import os, sys, time, pickle, imageio
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
import optax
from flax.training.train_state import TrainState
import mujoco
from math import exp

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

sys.path.insert(0, os.path.expanduser("~/panda_lego"))
from envs.lego_env import LegoEnv, build_model
from training.encoders import LegoAgent

# Config 
N_ENVS        = 4
N_STEPS       = 64           # longer rollouts = more GAE signal per update
N_EPOCHS      = 4            # more passes over each rollout
CLIP_EPS      = 0.15
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
ENT_COEF      = 0.01         # less entropy pressure
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
LR            = 3e-6
TOTAL_UPDATES = 10000
LOG_INTERVAL  = 10
SAVE_INTERVAL = 100

STD_START     = 0.15         # tighter noise; policy already knows to approach
STD_END       = 0.05
STD_DECAY     = 5000

TARGET_POS    = np.array([0.5, 0.15, 0.42])
BRICK_START   = np.array([0.35, 0.0, 0.42])
MAX_EP_STEPS  = 400          # longer episodes give policy time to recover from fumbles

# Grasp curriculum
# Slower gating; require success to hold for longer before advancing
CONTACT_HOLD_START  = 2    # resume from hold=3 (where last run stalled)
CONTACT_HOLD_MAX    = 8
CONTACT_HOLD_REQUIRED = 3
GRASP_SUCC_GATE     = 0.80   # slightly easier gate
GRASP_WINDOW        = 8      # must hold gate for longer before advancing

# Finger / contact config
FINGER_QPOS_IDXS = list(range(7, 23))   # 16 finger joints
FINGER_HOME      = 0.0
FINGER_CLOSED    = 0.5

# Contact detection
PALM_THRESH   = 0.08         # slightly more generous; 8cm palm proximity window
LIFT_THRESH   = 0.03         # brick must rise 3cm above BRICK_START z to count as lifted
BRICK_START_Z = 0.42         # z-height of brick at episode start

RECORD_INTERVAL = 1000           # record a video every N updates
RECORD_FPS      = 30             # frames per second in saved video
RECORD_WIDTH    = 640            # render resolution
RECORD_HEIGHT   = 480
VIDEO_DIR       = os.path.expanduser("~/panda_lego/videos")

ROBOT_XML  = os.path.expanduser("~/panda_lego/models/mjxpandamerged.xml")
ASSETS_DIR = os.path.expanduser("~/panda_lego/models/assets")
CKPT_DIR   = os.path.expanduser("~/panda_lego/checkpoints")

HOME_QPOS = np.array([
    0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785,
    0.0,  0.0,   0.0,  0.0,
    0.0,  0.0,   0.0,  0.0,
    0.0,  0.0,   0.0,  0.0,
    0.5,  0.0,   0.0,  0.0,
])


# Environment
class GraspEnv:
    def __init__(self, model):
        self.model   = model
        self.data    = mujoco.MjData(model)
        self.palm_id = model.body("palm").id
        self.nu      = model.nu
        self.obs_dim = 32   # unchanged from phases 1 & 2
        self.act_dim = self.nu
        self.steps        = 0
        self.contact_steps = 0  # consecutive steps with brick contact

        self.ctrl_low  = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = model.actuator_ctrlrange[:, 1].copy()
        self.ctrl_mid  = (self.ctrl_low + self.ctrl_high) / 2.0
        self.ctrl_half = (self.ctrl_high - self.ctrl_low) / 2.0

        # Find brick body ID once; used for contact detection
        try:
            self.brick_id = model.body("brick").id
        except Exception:
            # Fall back: brick is the first non-robot body (body index after arm)
            self.brick_id = 1
            for i in range(model.nbody):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if name and "brick" in name.lower():
                    self.brick_id = i
                    break

        # Collect geom IDs belonging to brick body and finger bodies
        self.brick_geom_ids  = self._geoms_for_body(self.brick_id)
        #finger_keywords = ["finger", "tip", "distal", "phalanx", "proximal", "medial", "base", "ff_", "mf_", "rf_", "th_"]
        self.finger_body_ids = set(range(10, 30))
        self.finger_geom_ids = set()
        for bid in self.finger_body_ids:
            self.finger_geom_ids |= self._geoms_for_body(bid)

    def _geoms_for_body(self, body_id):
        geom_ids = set()
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == body_id:
                geom_ids.add(g)
        return geom_ids

    def _finger_body_ids(self):
        ids = set()
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and any(k in name.lower() for k in ["finger", "tip", "distal", "phalanx"]):
                ids.add(i)
        # Fallback: if no finger names found, use all bodies with qpos idx >= 7
        if not ids:
            # Bodies attached to finger joints (qpos 7-22)
            for i in range(self.model.nbody):
                jnt_start = self.model.body_jntadr[i]
                jnt_num   = self.model.body_jntnum[i]
                for j in range(jnt_start, jnt_start + jnt_num):
                    if self.model.jnt_qposadr[j] >= 7:
                        ids.add(i)
        return ids

    def _has_brick_contact(self):
        for c in range(self.data.ncon):
            g1 = self.data.contact[c].geom1
            g2 = self.data.contact[c].geom2
            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]
        
            # Skip world/table body (body 0 = world)
            if b1 == 0 or b2 == 0:
                continue
            
            if (b1 == self.brick_id and b2 in self.finger_body_ids) or \
               (b2 == self.brick_id and b1 in self.finger_body_ids):
                return True
        return False

    def _mean_finger_closure(self):
        #Returns [0,1]; 0=fully open, 1=fully closed
        finger_qpos = self.data.qpos[FINGER_QPOS_IDXS]
        closure = np.mean(np.abs(finger_qpos - FINGER_HOME)) / max(abs(FINGER_CLOSED - FINGER_HOME), 1e-6)
        return float(np.clip(closure, 0.0, 1.0))

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:23] = HOME_QPOS
        self.data.ctrl[:]   = self.ctrl_mid
        self.data.qpos[23:26] = BRICK_START
        self.data.qpos[26:30] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.steps         = 0
        self.contact_steps = 0
        return self._get_obs()

    def _get_obs(self):
        # Identical to phases 1 & 2; checkpoint compatible
        robot_qpos = self.data.qpos[:23].copy()
        brick_pos  = self.data.qpos[23:26].copy()
        palm_pos   = self.data.xpos[self.palm_id].copy()
        return np.concatenate([robot_qpos, brick_pos, TARGET_POS, palm_pos])

    def step(self, action, contact_hold_required):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = self.ctrl_mid + action * self.ctrl_half
        mujoco.mj_step(self.model, self.data)
        self.steps += 1

        brick_pos = self.data.qpos[23:26].copy()
        palm_pos  = self.data.xpos[self.palm_id].copy()
        d_palm    = np.linalg.norm(palm_pos - brick_pos)

        has_contact    = self._has_brick_contact()
        finger_closure = self._mean_finger_closure()

        # Brick drift; how far has the brick moved laterally from its start?
        brick_xy_drift = np.linalg.norm(brick_pos[:2] - BRICK_START[:2])
        # Brick lift; how much has the brick risen above its start height?
        brick_lift     = max(0.0, brick_pos[2] - BRICK_START_Z)

        # Floor penalty; penalise any finger or palm position below table height
        TABLE_Z = 0.42   # same as BRICK_START_Z, the table surface
        FLOOR_MARGIN = 0.02  # fingers should stay 2cm above table
        palm_floor_violation = max(0.0, TABLE_Z + FLOOR_MARGIN - palm_pos[2])
        floor_penalty = -20.0 * palm_floor_violation

        # Contact streak
        # Only count contact when fingers are also closing (closure > 0.3)
        near_and_closing = has_contact and d_palm < PALM_THRESH and finger_closure > 0.3
        if near_and_closing:
            self.contact_steps += 1
        else:
            self.contact_steps = 0

        MIN_GRASP_STEPS = 30   # must maintain contact for at least this long to count as a true grasp success (prevents brief accidental contacts from scoring)

        # True success requires sustained contact AND the brick must actually be lifted
        lifted_enough = brick_lift >= LIFT_THRESH   # 3cm off table
        success = (self.contact_steps >= contact_hold_required and lifted_enough and self.steps >= MIN_GRASP_STEPS)

        # Reward shaping 
        # 1. Approach: dense reward for closing distance to brick
        approach_reward = exp(-4.0 * d_palm) * 0.3

        # 2. Closure-gated contact: reward for touching brick with closing fingers
        grasping_contact = float(near_and_closing)
        contact_bonus    = 3.0 * grasping_contact

        # 3. Finger closure reward: only paid when hand is near the brick, scaled by how closed the fingers are
        closure_reward = 0.3 * finger_closure * float(d_palm < PALM_THRESH)

        # 4. Streak reward: smooth dense signal for maintaining grasping contact
        streak_frac           = min(self.contact_steps, contact_hold_required) / contact_hold_required
        # Only reward streaks when the brick is actually being lifted off the table
        contact_streak_reward = 5.0 * streak_frac * min(brick_lift / LIFT_THRESH, 1.0)

        # 5. Anti-shove penalty: penalise lateral drift of the brick
        shove_penalty = -5.0 * brick_xy_drift

        # 6. Lift bonus: reward if the brick rises while being held
        lift_bonus = 50.0 * brick_lift   # much stronger incentive to actually lift

        brick_vel_z = float(self.data.qvel[25])
        lift_motion_reward = 15.0 * max(0.0, brick_vel_z) * float(has_contact)
        
        # 7. Terminal grasp bonus (one-shot on success)
        grasp_bonus = 100.0 * float(success)

        # 8. Small time penalty to discourage dithering
        time_penalty = -0.002

        reward = (approach_reward + contact_bonus + closure_reward +
                  contact_streak_reward + shove_penalty + lift_bonus +
                  lift_motion_reward +grasp_bonus + floor_penalty + time_penalty)

        # Terminate on success or timeout
        done = success or (self.steps >= MAX_EP_STEPS)
        return self._get_obs(), reward, done, d_palm, has_contact, finger_closure


# GAE 
def compute_gae(rewards, values, dones):
    n_steps, n_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    last_adv   = np.zeros(n_envs)
    for t in reversed(range(n_steps)):
        nv       = values[t+1] if t < n_steps - 1 else np.zeros(n_envs)
        delta    = rewards[t] + GAMMA * nv * (1 - dones[t]) - values[t]
        last_adv = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * last_adv
        advantages[t] = last_adv
    return advantages, advantages + values


# PPO loss (identical to previous phases)
def ppo_loss(params, agent, obs, tgt, actions, old_lp, advantages, returns, std):
    pred_a, values, _, _ = vmap(lambda o, t: agent.apply(params, o, t))(obs, tgt)
    log_p   = -0.5 * jnp.sum(((actions - pred_a) / std) ** 2, axis=-1)
    ratio   = jnp.exp(log_p - old_lp)
    pg_loss = -jnp.mean(jnp.minimum(
        advantages * ratio,
        advantages * jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
    ))
    vf_loss = jnp.mean((values - returns) ** 2)
    entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * std ** 2) * actions.shape[-1]
    return pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy, (pg_loss, vf_loss)


# Recording 
def record_rollout(mj_model, state, agent, forward_fn, update_idx, contact_hold_required):
    os.makedirs(VIDEO_DIR, exist_ok=True)

    rec_env = GraspEnv(mj_model)
    obs     = rec_env.reset()
    tgt     = jnp.array(TARGET_POS)

    renderer = mujoco.Renderer(mj_model, height=RECORD_HEIGHT, width=RECORD_WIDTH)

    frames = []
    done   = False
    # Run for at least MIN_RECORD_STEPS frames even after success, so videos are watchable
    MIN_RECORD_STEPS = 120   # 4 seconds at 30fps
    while rec_env.steps < max(MIN_RECORD_STEPS, MAX_EP_STEPS):
        if done and rec_env.steps >= MIN_RECORD_STEPS:
            break
        a, _ = forward_fn(state.params, jnp.array(obs), tgt)
        obs, _, done, _, _, _ = rec_env.step(np.array(a), contact_hold_required)

        renderer.update_scene(rec_env.data)
        frame = renderer.render()          # returns (H, W, 3) uint8 RGB
        frames.append(frame)

    renderer.close()

    if frames:
        video_path = os.path.join(VIDEO_DIR, f"grasp_update_{update_idx:06d}.mp4")
        imageio.mimwrite(video_path, frames, fps=RECORD_FPS, quality=8)
        print(f"  🎥 Video saved ({len(frames)} frames): {video_path}")


# Train 
def train():
    print("=== Panda-Lego Phase 3: Grasp (CPU sim + GPU nets) ===")
    print(f"JAX devices: {jax.devices()}")

    print("\nBuilding MuJoCo model...")
    mj_model = build_model(ROBOT_XML, ASSETS_DIR)
    envs     = [GraspEnv(mj_model) for _ in range(N_ENVS)]
    print(f"  nq={mj_model.nq}, nu={mj_model.nu}")
    print(f"  Brick body ID: {envs[0].brick_id}")
    print(f"  Brick geoms: {envs[0].brick_geom_ids}")
    print(f"  Finger body IDs: {envs[0].finger_body_ids}")
    print(f"  Finger geoms: {len(envs[0].finger_geom_ids)} geoms")

    _diag_env = envs[0]
    _diag_env.reset()
    print(f"  Contact diagnostic: ncon={_diag_env.data.ncon}, has_contact={_diag_env._has_brick_contact()}")

    _diag_env = envs[0]
    _diag_env.reset()

    print(f"  Contact diagnostic: ncon={_diag_env.data.ncon}, has_contact={_diag_env._has_brick_contact()}")

    agent  = LegoAgent(act_dim=envs[0].act_dim)
    key    = jax.random.PRNGKey(0)
    params = agent.init(key, jnp.zeros(32), jnp.array(TARGET_POS))
    n_p    = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Agent parameters: {n_p:,}")

    resume_path = os.path.expanduser("~/panda_lego/checkpoints/grasp_best_h02.pkl")  # last known-good policy
    with open(resume_path, "rb") as f:
        params = pickle.load(f)
    print(f"  Resumed from: {resume_path}")

    tx    = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR))
    state = TrainState.create(apply_fn=agent.apply, params=params, tx=tx)

    @jit
    def forward(params, obs, tgt):
        a, v, _, _ = agent.apply(params, obs, tgt)
        return a, v

    obs_list = [env.reset() for env in envs]
    ep_rews  = np.zeros(N_ENVS)
    ep_lens  = np.zeros(N_ENVS, dtype=int)
    all_rews = []
    all_successes = []
    tgt      = jnp.array(TARGET_POS)

    contact_hold_required = CONTACT_HOLD_START   # = 3, resuming from h02
    consec_good           = 0
    consec_bad            = 0
    best_success_rate     = 0.0   # FIX 4: track best success for checkpoint saving

    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"\nTraining: {TOTAL_UPDATES} updates | {N_ENVS} envs | {N_STEPS} steps/update")
    print(f"Grasp curriculum: contact hold 1 → {CONTACT_HOLD_MAX} steps")
    print(f"Gate: {GRASP_SUCC_GATE*100:.0f}% success x {GRASP_WINDOW} log intervals")
    print("-" * 70)

    for update in range(1, TOTAL_UPDATES + 1):
        t0 = time.time()

        frac        = min(update / STD_DECAY, 1.0)
        current_std = STD_START + frac * (STD_END - STD_START)

        obs_buf  = np.zeros((N_STEPS, N_ENVS, 32),              dtype=np.float32)
        act_buf  = np.zeros((N_STEPS, N_ENVS, envs[0].act_dim), dtype=np.float32)
        logp_buf = np.zeros((N_STEPS, N_ENVS),                  dtype=np.float32)
        val_buf  = np.zeros((N_STEPS, N_ENVS),                  dtype=np.float32)
        rew_buf  = np.zeros((N_STEPS, N_ENVS),                  dtype=np.float32)
        done_buf = np.zeros((N_STEPS, N_ENVS),                  dtype=np.float32)

        ep_contacts  = np.zeros(N_ENVS)   # contact rate this rollout
        ep_closures  = np.zeros(N_ENVS)   # mean finger closure this rollout
        ep_dpalms    = []
        ep_drifts    = []   # brick xy drift (shove metric)
        ep_lifts     = []   # brick z lift

        for s in range(N_STEPS):
            for i, env in enumerate(envs):
                obs  = jnp.array(obs_list[i])
                a, v = forward(state.params, obs, tgt)
                a    = np.array(a)
                noise   = np.random.normal(0, current_std, a.shape).astype(np.float32)
                a_noisy = np.clip(a + noise, -1.0, 1.0)
                lp      = float(-0.5 * np.sum(((a_noisy - a) / current_std) ** 2))

                next_obs, rew, done, d_palm, has_contact, closure = \
                    env.step(a_noisy, contact_hold_required)

                obs_buf[s, i]  = obs_list[i]
                act_buf[s, i]  = a_noisy
                logp_buf[s, i] = lp
                val_buf[s, i]  = float(v)
                rew_buf[s, i]  = rew
                done_buf[s, i] = float(done)

                ep_rews[i]    += rew
                ep_lens[i]    += 1
                ep_contacts[i] = ep_contacts[i] * 0.95 + float(has_contact) * 0.05
                ep_closures[i] = ep_closures[i] * 0.95 + closure * 0.05
                ep_dpalms.append(d_palm)
                brick_pos_log = env.data.qpos[23:26].copy()
                ep_drifts.append(float(np.linalg.norm(brick_pos_log[:2] - BRICK_START[:2])))
                ep_lifts.append(max(0.0, float(brick_pos_log[2] - BRICK_START_Z)))

                ep_true_success = np.zeros(N_ENVS)

                if done:
                    true_success = (env.contact_steps >= contact_hold_required and (env.data.qpos[25] - BRICK_START_Z) >= LIFT_THRESH)
                    all_rews.append(ep_rews[i])
                    all_successes.append(float(true_success))
                    ep_rews[i] = 0.0
                    ep_lens[i] = 0
                    obs_list[i] = env.reset()
                else:
                    obs_list[i] = next_obs

        # GAE
        adv, ret = compute_gae(rew_buf, val_buf, done_buf)
        adv_flat = adv.flatten().astype(np.float32)
        ret_flat = ret.flatten().astype(np.float32)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        obs_flat  = obs_buf.reshape(-1, 32)
        act_flat  = act_buf.reshape(-1, envs[0].act_dim)
        logp_flat = logp_buf.flatten()
        tgt_flat  = np.tile(TARGET_POS, (N_STEPS * N_ENVS, 1)).astype(np.float32)

        def loss_wrapper(params):
            return ppo_loss(params, agent, jnp.array(obs_flat), jnp.array(tgt_flat),
                            jnp.array(act_flat), jnp.array(logp_flat),
                            jnp.array(adv_flat), jnp.array(ret_flat),
                            current_std)

        grad_fn = jit(value_and_grad(loss_wrapper, has_aux=True))
        for _ in range(N_EPOCHS):
            (loss, aux), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)

        if update % LOG_INTERVAL == 0:
            mr      = np.mean(all_rews[-50:]) if all_rews else 0.0
            # Success = episode reward high enough to have gotten grasp_bonus
            suc     = np.mean(all_successes[-50:]) if all_successes else 0.0
            pg, vf  = aux
            ep_len  = ep_lens.mean()
            mean_dp = np.mean(ep_dpalms)
            mean_contact = float(np.mean(ep_contacts))
            mean_closure = float(np.mean(ep_closures))
            mean_drift = float(np.mean(ep_drifts)) if ep_drifts else 0.0
            mean_lift  = float(np.mean(ep_lifts))  if ep_lifts  else 0.0

            print(f"Update {update:5d} | Rew {mr:7.2f} | Succ {suc*100:5.1f}% | "
                  f"PG {float(pg):6.3f} | VF {float(vf):5.1f} | "
                  f"dPalm {mean_dp:.3f} | Contact {mean_contact:.2f} | "
                  f"Closure {mean_closure:.2f} | Drift {mean_drift:.3f} | Lift {mean_lift:.3f} | "
                  f"EpLen {ep_len:.0f} | Hold {contact_hold_required}/{CONTACT_HOLD_MAX} | {time.time()-t0:.1f}s")

            # FIX 4: save best checkpoint whenever we hit a new peak success rate
            if suc > best_success_rate:
                best_success_rate = suc
                best_path = f"{CKPT_DIR}/grasp_best.pkl"
                with open(best_path, "wb") as f:
                    pickle.dump(state.params, f)
                print(f"  ★ New best: {suc*100:.1f}% success → {best_path}")

            # Curriculum advance
            if suc >= GRASP_SUCC_GATE and mean_contact >= 0.7:
                consec_good += 1
                consec_bad = 0
            else:
                consec_good = 0
                consec_bad += 1

            if consec_good >= GRASP_WINDOW and contact_hold_required < CONTACT_HOLD_MAX:
                old_req = contact_hold_required
                contact_hold_required += 1
                consec_good = 0
                consec_bad  = 0
                print(f"  ↑ Contact hold increased: {old_req} → {contact_hold_required}")
                path = f"{CKPT_DIR}/grasp_best_h{old_req:02d}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(state.params, f)
                print(f"    → Saved: {path}")

            # Regression rollback: if success collapses after a hold advance, step back
            REGRESSION_WINDOW   = 16   # bad intervals before rolling back
            REGRESSION_SUC_FLOOR = 0.20
            if (consec_bad >= REGRESSION_WINDOW and
                    suc < REGRESSION_SUC_FLOOR and
                    contact_hold_required > CONTACT_HOLD_START):
                old_req = contact_hold_required
                contact_hold_required -= 1
                consec_bad  = 0
                consec_good = 0
                rollback_path = f"{CKPT_DIR}/grasp_best_h{contact_hold_required:02d}.pkl"
                if os.path.exists(rollback_path):
                    with open(rollback_path, "rb") as f:
                        restored_params = pickle.load(f)
                    state = state.replace(params=restored_params)
                    print(f"  ↓ ROLLBACK: Hold {old_req} → {contact_hold_required}, restored {rollback_path}")
                else:
                    print(f"  ↓ ROLLBACK: Hold {old_req} → {contact_hold_required} (no checkpoint to restore)")

            if contact_hold_required == CONTACT_HOLD_MAX and suc >= GRASP_SUCC_GATE:
                print(f" GRASP PHASE COMPLETE! {contact_hold_required}-step contact at {suc*100:.1f}% success!")

        if update % SAVE_INTERVAL == 0:
            path = f"{CKPT_DIR}/grasp_agent_{update}.pkl"
            with open(path, "wb") as f:
                pickle.dump(state.params, f)
            print(f"  → Checkpoint: {path}")

        if update % RECORD_INTERVAL == 0:
            print(f"  Recording rollout at update {update}...")
            record_rollout(mj_model, state, agent, forward, update, contact_hold_required)

    print("\n Phase 3 (Grasp) training complete!")
    return state

if __name__ == "__main__":
    train()
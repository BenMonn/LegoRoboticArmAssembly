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
N_ENVS        = 16
N_STEPS       = 128
N_EPOCHS      = 4
CLIP_EPS      = 0.15
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
ENT_COEF      = 0.01
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
LR            = 5e-5   # v4: lower LR for stability when resuming from best
TOTAL_UPDATES = 10000
LOG_INTERVAL  = 10
SAVE_INTERVAL = 100

STD_START     = 0.15
STD_END       = 0.08
STD_DECAY     = 8000

TARGET_POS    = np.array([0.5, 0.15, 0.42])
MAX_EP_STEPS  = 400

# Grasp curriculum
CONTACT_HOLD_START = 4
CONTACT_HOLD_MAX   = 8

GRASP_SUCC_GATE    = 0.30
GRASP_WINDOW       = 3

REGRESSION_WINDOW    = 16
REGRESSION_SUC_FLOOR = 0.20

# Finger / contact config
FINGER_QPOS_IDXS = list(range(7, 23))
FINGER_HOME      = 0.0
FINGER_CLOSED    = 0.5

# Contact / lift thresholds
PALM_THRESH = 0.08
LIFT_THRESH = 0.03

# Domain randomization
# Same schedule as reach / hold DR so the grasp policy sees the same
# distribution it was trained to approach from.
#
# BRICK_BASE_POS: the canonical spawn point.
# Warmup: fixed for DR_WARMUP updates so the policy can stabilise from the
# hold-DR checkpoint before noise is introduced.
# After warmup: XY / Z / yaw noise ramps linearly to DR_*_MAX.
BRICK_BASE_POS = np.array([0.35, 0.0, 0.42])
DR_WARMUP      = 1000
DR_XY_MAX      = 0.06          # ±6 cm  (matches reach/hold)
DR_Z_MAX       = 0.01          # ±1 cm
DR_YAW_MAX     = np.deg2rad(20)  # ±20°

# v4: DR freeze gate
DR_FREEZE_MAX    = 0.20   # cap DR at 20% until gate is passed
DR_FREEZE_THRESH = 0.15   # need 15% contact-gated success to unfreeze
DR_CONTACT_MIN   = 0.40   # contact must be above this for success to count
DR_FREEZE_WINDOW = 20     # consecutive LOG_INTERVAL updates above threshold

# Video recording
RECORD_INTERVAL = 1000
RECORD_FPS      = 30
RECORD_WIDTH    = 640
RECORD_HEIGHT   = 480
VIDEO_DIR       = os.path.expanduser("~/panda_lego/videos")

# Paths
ROBOT_XML  = os.path.expanduser("~/panda_lego/models/mjxpandamerged.xml")
ASSETS_DIR = os.path.expanduser("~/panda_lego/models/assets")
CKPT_DIR   = os.path.expanduser("~/panda_lego/checkpoints")

# Resume from the best grasp checkpoint
RESUME_PATH = os.path.expanduser(
    "~/panda_lego/checkpoints/grasp_best_h07.pkl"
)

HOME_QPOS = np.array([
    -0.0413, -0.5000,  0.1060, -1.8000, -0.2379,  2.0784,  0.6668,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.5, 0.3, 0.5, 0.3,
])

VF_CLIP_RANGE = 10.0 # was 50


# DR helper
def sample_brick_spawn(update, total_updates, rng, dr_unfrozen):
    if update <= DR_WARMUP:
        noise_frac = 0.0
    else:
        raw_frac = min((update - DR_WARMUP) / (total_updates - DR_WARMUP), 1.0)
        if dr_unfrozen:
            noise_frac = raw_frac
        else:
            noise_frac = min(raw_frac, DR_FREEZE_MAX)

    xy    = rng.uniform(-DR_XY_MAX,  DR_XY_MAX,  size=2) * noise_frac
    dz    = rng.uniform(-DR_Z_MAX,   DR_Z_MAX)            * noise_frac
    yaw   = rng.uniform(-DR_YAW_MAX, DR_YAW_MAX)          * noise_frac

    pos  = BRICK_BASE_POS + np.array([xy[0], xy[1], dz])
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    return pos.copy(), quat.copy()


# Environment
class GraspEnv:
    def __init__(self, model):
        self.model   = model
        self.data    = mujoco.MjData(model)
        self.palm_id = model.body("palm").id
        self.nu      = model.nu
        self.obs_dim = 32
        self.act_dim = self.nu
        self.steps         = 0
        self.contact_steps = 0

        # Per-episode brick spawn — updated on reset() so all reward calculations (drift, lift) reference the correct starting position
        self.episode_brick_start = BRICK_BASE_POS.copy()

        self.ctrl_low  = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = model.actuator_ctrlrange[:, 1].copy()
        self.ctrl_mid  = (self.ctrl_low + self.ctrl_high) / 2.0
        self.ctrl_half = (self.ctrl_high - self.ctrl_low) / 2.0

        try:
            self.brick_id = model.body("brick").id
        except Exception:
            self.brick_id = 1
            for i in range(model.nbody):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if name and "brick" in name.lower():
                    self.brick_id = i
                    break

        self.brick_geom_ids  = self._geoms_for_body(self.brick_id)
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

    def _has_brick_contact(self):
        for c in range(self.data.ncon):
            g1 = self.data.contact[c].geom1
            g2 = self.data.contact[c].geom2
            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]
            if b1 == 0 or b2 == 0:
                continue
            if (b1 == self.brick_id and b2 in self.finger_body_ids) or \
               (b2 == self.brick_id and b1 in self.finger_body_ids):
                return True
        return False

    def _mean_finger_closure(self):
        finger_qpos = self.data.qpos[FINGER_QPOS_IDXS]
        closure = np.mean(np.abs(finger_qpos - FINGER_HOME)) / max(abs(FINGER_CLOSED - FINGER_HOME), 1e-6)
        return float(np.clip(closure, 0.0, 1.0))

    def reset(self, brick_pos=None, brick_quat=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:23] = HOME_QPOS
        self.data.ctrl[:]   = self.ctrl_mid

        pos  = brick_pos  if brick_pos  is not None else BRICK_BASE_POS.copy()
        quat = brick_quat if brick_quat is not None else np.array([1., 0., 0., 0.])

        self.data.qpos[23:26] = pos
        self.data.qpos[26:30] = quat
        mujoco.mj_forward(self.model, self.data)

        # Store the actual spawn so drift/lift are computed relative to THIS episode's start, not a global constant
        self.episode_brick_start = pos.copy()
        self.steps         = 0
        self.contact_steps = 0
        return self._get_obs()

    def _get_obs(self):
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

        # Drift and lift are always relative to THIS episode's brick start
        brick_start_z  = self.episode_brick_start[2]
        brick_xy_drift = np.linalg.norm(brick_pos[:2] - self.episode_brick_start[:2])
        brick_lift     = max(0.0, brick_pos[2] - brick_start_z)

        # Floor penalty
        FLOOR_MARGIN = 0.02
        palm_floor_violation = max(0.0, brick_start_z + FLOOR_MARGIN - palm_pos[2])
        floor_penalty = -20.0 * palm_floor_violation

        # Contact streak
        near_and_closing = has_contact and d_palm < PALM_THRESH and finger_closure > 0.3
        if near_and_closing:
            self.contact_steps += 1
        else:
            self.contact_steps = 0

        # Success condition
        MIN_GRASP_STEPS = 30
        lifted_enough   = brick_lift >= LIFT_THRESH
        is_holding      = self.contact_steps >= contact_hold_required
        success         = is_holding and lifted_enough and self.steps >= MIN_GRASP_STEPS

        # Reward shaping

        # 1. Approach: always dense so the agent finds the brick
        approach_reward = exp(-4.0 * d_palm) * 0.3 * (1.0 + 2.0 * float(has_contact))

        # 2. Contact bonus: gated on near_and_closing
        grasping_contact = float(near_and_closing)
        contact_bonus    = 3.0 * grasping_contact

        # 3. Finger closure near brick
        closure_reward = 0.3 * finger_closure * float(d_palm < PALM_THRESH) * float(has_contact)

        # 4. Contact streak reward
        streak_frac           = min(self.contact_steps, contact_hold_required) / contact_hold_required
        contact_streak_reward = 1.0 * streak_frac * (1.0 if is_holding else 0.1) # was 2.0 * ...

        # 5. Shove penalty: always active
        MAX_DRIFT = 0.35
        if brick_xy_drift > MAX_DRIFT:
            shove_penalty = -50.0
            done = True
        else:
            shove_penalty = -1.0 * brick_xy_drift

        # 6. Lift bonus: gated on is_holding
        lift_bonus = 10.0 * (brick_lift ** 0.5) * float(is_holding)
        if brick_lift > 0.02:
            lift_bonus += 5.0
        if brick_lift > 0.05:
            lift_bonus += 10.0

        # 7. Terminal grasp bonus
        grasp_bonus = 20.0 * float(success)

        # 8. Per-step completion bonus while holding and lifting
        lift_frac        = min(brick_lift / LIFT_THRESH, 1.0)
        completion_bonus = 3.0 * float(is_holding) * lift_frac

        # 9. Palm upward velocity reward while holding
        palm_vel_z           = float(self.data.cvel[self.palm_id, 5])
        lift_velocity_reward = 0.5 * max(0.0, palm_vel_z) * float(is_holding)

        # 10. Time penalty
        time_penalty = -0.05 - 0.3 * max(0.0, d_palm - 0.05)

        reward = (approach_reward + contact_bonus + closure_reward +
                  contact_streak_reward + shove_penalty + lift_bonus +
                  grasp_bonus + completion_bonus + floor_penalty +
                  lift_velocity_reward + time_penalty)

        done = success or (self.steps >= MAX_EP_STEPS)
        return self._get_obs(), reward, done, d_palm, has_contact, finger_closure


# GAE
def compute_gae(rewards, values, dones, next_values):
    n_steps, n_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    last_adv   = np.zeros(n_envs)
    for t in reversed(range(n_steps)):
        nv       = next_values if t == n_steps - 1 else values[t + 1]
        delta    = rewards[t] + GAMMA * nv * (1.0 - dones[t]) - values[t]
        last_adv = delta + GAMMA * GAE_LAMBDA * (1.0 - dones[t]) * last_adv
        advantages[t] = last_adv
    return advantages, advantages + values


# PPO loss
def ppo_loss(params, agent, obs, tgt, actions, old_lp, advantages, returns, std, old_values):
    pred_a, values, _, _ = vmap(lambda o, t: agent.apply(params, o, t))(obs, tgt)
    log_p   = -0.5 * jnp.sum(((actions - pred_a) / std) ** 2, axis=-1)
    ratio   = jnp.exp(log_p - old_lp)
    pg_loss = -jnp.mean(jnp.minimum(
        advantages * ratio,
        advantages * jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS),
    ))
    vf_unclipped   = (values - returns) ** 2
    values_clipped = old_values + jnp.clip(values - old_values, -VF_CLIP_RANGE, VF_CLIP_RANGE)
    vf_clipped     = (values_clipped - returns) ** 2
    vf_loss        = jnp.mean(jnp.maximum(vf_unclipped, vf_clipped))
    entropy = 0.5 * jnp.log(2.0 * jnp.pi * jnp.e * std ** 2) * actions.shape[-1]
    return pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy, (pg_loss, vf_loss)


# Video recording
def record_rollout(mj_model, state, agent, forward_fn, update_idx, contact_hold_required):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    rec_env  = GraspEnv(mj_model)
    obs      = rec_env.reset()   # fixed spawn for deterministic video
    tgt      = jnp.array(TARGET_POS)
    renderer = mujoco.Renderer(mj_model, height=RECORD_HEIGHT, width=RECORD_WIDTH)
    frames   = []
    done     = False
    MIN_RECORD_STEPS = 120
    while rec_env.steps < max(MIN_RECORD_STEPS, MAX_EP_STEPS):
        if done and rec_env.steps >= MIN_RECORD_STEPS:
            break
        a, _ = forward_fn(state.params, jnp.array(obs), tgt)
        obs, _, done, _, _, _ = rec_env.step(np.array(a), contact_hold_required)
        renderer.update_scene(rec_env.data)
        frames.append(renderer.render())
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
    print(f"  Contact diagnostic: ncon={_diag_env.data.ncon}, "
          f"has_contact={_diag_env._has_brick_contact()}")

    agent  = LegoAgent(act_dim=envs[0].act_dim)
    key    = jax.random.PRNGKey(0)
    params = agent.init(key, jnp.zeros(32), jnp.array(TARGET_POS))
    n_p    = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Agent parameters: {n_p:,}")

    if os.path.exists(RESUME_PATH):
        with open(RESUME_PATH, "rb") as f:
            params = pickle.load(f)
        print(f"  Resumed from: {RESUME_PATH}")
    else:
        print(f"  WARNING: {RESUME_PATH} not found — starting from scratch.")
        print(f"  (Set RESUME_PATH to your best hold-DR checkpoint.)")

    schedule = optax.linear_schedule(5e-5, 5e-6, TOTAL_UPDATES)  # v4: lower LR
    tx = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(schedule))
    state = TrainState.create(apply_fn=agent.apply, params=params, tx=tx)

    @jit
    def forward(params, obs, tgt):
        a, v, _, _ = agent.apply(params, obs, tgt)
        return a, v

    @jit
    def loss_and_grad(params, obs_j, tgt_j, act_j, logp_j, adv_j, ret_j, std_j, old_val_j):
        return value_and_grad(
            lambda p: ppo_loss(p, agent, obs_j, tgt_j, act_j, logp_j,
                               adv_j, ret_j, std_j, old_val_j),
            has_aux=True,
        )(params)

    rng = np.random.default_rng(42)

    # Initial reset at base position (warmup period)
    obs_list = [env.reset() for env in envs]
    ep_rews          = np.zeros(N_ENVS)
    ep_lens          = np.zeros(N_ENVS, dtype=int)
    all_rews         = []
    all_successes    = []
    all_hold_streaks = []
    ep_max_hold      = np.zeros(N_ENVS, dtype=np.float32)
    tgt              = jnp.array(TARGET_POS)

    contact_hold_required = CONTACT_HOLD_START
    consec_good           = 0
    consec_bad            = 0
    best_success_rate     = 0.0

    # v4: DR freeze gate state
    dr_unfrozen              = False
    dr_freeze_consec         = 0     # consecutive log intervals with contact-gated success above thresh
    all_contact_successes    = []    # parallel to all_successes: 1 if success AND contact >= DR_CONTACT_MIN

    os.makedirs(CKPT_DIR, exist_ok=True)
    initial_ckpt_path = f"{CKPT_DIR}/grasp_best_h{CONTACT_HOLD_START:02d}.pkl"
    if not os.path.exists(initial_ckpt_path):
        with open(initial_ckpt_path, "wb") as f:
            pickle.dump(state.params, f)
        print(f"  Saved initial rollback checkpoint: {initial_ckpt_path}")

    print(f"\nTraining: {TOTAL_UPDATES} updates | {N_ENVS} envs | {N_STEPS} steps/update")
    print(f"Grasp curriculum: contact hold {CONTACT_HOLD_START} → {CONTACT_HOLD_MAX} steps")
    print(f"Gate: {GRASP_SUCC_GATE*100:.0f}% success x {GRASP_WINDOW} log intervals "
          f"(contact-gate REMOVED)")
    print(f"Domain randomization: warmup {DR_WARMUP} updates, "
          f"then XY ±{DR_XY_MAX*100:.0f}cm / Z ±{DR_Z_MAX*100:.0f}cm / "
          f"yaw ±{np.rad2deg(DR_YAW_MAX):.0f}° ramps to max")
    print(f"v4 DR freeze: capped at {DR_FREEZE_MAX*100:.0f}% until {DR_FREEZE_THRESH*100:.0f}% "
          f"contact-gated success for {DR_FREEZE_WINDOW} log intervals")
    print("-" * 70)

    for update in range(1, TOTAL_UPDATES + 1):
        t0 = time.time()

        frac        = min(update / STD_DECAY, 1.0)
        current_std = float(STD_START + frac * (STD_END - STD_START))
        std_j       = jnp.array(current_std)

        obs_buf  = np.zeros((N_STEPS, N_ENVS, 32),              dtype=np.float32)
        act_buf  = np.zeros((N_STEPS, N_ENVS, envs[0].act_dim), dtype=np.float32)
        logp_buf = np.zeros((N_STEPS, N_ENVS),                  dtype=np.float32)
        val_buf  = np.zeros((N_STEPS, N_ENVS),                  dtype=np.float32)
        rew_buf  = np.zeros((N_STEPS, N_ENVS),                  dtype=np.float32)
        done_buf = np.zeros((N_STEPS, N_ENVS),                  dtype=np.float32)

        ep_contacts = np.zeros(N_ENVS)
        ep_closures = np.zeros(N_ENVS)
        ep_dpalms   = []
        ep_drifts   = []
        ep_lifts    = []

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
                brick_pos_log  = env.data.qpos[23:26].copy()
                ep_drifts.append(float(np.linalg.norm(
                    brick_pos_log[:2] - env.episode_brick_start[:2])))
                ep_lifts.append(max(0.0, float(
                    brick_pos_log[2] - env.episode_brick_start[2])))
                ep_max_hold[i] = max(ep_max_hold[i], float(env.contact_steps))

                if done:
                    true_success = (
                        env.contact_steps >= contact_hold_required and
                        (env.data.qpos[25] - env.episode_brick_start[2]) >= LIFT_THRESH
                    )
                    # v4: contact-gated success for DR freeze check
                    contact_success = float(true_success and ep_contacts[i] >= DR_CONTACT_MIN)
                    all_rews.append(ep_rews[i])
                    all_successes.append(float(true_success))
                    all_contact_successes.append(contact_success)
                    all_hold_streaks.append(ep_max_hold[i])
                    ep_max_hold[i] = 0.0
                    ep_rews[i]     = 0.0
                    ep_lens[i]     = 0
                    # Each episode gets a fresh randomized spawn
                    new_pos, new_quat = sample_brick_spawn(update, TOTAL_UPDATES, rng, dr_unfrozen)
                    obs_list[i] = env.reset(brick_pos=new_pos, brick_quat=new_quat)
                else:
                    obs_list[i] = next_obs

        next_values = np.array([
            float(forward(state.params, jnp.array(obs_list[i]), tgt)[1])
            for i in range(N_ENVS)
        ])

        adv, ret = compute_gae(rew_buf, val_buf, done_buf, next_values)
        adv_flat = adv.flatten().astype(np.float32)
        ret_flat = ret.flatten().astype(np.float32)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        obs_j     = jnp.array(obs_buf.reshape(-1, 32))
        act_j     = jnp.array(act_buf.reshape(-1, envs[0].act_dim))
        logp_j    = jnp.array(logp_buf.flatten())
        adv_j     = jnp.array(adv_flat)
        ret_j     = jnp.array(ret_flat)
        tgt_j     = jnp.array(np.tile(TARGET_POS, (N_STEPS * N_ENVS, 1)).astype(np.float32))
        old_val_j = jnp.array(val_buf.flatten().astype(np.float32))

        for _ in range(N_EPOCHS):
            (loss, aux), grads = loss_and_grad(
                state.params, obs_j, tgt_j, act_j, logp_j, adv_j, ret_j, std_j, old_val_j
            )
            state = state.apply_gradients(grads=grads)

        # Logging
        if update % LOG_INTERVAL == 0:
            mr           = np.mean(all_rews[-50:])      if all_rews      else 0.0
            suc          = np.mean(all_successes[-50:]) if all_successes else 0.0
            # v4: contact-gated success for DR freeze check
            c_suc        = np.mean(all_contact_successes[-50:]) if all_contact_successes else 0.0
            pg, vf       = aux
            ep_len       = ep_lens.mean()
            mean_dp      = np.mean(ep_dpalms)
            mean_contact = float(np.mean(ep_contacts))
            mean_closure = float(np.mean(ep_closures))
            mean_drift   = float(np.mean(ep_drifts)) if ep_drifts else 0.0
            mean_lift    = float(np.mean(ep_lifts))  if ep_lifts  else 0.0
            mean_streak  = float(np.mean(all_hold_streaks[-50:])) if all_hold_streaks else 0.0
            # v4: noise_frac respects the DR freeze
            if update <= DR_WARMUP:
                noise_frac = 0.0
            else:
                raw_frac = min((update - DR_WARMUP) / (TOTAL_UPDATES - DR_WARMUP), 1.0)
                noise_frac = raw_frac if dr_unfrozen else min(raw_frac, DR_FREEZE_MAX)

            # v4: DR freeze gate logic
            if not dr_unfrozen:
                if c_suc >= DR_FREEZE_THRESH:
                    dr_freeze_consec += 1
                else:
                    dr_freeze_consec = 0
                if dr_freeze_consec >= DR_FREEZE_WINDOW:
                    dr_unfrozen = True
                    print(f"  ★ DR UNFROZEN at update {update}! "
                          f"Contact-gated success held at {c_suc*100:.1f}% "
                          f"for {DR_FREEZE_WINDOW} log intervals.")

            freeze_tag = "" if dr_unfrozen else f" [frozen≤{DR_FREEZE_MAX*100:.0f}%,{dr_freeze_consec}/{DR_FREEZE_WINDOW}]"

            print(f"Update {update:5d} | Rew {mr:7.2f} | Succ {suc*100:5.1f}% | "
                  f"CSuc {c_suc*100:5.1f}% | "
                  f"PG {float(pg):6.3f} | VF {float(vf):5.1f} | "
                  f"dPalm {mean_dp:.3f} | Contact {mean_contact:.2f} | "
                  f"Closure {mean_closure:.2f} | Drift {mean_drift:.3f} | "
                  f"Lift {mean_lift:.3f} | Streak {mean_streak:.1f} | "
                  f"EpLen {ep_len:.0f} | Hold {contact_hold_required}/{CONTACT_HOLD_MAX} | "
                  f"DR {noise_frac*100:.0f}%{freeze_tag} | {time.time()-t0:.1f}s")

            # Best checkpoint
            if suc > best_success_rate:
                best_success_rate = suc
                best_path = f"{CKPT_DIR}/grasp_best.pkl"
                with open(best_path, "wb") as f:
                    pickle.dump(state.params, f)
                print(f"  ★ New best: {suc*100:.1f}% success → {best_path}")

            # Curriculum advance: success-only gate
            if suc >= GRASP_SUCC_GATE:
                consec_good += 1
                consec_bad   = 0
            else:
                consec_good  = 0
                consec_bad  += 1

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

            # Regression rollback
            if (consec_bad >= REGRESSION_WINDOW and
                    suc < REGRESSION_SUC_FLOOR and
                    contact_hold_required > CONTACT_HOLD_START):
                old_req = contact_hold_required
                contact_hold_required -= 1
                consec_bad  = 0
                consec_good = 0
                rollback_path = f"{CKPT_DIR}/grasp_best_h{contact_hold_required:02d}.pkl"
                if not os.path.exists(rollback_path):
                    rollback_path = f"{CKPT_DIR}/grasp_best.pkl"
                if os.path.exists(rollback_path):
                    with open(rollback_path, "rb") as f:
                        restored_params = pickle.load(f)
                    state = state.replace(params=restored_params)
                    print(f"  ↓ ROLLBACK: Hold {old_req} → {contact_hold_required}, "
                          f"restored {rollback_path}")
                else:
                    print(f"  ↓ ROLLBACK: Hold {old_req} → {contact_hold_required} "
                          f"(no checkpoint found)")

            if contact_hold_required == CONTACT_HOLD_MAX and suc >= GRASP_SUCC_GATE:
                print(f"  GRASP PHASE COMPLETE! "
                      f"{contact_hold_required}-step contact at {suc*100:.1f}% success!")

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

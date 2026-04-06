import os, sys, time, pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
import optax
from flax.training.train_state import TrainState
import mujoco
import imageio

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

sys.path.insert(0, os.path.expanduser("~/panda_lego"))
from envs.lego_env import LegoEnv, build_model
from training.encoders import LegoAgent

# Config 
N_ENVS        = 4
N_STEPS       = 32
N_EPOCHS      = 2
CLIP_EPS      = 0.1
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
ENT_COEF      = 0.02
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
LR            = 3e-6
TOTAL_UPDATES = 10000
LOG_INTERVAL  = 10
SAVE_INTERVAL = 100

STD_START     = 0.4
STD_END       = 0.15
STD_DECAY     = 5000
TARGET_POS    = np.array([0.5, 0.15, 0.42])
MAX_EP_STEPS  = 300

# Threshold annealing 
THRESH_START      = 0.05
THRESH_MIN        = 0.05
THRESH_STEP       = 0.005
ANNEAL_SUCC_GATE  = 0.85
ANNEAL_WINDOW     = 5

# Video recording
VIDEO_INTERVAL = 1000
VIDEO_FPS = 30
VIDEO_DIR = os.path.expanduser("~/panda_lego/videos")

# Domain randomization for brick spawn
BRICK_BASE_POS = np.array([0.35, 0.0, 0.42])
DR_WARMUP      = 500          # updates before noise is introduced
DR_XY_MAX      = 0.06         # ±6 cm in X and Y at full randomization
DR_Z_MAX       = 0.01         # ±1 cm in Z (minor table-height variation)
DR_YAW_MAX     = np.deg2rad(20)  # ±20° yaw randomization

# Set to a .pkl path to resume, or None to train from scratch
# RESUME_CHECKPOINT = None
RESUME_CHECKPOINT = os.path.expanduser("~/panda_lego/checkpoints/reach_dr_agent_7600.pkl")

ROBOT_XML  = os.path.expanduser("~/panda_lego/models/mjxpandamerged.xml")
ASSETS_DIR = os.path.expanduser("~/panda_lego/models/assets")
CKPT_DIR   = os.path.expanduser("~/panda_lego/checkpoints")

HOME_QPOS = np.array([
    # Arm: IK-computed pose — palm at [0.35, 0.0, 0.57], 15 cm above brick, fingers pointing forward/inward, min hand Z = 0.46 (4 cm above table)
    -0.0413, -0.5000,  0.1060, -1.8000, -0.2379,  2.0784,  0.6668,
    # Hand: open, fingers uncurled, thumb retracted so nothing pokes downward
    0.0,  0.0,  0.0,  0.0,   # index
    0.0,  0.0,  0.0,  0.0,   # middle
    0.0,  0.0,  0.0,  0.0,   # ring
    0.5,  0.3,  0.5,  0.3,   # thumb
])


def get_brick_start(update, total_updates, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if update <= DR_WARMUP:
        noise_frac = 0.0
    else:
        noise_frac = min((update - DR_WARMUP) / (total_updates - DR_WARMUP), 1.0)

    xy_noise  = rng.uniform(-DR_XY_MAX,   DR_XY_MAX,  size=2) * noise_frac
    z_noise   = rng.uniform(-DR_Z_MAX,    DR_Z_MAX)            * noise_frac
    yaw_noise = rng.uniform(-DR_YAW_MAX,  DR_YAW_MAX)          * noise_frac

    pos = BRICK_BASE_POS + np.array([xy_noise[0], xy_noise[1], z_noise])

    # yaw → quaternion [w, x, y, z]  (rotation about world Z)
    quat = np.array([
        np.cos(yaw_noise / 2),
        0.0,
        0.0,
        np.sin(yaw_noise / 2),
    ])

    return pos, quat


# CPU Environment
class CPUEnv:
    def __init__(self, model):
        self.model   = model
        self.data    = mujoco.MjData(model)
        self.palm_id = model.body("palm").id
        self.nq      = model.nq
        self.nu      = model.nu
        self.obs_dim = 32
        self.act_dim = self.nu
        self.steps   = 0

        self.ctrl_low  = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = model.actuator_ctrlrange[:, 1].copy()
        self.ctrl_mid  = (self.ctrl_low + self.ctrl_high) / 2.0
        self.ctrl_half = (self.ctrl_high - self.ctrl_low) / 2.0
        # Reference ctrl for action normalisation: zero action = hold home pose
        # Clipped to actuator range so HOME_QPOS values outside range are safe
        self.ctrl_ref  = np.clip(HOME_QPOS, self.ctrl_low, self.ctrl_high)

    def reset(self, brick_pos=None, brick_quat=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:23] = HOME_QPOS
        # Position controllers: set ctrl to match qpos so the arm holds its pose at reset instead of immediately torquing toward ctrl_mid
        # The action space is still normalised via ctrl_mid/ctrl_half at step time
        self.data.ctrl[:23] = HOME_QPOS  # arm + hand joints
        self.data.ctrl[23:] = 0.0        # any remaining actuators

        pos  = brick_pos  if brick_pos  is not None else BRICK_BASE_POS.copy()
        quat = brick_quat if brick_quat is not None else np.array([1., 0., 0., 0.])

        self.data.qpos[23:26] = pos
        self.data.qpos[26:30] = quat

        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        robot_qpos = self.data.qpos[:23].copy()
        brick_pos  = self.data.qpos[23:26].copy()
        palm_pos   = self.data.xpos[self.palm_id].copy()
        return np.concatenate([robot_qpos, brick_pos, TARGET_POS, palm_pos])

    def step(self, action, success_thresh):
        action = np.clip(action, -1.0, 1.0)
        ctrl   = self.ctrl_ref + action * self.ctrl_half
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        self.steps += 1

        obs       = self._get_obs()
        brick_pos = self.data.qpos[23:26]
        palm_pos  = self.data.xpos[self.palm_id]
        d_palm    = np.linalg.norm(palm_pos - brick_pos)

        # Shallower decay: stronger gradient signal in the 0.1–0.4m range
        approach_reward = np.exp(-3.0 * d_palm)
        # Two near-bonus rings. 30cm ring omitted — fires from home pose for free
        near_bonus  = 1.5 * (d_palm < 0.15)   # first real goal: 15 cm
        near_bonus += 2.0 * (d_palm < 0.07)   # close approach: 7 cm
        success       = float(d_palm < success_thresh)
        success_bonus = 20.0 * success
        time_penalty  = -0.01
        reward        = approach_reward + near_bonus + success_bonus + time_penalty

        genuine_success = bool(success)
        done = genuine_success or self.steps >= MAX_EP_STEPS
        return obs, reward, done, genuine_success


# GAE
def compute_gae(rewards, values, dones):
    n_steps, n_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    last_adv   = np.zeros(n_envs)
    for t in reversed(range(n_steps)):
        nv       = values[t + 1] if t < n_steps - 1 else np.zeros(n_envs)
        delta    = rewards[t] + GAMMA * nv * (1 - dones[t]) - values[t]
        last_adv = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * last_adv
        advantages[t] = last_adv
    return advantages, advantages + values


# PPO loss 
def ppo_loss(params, agent, obs, tgt, actions, old_lp, advantages, returns, std):
    pred_a, values, _, _ = vmap(lambda o, t: agent.apply(params, o, t))(obs, tgt)
    log_p   = -0.5 * jnp.sum(((actions - pred_a) / std) ** 2, axis=-1)
    ratio   = jnp.exp(log_p - old_lp)
    pg_loss = -jnp.mean(jnp.minimum(
        advantages * ratio,
        advantages * jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS),
    ))
    vf_loss = jnp.mean((values - returns) ** 2)
    entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * std ** 2) * actions.shape[-1]
    return pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy, (pg_loss, vf_loss)

def record_video(params, agent, mj_model, update, success_thresh,
                 width=640, height=480, max_steps=MAX_EP_STEPS):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    out_path = os.path.join(VIDEO_DIR, f"reach_update_{update:05d}.mp4")
    env = CPUEnv(mj_model)
    obs = env.reset(brick_pos=BRICK_BASE_POS.copy(),
                    brick_quat=np.array([1., 0., 0., 0.]))
    tgt = jnp.array(TARGET_POS)

    @jit
    def greedy(params, obs, tgt):
        a, _, _, _ = agent.apply(params, obs, tgt)
        return a

    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    frames = []
    for _ in range(max_steps):
        renderer.update_scene(env.data)
        frames.append(renderer.render())
        a = np.array(greedy(params, jnp.array(obs), tgt))
        obs, _, done, _ = env.step(a, success_thresh)
        if done:
            for _ in range(15):
                renderer.update_scene(env.data)
                frames.append(renderer.render())
            break
    renderer.close()
    imageio.mimwrite(out_path, frames, fps=VIDEO_FPS, quality=7)
    print(f"  → Video saved: {out_path}  ({len(frames)} frames)")

# Training loop
def train():
    print("=== Panda-Lego Phase 1: Reach (with Domain Randomization) ===")
    print(f"JAX devices: {jax.devices()}")

    print("\nBuilding MuJoCo model...")
    mj_model = build_model(ROBOT_XML, ASSETS_DIR)
    envs     = [CPUEnv(mj_model) for _ in range(N_ENVS)]
    print(f"  nq={mj_model.nq}, nu={mj_model.nu}")

    agent  = LegoAgent(act_dim=envs[0].act_dim)
    key    = jax.random.PRNGKey(0)
    params = agent.init(key, jnp.zeros(32), jnp.array(TARGET_POS))
    n_p    = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Agent parameters: {n_p:,}")

    if RESUME_CHECKPOINT is not None:
        with open(RESUME_CHECKPOINT, "rb") as f:
            params = pickle.load(f)
        print(f"  Resumed training from {RESUME_CHECKPOINT}")
    else:
        print("  Training from scratch (fresh random weights)")

    tx    = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR))
    state = TrainState.create(apply_fn=agent.apply, params=params, tx=tx)

    @jit
    def forward(params, obs, tgt):
        a, v, _, _ = agent.apply(params, obs, tgt)
        return a, v

    rng = np.random.default_rng(42)

    # Initial reset at fixed position during warmup
    brick_pos, brick_quat = get_brick_start(1, TOTAL_UPDATES, rng)
    obs_list = [env.reset(brick_pos=brick_pos, brick_quat=brick_quat) for env in envs]
    ep_rews  = np.zeros(N_ENVS)
    ep_lens  = np.zeros(N_ENVS, dtype=int)
    all_rews = []
    tgt      = jnp.array(TARGET_POS)

    success_thresh = THRESH_MIN if RESUME_CHECKPOINT is not None else THRESH_START
    consec_good    = 0

    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"\nTraining: {TOTAL_UPDATES} updates | {N_ENVS} envs | {N_STEPS} steps/update")
    print(f"Domain randomization: warmup {DR_WARMUP} updates, "
          f"then XY ±{DR_XY_MAX*100:.0f}cm / Z ±{DR_Z_MAX*100:.0f}cm / "
          f"yaw ±{np.rad2deg(DR_YAW_MAX):.0f}° ramps to max")
    print(f"Thresh annealing: {THRESH_START} → {THRESH_MIN} "
          f"(step {THRESH_STEP}, gate {ANNEAL_SUCC_GATE*100:.0f}% x {ANNEAL_WINDOW} intervals)")
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

        for s in range(N_STEPS):
            for i, env in enumerate(envs):
                obs      = jnp.array(obs_list[i])
                a, v     = forward(state.params, obs, tgt)
                a        = np.array(a)
                noise    = np.random.normal(0, current_std, a.shape).astype(np.float32)
                a_noisy  = np.clip(a + noise, -1.0, 1.0)
                lp       = float(-0.5 * np.sum(((a_noisy - a) / current_std) ** 2))

                next_obs, rew, done, genuine = env.step(a_noisy, success_thresh)

                obs_buf[s, i]  = obs_list[i]
                act_buf[s, i]  = a_noisy
                logp_buf[s, i] = lp
                val_buf[s, i]  = float(v)
                rew_buf[s, i]  = rew
                done_buf[s, i] = float(done)

                ep_rews[i] += rew
                ep_lens[i] += 1

                if done:
                    all_rews.append((ep_rews[i], float(genuine)))
                    ep_rews[i] = 0.0
                    ep_lens[i] = 0
                    # Each episode gets a freshly randomized brick position
                    new_pos, new_quat = get_brick_start(update, TOTAL_UPDATES, rng)
                    obs_list[i] = env.reset(brick_pos=new_pos, brick_quat=new_quat)
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
            return ppo_loss(
                params, agent,
                jnp.array(obs_flat), jnp.array(tgt_flat),
                jnp.array(act_flat), jnp.array(logp_flat),
                jnp.array(adv_flat), jnp.array(ret_flat),
                current_std,
            )

        grad_fn = jit(value_and_grad(loss_wrapper, has_aux=True))
        for _ in range(N_EPOCHS):
            (loss, aux), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)

        if update % LOG_INTERVAL == 0:
            recent     = all_rews[-50:] if all_rews else []
            mr         = np.mean([r for r, _ in recent]) if recent else 0.0
            suc        = np.mean([s for _, s in recent]) if recent else 0.0
            pg, vf     = aux
            ep_len     = ep_lens.mean()
            d_palms    = [
                np.linalg.norm(obs_buf[s, i, 29:32] - obs_buf[s, i, 23:26])
                for s in range(N_STEPS) for i in range(N_ENVS)
            ]
            mean_dpalm = np.mean(d_palms)
            noise_frac = 0.0 if update <= DR_WARMUP else min(
                (update - DR_WARMUP) / (TOTAL_UPDATES - DR_WARMUP), 1.0)

            print(f"Update {update:4d} | Rew {mr:7.3f} | Succ {suc*100:5.1f}% | "
                  f"PG {float(pg):6.3f} | VF {float(vf):5.1f} | dPalm {mean_dpalm:.3f} | "
                  f"EpLen {ep_len:.0f} | Thresh {success_thresh:.3f} | "
                  f"Std {current_std:.3f} | DR {noise_frac*100:.0f}% | {time.time()-t0:.1f}s")

            if suc >= ANNEAL_SUCC_GATE:
                consec_good += 1
            else:
                consec_good = 0

            if consec_good >= ANNEAL_WINDOW and success_thresh > THRESH_MIN:
                old_thresh     = success_thresh
                success_thresh = round(max(success_thresh - THRESH_STEP, THRESH_MIN), 3)
                consec_good    = 0
                print(f"  Threshold annealed: {old_thresh:.3f} → {success_thresh:.3f}")
                path = f"{CKPT_DIR}/reach_dr_best_{int(old_thresh*1000):03d}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(state.params, f)
                print(f"  → Saved: {path}")

            if success_thresh == THRESH_MIN and suc >= ANNEAL_SUCC_GATE:
                print(f"  FINAL THRESHOLD {THRESH_MIN}m ACHIEVED at {suc*100:.1f}% success!")

        if update % SAVE_INTERVAL == 0:
            path = f"{CKPT_DIR}/reach_dr_agent_{update}.pkl"
            with open(path, "wb") as f:
                pickle.dump(state.params, f)
            print(f"  → Checkpoint: {path}")

        if update % VIDEO_INTERVAL == 0:
            print(f"  Recording video at update {update}...")
            try:
                record_video(state.params, agent, mj_model, update, success_thresh)
            except Exception as e:
                print(f"  Warning: video capture failed ({e})")

    print("\n Reach (DR) training complete!")
    return state


if __name__ == "__main__":
    train()

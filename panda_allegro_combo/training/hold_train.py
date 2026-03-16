import os, sys, time, pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
import optax
from flax.training.train_state import TrainState
import mujoco

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
ENT_COEF      = 0.02       # kept higher for exploration
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
LR            = 8e-6       # same conservative LR as successful reach run
TOTAL_UPDATES = 10000
LOG_INTERVAL  = 10
SAVE_INTERVAL = 100
TARGET_POS    = np.array([0.5, 0.15, 0.42])
MAX_EP_STEPS  = 300

# Distance threshold — fixed at 5cm 
SUCCESS_THRESH = 0.05

# Hold curriculum: start at 1 step, grow to HOLD_STEPS_MAX
# Each time policy hits ANNEAL_SUCC_GATE% for ANNEAL_WINDOW intervals, increase the required hold by HOLD_STEP_SIZE
HOLD_STEPS_START  = 1       # start easy: just 1 step within range = done
HOLD_STEPS_MAX    = 10      # target: 10 consecutive steps (~100ms at typical dt)
HOLD_STEP_SIZE    = 1       # grow by 1 step each anneal
ANNEAL_SUCC_GATE  = 0.85
ANNEAL_WINDOW     = 5

# Reward shaping
APPROACH_SCALE    = 1.0     # exp(-5*d) component
HOLD_BONUS        = 0.5     # per-step bonus for each step palm is within thresh
SUCCESS_BONUS     = 25.0    # large terminal bonus for completing hold
TIME_PENALTY      = -0.005  # tiny per-step cost to prefer fast solutions

ROBOT_XML  = os.path.expanduser("~/panda_lego/models/mjxpandamerged.xml")
ASSETS_DIR = os.path.expanduser("~/panda_lego/models/assets")
CKPT_DIR   = os.path.expanduser("~/panda_lego/checkpoints")

# Resume from the best reach checkpoint
RESUME_PATH = os.path.expanduser("~/panda_lego/checkpoints/agent_10000.pkl")

HOME_QPOS = np.array([
    0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785,
    0.0,  0.0,   0.0,  0.0,
    0.0,  0.0,   0.0,  0.0,
    0.0,  0.0,   0.0,  0.0,
    0.5,  0.0,   0.0,  0.0,
])

# CPU environment with hold counter
class CPUEnv:
    def __init__(self, model):
        self.model      = model
        self.data       = mujoco.MjData(model)
        self.palm_id    = model.body("palm").id
        self.nq         = model.nq
        self.nu         = model.nu
        self.obs_dim    = 33   # +1 for normalised hold_count
        self.act_dim    = self.nu
        self.steps      = 0
        self.hold_count = 0    # consecutive steps within SUCCESS_THRESH

        self.ctrl_low   = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high  = model.actuator_ctrlrange[:, 1].copy()
        self.ctrl_mid   = (self.ctrl_low + self.ctrl_high) / 2.0
        self.ctrl_half  = (self.ctrl_high - self.ctrl_low) / 2.0

    def reset(self, brick_start=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:23] = HOME_QPOS
        self.data.ctrl[:]   = self.ctrl_mid

        start = brick_start if brick_start is not None else np.array([0.35, 0.0, 0.42])
        self.data.qpos[23:26] = start
        self.data.qpos[26:30] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)
        self.steps      = 0
        self.hold_count = 0
        return self._get_obs()

    def _get_obs(self):
        robot_qpos = self.data.qpos[:23].copy()
        brick_pos  = self.data.qpos[23:26].copy()
        palm_pos   = self.data.xpos[self.palm_id].copy()
        # Normalise hold_count to [0,1] so the network can read it
        hold_norm  = np.array([self.hold_count / HOLD_STEPS_MAX])
        return np.concatenate([robot_qpos, brick_pos, TARGET_POS, palm_pos, hold_norm])

    def step(self, action, hold_steps_required):
        action = np.clip(action, -1.0, 1.0)
        ctrl   = self.ctrl_mid + action * self.ctrl_half
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        self.steps += 1

        obs       = self._get_obs()
        brick_pos = self.data.qpos[23:26]
        palm_pos  = self.data.xpos[self.palm_id]
        d_palm    = np.linalg.norm(palm_pos - brick_pos)

        # Update hold counter
        if d_palm < SUCCESS_THRESH:
            self.hold_count += 1
        else:
            self.hold_count = 0   # reset if palm drifts away

        # Reward
        approach_reward = APPROACH_SCALE * np.exp(-5.0 * d_palm)
        holding         = float(d_palm < SUCCESS_THRESH)
        hold_bonus      = HOLD_BONUS * holding            # per-step near bonus
        time_pen        = TIME_PENALTY

        success = self.hold_count >= hold_steps_required
        success_bonus = SUCCESS_BONUS * float(success)

        reward = approach_reward + hold_bonus + success_bonus + time_pen

        done = success or (self.steps >= MAX_EP_STEPS)
        return obs, reward, done, d_palm

# GAE (generalized advantage estimation)
def compute_gae(rewards, values, dones):
    n_steps, n_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    last_adv   = np.zeros(n_envs)
    for t in reversed(range(n_steps)):
        nv       = values[t+1] if t < n_steps-1 else np.zeros(n_envs)
        delta    = rewards[t] + GAMMA * nv * (1 - dones[t]) - values[t]
        last_adv = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * last_adv
        advantages[t] = last_adv
    return advantages, advantages + values

# PPO loss
def ppo_loss(params, agent, obs, tgt, actions, old_lp, advantages, returns):
    pred_a, values, _, _ = vmap(lambda o, t: agent.apply(params, o, t))(obs, tgt)
    std     = 0.15                         # same floor as reach run
    log_p   = -0.5 * jnp.sum(((actions - pred_a) / std)**2, axis=-1)
    ratio   = jnp.exp(log_p - old_lp)
    pg_loss = -jnp.mean(jnp.minimum(
        advantages * ratio,
        advantages * jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
    ))
    vf_loss = jnp.mean((values - returns)**2)
    entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * std**2) * actions.shape[-1]
    return pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy, (pg_loss, vf_loss)

# Train
def train():
    print("=== Panda-Lego Phase 2: Reach + Hold ===")
    print(f"JAX devices: {jax.devices()}")

    print("\nBuilding MuJoCo model...")
    mj_model = build_model(ROBOT_XML, ASSETS_DIR)
    envs     = [CPUEnv(mj_model) for _ in range(N_ENVS)]
    print(f"  nq={mj_model.nq}, nu={mj_model.nu}")

    # keep obs_dim=32 and don't include hold_count in obs

    OBS_DIM = 32   # match original network — hold_norm stripped from obs
    agent   = LegoAgent(act_dim=envs[0].act_dim)
    key     = jax.random.PRNGKey(0)
    params  = agent.init(key, jnp.zeros(OBS_DIM), jnp.array(TARGET_POS))
    n_p     = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Agent parameters: {n_p:,}")

    # Resume
    if os.path.exists(RESUME_PATH):
        with open(RESUME_PATH, "rb") as f:
            params = pickle.load(f)
        print(f" Resumed from {RESUME_PATH}")
    else:
        print(f" Checkpoint not found at {RESUME_PATH}, starting fresh")

    tx    = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR))
    state = TrainState.create(apply_fn=agent.apply, params=params, tx=tx)

    @jit
    def forward(params, obs, tgt):
        a, v, _, _ = agent.apply(params, obs, tgt)
        return a, v

    brick_start = np.array([0.35, 0.0, 0.42])
    # Pass only the first 32 dims to the network (strip hold_norm)
    obs_list = [env.reset(brick_start=brick_start)[:OBS_DIM] for env in envs]
    # Store full obs internally for env, truncate for network
    full_obs_list = [env.reset(brick_start=brick_start) for env in envs]
    obs_list      = [o[:OBS_DIM] for o in full_obs_list]

    ep_rews  = np.zeros(N_ENVS)
    ep_lens  = np.zeros(N_ENVS, dtype=int)
    all_rews = []
    tgt      = jnp.array(TARGET_POS)

    # Hold curriculum state
    hold_steps_required = HOLD_STEPS_START
    consec_good         = 0

    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"\nTraining: {TOTAL_UPDATES} updates | {N_ENVS} envs | {N_STEPS} steps/update")
    print(f"Hold curriculum: {HOLD_STEPS_START} → {HOLD_STEPS_MAX} steps "
          f"(gate {ANNEAL_SUCC_GATE*100:.0f}% x {ANNEAL_WINDOW} intervals)")
    print(f"Distance threshold: fixed {SUCCESS_THRESH}m")
    print("-" * 60)

    for update in range(1, TOTAL_UPDATES + 1):
        t0 = time.time()

        obs_buf  = np.zeros((N_STEPS, N_ENVS, OBS_DIM),           dtype=np.float32)
        act_buf  = np.zeros((N_STEPS, N_ENVS, envs[0].act_dim),   dtype=np.float32)
        logp_buf = np.zeros((N_STEPS, N_ENVS),                    dtype=np.float32)
        val_buf  = np.zeros((N_STEPS, N_ENVS),                    dtype=np.float32)
        rew_buf  = np.zeros((N_STEPS, N_ENVS),                    dtype=np.float32)
        done_buf = np.zeros((N_STEPS, N_ENVS),                    dtype=np.float32)
        dpalm_buf = np.zeros((N_STEPS, N_ENVS),                   dtype=np.float32)

        std = 0.15   # fixed floor

        for s in range(N_STEPS):
            for i, env in enumerate(envs):
                obs = jnp.array(obs_list[i])
                a, v = forward(state.params, obs, tgt)
                a = np.array(a)
                noise   = np.random.normal(0, std, a.shape).astype(np.float32)
                a_noisy = np.clip(a + noise, -1.0, 1.0)
                lp      = float(-0.5 * np.sum(((a_noisy - a) / std)**2))

                next_full_obs, rew, done, d_palm = env.step(a_noisy, hold_steps_required)
                next_obs = next_full_obs[:OBS_DIM]

                obs_buf[s, i]   = obs_list[i]
                act_buf[s, i]   = a_noisy
                logp_buf[s, i]  = lp
                val_buf[s, i]   = float(v)
                rew_buf[s, i]   = rew
                done_buf[s, i]  = float(done)
                dpalm_buf[s, i] = d_palm

                ep_rews[i] += rew
                ep_lens[i] += 1

                if done:
                    all_rews.append(ep_rews[i])
                    ep_rews[i] = 0.0
                    ep_lens[i] = 0
                    new_obs = env.reset(brick_start=brick_start)
                    obs_list[i] = new_obs[:OBS_DIM]
                else:
                    obs_list[i] = next_obs

        # GAE
        adv, ret = compute_gae(rew_buf, val_buf, done_buf)
        adv_flat = adv.flatten().astype(np.float32)
        ret_flat = ret.flatten().astype(np.float32)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        obs_flat  = obs_buf.reshape(-1, OBS_DIM)
        act_flat  = act_buf.reshape(-1, envs[0].act_dim)
        logp_flat = logp_buf.flatten()
        tgt_flat  = np.tile(TARGET_POS, (N_STEPS * N_ENVS, 1)).astype(np.float32)

        def loss_wrapper(params):
            return ppo_loss(params, agent,
                            jnp.array(obs_flat), jnp.array(tgt_flat),
                            jnp.array(act_flat), jnp.array(logp_flat),
                            jnp.array(adv_flat), jnp.array(ret_flat))

        grad_fn = jit(value_and_grad(loss_wrapper, has_aux=True))
        for _ in range(N_EPOCHS):
            (loss, aux), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)

        if update % LOG_INTERVAL == 0:
            mr      = np.mean(all_rews[-50:]) if all_rews else 0.0
            # Success = episode reward > threshold that indicates hold completed
            # Use a heuristic: reward > (SUCCESS_BONUS * 0.8) in last 50 eps
            suc_threshold = SUCCESS_BONUS * 0.8
            suc     = np.mean([r > suc_threshold for r in all_rews[-50:]]) if all_rews else 0.0
            pg, vf  = aux
            ep_len  = ep_lens.mean()
            mean_dpalm = dpalm_buf.mean()

            print(f"Update {update:5d} | Rew {mr:7.2f} | Succ {suc*100:5.1f}% | "
                  f"PG {float(pg):6.3f} | VF {float(vf):5.1f} | "
                  f"dPalm {mean_dpalm:.3f} | EpLen {ep_len:.0f} | "
                  f"Hold {hold_steps_required}/{HOLD_STEPS_MAX} | {time.time()-t0:.1f}s")

            # Curriculum: increase hold requirement
            if suc >= ANNEAL_SUCC_GATE:
                consec_good += 1
            else:
                consec_good = 0

            if consec_good >= ANNEAL_WINDOW and hold_steps_required < HOLD_STEPS_MAX:
                old_hold = hold_steps_required
                hold_steps_required = min(hold_steps_required + HOLD_STEP_SIZE,
                                          HOLD_STEPS_MAX)
                consec_good = 0
                print(f" Hold requirement increased: {old_hold} → {hold_steps_required} steps")
                path = f"{CKPT_DIR}/hold_best_h{old_hold:02d}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(state.params, f)
                print(f"  → Saved: {path}")

            if hold_steps_required == HOLD_STEPS_MAX and suc >= ANNEAL_SUCC_GATE:
                print(f" HOLD PHASE COMPLETE! {HOLD_STEPS_MAX}-step hold at {suc*100:.1f}% success!")

        if update % SAVE_INTERVAL == 0:
            path = f"{CKPT_DIR}/hold_agent_{update}.pkl"
            with open(path, "wb") as f:
                pickle.dump(state.params, f)
            print(f"  → Checkpoint: {path}")

    print("\n Hold phase training complete!")
    return state

if __name__ == "__main__":
    train()
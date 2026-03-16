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
ENT_COEF      = 0.02   # raised from 0.01 to force more exploration
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
LR            = 8e-6   # lowered from 3e-5 (was too aggressive post-curriculum)
TOTAL_UPDATES = 10000
LOG_INTERVAL  = 10
SAVE_INTERVAL = 100

# action std schedule — start high for exploration, decay slowly
STD_START     = 0.4    # slightly higher than original 0.3 to re-explore
STD_END       = 0.15   # floor: still precise enough for 5cm task
STD_DECAY     = 5000   # number of updates to decay over
TARGET_POS    = np.array([0.5, 0.15, 0.42])
MAX_EP_STEPS  = 300

# Auto-annealing: start at 0.08, shrink by 0.005 each time
# policy achieves >= ANNEAL_SUCC_GATE% for ANNEAL_WINDOW consecutive log intervals
THRESH_START      = 0.08
THRESH_MIN        = 0.05
THRESH_STEP       = 0.005
ANNEAL_SUCC_GATE  = 0.85   # must hit 90% success to shrink
ANNEAL_WINDOW     = 5      # must hold it for 5 consecutive log intervals

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

def get_brick_start(update, total_updates):
    return np.array([0.35, 0.0, 0.42])

# CPU environment — success threshold passed in at step time
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

    def reset(self, brick_start=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:23] = HOME_QPOS
        self.data.ctrl[:] = self.ctrl_mid

        start = brick_start if brick_start is not None else np.array([0.5, 0.0, 0.44])
        self.data.qpos[23:26] = start
        self.data.qpos[26:30] = [1, 0, 0, 0]

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
        ctrl   = self.ctrl_mid + action * self.ctrl_half
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        self.steps += 1

        obs       = self._get_obs()
        brick_pos = self.data.qpos[23:26]
        palm_pos  = self.data.xpos[self.palm_id]

        d_palm  = np.linalg.norm(palm_pos - brick_pos)

        approach_reward = np.exp(-5.0 * d_palm) #- 1.0
        near_bonus = 1.0 * (d_palm < 0.07)
        success = float(d_palm < success_thresh)
        success_bonus = 20.0 * success
        time_penalty = -0.01
        reward = approach_reward + near_bonus + success_bonus + time_penalty
        #reward = float(np.clip(approach_reward + 10.0 * success, -1.0, 10.0))

        done = bool(success) or self.steps >= MAX_EP_STEPS
        return obs, reward, done

# GAE (Generalized Advantage Estimation)
def compute_gae(rewards, values, dones):
    n_steps, n_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    last_adv   = np.zeros(n_envs)
    for t in reversed(range(n_steps)):
        nv       = values[t+1] if t < n_steps-1 else np.zeros(n_envs)
        delta    = rewards[t] + GAMMA * nv * (1-dones[t]) - values[t]
        last_adv = delta + GAMMA * GAE_LAMBDA * (1-dones[t]) * last_adv
        advantages[t] = last_adv
    return advantages, advantages + values

# PPO loss
def ppo_loss(params, agent, obs, tgt, actions, old_lp, advantages, returns, std):
    pred_a, values, _, _ = vmap(lambda o, t: agent.apply(params, o, t))(obs, tgt)
    log_p   = -0.5 * jnp.sum(((actions - pred_a) / std)**2, axis=-1)
    ratio   = jnp.exp(log_p - old_lp)
    pg_loss = -jnp.mean(jnp.minimum(
        advantages * ratio,
        advantages * jnp.clip(ratio, 1-CLIP_EPS, 1+CLIP_EPS)
    ))
    vf_loss = jnp.mean((values - returns)**2)
    # entropy is now meaningful because std changes over training
    entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * std**2) * actions.shape[-1]
    return pg_loss + VF_COEF*vf_loss - ENT_COEF*entropy, (pg_loss, vf_loss)

# Train
def train():
    print("=== Panda-Lego Pipeline (CPU sim + GPU nets) ===")
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

    resume_path = os.path.expanduser("~/panda_lego/checkpoints/agent_6700.pkl")
    with open(resume_path, "rb") as f:
        params = pickle.load(f)
    print(f" Resumed training from {resume_path}")

    # fresh optimizer at lower LR — don't inherit momentum from collapsed run
    tx    = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR))
    state = TrainState.create(apply_fn=agent.apply, params=params, tx=tx)

    @jit
    def forward(params, obs, tgt):
        a, v, _, _ = agent.apply(params, obs, tgt)
        return a, v

    brick_start = get_brick_start(1, TOTAL_UPDATES)
    obs_list = [env.reset(brick_start=brick_start) for env in envs]
    ep_rews  = np.zeros(N_ENVS)
    ep_lens  = np.zeros(N_ENVS, dtype=int)
    all_rews = []
    tgt      = jnp.array(TARGET_POS)

    # resume at 0.055 (one step above the threshold that caused collapse)
    # this gives the policy a chance to stabilise before hitting 0.05 again
    success_thresh = 0.055
    consec_good    = 0

    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"\nTraining: {TOTAL_UPDATES} updates | {N_ENVS} envs | {N_STEPS} steps/update")
    print(f"Thresh annealing: {THRESH_START} → {THRESH_MIN} (step {THRESH_STEP}, gate {ANNEAL_SUCC_GATE*100:.0f}% x {ANNEAL_WINDOW} intervals)")
    print("-" * 60)

    for update in range(1, TOTAL_UPDATES + 1):
        t0 = time.time()
        brick_start = get_brick_start(update, TOTAL_UPDATES)

        # linearly decay std from STD_START → STD_END over STD_DECAY updates
        # this gives early exploration then tightens the policy as it stabilises
        frac = min(update / STD_DECAY, 1.0)
        current_std = STD_START + frac * (STD_END - STD_START)

        obs_buf  = np.zeros((N_STEPS, N_ENVS, 32),              dtype=np.float32)
        act_buf  = np.zeros((N_STEPS, N_ENVS, envs[0].act_dim), dtype=np.float32)
        logp_buf = np.zeros((N_STEPS, N_ENVS),                  dtype=np.float32)
        val_buf  = np.zeros((N_STEPS, N_ENVS),                  dtype=np.float32)
        rew_buf  = np.zeros((N_STEPS, N_ENVS),                  dtype=np.float32)
        done_buf = np.zeros((N_STEPS, N_ENVS),                  dtype=np.float32)

        for s in range(N_STEPS):
            for i, env in enumerate(envs):
                obs = jnp.array(obs_list[i])
                a, v = forward(state.params, obs, tgt)
                a = np.array(a)
                noise   = np.random.normal(0, current_std, a.shape).astype(np.float32)
                a_noisy = np.clip(a + noise, -1.0, 1.0)
                lp = float(-0.5 * np.sum(((a_noisy - a) / current_std)**2))

                next_obs, rew, done = env.step(a_noisy, success_thresh)

                obs_buf[s, i]  = obs_list[i]
                act_buf[s, i]  = a_noisy
                logp_buf[s, i] = lp
                val_buf[s, i]  = float(v)
                rew_buf[s, i]  = rew
                done_buf[s, i] = float(done)

                ep_rews[i] += rew
                ep_lens[i] += 1

                if done:
                    all_rews.append(ep_rews[i])
                    ep_rews[i] = 0.0
                    ep_lens[i] = 0
                    obs_list[i] = env.reset(brick_start=brick_start)
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
            mr  = np.mean(all_rews[-50:]) if all_rews else 0.0
            suc = np.mean([r > 2.0 for r in all_rews[-50:]]) if all_rews else 0.0
            pg, vf = aux
            ep_len = ep_lens.mean()
            d_palms = [np.linalg.norm(obs_buf[s, i, 29:32] - obs_buf[s, i, 23:26])
                       for s in range(N_STEPS) for i in range(N_ENVS)]
            mean_dpalm = np.mean(d_palms)

            print(f"Update {update:4d} | Rew {mr:7.3f} | Succ {suc*100:5.1f}% | "
                  f"PG {float(pg):6.3f} | VF {float(vf):5.1f} | dPalm {mean_dpalm:.3f} | "
                  f"EpLen {ep_len:.0f} | Thresh {success_thresh:.3f} | Std {current_std:.3f} | {time.time()-t0:.1f}s")

            # Auto-anneal logic
            if suc >= ANNEAL_SUCC_GATE:
                consec_good += 1
            else:
                consec_good = 0

            if consec_good >= ANNEAL_WINDOW and success_thresh > THRESH_MIN:
                old_thresh = success_thresh
                success_thresh = round(max(success_thresh - THRESH_STEP, THRESH_MIN), 3)
                consec_good = 0
                print(f" Threshold annealed: {old_thresh:.3f} → {success_thresh:.3f}")
                path = f"{CKPT_DIR}/best_{int(old_thresh*1000):03d}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(state.params, f)
                print(f"  → Saved: {path}")

            if success_thresh == THRESH_MIN and suc >= ANNEAL_SUCC_GATE:
                print(f" FINAL THRESHOLD {THRESH_MIN}m ACHIEVED at {suc*100:.1f}% success!")

        if update % SAVE_INTERVAL == 0:
            path = f"{CKPT_DIR}/agent_{update}.pkl"
            with open(path, "wb") as f:
                pickle.dump(state.params, f)
            print(f"  → Checkpoint: {path}")

    print("\n Training complete!")
    return state

if __name__ == "__main__":
    train()
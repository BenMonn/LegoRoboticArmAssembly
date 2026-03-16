"""
training/pretrain_bc.py  (v3 - Cartesian proportional controller demos)

Generates demos using a closed-loop Cartesian controller:
  Each step: compute Jacobian, compute palm->brick error,
  compute joint velocity via J^T * err, convert to normalised action.

This produces (obs, action) pairs where action actually drives the palm
toward the brick from the current state — true closed-loop behavior.
"""

import os, sys, time, pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import optax
import mujoco

sys.path.insert(0, os.path.expanduser("~/panda_lego"))
from envs.lego_env import build_model
from training.encoders import LegoAgent

ROBOT_XML  = os.path.expanduser("~/panda_lego/models/mjxpandamerged.xml")
ASSETS_DIR = os.path.expanduser("~/panda_lego/models/assets")
CKPT_DIR   = os.path.expanduser("~/panda_lego/checkpoints")
TARGET_POS = np.array([0.5, 0.15, 0.42])

HOME_QPOS = np.array([
    0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785,
    0.0,  0.0,   0.0,  0.0,
    0.0,  0.0,   0.0,  0.0,
    0.0,  0.0,   0.0,  0.0,
    0.5,  0.0,   0.0,  0.0,
])

BRICK_X_RANGE = (0.30, 0.50)
BRICK_Y_RANGE = (-0.15, 0.15)
BRICK_Z       = 0.42

N_DEMOS    = 500
SIM_STEPS  = 300     # physics steps per demo
TRAJ_LEN   = 60      # samples saved per demo
CART_GAIN  = 3.0     # Cartesian proportional gain
ACTION_SCALE = 2.0   # amplify joint velocity signal
SUCCESS_D  = 0.03    # stop demo early if this close

BC_EPOCHS  = 300
BC_LR      = 3e-4
BATCH_SIZE = 256


def generate_demo(model, data, palm_id, brick_pos):
    """
    Closed-loop Cartesian proportional controller demo.
    Each step computes J^T * (brick - palm) as joint velocity,
    scales and clips to action range, steps physics.
    """
    ctrl_low  = model.actuator_ctrlrange[:, 0]
    ctrl_high = model.actuator_ctrlrange[:, 1]
    ctrl_mid  = (ctrl_low + ctrl_high) / 2.0
    ctrl_half = (ctrl_high - ctrl_low) / 2.0

    mujoco.mj_resetData(model, data)
    data.qpos[:23] = HOME_QPOS
    data.qpos[23:26] = brick_pos
    data.qpos[26:30] = [1, 0, 0, 0]
    data.ctrl[:]  = ctrl_mid   # start at midpoint
    mujoco.mj_forward(model, data)

    trajectory = []
    min_dist = 999.0

    for step in range(SIM_STEPS):
        mujoco.mj_forward(model, data)
        palm = data.xpos[palm_id].copy()
        err  = brick_pos - palm
        dist = np.linalg.norm(err)
        min_dist = min(min_dist, dist)

        # Cartesian Jacobian for palm
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, None, palm_id)
        J = jacp[:, :7]  # arm joints only

        # Joint velocity via J^T * cartesian_error
        dq = J.T @ (err * CART_GAIN)

        # Convert joint velocity to normalised action:
        # Current ctrl + delta, then normalise to [-1,1]
        target_qpos = data.qpos[:7] + dq * ACTION_SCALE
        for i in range(7):
            lo, hi = model.jnt_range[i]
            target_qpos[i] = np.clip(target_qpos[i], lo, hi)

        target_ctrl = np.zeros(model.nu)
        target_ctrl[:7] = target_qpos
        target_ctrl[7:] = HOME_QPOS[7:23]
        action = (target_ctrl - ctrl_mid) / (ctrl_half + 1e-8)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Record obs + action
        robot_qpos = data.qpos[:23].copy()
        brick      = data.qpos[23:26].copy()
        obs = np.concatenate([robot_qpos, brick, TARGET_POS, palm]).astype(np.float32)
        trajectory.append((obs, action))

        # Apply action to physics
        data.ctrl[:7] = target_qpos
        data.ctrl[7:] = HOME_QPOS[7:23]
        mujoco.mj_step(model, data)

        if dist < SUCCESS_D:
            break

    # Subsample
    idx = np.linspace(0, len(trajectory)-1, TRAJ_LEN, dtype=int)
    trajectory = [trajectory[i] for i in idx]
    return trajectory, min_dist


def generate_dataset(model):
    data    = mujoco.MjData(model)
    palm_id = model.body("palm").id
    obs_list, act_list = [], []
    successes = 0

    print(f"\n=== Phase 1: Generating {N_DEMOS} Cartesian controller demos ===")
    t0 = time.time()

    for i in range(N_DEMOS):
        bx = np.random.uniform(*BRICK_X_RANGE)
        by = np.random.uniform(*BRICK_Y_RANGE)
        brick_pos = np.array([bx, by, BRICK_Z])

        traj, min_dist = generate_demo(model, data, palm_id, brick_pos)
        for obs, act in traj:
            obs_list.append(obs)
            act_list.append(act)
        if min_dist < SUCCESS_D * 2:
            successes += 1

        if (i + 1) % 50 == 0:
            print(f"  Demo {i+1:4d}/{N_DEMOS} | "
                  f"Reach rate: {successes/(i+1)*100:.0f}% | "
                  f"Samples: {len(obs_list):,} | {time.time()-t0:.0f}s")

    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.float32)
    print(f"\nDataset: {obs_arr.shape[0]:,} samples | "
          f"Reached: {successes}/{N_DEMOS} ({successes/N_DEMOS*100:.0f}%)")
    return obs_arr, act_arr


def bc_loss(params, agent, obs, tgt, actions):
    pred_a, _, _, _ = vmap(lambda o, t: agent.apply(params, o, t))(obs, tgt)
    return jnp.mean((pred_a - actions) ** 2)


def pretrain_bc(agent, params, obs_data, act_data):
    print(f"\n=== Phase 2: BC pretraining ({BC_EPOCHS} epochs) ===")
    tx        = optax.adam(BC_LR)
    opt_state = tx.init(params)
    tgt_data  = np.tile(TARGET_POS, (obs_data.shape[0], 1)).astype(np.float32)
    grad_fn   = jit(value_and_grad(lambda p, o, t, a: bc_loss(p, agent, o, t, a)))

    n = obs_data.shape[0]
    best_loss, best_params = float('inf'), params

    for epoch in range(BC_EPOCHS):
        idx = np.random.permutation(n)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            o = jnp.array(obs_data[idx[start:end]])
            a = jnp.array(act_data[idx[start:end]])
            t = jnp.array(tgt_data[idx[start:end]])
            loss, grads = grad_fn(params, o, t, a)
            updates, opt_state_new = tx.update(grads, opt_state)
            params    = optax.apply_updates(params, updates)
            opt_state = opt_state_new
            epoch_loss += float(loss); n_batches += 1

        avg = epoch_loss / n_batches
        if avg < best_loss:
            best_loss, best_params = avg, params
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:4d}/{BC_EPOCHS} | Loss {avg:.6f} | Best {best_loss:.6f}")

    print(f"\nBC done. Best loss: {best_loss:.6f}")
    return best_params


def main():
    print("=== BC Pretraining (Cartesian controller) ===")
    print(f"JAX devices: {jax.devices()}")
    print("\nBuilding MuJoCo model...")
    model = build_model(ROBOT_XML, ASSETS_DIR)
    print(f"  nq={model.nq}, nu={model.nu}")

    agent  = LegoAgent(act_dim=model.nu)
    key    = jax.random.PRNGKey(42)
    params = agent.init(key, jnp.zeros(32), jnp.array(TARGET_POS))
    n_p    = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Agent parameters: {n_p:,}")

    obs_data, act_data = generate_dataset(model)
    params = pretrain_bc(agent, params, obs_data, act_data)

    os.makedirs(CKPT_DIR, exist_ok=True)
    path = f"{CKPT_DIR}/bc_pretrained.pkl"
    with open(path, "wb") as f:
        pickle.dump(params, f)
    print(f"\n✅ Saved: {path}")
    print("   Now run: python ~/panda_lego/training/train.py")


if __name__ == "__main__":
    main()
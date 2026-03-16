import os
import re
import numpy as np
import mujoco
import mujoco.mjx as mjx
import jax
import jax.numpy as jnp
from jax import jit, vmap

# Task bodies to inject into the robot XML

TASK_BODIES = """
    <!-- Table -->
    <body name="table" pos="0.5 0 0.4">
      <geom name="table_top" type="box" size="0.4 0.4 0.02"
            rgba="0.8 0.7 0.6 1" contype="1" conaffinity="1"/>
    </body>

    <!-- LEGO brick placeholder (standard 2x4 brick approx dimensions) -->
    <body name="brick" pos="0.5 0.0 0.44">
      <freejoint name="brick_joint"/>
      <inertial mass="0.01" pos="0 0 0" diaginertia="1e-5 1e-5 1e-5"/>
      <geom name="brick_geom" type="box" size="0.016 0.032 0.0096"
            rgba="1.0 0.2 0.2 1" contype="1" conaffinity="1"/>
      <site name="brick_site" pos="0 0 0" size="0.005"/>
    </body>

    <!-- Target marker (visual only, no collision) -->
    <body name="target_marker" pos="0.5 0.15 0.42">
      <geom name="target_geom" type="box" size="0.016 0.032 0.001"
            rgba="0.0 1.0 0.0 0.4" contype="0" conaffinity="0"/>
    </body>
"""

# Build the combined model

def build_model(robot_xml_path: str, assets_dir: str) -> mujoco.MjModel:
    with open(robot_xml_path, "r") as f:
        xml_str = f.read()

    # Fix meshdir to absolute path
    assets_abs = os.path.abspath(assets_dir)
    xml_str = re.sub(r'meshdir="[^"]*"', f'meshdir="{assets_abs}"', xml_str)
    # Disable all collision geometry for training (saves huge amounts of MJX memory)
    xml_str = xml_str.replace('contype="1" conaffinity="1"', 'contype="0" conaffinity="0"')

    # Inject task bodies just before </worldbody>
    xml_str = xml_str.replace("</worldbody>", TASK_BODIES + "\n  </worldbody>")

    # Write temp file in same directory as robot XML so relative paths resolve
    tmp_path = os.path.join(
        os.path.dirname(os.path.abspath(robot_xml_path)),
        "_lego_env_tmp.xml"
    )
    with open(tmp_path, "w") as f:
        f.write(xml_str)

    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return model

# Environment class
class LegoEnv:

    BRICK_START_POS = np.array([0.5,  0.0,  0.44])
    TARGET_POS      = np.array([0.5,  0.15, 0.42])
    SUCCESS_THRESH  = 0.02   # metres
    MAX_STEPS       = 500

    def __init__(self, robot_xml_path: str, assets_dir: str):
        print("Building MuJoCo model...")
        self.mj_model = build_model(robot_xml_path, assets_dir)
        self.mj_data  = mujoco.MjData(self.mj_model)

        # Upload model to GPU
        self.mjx_model = mjx.put_model(self.mj_model)

        # Body IDs
        self.brick_body_id  = self.mj_model.body("brick").id
        self.palm_body_id   = self.mj_model.body("palm").id
        self.target_body_id = self.mj_model.body("target_marker").id

        # Dimensions
        self.nq      = self.mj_model.nq
        self.nv      = self.mj_model.nv
        self.nu      = self.mj_model.nu
        self.obs_dim = 32
        self.act_dim = self.nu

        print(f"  nq={self.nq}, nv={self.nv}, nu={self.nu}")
        print(f"  obs_dim={self.obs_dim}, act_dim={self.act_dim}")
        print("  Model built successfully.")

    # Reset
    def reset(self) -> mjx.Data:
        """Reset to home config. Returns MJX data on GPU."""
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        home_qpos = np.array([
            0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785,  # panda
            0.0,  0.0,   0.0,  0.0,                        # index finger
            0.0,  0.0,   0.0,  0.0,                        # middle finger
            0.0,  0.0,   0.0,  0.0,                        # ring finger
            0.5,  0.0,   0.0,  0.0,                        # thumb
        ])
        self.mj_data.qpos[:23] = home_qpos

        # Brick freejoint: 7 values (pos xyz + quat wxyz)
        brick_start = 23
        self.mj_data.qpos[brick_start:brick_start+3] = self.BRICK_START_POS
        self.mj_data.qpos[brick_start+3:brick_start+7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.mj_model, self.mj_data)
        return mjx.put_data(self.mj_model, self.mj_data)

    # Observation
    def get_obs(self, mjx_data: mjx.Data) -> jnp.ndarray:
        robot_qpos = mjx_data.qpos[:23]
        brick_pos  = mjx_data.qpos[23:26]
        target_pos = jnp.array(self.TARGET_POS)
        palm_pos   = mjx_data.xpos[self.palm_body_id]
        return jnp.concatenate([robot_qpos, brick_pos, target_pos, palm_pos])

    # Reward
    def compute_reward(self, mjx_data: mjx.Data) -> jnp.ndarray:
        brick_pos  = mjx_data.qpos[23:26]
        palm_pos   = mjx_data.xpos[self.palm_body_id]
        target_pos = jnp.array(self.TARGET_POS)

        d_palm_brick   = jnp.linalg.norm(palm_pos - brick_pos)
        d_brick_target = jnp.linalg.norm(brick_pos - target_pos)
        success        = jnp.float32(d_brick_target < self.SUCCESS_THRESH)

        return -d_palm_brick - d_brick_target + 5.0 * success

    # Step
    @staticmethod
    def _step_fn(mjx_model, mjx_data, action):
        action   = jnp.clip(action, -1.0, 1.0)
        mjx_data = mjx_data.replace(ctrl=action)
        mjx_data = mjx.step(mjx_model, mjx_data)
        return mjx_data

    def step(self, mjx_data: mjx.Data, action: jnp.ndarray):
        mjx_data  = self._step_fn(self.mjx_model, mjx_data, action)
        obs       = self.get_obs(mjx_data)
        reward    = self.compute_reward(mjx_data)
        brick_pos = mjx_data.qpos[23:26]
        done = jnp.float32(
            jnp.linalg.norm(brick_pos - jnp.array(self.TARGET_POS)) < self.SUCCESS_THRESH
        )
        return mjx_data, obs, reward, done

# Batched environment (N parallel envs on GPU)
class BatchedLegoEnv:
    def __init__(self, base_env: LegoEnv, n_envs: int = 8):
        self.env      = base_env
        self.n_envs   = n_envs
        self._jit_step = jit(vmap(LegoEnv._step_fn, in_axes=(None, 0, 0)))

    def reset_all(self) -> mjx.Data:
        base_data = self.env.reset()
        return jax.tree_util.tree_map(
            lambda x: jnp.stack([x] * self.n_envs), base_data
        )

    def step_all(self, batched_data: mjx.Data, actions: jnp.ndarray) -> mjx.Data:
        return self._jit_step(self.env.mjx_model, batched_data, actions)

# Quick test
if __name__ == "__main__":
    ROBOT_XML  = os.path.expanduser("~/panda_lego/models/mjxpandamerged.xml")
    ASSETS_DIR = os.path.expanduser("~/panda_lego/models/assets")

    print("=== Testing LegoEnv ===")
    env = LegoEnv(ROBOT_XML, ASSETS_DIR)

    state = env.reset()
    print(f"Reset OK  -- qpos shape: {state.qpos.shape}")

    obs = env.get_obs(state)
    print(f"Obs shape : {obs.shape}")
    print(f"  Panda joints (7) : {np.array(obs[:7]).round(3)}")
    print(f"  Brick pos    (3) : {np.array(obs[23:26]).round(3)}")
    print(f"  Target pos   (3) : {np.array(obs[26:29]).round(3)}")
    print(f"  Palm pos     (3) : {np.array(obs[29:32]).round(3)}")

    action = jnp.zeros(env.act_dim)
    state, obs, reward, done = env.step(state, action)
    print(f"Step OK   -- reward={float(reward):.4f}, done={float(done):.0f}")

    print("\n=== Testing BatchedLegoEnv (8 envs) ===")
    batched_env   = BatchedLegoEnv(env, n_envs=8)
    batched_state = batched_env.reset_all()
    print(f"Batched qpos shape: {batched_state.qpos.shape}")

    batch_actions = jnp.zeros((8, env.act_dim))
    batched_state = batched_env.step_all(batched_state, batch_actions)
    print("Batched step OK")

    print("\n✅ All environment tests passed!")
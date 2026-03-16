import os, sys, pickle, argparse
import numpy as np
import mujoco
import jax
import jax.numpy as jnp
from jax import jit
import imageio

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Force CPU-only MuJoCo rendering (no display needed)
os.environ["MUJOCO_GL"] = "osmesa"   # or "egl" if osmesa not available

sys.path.insert(0, os.path.expanduser("~/panda_lego"))
from envs.lego_env import LegoEnv, build_model
from training.encoders import LegoAgent

# Config (must match training)
TARGET_POS = np.array([0.5, 0.15, 0.42])
MAX_EP_STEPS = 300
SUCCESS_THRESH = 0.05
HOLD_STEPS_REQUIRED = 10
OBS_DIM = 32

HOME_QPOS = np.array([
    0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785,
    0.0,  0.0,   0.0,  0.0,
    0.0,  0.0,   0.0,  0.0,
    0.0,  0.0,   0.0,  0.0,
    0.5,  0.0,   0.0,  0.0,
])

BRICK_START = np.array([0.35, 0.0, 0.42])

ROBOT_XML  = os.path.expanduser("~/panda_lego/models/mjxpandamerged.xml")
ASSETS_DIR = os.path.expanduser("~/panda_lego/models/assets")

# Camera angles (try multiple for best view)
CAMERAS = [
    # (azimuth, elevation, distance, lookat_xyz)
    (135, -25, 1.8, [0.45, 0.1, 0.45]),   # main 3/4 view
    (90,  -20, 1.5, [0.40, 0.0, 0.44]),   # side view
    (180, -30, 1.6, [0.42, 0.1, 0.44]),   # front view
]
SELECTED_CAM = 0   # change to 1 or 2 to try different angles

# Environment
class RenderEnv:
    def __init__(self, model):
        self.model      = model
        self.data       = mujoco.MjData(model)
        self.palm_id    = model.body("palm").id
        self.ctrl_low   = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high  = model.actuator_ctrlrange[:, 1].copy()
        self.ctrl_mid   = (self.ctrl_low + self.ctrl_high) / 2.0
        self.ctrl_half  = (self.ctrl_high - self.ctrl_low) / 2.0
        self.steps      = 0
        self.hold_count = 0

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:23] = HOME_QPOS
        self.data.ctrl[:]   = self.ctrl_mid
        self.data.qpos[23:26] = BRICK_START
        self.data.qpos[26:30] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.hold_count = 0
        return self._obs()

    def _obs(self):
        robot_qpos = self.data.qpos[:23].copy()
        brick_pos  = self.data.qpos[23:26].copy()
        palm_pos   = self.data.xpos[self.palm_id].copy()
        return np.concatenate([robot_qpos, brick_pos, TARGET_POS, palm_pos])

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = self.ctrl_mid + action * self.ctrl_half
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        d = np.linalg.norm(self.data.xpos[self.palm_id] - self.data.qpos[23:26])
        if d < SUCCESS_THRESH:
            self.hold_count += 1
        else:
            self.hold_count = 0
        success = self.hold_count >= HOLD_STEPS_REQUIRED
        done    = success or (self.steps >= MAX_EP_STEPS)
        rew     = float(np.exp(-5.0 * d)) + 0.5 * float(d < SUCCESS_THRESH) + 25.0 * float(success) - 0.005
        return self._obs(), rew, done, d

    def palm_pos(self):
        return self.data.xpos[self.palm_id].copy()

    def brick_pos(self):
        return self.data.qpos[23:26].copy()


# Renderer
def setup_camera(renderer, cam_idx=0):
    az, el, dist, lookat = CAMERAS[cam_idx]
    renderer.scene.camera.azimuth   = az
    renderer.scene.camera.elevation = el
    renderer.scene.camera.distance  = dist
    renderer.scene.camera.lookat[:] = lookat


def add_text_overlay(frame, lines, top=True):
    # Using PIL if available, otherwise skip overlay
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except:
            font = ImageFont.load_default()
            font_small = font
        y = 20 if top else frame.shape[0] - 30 * len(lines) - 10
        for i, (text, big) in enumerate(lines):
            draw.text((18, y + i*30), text, font=(font if big else font_small),
                      fill=(255, 255, 255),
                      stroke_width=2, stroke_fill=(0, 0, 0))
        return np.array(img)
    except ImportError:
        return frame  # no PIL, skip overlay


def render_episodes(ckpt_path, out_path, n_episodes=3, width=1280, height=720, fps=30, std=0.05):
    print(f"Loading checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        params = pickle.load(f)

    print("Building model...")
    mj_model = build_model(ROBOT_XML, ASSETS_DIR)
    env      = RenderEnv(mj_model)

    agent  = LegoAgent(act_dim=mj_model.nu)
    key    = jax.random.PRNGKey(42)
    dummy_params = agent.init(key, jnp.zeros(OBS_DIM), jnp.array(TARGET_POS))
    tgt    = jnp.array(TARGET_POS)

    @jit
    def forward(params, obs, tgt):
        a, v, _, _ = agent.apply(params, obs, tgt)
        return a

    # Set up offscreen renderer
    print(f"Setting up renderer {width}x{height}...")
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    all_frames = []

    for ep in range(n_episodes):
        print(f"  Rolling out episode {ep+1}/{n_episodes}...")
        obs      = env.reset()
        ep_rew   = 0.0
        ep_frames = []
        palm_trail = []   # for trajectory overlay
        success  = False

        for step in range(MAX_EP_STEPS):
            # Render
            renderer.update_scene(env.data, camera=-1)   # use default free cam
            # Manually set camera
            cam = renderer.scene.camera
            az, el, dist, lookat = CAMERAS[SELECTED_CAM]
            # Use mujoco camera manipulation
            frame = renderer.render()

            # Overlay
            d_palm = np.linalg.norm(env.palm_pos() - env.brick_pos())
            hold_pct = min(100, int(env.hold_count / HOLD_STEPS_REQUIRED * 100))
            status = "✓ HOLDING" if env.hold_count > 0 else "REACHING"
            if success:
                status = "✓ SUCCESS"

            lines = [
                (f"Episode {ep+1}/{n_episodes}", True),
                (f"Step {step:3d}  |  Return {ep_rew:6.1f}", False),
                (f"Palm dist: {d_palm:.3f}m  |  {status}  [{hold_pct:3d}%]", False),
            ]
            frame = add_text_overlay(frame, lines)
            ep_frames.append(frame)

            # Step
            a    = np.array(forward(params, jnp.array(obs[:OBS_DIM]), tgt))
            noise = np.random.normal(0, std, a.shape).astype(np.float32)
            a_noisy = np.clip(a + noise, -1.0, 1.0)
            obs, rew, done, d = env.step(a_noisy)
            ep_rew += rew

            if done:
                success = True
                # Freeze last frame for 1s to show success
                success_frame = add_text_overlay(ep_frames[-1], [
                    (f"Episode {ep+1}/{n_episodes}", True),
                    (f"✓ HOLD COMPLETE  |  Return {ep_rew:.1f}", False),
                    (f"Steps: {step+1}  |  Palm dist: {d:.3f}m", False),
                ])
                ep_frames.extend([success_frame] * fps)
                break

        all_frames.extend(ep_frames)
        print(f"    Episode {ep+1}: {len(ep_frames)} frames, return={ep_rew:.1f}, success={success}")

    print(f"Writing {len(all_frames)} frames to {out_path}...")
    try:
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                     quality=8, macro_block_size=1)
        for f in all_frames:
            writer.append_data(f)
        writer.close()
    except Exception as e:
        print(f"  libx264 failed ({e}), trying ffmpeg fallback...")
        imageio.mimsave(out_path, all_frames, fps=fps)

    print(f"\n✅ Video saved: {out_path}  ({len(all_frames)/fps:.1f}s)")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.expanduser(
                        "~/panda_lego/checkpoints/hold_agent_10000.pkl"))
    parser.add_argument("--out",       default="demo.mp4")
    parser.add_argument("--episodes",  type=int,   default=3)
    parser.add_argument("--width",     type=int,   default=1280)
    parser.add_argument("--height",    type=int,   default=720)
    parser.add_argument("--fps",       type=int,   default=30)
    parser.add_argument("--std",       type=float, default=0.05,
                        help="Action noise std (0=deterministic, 0.05=slight noise)")
    parser.add_argument("--cam",       type=int,   default=0,
                        help="Camera index: 0=3/4 view, 1=side, 2=front")
    args = parser.parse_args()

    SELECTED_CAM = args.cam

    render_episodes(
        ckpt_path  = args.checkpoint,
        out_path   = args.out,
        n_episodes = args.episodes,
        width      = args.width,
        height     = args.height,
        fps        = args.fps,
        std        = args.std,
    )

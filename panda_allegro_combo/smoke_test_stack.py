import numpy as np
from stack_env import PandaBrickStackPhase1, mat2quat_wxyz
import mujoco
import mujoco.viewer
import time
import imageio.v2 as imageio

XML_PATH = "/path/to/your/scene.xml"

#export MUJOCO_GL="egl"

import os, json
from pathlib import Path

def to_jsonable(x):
    if x is None:
        return None
    # numpy scalars (includes np.bool_)
    if isinstance(x, (np.generic,)):
        return x.item()
    # python scalars
    if isinstance(x, (bool, int, float, str)):
        return x
    # numpy arrays
    if isinstance(x, np.ndarray):
        return x.tolist()
    # dict
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    # list/tuple
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    # fallback
    return str(x)

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def rotmat_to_quat_wxyz(R):
    # R: 3x3 rotation matrix
    # returns quaternion [w, x, y, z]
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)

def make_q_down(yaw_deg=0.0, palm_normal_is_plus_z=True):
    # want palm normal aligned with world DOWN:
    world_down = np.array([0.0, 0.0, -1.0])

    # Choose which site axis represents palm normal
    # If site +Z is palm normal, align +Z -> world_down
    # If site -Z is palm normal, align -Z -> world_down, i.e. +Z -> -world_down
    z_axis = world_down if palm_normal_is_plus_z else -world_down
    z_axis = normalize(z_axis)

    # Pick a reference x-axis that isn't parallel to z
    x_ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(x_ref, z_axis)) > 0.95:
        x_ref = np.array([0.0, 1.0, 0.0])

    # Make an orthonormal frame: x, y, z
    y_axis = normalize(np.cross(z_axis, x_ref))
    x_axis = normalize(np.cross(y_axis, z_axis))

    R = np.column_stack([x_axis, y_axis, z_axis])  # columns are basis vectors

    # Optional: apply a yaw about z_axis (world-down direction)
    yaw = np.deg2rad(yaw_deg)
    cz, sz = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([[cz, -sz, 0],
                      [sz,  cz, 0],
                      [ 0,   0, 1]], dtype=np.float64)
    R = R @ R_yaw

    return rotmat_to_quat_wxyz(R)

class StepRecorder:
    def __init__(self, model, width=640, height=480, camera_name="global_cam",
                 save_dir="rollouts", every_k=1, record_images=True):
        self.model = model
        self.W = int(width)
        self.H = int(height)
        self.every_k = int(every_k)
        self.record_images = bool(record_images)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in XML")

        self.renderer = mujoco.Renderer(model, height=self.H, width=self.W)

        # buffers
        self._qpos = []
        self._qvel = []
        self._ctrl = []
        self._action = []
        self._reward = []
        self._terminated = []
        self._truncated = []
        self._stage = []

        self._info_f = None
        self._writer = None
        self._step_idx = 0
        self._ep_dir = None
        self._meta = {}

    def start_episode(self, ep_idx: int, meta: dict = None, fps=30, write_video=True):
        # clear buffers
        meta = meta or {}
        self._qpos.clear(); self._qvel.clear(); self._ctrl.clear()
        self._action.clear(); self._reward.clear()
        self._terminated.clear(); self._truncated.clear(); self._stage.clear()
        self._step_idx = 0

        self._ep_dir = self.save_dir / f"ep_{ep_idx:03d}"
        self._ep_dir.mkdir(parents=True, exist_ok=True)

        self._info_f = open(self._ep_dir / "info.jsonl", "w", encoding="utf-8")

        self._meta = dict(meta or {})
        self._meta["episode_index"] = int(ep_idx)
        self._meta["camera_name"] = self._meta.get("camera_name", None)
        self._meta["width"] = self.W
        self._meta["height"] = self.H
        self._meta["every_k"] = self.every_k

        # stream MP4 writer (avoids storing all frames in RAM)
        if write_video and self.record_images:
            if imageio is None:
                print("imageio not installed; skipping video write.")
                self._writer = None
            else:
                mp4_path = self._ep_dir / "frames.mp4"
                self._writer = imageio.get_writer(str(mp4_path), fps=fps, codec="libx264", quality=8)
        else:
            self._writer = None

    def capture(self, data: mujoco.MjData, info: dict, action=None, reward=None,
                terminated=None, truncated=None, stage=None):
        if (self._step_idx % self.every_k) != 0:
            self._step_idx += 1
            return

        # render
        if self.record_images and (self._writer is not None):
            self.renderer.update_scene(data, camera=self.cam_id)
            rgb = self.renderer.render()  # uint8 (H,W,3)
            self._writer.append_data(rgb)

        # state
        self._qpos.append(data.qpos.copy())
        self._qvel.append(data.qvel.copy())
        self._ctrl.append(data.ctrl.copy())

        # step fields
        if action is None:
            action = np.zeros((self.model.nu,), dtype=np.float32)
        self._action.append(np.array(action, dtype=np.float32).copy())

        self._reward.append(float(reward) if reward is not None else 0.0)
        self._terminated.append(bool(terminated) if terminated is not None else False)
        self._truncated.append(bool(truncated) if truncated is not None else False)
        self._stage.append(int(stage) if stage is not None else -1)

        # info jsonl
        safe = to_jsonable(info or {})
        safe["_step"] = int(self._step_idx)
        safe["_stage"] = int(self._stage[-1])
        self._info_f.write(json.dumps(safe) + "\n")

        self._step_idx += 1

    def end_episode(self, final_meta: dict = None):
        if self._info_f is not None:
            self._info_f.close()
            self._info_f = None

        if self._writer is not None:
            self._writer.close()
            self._writer = None

        # save trajectory arrays
        qpos = np.stack(self._qpos, axis=0) if self._qpos else np.zeros((0, self.model.nq))
        qvel = np.stack(self._qvel, axis=0) if self._qvel else np.zeros((0, self.model.nv))
        ctrl = np.stack(self._ctrl, axis=0) if self._ctrl else np.zeros((0, self.model.nu))
        action = np.stack(self._action, axis=0) if self._action else np.zeros((0, self.model.nu))
        reward = np.asarray(self._reward, dtype=np.float32)
        terminated = np.asarray(self._terminated, dtype=np.bool_)
        truncated = np.asarray(self._truncated, dtype=np.bool_)
        stage = np.asarray(self._stage, dtype=np.int32)

        np.savez_compressed(
            self._ep_dir / "traj.npz",
            qpos=qpos, qvel=qvel, ctrl=ctrl,
            action=action, reward=reward,
            terminated=terminated, truncated=truncated,
            stage=stage
        )

        # write meta JSON
        meta = dict(self._meta)
        meta["num_steps_recorded"] = int(len(reward))
        if final_meta:
            meta.update(to_jsonable(final_meta))
        with open(self._ep_dir / "episode_meta.json", "w", encoding="utf-8") as f:
            json.dump(to_jsonable(meta), f, indent=2)

    def close(self):
        # Close writer/file handles if open
        try:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
        except Exception:
            pass
        try:
            if self._info_f is not None:
                self._info_f.close()
                self._info_f = None
        except Exception:
            pass
        # Close renderer GL context
        try:
            if self.renderer is not None:
                self.renderer.close()
        except Exception:
            pass

def main():
    env = PandaBrickStackPhase1(
        xml_path=XML_PATH,
        frame_skip=10,
        xy_thresh=0.03,
        success_hold=20,
        render_mode=None,
    )

    palm_site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "palm_site")
    if palm_site_id < 0:
        print("Warning: palm_site not found in model XML. Using grasp_site_id instead.")
        palm_site_id = env.grasp_site_id

    global_cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "global_cam")

    if global_cam_id < 0:
        raise ValueError("global_cam camera not found in the model")
    
    rec = StepRecorder(env.model, width=640, height=480, camera_name="global_cam",
                       save_dir="smoke_test_rollouts", every_k=1, record_images=True)
    
    n_episodes = 200
    max_steps = 1000
    COLLECT_FIXED_HORIZON = True

    print("Action space:", env.action_space, "shape:", env.action_space.shape)
    print("Obs space:", env.observation_space, "shape:", env.observation_space.shape)

    view_episodes = {0, n_episodes - 1}   # first and last
    viewer = None

    DEBUG_FORCE_SUCCESS = False

    try:
        for ep in range(n_episodes):
            stage = 0  # 0=above b2, 1=descend b2, 2=close+lift, 3=carry above b1, 4=hover above b1, 5=descend place, 6=release+lift

            obs, info = env.reset(seed=ep)
            mujoco.mj_forward(env.model, env.data)

            q_down = make_q_down(yaw_deg=0.0, palm_normal_is_plus_z=True)

            R0 = env.data.site_xmat[env.palm_site_id].reshape(3, 3).copy()
            q_down = mat2quat_wxyz(R0)

            assert obs.shape == env.observation_space.shape

            p1 = env.data.xpos[env.brick1_body].copy()
            p2 = env.data.xpos[env.brick2_body].copy()
            print("brick1 xyz:", p1, "brick2 xyz:", p2)

            # Safety / table parameters
            table_z = 0.0  # ground plane is at z=0; bricks sit on it
            min_clearance = 0.01
            min_site_z = table_z + env.half_h + min_clearance   # don't let grasp site target go below this

            # open viewer only for first & last episode
            if ep in view_episodes:
                viewer = mujoco.viewer.launch_passive(env.model, env.data)

                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = global_cam_id

                time.sleep(0.1)
            
                if DEBUG_FORCE_SUCCESS:
                    env.debug_force_success()  # for debugging only

                #for _ in range(60):
                #    viewer.sync()
                #    time.sleep(0.02)

                #viewer.close()
                #viewer = None
            else:
                viewer = None

            env._closed_once = False  # reset latch for smoke test
            last_info = {}

            meta = {
                "seed": int(ep),
                "xml_path": XML_PATH,
                "frame_skip": int(env.frame_skip),
                "xy_thresh": float(env.xy_thresh),
                "z_thresh": float(env.z_thresh),
                "success_hold": int(env.success_hold),
                "max_steps": int(max_steps),
                "camera_name": "global_cam",
            }
            rec.start_episode(ep, meta=meta, fps=30, write_video=True)

            stage_t0 = 0
            p2_grasp_ref = None

            for t in range(max_steps):
                # read state
                p1 = env.data.xpos[env.brick1_body].copy()
                p2 = env.data.xpos[env.brick2_body].copy()
                ph = env.data.site_xpos[env.palm_site_id].copy()

                xy_err_to_b2 = float(np.linalg.norm(p2[:2] - ph[:2]))
                dist_to_b2   = float(np.linalg.norm(p2 - ph))

                # heights (tune these)
                z_above = 0.10
                z_grasp = -0.01
                z_lift  = 0.12
                z_place = float(p1[2] + env.full_h + 0.002)

                table_z = 0.0
                min_site_z = table_z + 0.08   # tune (0.08–0.12 is a good start)
                stage_t0 = 0

                if t == 0:
                    stage = 0
                    stage_t0 = 0

                if 'prev_stage' not in locals():
                    prev_stage = stage
                if stage != prev_stage:
                    stage_t0 = t
                    prev_stage = stage

                # FSM for stacking
                if stage == 0:
                    # move above brick2 (XY align at safe height)
                    p_target = np.array([p2[0], p2[1], p2[2] + z_above], dtype=np.float64)
                    grip = -1.0
                    if xy_err_to_b2 < 0.08:
                        stage = 1

                elif stage == 1:
                    # descend to grasp height
                    p_target = np.array([p2[0], p2[1], p2[2] + z_grasp], dtype=np.float64)
                    grip = -1.0

                    xy_err_to_b2 = float(np.linalg.norm(p2[:2] - ph[:2]))
                    z_err = abs(float(ph[2]) - float(p_target[2]))

                    if (xy_err_to_b2 < 0.02) and (z_err < 0.02):
                        stage = 2

                    if (t - stage_t0) > 200:
                        print("Taking too long to reach brick2; moving on.")
                        stage = 2

                elif stage == 2:
                    # close while HOLDING at grasp height (do NOT lift yet)
                    p_target = np.array([p2[0], p2[1], p2[2] + z_grasp], dtype=np.float64)
                    
                    xy_err = float(np.linalg.norm(p2[:2] - ph[:2]))
                    z_err = float(abs(ph[2] - p_target[2]))
                    dist = float(np.linalg.norm(p2 - ph))

                    if (xy_err < 0.015) and (z_err < 0.010) and (dist < 0.035):
                        grip = +1.0
                    else:
                        grip = -1.0

                    # give it time to close and latch grasp
                    if env.grasped and (p2_grasp_ref is None):
                        stage = 3
                    #elif (t - stage_t0) > 80:   # timeout fallback so you don't stall forever
                    #    print("Failed to grasp brick2 in time; moving on.")
                    #    stage = 0

                elif stage == 3:
                    # lift straight up once grasped (or after timeout)
                    p_target = np.array([ph[0], ph[1], p2[2] + z_lift], dtype=np.float64)
                    grip = +1.0
                    if ph[2] > p2[2] + 0.08:
                        stage = 4
               
                    #if (not env.grasped) and (t - stage_t0) > 10:
                    #    print("Lost grasp on brick2 during lift; retrying.")
                    #    stage = 0

                elif stage == 4:
                    # carry above brick1
                    p_target = np.array([p1[0], p1[1], p1[2] + z_above], dtype=np.float64)
                    grip = +1.0
                    if float(np.linalg.norm(p1[:2] - ph[:2])) < 0.02:
                        stage = 5

                elif stage == 5:
                    # hover above brick1 briefly
                    p_target = np.array([p1[0], p1[1], p1[2] + z_above], dtype=np.float64)
                    grip = +1.0
                    if (t - stage_t0) > 20:
                        stage = 6

                elif stage == 6:
                    # descend to place
                    p_target = np.array([p1[0], p1[1], z_place], dtype=np.float64)
                    grip = +1.0
                    if abs(float(ph[2]) - z_place) < 0.01:
                        stage = 7

                else:  # stage == 7
                    # release then lift away
                    p_target = np.array([p1[0], p1[1], p1[2] + z_above], dtype=np.float64)
                    grip = -1.0

                # hard clamp target z so IK never drives into table/ground
                table_z = 0.0
                p_target[2] = max(p_target[2], 0.015)

                # IK + action
                q_down = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)  # example: 180deg about x

                dq = env._ik_dq_to_site_pos(env.palm_site_id, p_target, q_target_wxyz=None, kp_pos=8.0, kp_ori=1.5, damping=2e-2, max_dq=0.06)

                action = np.concatenate([dq, [grip]]).astype(np.float32)

                obs, reward, terminated, truncated, info = env.step(action)
                rec.capture(data=env.data, action=action, reward=reward, terminated=terminated, truncated=truncated, info=info, stage=stage)
                last_info = info

                # update the window (only if viewer is open)
                if viewer is not None:
                    viewer.sync()

                if t == 0:
                    print("info keys:", sorted(info.keys()))

                if (t % 20) == 0:
                    p2 = env.data.xpos[env.brick2_body].copy()
                    ph = env.data.site_xpos[env.palm_site_id].copy()
                    dist = float(np.linalg.norm(p2 - ph))
                    print(f"t={t:03d} grip ={action[7]:+.2f} grasped={env.grasped} dist={dist:.3f}")
                    print(f"t={t:03d} stage={stage} grip={action[7]:+.2f} grasped={env.grasped} dist={dist:.3f}")

                if (t % 20) == 0 and env.grasped:
                    print("stage", stage, "ph.z", ph[2], "p2_live.z", p2[2], "p2_ref.z", None if p2_grasp_ref is None else p2_grasp_ref[2])

                if (not COLLECT_FIXED_HORIZON) and (terminated or truncated):
                    break

            final_meta = {
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "final_info": last_info,
                "final_stage": int(stage),
            }
            rec.end_episode(final_meta=final_meta)
        
            print(
                f"ep {ep:03d} | ran {t+1:03d} steps | "
                f"xy_err={float(last_info.get('xy_err', np.nan)):.4f} | "
                f"z_gap={float(last_info.get('z_gap', np.nan)):.4f} | "
                f"success={bool(last_info.get('success', False))}"
            )

            print(f"... xy_err={float(last_info.get('xy_err', np.nan)):.4f} "
                f"z_gap={float(last_info.get('z_gap', np.nan)):.4f} ...")

            xy_err_to_b2 = float(np.linalg.norm(p2[:2] - ph[:2]))
            print("xy_err_to_b2:", xy_err_to_b2, "ph:", ph, "p2:", p2)

            # Keep last frame open briefly before closing
            if viewer is not None:
                for _ in range(60):
                    viewer.sync()
                    time.sleep(0.02)
                viewer.close()
                viewer = None

    finally:
        try:
            rec.close()
        except Exception:
            pass

    print("\nSmoke test complete.")

if __name__ == "__main__":
    main()

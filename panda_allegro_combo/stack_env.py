import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

def quat_from_yaw(yaw):
    # wxyz quaternion for rotation about z
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)

def quat_conj(q):
    # q = [w,x,y,z]
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def quat_rotate(q, v):
    # rotate vector v by quaternion q (wxyz)
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]

def quat_err_axis_angle(q_target_wxyz, q_current_wxyz):
    q_target = np.asarray(q_target_wxyz, dtype=np.float64)
    q_cur    = np.asarray(q_current_wxyz, dtype=np.float64)
    q_target = q_target / (np.linalg.norm(q_target) + 1e-12)
    q_cur    = q_cur    / (np.linalg.norm(q_cur) + 1e-12)

    q_err = quat_mul(q_target, quat_conj(q_cur))
    q_err = q_err / (np.linalg.norm(q_err) + 1e-12)

    # Ensure shortest path
    if q_err[0] < 0:
        q_err = -q_err

    w, x, y, z = q_err
    v = np.array([x, y, z], dtype=np.float64)
    v_norm = np.linalg.norm(v)

    # If very small angle, linearize
    if v_norm < 1e-9:
        return 2.0 * v

    angle = 2.0 * np.arctan2(v_norm, w)  # in [0, pi]
    axis  = v / v_norm
    return axis * angle

def mat2quat_wxyz(R):
    # Convert 3x3 rotation matrix to quaternion (w,x,y,z)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.trace(R)
    if t > 0.0:
        S = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)

def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)

def quat_err_axis_angle(q_target_wxyz, q_cur_wxyz):
    q_t = quat_normalize(q_target_wxyz)
    q_c = quat_normalize(q_cur_wxyz)

    # q_err = q_t * conj(q_c)
    q_err = quat_mul(q_t, quat_conj(q_c))
    q_err = quat_normalize(q_err)

    # Ensure shortest rotation (avoid discontinuity)
    if q_err[0] < 0.0:
        q_err = -q_err

    # axis-angle from quaternion
    w = np.clip(q_err[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)

    s = np.sqrt(max(1e-12, 1.0 - w*w))  # sin(angle/2)
    axis = q_err[1:] / s                # (x,y,z)

    return axis * angle                 # 3-vector

class PandaBrickStackPhase1(gym.Env):
    def __init__(self, xml_path, frame_skip=10, ctrl_dt=0.02,
                 xy_thresh=0.03, success_hold=20, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.ctrl_dt = ctrl_dt
        self.render_mode = render_mode

        self._cache_ctrl_indices()

        self.palm_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "palm_site")

        if self.palm_site_id < 0:
            raise ValueError("grasp_site not found in model XML.")

        # IDs
        self.brick1_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "brick1")
        self.brick2_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "brick2")

        self.brick1_jnt  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "brick1_free")
        self.brick2_jnt  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "brick2_free")

        self.brick1_adr  = self.model.jnt_qposadr[self.brick1_jnt]
        self.brick2_adr  = self.model.jnt_qposadr[self.brick2_jnt]

        self.brick1_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "brick1_geom")
        self.brick2_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "brick2_geom")

        # Hand synergy postures (order matches self.hand_act_names)
        self.hand_open_q = np.array([
            0.0, 0.3, 0.7, 0.0,   # ff
            0.0, 0.3, 0.7, 0.0,   # mf
            0.0, 0.3, 0.7, 0.0,   # rf
            0.2, 0.4, 0.2, 0.0    # thumb
        ], dtype=np.float32)

        self.hand_close_q = np.array([
            0.2, 1.0, 1.2, 0.7,   # ff
            0.2, 1.0, 1.2, 0.7,   # mf
            0.2, 1.0, 1.2, 0.7,   # rf
            0.6, 1.0, 0.8, 0.6    # thumb
        ], dtype=np.float32)

        # keep a filtered "grip state" so it closes smoothly over multiple steps (optional, but looks nicer than teleporting fingers)
        self._grip_alpha = 0.0

        #step count
        self._step_count = 0

        # Brick half-height
        self.half_h = 0.0115
        self.full_h = 2.0 * self.half_h

        self.xy_thresh = float(xy_thresh)
        self.z_thresh  = float(0.9 * self.full_h)  # generous
        self.success_hold = int(success_hold)
        self._succ_count = 0

        self.sid_ff = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ff_tip_contact")
        self.sid_mf = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "mf_tip_contact")
        self.sid_rf = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "rf_tip_contact")
        self.sid_th = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "th_tip_contact")

        self._has_touch = all(sid >= 0 for sid in [self.sid_ff, self.sid_mf, self.sid_rf, self.sid_th])

        self._grip_alpha = 0.0

        # Phase-1 action space:
        # Delta-q for 7 Panda joints + grip open/close
        self.action_space = gym.spaces.Box(low=-0.02, high=0.02, shape=(8,), dtype=np.float32)

        # Observation: brick poses only for Phase 1 debugging
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        self.brick2_grasp_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "brick2_grasp")

        if self.brick2_grasp_sid < 0:
            raise ValueError("brick2_grasp site not found in model XML.")

        print("xy_thresh =", self.xy_thresh)
        print("z_thresh  =", self.z_thresh, " (full brick height =", self.full_h, ")")

        jid6 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint6")
        print("joint6 range:", self.model.jnt_range[jid6])

    def _cache_ctrl_indices(self):
        # Panda actuators
        self.panda_act_names = [f"actuator{i}" for i in range(1, 8)]
        self.panda_aid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.panda_act_names]

        # Allegro actuators
        self.hand_act_names = ["ffa0","ffa1","ffa2","ffa3", "mfa0","mfa1","mfa2","mfa3", "rfa0","rfa1","rfa2","rfa3", "tha0","tha1","tha2","tha3",]
        self.hand_aid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.hand_act_names]

        # Ctrl indices
        self.panda_ctrl = np.array(self.panda_aid, dtype=int)  # length 7
        self.hand_ctrl  = np.array(self.hand_aid, dtype=int)   # length 16

    def hand_open(self):
        # Open posture (radians)
        q_open = np.array([
            0.0, 0.3, 0.7, 0.0,   # ff
            0.0, 0.3, 0.7, 0.0,   # mf
            0.0, 0.3, 0.7, 0.0,   # rf
            0.2, 0.4, 0.2, 0.0    # thumb
        ], dtype=np.float32)
        self.data.ctrl[self.hand_ctrl] = q_open

    def hand_close(self):   
        # Grasped posture
        q_close = np.array([
            0.2, 1.0, 1.2, 0.7,   # ff
            0.2, 1.0, 1.2, 0.7,   # mf
            0.2, 1.0, 1.2, 0.7,   # rf
            0.6, 1.0, 0.8, 0.6    # thumb
        ], dtype=np.float32)
        self.data.ctrl[self.hand_ctrl] = q_close

    def _touch_flags(self):
        # Returns bools: finger contact flags (ff, mf, rf, th)
        if not getattr(self, "_has_touch", False):
            return False, False, False, False

        def read(sid):
            adr = self.model.sensor_adr[sid]
            # touch sensor outputs a scalar; >0 means contact
            return float(self.data.sensordata[adr]) > 0.0

        return read(self.sid_ff), read(self.sid_mf), read(self.sid_rf), read(self.sid_th)

    def step_sim(self, n=50):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def _try_grasp(self, dist_thresh=0.03):
        p_h = self.data.site_xpos[self.palm_site_id].copy()
        R_h = self.data.site_xmat[self.palm_site_id].reshape(3,3).copy()
        q_h = mat2quat_wxyz(R_h)  # wxyz

        p_b, q_b = self._brick_pose(self.brick2_body)  # body pose (center)

        d = np.linalg.norm(p_b - p_h)
        if d > dist_thresh:
            return False, None

        ff, mf, rf, th = self._touch_flags()
        touched = ff or mf or rf or th
        if not touched:
            return False, {"touch_ff": ff, "touch_mf": mf, "touch_rf": rf, "touch_th": th}

        # latch grasp offsets (hand frame -> brick)
        q_h_conj = quat_conj(q_h)

        # position offset in HAND frame
        self._grasp_p_off = quat_rotate(q_h_conj, (p_b - p_h))

        # orientation offset: q_b = q_h * q_off  => q_off = conj(q_h) * q_b
        self._grasp_q_off = quat_mul(q_h_conj, q_b)

        self.grasped = True
        return True, {"touch_ff": ff, "touch_mf": mf, "touch_rf": rf, "touch_th": th}

    def _apply_grasp(self):
        if (self._grasp_p_off is None) or (self._grasp_q_off is None):
            return

        p_h = self.data.site_xpos[self.palm_site_id].copy()
        R_h = self.data.site_xmat[self.palm_site_id].reshape(3,3).copy()
        q_h = mat2quat_wxyz(R_h)
        q_h = q_h / (np.linalg.norm(q_h) + 1e-9)

        p_b = p_h + quat_rotate(q_h, self._grasp_p_off)
        q_b = quat_mul(q_h, self._grasp_q_off)
        q_b = q_b / (np.linalg.norm(q_b) + 1e-9)

        self.data.qpos[self.brick2_adr:self.brick2_adr+3] = p_b
        self.data.qpos[self.brick2_adr+3:self.brick2_adr+7] = q_b

        dof_adr = self.model.jnt_dofadr[self.brick2_jnt]
        self.data.qvel[dof_adr:dof_adr+6] = 0.0

    def _set_freejoint(self, jnt_id, qpos_adr, pos, quat_wxyz):
        self.data.qpos[qpos_adr:qpos_adr+3] = pos
        self.data.qpos[qpos_adr+3:qpos_adr+7] = quat_wxyz
        # self.data.qvel[:] = 0.0 zero vel for simplicity
        dof_adr = self.model.jnt_dofadr[jnt_id]
        self.data.qvel[dof_adr:dof_adr+6] = 0.0

    def _brick_pose(self, body_id):
        p = self.data.xpos[body_id].copy()
        q = self.data.xquat[body_id].copy()  # wxyz
        return p, q

    def _get_obs(self):
        p1, q1 = self._brick_pose(self.brick1_body)
        p2, q2 = self._brick_pose(self.brick2_body)
        return np.concatenate([p1, q1, p2, q2]).astype(np.float32)

    def _success_now(self):
        p1, _ = self._brick_pose(self.brick1_body)
        p2, _ = self._brick_pose(self.brick2_body)

        xy_err = np.linalg.norm(p2[:2] - p1[:2])
        z_gap  = (p2[2] - p1[2])

        return (xy_err < self.xy_thresh) and (z_gap > self.z_thresh), float(xy_err), float(z_gap)
    
    def debug_force_success(self):
        p1, q1 = self._brick_pose(self.brick1_body)
        p2 = p1.copy()
        p2[2] = p1[2] + self.full_h
        self._set_freejoint(self.brick2_jnt, self.brick2_adr, p2, q1)
        mujoco.mj_forward(self.model, self.data)
        ok, xy_err, z_gap = self._success_now()
        print("FORCED success:", ok, "xy_err:", xy_err, "z_gap:", z_gap)

    def hold_current_targets(self, steps=20):
        mujoco.mj_forward(self.model, self.data)
        for _ in range(steps):
            # keep Panda where it is (position actuators)
            self.data.ctrl[self.panda_ctrl] = self.data.qpos[:7].astype(np.float32)
            mujoco.mj_step(self.model, self.data)

    def _ik_dq_to_site_pos(
        self,
        site_id: int,
        p_target: np.ndarray,
        q_target_wxyz: np.ndarray = None,
        kp_pos: float = 2.0,
        kp_ori: float = 0.4,      # keep this modest
        damping: float = 1e-3,
        max_dq: float = 0.05,
    ):
        mujoco.mj_forward(self.model, self.data)

        # Current site pos/orientation
        p_cur = self.data.site_xpos[site_id].copy()
        R_cur = self.data.site_xmat[site_id].reshape(3, 3).copy()
        q_cur = mat2quat_wxyz(R_cur)

        p_target = np.asarray(p_target, dtype=np.float64).reshape(3,)
        e_pos = (p_target - p_cur).astype(np.float64)  # (3,)

        # Jacobians wrt all DoFs
        Jp = np.zeros((3, self.model.nv), dtype=np.float64)
        Jr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, Jp, Jr, site_id)

        # Arm joints only
        Jp = Jp[:, :7]   # (3,7)
        Jr = Jr[:, :7]   # (3,7)

        if q_target_wxyz is None:
            # Position-only IK (original behavior)
            A = Jp @ Jp.T + (damping * np.eye(3))
            b = (kp_pos * e_pos)
            x = np.linalg.solve(A, b)
            dq = Jp.T @ x
        else:
            q_target = np.asarray(q_target_wxyz, dtype=np.float64).reshape(4,)
            e_ori = quat_err_axis_angle(q_target, q_cur).astype(np.float64)  # (3,)

            # Weighted 6D task: [pos; ori]
            e6 = np.concatenate([kp_pos * e_pos, kp_ori * e_ori], axis=0)    # (6,)
            J6 = np.vstack([Jp, Jr])                                         # (6,7)

            A = J6 @ J6.T + (damping * np.eye(6))
            x = np.linalg.solve(A, e6)
            dq = J6.T @ x

        # weight joints (smaller on base joints, larger on wrist)
        w = np.array([0.03, 0.08, 0.0001, 1.0, 0.0001, 4.0, 0.00001], dtype=np.float64)
        dq = (w * dq).astype(np.float64)

        # nullspace-like bias toward nominal posture (small)
        q = self.data.qpos[:7].copy().astype(np.float64)
        #dq += 0.01 * (self._q_nom - q)
        #dq += 0.001 * (self._q_nom - q) # tiny bias

        # per-joint speed limits (rad per control step)
        dq_lim = np.array([0.03, 0.04, 0.06, 0.06, 0.08, 0.24, 0.12], dtype=np.float64)
        dq = np.clip(dq, -dq_lim, dq_lim).astype(np.float32)

        return dq

    def domain_randomize(self, rng: np.random.Generator):
        # Randomize light positions and intensities (if lights exist)
        if self.model.nlight > 0:
            for i in range(self.model.nlight):
                # jitter position a bit
                self.model.light_pos[i, 0] += rng.uniform(-0.2, 0.2)
                self.model.light_pos[i, 1] += rng.uniform(-0.2, 0.2)
                self.model.light_pos[i, 2] = max(0.5, self.model.light_pos[i, 2] + rng.uniform(-0.2, 0.2))

                # jitter diffuse intensity
                self.model.light_diffuse[i, :] = np.clip(
                    self.model.light_diffuse[i, :] * rng.uniform(0.6, 1.4),
                    0.05, 1.0
                )
        def random_rgba():
            return np.array([rng.uniform(0.2, 1.0),
                     rng.uniform(0.2, 1.0),
                     rng.uniform(0.2, 1.0),
                     1.0], dtype=np.float32)

        if self.brick1_gid >= 0: self.model.geom_rgba[self.brick1_gid] = random_rgba()
        if self.brick2_gid >= 0: self.model.geom_rgba[self.brick2_gid] = random_rgba()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Table height
        table_z = 0.0

        p1 = np.array([0.8, 0.0, table_z + self.half_h])

        yaw1 = rng.uniform(-np.pi, np.pi)
        q1 = quat_from_yaw(yaw1)

        p2 = np.array([0.8, 0.25, table_z + self.half_h])

        yaw2 = rng.uniform(-np.pi, np.pi)
        q2 = quat_from_yaw(yaw2)

        self._set_freejoint(self.brick1_jnt, self.brick1_adr, p1, q1)
        self._set_freejoint(self.brick2_jnt, self.brick2_adr, p2, q2)

        self.domain_randomize(rng)

        mujoco.mj_forward(self.model, self.data)

        self.hold_current_targets()
        self.hand_open()
        self.step_sim(10)
        self.grasped = False
        self._grasp_p_off = None
        self._grasp_q_off = None

        p1, _ = self._brick_pose(self.brick1_body)
        p2, _ = self._brick_pose(self.brick2_body)
        print("xpos brick1 z:", p1[2], "xpos brick2 z:", p2[2], "z gap:", p2[2]-p1[2])

        ok, xy_err, z_gap = self._success_now()
        print("Reset xy_err", xy_err)

        self._q_nom = self.data.qpos[:7].copy().astype(np.float64)

        self._step_count = 0

        self._succ_count = 0
        obs = self._get_obs()
        info = {"success": False}
        return obs, info
    
    def goto_qpos(self, q_arm_target, steps=150):
        q_arm_target = np.asarray(q_arm_target, dtype=np.float32)
        assert q_arm_target.shape == (7,)
        for _ in range(steps):
            self.data.ctrl[self.panda_ctrl] = q_arm_target
            mujoco.mj_step(self.model, self.data)

    def scripted_pick_place(self):
        # open hand
        self.hand_open()
        self.step_sim(50)

        # go to a known "pregrasp" arm pose
        self.goto_qpos(self.q_pregrasp, steps=200)

        # close hand (grasp)
        self.hand_close()
        self.step_sim(120)

        # lift pose
        self.goto_qpos(self.q_lift, steps=200)

        # move-to-place pose
        self.goto_qpos(self.q_place, steps=250)

        # open hand (release)
        self.hand_open()
        self.step_sim(120)

        # retract
        self.goto_qpos(self.q_retract, steps=200)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        dq = action[:7]

        if not hasattr(self, "_dq_filt"):
            self._dq_filt = np.zeros(7, dtype=np.float32)

        alpha = 1.0 # higher = more responsive, lower = smoother
        self._dq_filt = alpha * dq + (1.0 - alpha) * self._dq_filt
        dq = self._dq_filt

        grip = action[7]
        #dq = np.asarray(action, dtype=np.float32)
        assert dq.shape == (7,)

        self._step_count += 1

        # current arm joint positions
        q = self.data.qpos[:7].copy().astype(np.float32)
        q_target = q + dq

        # clip to joint ranges
        for i, jname in enumerate([f"joint{i+1}" for i in range(7)]):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            lo, hi = self.model.jnt_range[jid]
            q_target[i] = np.clip(q_target[i], lo, hi)

        # command panda actuators (position actuators)
        self.data.ctrl[self.panda_ctrl] = q_target

        if (self._step_count % 20) == 0:
            # ctrl is desired position, actuator_force tells you if you’re hitting limits
            print("ctrl (arm):", self.data.ctrl[self.panda_ctrl])
            print("actuator_force (first 7):", self.data.actuator_force[:7])

        # Map grip [-1,1] -> alpha [0,1]
        alpha_cmd = (float(grip) + 1.0) * 0.5
        alpha_cmd = np.clip(alpha_cmd, 0.0, 1.0)

        # Smooth it a bit so fingers don't teleport
        # (tune 0.2–0.5; higher = faster)
        self._grip_alpha = 0.3 * alpha_cmd + 0.7 * getattr(self, "_grip_alpha", 0.0)

        q_hand_target = (1.0 - self._grip_alpha) * self.hand_open_q + self._grip_alpha * self.hand_close_q
        self.data.ctrl[self.hand_ctrl] = q_hand_target

        # Decide attempt grasp and release based on alpha thresholds
        closing = self._grip_alpha > 0.7
        opening = self._grip_alpha < 0.3

        if closing and (not self.grasped):
            self._try_grasp(dist_thresh=0.04)

        if opening and self.grasped:
            # release latch
            self.grasped = False
            self._grasp_p_off = None
            self._grasp_q_off = None
            mujoco.mj_forward(self.model, self.data)

        # step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self.grasped:
                self._apply_grasp()
                mujoco.mj_forward(self.model, self.data)

        # evaluate task success same as before
        ok, xy_err, z_gap = self._success_now()
        self._succ_count = self._succ_count + 1 if ok else 0
        success = self._succ_count >= self.success_hold

        reward = 0.0
        reward += 1.0 if ok else 0.0
        reward -= 2.0 * xy_err
        reward -= 1.0 * max(0.0, self.z_thresh - z_gap)
        reward -= 0.01

        terminated = bool(success)
        truncated = False
        obs = self._get_obs()

        p_h = self.data.site_xpos[self.palm_site_id].copy()
        p_b, _ = self._brick_pose(self.brick2_body)
        dist_to_brick = float(np.linalg.norm(p_b - p_h))

        p2 = self.data.xpos[self.brick2_body].copy()
        ph = self.data.site_xpos[self.palm_site_id].copy()
        dist = float(np.linalg.norm(p2 - ph))

        info = {"success": success, "success_now": ok, "xy_err": xy_err, "z_gap": z_gap, "grasped": bool(self.grasped), "dist_to_brick": dist, "grip": float(grip)}

        if (self._step_count % 20) == 0:
            ff, mf, rf, th = self._touch_flags()
            print("touch:", ff, mf, rf, th, "grip_alpha:", getattr(self, "_grip_alpha", None), "grasped:", self.grasped)

        if (self._step_count % 50) == 0:
            print("arm qpos:", self.data.qpos[:7])

        if (self._step_count % 20) == 0:
            print("q6:", float(self.data.qpos[5]),
            "ctrl6:", float(self.data.ctrl[self.panda_ctrl][5]),
            "tau6:", float(self.data.actuator_force[5]))

        if (self._step_count % 20) == 0:
            pos_err = self.data.ctrl[self.panda_ctrl] - self.data.qpos[:7]
            print("pos_err:", np.round(pos_err, 3))
            print("tau:", np.round(self.data.actuator_force[:7], 1))

        return obs, reward, terminated, truncated, info

class StackEnv(gym.Env):
    def __init__(self, xml_path, frame_skip=10, ctrl_dt=0.02, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.ctrl_dt = ctrl_dt
        self.render_mode = render_mode

        #self._cache_ctrl_indices()

        # Optional: safe home pose for robot arm (if present)
        if hasattr(self, "panda_ctrl"):
            self.q_home = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.5, 0.5], dtype=np.float32)
            self.arm_qpos_adr = 0  # assuming arm joints start at qpos[0]

        # Observation: brick poses only for debugging
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        print("StackEnv initialized.")

    def _get_body_pose(self, body_name: str):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        pos = self.data.xpos[bid].copy()   # (3,)
        quat = self.data.xquat[bid].copy() # (4,) wxyz
        return pos, quat

    def _get_obs(self):
        p1, q1 = self._get_body_pose("brick1")
        p2, q2 = self._get_body_pose("brick2")
        obs = np.concatenate([p1, q1, p2, q2]).astype(np.float32)
        return obs
    
    def is_stacked(self, xy_tol=0.01, z_tol=0.004):
        b1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "brick1")
        b2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "brick2")

        p1 = self.data.xpos[b1]
        p2 = self.data.xpos[b2]

        xy_ok = np.linalg.norm(p2[:2] - p1[:2]) < xy_tol
        z_ok  = abs((p2[2] - p1[2]) - 0.023) < z_tol

        return bool(xy_ok and z_ok)

    def step(self, action):
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 1) Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        BRICK_SZ = 0.0115               # half-height from geom size
        BRICK_H  = 2.0 * BRICK_SZ       # 0.023
        EPS      = 0.001                # 1 mm gap
        SETTLE_STEPS = 200

        def set_free_body_pose(model, data, body_name, pos, quat_wxyz=(1,0,0,0)):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

            jadr = model.body_jntadr[bid]
            jnum = model.body_jntnum[bid]
            if jnum < 1:
                raise ValueError(f"Body '{body_name}' has no joints. Did you forget to add <freejoint/>?")

            # First joint on this body (for free bodies, this is the freejoint)
            jid = jadr

            qadr = model.jnt_qposadr[jid]
            vadr = model.jnt_dofadr[jid]

            # free joint qpos: [x y z qw qx qy qz]
            data.qpos[qadr:qadr+3] = np.asarray(pos, dtype=float)
            data.qpos[qadr+3:qadr+7] = np.asarray(quat_wxyz, dtype=float)

            # free joint qvel: 6 dof
            data.qvel[vadr:vadr+6] = 0.0

        def teleport_stacked_on_ground(model, data, x=0.5, y=0.0, yaw=0.0):
            # yaw about z
            qw = np.cos(yaw/2.0)
            qz = np.sin(yaw/2.0)
            quat = (qw, 0.0, 0.0, qz)

            z1 = BRICK_SZ
            z2 = BRICK_SZ + BRICK_H + EPS

            set_free_body_pose(model, data, "brick1", (x, y, z1), quat)
            set_free_body_pose(model, data, "brick2", (x, y, z2), quat)

            mujoco.mj_forward(model, data)

        def teleport_side_by_side_on_ground(model, data, x=0.5, y=0.0, yaw=0.0,
                                    sep=0.06):
            # sep is center-to-center separation in meters (choose > brick width)
            qw = np.cos(yaw/2.0)
            qz = np.sin(yaw/2.0)
            quat = (qw, 0.0, 0.0, qz)

            z = BRICK_SZ  # both on ground

            # place brick1 and brick2 separated in x (or y)
            set_free_body_pose(model, data, "brick1", (x - sep/2, y, z), quat)
            set_free_body_pose(model, data, "brick2", (x + sep/2, y, z), quat)

            mujoco.mj_forward(model, data)

        def teleport_near_miss_on_ground(model, data, x=0.5, y=0.0, yaw=0.0,
                                xy_miss=0.012):
            # "almost stacked" but slightly off in XY so it should fall off during settle
            qw = np.cos(yaw/2.0)
            qz = np.sin(yaw/2.0)
            quat = (qw, 0.0, 0.0, qz)

            z1 = BRICK_SZ
            z2 = BRICK_SZ + BRICK_H + EPS

            # offset brick2 slightly
            dx = np.random.uniform(-xy_miss, xy_miss)
            dy = np.random.uniform(-xy_miss, xy_miss)

            set_free_body_pose(model, data, "brick1", (x, y, z1), quat)
            set_free_body_pose(model, data, "brick2", (x + dx, y + dy, z2), quat)

            mujoco.mj_forward(model, data)

        def settle(model, data, steps=SETTLE_STEPS):
            for _ in range(steps):
                mujoco.mj_step(model, data)

        def hardcoded_scene_reset(model, data):
            # Randomize base pose
            x = 0.5 + np.random.uniform(-0.03, 0.03)
            y = 0.0 + np.random.uniform(-0.03, 0.03)
            yaw = np.random.uniform(-np.deg2rad(10), np.deg2rad(10))

            # Pick a mode (roughly equal)
            mode = np.random.choice(["stacked", "side_by_side", "near_miss"])

            if mode == "stacked":
                teleport_stacked_on_ground(model, data, x=x, y=y, yaw=yaw)
                settle(model, data, steps=200)

            elif mode == "side_by_side":
                teleport_side_by_side_on_ground(model, data, x=x, y=y, yaw=yaw, sep=0.06)
                settle(model, data, steps=200)

            elif mode == "near_miss":
                teleport_near_miss_on_ground(model, data, x=x, y=y, yaw=yaw, xy_miss=0.012)
                settle(model, data, steps=200)

            return mode

        # 2) Put robot in safe home pose (if present)
        if hasattr(self, "q_home"):
            self.data.qpos[self.arm_qpos_adr:self.arm_qpos_adr+len(self.q_home)] = self.q_home
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

        # 3) HARD-CODED SUCCESS STATE 
        mode = hardcoded_scene_reset(self.model, self.data)
        self.last_reset_mode = mode

        # 4) Final forward pass
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs()

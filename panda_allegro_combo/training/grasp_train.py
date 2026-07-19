import os, sys, time, pickle, imageio
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
import optax
from flax.training.train_state import TrainState
import mujoco
from math import exp
from scipy.spatial import ConvexHull
from itertools import product

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

INDEX_QPOS_IDXS  = list(range(7, 11))    # ffj0..3
THUMB_QPOS_IDXS  = list(range(19, 23))   # thj0..3

INDEX_HOME, INDEX_CLOSED = 0.0, 0.5
THUMB_HOME = 0.0
THUMB_HOME_VEC = np.array([1.2, 0.0, 0.728, 0.779])
THUMB_CLOSED_VEC = np.array([1.9, 0.1, 1.644, 1.719])
THUMB_CLOSED = 0.5
PINCH_CLOSURE_THRESH     = 0.5  # tune against rollout data once training starts

# Contact / lift thresholds
PALM_THRESH = 0.06   
LIFT_THRESH = 0.03

# Domain randomization
BRICK_BASE_POS = np.array([0.6, 0.0, 0.42 + 0.0096])
DR_WARMUP      = 1000
DR_XY_MAX      = 0.06         
DR_Z_MAX       = 0.01         
DR_YAW_MAX     = np.deg2rad(20) 

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
VIDEO_DIR       = os.path.expanduser("~/panda_allegro_combo/videos")

# Paths
ROBOT_XML  = os.path.expanduser("~/panda_allegro_combo/models/mjxpandamerged.xml")
ASSETS_DIR = os.path.expanduser("~/panda_allegro_combo/models/assets")
CKPT_DIR   = os.path.expanduser("~/panda_allegro_combo/checkpoints")

# Resume from the best checkpoint
RESUME_PATH = os.path.expanduser("~/panda_allegro_combo/checkpoints/hold_dr_agent_1000.pkl")

HOME_QPOS = np.array([
    -0.0413, -0.6000,  0.1060, -1.8000, -0.2379,  2.0784,  0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.5, 0.0, 0.5, 0.3,
])

VF_CLIP_RANGE = 5.0 

MIDDLE_QPOS_IDXS = list(range(11,15))
RING_QPOS_IDXS = list(range(15,19))


# DR
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
        self.obs_dim = 38       #was 32
        self.act_dim = self.nu
        self.steps         = 0
        self.contact_steps = 0

        # Per-episode brick spawn, updated on reset() so all reward calculations (drift, lift) reference the correct starting position
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

        self.index_body_ids = self._geoms_for_body_name_prefix("ff_")
        self.middle_body_ids = self._geoms_for_body_name_prefix("mf_")
        self.ring_body_ids = self._geoms_for_body_name_prefix("rf_")
        self.thumb_body_ids = self._geoms_for_body_name_prefix("th_")

        self.index_tip_site_id = model.site("ff_tip_site").id
        self.thumb_tip_site_id = model.site("th_tip_site").id

    def _geoms_for_body(self, body_id):
        geom_ids = set()
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == body_id:
                geom_ids.add(g)
        return geom_ids
    
    def _geoms_for_body_name_prefix(self, prefix):
        body_ids = set()
        for b in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            if name and name.startswith(prefix):
                body_ids.add(b)
        return body_ids
    
    def _pinch_closure(self):
        index_qpos = self.data.qpos[INDEX_QPOS_IDXS]   # ffj0..3, 4 joints
        thumb_qpos = self.data.qpos[THUMB_QPOS_IDXS]    # thj0..3, 4 joints

        index_closure = np.mean(np.abs(index_qpos - INDEX_HOME)) / max(abs(INDEX_CLOSED - INDEX_HOME), 1e-6)
        thumb_range = np.abs(THUMB_CLOSED_VEC - THUMB_HOME_VEC)
        thumb_closure = np.mean(np.abs(thumb_qpos - THUMB_HOME_VEC) / np.maximum(thumb_range, 1e-6))
        return float(np.clip(index_closure, 0, 1)), float(np.clip(thumb_closure, 0, 1))

    def _finger_label(self, body_id):
        if body_id in self.index_body_ids:
            return "index"
        if body_id in self.middle_body_ids:
            return "middle"
        if body_id in self.ring_body_ids:
            return "ring"
        if body_id in self.thumb_body_ids:
            return "thumb"
        return None

    def _has_brick_contact(self):
        contacts = []
        brick_pos = self.data.qpos[23:26].copy()
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g1 = con.geom1
            g2 = con.geom2
            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]
            if b1 == 0 or b2 == 0:
                continue

            finger_body = None
            if b1 == self.brick_id:
                finger_body = b2
            elif b2 == self.brick_id:
                finger_body = b1
            if finger_body is None:
                continue

            label = self._finger_label(finger_body)
            if label is None or label in ("middle", "ring"):
                continue

            pos    = con.pos.copy()
            normal = con.frame[:3].copy()  # contact normal, world frame

            if b1 == self.brick_id:
                normal = -normal

            mu1 = self.model.geom_friction[g1, 0]
            mu2 = self.model.geom_friction[g2, 0]
            mu  = float(np.sqrt(mu1 * mu2))

            contacts.append((pos, normal, mu, label))
        return contacts

    def _mean_finger_closure(self):
        finger_qpos = self.data.qpos[FINGER_QPOS_IDXS]
        closure = np.mean(np.abs(finger_qpos - FINGER_HOME)) / max(abs(FINGER_CLOSED - FINGER_HOME), 1e-6)
        return float(np.clip(closure, 0.0, 1.0))

    def _epsilon_quality(self, contacts, brick_centroid, n_friction_edges=6):
        n = len(contacts)
        if n < 2:
            return 0.0

        if n == 2:
            (p1, n1, mu1, _), (p2, n2, mu2, _) = contacts
            n1 = n1 / (np.linalg.norm(n1) + 1e-8)
            n2 = n2 / (np.linalg.norm(n2) + 1e-8)

            #Opposition: normals should point toward each other
            opposition = float(np.clip(np.dot(n1, -n2), 0.0, 1.0))

            #Axis alignment: each normal should align with the grasp axis
            grasp_axis = p1 - p2
            axis_len = np.linalg.norm(grasp_axis)
            if axis_len < 1e-6:
                return 0.0
            grasp_axis /= axis_len
            align1 = float(np.clip(np.dot(n1, -grasp_axis), 0.0, 1.0))
            align2 = float(np.clip(np.dot(n2,  grasp_axis), 0.0, 1.0))
            axis_alignment = 0.5 * (align1 + align2)

            #Friction margin: are the normals inside the friction cones
            sin1 = np.linalg.norm(np.cross(n1, -grasp_axis))
            sin2 = np.linalg.norm(np.cross(n2,  grasp_axis))
            margin1 = float(np.clip(1.0 - sin1 / (mu1 + 1e-8), 0.0, 1.0))
            margin2 = float(np.clip(1.0 - sin2 / (mu2 + 1e-8), 0.0, 1.0))
            friction_margin = 0.5 * (margin1 + margin2)

            return opposition * axis_alignment * friction_margin

        # n >= 3: original Ferrari-Canny wrench-space method
        wrenches = []
        for pos, normal, mu, _label in contacts:
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            tmp = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            t1 = np.cross(normal, tmp)
            t1 /= (np.linalg.norm(t1) + 1e-8)
            t2 = np.cross(normal, t1)

            r = pos - brick_centroid  # moment arm

            for k in range(n_friction_edges):
                theta = 2.0 * np.pi * k / n_friction_edges
                force_dir = normal + mu * (np.cos(theta) * t1 + np.sin(theta) * t2)
                force_dir /= (np.linalg.norm(force_dir) + 1e-8)
                torque = np.cross(r, force_dir)
                wrenches.append(np.concatenate([force_dir, torque]))

        wrenches = np.array(wrenches)  # shape (n*n_friction_edges, 6)

        try:
            hull = ConvexHull(wrenches)
        except Exception:
            return 0.0

        offsets = hull.equations[:, -1]
        if np.any(offsets > 0):
            return 0.0

        facet_normals = hull.equations[:, :-1]
        norms = np.linalg.norm(facet_normals, axis=1)
        distances = -offsets / (norms + 1e-8)
        return float(np.min(distances))
    
    def _thumb_opposition_reward(self):
        index_tip = self.data.site_xpos[self.index_tip_site_id]
        thumb_tip = self.data.site_xpos[self.thumb_tip_site_id]
        brick_pos = self.data.qpos[23:26]

        # Vector from brick center to index tip, and to thumb tip.
        # Good opposition: these point in roughly opposite directions.
        v_index = index_tip - brick_pos
        v_thumb = thumb_tip - brick_pos
        n_index = v_index / (np.linalg.norm(v_index) + 1e-8)
        n_thumb = v_thumb / (np.linalg.norm(v_thumb) + 1e-8)

        opposition = -float(np.dot(n_index, n_thumb))  # -1 (same side) to +1 (opposite sides)
        return np.clip(opposition, -1.0, 1.0)

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
        ff_tip_pos = self.data.site_xpos[self.index_tip_site_id].copy()
        th_tip_pos = self.data.site_xpos[self.thumb_tip_site_id].copy()
        return np.concatenate([robot_qpos, brick_pos, TARGET_POS, palm_pos, ff_tip_pos - brick_pos, th_tip_pos - brick_pos])

    def step(self, action, contact_hold_required):
        action = np.clip(action, -1.0, 1.0)
        action[MIDDLE_QPOS_IDXS] = 0.0
        action[RING_QPOS_IDXS] = 0.0
        action[6] = 0.0
        action[20] = 0.0
        self.data.ctrl[:] = self.ctrl_mid + action * self.ctrl_half
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            self.data.qpos[MIDDLE_QPOS_IDXS] = 0.0
            self.data.qpos[RING_QPOS_IDXS] = 0.0
            self.data.qvel[MIDDLE_QPOS_IDXS] = 0.0
            self.data.qvel[RING_QPOS_IDXS] = 0.0
            self.data.qpos[6] = 0.0
            self.data.qvel[6] = 0.0
            self.data.qpos[20] = 0.0
            self.data.qvel[20] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1

        brick_pos = self.data.qpos[23:26].copy()
        palm_pos  = self.data.xpos[self.palm_id].copy()
        d_palm    = np.linalg.norm(palm_pos - brick_pos)

        #lateral approach
        approach_vec = brick_pos - palm_pos
        approach_vec[2] = 0.0

        #horizontal approach
        horiz_dist = np.linalg.norm(approach_vec)
        lateral_approach = 0.3 * float(horiz_dist < 0.20) * float(d_palm < 0.20)

        ff_tip_pos = self.data.site_xpos[self.index_tip_site_id]
        th_tip_pos = self.data.site_xpos[self.thumb_tip_site_id]
        d_index = np.linalg.norm(ff_tip_pos - brick_pos)
        d_thumb = np.linalg.norm(th_tip_pos - brick_pos)

        fingertip_guidance = (exp(-8.0 * d_index) * 1.5 + exp(-8.0 * d_thumb) * 1.5) * float(d_palm < 0.15)

        contacts = self._has_brick_contact()
        has_contact = len(contacts) > 0
        index_closure, thumb_closure = self._pinch_closure()

        # Drift and lift are always relative to THIS episode's brick start
        brick_start_z  = self.episode_brick_start[2]
        brick_xy_drift = np.linalg.norm(brick_pos[:2] - self.episode_brick_start[:2])
        brick_lift     = max(0.0, brick_pos[2] - brick_start_z)

        # Floor penalty
        TABLE_Z = 0.42
        FLOOR_MARGIN = 0.01
        palm_floor_violation = max(0.0, TABLE_Z + FLOOR_MARGIN - palm_pos[2])
        floor_penalty = -20.0 * palm_floor_violation


        near_and_closing = (has_contact and d_palm < PALM_THRESH and
                             index_closure > PINCH_CLOSURE_THRESH and
                             thumb_closure > PINCH_CLOSURE_THRESH)
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
        approach_reward = exp(-4.0 * d_palm) * 0.5 * (1.0 + 2.0 * float(has_contact)) * float(d_palm > 0.08)

        # 2. Contact bonus: gated on near_and_closing
        grasping_contact = float(near_and_closing)
        contact_bonus    = 3.0 * grasping_contact

        # 4. Contact streak reward
        streak_frac           = min(self.contact_steps, contact_hold_required) / contact_hold_required
        contact_streak_reward = 3.0 * (self.contact_steps / contact_hold_required) #1.0 * streak_frac * (1.0 if is_holding else 0.1) # was 2.0 * ...

        # Initialize done
        done = False

        # 5. Shove penalty
        shove_penalty = -1.0 * max(0.0, brick_xy_drift - 0.10)

        # 6. Lift bonus: gated on is_holding
        lift_bonus = 10.0 * (brick_lift ** 0.5) * float(is_holding)
        if is_holding and brick_lift > 0.02:
            lift_bonus += 5.0
        if is_holding and brick_lift > 0.05:
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
        time_penalty = -0.01 - 0.3 * max(0.0, d_palm - 0.05)

        # 11. Force-closure quality
        pinch_contacts  = [c for c in contacts if c[3] in ("index", "thumb")]
        grasp_quality   = self._epsilon_quality(pinch_contacts, brick_pos)
        closure_reward  = 1.5 * grasp_quality * float(d_palm < PALM_THRESH)

        # 12. Thumb opposition shaping
        opposition_reward = 0.5 * self._thumb_opposition_reward()

        #finger closing reward
        finger_closing_reward = 1.5 * (index_closure + thumb_closure) * float(d_palm < 0.20)

        #lift guidance
        if self.contact_steps >= 2:
            lift_guidance = 2.0 * max(0.0, brick_pos[2] - BRICK_BASE_POS[2])
        else:
            lift_guidance = 0.0

        #open penalty
        if d_palm < 0.20:
            open_penalty = -2.0 * (1.0 - index_closure) - 2.0 * (1.0 - thumb_closure)
        else:
            open_penalty = 0.0

        #panda joint limits
        JOINT_LIMITS_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        JOINT_LIMITS_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        arm_qpos = self.data.qpos[:7]
        low_violation = np.maximum(0.0, JOINT_LIMITS_LOW - arm_qpos)
        high_violation = np.maximum(0.0, arm_qpos - JOINT_LIMITS_HIGH)
        joint_limit_penalty = -2.0 * float(np.sum(low_violation + high_violation))
        joint1_penalty = -1.0 * abs(self.data.qpos[0])

        # palm orientation reward — compute palm_down from rotation matrix
        palm_mat  = self.data.xmat[self.palm_id].reshape(3, 3)
        palm_z    = palm_mat[:, 2]
        palm_down = float(np.clip(-palm_z[2], 0.0, 1.0))
        proximity_weight = float(np.clip(1.0 - d_palm / 0.40, 0.0, 1.0))
        palm_orientation_reward = 0.4 * palm_down * proximity_weight

        reward = (approach_reward + contact_bonus + closure_reward + lift_guidance +
                contact_streak_reward + shove_penalty + lift_bonus + fingertip_guidance +
                grasp_bonus + completion_bonus + floor_penalty + finger_closing_reward +
                lift_velocity_reward + time_penalty + opposition_reward + palm_orientation_reward + 
                open_penalty + lateral_approach + joint_limit_penalty + joint1_penalty)

        # Floor penalty, AFTER reward is defined
        TABLE_Z = 0.42  # table surface at height z
        fingertip_ids = [self.model.body(name).id for name in ['ff_tip', 'th_tip']]
        min_fingertip_z = float(min(float(self.data.xpos[fid][2]) for fid in fingertip_ids))
        if min_fingertip_z < TABLE_Z + 0.01:
            reward -= 10.0
            done = True

        done = done or success or (self.steps >= MAX_EP_STEPS)
        finger_closure = 0.5 * (index_closure + thumb_closure)  # kept for logging compatibility
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
        renderer.update_scene(rec_env.data, camera="global_cam")
        frames.append(renderer.render())
    renderer.close()
    if frames:
        video_path = os.path.join(VIDEO_DIR, f"grasp_update_{update_idx:06d}.mp4")
        imageio.mimwrite(video_path, frames, fps=RECORD_FPS, quality=8)
        print(f"Video saved ({len(frames)} frames): {video_path}")


# Train
def train():
    print("Panda-Lego: Grasp (CPU sim + GPU nets)")
    print(f"JAX devices: {jax.devices()}")

    print("\nBuilding MuJoCo model...")
    mj_model = build_model(ROBOT_XML, ASSETS_DIR)
    envs     = [GraspEnv(mj_model) for _ in range(N_ENVS)]
    print(f"  nq={mj_model.nq}, nu={mj_model.nu}")
    print(f"  Brick body ID: {envs[0].brick_id}")
    print(f"  Index body IDs: {envs[0].index_body_ids}")
    print(f"  Middle body IDs: {envs[0].middle_body_ids}")
    print(f"  Ring body IDs: {envs[0].ring_body_ids}")
    print(f"  Thumb body IDs: {envs[0].thumb_body_ids}")

    _diag_env = envs[0]
    _diag_env.reset()
    print(f"  Contact diagnostic: ncon={_diag_env.data.ncon}, "
          f"has_contact={_diag_env._has_brick_contact()}")

    agent  = LegoAgent(act_dim=envs[0].act_dim)
    key    = jax.random.PRNGKey(0)
    params = agent.init(key, jnp.zeros(38), jnp.array(TARGET_POS))
    n_p    = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Agent parameters: {n_p:,}")

    if RESUME_PATH is not None and os.path.exists(RESUME_PATH):
        with open(RESUME_PATH, "rb") as f:
            params = pickle.load(f)
        print(f"  Resumed from: {RESUME_PATH}")
    else:
        print(f"  WARNING: {RESUME_PATH} not found — starting from scratch.")
        print(f"  (Set RESUME_PATH to your best hold-DR checkpoint.)")

    # Separate optimizers for policy and VF heads with independent gradient clipping.
    schedule = optax.linear_schedule(5e-5, 5e-6, TOTAL_UPDATES)

    # Single optimizer with tighter global gradient clip.
    tx = optax.chain(
        optax.clip_by_global_norm(0.3),
        optax.adam(schedule),
    )
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

    # DR freeze gate state
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

        obs_buf  = np.zeros((N_STEPS, N_ENVS, 38),              dtype=np.float32)
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
                    # contact-gated success for DR freeze check
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

        obs_j     = jnp.array(obs_buf.reshape(-1, 38))
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
            # contact-gated success for DR freeze check
            c_suc        = np.mean(all_contact_successes[-50:]) if all_contact_successes else 0.0
            pg, vf       = aux
            ep_len       = ep_lens.mean()
            mean_dp      = np.mean(ep_dpalms)
            mean_contact = float(np.mean(ep_contacts))
            mean_closure = float(np.mean(ep_closures))
            mean_drift   = float(np.mean(ep_drifts)) if ep_drifts else 0.0
            mean_lift    = float(np.mean(ep_lifts))  if ep_lifts  else 0.0
            mean_streak  = float(np.mean(all_hold_streaks[-50:])) if all_hold_streaks else 0.0
            #  noise_frac respects the DR freeze
            if update <= DR_WARMUP:
                noise_frac = 0.0
            else:
                raw_frac = min((update - DR_WARMUP) / (TOTAL_UPDATES - DR_WARMUP), 1.0)
                noise_frac = raw_frac if dr_unfrozen else min(raw_frac, DR_FREEZE_MAX)

            # DR freeze gate logic
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

from curses.ascii import ctrl
import os, sys, time, pickle, imageio
from turtle import forward, update
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

sys.path.insert(0, os.path.expanduser("~/panda_allegro_combo"))
from envs.lego_env import build_model
from training.encoders import LegoAgentHyper, LegoAgent

# Config

N_ENVS        = 16
N_STEPS       = 128
N_EPOCHS      = 4
CLIP_EPS      = 0.05
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
ENT_COEF      = 0.02      # lower entropy for fine-tuning
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
TOTAL_UPDATES = 3000
LOG_INTERVAL  = 10
SAVE_INTERVAL = 100

# LR: start low to fine-tune a pre-trained policy
LR_START = 5e-6
LR_END   = 1e-6

STD_START = 0.10
STD_END   = 0.06
STD_DECAY = 2000

VF_CLIP_RANGE = 10.0

MAX_EP_STEPS = 400

# Freeze encoders for this many updates; only train hypernet + policy trunk
FREEZE_ENCODER_UPDATES = 500

# Grasp curriculum (fixed for hypernet phase; don't complicate with curriculum)
CONTACT_HOLD_REQUIRED = 4

# Contact / lift thresholds (same as grasp phase)
PALM_THRESH = 0.06
LIFT_THRESH = 0.03
FINGER_QPOS_IDXS = list(range(7, 23))
FINGER_HOME      = 0.0
FINGER_CLOSED    = 0.5

TARGET_POS = np.array([0.5, 0.15, 0.42])   # alias

INDEX_QPOS_IDXS  = list(range(7, 11))    # ffj0..3
THUMB_QPOS_IDXS  = list(range(19, 23))   # thj0..3

INDEX_HOME, INDEX_CLOSED = 0.0, 0.5
THUMB_HOME_VEC   = np.array([1.2, 0.0, 0.728, 0.779])
THUMB_CLOSED_VEC = np.array([1.9, 0.1, 1.644, 1.719])
PINCH_CLOSURE_THRESH = 0.5

MIDDLE_QPOS_IDXS = list(range(11,15))
RING_QPOS_IDXS = list(range(15,19))

# Video
RECORD_INTERVAL = 500
RECORD_FPS      = 30
RECORD_WIDTH    = 640
RECORD_HEIGHT   = 480
VIDEO_DIR       = os.path.expanduser("~/panda_lego/videos/hypernet")

# Paths
ROBOT_XML  = os.path.expanduser("~/panda_lego/models/mjxpandamerged.xml")
ASSETS_DIR = os.path.expanduser("~/panda_lego/models/assets")
CKPT_DIR   = os.path.expanduser("~/panda_lego/checkpoints/hypernet")

# Base grasp checkpoint to transfer weights from
GRASP_CKPT = os.path.expanduser("~/panda_lego/checkpoints/grasp_agent_5000.pkl")

# Goal positions for hypernet experiment

GOAL_POSITIONS = np.array([
    [0.6,   0.0,  0.4296],   # 0: center (trained position)
    [0.55,  0.0,  0.4296],   # 1: closer
    [0.65,  0.0,  0.4296],   # 2: further
    [0.6,   0.05, 0.4296],   # 3: left
    [0.6,  -0.05, 0.4296],   # 4: right
], dtype=np.float32)

GOAL_NAMES = ["center", "left", "right", "forward", "backward"]
N_GOALS    = len(GOAL_POSITIONS)

HOME_QPOS = np.array([
    0.0, -0.54, 0.0, -2.37, 0.0, 2.9, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    1.2, 0.0, 0.728, 0.779
])

FIXED_TARGET = np.array([0.5, 0.15, 0.42])  # matches grasp training


# Environment

from scipy.spatial import ConvexHull

class HyperEnv:
    def __init__(self, model):
        self.model   = model
        self.data    = mujoco.MjData(model)
        self.palm_id = model.body("palm").id
        self.nu      = model.nu
        self.obs_dim = 38
        self.act_dim = self.nu
        self.steps         = 0
        self.contact_steps = 0

        # hypernet addition: goal tracking
        self.current_goal_idx = 0
        self.episode_brick_start = GOAL_POSITIONS[0].copy()

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

        self.index_body_ids  = self._geoms_for_body_name_prefix("ff_")
        self.middle_body_ids = self._geoms_for_body_name_prefix("mf_")
        self.ring_body_ids   = self._geoms_for_body_name_prefix("rf_")
        self.thumb_body_ids  = self._geoms_for_body_name_prefix("th_")

        self.index_tip_site_id = model.site("ff_tip_site").id
        self.thumb_tip_site_id = model.site("th_tip_site").id

    def _geoms_for_body_name_prefix(self, prefix):
        body_ids = set()
        for b in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            if name and name.startswith(prefix):
                body_ids.add(b)
        return body_ids

    def _pinch_closure(self):
        index_qpos = self.data.qpos[INDEX_QPOS_IDXS]
        thumb_qpos = self.data.qpos[THUMB_QPOS_IDXS]
        index_closure = np.mean(np.abs(index_qpos - INDEX_HOME)) / max(abs(INDEX_CLOSED - INDEX_HOME), 1e-6)
        thumb_range = np.abs(THUMB_CLOSED_VEC - THUMB_HOME_VEC)
        thumb_closure = np.mean(np.abs(thumb_qpos - THUMB_HOME_VEC) / np.maximum(thumb_range, 1e-6))
        return float(np.clip(index_closure, 0, 1)), float(np.clip(thumb_closure, 0, 1))

    def _finger_label(self, body_id):
        if body_id in self.index_body_ids:  return "index"
        if body_id in self.middle_body_ids: return "middle"
        if body_id in self.ring_body_ids:   return "ring"
        if body_id in self.thumb_body_ids:  return "thumb"
        return None

    def _has_brick_contact(self):
        contacts = []
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g1, g2 = con.geom1, con.geom2
            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]
            if b1 == 0 or b2 == 0:
                continue
            finger_body = b2 if b1 == self.brick_id else (b1 if b2 == self.brick_id else None)
            if finger_body is None:
                continue
            label = self._finger_label(finger_body)
            if label is None or label in ("middle", "ring"):
                continue
            pos    = con.pos.copy()
            normal = con.frame[:3].copy()
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
            opposition = float(np.clip(np.dot(n1, -n2), 0.0, 1.0))
            grasp_axis = p1 - p2
            axis_len = np.linalg.norm(grasp_axis)
            if axis_len < 1e-6:
                return 0.0
            grasp_axis /= axis_len
            align1 = float(np.clip(np.dot(n1, -grasp_axis), 0.0, 1.0))
            align2 = float(np.clip(np.dot(n2,  grasp_axis), 0.0, 1.0))
            axis_alignment = 0.5 * (align1 + align2)
            sin1 = np.linalg.norm(np.cross(n1, -grasp_axis))
            sin2 = np.linalg.norm(np.cross(n2,  grasp_axis))
            margin1 = float(np.clip(1.0 - sin1 / (mu1 + 1e-8), 0.0, 1.0))
            margin2 = float(np.clip(1.0 - sin2 / (mu2 + 1e-8), 0.0, 1.0))
            friction_margin = 0.5 * (margin1 + margin2)
            return opposition * axis_alignment * friction_margin

        wrenches = []
        for pos, normal, mu, _label in contacts:
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            tmp = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            t1 = np.cross(normal, tmp)
            t1 /= (np.linalg.norm(t1) + 1e-8)
            t2 = np.cross(normal, t1)
            r = pos - brick_centroid
            for k in range(n_friction_edges):
                theta = 2.0 * np.pi * k / n_friction_edges
                force_dir = normal + mu * (np.cos(theta) * t1 + np.sin(theta) * t2)
                force_dir /= (np.linalg.norm(force_dir) + 1e-8)
                torque = np.cross(r, force_dir)
                wrenches.append(np.concatenate([force_dir, torque]))
        wrenches = np.array(wrenches)
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
        v_index = index_tip - brick_pos
        v_thumb = thumb_tip - brick_pos
        n_index = v_index / (np.linalg.norm(v_index) + 1e-8)
        n_thumb = v_thumb / (np.linalg.norm(v_thumb) + 1e-8)
        opposition = -float(np.dot(n_index, n_thumb))
        return np.clip(opposition, -1.0, 1.0)

    # hypernet addition: goal_idx replaces brick_pos/brick_quat args
    def reset(self, goal_idx=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:23] = HOME_QPOS
        self.data.ctrl[:]   = self.ctrl_mid

        if goal_idx is None:
            goal_idx = np.random.randint(N_GOALS)
        self.current_goal_idx = goal_idx
        pos  = GOAL_POSITIONS[goal_idx].copy()
        quat = np.array([1., 0., 0., 0.])

        self.data.qpos[23:26] = pos
        self.data.qpos[26:30] = quat
        mujoco.mj_forward(self.model, self.data)

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
        return np.concatenate([robot_qpos, brick_pos, TARGET_POS, palm_pos,
                                ff_tip_pos - brick_pos, th_tip_pos - brick_pos])

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

        approach_vec = brick_pos - palm_pos
        approach_vec[2] = 0.0
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

        brick_start_z  = self.episode_brick_start[2]
        brick_xy_drift = np.linalg.norm(brick_pos[:2] - self.episode_brick_start[:2])
        brick_lift     = max(0.0, brick_pos[2] - brick_start_z)

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

        MIN_GRASP_STEPS = 30
        lifted_enough = brick_lift >= LIFT_THRESH
        is_holding    = self.contact_steps >= contact_hold_required
        success       = is_holding and lifted_enough and self.steps >= MIN_GRASP_STEPS

        approach_reward = exp(-4.0 * d_palm) * 0.5 * (1.0 + 2.0 * float(has_contact)) * float(d_palm > 0.08)
        grasping_contact = float(near_and_closing)
        contact_bonus    = 3.0 * grasping_contact
        contact_streak_reward = 3.0 * (self.contact_steps / contact_hold_required)

        done = False
        shove_penalty = -1.0 * max(0.0, brick_xy_drift - 0.10)

        lift_bonus = 10.0 * (brick_lift ** 0.5) * float(is_holding)
        if is_holding and brick_lift > 0.02:
            lift_bonus += 5.0
        if is_holding and brick_lift > 0.05:
            lift_bonus += 10.0

        grasp_bonus = 20.0 * float(success)
        lift_frac = min(brick_lift / LIFT_THRESH, 1.0)
        completion_bonus = 3.0 * float(is_holding) * lift_frac

        palm_vel_z = float(self.data.cvel[self.palm_id, 5])
        lift_velocity_reward = 0.5 * max(0.0, palm_vel_z) * float(is_holding)

        time_penalty = -0.01 - 0.3 * max(0.0, d_palm - 0.05)

        pinch_contacts = [c for c in contacts if c[3] in ("index", "thumb")]
        grasp_quality  = self._epsilon_quality(pinch_contacts, brick_pos)
        closure_reward = 1.5 * grasp_quality * float(d_palm < PALM_THRESH)

        opposition_reward = 0.5 * self._thumb_opposition_reward()
        finger_closing_reward = 1.5 * (index_closure + thumb_closure) * float(d_palm < 0.20)

        if self.contact_steps >= 2:
            lift_guidance = 2.0 * max(0.0, brick_pos[2] - self.episode_brick_start[2])
        else:
            lift_guidance = 0.0

        if d_palm < 0.20:
            open_penalty = -2.0 * (1.0 - index_closure) - 2.0 * (1.0 - thumb_closure)
        else:
            open_penalty = 0.0

        JOINT_LIMITS_LOW  = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        JOINT_LIMITS_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        arm_qpos = self.data.qpos[:7]
        low_violation  = np.maximum(0.0, JOINT_LIMITS_LOW - arm_qpos)
        high_violation = np.maximum(0.0, arm_qpos - JOINT_LIMITS_HIGH)
        joint_limit_penalty = -2.0 * float(np.sum(low_violation + high_violation))
        joint1_penalty = -1.0 * abs(self.data.qpos[0])

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

        fingertip_ids = [self.model.body(name).id for name in ['ff_tip', 'th_tip']]
        min_fingertip_z = float(min(float(self.data.xpos[fid][2]) for fid in fingertip_ids))
        if min_fingertip_z < TABLE_Z + 0.01:
            reward -= 10.0
            done = True

        done = done or success or (self.steps >= MAX_EP_STEPS)
        finger_closure = 0.5 * (index_closure + thumb_closure)
        return self._get_obs(), reward, done, d_palm, has_contact, finger_closure


# GAE (identical to grasp train)

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


# PPO loss for hypernet agent

def ppo_loss_hyper(params, agent, obs, tgt, actions, old_lp,
                   advantages, returns, std, old_values):
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


# Weight transfer from grasp checkpoint

def load_hypernet_params(grasp_ckpt_path, hyper_agent, key):
    # Init new agent to get the right pytree structure
    dummy_obs = jnp.zeros(38)
    dummy_tgt = jnp.array(GOAL_POSITIONS[0])
    new_params = hyper_agent.init(key, dummy_obs, dummy_tgt)

    if not os.path.exists(grasp_ckpt_path):
        print(f"  WARNING: Grasp checkpoint not found at {grasp_ckpt_path}")
        print(f"  Starting hypernet from scratch.")
        return new_params

    with open(grasp_ckpt_path, "rb") as f:
        old_params = pickle.load(f)

    old_p = old_params['params']
    new_p = new_params['params']

    # Transfer matching submodules
    transferred = dict(new_p)
    transfer_keys = []

    if 'state_encoder' in old_p:
        transferred['state_encoder'] = old_p['state_encoder']
        transfer_keys.append('state_encoder')

    if 'goal_encoder' in old_p:
        transferred['goal_encoder'] = old_p['goal_encoder']
        transfer_keys.append('goal_encoder')

    if 'value_fn' in old_p:
        transferred['value_fn'] = old_p['value_fn']
        transfer_keys.append('value_fn')

    # Policy trunk to hyper_policy trunk (same Dense layers, same dims)
    if 'policy' in old_p:
        transferred['hyper_policy'] = old_p['policy']
        transfer_keys.append('hyper_policy (from policy)')

    print(f"  Transferred weights: {transfer_keys}")
    print(f"  Randomly initialized: hypernet")

    # Initialize hypernet so it outputs the base policy's final layer weights
    # Zero all kernels so output is constant (independent of z_g)
    # Set output biases to match policy's Dense_2 kernel/bias
    # Initialize hypernet to reproduce base policy final layer
    if 'policy' in old_p:
        policy_W = old_p['policy']['Dense_2']['kernel']  # (256, 23)
        policy_b = old_p['policy']['Dense_2']['bias']    # (23,)

        hyper_p = jax.tree_util.tree_map(lambda x: x * 0.0, new_p['hypernet'])
        hyper_p = dict(hyper_p)
        hyper_p['Dense_2'] = {
            'kernel': jnp.zeros_like(new_p['hypernet']['Dense_2']['kernel']),
            'bias':   policy_W.flatten(),  # (5888,)
        }
        hyper_p['Dense_3'] = {
            'kernel': jnp.zeros_like(new_p['hypernet']['Dense_3']['kernel']),
            'bias':   policy_b,            # (23,)
        }
        transferred['hypernet'] = hyper_p
        print(f"  Hypernet initialized to reproduce base policy final layer")
    else:
        transferred['hypernet'] = jax.tree_util.tree_map(
            lambda x: x * 0.0, new_p['hypernet'])

    return {'params': transferred}


# Video recording

def record_rollout_hyper(mj_model, state, agent, forward_fn, update_idx, goal_idx=None):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    rec_env = HyperEnv(mj_model)

    # Record one rollout per goal if goal_idx is None, else just that goal
    goals_to_record = range(N_GOALS) if goal_idx is None else [goal_idx]

    for gidx in goals_to_record:
        obs  = rec_env.reset(goal_idx=gidx)
        tgt  = jnp.array(GOAL_POSITIONS[gidx])
        renderer = mujoco.Renderer(mj_model, height=RECORD_HEIGHT, width=RECORD_WIDTH)
        frames   = []
        done     = False
        MIN_RECORD_STEPS = 120

        while rec_env.steps < max(MIN_RECORD_STEPS, MAX_EP_STEPS):
            if done and rec_env.steps >= MIN_RECORD_STEPS:
                break

            renderer.update_scene(rec_env.data, camera="global_cam")
            frame = renderer.render().copy()

            has_contact = rec_env._has_brick_contact()
            brick_pos   = rec_env.data.qpos[23:26]
            lift        = max(0.0, brick_pos[2] - rec_env.episode_brick_start[2])

            bar_color = [0, 220, 0] if has_contact else [220, 0, 0]
            frame[:8, :]  = bar_color

            lift_frac = min(lift / LIFT_THRESH, 1.0)
            lift_px   = int(lift_frac * RECORD_WIDTH)
            frame[8:16, :lift_px] = [0, 100, 255]

            frames.append(frame)

            a, _ = forward_fn(state.params, jnp.array(obs), tgt)
            obs, _, done, _, _, _ = rec_env.step(np.array(a), CONTACT_HOLD_REQUIRED)

        renderer.close()
        if frames:
            goal_name  = GOAL_NAMES[gidx]
            video_path = os.path.join(
                VIDEO_DIR, f"hyper_update_{update_idx:06d}_{goal_name}.mp4"
            )
            imageio.mimwrite(video_path, frames, fps=RECORD_FPS, quality=8)
            print(f"  🎥 [{goal_name}] Video saved ({len(frames)} frames): {video_path}")


# Main training loop

def train():
    print("=== Panda-Lego Hypernet Phase ===")
    print(f"JAX devices: {jax.devices()}")
    print(f"Goal positions ({N_GOALS}):")
    for i, (name, pos) in enumerate(zip(GOAL_NAMES, GOAL_POSITIONS)):
        print(f"  [{i}] {name}: {pos}")

    print("\nBuilding MuJoCo model...")
    mj_model = build_model(ROBOT_XML, ASSETS_DIR, disable_collisions=True)
    envs     = [HyperEnv(mj_model) for _ in range(N_ENVS)]
    print(f"  nq={mj_model.nq}, nu={mj_model.nu}")

    # Assign each env a goal and distribute evenly across goals
    env_goals = [i % N_GOALS for i in range(N_ENVS)]
    print(f"  Goal assignment per env: {env_goals}")

    # Build hypernet agent
    agent = LegoAgentHyper(act_dim=envs[0].act_dim)
    key   = jax.random.PRNGKey(42)

    print(f"\nLoading weights from: {GRASP_CKPT}")
    params = load_hypernet_params(GRASP_CKPT, agent, key)
    n_p = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Total parameters: {n_p:,}")

    # Optimizer
    schedule = optax.linear_schedule(LR_START, LR_END, TOTAL_UPDATES)
    tx = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(schedule))
    state = TrainState.create(apply_fn=agent.apply, params=params, tx=tx)

    @jit
    def forward(params, obs, tgt):
        a, v, _, _ = agent.apply(params, obs, tgt)
        return a, v

    @jit
    def loss_and_grad(params, obs_j, tgt_j, act_j, logp_j, adv_j, ret_j, std_j, old_val_j):
        return value_and_grad(
            lambda p: ppo_loss_hyper(p, agent, obs_j, tgt_j, act_j, logp_j,
                                     adv_j, ret_j, std_j, old_val_j),
            has_aux=True,
        )(params)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # Per-goal contact tracking for ablation logging
    goal_contact_accum = np.zeros(N_GOALS)
    goal_contact_count = np.zeros(N_GOALS, dtype=int)

    # Reset all envs with assigned goals
    obs_list = [envs[i].reset(goal_idx=env_goals[i]) for i in range(N_ENVS)]
    ep_rews       = np.zeros(N_ENVS)
    ep_lens       = np.zeros(N_ENVS, dtype=int)
    all_rews      = []
    all_successes = []
    best_success  = 0.0

    print(f"\nTraining: {TOTAL_UPDATES} updates | {N_ENVS} envs | {N_STEPS} steps/update")
    print(f"Encoder freeze: first {FREEZE_ENCODER_UPDATES} updates")
    print("-" * 70)

    inv_weights = np.ones(N_GOALS)  # initialized flat; updated each update

    for update in range(1, TOTAL_UPDATES + 1):

        frac        = min(update / STD_DECAY, 1.0)
        current_std = float(STD_START + frac * (STD_END - STD_START))
        std_j       = jnp.array(current_std)

        obs_buf  = np.zeros((N_STEPS, N_ENVS, 38),              dtype=np.float32)
        tgt_buf  = np.zeros((N_STEPS, N_ENVS, 3),               dtype=np.float32)
        act_buf  = np.zeros((N_STEPS, N_ENVS, envs[0].act_dim), dtype=np.float32)
        logp_buf = np.zeros((N_STEPS, N_ENVS),                  dtype=np.float32)
        val_buf  = np.zeros((N_STEPS, N_ENVS),                  dtype=np.float32)
        rew_buf  = np.zeros((N_STEPS, N_ENVS),                  dtype=np.float32)
        done_buf = np.zeros((N_STEPS, N_ENVS),                  dtype=np.float32)
        goal_buf = np.zeros((N_STEPS, N_ENVS),                  dtype=np.int32)

        ep_contacts = np.zeros(N_ENVS)
        ep_dpalms   = []
        ep_lifts    = []

        for s in range(N_STEPS):
            for i, env in enumerate(envs):
                goal_idx = env.current_goal_idx
                tgt  = jnp.array(GOAL_POSITIONS[goal_idx])
                obs  = jnp.array(obs_list[i])
                a, v = forward(state.params, obs, tgt)
                a        = np.array(a)
                noise    = np.random.normal(0, current_std, a.shape).astype(np.float32)
                a_noisy  = np.clip(a + noise, -1.0, 1.0)
                lp       = float(-0.5 * np.sum(((a_noisy - a) / current_std) ** 2))

                next_obs, rew, done, d_palm, has_contact, closure = env.step(a_noisy, CONTACT_HOLD_REQUIRED)

                obs_buf[s, i]  = obs_list[i]
                tgt_buf[s, i]  = GOAL_POSITIONS[env_goals[i]]
                act_buf[s, i]  = a_noisy
                logp_buf[s, i] = lp
                val_buf[s, i]  = float(v)
                rew_buf[s, i]  = rew
                done_buf[s, i] = float(done)
                goal_buf[s, i] = goal_idx

                ep_rews[i]    += rew
                ep_lens[i]    += 1
                ep_contacts[i] = ep_contacts[i] * 0.95 + float(has_contact) * 0.05
                ep_dpalms.append(d_palm)
                brick_pos_log = env.data.qpos[23:26].copy()
                ep_lifts.append(max(0.0, float(brick_pos_log[2] - env.episode_brick_start[2])))

                # Per-goal contact tracking
                goal_contact_accum[goal_idx] += float(has_contact)
                goal_contact_count[goal_idx] += 1

                if done:
                    true_success = (
                        env.contact_steps >= CONTACT_HOLD_REQUIRED and
                        (env.data.qpos[25] - env.episode_brick_start[2]) >= LIFT_THRESH
                    )
                    all_rews.append(ep_rews[i])
                    all_successes.append(float(true_success))
                    ep_rews[i] = 0.0
                    ep_lens[i] = 0

                    # Resample goal each episode 
                    new_goal = np.random.randint(N_GOALS)
                    env_goals[i] = new_goal
                    obs_list[i]  = env.reset(goal_idx=new_goal)
                else:
                    obs_list[i] = next_obs

        # Bootstrap values for GAE
        next_values = np.array([
            float(forward(
                state.params,
                jnp.array(obs_list[i]),
                jnp.array(GOAL_POSITIONS[env_goals[i]])
            )[1])
            for i in range(N_ENVS)
        ])

        adv, ret = compute_gae(rew_buf, val_buf, done_buf, next_values)
        adv_flat = adv.flatten().astype(np.float32)
        ret_flat = ret.flatten().astype(np.float32)

        # Per-goal advantage normalization + inverse contact rate weighting
        goal_flat = goal_buf.flatten()
        adv_normalized = np.zeros_like(adv_flat)

        # Compute per-goal contact rates from this update's accumulators
        goal_rates = np.array([
            goal_contact_accum[g] / goal_contact_count[g]
            if goal_contact_count[g] > 0 else 0.1
            for g in range(N_GOALS)
        ])

        # Inverse weighting: goals with low contact get upweighted
        # floor at 0.05 so zero-contact goals don't get infinite weight
        inv_weights = 1.0 / (goal_rates + 0.05)
        inv_weights = inv_weights / inv_weights.mean()  # normalize so mean weight = 1

        for g in range(N_GOALS):
            mask = (goal_flat == g)
            if mask.sum() > 1:
                g_adv = adv_flat[mask]
                adv_normalized[mask] = (
                    (g_adv - g_adv.mean()) / (g_adv.std() + 1e-8)
                ) * inv_weights[g]
            elif mask.sum() == 1:
                adv_normalized[mask] = 0.0

        adv_flat = adv_normalized

        obs_j     = jnp.array(obs_buf.reshape(-1, 38))
        # Per-sample goal vectors, critical difference from grasp train
        tgt_j     = jnp.array(tgt_buf.reshape(-1, 3))
        act_j     = jnp.array(act_buf.reshape(-1, envs[0].act_dim))
        logp_j    = jnp.array(logp_buf.flatten())
        adv_j     = jnp.array(adv_flat)
        ret_j     = jnp.array(ret_flat)
        old_val_j = jnp.array(val_buf.flatten().astype(np.float32))

        for _ in range(N_EPOCHS):
            (loss, aux), grads = loss_and_grad(
                state.params, obs_j, tgt_j, act_j, logp_j, adv_j, ret_j, std_j, old_val_j
            )

            # Freeze encoders for first FREEZE_ENCODER_UPDATES updates
            if update <= FREEZE_ENCODER_UPDATES:
                import flax
                frozen_grad_params = dict(grads['params'])
                frozen_grad_params['state_encoder'] = jax.tree_util.tree_map(
                    jnp.zeros_like, grads['params']['state_encoder']
                )
                frozen_grad_params['goal_encoder'] = jax.tree_util.tree_map(
                    jnp.zeros_like, grads['params']['goal_encoder']
                )
                grads = {'params': frozen_grad_params}

            state = state.apply_gradients(grads=grads)

        # Logging
        if update % LOG_INTERVAL == 0:
            mr      = np.mean(all_rews[-50:])      if all_rews      else 0.0
            suc     = np.mean(all_successes[-50:]) if all_successes else 0.0
            pg, vf  = aux
            mean_dp = np.mean(ep_dpalms) if ep_dpalms else 0.0
            mean_contact = float(np.mean(ep_contacts))
            mean_lift    = float(np.mean(ep_lifts)) if ep_lifts else 0.0
            frozen_str   = " [FROZEN]" if update <= FREEZE_ENCODER_UPDATES else ""

            print(f"Update {update:5d} | Rew {mr:7.2f} | Succ {suc*100:5.1f}% | "
                  f"PG {float(pg):6.3f} | VF {float(vf):5.1f} | "
                  f"dPalm {mean_dp:.3f} | Contact {mean_contact:.2f} | "
                  f"Lift {mean_lift:.3f}{frozen_str}")

            # Per-goal contact rates
            goal_strs = []
            for g in range(N_GOALS):
                if goal_contact_count[g] > 0:
                    rate = goal_contact_accum[g] / goal_contact_count[g]
                    goal_strs.append(f"{GOAL_NAMES[g]}={rate:.2f}")
                else:
                    goal_strs.append(f"{GOAL_NAMES[g]}=N/A")
            print(f"         Contact/goal: {' | '.join(goal_strs)}")

            # Log per-goal advantage weights to see reweighting in action
            weight_strs = [f"{GOAL_NAMES[g]}={inv_weights[g]:.2f}" for g in range(N_GOALS)]
            print(f"         Adv weights:  {' | '.join(weight_strs)}")

            # Reset per-goal accumulators each log interval
            goal_contact_accum[:] = 0.0
            goal_contact_count[:] = 0

            if suc > best_success:
                best_success = suc
                best_path = f"{CKPT_DIR}/hyper_best.pkl"
                with open(best_path, "wb") as f:
                    pickle.dump(state.params, f)
                print(f"  ★ New best: {suc*100:.1f}% → {best_path}")

        if update % SAVE_INTERVAL == 0:
            path = f"{CKPT_DIR}/hyper_agent_{update}.pkl"
            with open(path, "wb") as f:
                pickle.dump(state.params, f)
            print(f"  → Checkpoint: {path}")

        if update % RECORD_INTERVAL == 0:
            print(f"  Recording rollouts at update {update}...")
            record_rollout_hyper(mj_model, state, agent, forward, update)

    print("\n Hypernet training complete!")
    return state


# Ablation: run fixed LegoAgent on all goal positions for comparison

def run_ablation_baseline(n_episodes_per_goal=100):
    print("\n Ablation Baseline: Fixed Policy on All Goals ")

    mj_model   = build_model(ROBOT_XML, ASSETS_DIR, disable_collisions=False)
    fixed_env  = HyperEnv(mj_model)
    fixed_agent = LegoAgent(act_dim=fixed_env.act_dim)
    key         = jax.random.PRNGKey(0)

    dummy_obs = jnp.zeros(38)
    dummy_tgt = jnp.array(GOAL_POSITIONS[0])
    params    = fixed_agent.init(key, dummy_obs, dummy_tgt)

    if os.path.exists(GRASP_CKPT):
        with open(GRASP_CKPT, "rb") as f:
            params = pickle.load(f)
        print(f"  Loaded: {GRASP_CKPT}")
    else:
        print(f"  WARNING: {GRASP_CKPT} not found, using random params")

    # Fixed target, the center position the base policy was trained on
    FIXED_TARGET = jnp.array([0.5, 0.15, 0.42])

    @jit
    def fixed_forward(params, obs):
        a, v, _, _ = fixed_agent.apply(params, obs, FIXED_TARGET)
        return a, v

    results = {}
    for gidx, (name, goal_pos) in enumerate(zip(GOAL_NAMES, GOAL_POSITIONS)):
        contacts = []
        for ep in range(n_episodes_per_goal):
            obs  = fixed_env.reset(goal_idx=gidx)
            ep_contact = 0
            ep_steps   = 0
            done = False
            while not done and ep_steps < MAX_EP_STEPS:
                a, _ = fixed_forward(params, jnp.array(obs))
                obs, _, done, _, has_contact, _ = fixed_env.step(np.array(a), CONTACT_HOLD_REQUIRED)
                ep_contact += int(has_contact)
                ep_steps   += 1
            contacts.append(ep_contact / max(ep_steps, 1))

        mean_contact = np.mean(contacts)
        results[name] = mean_contact
        print(f"  [{name}] contact rate: {mean_contact:.3f} "
              f"({np.std(contacts):.3f} std) over {n_episodes_per_goal} episodes")

    print("\nBaseline summary (fixed policy, fixed target = center):")
    for name, rate in results.items():
        bar = "█" * int(rate * 20)
        print(f"  {name:10s}: {rate:.3f} {bar}")

    # Save results
    results_path = os.path.join(CKPT_DIR, "ablation_baseline.pkl")
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Saved baseline results: {results_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true",
                        help="Run fixed-policy ablation baseline instead of training")
    parser.add_argument("--ablation-episodes", type=int, default=100,
                        help="Episodes per goal for ablation baseline")
    args = parser.parse_args()

    if args.ablation:
        run_ablation_baseline(n_episodes_per_goal=args.ablation_episodes)
    else:
        train()

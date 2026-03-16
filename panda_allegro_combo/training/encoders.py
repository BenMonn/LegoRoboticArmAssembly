"""
training/encoders.py

Simple goal encoder and state encoder networks.

GoalEncoder:  target_pose (3,)  → z_g (latent_dim,)
StateEncoder: observation (32,) → z_t (latent_dim,)

Both are small MLPs for now. SimCLR contrastive training
will be layered on top of StateEncoder in a later phase.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence


# ---------------------------------------------------------------------------
# Shared MLP backbone
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    features: Sequence[int]
    activate_final: bool = False

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i < len(self.features) - 1 or self.activate_final:
                x = nn.relu(x)
        return x


# ---------------------------------------------------------------------------
# Goal Encoder   z_g = f(x_g)
# ---------------------------------------------------------------------------

class GoalEncoder(nn.Module):
    """
    Encodes a target brick pose into a latent goal vector z_g.

    Input:  target_pos (3,)  -- x, y, z of target placement
    Output: z_g (latent_dim,) -- normalized latent goal embedding

    In Phase 1 of your thesis this will be replaced with a richer
    encoder over LEGO Studio instruction representations. For now
    it just learns a useful embedding of the 3D target position.
    """
    latent_dim: int = 32
    hidden_dims: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, target_pos: jnp.ndarray) -> jnp.ndarray:
        x = target_pos
        for feat in self.hidden_dims:
            x = nn.relu(nn.Dense(feat)(x))
        z_g = nn.Dense(self.latent_dim)(x)
        # L2 normalize so embedding lives on a hypersphere
        # This helps with downstream distance-based losses (e.g. InfoNCE)
        z_g = z_g / (jnp.linalg.norm(z_g, keepdims=True) + 1e-8)
        return z_g


# ---------------------------------------------------------------------------
# State Encoder   z_t = f(x_t)
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    """
    Encodes a robot observation into a latent state vector z_t.

    Input:  obs (32,)
              [robot_qpos(23), brick_pos(3), target_pos(3), palm_pos(3)]
    Output: z_t (latent_dim,)

    In Phase 2 of your thesis this encoder will be trained with
    SimCLR + InfoNCE contrastive loss. For now it's trained end-to-end
    with the policy via RL.
    """
    latent_dim: int = 64
    hidden_dims: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        for feat in self.hidden_dims:
            x = nn.relu(nn.Dense(feat)(x))
        z_t = nn.Dense(self.latent_dim)(x)
        z_t = z_t / (jnp.linalg.norm(z_t, keepdims=True) + 1e-8)
        return z_t


# ---------------------------------------------------------------------------
# Policy   pi(z_t, z_g) → action
# ---------------------------------------------------------------------------

class Policy(nn.Module):
    """
    Simple MLP policy conditioned on state and goal embeddings.

    Input:  [z_t (latent_dim,), z_g (latent_dim,)]  concatenated
    Output: action (act_dim,) in [-1, 1]

    This is the base policy from Phase 4 of your thesis.
    Hypernet modulation and diffusion policy will replace/extend
    this in later phases.
    """
    act_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, z_t: jnp.ndarray, z_g: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([z_t, z_g], axis=-1)
        for feat in self.hidden_dims:
            x = nn.relu(nn.Dense(feat)(x))
        action = nn.tanh(nn.Dense(self.act_dim)(x))
        return action


# ---------------------------------------------------------------------------
# Value function   V(z_t, z_g) → scalar
# ---------------------------------------------------------------------------

class ValueFunction(nn.Module):
    """
    Critic network for PPO / actor-critic training.
    Estimates expected return from (z_t, z_g).
    """
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, z_t: jnp.ndarray, z_g: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([z_t, z_g], axis=-1)
        for feat in self.hidden_dims:
            x = nn.relu(nn.Dense(feat)(x))
        value = nn.Dense(1)(x)
        return jnp.squeeze(value, axis=-1)


# ---------------------------------------------------------------------------
# Combined agent (all networks in one place)
# ---------------------------------------------------------------------------

class LegoAgent(nn.Module):
    """
    Bundles all networks. Makes parameter handling cleaner.
    """
    latent_dim_goal:  int = 32
    latent_dim_state: int = 64
    act_dim:          int = 23
    goal_input_dim:   int = 3

    def setup(self):
        self.goal_encoder  = GoalEncoder(latent_dim=self.latent_dim_goal)
        self.state_encoder = StateEncoder(latent_dim=self.latent_dim_state)
        self.policy        = Policy(act_dim=self.act_dim)
        self.value_fn      = ValueFunction()

    def __call__(self, obs: jnp.ndarray, target_pos: jnp.ndarray):
        z_t    = self.state_encoder(obs)
        z_g    = self.goal_encoder(target_pos)
        action = self.policy(z_t, z_g)
        value  = self.value_fn(z_t, z_g)
        return action, value, z_t, z_g

    def get_action(self, obs: jnp.ndarray, target_pos: jnp.ndarray):
        z_t = self.state_encoder(obs)
        z_g = self.goal_encoder(target_pos)
        return self.policy(z_t, z_g)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import jax

    key = jax.random.PRNGKey(0)

    agent = LegoAgent(act_dim=23)

    # Dummy inputs
    obs        = jnp.ones((32,))
    target_pos = jnp.array([0.5, 0.15, 0.42])

    # Initialize
    params = agent.init(key, obs, target_pos)
    print("Agent parameter shapes:")
    print(jax.tree_util.tree_map(lambda x: x.shape, params))

    # Forward pass
    action, value, z_t, z_g = agent.apply(params, obs, target_pos)
    print(f"\naction shape : {action.shape}")   # (23,)
    print(f"value        : {value:.4f}")
    print(f"z_t shape    : {z_t.shape}")        # (64,)
    print(f"z_g shape    : {z_g.shape}")        # (32,)
    print(f"action range : [{action.min():.3f}, {action.max():.3f}]")

    # Batched forward pass (256 envs)
    batch_obs        = jnp.ones((256, 32))
    batch_target_pos = jnp.tile(target_pos, (256, 1))
    batched_forward  = jax.vmap(lambda o, t: agent.apply(params, o, t))
    actions, values, _, _ = batched_forward(batch_obs, batch_target_pos)
    print(f"\nBatched actions shape: {actions.shape}")  # (256, 23)
    print("\n✅ Encoder tests passed!")
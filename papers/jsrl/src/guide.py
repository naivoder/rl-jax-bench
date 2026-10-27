"""Guide policy pi_g: weak PPO snapshot from the earlier DQfD reproduction (data/guides/ppo_<Game>_subopt.msgpack, sha256 in data/guides/SHA256.txt).
PUBLIC teacher (sees only the observation): argmax of the PPO logits, replaced by a uniform random action with probability eps_g
(the noise at which its demos were collected and its true score was measured)."""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
from flax import serialization

from .model import ActorCritic

GUIDE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "guides")


def load_guide(env, cfg, path: str | None = None):
    """-> (net, params) for the PPO actor-critic of this game."""
    net = ActorCritic(env.n_actions, channels=cfg["network"]["conv_channels"], hidden=cfg["ppo"]["hidden"])
    tmpl = net.init(jax.random.PRNGKey(0), jnp.zeros((1,) + env.obs_shape, jnp.float32))
    path = path or os.path.join(GUIDE_DIR, f"ppo_{env.game}_subopt.msgpack")
    with open(path, "rb") as f:
        params = serialization.from_bytes(tmpl, f.read())
    return net, params


def make_guide_fn(env, cfg, eps_g: float, path: str | None = None):
    """guide_fn(obs_bool (H,W,C), key) -> int32 action.  Single observation (vmapped by the callers)."""
    net, params = load_guide(env, cfg, path)

    def guide_fn(obs, key):
        logits, _ = net.apply(params, obs[None].astype(jnp.float32))
        k1, k2 = jax.random.split(key)
        rand = jax.random.randint(k1, (), 0, env.n_actions)
        return jnp.where(jax.random.uniform(k2) < eps_g, rand, jnp.argmax(logits[0])).astype(jnp.int32)

    return guide_fn

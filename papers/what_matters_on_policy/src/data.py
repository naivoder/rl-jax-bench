"""Environments and running statistics. This paper has no dataset: the data are on-policy rollouts.

Environments (README): brax 0.14.2 ``halfcheetah``/``walker2d``/``hopper`` on the ``mjx`` backend,
labelled PROXY for gym ``*-v1`` on MuJoCo 2.0. Episode limit 1000 (paper §3.6) and the time-limit cut is TERMINAL
(C22 = False): done = brax-done OR t == 1000, and the value of the next state is never bootstrapped.

Auto-reset (ours, not brax's AutoResetWrapper, which re-uses one fixed initial state per env): every iteration a
fresh reset state is drawn per env from fold_in(k_env, iteration + 1); an env whose episode ends during the
iteration restarts from it. Each env thus restarts from an independent random initial state, as gym does.

Running statistics = official ``AverageMeanStd`` (seed_rl running_statistics.py): exact batched Welford,
population variance, std = sqrt(max(1e-12, summed_var) / max(1e-12, count)) clipped to [1e-6, 1e6]; before any
data mean 0 / std 1. Used for the observation normaliser (C64) and the value-target normaliser (C66).
"""
from __future__ import annotations

import hashlib
import warnings
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

STD_MIN, STD_MAX = 1e-6, 1e6


# ------------------------------------------------------------------ AverageMeanStd (official semantics)
class MeanStd(NamedTuple):
    count: jax.Array        # (size) float32
    mean: jax.Array         # (size)
    summed_var: jax.Array   # (size)


def meanstd_init(size: int, dtype=jnp.float32) -> MeanStd:
    z = jnp.zeros((size,), dtype)
    return MeanStd(z, z, z)


def meanstd_update(ms: MeanStd, data) -> MeanStd:
    """data: (..., size). Official update: count += n; mean += sum(x - m_old)/count; S += sum((x-m_old)(x-m_new))."""
    flat = data.reshape(-1, data.shape[-1])
    count = ms.count + flat.shape[0]
    d_old = flat - ms.mean
    mean = ms.mean + d_old.sum(0) / count
    s = ms.summed_var + (d_old * (flat - mean)).sum(0)
    return MeanStd(count, mean, s)


def meanstd_get(ms: MeanStd):
    minval = STD_MIN * STD_MIN
    std = jnp.sqrt(jnp.maximum(minval, ms.summed_var) / jnp.maximum(minval, ms.count))
    return ms.mean, jnp.clip(std, STD_MIN, STD_MAX)


def normalize_obs(ms: MeanStd, obs, on, clip):
    """C64/C65: (o - mean)/std then clip to +-clip; ``on`` False = FixedMeanStd identity and no clipping."""
    mean, std = meanstd_get(ms)
    y = jnp.clip((obs - mean) / std, -clip, clip)
    return jnp.where(on, y, obs)


# ------------------------------------------------------------------ environment
class EnvSpec(NamedTuple):
    name: str
    obs_dim: int
    act_dim: int
    episode_length: int


def make_env(name: str, backend: str = "mjx"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from brax import envs as benvs
        env = benvs.get_environment(name, backend=backend)
    return env, EnvSpec(name, int(env.observation_size), int(env.action_size), 1000)


class EnvState(NamedTuple):
    bstate: Any          # brax State, leading (N)
    t: jax.Array         # (N) int32 steps into the current episode
    ep_ret: jax.Array    # (N) running undiscounted return (training-episode logging)


def env_reset(env, keys) -> EnvState:
    s = jax.vmap(env.reset)(keys)
    n = keys.shape[0]
    return EnvState(s, jnp.zeros((n,), jnp.int32), jnp.zeros((n,), jnp.float32))


def env_step(env, es: EnvState, action, reset_pool, episode_length: int):
    """Step N envs; ended envs restart from ``reset_pool`` (brax State with leading N). Returns
    (new_state, next_obs, reward, done, finished_return) where next_obs is the reset obs for ended envs (SEED
    convention: the terminal obs is never seen by the learner; the time-limit cut is terminal, C22 = False)."""
    s2 = jax.vmap(env.step)(es.bstate, action)
    t2 = es.t + 1
    done = jnp.logical_or(s2.done > 0.5, t2 >= episode_length)
    ret = es.ep_ret + s2.reward
    sel = lambda a, b: jnp.where(done.reshape(done.shape + (1,) * (a.ndim - 1)), a, b)
    new_b = jax.tree.map(sel, reset_pool, s2)
    new = EnvState(new_b, jnp.where(done, 0, t2), jnp.where(done, 0.0, ret))
    return new, new_b.obs, s2.reward, done, jnp.where(done, ret, 0.0)


def env_fingerprint(env, spec: EnvSpec, n_steps: int = 1000) -> dict:
    """sha256 of a fixed-key random-action trajectory (obs, reward, done) and of the env source/XML (README)."""
    import inspect
    import numpy as np
    s = env.reset(jax.random.PRNGKey(0))
    acts = jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(0), 999), (n_steps, spec.act_dim),
                              minval=-1.0, maxval=1.0)

    def body(c, a):
        c = env.step(c, a)
        return c, (c.obs, c.reward, c.done)

    _, (o, r, d) = jax.jit(lambda s, a: jax.lax.scan(body, s, a))(s, acts)
    h = hashlib.sha256()
    for x in (o, r, d):
        h.update(np.asarray(x, np.float32).tobytes())
    src = inspect.getsourcefile(type(env))
    return {"traj_sha256": h.hexdigest(), "source": src,
            "source_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
            "first_reward": float(r[0]), "sum_reward": float(np.asarray(r).sum())}

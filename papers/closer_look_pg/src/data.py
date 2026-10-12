"""Environment layer: brax MJX Hopper/Walker2d proxies, running normalisers, single-actor rollouts.

This paper has no dataset; "data" here is on-policy experience. Semantics follow the official
companion code (MadryLab/implementation-matters@5ee6ecb, custom_env.py / agent.py) re-derived, not copied:

* every collection is a FRESH rollout from env reset (agent.run_trajectories resets);
* the 1000-step time limit sets done=1 and is treated as terminal, no bootstrap (gym-v2 TimeLimit);
* obs normaliser = running mean/std pushed BEFORE normalising (ZFilter); it pushes the post-step obs every
  step and, on done, also the reset obs (custom_env.Env.step / reset);
* PPO reward normaliser = r / (std of the running discounted return + 1e-8), clipped; the return
  accumulator resets on env reset (RewardFilter);
* RunningStat.var = S/(n-1) for n>1 else mean^2 (joschu/modular_rl convention used by the official code).

``update_norm=False`` freezes both normalisers : instrument rollouts never touch them.
"""
from __future__ import annotations

import hashlib
import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from brax import envs as brax_envs

from .model import gauss_logp


# ------------------------------------------------------------------ running statistics
class RunStat(NamedTuple):
    n: jax.Array      # scalar float32 count
    m: jax.Array      # mean
    s: jax.Array      # sum of squared deviations


def runstat_init(shape) -> RunStat:
    return RunStat(jnp.zeros((), jnp.float32), jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32))


def runstat_push(rs: RunStat, x, mask=True) -> RunStat:
    """Welford update (joschu RunningStat.push); ``mask`` False leaves the stat unchanged."""
    n1 = rs.n + 1.0
    m1 = jnp.where(n1 == 1.0, x, rs.m + (x - rs.m) / n1)
    s1 = jnp.where(n1 == 1.0, rs.s, rs.s + (x - rs.m) * (x - m1))
    mask = jnp.asarray(mask)
    return RunStat(jnp.where(mask, n1, rs.n), jnp.where(mask, m1, rs.m), jnp.where(mask, s1, rs.s))


def runstat_std(rs: RunStat):
    var = jnp.where(rs.n > 1.0, rs.s / jnp.maximum(rs.n - 1.0, 1.0), rs.m ** 2)
    return jnp.sqrt(var)


class NormState(NamedTuple):
    obs: RunStat
    rew: RunStat


def norm_init(obs_dim: int) -> NormState:
    return NormState(runstat_init((obs_dim,)), runstat_init(()))


def obs_filter(rs: RunStat, x, clip, update, mask=True):
    rs = runstat_push(rs, x, jnp.logical_and(update, mask))      # update may be a python or traced bool
    y = (x - rs.m) / (runstat_std(rs) + 1e-8)
    if clip is not None:
        y = jnp.clip(y, -clip, clip)
    return rs, y


# ------------------------------------------------------------------ environment
class EnvSpec(NamedTuple):
    name: str
    obs_dim: int
    act_dim: int


def make_env(name: str, backend: str = "mjx", post_step_kinematics: bool = False):
    """brax MJX proxy env. post_step_kinematics=True  refreshes the forward kinematics after each
    env step: brax's MJX pipeline sets State.x = data.xpos computed BEFORE the last substep's integration, so the
    env's forward-reward displacement x.pos[0,0], its height x.pos[0,2] (obs[0] and the healthy check) lag qpos by
    one 2 ms substep; gym/gymnasium read qpos after do_simulation. False reproduces the pre-fix (full-scale) env."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = brax_envs.get_environment(name, backend=backend)
    if post_step_kinematics:
        _refresh_kinematics_after_step(env)
    return env, EnvSpec(name, env.observation_size, env.action_size)


def _refresh_kinematics_after_step(env):
    """Shadow env.pipeline_step with a version whose returned State.x is the forward kinematics of the FINAL qpos
    (mjx.kinematics on the post-step data). Only x (body positions/orientations) is replaced; q, qd and the contact
    data are untouched, and the brax env's own reward/obs/done code then reads post-step positions as gym does."""
    from brax.base import Transform
    from mujoco import mjx
    base_step = env.pipeline_step
    sys_ = env.sys

    def pipeline_step(pipeline_state, action):
        ps = base_step(pipeline_state, action)
        d = mjx.kinematics(sys_, ps)
        return ps.replace(x=Transform(pos=d.xpos[1:], rot=d.xquat[1:]))

    env.pipeline_step = pipeline_step
    env._post_step_kinematics = True
    return env


def env_name_of(env) -> str:
    return {"Hopper": "hopper", "Walker2d": "walker2d"}[type(env).__name__]


def env_fingerprint(env, n_steps: int = 1000) -> dict:
    """SHA-256 of the model definition + of the first ``n_steps`` (obs, reward, done) under a fixed
    action sequence drawn from fold_in(PRNGKey(0), 999)."""
    from etils import epath
    xml = (epath.resource_path("brax") / f"envs/assets/{env_name_of(env)}.xml").read_bytes()
    sys_hash = hashlib.sha256(xml).hexdigest()
    key = jax.random.fold_in(jax.random.PRNGKey(0), 999)
    acts = jax.random.uniform(key, (n_steps, env.action_size), minval=-1.0, maxval=1.0)

    def body(st, a):
        st2 = env.step(st, a)
        st2 = jax.lax.cond(st2.done > 0, lambda: env.reset(jax.random.PRNGKey(1)), lambda: st2)
        return st2, (st2.obs, st2.reward, st2.done)

    _, (o, r, d) = jax.jit(lambda: jax.lax.scan(body, env.reset(jax.random.PRNGKey(1)), acts))()
    import numpy as np
    traj = np.concatenate([np.asarray(o).ravel(), np.asarray(r).ravel(), np.asarray(d).ravel()]).astype(np.float32)
    return {"model_sha256": sys_hash, "trajectory_sha256": hashlib.sha256(traj.tobytes()).hexdigest()}


# ------------------------------------------------------------------ rollouts
class Traj(NamedTuple):
    obs: jax.Array        # [T, obs_dim] normalised policy input s_t
    act: jax.Array        # [T, act_dim]
    logp: jax.Array       # [T] log pi_old(a_t|s_t)
    rew: jax.Array        # [T] reward in training units (normalised for PPO, raw for TRPO)
    rew_raw: jax.Array    # [T] raw env reward
    not_done: jax.Array   # [T] 0 at terminal steps (incl. time limit)
    ep_ret: jax.Array     # [T] raw return of the episode that ended at t (valid where not_done==0)
    ep_len: jax.Array     # [T] its length
    ep_idx: jax.Array     # [T] index of the episode (within this rollout) that step t belongs to
    rew_clip_hit: jax.Array  # [T] 1 where the reward clip was active


def make_rollout(env, cfg: dict, nets):
    """Returns rollout(pi_flat, norm, key, T, update_norm) -> (Traj, norm') for ONE actor from reset."""
    ep_len_max = int(cfg["episode_length"])
    gamma = float(cfg["gamma"])
    obs_clip = cfg.get("obs_clip")
    reward_norm = bool(cfg.get("reward_norm", False))
    rew_clip = cfg.get("reward_clip")

    def rollout(pi_flat, norm: NormState, key, T: int, update_norm: bool):
        k_reset0, k_steps = jax.random.split(key)
        st0 = env.reset(k_reset0)
        obs_rs, o0 = obs_filter(norm.obs, st0.obs, obs_clip, update_norm)
        carry0 = (st0, o0, jnp.int32(0), jnp.float32(0.0), jnp.float32(0.0), jnp.int32(0),
                  obs_rs, norm.rew)

        def step(carry, k):
            st, o, t, ep_ret, rf_ret, ep_i, obs_rs, rew_rs = carry
            k_a, k_r = jax.random.split(k)
            mean, log_std = nets.pi_apply(pi_flat, o)
            a = mean + jnp.exp(log_std) * jax.random.normal(k_a, mean.shape)
            logp = gauss_logp(mean, log_std, a)
            st1 = env.step(st, a)
            r_raw = st1.reward
            t1 = t + 1
            done = jnp.logical_or(st1.done > 0.5, t1 >= ep_len_max)
            ep_ret1 = ep_ret + r_raw
            # obs filter sees the post-step observation first (Env.step), reward filter next
            obs_rs, o1 = obs_filter(obs_rs, st1.obs, obs_clip, update_norm)
            if reward_norm:
                rf1 = rf_ret * gamma + r_raw
                rew_rs = runstat_push(rew_rs, rf1, update_norm)
                r = r_raw / (runstat_std(rew_rs) + 1e-8)
                hit = (jnp.abs(r) > rew_clip).astype(jnp.float32) if rew_clip is not None else jnp.float32(0)
                r = jnp.clip(r, -rew_clip, rew_clip) if rew_clip is not None else r
            else:
                rf1, r, hit = rf_ret, r_raw, jnp.float32(0)
            # on done: reset env, push the reset obs, zero the return accumulator (Env.reset)
            st_next = jax.lax.cond(done, lambda: env.reset(k_r), lambda: st1)
            obs_rs_r, o_r = obs_filter(obs_rs, st_next.obs, obs_clip, update_norm, mask=done)
            o_next = jnp.where(done, o_r, o1)
            out = Traj(o, a, logp, r, r_raw, 1.0 - done.astype(jnp.float32), ep_ret1, t1.astype(jnp.float32),
                       ep_i, hit)
            carry = (st_next, o_next, jnp.where(done, 0, t1), jnp.where(done, 0.0, ep_ret1),
                     jnp.where(done, 0.0, rf1), ep_i + done.astype(jnp.int32), obs_rs_r, rew_rs)
            return carry, out

        carry, traj = jax.lax.scan(step, carry0, jax.random.split(k_steps, T))
        return traj, NormState(carry[6], carry[7])

    return rollout


def completed_episode_mean(traj: Traj, official: bool = True):
    """Mean raw return of episodes completed in a single-actor rollout ([T] arrays).

    official=True reproduces agent.run_trajectories: an episode completing at the last step is counted only
    if no other episode completed . Returns (mean, count); mean=-1 when none completed (official)."""
    T = traj.not_done.shape[-1]
    done = traj.not_done < 0.5
    if official:
        last = jnp.arange(T) == T - 1
        others = jnp.sum(done & ~last) > 0
        done = done & ~(last & others)
    c = jnp.sum(done)
    m = jnp.sum(jnp.where(done, traj.ep_ret, 0.0)) / jnp.maximum(c, 1)
    return jnp.where(c > 0, m, -1.0), c


def unbiased_episode_returns(traj: Traj, ep_len_max: int):
    """Per-actor first floor(T/ep_len_max) episodes always complete, so their returns are an unbiased sample of
    the episode-return distribution (later, completed-only episodes are length-biased). Works on [..., T]."""
    T = traj.not_done.shape[-1]
    k = max(T // ep_len_max, 1)
    sel = (traj.not_done < 0.5) & (traj.ep_idx < k)
    return sel, traj.ep_ret

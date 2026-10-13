"""Environments and the VecNormalize-style wrappers (paper C1, K5-K9). There is no dataset in this paper: the "data" are
on-policy rollouts from vectorized, auto-resetting environments.

* gymnax classic control (CartPole-v1, Acrobot-v1, MountainCar-v0, Pendulum-v1). gymnax 1.0's ``step``
  returns (obs_after_autoreset, state, r, terminated, truncated, info{final_observation}), which is exactly the
  gymnasium-0.29 vector-env contract the CleanRL oracle sees.
* masked-velocity CartPole: obs[1] = obs[3] = 0.
* Brax hopper (PROXY): our own auto-reset so the final obs survives.

Wrapper semantics (K6-K9), two modes:
* ``baselines`` (default): one obs RMS and one return RMS shared by the N envs (VecNormalize). Per vector step:
  ret <- gamma*ret + r; obs_rms.update(obs_batch); ret_rms.update(ret); r~ = clip(r/sqrt(var+eps), +-10);
  ret[done] = 0 (done = terminated OR truncated). Obs = clip((o-mu)/sqrt(var+eps), +-10). The reset also updates.
* ``gymnasium029``: per-env wrappers (gymnasium 0.29.1 NormalizeObservation/NormalizeReward inside SyncVectorEnv):
  the terminal obs AND the auto-reset obs both update the obs RMS; returns <- returns*gamma*(1-terminated) + r
  (never reset on truncation), return RMS updated with it, r~ = clip(r/sqrt(var+eps), +-10).
* ``gym021``: the post's OWN continuous code (ppo-implementation-details@fbef824, a post-publication revision whose PPO
  files equal the pre-publication 339df2e) ppo_continuous_action.py on gym 0.21.0,
  found in authors'-code check): per-env wrappers like ``gymnasium029`` for the observation (terminal obs AND auto-reset obs both
  update the per-env obs RMS), but gym 0.21 NormalizeReward has the baselines reward order (ret <- gamma*ret + r; RMS
  update; scale; THEN ret = 0 on done, done = terminated OR truncated), with a per-env return RMS.
The raw (unscaled) reward feeds the episodic-return metric (RecordEpisodeStatistics is innermost).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


# ------------------------------------------------------------------ running mean / std (K6, K8)
class RMS(NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: jax.Array


def rms_init(shape=(), dtype=jnp.float32) -> RMS:
    # official code (baselines running_mean_std.py / gymnasium RunningMeanStd): mean 0, var 1, count 1e-4
    return RMS(jnp.zeros(shape, dtype), jnp.ones(shape, dtype), jnp.asarray(1e-4, dtype))


def rms_update(rms: RMS, x) -> RMS:
    """Parallel (Chan et al.) update from a batch x with leading batch axis."""
    b_mean, b_var, b_n = x.mean(0), x.var(0), x.shape[0]
    delta = b_mean - rms.mean
    tot = rms.count + b_n
    mean = rms.mean + delta * b_n / tot
    m2 = rms.var * rms.count + b_var * b_n + delta ** 2 * rms.count * b_n / tot
    return RMS(mean, m2 / tot, tot)


def rms_normalize(rms: RMS, x, eps, clip):
    return jnp.clip((x - rms.mean) / jnp.sqrt(rms.var + eps), -clip, clip)


def scale_reward_baselines(ret, ret_rms: RMS, r, done, gamma, eps, clip):
    """VecNormalize.step_wait (K8, K9): ret <- gamma*ret + r; update the SHARED RMS with ret (N,);
    r~ = clip(r / sqrt(var + eps)); THEN ret[done] = 0 (done = terminated OR truncated). Shapes (N,)."""
    ret = ret * gamma + r
    ret_rms = rms_update(ret_rms, ret)
    rew = jnp.clip(r / jnp.sqrt(ret_rms.var + eps), -clip, clip)
    return ret * (1.0 - done.astype(ret.dtype)), ret_rms, rew


def scale_reward_gymnasium029(ret, ret_rms: RMS, r, terminated, gamma, eps, clip):
    """gymnasium 0.29.1 NormalizeReward + TransformReward(clip), ONE env (scalars): returns <- returns*gamma*(1-term)
    + r (zeroed BEFORE adding; never on truncation); RMS update with returns; r~ = clip(r / sqrt(var + eps))."""
    ret = ret * gamma * (1.0 - terminated.astype(ret.dtype)) + r
    ret_rms = rms_update(ret_rms, ret[None])
    return ret, ret_rms, jnp.clip(r / jnp.sqrt(ret_rms.var + eps), -clip, clip)


# ------------------------------------------------------------------ base environments
class EnvSpec(NamedTuple):
    obs_dim: int
    continuous: bool
    n_out: int            # n_actions (discrete) or action dim (continuous)
    act_low: Any
    act_high: Any
    max_steps: int


def _gymnax_env(env_id: str, mask_velocity: bool):
    import gymnax
    env, params = gymnax.make(env_id)
    cont = env_id == "Pendulum-v1"
    obs_dim = env.observation_space(params).shape[0]
    if cont:
        sp = env.action_space(params)
        spec = EnvSpec(obs_dim, True, sp.shape[0] if sp.shape else 1, float(sp.low), float(sp.high),
                       int(params.max_steps_in_episode))
    else:
        spec = EnvSpec(obs_dim, False, env.action_space(params).n, None, None, int(params.max_steps_in_episode))
    mask = jnp.array([1.0, 0.0, 1.0, 0.0]) if mask_velocity else None

    def obs_fn(o):
        o = o.astype(jnp.float32)
        return o * mask if mask is not None else o

    def reset(key):
        o, s = env.reset(key, params)
        return obs_fn(o), s

    def step(key, state, action):
        o, s, r, term, trunc, info = env.step(key, state, action, params)
        return obs_fn(o), s, r.astype(jnp.float32), term, trunc, obs_fn(info["final_observation"])

    return reset, step, spec


def _brax_env(name: str, episode_length: int = 1000):
    from brax import envs as benvs
    env = benvs.get_environment(name, backend="generalized")
    spec = EnvSpec(env.observation_size, True, env.action_size, -1.0, 1.0, episode_length)

    def reset(key):
        s = env.reset(key)
        return s.obs, (s, jnp.int32(0))

    def step(key, state, action):
        s, t = state
        s2 = env.step(s, action)
        t2 = t + 1
        term = s2.done > 0.5
        trunc = jnp.logical_and(t2 >= episode_length, jnp.logical_not(term))
        done = jnp.logical_or(term, trunc)
        sr = env.reset(key)
        new = jax.tree.map(lambda a, b: jnp.where(done, a, b), sr, s2)
        return (jnp.where(done, sr.obs, s2.obs), (new, jnp.where(done, 0, t2)), s2.reward, term, trunc, s2.obs)

    return reset, step, spec


def make_base_env(cfg: dict):
    if cfg["env_impl"] == "gymnax":
        return _gymnax_env(cfg["env_id"], cfg.get("mask_velocity", False))
    if cfg["env_impl"] == "brax":
        return _brax_env(cfg["env_id"].split("-", 1)[1])
    raise ValueError(cfg["env_impl"])


# ------------------------------------------------------------------ vectorized env + wrappers
class VecState(NamedTuple):
    env_state: Any
    obs_rms: RMS       # shared (baselines) or per-env with leading N axis (gymnasium029)
    ret_rms: RMS
    ret: jax.Array     # (N,) discounted-return accumulator (K8)
    ep_ret: jax.Array  # (N,) raw episodic return (metric)
    ep_len: jax.Array


class StepOut(NamedTuple):
    obs: jax.Array        # (N, obs) what the agent sees next (normalized if obs_norm)
    reward: jax.Array     # (N,) training reward (scaled + clipped if reward_scaling)
    done: jax.Array       # (N,) terminated OR truncated
    terminated: jax.Array
    truncated: jax.Array
    final_obs: jax.Array  # (N, obs) terminal obs, as the agent would see it (for the truncation-bootstrap flag)
    ep_ret_done: jax.Array  # (N,) raw return of the episode that just finished (0 where not done)
    ep_len_done: jax.Array


class VecEnv:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.reset_one, self.step_one, self.spec = make_base_env(cfg)
        self.N = cfg["num_envs"]
        self.mode = cfg["reward_scaling_mode"]
        assert self.mode in ("baselines", "gymnasium029", "gym021")
        self.obs_norm, self.rew_scale = cfg["obs_norm"], cfg["reward_scaling"]
        self.eps, self.oclip, self.rclip, self.gamma = cfg["norm_eps"], cfg["obs_clip"], cfg["reward_clip"], cfg["gamma"]

    def _prep_action(self, a):
        if self.spec.continuous:  # paper K5: clip for the env only; the unclipped action is stored
            return jnp.clip(a, self.spec.act_low, self.spec.act_high)
        return a

    def reset(self, key):
        obs, es = jax.vmap(self.reset_one)(jax.random.split(key, self.N))
        return self.wrap_reset(obs, es)

    def wrap_reset(self, obs, es):
        """Wrapper part of reset (the initial obs update the obs RMS, as VecNormalize.reset / gymnasium reset do)."""
        N, d = self.N, self.spec.obs_dim
        if self.mode == "baselines":
            obs_rms, ret_rms = rms_init((d,)), rms_init(())
        else:
            obs_rms = jax.vmap(lambda _: rms_init((d,)))(jnp.arange(N))
            ret_rms = jax.vmap(lambda _: rms_init(()))(jnp.arange(N))
        if self.obs_norm:
            if self.mode == "baselines":
                obs_rms = rms_update(obs_rms, obs)
                obs = rms_normalize(obs_rms, obs, self.eps, self.oclip)
            else:
                obs_rms = jax.vmap(lambda r, o: rms_update(r, o[None]))(obs_rms, obs)
                obs = jax.vmap(lambda r, o: rms_normalize(r, o, self.eps, self.oclip))(obs_rms, obs)
        z = jnp.zeros((N,), jnp.float32)
        return VecState(es, obs_rms, ret_rms, z, z, z), obs

    def step(self, key, vs: VecState, action):
        o, es, r, term, trunc, fin = jax.vmap(self.step_one)(jax.random.split(key, self.N), vs.env_state,
                                                               self._prep_action(action))
        trunc = jnp.logical_and(trunc, jnp.logical_not(term))
        done = jnp.logical_or(term, trunc)
        df = done.astype(jnp.float32)
        ep_ret, ep_len = vs.ep_ret + r, vs.ep_len + 1.0
        ep_ret_done, ep_len_done = ep_ret * df, ep_len * df
        ep_ret, ep_len = ep_ret * (1 - df), ep_len * (1 - df)
        obs_rms, ret_rms, ret, rew = vs.obs_rms, vs.ret_rms, vs.ret, r
        if self.mode == "baselines":
            if self.obs_norm:
                obs_rms = rms_update(obs_rms, o)
                fin = rms_normalize(obs_rms, fin, self.eps, self.oclip)
                o = rms_normalize(obs_rms, o, self.eps, self.oclip)
            if self.rew_scale:
                ret, ret_rms, rew = scale_reward_baselines(ret, ret_rms, r, done, self.gamma, self.eps, self.rclip)
        else:
            if self.obs_norm:
                def per_env(rm, oo, ff, d):
                    rm1 = rms_update(rm, ff[None])                      # terminal obs (== next obs when not done)
                    fn = rms_normalize(rm1, ff, self.eps, self.oclip)
                    rm2 = jax.tree.map(lambda a, b: jnp.where(d, a, b), rms_update(rm1, oo[None]), rm1)
                    on = jnp.where(d, rms_normalize(rm2, oo, self.eps, self.oclip), fn)
                    return rm2, on, fn
                obs_rms, o, fin = jax.vmap(per_env)(obs_rms, o, fin, done)
            if self.rew_scale and self.mode == "gymnasium029":
                ret, ret_rms, rew = jax.vmap(
                    lambda rt, rm, rr, tm: scale_reward_gymnasium029(rt, rm, rr, tm, self.gamma, self.eps, self.rclip)
                )(ret, ret_rms, r, term)
            elif self.rew_scale:  # gym021: baselines order, one RMS per env (gym 0.21 wrappers inside SyncVectorEnv)
                def per_env_b(rt, rm, rr, dn):
                    rt1, rm1, rw1 = scale_reward_baselines(rt[None], rm, rr[None], dn[None], self.gamma, self.eps,
                                                           self.rclip)
                    return rt1[0], rm1, rw1[0]
                ret, ret_rms, rew = jax.vmap(per_env_b)(ret, ret_rms, r, done)
        vs = VecState(es, obs_rms, ret_rms, ret, ep_ret, ep_len)
        return vs, StepOut(o, rew, done, term, trunc, fin, ep_ret_done, ep_len_done)

    def normalize_frozen(self, vs: VecState, obs):
        """Normalize with the frozen RMS (eval only; never updates; train-only stats rule)."""
        if not self.obs_norm:
            return obs
        if self.mode == "baselines":
            return rms_normalize(vs.obs_rms, obs, self.eps, self.oclip)
        rm = jax.tree.map(lambda x: x[0], vs.obs_rms)
        return rms_normalize(rm, obs, self.eps, self.oclip)

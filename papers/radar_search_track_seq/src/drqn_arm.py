"""Descriptive extra arm (NOT in the paper): the same three track-list encoders
inside the differential-tested value learner rl_jax_bench.drqn (imported, unmodified).

  Q network: [TrackEncoder(z_TL); NatureCNN(raster)] -> Dense 128 tanh -> Dense 128 tanh -> 361 joint (az, el) values
  learner: drqn.update / sequence_loss with a custom unroll_fn (feed-forward, trace_len = 1, double-Q target),
              uniform replay (drqn.TraceBuffer, flat compact observations), hard target sync, linear epsilon decay
  options of the core kept at defaults: masked act (all 361 actions legal here), n_step = 1, no SMDP discount.
Declared settings (ASSUMED, descriptive arm only; drqn.py notes the recurrent learner is step-size sensitive):
  gamma 0.99, Adam lr 1e-4, Huber delta 1.0, batch 32, buffer 50 000 transitions, 1 gradient step per env step
  (8 envs), target period 1 000 gradient steps, epsilon 1.0 -> 0.05 over the first 50 % of the run, learning starts
  after 5 000 transitions, max grad norm 10.
Lanes (arm x seed) are vmapped; arms differ only in the encoder (one process per encoder).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common as C  # noqa: E402

import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from rl_jax_bench import drqn  # noqa: E402

from src import data  # noqa: E402
from src.configs.config import Config, EnvConfig, NetConfig  # noqa: E402
from src.model import TrackEncoder, NatureCNN, raster_of, ortho, SQ2  # noqa: E402

DRQN_ARMS = {"drqn_flat": (11, "flat"), "drqn_bigru": (12, "bigru"), "drqn_mhsa": (13, "mhsa")}


@dataclass(frozen=True)
class DRQNSettings:
    gamma: float = 0.99
    lr: float = 1e-4
    huber: float = 1.0
    batch: int = 32
    buffer: int = 50_000
    n_envs: int = 8
    target_period: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_frac: float = 0.5
    learning_starts: int = 5000
    max_grad_norm: float = 10.0


def obs_dim(e: EnvConfig) -> int:
    return e.k_max * 7 + e.k_max + 4 * e.n_sv


def flatten_obs(o: data.Obs):
    return jnp.concatenate([o.tl.reshape(o.tl.shape[:-2] + (-1,)), o.tl_mask.astype(jnp.float32), o.sv_az, o.sv_el,
                            o.sv_age, o.sv_valid.astype(jnp.float32)], -1)


def unflatten_obs(v, e: EnvConfig) -> data.Obs:
    K, N = e.k_max, e.n_sv
    i = 0

    def take(n):
        nonlocal i
        x = v[..., i:i + n]
        i += n
        return x
    tl = take(K * 7).reshape(v.shape[:-1] + (K, 7))
    m = take(K) > 0.5
    az, el, age, val = take(N), take(N), take(N), take(N) > 0.5
    return data.Obs(tl, m, az, el, age, val)


class QNet(nn.Module):
    cfg: NetConfig
    ecfg: EnvConfig

    @nn.compact
    def __call__(self, obs: data.Obs):
        z = jnp.concatenate([TrackEncoder(self.cfg, name="track_enc")(obs.tl, obs.tl_mask),
                             NatureCNN(self.cfg.cnn_out, name="cnn")(raster_of(obs, self.ecfg))], -1)
        for i, h in enumerate(self.cfg.mlp):
            z = jnp.tanh(nn.Dense(h, kernel_init=ortho(SQ2), name=f"q_{i}")(z))
        return nn.Dense(self.ecfg.n_a ** 2, name="q_out")(z)


def make(cfg: Config, s: DRQNSettings, encoder: str, total_steps: int):
    ecfg = replace(cfg.env, gospa_metrics=False)
    net = QNet(replace(cfg.net, encoder=encoder), ecfg)
    D = obs_dim(ecfg)
    A = ecfg.n_a ** 2
    qcfg = drqn.QConfig(n_actions=A, obs_dim=D, gamma=s.gamma, lr=s.lr, trace_len=1, batch_size=s.batch,
                        buffer_traces=s.buffer, target_period=s.target_period, huber_delta=s.huber,
                        max_grad_norm=s.max_grad_norm, double=True)

    def unroll_fn(params, inputs, carry0, resets=None):
        return net.apply(params, unflatten_obs(inputs, ecfg)), carry0

    T = ecfg.T
    nE = s.n_envs
    n_iter = total_steps // nE

    def init(k_lane):
        params = net.init(jax.random.fold_in(k_lane, 0), unflatten_obs(jnp.zeros((1, D)), ecfg))
        st = drqn.QState(params, jax.tree.map(jnp.copy, params), drqn.make_optimizer(qcfg).init(params),
                         jnp.zeros((), jnp.int32))
        envs = jax.vmap(lambda e: data.reset(_scen(k_lane, e, 0), ecfg))(jnp.arange(nE))
        obs = jax.vmap(lambda x: data.observe(x, ecfg))(envs)
        return st, drqn.buffer_init(qcfg), envs, obs

    def step(carry, g, k_lane):
        st, buf, envs, obs = carry
        eps = drqn.linear_epsilon(g * nE, s.eps_start, s.eps_end, s.eps_frac * total_steps)
        q = net.apply(st.params, obs)
        ka = jax.random.fold_in(jax.random.fold_in(k_lane, 4), g)
        greedy = drqn.greedy(q)
        rnd = jax.random.randint(jax.random.fold_in(ka, 1), (nE,), 0, A)
        a = jnp.where(jax.random.uniform(ka, (nE,)) < eps, rnd, greedy).astype(jnp.int32)
        act2 = jnp.stack([a % ecfg.n_a, a // ecfg.n_a], -1)
        envs2, obs2, info = jax.vmap(lambda e, x: data.step(e, x, ecfg))(envs, act2)
        r = (info["r_sv"] + info["rtl_flip"]).astype(jnp.float32)
        o_flat, o2_flat = flatten_obs(obs), flatten_obs(obs2)
        # time-limit truncation: bootstrap through the final observation (discount stays gamma, Pardo et al.)
        for e in range(nE):
            buf = drqn.buffer_add(buf, dict(obs=jnp.stack([o_flat[e], o2_flat[e]]), act=a[e:e + 1],
                                            rew=r[e:e + 1], disc=jnp.full((1,), s.gamma, jnp.float32),
                                            resets=jnp.zeros((2,), bool), carry0=()))
        trunc = (g + 1) % T == 0
        ep = (g + 1) // T

        def do_reset(_):
            en = jax.vmap(lambda e: data.reset(_scen(k_lane, e, ep), ecfg))(jnp.arange(nE))
            return en, jax.vmap(lambda x: data.observe(x, ecfg))(en)
        envs3, obs3 = jax.lax.cond(trunc, do_reset, lambda _: (envs2, obs2), None)
        batch = drqn.buffer_sample(buf, jax.random.fold_in(jax.random.fold_in(k_lane, 5), g), s.batch)
        st_new, inf = drqn.update(st, batch, qcfg, unroll_fn)
        learn = buf.count * 1 >= s.learning_starts
        st = jax.tree.map(lambda n_, o_: jnp.where(learn, n_, o_), st_new, st)
        st = drqn.sync(st, drqn.sync_due(st, s.target_period) & learn)
        return (st, buf, envs3, obs3), dict(r=r.mean(), loss=inf["loss"], q=inf["q_mean"], eps=eps,
                                            cov=info["cov_sum"].mean())

    def run_chunk(carry, g0, n, k_lane):
        return jax.lax.scan(lambda c, g: step(c, g, k_lane), carry, g0 + jnp.arange(n))

    return net, init, run_chunk, n_iter


def _scen(k_lane, e, ep):
    return jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(k_lane, 3), e), ep)


def lane_key(cfg: Config, arm: str, seed: int):
    return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(cfg.seed_root), DRQN_ARMS[arm][0]), seed)


def make_policy(net, params):
    """Greedy policy for eval.rollout_eval."""
    def pinit(key):
        return jnp.int32(0)

    def policy(p, s, obs, key):
        a = drqn.greedy(net.apply(params, obs))
        n = net.ecfg.n_a
        return jnp.stack([a % n, a // n]).astype(jnp.int32), p
    return pinit, policy

"""Per-lane streaming agents: one env step + one update per call, pure and vmap-safe (Alg. 7-10, official scripts).

All arms of an experiment share ONE step function; they differ only through a per-lane config pytree of DYNAMIC
values (``QLane``, ``ACLane``, ``TDLane``), so arms x seeds run in one vmapped ``lax.scan`` program and an arm
differs from another only in the switched mechanism (LayerNorm, init, obs/reward normalization, optimizer,
lambda, target network).

Per-step PRNG (key tree in src/train.py): k_t = fold_in(step_key, t) -> split(5) = (eps coin | Gaussian noise, random action,
sticky coin, env step, env reset). Seeds pair across arms (common random numbers).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from . import algorithm as alg
from . import model as mdl
from .data import Env, sticky

OPT_OBGD, OPT_ADAM = 0, 1


# ======================================================================================= stream Q(lambda) family
class QLane(NamedTuple):
    use_ln: Any          # 1 = LayerNorm before every hidden activation
    init: Any            # mdl.INIT_SPARSE / INIT_DENSE0 / INIT_TORCH
    obs_norm: Any        # 1 = NormalizeObservation
    rew_scale: Any       # 1 = ScaleReward
    opt: Any             # OPT_OBGD / OPT_ADAM
    lr: Any              # ObGD alpha (1.0) or Adam lr
    kappa: Any
    lam: Any
    eps_end: Any
    eps_frac: Any        # exploration fraction of total steps
    use_target: Any      # 1 = DQN target network (DQN1 arm)
    target_period: Any
    loss_mult: Any       # pseudo-gradient multiplier: 1 (TD semi-gradient) or 2 (MSE loss, DQN1)
    watkins_paper: Any   # 0 = official cut (after the update), 1 = paper Alg. 8 (before accumulating)
    adam_eps: Any        # Adam epsilon: 1e-4 classic arms (App. F.2); 1e-8 DQN1 (CleanRL = torch default)


class QDiag(NamedTuple):
    n: jax.Array
    n_active: jax.Array      # steps with M > 1 (bound active)
    sum_step: jax.Array      # sum alpha_eff (ObGD lanes)
    max_viol: jax.Array      # max(||dw||_1 - min(|delta|,1)/kappa) -- must be <= ~1e-6 (P1)
    n_overshoot: jax.Array   # sign(delta_after * delta) == -1 on the same transition
    sum_dw: jax.Array
    sum_absdelta: jax.Array


def qdiag_zero():
    z = jnp.float32(0.0)
    return QDiag(z, z, z, jnp.float32(-jnp.inf), z, z, z)


class QCarry(NamedTuple):
    params: Any
    trace: Any
    adam: alg.AdamState
    target: Any
    obs_w: alg.Welford
    rew_w: alg.Welford
    u: jax.Array
    env_state: Any
    obs: jax.Array       # current (normalized if obs_norm) observation
    last_exec: jax.Array
    ep_ret: jax.Array
    ep_len: jax.Array
    diag: QDiag


class QAgent:
    def __init__(self, env: Env, total_steps: int, gamma: float = 0.99, eps_start: float = 1.0,
                 overshoot_diag: bool = True, hidden: int = 128):
        self.env, self.T, self.gamma, self.eps_start = env, total_steps, gamma, eps_start
        self.net = mdl.QNetMinAtar(env.n_actions, hidden)
        self.tmpl = mdl.template(self.net, env.obs_shape)
        self.overshoot_diag = overshoot_diag

    def q(self, params, obs, use_ln):
        return self.net.apply({"params": params}, obs, use_ln)

    def _obs_in(self, lane, w, obs_raw):
        w2, on = alg.normalize_obs(w, obs_raw)
        return alg.welford_select(lane.obs_norm > 0.5, w2, w), jnp.where(lane.obs_norm > 0.5, on, obs_raw)

    def init(self, run_key, lane: QLane) -> QCarry:
        init_key = jax.random.fold_in(run_key, 0)
        env_key = jax.random.fold_in(run_key, 1)
        params = mdl.init_select(jax.random.fold_in(init_key, 0), self.tmpl, lane.init)
        st, obs = self.env.reset(jax.random.fold_in(env_key, 0))
        w, on = self._obs_in(lane, alg.welford_init(self.env.obs_shape), obs)
        return QCarry(params, alg.tree_zeros(params), alg.adam_init(params), params, w, alg.welford_init(),
                      jnp.float32(0.0), st, on, jnp.int32(0), jnp.float32(0.0), jnp.int32(0), qdiag_zero())

    def learn(self, params, trace, adam, target, obs, a, r_t, obs2, done, nongreedy, lane: QLane, t):
        """One stream-Q(lambda) / classic-Q / DQN1 update on (S, A, R, S', done) (official update_params)."""
        donef = done.astype(jnp.float32)
        q_s, vjp = jax.vjp(lambda p: self.q(p, obs, lane.use_ln), params)
        boot_p = alg.tree_where(lane.use_target > 0.5, target, params)   # DQN1: max over the target network
        q_next = self.q(boot_p, obs2, lane.use_ln)
        delta = r_t + self.gamma * (1.0 - donef) * jnp.max(q_next) - q_s[a]
        (grad,) = vjp(jax.nn.one_hot(a, self.env.n_actions, dtype=q_s.dtype))
        gl = self.gamma * lane.lam
        # Watkins cut: paper Alg. 8 (before accumulating) vs official (after the update, D3)
        pre = jnp.logical_and(lane.watkins_paper > 0.5, nongreedy)
        trace = alg.tree_where(pre, alg.tree_zeros(trace), trace)
        trace = alg.trace_accumulate(trace, grad, gl)
        p_obgd, info = alg.obgd_apply(params, trace, delta, lane.lr, lane.kappa)
        g_loss = alg.tree_scale(-lane.loss_mult * delta, trace)
        p_adam, adam2, dw_adam = alg.adam_apply(params, adam, g_loss, lane.lr, eps=lane.adam_eps)
        use_adam = lane.opt == OPT_ADAM
        new_params = alg.tree_where(use_adam, p_adam, p_obgd)
        adam = alg.tree_where(use_adam, adam2, adam)
        post_cut = jnp.logical_and(lane.watkins_paper < 0.5, nongreedy)
        trace = alg.tree_where(jnp.logical_or(done, post_cut), alg.tree_zeros(trace), trace)
        target = alg.tree_where(jnp.logical_and(lane.use_target > 0.5, (t + 1) % lane.target_period == 0),
                                new_params, target)
        return new_params, trace, adam, target, delta, info, dw_adam, use_adam

    def eps(self, t, lane: QLane):
        """epsilon at 0-based global step t (official sample_action: time_step = t + 1 before the schedule)."""
        return alg.linear_schedule(self.eps_start, lane.eps_end, lane.eps_frac * self.T, (t + 1).astype(jnp.float32))

    def act(self, c: QCarry, t, lane: QLane, k):
        """epsilon-greedy action and the Watkins non-greedy flag (official sample_action)."""
        q_s = self.q(c.params, c.obs, lane.use_ln)
        greedy = jnp.argmax(q_s).astype(jnp.int32)
        explore = jax.random.uniform(k[0]) < self.eps(t, lane)
        rand_a = jax.random.randint(k[1], (), 0, self.env.n_actions, jnp.int32)
        a = jnp.where(explore, rand_a, greedy)
        return a, jnp.logical_and(explore, rand_a != greedy)

    def step(self, c: QCarry, t, lane: QLane, step_key):
        """One official main-loop iteration at 0-based global step t (official time_step = t + 1)."""
        k = jax.random.split(jax.random.fold_in(step_key, t), 5)
        a, nongreedy = self.act(c, t, lane, k)
        a_exec = sticky(k[2], c.last_exec, a)
        env_out = self.env.step(k[3], c.env_state, a_exec)
        new, out, _ = self.observe(c, t, lane, a, nongreedy, a_exec, env_out, self.env.reset(k[4]))
        return new, out

    def observe(self, c: QCarry, t, lane: QLane, a, nongreedy, a_exec, env_out, reset_out):
        """Everything after the env step: wrappers, the update, diagnostics, episode-end reset. Split from ``step``
        so the oracle closed-loop test can drive it with the AUTHORS' actions and raw env outputs. Returns
        (carry, (ep_ret | nan, ep_len | 0), delta)."""
        st2, obs2_raw, r, done = env_out
        donef = done.astype(jnp.float32)
        ep_ret, ep_len = c.ep_ret + r, c.ep_len + 1
        # wrappers (official order: NormalizeObservation inside ScaleReward; independent)
        rew_w2, u2, r_s = alg.scale_reward(c.rew_w, c.u, r, donef, self.gamma)
        rs_on = lane.rew_scale > 0.5
        rew_w = alg.welford_select(rs_on, rew_w2, c.rew_w)
        u = jnp.where(rs_on, u2, c.u)
        r_t = jnp.where(rs_on, r_s, r)
        obs_w, obs2 = self._obs_in(lane, c.obs_w, obs2_raw)
        params, trace, adam, target, delta, info, dw_adam, use_adam = self.learn(
            c.params, c.trace, c.adam, c.target, c.obs, a, r_t, obs2, done, nongreedy, lane, t)
        # diagnostics
        d = c.diag
        bound = jnp.minimum(jnp.abs(delta), 1.0) / lane.kappa
        is_obgd = jnp.logical_not(use_adam)
        dw = jnp.where(use_adam, dw_adam, info.dw_l1)
        if self.overshoot_diag:
            q_s2 = self.q(params, c.obs, lane.use_ln)
            q_n2 = self.q(alg.tree_where(lane.use_target > 0.5, target, params), obs2, lane.use_ln)
            delta_after = r_t + self.gamma * (1.0 - donef) * jnp.max(q_n2) - q_s2[a]
            ovs = (jnp.sign(delta_after * delta) == -1).astype(jnp.float32)
        else:
            ovs = jnp.float32(0.0)
        diag = QDiag(d.n + 1, d.n_active + jnp.where(is_obgd, (info.M > 1).astype(jnp.float32), 0.0),
                     d.sum_step + jnp.where(is_obgd, info.step, 0.0),
                     jnp.maximum(d.max_viol, jnp.where(is_obgd, info.dw_l1 - bound, -jnp.inf)),
                     d.n_overshoot + ovs, d.sum_dw + dw, d.sum_absdelta + jnp.abs(delta))
        # episode end: reset env, normalize the reset obs too (D10)
        st_r, obs_r_raw = reset_out
        obs_w_r, obs_r = self._obs_in(lane, obs_w, obs_r_raw)
        env_state = alg.tree_where(done, st_r, st2)
        obs_w = alg.welford_select(done, obs_w_r, obs_w)
        obs_next = jnp.where(done, obs_r, obs2)
        out = (jnp.where(done, ep_ret, jnp.nan), jnp.where(done, ep_len, 0))
        new = QCarry(params, trace, adam, target, obs_w, rew_w, u, env_state, obs_next, a_exec,
                     jnp.where(done, 0.0, ep_ret), jnp.where(done, 0, ep_len), diag)
        return new, out, delta


# ======================================================================================= stream AC(lambda)
class ACLane(NamedTuple):
    use_ln: Any
    init: Any
    norm: Any            # 1 = NormalizeObservation + ScaleReward (ablated jointly)
    opt: Any
    lr: Any              # ObGD alpha or Adam lr (both nets)
    kappa_pi: Any
    kappa_v: Any
    lam: Any
    tau: Any             # entropy coefficient
    adam_eps: Any        # Adam epsilon (App. F.4: 1e-4)


class ACDiag(NamedTuple):
    n: jax.Array
    n_active_v: jax.Array
    n_active_pi: jax.Array
    max_viol: jax.Array
    n_overshoot: jax.Array
    sum_absdelta: jax.Array
    sum_std: jax.Array


def acdiag_zero():
    z = jnp.float32(0.0)
    return ACDiag(z, z, z, jnp.float32(-jnp.inf), z, z, z)


class ACCarry(NamedTuple):
    pi: Any
    v: Any
    z_pi: Any
    z_v: Any
    adam_pi: alg.AdamState
    adam_v: alg.AdamState
    obs_w: alg.Welford
    rew_w: alg.Welford
    u: jax.Array
    env_state: Any
    obs: jax.Array       # agent input: normalized obs ++ time feature
    epi_time: jax.Array
    ep_ret: jax.Array
    ep_len: jax.Array
    diag: ACDiag


class ACAgent:
    def __init__(self, env: Env, gamma: float = 0.99, episode_limit: int = 1000, overshoot_diag: bool = True,
                 hidden: int = 128):
        self.env, self.gamma, self.Tlim = env, gamma, episode_limit
        n_in = (env.obs_shape[0] + 1,)  # + time feature (AddTimeInfo)
        self.actor = mdl.GaussianActor(env.n_actions, hidden)
        self.critic = mdl.ValueNet(hidden)
        self.t_pi = mdl.template(self.actor, n_in)
        self.t_v = mdl.template(self.critic, n_in)
        self.overshoot_diag = overshoot_diag

    def pi(self, p, x, use_ln):
        return self.actor.apply({"params": p}, x, use_ln)

    def v(self, p, x, use_ln):
        return self.critic.apply({"params": p}, x, use_ln)

    def _obs_in(self, lane, w, obs_raw, time_feat):
        w2, on = alg.normalize_obs(w, obs_raw)
        use = lane.norm > 0.5
        return alg.welford_select(use, w2, w), jnp.concatenate([jnp.where(use, on, obs_raw), time_feat[None]])

    def init(self, run_key, lane: ACLane) -> ACCarry:
        init_key = jax.random.fold_in(run_key, 0)
        env_key = jax.random.fold_in(run_key, 1)
        p_pi = mdl.init_select(jax.random.fold_in(init_key, 0), self.t_pi, lane.init)
        p_v = mdl.init_select(jax.random.fold_in(init_key, 1), self.t_v, lane.init)
        st, obs = self.env.reset(jax.random.fold_in(env_key, 0))
        t0 = jnp.float32(-0.5)
        w, on = self._obs_in(lane, alg.welford_init(self.env.obs_shape), obs, t0)
        return ACCarry(p_pi, p_v, alg.tree_zeros(p_pi), alg.tree_zeros(p_v), alg.adam_init(p_pi), alg.adam_init(p_v),
                       w, alg.welford_init(), jnp.float32(0.0), st, on, t0, jnp.float32(0.0), jnp.int32(0),
                       acdiag_zero())

    def learn(self, pi, v, z_pi, z_v, adam_pi, adam_v, obs, a, r_t, obs2, done, lane: ACLane):
        """One stream-AC(lambda) update (official StreamAC.update_params); truncation bootstrapped as terminal (D12)."""
        donef = done.astype(jnp.float32)
        v_s, vjp_v = jax.vjp(lambda p: self.v(p, obs, lane.use_ln), v)
        v_n = self.v(v, obs2, lane.use_ln)
        delta = r_t + self.gamma * (1.0 - donef) * v_n - v_s
        (g_v,) = vjp_v(jnp.ones_like(v_s))
        sgn = jnp.sign(delta)

        def pi_obj(p):
            m, s = self.pi(p, obs, lane.use_ln)
            return alg.gaussian_logp(a, m, s) + lane.tau * sgn * alg.gaussian_entropy(s)

        g_pi = jax.grad(pi_obj)(pi)
        gl = self.gamma * lane.lam
        z_pi = alg.trace_accumulate(z_pi, g_pi, gl)
        z_v = alg.trace_accumulate(z_v, g_v, gl)
        pi_o, i_pi = alg.obgd_apply(pi, z_pi, delta, lane.lr, lane.kappa_pi)
        v_o, i_v = alg.obgd_apply(v, z_v, delta, lane.lr, lane.kappa_v)
        pi_a, adam_pi2, _ = alg.adam_apply(pi, adam_pi, alg.tree_scale(-delta, z_pi), lane.lr, eps=lane.adam_eps)
        v_a, adam_v2, _ = alg.adam_apply(v, adam_v, alg.tree_scale(-delta, z_v), lane.lr, eps=lane.adam_eps)
        use_adam = lane.opt == OPT_ADAM
        p_pi = alg.tree_where(use_adam, pi_a, pi_o)
        p_v = alg.tree_where(use_adam, v_a, v_o)
        adam_pi = alg.tree_where(use_adam, adam_pi2, adam_pi)
        adam_v = alg.tree_where(use_adam, adam_v2, adam_v)
        z_pi = alg.tree_where(done, alg.tree_zeros(z_pi), z_pi)
        z_v = alg.tree_where(done, alg.tree_zeros(z_v), z_v)
        return p_pi, p_v, z_pi, z_v, adam_pi, adam_v, delta, i_pi, i_v, use_adam

    def step(self, c: ACCarry, t, lane: ACLane, step_key):
        k = jax.random.split(jax.random.fold_in(step_key, t), 5)
        mu, std = self.pi(c.pi, c.obs, lane.use_ln)
        a = mu + std * jax.random.normal(k[0], mu.shape)
        env_out = self.env.step(k[3], c.env_state, jnp.clip(a, -1.0, 1.0))
        new, out, _ = self.observe(c, lane, a, env_out, self.env.reset(k[4]))
        return new, out

    def observe(self, c: ACCarry, lane: ACLane, a, env_out, reset_out):
        """Everything after the env step (wrappers, update, diagnostics, reset); split from ``step`` for the oracle
        closed-loop test. ``a`` is the UNCLIPPED sample (the update's log-prob uses it; the env got clip(a)).
        Returns (carry, (ep_ret | nan, ep_len | 0), delta)."""
        st2, obs2_raw, r, term = env_out
        mu, std = self.pi(c.pi, c.obs, lane.use_ln)
        ep_ret, ep_len = c.ep_ret + r, c.ep_len + 1
        trunc = jnp.logical_and(ep_len >= self.Tlim, jnp.logical_not(term))
        done = jnp.logical_or(term, trunc)
        donef = done.astype(jnp.float32)
        rew_w2, u2, r_s = alg.scale_reward(c.rew_w, c.u, r, donef, self.gamma)
        nm = lane.norm > 0.5
        rew_w = alg.welford_select(nm, rew_w2, c.rew_w)
        u = jnp.where(nm, u2, c.u)
        r_t = jnp.where(nm, r_s, r)
        obs_w, obs2 = self._obs_in(lane, c.obs_w, obs2_raw, c.epi_time)
        epi_time = -0.5 + ep_len.astype(jnp.float32) / self.Tlim   # official float64 running sum, without float32 drift
        p_pi, p_v, z_pi, z_v, adam_pi, adam_v, delta, i_pi, i_v, use_adam = self.learn(
            c.pi, c.v, c.z_pi, c.z_v, c.adam_pi, c.adam_v, c.obs, a, r_t, obs2, done, lane)
        d = c.diag
        is_obgd = jnp.logical_not(use_adam)
        viol = jnp.maximum(i_v.dw_l1 - jnp.minimum(jnp.abs(delta), 1.0) / lane.kappa_v,
                           i_pi.dw_l1 - jnp.minimum(jnp.abs(delta), 1.0) / lane.kappa_pi)
        if self.overshoot_diag:
            da = r_t + self.gamma * (1.0 - donef) * self.v(p_v, obs2, lane.use_ln) - self.v(p_v, c.obs, lane.use_ln)
            ovs = (jnp.sign(da * delta) == -1).astype(jnp.float32)
        else:
            ovs = jnp.float32(0.0)
        diag = ACDiag(d.n + 1, d.n_active_v + jnp.where(is_obgd, (i_v.M > 1).astype(jnp.float32), 0.0),
                      d.n_active_pi + jnp.where(is_obgd, (i_pi.M > 1).astype(jnp.float32), 0.0),
                      jnp.maximum(d.max_viol, jnp.where(is_obgd, viol, -jnp.inf)), d.n_overshoot + ovs,
                      d.sum_absdelta + jnp.abs(delta), d.sum_std + jnp.mean(std))
        # reset (AddTimeInfo.reset: time -0.5; NormalizeObservation sees the reset obs)
        st_r, obs_r_raw = reset_out
        t0 = jnp.float32(-0.5)
        obs_w_r, obs_r = self._obs_in(lane, obs_w, obs_r_raw, t0)
        out = (jnp.where(done, ep_ret, jnp.nan), jnp.where(done, ep_len, 0))
        new = ACCarry(p_pi, p_v, z_pi, z_v, adam_pi, adam_v, alg.welford_select(done, obs_w_r, obs_w), rew_w, u,
                      alg.tree_where(done, st_r, st2), jnp.where(done, obs_r, obs2), jnp.where(done, t0, epi_time),
                      jnp.where(done, 0.0, ep_ret), jnp.where(done, 0, ep_len), diag)
        return new, out, delta


# ======================================================================================= stream TD(lambda) (E8)
class TDLane(NamedTuple):
    use_ln: Any
    init: Any
    norm: Any            # obs norm + reward scaling
    opt: Any
    lr: Any
    kappa: Any
    lam: Any
    adam_eps: Any        # Adam epsilon (App. F.1: 1e-4)


class TDCarry(NamedTuple):
    params: Any
    trace: Any
    adam: alg.AdamState
    obs_w: alg.Welford
    rew_w: alg.Welford
    u: jax.Array
    obs: jax.Array
    n_active: jax.Array
    n_overshoot: jax.Array


class TDAgent:
    """Official stream_td.py loop on a fixed stream: transition k (1-based) is obs[k-1] -> obs[k] with cumulant
    c[k+1] (ETTEnvironment._calculate_reward); obs = bias-corrected EMA memory traces (precomputed, float64)."""

    def __init__(self, n_obs: int, gamma: float = 0.99, hidden: int = 128):
        self.gamma = gamma
        self.net = mdl.ValueNet(hidden)
        self.tmpl = mdl.template(self.net, (n_obs,))

    def v(self, p, x, use_ln):
        return self.net.apply({"params": p}, x, use_ln)

    def _obs_in(self, lane, w, x):
        w2, on = alg.normalize_obs(w, x)
        use = lane.norm > 0.5
        return alg.welford_select(use, w2, w), jnp.where(use, on, x)

    def init(self, run_key, lane: TDLane, obs0) -> TDCarry:
        p = mdl.init_select(jax.random.fold_in(jax.random.fold_in(run_key, 0), 0), self.tmpl, lane.init)
        w, on = self._obs_in(lane, alg.welford_init(obs0.shape), obs0)
        z = jnp.float32(0.0)
        return TDCarry(p, alg.tree_zeros(p), alg.adam_init(p), w, alg.welford_init(), z, on, z, z)

    def step(self, c: TDCarry, x_next, c_raw, done, lane: TDLane):
        donef = done.astype(jnp.float32)
        obs_w, s2 = self._obs_in(lane, c.obs_w, x_next)
        rew_w2, u2, r_s = alg.scale_reward(c.rew_w, c.u, c_raw, donef, self.gamma)
        nm = lane.norm > 0.5
        rew_w = alg.welford_select(nm, rew_w2, c.rew_w)
        u = jnp.where(nm, u2, c.u)
        r_t = jnp.where(nm, r_s, c_raw)
        v_s, vjp = jax.vjp(lambda p: self.v(p, c.obs, lane.use_ln), c.params)
        delta = r_t + self.gamma * (1.0 - donef) * self.v(c.params, s2, lane.use_ln) - v_s
        (g,) = vjp(jnp.ones_like(v_s))
        trace = alg.trace_accumulate(c.trace, g, self.gamma * lane.lam)
        p_o, info = alg.obgd_apply(c.params, trace, delta, lane.lr, lane.kappa)
        p_a, adam2, _ = alg.adam_apply(c.params, c.adam, alg.tree_scale(-delta, trace), lane.lr, eps=lane.adam_eps)
        use_adam = lane.opt == OPT_ADAM
        params = alg.tree_where(use_adam, p_a, p_o)
        adam = alg.tree_where(use_adam, adam2, c.adam)
        trace = alg.tree_where(done, alg.tree_zeros(trace), trace)
        da = r_t + self.gamma * (1.0 - donef) * self.v(params, s2, lane.use_ln) - self.v(params, c.obs, lane.use_ln)
        sigma = jnp.where(nm, jnp.sqrt(alg.welford_var(rew_w) + alg.REW_EPS), 1.0)
        pred = self.v(params, s2, lane.use_ln) * sigma   # official: predict(s') after the update, rescaled
        cum = r_t * sigma
        new = TDCarry(params, trace, adam, obs_w, rew_w, u, s2,
                      c.n_active + jnp.where(use_adam, 0.0, (info.M > 1).astype(jnp.float32)),
                      c.n_overshoot + (jnp.sign(da * delta) == -1).astype(jnp.float32))
        return new, (pred, cum)

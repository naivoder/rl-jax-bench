"""Exact (closed-form / dynamic-programming) solutions of the paper's toy tasks -- the toy-stage ground truth
Pure JAX; call under `jax.experimental.enable_x64()` with float64 inputs for 1e-6 checks.

Conventions: finite-horizon arrays are indexed by REMAINING steps k = T - t (k = 0 is the zero terminal layer);
policy tables for time-aware policies are indexed by ELAPSED step t = 0..T-1.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .algorithm import gae_peb


# ===================================================================== generic deterministic tabular MDPs
def _boot(tab):
    return 1.0 - jnp.asarray(tab.is_terminal, jnp.float64)[jnp.asarray(tab.NX)]


def finite_horizon_q(tab, gamma, T: int):
    """q*(s, a, k) for k = 0..T remaining steps : Q[k] = R + gamma (1 - D') max Q[k-1](s')."""
    NX, R, boot = jnp.asarray(tab.NX), jnp.asarray(tab.R, jnp.float64), _boot(tab)
    term = jnp.asarray(tab.is_terminal)
    Qs = [jnp.zeros_like(R)]
    for _ in range(T):
        V = jnp.where(term, 0.0, Qs[-1].max(1))
        Q = R + gamma * boot * V[NX]
        Qs.append(jnp.where(term[:, None], 0.0, Q))
    return jnp.stack(Qs)


def infinite_q(tab, gamma, iters: int = 20000, pb=None):
    """Fixed point of Q = R + gamma (1 - D') pb(s) max Q(s'). pb = None -> 1 (the infinite-horizon / PEB target);
    pb = p_b(s) -> the time-unaware standard agent's fixed point ."""
    NX, R, boot = jnp.asarray(tab.NX), jnp.asarray(tab.R, jnp.float64), _boot(tab)
    term = jnp.asarray(tab.is_terminal)
    pbm = jnp.ones(R.shape[0], jnp.float64) if pb is None else jnp.asarray(pb, jnp.float64)

    def body(_, Q):
        V = jnp.where(term, 0.0, Q.max(1))
        Qn = R + gamma * boot * pbm[:, None] * V[NX]
        return jnp.where(term[:, None], 0.0, Qn)

    return jax.lax.fori_loop(0, iters, body, jnp.zeros_like(R))


def bootstrap_fraction(tab, T: int):
    """p_b(s) = P(t < T-1 | a visit of s) under uniform-random behavior from tab.start ."""
    NX = np.asarray(tab.NX)
    term = np.asarray(tab.is_terminal)
    S, A = NX.shape
    d = np.zeros((T, S))
    d[0] = np.asarray(tab.start, np.float64)
    for t in range(1, T):
        for s in range(S):
            if term[s] or d[t - 1, s] == 0:
                continue
            for a in range(A):
                j = NX[s, a]
                if not term[j]:
                    d[t, j] += d[t - 1, s] / A
    tot = d.sum(0)
    return np.where(tot > 0, d[: T - 1].sum(0) / np.maximum(tot, 1e-300), 0.0), d


def policy_value_finite(tab, gamma, pi_t):
    """V(s) at t = 0 of a time-indexed stochastic policy pi_t: (T, S, A) indexed by ELAPSED step."""
    NX, R, boot = jnp.asarray(tab.NX), jnp.asarray(tab.R, jnp.float64), _boot(tab)
    term = jnp.asarray(tab.is_terminal)
    T = pi_t.shape[0]
    V = jnp.zeros(R.shape[0], jnp.float64)
    for t in reversed(range(T)):
        Q = R + gamma * boot * V[NX]
        V = jnp.where(term, 0.0, (pi_t[t] * Q).sum(1))
    return V


def policy_value_infinite(tab, gamma, pi, iters: int = 20000):
    """Infinite-horizon V of a stationary stochastic policy pi: (S, A) (gamma < 1, or proper policies)."""
    NX, R, boot = jnp.asarray(tab.NX), jnp.asarray(tab.R, jnp.float64), _boot(tab)
    term = jnp.asarray(tab.is_terminal)

    def body(_, V):
        Q = R + gamma * boot * V[NX]
        return jnp.where(term, 0.0, (pi * Q).sum(1))

    return jax.lax.fori_loop(0, iters, body, jnp.zeros(R.shape[0], jnp.float64))


def greedy_onehot(Q, first=True):
    """Deterministic first-argmax policy (Q: (..., A)) or uniform over ties (first=False)."""
    if first:
        return jax.nn.one_hot(jnp.argmax(Q, -1), Q.shape[-1], dtype=jnp.float64)
    m = (Q >= Q.max(-1, keepdims=True) - 1e-9).astype(jnp.float64)
    return m / m.sum(-1, keepdims=True)


def eps_greedy(Q, eps, first=True):
    A = Q.shape[-1]
    return (1.0 - eps) * greedy_onehot(Q, first) + eps / A


# ===================================================================== Last Moment 
def last_moment_J(p, T: int, gamma):
    """Exact return from A at t = 0 of the memoryless policy 'jump w.p. p' ."""
    tot = 0.0
    for j in range(T):
        tail = sum(gamma ** i for i in range(1, T - j))
        tot = tot + (1.0 - p) ** j * p * gamma ** j * (1.0 - tail)
    return tot


def last_moment_best_memoryless(T: int, gamma, n_grid: int = 200001):
    p = jnp.linspace(0.0, 1.0, n_grid, dtype=jnp.float64)
    J = last_moment_J(p, T, gamma)
    i = jnp.argmax(J)
    return p[i], J[i]


# ===================================================================== Queue of Cars 
def qoc_policy_value(pd, outcomes, gamma, n_slots: int = 9, exit_reward: float = 1.0):
    """V(slot, t = 0) of a policy with P(dangerous | slot, elapsed t) = pd: (T, n_slots). Exact backward DP.
    outcomes: (2, 3) probabilities of (advance, crash, stay) for safe / dangerous (src/data.qoc_outcomes)."""
    T = pd.shape[0]
    out = jnp.asarray(outcomes, pd.dtype)
    V = jnp.zeros(n_slots + 1, pd.dtype)  # V[n_slots] = exit (terminal, value 0)
    x = jnp.arange(n_slots)
    for t in reversed(range(T)):
        adv_val = jnp.where(x + 1 >= n_slots, exit_reward, gamma * V[jnp.minimum(x + 1, n_slots)])
        stay_val = gamma * V[x]
        q = out[:, 0:1] * adv_val[None] + out[:, 2:3] * stay_val[None]  # (2, n_slots); crash contributes 0
        Vn = (1.0 - pd[t]) * q[0] + pd[t] * q[1]
        V = jnp.concatenate([Vn, jnp.zeros(1, pd.dtype)])
    return V[:n_slots]


def qoc_optimal(outcomes, gamma, T: int, n_slots: int = 9, exit_reward: float = 1.0):
    """Time-aware optimum: (V*(0, t=0), optimal P(dangerous | slot, t) with 0.5 on exact ties)."""
    out = np.asarray(outcomes, np.float64)
    V = np.zeros(n_slots + 1)
    P = np.zeros((T, n_slots))
    for t in reversed(range(T)):
        Vn = np.zeros(n_slots + 1)
        for xx in range(n_slots):
            adv = exit_reward if xx + 1 >= n_slots else gamma * V[xx + 1]
            q = out[:, 0] * adv + out[:, 2] * gamma * V[xx]
            Vn[xx] = q.max()
            P[t, xx] = 1.0 if q[1] > q[0] + 1e-12 else (0.5 if abs(q[1] - q[0]) <= 1e-12 else 0.0)
        V = Vn
    return V[0], P


def qoc_best_memoryless(outcomes, gamma, T: int, n_slots: int = 9, n_restarts: int = 256, steps: int = 3000,
                        seed: int = 0):
    """max over P(dangerous | slot) in [0,1]^9 (time-unaware) of the exact return, by projected-free gradient
    ascent on logits (exact gradients by autodiff of qoc_policy_value) from many random + all-512-deterministic
    starts. Returns (best value, best p)."""
    import optax

    def J(logits):
        pd = jnp.broadcast_to(jax.nn.sigmoid(logits), (T, n_slots))
        return qoc_policy_value(pd, outcomes, gamma, n_slots)[0]

    det = np.array([[(m >> i) & 1 for i in range(n_slots)] for m in range(2 ** n_slots)], np.float64) * 8.0 - 4.0
    rnd = np.random.default_rng(seed).normal(0.0, 2.0, (n_restarts, n_slots))
    L0 = jnp.asarray(np.concatenate([det, rnd]), jnp.float64)
    opt = optax.adam(0.05)

    def run(l0):
        st = opt.init(l0)

        def body(c, _):
            l, st = c
            g = jax.grad(lambda z: -J(z))(l)
            u, st = opt.update(g, st)
            return (optax.apply_updates(l, u), st), None

        (l, _), _ = jax.lax.scan(body, (l0, st), None, length=steps)
        return J(l), jax.nn.sigmoid(l)

    vals, ps = jax.jit(jax.vmap(run))(L0)
    i = int(jnp.argmax(vals))
    return float(vals[i]), np.asarray(ps[i])


# ===================================================================== constant-reward aliased chain 
def chain_fixed_point(gamma, lam, T: int, peb: bool, iters: int = 5000):
    """The time-unaware critic's fixed point on a constant-reward chain (r = 1, every episode lasts T steps):
    v = mean_t G^lambda_t(v), with G^lambda computed by OUR gae_peb on one episode (resets at the last step only).
     closed form: v = mean(a)/(1 - mean(b)); with PEB v = 1/(1-gamma)."""
    r = jnp.ones(T, jnp.float64)
    resets = jnp.zeros(T, jnp.float64).at[-1].set(1.0)
    terms = jnp.zeros(T, jnp.float64)

    def G_mean(v):
        vals = jnp.full(T, v)
        _, ret = gae_peb(r, vals, vals, terms, resets, gamma, lam, jnp.float64(1.0 if peb else 0.0))
        return ret.mean()

    # G_mean is affine in v: v = a + b v  =>  v = a / (1 - b)  (exact, avoids slow iteration at gamma = 1)
    a = G_mean(jnp.float64(0.0))
    b = G_mean(jnp.float64(1.0)) - a
    return a / (1.0 - b)


def chain_fixed_point_closed_form(gamma, lam, T: int):
    a = np.zeros(T)
    b = np.zeros(T)
    a[0] = 1.0
    for k in range(1, T):
        a[k] = 1 + gamma * lam * a[k - 1]
        b[k] = gamma * (1 - lam) + gamma * lam * b[k - 1]
    return a.mean() / (1 - b.mean())


# ===================================================================== Difficult Gridworld 
def dgrid_eps_greedy_return(tab, Q, eps, T: int, start: int, first=True):
    """Exact expected (undiscounted) return of the ε-greedy policy w.r.t. Q from `start`, truncated at T."""
    pi = eps_greedy(jnp.asarray(Q, jnp.float64), eps, first)
    return policy_value_finite(tab, 1.0, jnp.broadcast_to(pi, (T,) + pi.shape))[start]

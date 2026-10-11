"""Exact oracles: dynamic-programming values (the learning-curve metric, A-VALUE), and brute-force trajectory
enumeration giving the exact hindsight distributions (h_k, h_occ, the learner's pair fixed point, h_z) and
exact expected update directions. Used by the exact checks and by the metric of every learning curve.

The enumeration side is plain numpy and written independently of ``algorithm.py`` (it re-derives the
segment/pair logic), so agreement between the two is a real check, not a tautology.
"""
from __future__ import annotations

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

from .data import Episode, TabEnv
from .model import policy


# ---------------------------------------------------------------- DP (jax; differentiable in theta)
def dp_values(ea: dict, theta: jnp.ndarray, L: int, gamma: float = 1.0) -> jnp.ndarray:
    """V^pi(s) for every true state; pi acts on observations. Acyclic envs: L sweeps are exact."""
    pi_s = policy(theta)[ea["obs"]]
    V = jnp.zeros(ea["r_mean"].shape)
    for _ in range(L):
        V = ea["r_mean"] + gamma * (~ea["terminal"]) * jnp.einsum("sa,sat,t->s", pi_s, ea["P"], V)
    return V


def dp_q(ea: dict, theta: jnp.ndarray, L: int, gamma: float = 1.0) -> jnp.ndarray:
    V = dp_values(ea, theta, L, gamma)
    return ea["r_mean"][:, None] + gamma * jnp.einsum("sat,t->sa", ea["P"], V)


def start_value(ea: dict, theta, s0: int, L: int) -> jnp.ndarray:
    return dp_values(ea, theta, L)[s0]


# ---------------------------------------------------------------- enumeration (numpy)
def enumerate_trajectories(env: TabEnv, pi_obs: np.ndarray):
    """All trajectories from s0 with probability > 0 under pi (acyclic env), mean rewards.
    Returns list of (prob, states, actions) with actions[-1] = None for the terminal step."""
    out = []

    def rec(s, p, xs, acts):
        xs = xs + [s]
        if env.terminal[s]:
            out.append((p, xs, acts + [None]))
            return
        for a in range(env.P.shape[1]):
            pa = pi_obs[env.obs[s], a]
            if pa == 0:
                continue
            for s2 in np.nonzero(env.P[s, a] > 0)[0]:
                rec(int(s2), p * pa * env.P[s, a, s2], xs, acts + [a])

    rec(env.s0, 1.0, [], [])
    assert abs(sum(t[0] for t in out) - 1.0) < 1e-12
    return out


def trajectories_to_batch(env: TabEnv, trajs) -> tuple[np.ndarray, Episode]:
    """Pad enumerated trajectories into a numpy Episode batch (mean rewards)."""
    n, L = len(trajs), env.L
    st = np.zeros((n, L), np.int64)
    ob = np.zeros((n, L), np.int64)
    ac = np.zeros((n, L), np.int64)
    rw = np.zeros((n, L))
    va = np.zeros((n, L), bool)
    de = np.zeros((n, L), bool)
    for k, (_, xs, acts) in enumerate(trajs):
        for t, (s, a) in enumerate(zip(xs, acts)):
            st[k, t], ob[k, t], rw[k, t], va[k, t] = s, env.obs[s], env.r_mean[s], True
            if a is not None:
                ac[k, t], de[k, t] = a, True
    probs = np.array([t[0] for t in trajs])
    return probs, Episode(st, ob, ac, rw, va, de)


def np_pairs(xs, acts, seg_len, L, seg_mode="segment"):
    """Independent re-derivation of the hindsight training pairs (i, j) for one trajectory (A-SEG):
    MC -> all j >= i; segments of seg_len decision steps -> j in [i, end) U {end} (end = bootstrap step or the
    terminal step appended at the boundary)."""
    T = len(xs)
    pairs = []
    for i in range(T):
        if acts[i] is None:
            continue
        if seg_len is None:
            js = range(i, T)
        else:
            end = (i // seg_len + 1) * seg_len if seg_mode == "segment" else i + seg_len
            js = [j for j in range(i, min(end + 1, T))]
        pairs += [(i, j) for j in js]
    return pairs


def exact_pair_hindsight(env: TabEnv, pi_obs: np.ndarray, seg_len=None, seg_mode="segment"):
    """Fixed point of the cross-entropy pair training: h*(a|ox,oy) = E[#pairs (i,j) with o_i=ox, o_j=oy,
    A_i=a] / E[#pairs with o_i=ox, o_j=oy]. Returns dict (ox, oy) -> [A] and the visit weights."""
    num = defaultdict(lambda: np.zeros(env.P.shape[1]))
    den = defaultdict(float)
    for p, xs, acts in enumerate_trajectories(env, pi_obs):
        for i, j in np_pairs(xs, acts, seg_len, env.L, seg_mode):
            key = (int(env.obs[xs[i]]), int(env.obs[xs[j]]))
            num[key][acts[i]] += p
            den[key] += p
    return {k: num[k] / den[k] for k in den}, dict(den)


def exact_hk(env: TabEnv, pi_obs: np.ndarray):
    """Per-depth hindsight from s0: h_k(a|s0, y) = P(A_0=a | X_k=y) (Lemma 1); dict (k, oy) -> [A]."""
    num = defaultdict(lambda: np.zeros(env.P.shape[1]))
    den = defaultdict(float)
    for p, xs, acts in enumerate_trajectories(env, pi_obs):
        for k in range(1, len(xs)):
            key = (k, int(env.obs[xs[k]]))
            num[key][acts[0]] += p
            den[key] += p
    return {k: num[k] / den[k] for k in den}


def exact_return_hindsight(env: TabEnv, pi_obs: np.ndarray, centres: np.ndarray, gamma: float = 1.0):
    """Noise-free envs: h_z*(a|ox, bin) over all sources i (the Alg. 2 training fixed point) and the expected
    number of training samples per episode for each (ox, bin)."""
    num = defaultdict(lambda: np.zeros(env.P.shape[1]))
    den = defaultdict(float)
    for p, xs, acts in enumerate_trajectories(env, pi_obs):
        r = env.r_mean[xs]
        for i in range(len(xs)):
            if acts[i] is None:
                continue
            Z = float(np.sum(r[i:] * gamma ** np.arange(len(xs) - i)))
            b = int(np.argmin(np.abs(centres - Z)))
            key = (int(env.obs[xs[i]]), b)
            num[key][acts[i]] += p
            den[key] += p
    return {k: num[k] / den[k] for k in den}, dict(den)


def gather_pair_h(batch: Episode, h_star: dict, n_actions: int) -> np.ndarray:
    """[n, L, L, A] exact pair hindsight for every (i, j) of every trajectory (uniform where undefined)."""
    n, L = batch.obs.shape
    out = np.full((n, L, L, n_actions), 1.0 / n_actions)
    for k in range(n):
        for i in range(L):
            for j in range(L):
                key = (int(batch.obs[k, i]), int(batch.obs[k, j]))
                if key in h_star:
                    out[k, i, j] = h_star[key]
    return out


def gather_return_h(batch: Episode, hz_star: dict, centres: np.ndarray, n_actions: int, gamma=1.0) -> np.ndarray:
    n, L = batch.obs.shape
    out = np.full((n, L, n_actions), 1.0 / n_actions)
    for k in range(n):
        for i in range(L):
            if not batch.decision[k, i]:
                continue
            Z = float(np.sum(batch.reward[k, i:] * batch.valid[k, i:] * gamma ** np.arange(L - i)))
            key = (int(batch.obs[k, i]), int(np.argmin(np.abs(centres - Z))))
            if key in hz_star:
                out[k, i] = hz_star[key]
    return out


def to_jnp_episode(ep: Episode) -> Episode:
    return Episode(*(jnp.asarray(x) for x in ep))


def true_policy_gradient(ea: dict, theta: jnp.ndarray, s0: int, L: int) -> jnp.ndarray:
    """d V^pi(s0) / d theta by autodiff of the DP value."""
    return jax.grad(lambda th: start_value(ea, th, s0, L))(theta)

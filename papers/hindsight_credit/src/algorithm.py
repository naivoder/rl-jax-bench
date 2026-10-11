"""Per-episode update directions for State HCA, Return HCA and the n-step advantage actor-critic ("Policy
Gradient") baseline, plus the Fig. 3-right counterfactual estimators. Pure jnp functions of one padded
``Episode``; ``jax.vmap`` over runs happens in ``train.py``.

Every function returns *ascent directions* (the official snippet's raw gradients ``dlogits``, ``dV``,
``dlogits_h``), summed over the episode with the pre-update tables (official code, A-UPDATE). ``train.apply_update``
multiplies them by the learning rates. The arms differ only in how ``dtheta`` (the credit) is formed; the
episode sampler, the schedule and the value metric are shared.

Index conventions: ``i`` = source step (a decision step), ``j`` = later step, ``end[i]`` = the step a
T-step segment bootstraps from (A-SEG). With ``seg_len=None`` (Monte Carlo) there is no bootstrap.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from .data import Episode
from .model import Params, hindsight, policy


class Windows(NamedTuple):
    win: jnp.ndarray      # [L, L] reward j counts for source i (j >= i, inside i's segment, terminal appended)
    boot: jnp.ndarray     # [L]    source i bootstraps from V(obs[end[i]])
    end: jnp.ndarray      # [L]    bootstrap step index (clipped into range)
    hpair: jnp.ndarray    # [L, L] (i, j) is a hindsight training pair (Alg. 1: i <= j <= T; official j >= i)
    disc: jnp.ndarray     # [L, L] gamma^(j - i) for j >= i, else 0
    disc_end: jnp.ndarray  # [L]   gamma^(end[i] - i)


def windows(ep: Episode, seg_len: Optional[int], gamma: float, seg_mode: str = "segment") -> Windows:
    """A-SEG: non-overlapping segments of ``seg_len`` decision steps from t = 0 (A3C style); every source in a
    segment bootstraps from the first step after it if that step is a decision state; a terminal reward-only
    step at the segment boundary is appended to the segment as a reward (V(terminal) := its reward).
    ``seg_mode='sliding'`` (sensitivity): every source i has its own window [i, i + seg_len) and
    bootstraps from step i + seg_len (the textbook n-step return); same terminal and pair conventions."""
    L = ep.valid.shape[0]
    t = jnp.arange(L)
    sl = L if seg_len is None else seg_len
    if seg_mode == "segment":
        end = (t // sl + 1) * sl
    elif seg_mode == "sliding":
        end = t + sl
    else:
        raise ValueError(seg_mode)
    end_c = jnp.minimum(end, L - 1)
    in_range = end < L
    end_dec = in_range & ep.decision[end_c]
    end_term = in_range & ep.valid[end_c] & ~ep.decision[end_c]
    I, J = t[:, None], t[None, :]
    src = ep.decision[:, None]
    win = (J >= I) & ((J < end[:, None]) | ((J == end[:, None]) & end_term[:, None])) & ep.valid[None, :] & src
    boot = end_dec & ep.decision
    hpair = (win | ((J == end_c[:, None]) & boot[:, None])) & src
    dt = (J - I).astype(jnp.float64)
    disc = jnp.where(J >= I, jnp.power(gamma, jnp.maximum(dt, 0.0)), 0.0)
    disc_end = jnp.power(gamma, (end_c - t).astype(jnp.float64))
    return Windows(win, boot, end_c, hpair, disc, disc_end)


def segment_returns(ep: Episode, w: Windows, V: jnp.ndarray) -> jnp.ndarray:
    """G_i = sum_{j in win_i} gamma^(j-i) R_j + boot_i gamma^(T-i) V(X_T)  (Alg. 1 l.13; PG n-step return)."""
    G = jnp.sum(w.win * w.disc * ep.reward[None, :], axis=1)
    return G + jnp.where(w.boot, w.disc_end * V[ep.obs[w.end]], 0.0)


def mc_returns(ep: Episode, gamma: float) -> jnp.ndarray:
    """Full Monte-Carlo return from every step (Alg. 2; A-RET-MC)."""
    w = windows(ep, None, gamma)
    return jnp.sum((w.disc > 0) * ep.valid[None, :] * w.disc * ep.reward[None, :], axis=1)


def _scatter_rows(shape, idx, vals, mask):
    return jnp.zeros(shape).at[idx].add(vals * mask.reshape(mask.shape + (1,) * (vals.ndim - mask.ndim)))


class Grads(NamedTuple):
    theta: jnp.ndarray
    psi: jnp.ndarray
    phi: jnp.ndarray
    V: jnp.ndarray
    rhat: jnp.ndarray


def _zeros_like(p: Params) -> Grads:
    return Grads(*(jnp.zeros_like(x) for x in p))


# ---------------------------------------------------------------- State HCA (Alg. 1 / Eq. 7 / official)
def state_hca_credit(pi_o, h_pair, h_end, rew, w: Windows, V_end, variant: str, rhat_o=None):
    """Per-source logit credit G_hca[i, b] such that dtheta[o_i, b] = G_hca[i,b] - pi(b) sum_a G_hca[i,a].

    official: G_hca[i,b] = sum_{j in win_i, j >= i} gamma^(j-i) R_j (h(b|o_i,o_j) - pi(b|o_i))
    paper    (Alg. 1)   : the j = i term is replaced by pi(b) (rhat(o_i,b) - sum_a pi(a) rhat(o_i,a)).
    Both add the bootstrap term boot_i gamma^(T-i) V(X_T) (h(b|o_i,o_T) - pi(b|o_i)) (A-BOOT-FORM)."""
    L = rew.shape[0]
    t = jnp.arange(L)
    win = w.win
    if variant == "paper":
        win = win & (t[None, :] > t[:, None])
    elif variant != "official":
        raise ValueError(variant)
    coef = (win * w.disc * rew[None, :])[..., None]                       # [L, L, 1]
    G = jnp.sum(coef * (h_pair - pi_o[:, None, :]), axis=1)               # [L, A]
    G = G + jnp.where(w.boot, w.disc_end * V_end, 0.0)[:, None] * (h_end - pi_o)
    if variant == "paper":
        G = G + pi_o * (rhat_o - jnp.sum(pi_o * rhat_o, -1, keepdims=True))
    return G


def state_hca_grads(params: Params, ep: Episode, seg_len, gamma: float, variant: str = "official",
                    h_pair_override=None, seg_mode: str = "segment") -> Grads:
    pi = policy(params.theta)
    o = ep.obs
    pi_o = pi[o]                                                           # [L, A]
    w = windows(ep, seg_len, gamma, seg_mode)
    h_pair = hindsight(params.psi[o[:, None], o[None, :]]) if h_pair_override is None else h_pair_override
    h_end = h_pair[jnp.arange(o.shape[0]), w.end]                          # [L, A]
    V_end = params.V[o[w.end]]
    G = state_hca_credit(pi_o, h_pair, h_end, ep.reward, w, V_end, variant, params.rhat[o])
    dth_i = G - pi_o * jnp.sum(G, -1, keepdims=True)
    A = params.theta.shape[-1]
    onehot = jax.nn.one_hot(ep.action, A)
    src = ep.decision
    g = _zeros_like(params)
    dtheta = _scatter_rows(params.theta.shape, o, dth_i, src)
    dpsi = jnp.zeros(params.psi.shape).at[o[:, None], o[None, :]].add(
        w.hpair[..., None] * (onehot[:, None, :] - h_pair))
    Gi = segment_returns(ep, w, params.V)
    dV = _scatter_rows(params.V.shape, o, Gi - params.V[o], src)
    drhat = jnp.zeros(params.rhat.shape)
    if variant == "paper":
        drhat = drhat.at[o, ep.action].add(src * (ep.reward - params.rhat[o, ep.action]))
    return g._replace(theta=dtheta, psi=dpsi, V=dV, rhat=drhat)


# ---------------------------------------------------------------- Return HCA (Alg. 2 / Eq. 8 / official)
def return_bins(Z: jnp.ndarray, centres: jnp.ndarray) -> jnp.ndarray:
    """Nearest bin centre (official ReturnHCA: argmin |bins - G|, first index on ties)."""
    return jnp.argmin(jnp.abs(centres[None, :] - Z[:, None]), axis=1)


def return_hca_grads(params: Params, ep: Episode, gamma: float, centres: jnp.ndarray, hz_override=None) -> Grads:
    pi = policy(params.theta)
    o = ep.obs
    pi_o = pi[o]
    Z = mc_returns(ep, gamma)
    b = return_bins(Z, centres)
    hz = hindsight(params.phi[o, b]) if hz_override is None else hz_override   # [L, A]
    A = params.theta.shape[-1]
    onehot = jax.nn.one_hot(ep.action, A)
    a = ep.action
    Ghca = Z * (1.0 - pi_o[jnp.arange(o.shape[0]), a] / hz[jnp.arange(o.shape[0]), a])
    src = ep.decision
    g = _zeros_like(params)
    dtheta = _scatter_rows(params.theta.shape, o, (onehot - pi_o) * Ghca[:, None], src)
    dphi = jnp.zeros(params.phi.shape).at[o, b].add(src[:, None] * (onehot - hz))
    dV = _scatter_rows(params.V.shape, o, Z - params.V[o], src)       # trained, never used (as in the official code)
    return g._replace(theta=dtheta, phi=dphi, V=dV)


# ---------------------------------------------------------------- Policy Gradient baseline (paper §2, §5)
def pg_grads(params: Params, ep: Episode, seg_len, gamma: float, seg_mode: str = "segment") -> Grads:
    """n-step advantage actor-critic: dtheta[o_i] += (1{a=A_i} - pi) (G_i - V(o_i)), dV[o_i] += G_i - V(o_i)."""
    pi = policy(params.theta)
    o = ep.obs
    pi_o = pi[o]
    w = windows(ep, seg_len, gamma, seg_mode)
    adv = segment_returns(ep, w, params.V) - params.V[o]
    onehot = jax.nn.one_hot(ep.action, params.theta.shape[-1])
    src = ep.decision
    g = _zeros_like(params)
    return g._replace(theta=_scatter_rows(params.theta.shape, o, (onehot - pi_o) * adv[:, None], src),
                      V=_scatter_rows(params.V.shape, o, adv, src))


def episode_grads(method: str, params: Params, ep: Episode, *, seg_len, gamma, centres, variant="official",
                  seg_mode="segment"):
    if method == "state_hca":
        return state_hca_grads(params, ep, seg_len, gamma, variant, seg_mode=seg_mode)
    if method == "return_hca":
        return return_hca_grads(params, ep, gamma, centres)
    if method == "pg":
        return pg_grads(params, ep, seg_len, gamma, seg_mode)
    raise ValueError(method)


# ---------------------------------------------------------------- Fig. 3-right counterfactual estimators
def counterfactual_q(method: str, params: Params, ep: Episode, gamma: float, centres, immediate: str = "rhat"):
    """All-action estimate of Q(x0, .) from ONE rollout of a fixed policy (A-F3R), with the current tables.

    state  (Thm 1/6): Qhat(x0,a) = imm(a) + sum_{k>=1} gamma^k h(a|x0,X_k)/pi(a|x0) R_k
           imm = rhat(x0,a) ('rhat', Alg. 1), the true r(x0) ('exact'), or the official j = 0 term
           R_0 h(a|x0,x0)/pi(a|x0) ('official').
    return (Thm 5):   Qhat(x0,a) = Z h_z(a|x0,Z)/pi(a|x0)
    pg (indicator):   Qhat(x0,a) = 1{A_0=a} Z  (the caller subtracts the baseline; only the difference
                      short - long is reported, see train.run_fig3r)."""
    pi = policy(params.theta)
    o = ep.obs
    pi0 = pi[o[0]]
    L = o.shape[0]
    if method == "state_hca":
        h = hindsight(params.psi[o[0], o])                                     # [L, A]
        k = jnp.arange(L)
        w = (k >= 1) & ep.valid
        q = jnp.sum((w * gamma ** k * ep.reward)[:, None] * h, axis=0) / pi0
        if immediate == "rhat":
            q = q + params.rhat[o[0]]
        elif immediate == "exact":
            q = q + ep.reward[0]  # deterministic state reward of x0 in every paper task
        elif immediate == "official":
            q = q + ep.reward[0] * h[0] / pi0
        else:
            raise ValueError(immediate)
        return q
    Z = mc_returns(ep, gamma)[0]
    if method == "return_hca":
        hz = hindsight(params.phi[o[0], return_bins(Z[None], centres)[0]])
        return Z * hz / pi0
    if method == "pg":
        return jax.nn.one_hot(ep.action[0], pi.shape[-1]) * Z
    raise ValueError(method)

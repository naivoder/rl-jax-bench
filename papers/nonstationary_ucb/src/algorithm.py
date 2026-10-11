"""The policies: UCB-1, D-UCB, SW-UCB (paper Algorithms 1-3), EXP3.S (Auer et al. 2002, Fig. 6) and the
labelled ORACLE-RESTART-UCB1 reference, plus the jitted, vmapped lax.scan runner.

Re-implemented from hand derivation (no code was copied; no official code exists, README).

Conventions ():
  * the index at round t uses the t-1 completed plays (D-UCB: most recent play has weight 1).
  * UCB-1 uses log t and SW-UCB log(min(t, tau)) with t the CURRENT round index; D-UCB uses
    log n_t(gamma), n_t(gamma) = sum_i N_t(gamma, i) (D-4).
  * ties among maximal indices are broken uniformly at random (exact float64 equality);
    ``tie="lowest"`` is the ablation (lowest arm index).
  * an arm with zero (window/discounted) count gets index +inf (the padding's limit).
  * EXP3.S runs in log space; the draw is inverse-CDF on one policy-branch uniform.
  * UCB-type policies play arm t at rounds t = 1..K; EXP3.S has no forced initialisation.

The ONLY differences between UCB-1, D-UCB and SW-UCB are the statistics (undiscounted / discounted /
windowed), the log argument and the padding constant (2B for D-UCB, B otherwise) -- the mechanism under test.
Everything else (environment draws, tie-break, initialisation, accumulators) is shared code.
"""
from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from . import common  # noqa: F401  (enables float64 before any array is built)
from .model import Acc, CountStats, Exp3SStats, Params, RestartStats, WindowStats

F64 = jnp.float64
POLICIES = ("ucb1", "ducb", "swucb", "exp3s", "oracle_ucb1", "ducb_lit")
PAPER_POLICIES = ("ucb1", "exp3s", "ducb", "swucb")  # the four arms of paper Figure 1
POLICY_ID = {"ucb1": 0, "ducb": 1, "swucb": 2, "exp3s": 3, "oracle_ucb1": 4, "ducb_lit": 5}  # PRNG key tree (the key tree in src/algorithm.py)
LABELS = {"ucb1": "UCB-1", "ducb": "D-UCB", "swucb": "SW-UCB", "exp3s": "EXP3.S",
          "oracle_ucb1": "ORACLE-RESTART-UCB1 (EXTENSION reference)",
          "ducb_lit": "D-UCB, literal 'discount then decide' time convention (ABLATION of )"}


# ---------------------------------------------------------------- hyperparameter formulas (paper §5, Remarks 3/9)
def paper_gamma(T: int) -> float:
    """paper §5: gamma = 1 - 1/(4 sqrt T) (A5 reading)."""
    return 1.0 - 1.0 / (4.0 * math.sqrt(T))


def paper_tau(T: int) -> int:
    """paper §5: tau = 4 sqrt(T log T), rounded."""
    return int(round(4.0 * math.sqrt(T * math.log(T))))


def exp3s_g(K: int, S: float, T: int) -> float:
    """Auer et al. 2002 Cor. 8.3 / paper §5: g = min(1, sqrt(K (S ln(KT) + e) / ((e-1) T)))."""
    return min(1.0, math.sqrt(K * (S * math.log(K * T) + math.e) / ((math.e - 1.0) * T)))


def make_params(B=1.0, xi=0.5, gamma=1.0, tau=1, g=0.0, alpha=0.0, eta_mult=1.0) -> Params:
    return Params(jnp.asarray(B, F64), jnp.asarray(xi, F64), jnp.asarray(gamma, F64),
                  jnp.asarray(tau, jnp.int32), jnp.asarray(g, F64), jnp.asarray(alpha, F64), jnp.asarray(eta_mult, F64))


# ---------------------------------------------------------------- indices (hand derivation)
def ucb1_index(n, s, t, xi, B):
    """X̄_i + B sqrt(xi log t / N_i); +inf when N_i = 0 (paper Alg. 1).

    MIXED time convention (declared as /A7): N_i and the sum s are statistics of the
    t-1 COMPLETED plays, while the log uses the CURRENT round index t (the author's stationary MATLAB policyUCB.m).
    The paper writes X̄_t, N_t and log t at the same index t; the two readings differ only by log t vs log(t-1).
    """
    ns = jnp.where(n > 0, n, 1.0)
    pad = B * jnp.sqrt(xi * jnp.log(jnp.asarray(t, F64)) / ns)
    return jnp.where(n > 0, s / ns + pad, jnp.inf)


def ducb_index(n, s, xi, B):
    """X̄_t(gamma,i) + 2B sqrt(xi log n_t(gamma) / N_t(gamma,i)), n_t = sum_i N_t(gamma,i) (paper Alg. 2; D-4).

    2B is the AS-PRINTED constant (paper §1.2 p.4). The paper contradicts itself here: the next sentence says that for
    gamma = 1 D-UCB "boils down to the standard UCB-1", which (p.3 UCB-1 and p.5 SW-UCB both use B) holds only with
    padding B, and Figure 1 is reproduced only with B (the padding-B arm is run as xi/4 with 2B).

    After the K initial plays n_t >= 1, so the clamp max(n_t, 1) is inactive; it only keeps the unused
    pre-initialisation values finite. N_t(gamma,i) = 0 (float underflow after ~2.8e5 unplayed steps) -> +inf.
    """
    ntot = jnp.maximum(jnp.sum(n), 1.0)
    ns = jnp.where(n > 0, n, 1.0)
    pad = 2.0 * B * jnp.sqrt(xi * jnp.log(ntot) / ns)
    return jnp.where(n > 0, s / ns + pad, jnp.inf)


def swucb_index(n, s, t, tau, xi, B):
    """X̄_t(tau,i) + B sqrt(xi log min(t, tau) / N_t(tau,i)); +inf when N_t(tau,i) = 0 (paper Alg. 3).

    MIXED time convention: the window holds the last tau COMPLETED plays (rounds t-tau..t-1), while the
    log uses min(t, tau) with the CURRENT round t (as UCB-1 above). Only rounds t <= tau are affected (log t vs log(t-1)).
    """
    m = jnp.minimum(jnp.asarray(t, F64), jnp.asarray(tau, F64))
    nf = n.astype(F64)
    ns = jnp.where(n > 0, nf, 1.0)
    pad = B * jnp.sqrt(xi * jnp.log(m) / ns)
    return jnp.where(n > 0, s / ns + pad, jnp.inf)


def argmax_tiebreak(index, key, tie: str):
    """argmax with ties broken uniformly at random or by lowest arm index (ablation)."""
    is_max = index == jnp.max(index)
    if tie == "random":
        r = jax.random.uniform(key, index.shape, dtype=F64)
        return jnp.argmax(jnp.where(is_max, r, -1.0)).astype(jnp.int32)
    if tie == "lowest":
        return jnp.argmax(is_max).astype(jnp.int32)
    raise ValueError(f"unknown tie mode {tie!r}")


def exp3s_probs(logw, g, K):
    """p_i = (1 - g) w_i / sum_j w_j + g / K (Auer et al. 2002 Fig. 6)."""
    return (1.0 - g) * jax.nn.softmax(logw) + g / K


def exp3s_update_logw(logw, a, x, p, g, alpha, K, eta_mult=1.0):
    """log-space version of w_j <- w_j exp(g xhat_j / K) + (e alpha / K) sum_i w_i, max-normalised.

    ``eta_mult`` scales the learning rate g/K (recheck RC3 EXTENSION). 1.0 is Auer's coupled EXP3.S, the paper's
    arm; (1.0 * g) == g exactly, so the default is bit-identical to the pre-RC3 update (tested).
    """
    xhat = jnp.where(jnp.arange(K) == a, x / p[a], 0.0)
    share = jnp.log(math.e * alpha / K) + jax.scipy.special.logsumexp(logw)
    new = jnp.logaddexp(logw + (eta_mult * g) * xhat / K, share)
    return new - jnp.max(new)


# ---------------------------------------------------------------- per-policy init / select / update
def init_state(name: str, K: int, buf_len: int = 1):
    zf, zi = jnp.zeros(K, F64), jnp.zeros(K, jnp.int32)
    if name in ("ucb1", "ducb", "ducb_lit"):
        return CountStats(zf, zf)
    if name == "oracle_ucb1":
        return RestartStats(zf, zf, jnp.asarray(0, jnp.int32))
    if name == "swucb":
        return WindowStats(zi, zf, jnp.zeros(buf_len, jnp.int32), jnp.zeros(buf_len, F64))
    if name == "exp3s":
        return Exp3SStats(zf)
    raise ValueError(name)


def pre_step(name, state, t, reset):
    """ORACLE-RESTART-UCB1 forgets everything at a true breakpoint; no-op for every other policy."""
    if name != "oracle_ucb1":
        return state
    return RestartStats(jnp.where(reset, 0.0, state.n), jnp.where(reset, 0.0, state.s),
                        jnp.where(reset, t - 1, state.t0).astype(jnp.int32))


def select(name, state, t, params: Params, key, K: int, tie: str):
    """Choose I_t at round t (1-indexed). Returns (action int32, aux (K,) = index or EXP3.S probs, bad flag)."""
    if name == "exp3s":
        p = exp3s_probs(state.logw, params.g, K)
        u = jax.random.uniform(key, (), dtype=F64)
        c = jnp.cumsum(p)
        a = jnp.minimum(jnp.sum(c < u * c[-1]), K - 1).astype(jnp.int32)
        return a, p, ~jnp.all(jnp.isfinite(state.logw)) | ~jnp.all(jnp.isfinite(p))
    if name == "ucb1":
        idx, t_loc = ucb1_index(state.n, state.s, t, params.xi, params.B), t
    elif name == "oracle_ucb1":
        t_loc = t - state.t0
        idx = ucb1_index(state.n, state.s, t_loc, params.xi, params.B)
    elif name == "ducb":
        idx, t_loc = ducb_index(state.n, state.s, params.xi, params.B), t
    elif name == "ducb_lit":  # ablation: statistics 'at round t' sum gamma^{t-s} over s < t (one extra discount)
        idx, t_loc = ducb_index(params.gamma * state.n, params.gamma * state.s, params.xi, params.B), t
    elif name == "swucb":
        idx, t_loc = swucb_index(state.n, state.s, t, params.tau, params.xi, params.B), t
    else:
        raise ValueError(name)
    a_idx = argmax_tiebreak(idx, key, tie)
    a = jnp.where(t_loc <= K, t_loc - 1, a_idx).astype(jnp.int32)  # round-robin initialisation
    return a, idx, jnp.any(jnp.isnan(idx))


def update(name, state, a, x, t, params: Params, aux, K: int):
    """Fold the play (I_t = a, X_t(a) = x) of round t into the statistics."""
    oh = (jnp.arange(K) == a)
    ohf = oh.astype(F64)
    if name == "ucb1":
        return CountStats(state.n + ohf, state.s + ohf * x)
    if name == "oracle_ucb1":
        return RestartStats(state.n + ohf, state.s + ohf * x, state.t0)
    if name in ("ducb", "ducb_lit"):  # D-3: N <- gamma N + 1{I=i}, S <- gamma S + 1{I=i} X
        return CountStats(params.gamma * state.n + ohf, params.gamma * state.s + ohf * x)
    if name == "swucb":  # D-7: pop round t - tau (same ring slot) once t > tau, push round t
        pos = jnp.mod(t - 1, params.tau)
        pop = t > params.tau
        old_a, old_x = state.buf_a[pos], state.buf_x[pos]
        ooh = (jnp.arange(K) == old_a) & pop
        n = state.n - ooh.astype(jnp.int32) + oh.astype(jnp.int32)
        s = state.s - ooh.astype(F64) * old_x + ohf * x
        return WindowStats(n, s, state.buf_a.at[pos].set(a), state.buf_x.at[pos].set(x))
    if name == "exp3s":
        return Exp3SStats(exp3s_update_logw(state.logw, a, x, aux, params.g, params.alpha, K, params.eta_mult))
    raise ValueError(name)


# ---------------------------------------------------------------- environment draws (PRNG key tree, the key tree in src/algorithm.py)
def env_uniforms(k_env_r, t, K: int):
    """u_t(i) = uniform(fold_in(k_env_r, t), (K,)); X_t(i) = 1{u_t(i) < p_t(i)} (CRN)."""
    return jax.random.uniform(jax.random.fold_in(k_env_r, t), (K,), dtype=F64)


def rep_keys(seed: int, exp_id: int, cfg_idx: int, policy_id: int, R: int, r0: int = 0):
    """Replication keys: env keys are shared by every policy of a configuration (CRN); policy keys differ.

    root = PRNGKey(seed); k_cfg = fold_in(fold_in(root, exp_id), cfg_idx)
    k_env_r = fold_in(fold_in(k_cfg, 0), r) ; k_pol_r = fold_in(fold_in(fold_in(k_cfg, 1), policy_id), r)
    """
    k_cfg = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed), exp_id), cfg_idx)
    k_env = jax.random.fold_in(k_cfg, 0)
    k_pol = jax.random.fold_in(jax.random.fold_in(k_cfg, 1), policy_id)
    r = jnp.arange(r0, r0 + R)
    return (jax.vmap(lambda i: jax.random.fold_in(k_env, i))(r),
            jax.vmap(lambda i: jax.random.fold_in(k_pol, i))(r))


def init_acc(K: int) -> Acc:
    return Acc(jnp.asarray(0.0, F64), jnp.asarray(0, jnp.int32), jnp.zeros(K, jnp.int32),
               jnp.asarray(0.0, F64), jnp.asarray(False))


def make_step(name: str, K: int, tie: str):
    """One round for one replication. Returns (state, acc, action, aux)."""
    def step(state, acc: Acc, t, mu, reset, k_env_r, k_pol_r, params: Params):
        u = env_uniforms(k_env_r, t, K)
        x_all = (u < mu).astype(F64)
        state = pre_step(name, state, t, reset)
        a, aux, bad = select(name, state, t, params, jax.random.fold_in(k_pol_r, t), K, tie)
        state = update(name, state, a, x_all[a], t, params, aux, K)
        gap = jnp.max(mu) - mu[a]  # pseudo-regret increment (A12)
        acc = Acc(acc.regret + gap, acc.arm1 + (a == 0).astype(jnp.int32),
                  acc.ntilde + ((jnp.arange(K) == a) & (gap > 0)).astype(jnp.int32),
                  acc.usum + jnp.sum(u), acc.bad | bad)
        return state, acc, a, aux
    return step


# ---------------------------------------------------------------- vmapped chunk runner
@functools.lru_cache(maxsize=None)
def make_chunk_runner(name: str, K: int, tie: str, log_every: int):
    """jit(vmap over replications) of a scan over one chunk of rounds, logging every ``log_every`` rounds.

    Inputs: state/acc batched over R; keys (R, 2); params (unbatched); means (n_log, log_every, K);
    resets (n_log, log_every); t_start = first round of the chunk. Returns the new state/acc and the
    (R, n_log) curves of cumulative regret and arm-1 count. Nothing leaves the device inside the scan;
    the NaN/Inf guard fires once per chunk (jax.debug.callback in common.nan_guard).
    """
    step = make_step(name, K, tie)

    def one_rep(state, acc, k_env_r, k_pol_r, params, means, resets, t_start):
        n_log = means.shape[0]
        ts = (t_start + jnp.arange(n_log * log_every, dtype=jnp.int32)).reshape(n_log, log_every)

        def inner(carry, xs):
            st, ac = carry
            t, mu, rs = xs
            st, ac, _, _ = step(st, ac, t, mu, rs, k_env_r, k_pol_r, params)
            return (st, ac), None

        def outer(carry, xs):
            carry, _ = jax.lax.scan(inner, carry, xs)
            return carry, (carry[1].regret, carry[1].arm1)

        (state, acc), (reg, arm1) = jax.lax.scan(outer, (state, acc), (ts, means, resets))
        return state, acc, reg, arm1

    vm = jax.vmap(one_rep, in_axes=(0, 0, 0, 0, None, None, None, None))

    def chunk(state, acc, k_env, k_pol, params, means, resets, t_start):
        state, acc, reg, arm1 = vm(state, acc, k_env, k_pol, params, means, resets, t_start)
        common.nan_guard({"regret": acc.regret, "curve": reg, "bad": jnp.where(acc.bad, jnp.nan, 0.0)},
                         {"regret": acc.regret, "bad": acc.bad, "t_start": t_start}, tag=POLICY_ID[name])
        return state, acc, reg, arm1

    return jax.jit(chunk)


def batched_init(name: str, K: int, R: int, buf_len: int):
    st = init_state(name, K, buf_len)
    ac = init_acc(K)
    rep = lambda x: jnp.broadcast_to(x, (R,) + x.shape)  # noqa: E731
    return jax.tree_util.tree_map(rep, st), jax.tree_util.tree_map(rep, ac)


# ---------------------------------------------------------------- single-replication trajectory (tests, Toy-check)
@functools.partial(jax.jit, static_argnames=("name", "K", "tie", "buf_len"))
def trajectory(name, K, tie, buf_len, params, means, resets, k_env_r, k_pol_r):
    """Run one replication over all rounds and return (actions (T,), aux (T, K), final acc, final state)."""
    step = make_step(name, K, tie)
    T = means.shape[0]

    def body(carry, xs):
        st, ac = carry
        t, mu, rs = xs
        st, ac, a, aux = step(st, ac, t, mu, rs, k_env_r, k_pol_r, params)
        return (st, ac), (a, aux)

    ts = jnp.arange(1, T + 1, dtype=jnp.int32)
    (st, ac), (acts, aux) = jax.lax.scan(body, (init_state(name, K, buf_len), init_acc(K)), (ts, means, resets))
    return acts, aux, ac, st


def as_device_env(means: np.ndarray, resets: np.ndarray):
    return jnp.asarray(means, F64), jnp.asarray(resets)

"""The paper's two mechanisms and every learner they are tested with, as pure jit/vmap-safe JAX functions.

* `gae_peb`            -- GAE with separate termination / reset flags (paper Eq. 6, §3.1 last paragraph; 
                          §8; Tonic replays/utils.py lambda_returns). peb = 0 reproduces baselines ppo2 GAE
                          (timeout = terminal), checked against the vendored PPO-details `gae` in tests.
* `tabular_q`          -- tabular Q-learning, standard / time-aware (one table per step) / PEB, with ε-greedy behavior
                          (ε = 1 = uniform random, paper §2.2 "with random actions"); E1 and E2.
* `reinforce_jump`     -- memoryless Monte-Carlo policy gradient on Last Moment (time-unaware, no TD leakage;
                          ) -- E1.
* `buffer_q`           -- Zhang & Sutton's Buffer-Q (experience replay) with/without PEB on the Difficult Gridworld
                          (paper §3.4) -- E9.
PPO's clipped surrogate, value loss, optimizer and networks are the vendored PPO-details functions (src/ppo8).

Arms are selected by TRACED integer/float flags so that every arm of an experiment is one vmapped batch and shares
every PRNG key (common random numbers); no flag ever enters the key tree.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

AGENT_STANDARD, AGENT_TA, AGENT_PEB = 0, 1, 2   # paper §2.2 / §2.2 / §3.1 (Figure 1b, 1c, 1d)


# ===================================================================== GAE with partial-episode bootstrapping
def gae_peb(rewards, values, final_values, terms, resets, gamma, lam, peb):
    """rewards, values, final_values, terms, resets: (M, ...) with time on axis 0.

    final_values[t] = V(s'_t), the value of the TRUE next observation of transition t (pre-reset at an episode end;
    = V(s_{t+1}) otherwise; at t = M-1 it closes the rollout).
      term_eff_t = term_t           if peb   (bootstrap through timeouts, paper Eq. 6)
                 = reset_t          if not   (every episode end is terminal: the paper's standard agent, Eq. 4)
      delta_t    = r_t + gamma (1 - term_eff_t) V(s'_t) - V(s_t)
      A_t        = delta_t + gamma lam (1 - reset_t) A_{t+1}         (trace always cut at a reset; A_M = 0)
    Returns (advantages, value targets = A + V).
    """
    terms = terms.astype(values.dtype)
    resets = resets.astype(values.dtype)
    term_eff = jnp.where(peb > 0.5, terms, resets)
    deltas = rewards + gamma * (1.0 - term_eff) * final_values - values

    def body(nxt, x):
        d, rs = x
        a = d + gamma * lam * (1.0 - rs) * nxt
        return a, a

    _, adv = jax.lax.scan(body, jnp.zeros_like(values[0]), (deltas, resets), reverse=True)
    return adv, adv + values


# ===================================================================== helpers
def argmax_random_ties(key, q):
    """Uniform choice among the maximisers of q (last axis)."""
    m = q >= q.max(-1, keepdims=True)
    return jax.random.categorical(key, jnp.where(m, 0.0, -jnp.inf), axis=-1)


def first_argmax(q):
    return jnp.argmax(q, -1)


# ===================================================================== tabular Q-learning (E1, E2)
class TabQOut(NamedTuple):
    Q: jax.Array        # (T+1, S, A); standard / PEB use slice 0 only, TA uses slice t (Q[T] = 0)
    N: jax.Array        # visit counts per (table, s, a)
    returns: jax.Array  # (n_episodes) undiscounted return of each training episode (behavior policy)


def tabular_q(exp_key, NX, R, TERM_NEXT, start, *, agent, T: int, gamma, eps, n_episodes: int, lr_power):
    """Tabular Q-learning (paper §2.2 / §3.1) for one seed.

    exp_key   : the experiment key of this seed (key tree: k_env = fold_in(exp, 1), k_act = fold_in(exp, 2)).
    NX, R     : (S, A) next state / reward. TERM_NEXT: (S, A) bool, transition enters an environmental terminal.
    start     : (S) start distribution.
    agent     : AGENT_STANDARD (timeout = terminal, time-unaware), AGENT_TA (one table per elapsed step, timeout =
                terminal -- Eq. 4), AGENT_PEB (time-unaware, bootstrap at timeouts -- Eq. 6). Traced int.
    eps       : behavior ε-greedy w.r.t. the agent's own current table (1.0 = uniform random). Traced float.
    lr        : alpha = (1 + n(table, s, a)) ** (-lr_power)  ( 'decaying learning rate').
    """
    S, A = R.shape
    k_env = jax.random.fold_in(exp_key, 1)
    k_act = jax.random.fold_in(exp_key, 2)
    is_ta = agent == AGENT_TA
    is_peb = agent == AGENT_PEB
    logits_start = jnp.log(start)

    def episode(carry, e):
        Q, N = carry
        s = jax.random.categorical(jax.random.fold_in(jax.random.fold_in(k_env, 0), e), logits_start)
        alive = jnp.bool_(True)
        ret = jnp.float32(0.0)
        for t in range(T):  # T is static and small (<= 10)
            g = e * T + t
            k = jax.random.fold_in(k_act, g)
            k1, k2, k3 = jax.random.split(k, 3)
            ti = jnp.where(is_ta, t, 0)
            tn = jnp.where(is_ta, t + 1, 0)
            greedy = argmax_random_ties(k3, Q[ti, s])
            a = jnp.where(jax.random.uniform(k1) < eps, jax.random.randint(k2, (), 0, A), greedy)
            s2, r, term = NX[s, a], R[s, a], TERM_NEXT[s, a]
            timeout = t == T - 1
            boot = (1.0 - term) * jnp.where(is_peb, 1.0, 1.0 - timeout)
            y = r + gamma * boot * Q[tn, s2].max()
            n = N[ti, s, a]
            alpha = (1.0 + n) ** (-lr_power)
            upd = jnp.where(alive, alpha * (y - Q[ti, s, a]), 0.0)
            Q = Q.at[ti, s, a].add(upd)
            N = N.at[ti, s, a].add(alive.astype(N.dtype))
            ret = ret + jnp.where(alive, r, 0.0)
            alive = jnp.logical_and(alive, jnp.logical_not(term))
            s = s2
        return (Q, N), ret

    Q0 = jnp.zeros((T + 1, S, A), jnp.float32)
    N0 = jnp.zeros((T + 1, S, A), jnp.float32)
    (Q, N), rets = jax.lax.scan(episode, (Q0, N0), jnp.arange(n_episodes))
    return TabQOut(Q, N, rets)


# ===================================================================== REINFORCE on Last Moment (E1, time-unaware MC)
def reinforce_jump(exp_key, *, T: int, gamma, n_episodes: int, lr, baseline_rate=0.05, theta0=0.0):
    """Memoryless jump probability p = sigmoid(theta) in state A (the only state with a choice), trained by
    REINFORCE with a running-mean baseline. Exact J(p) is in src/exact.py. Returns the theta trajectory."""
    k_act = jax.random.fold_in(exp_key, 2)

    def episode(carry, e):
        th, b = carry
        p = jax.nn.sigmoid(th)
        inA = jnp.bool_(True)
        G = 0.0
        score = 0.0
        for t in range(T):
            jump = jax.random.uniform(jax.random.fold_in(k_act, e * T + t)) < p
            r = jnp.where(inA, jnp.where(jump, 1.0, 0.0), -1.0)
            score = score + jnp.where(inA, jump.astype(jnp.float32) - p, 0.0)  # d/dtheta log Bernoulli(sigmoid)
            G = G + gamma ** t * r
            inA = jnp.logical_and(inA, jnp.logical_not(jump))
        th = th + lr * (G - b) * score
        b = b + baseline_rate * (G - b)
        return (th, b), th

    (th, _), traj = jax.lax.scan(episode, (jnp.float32(theta0), jnp.float32(0.0)), jnp.arange(n_episodes))
    return th, traj


# ===================================================================== Buffer-Q (E9, paper §3.4)
class BufQOut(NamedTuple):
    Q: jax.Array
    bin_ret_sum: jax.Array   # (n_bins) sum of returns of episodes ENDING in the bin
    bin_cnt: jax.Array       # (n_bins)
    bin_len_sum: jax.Array


def buffer_q(exp_key, NX, goal: int, start: int, *, peb, B: int, total_steps: int, bin_steps: int, T: int = 200,
             eps: float = 0.1, lr: float = 0.1, minibatch: int = 10, min_buffer: int = 10, sequential: bool = True):
    """Buffer-Q (Zhang & Sutton 2017 §4,..28) with r = -1 per step, gamma = 1 (paper §3.4).

    Each env step: ε-greedy action (ties uniform); store (s, a, flag) in a FIFO of size B (s' = NX[s, a] is
    deterministic, so it is recomputed); once the buffer holds >= min_buffer transitions, draw `minibatch` indices
    uniformly WITH replacement from the stored ones and apply the tabular updates sequentially (or one
    averaged update per sample if sequential=False). flag = terminal: goal OR (timeout AND NOT peb).
    """
    k_act = jax.random.fold_in(exp_key, 2)
    k_mb = jax.random.fold_in(exp_key, 3)
    NX = jnp.asarray(NX, jnp.int32)
    A = NX.shape[1]
    n_bins = total_steps // bin_steps
    buf0 = jnp.zeros((B,), jnp.int32)

    def step(c, g):
        Q, buf, n_stored, s, t, ep_ret = c
        k1, k2, k3 = jax.random.split(jax.random.fold_in(k_act, g), 3)
        greedy = argmax_random_ties(k3, Q[s])
        a = jnp.where(jax.random.uniform(k1) < eps, jax.random.randint(k2, (), 0, A), greedy)
        s2 = NX[s, a]
        at_goal = s2 == goal
        t2 = t + 1
        timeout = jnp.logical_and(t2 >= T, jnp.logical_not(at_goal))
        flag = jnp.logical_or(at_goal, jnp.logical_and(timeout, peb < 0.5))
        code = ((s * A + a) * 2 + flag.astype(jnp.int32)).astype(jnp.int32)
        buf = buf.at[jnp.mod(n_stored, B)].set(code)
        n_stored = n_stored + 1
        size = jnp.minimum(n_stored, B)
        idx = jax.random.randint(jax.random.fold_in(k_mb, g), (minibatch,), 0, size)
        codes = buf[idx]
        do = n_stored >= min_buffer

        def upd_seq(Q):
            def one(i, Q):
                cd = codes[i]
                fl = cd % 2
                sa = cd // 2
                ss, aa = sa // A, sa % A
                y = -1.0 + (1.0 - fl) * Q[NX[ss, aa]].max()
                return Q.at[ss, aa].add(lr * (y - Q[ss, aa]))
            return jax.lax.fori_loop(0, minibatch, one, Q)

        def upd_mean(Q):
            fl = codes % 2
            sa = codes // 2
            ss, aa = sa // A, sa % A
            y = -1.0 + (1.0 - fl) * Q[NX[ss, aa]].max(-1)
            d = jnp.zeros_like(Q).at[ss, aa].add(y - Q[ss, aa])
            cnt = jnp.zeros_like(Q).at[ss, aa].add(1.0)
            return Q + lr * d / jnp.maximum(cnt, 1.0)

        Qn = upd_seq(Q) if sequential else upd_mean(Q)
        Q = jnp.where(do, Qn, Q)
        ep_ret = ep_ret - 1.0
        end = jnp.logical_or(at_goal, timeout)
        out = (jnp.where(end, ep_ret, 0.0), end.astype(jnp.float32), jnp.where(end, t2, 0).astype(jnp.float32))
        s = jnp.where(end, start, s2).astype(jnp.int32)
        t = jnp.where(end, 0, t2).astype(jnp.int32)
        ep_ret = jnp.where(end, 0.0, ep_ret).astype(jnp.float32)
        return (Q, buf, n_stored, s, t, ep_ret), out

    def one_bin(c, b):
        c, (r, n, ln) = jax.lax.scan(step, c, b * bin_steps + jnp.arange(bin_steps))
        return c, (r.sum(), n.sum(), ln.sum())

    c0 = (jnp.zeros(NX.shape, jnp.float32), buf0, jnp.int32(0), jnp.int32(start), jnp.int32(0), jnp.float32(0.0))
    (Q, *_), (rs, ns, ls) = jax.lax.scan(one_bin, c0, jnp.arange(n_bins))
    return BufQOut(Q, rs, ns, ls)

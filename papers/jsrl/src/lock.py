"""lock toy: the combination lock of Theorem 4.1 / A.3 (paper Appendix A.5, Figure 15), exactly computable.

Layer h has two states, red s*_h and blue b_h.  From s*_h the action a*_h leads to s*_{h+1}; any other action, and every action from blue,
leads to blue.  The reward is 1 only for (s*_{H-1}, a*_{H-1}) (0-indexed layers).  V* = 1.  The lock is an episodic MDP of H steps.

Contents
  * numpy lock dynamics and the explore-then-commit oracle of Algorithm 2, with
      - `exact_etc`  : EXACT probability (fractions.Fraction) that the final policy is optimal, by dynamic programming over the
                       distribution of the learned policy tables with REAL lock rollouts for every enumerated draw sequence;
      - `mc_etc`     : vectorised Monte Carlo of the same algorithm;
      - `scratch_first_reward`: 0-initialised uniform exploration, episodes until the first reward (Theorem 4.1);
      - closed forms `q_stage`, `p_opt_closed`, `p_opt_cov_closed`.
  * `LockEnv`: the lock as a jittable env with the MinAtar interface of src/envs.py (obs bool (10,10,2)), so that the REAL training code
    (src/algorithm.py Fns.interact_chunk / evaluate) runs on it for lock check (e) (the switching rule) and (g) (descriptive learner comparison).
"""
from __future__ import annotations

import itertools
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np


def astar(h: int, A: int) -> int:
    """The correct action at layer h (varies with h so that 'always action 0' is not the solution)."""
    return (3 * h + 1) % A


# ------------------------------------------------------------------------------------------------ numpy dynamics
def lock_return(h0: int, a0: int, table, H: int, A: int) -> int:
    """Return of: start in the RED state of layer h0, take a0, then follow `table[l]` (the chosen action at red layer l) for l > h0.
    Real lock transitions, not the closed form."""
    red = True
    for l in range(h0, H):
        a = a0 if l == h0 else table[l]
        if red and a == astar(l, A):
            red = True
        else:
            red = False
    return int(red)          # reached the end still red  <=>  the last step was (s*_{H-1}, a*_{H-1})  <=>  reward 1


def _choose_dist(means: list[Fraction]) -> dict[int, Fraction]:
    mx = max(means)
    tied = [a for a, m in enumerate(means) if m == mx]
    return {a: Fraction(1, len(tied)) for a in tied}


def exact_etc(H: int, A: int, m: int, p: float | Fraction = 1) -> Fraction:
    """EXACT P(final policy optimal) for the explore-then-commit oracle, stages H-1, ..., 0, m rounds each.
    Round at stage h: the guide reaches s*_h with probability p**h (p = 1: always; coverage variant: the guide takes a*_l w.p. p per step); if it
    does, a_h ~ Uniform(A), the return comes from `lock_return` with the already-learned table; if not, the round carries no information
    about s*_h.  Commit: argmax of the empirical mean return per action (untried actions at 0), ties uniform.
    Dynamic programming over {suffix table: probability}; each round outcome is enumerated (A + 1 outcomes)."""
    p = Fraction(p).limit_denominator(10 ** 6) if not isinstance(p, Fraction) else p
    dist = {(): Fraction(1)}
    for h in range(H - 1, -1, -1):
        reach = p ** h
        nd: dict[tuple, Fraction] = {}
        for suffix, pr in dist.items():
            table = {h + 1 + i: a for i, a in enumerate(suffix)}      # chosen actions at layers h+1 .. H-1
            ret = [lock_return(h, a, table, H, A) for a in range(A)]
            # round outcome: 'miss' (guide derailed) or action a; enumerate multisets via counts for efficiency
            outcomes = [(None, 1 - reach)] + [(a, reach / A) for a in range(A)]
            for seq in itertools.product(range(A + 1), repeat=m):     # index A = miss
                w = Fraction(1)
                n = [0] * A; s = [0] * A
                for o in seq:
                    if o == A:
                        w *= outcomes[0][1]
                    else:
                        w *= outcomes[1 + o][1]
                        n[o] += 1; s[o] += ret[o]
                if w == 0:
                    continue
                means = [Fraction(s[a], n[a]) if n[a] else Fraction(0) for a in range(A)]
                for a, wc in _choose_dist(means).items():
                    key = (a,) + suffix
                    nd[key] = nd.get(key, 0) + pr * w * wc
        dist = nd
    opt = tuple(astar(l, A) for l in range(H))
    return dist.get(opt, Fraction(0))


def q_stage(A: int, m: int, reach: float = 1.0) -> float:
    """P(stage wrong | later stages correct) = (1 - reach/A)^m (A-1)/A   (reach = p^h; reach = 1 gives ((A-1)/A)^(m+1))."""
    return (1.0 - reach / A) ** m * (A - 1) / A


def p_opt_closed(H: int, A: int, m: int) -> float:
    return (1.0 - ((A - 1) / A) ** (m + 1)) ** H


def p_opt_cov_closed(H: int, A: int, m: int, p: float) -> float:
    out = 1.0
    for h in range(H):
        out *= 1.0 - q_stage(A, m, p ** h)
    return out


def mc_etc(H: int, A: int, m: int, N: int, rng: np.random.Generator, p: float = 1.0) -> np.ndarray:
    """Vectorised Monte Carlo of the same algorithm; returns a bool array (N,) 'final policy optimal'.  Returns come from the lock
    (a round pays 1 iff the guide reached s*_h, a_h = a*_h and every later chosen action is correct)."""
    pol = np.zeros((N, H), np.int64)
    star = np.array([astar(h, A) for h in range(H)])
    for h in range(H - 1, -1, -1):
        reached = rng.random((N, m)) < p ** h
        a = rng.integers(0, A, (N, m))
        later_ok = (pol[:, h + 1:] == star[h + 1:]).all(axis=1) if h < H - 1 else np.ones(N, bool)
        r = reached & (a == star[h]) & later_ok[:, None]
        mean = np.zeros((N, A))
        for act in range(A):
            sel = reached & (a == act)
            n = sel.sum(1)
            mean[:, act] = np.where(n > 0, (r & sel).sum(1) / np.maximum(n, 1), 0.0)
        noise = rng.random((N, A)) * 1e-9                     # uniform tie-break (means differ by >= 1/m >> 1e-9)
        pol[:, h] = np.argmax(mean + noise, axis=1)
    return (pol == star).all(axis=1)


def scratch_first_reward(H: int, A: int, n_learners: int, rng: np.random.Generator, max_episodes: int = 10 ** 8) -> np.ndarray:
    """0-initialised eps-greedy: every Q stays 0 until the first reward, so every action is uniform (ties broken uniformly), whatever eps.
    Episodes until the first reward for n_learners independent learners (explicit episode simulation)."""
    star = np.array([astar(h, A) for h in range(H)])
    out = np.zeros(n_learners, np.int64)
    alive = np.ones(n_learners, bool)
    ep = 0
    while alive.any() and ep < max_episodes:
        k = int(alive.sum())
        a = rng.integers(0, A, (k, H))
        got = (a == star).all(axis=1)
        ep += 1
        idx = np.nonzero(alive)[0]
        out[idx[got]] = ep
        alive[idx[got]] = False
    return out


# ------------------------------------------------------------------------------------------------ Algorithm 2 literally
def alg2_eps_greedy(H: int, A: int, T: int, N: int, rng: np.random.Generator, p: float = 1.0, c_eps: float = 1.0, eps_exp: float = 1.0 / 3.0):
    """The paper's Algorithm 2 (Appendix A.5.3) as written, on the lock: iterations h = H-1, ..., 0; at each, ceil(T/H) rounds of an eps-greedy
    contextual-bandit ExplorationOracle (Corollary A.4's oracle: eps_t = min(1, (c_eps A / t)^(1/3)), greedy on the empirical mean reward of the
    lock state s*_h, ties uniform, untried actions at 0); at round t the policy rolled out is guide for steps < h, pi_t at step h, the ALREADY FIXED
    mixtures pi_l (l > h) after; the CB reward is the return from step h (1 iff the suffix is correct); the guide reaches s*_h w.p. p^h (p = 1:
    C = 1).  Line 5: pi_h = Unif({pi_t}), i.e. a round index drawn uniformly at every visit; its probability of the correct action at s*_h is the
    mean over rounds of pi_t's probability (computed EXACTLY from the oracle's state, not by sampling).
    Returns (subopt (N,) = V* - V^pi = 1 - prod_l q_l, layer_sum (N,) = sum_l (1 - q_l) (the Theorem A.3 sum, an upper bound on subopt), q (N, H)).
    Same-seed streams are vectorised over N independent learners.
    eps_exp (default 1/3 = the declared oracle, bit-identical to the original code): the exponent of the oracle's schedule,
    eps_t = min(1, (c_eps A / t)^eps_exp); the control in scripts/alg2_control.py uses 1/2 to show the fitted T-slope tracks the oracle."""
    R = -(-T // H)
    star = np.array([astar(h, A) for h in range(H)])
    q = np.ones((N, H))
    idx = np.arange(N)
    for h in range(H - 1, -1, -1):
        Qsuf = q[:, h + 1:].prod(axis=1) if h < H - 1 else np.ones(N)          # P(the fixed suffix mixture is correct from s*_{h+1})
        n = np.zeros((N, A)); sm = np.zeros((N, A)); pc_sum = np.zeros(N)
        reach = p ** h
        for t in range(1, R + 1):
            eps = min(1.0, (c_eps * A / t) ** eps_exp)
            mean = np.where(n > 0, sm / np.maximum(n, 1), 0.0)
            ties = mean >= mean.max(axis=1, keepdims=True) - 1e-12
            pc_sum += eps / A + (1.0 - eps) * ties[:, star[h]] / ties.sum(axis=1)
            greedy = np.argmax(np.where(ties, rng.random((N, A)), -1.0), axis=1)   # uniform among the tied maximisers
            a = np.where(rng.random(N) < eps, rng.integers(0, A, N), greedy)
            reached = rng.random(N) < reach
            r = reached & (a == star[h]) & (rng.random(N) < Qsuf)
            n[idx, a] += reached; sm[idx, a] += r
        q[:, h] = pc_sum / R
    return 1.0 - q.prod(axis=1), (1.0 - q).sum(axis=1), q


def alg2_reference(H: int, A: int, T: int, rng: np.random.Generator, p: float = 1.0, c_eps: float = 1.0):
    """Plain-python single-learner reference of `alg2_eps_greedy` (tests only): explicit per-action lists, real lock rollouts with sampled round
    indices for the suffix (no closed-form product), the final policy's V estimated by 2,000 rollouts of the mixture.  Returns the MC V-hat."""
    R = -(-T // H)
    star = [astar(h, A) for h in range(H)]
    probs = [[] for _ in range(H)]            # probs[h][t] = pi_t's P(correct action at s*_h)
    for h in range(H - 1, -1, -1):
        rew = [[] for _ in range(A)]
        for t in range(1, R + 1):
            eps = min(1.0, (c_eps * A / t) ** (1.0 / 3.0))
            mean = [float(np.mean(x)) if x else 0.0 for x in rew]
            best = max(mean); tied = [a for a in range(A) if mean[a] >= best - 1e-12]
            probs[h].append(eps / A + (1 - eps) * (star[h] in tied) / len(tied))
            a = int(rng.integers(0, A)) if rng.random() < eps else int(rng.choice(tied))
            # real rollout: guide to layer h (reaches s*_h w.p. p^h), pi_t's action a, then the mixture at every later layer
            if rng.random() < p ** h:
                red = (a == star[h])
                for l in range(h + 1, H):
                    tt = int(rng.integers(0, R))
                    al = star[l] if rng.random() < probs[l][tt] else (star[l] + 1) % A
                    red = red and (al == star[l])
                rew[a].append(float(red))
    ok = 0
    for _ in range(2000):                      # V of the final mixture policy by rollout
        red = True
        for l in range(H):
            tt = int(rng.integers(0, R))
            red = red and (rng.random() < probs[l][tt])
        ok += red
    return ok / 2000


# ------------------------------------------------------------------------------------------------ jittable env for the real code
class LockEnv:
    """Lock with the MinAtar interface (src/envs.py): obs bool (10, 10, 2): channel 0 one-hot marks the cell (layer // 10, layer % 10), channel 1
    marks 'red' in the same cell.  Episode = H steps (terminal after the last step; the cap is never hit).  H <= 100."""

    def __init__(self, H: int, A: int = 2):
        assert H <= 100
        self.H, self.A, self.game = H, A, "Lock"
        self.obs_shape, self.n_actions, self.max_steps = (10, 10, 2), A, H
        self.asterix_spawn_repair = False
        self.star = jnp.array([astar(h, A) for h in range(H)], jnp.int32)

    def _obs(self, layer, red):
        l = jnp.minimum(layer, 99)
        o = jnp.zeros((10, 10, 2), bool)
        o = o.at[l // 10, l % 10, 0].set(True)
        return o.at[l // 10, l % 10, 1].set(red)

    def reset(self, key):
        return self._obs(jnp.zeros((), jnp.int32), jnp.array(True)), (jnp.zeros((), jnp.int32), jnp.array(True))

    def step(self, key, state, action):
        layer, red = state
        ok = red & (action == self.star[jnp.minimum(layer, self.H - 1)])
        layer2 = layer + 1
        term = layer2 >= self.H
        r = jnp.where(term & ok, 1.0, 0.0).astype(jnp.float32)
        return self._obs(layer2, ok), (layer2, ok), r, term, jnp.zeros((), bool)

    # guide: a*_h at red layers except the last layer (wrong there), action 0 in blue (the guide sees the observation only)
    def guide_fn(self, obs, key):
        layer = jnp.argmax(obs[:, :, 0].reshape(-1)).astype(jnp.int32)
        red = obs[layer // 10, layer % 10, 1]
        a_star = self.star[jnp.minimum(layer, self.H - 1)]
        a = jnp.where(layer == self.H - 1, (a_star + 1) % self.A, a_star)
        return jnp.where(red, a, 0).astype(jnp.int32)

    def decode(self, obs_bool_batch):
        """(B, 10, 10, 2) bool -> (layer (B,), red (B,))."""
        flat0 = obs_bool_batch[..., 0].reshape(obs_bool_batch.shape[0], -1)
        flat1 = obs_bool_batch[..., 1].reshape(obs_bool_batch.shape[0], -1)
        layer = jnp.argmax(flat0, axis=1)
        red = jnp.take_along_axis(flat1, layer[:, None], axis=1)[:, 0]
        return layer, red

"""Environments ("data") for the HCA reproduction. The paper has no dataset: every task is a tiny tabular
MDP/POMDP defined in code, so this module holds the env definitions, the episode sampler and the env checksum.

Reward convention (A-ENV-*): a reward attached to a *state* is collected at the step the
agent is in that state, ``X_k = y => R_k = r(y) + noise``. Terminal states are visited for one reward-only step
(no action, no update source, A-TERMINAL). Every env is acyclic, so an episode has at most ``L`` steps.

An env is a ``TabEnv`` of numpy arrays (converted to jnp inside jitted code):
  P[s, a, s']   transition probabilities (rows of terminal states are ignored)
  r_mean[s]     mean state reward, r_std[s] its Gaussian noise std
  obs[s]        observation id (aliasing = several states sharing an id)
  terminal[s]   reward-only terminal state
  s0, L         start state, max steps per episode (incl. the terminal step)
"""
from __future__ import annotations

import hashlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

N_ACTIONS = 2  # every task has two actions per state (paper §5, Fig. 2; A-DE-ACT)


class TabEnv(NamedTuple):
    P: np.ndarray
    r_mean: np.ndarray
    r_std: np.ndarray
    obs: np.ndarray
    terminal: np.ndarray
    s0: int
    L: int
    n_obs: int
    name: str


# ---------------------------------------------------------------- env constructors
def shortcut(n: int = 5, absorb: float = 0.1, step_reward: float = -1.0, goal_reward: float = 1.0) -> TabEnv:
    """Shortcut chain (paper §5, Fig. 2 left; A-ENV-SC). States s_0..s_{n-1}, G = n, D = n+1.
    Action 0 = short (-> G), action 1 = long (-> s_{k+1}; s_{n-1} long -> G). Every transition is redirected
    to D with prob ``absorb``. Reward -1 at each decision state, +1 at G, 0 at D."""
    S = n + 2
    G, D = n, n + 1
    P = np.zeros((S, N_ACTIONS, S))
    for k in range(n):
        P[k, 0, G] += 1 - absorb
        P[k, 1, (k + 1) if k < n - 1 else G] += 1 - absorb
        P[k, :, D] += absorb
    P[G, :, G] = P[D, :, D] = 1.0  # unused (terminal)
    r_mean = np.array([step_reward] * n + [goal_reward, 0.0])
    terminal = np.array([False] * n + [True, True])
    return TabEnv(P, r_mean, np.zeros(S), np.arange(S), terminal, 0, n + 1, S, f"shortcut_n{n}")


def delayed_effect(n: int = 3, sigma: float = 0.0, aliasing: str = "per_position") -> TabEnv:
    """Delayed effect (paper §5, Fig. 2 centre; A-DE-*). x0 = 0; branch b in {+,-}: chain states
    1+b*n+(j-1), j=1..n; F+ = 2n+1 (+1), F- = 2n+2 (-1). Action 0 at x0 -> + branch. Chain actions both
    advance (irrelevant). Chain rewards N(0, sigma^2). Observations: x0 -> 0, chain (b, j) -> j
    (per_position) or 1 (shared), F+ -> n+1, F- -> n+2."""
    S = 2 * n + 3
    Fp, Fm = 2 * n + 1, 2 * n + 2
    P = np.zeros((S, N_ACTIONS, S))
    obs = np.zeros(S, dtype=np.int64)
    r_mean = np.zeros(S)
    r_std = np.zeros(S)
    for b in range(2):
        first = 1 + b * n
        P[0, b, first] = 1.0
        for j in range(1, n + 1):
            s = first + j - 1
            nxt = s + 1 if j < n else (Fp if b == 0 else Fm)
            P[s, :, nxt] = 1.0
            obs[s] = j if aliasing == "per_position" else 1
            r_std[s] = sigma
    obs[Fp], obs[Fm] = n + 1, n + 2
    P[Fp, :, Fp] = P[Fm, :, Fm] = 1.0
    r_mean[Fp], r_mean[Fm] = 1.0, -1.0
    terminal = np.zeros(S, dtype=bool)
    terminal[[Fp, Fm]] = True
    return TabEnv(P, r_mean, r_std, obs, terminal, 0, n + 2, n + 3, f"delayed_effect_n{n}_s{sigma}_{aliasing}")


def ambiguous_bandit(eps: float = 0.1, sigma: float = 1.5, hidden: bool = False,
                     means: tuple = (1.0, 2.0)) -> TabEnv:
    """Ambiguous bandit (paper §5, Fig. 2 right; A-AB-*). x0 = 0, arm states s1 = 1, s2 = 2 (terminal).
    Action i -> s_{i+1} w.p. 1-eps, the other arm w.p. eps. Reward in s_i ~ N(means[i], sigma^2) at step 1.
    Hidden: both arm states share one observation."""
    P = np.zeros((3, N_ACTIONS, 3))
    P[0, 0, 1], P[0, 0, 2] = 1 - eps, eps
    P[0, 1, 2], P[0, 1, 1] = 1 - eps, eps
    P[1, :, 1] = P[2, :, 2] = 1.0
    obs = np.array([0, 1, 1]) if hidden else np.array([0, 1, 2])
    return TabEnv(P, np.array([0.0, *means]), np.array([0.0, sigma, sigma]), obs,
                  np.array([False, True, True]), 0, 2, int(obs.max()) + 1,
                  f"bandit_e{eps}_s{sigma}_{'hidden' if hidden else 'observed'}")


def thm7_counterexample(c: float = 0.5) -> TabEnv:
    """Theorem 7 counterexample: x --a--> y --> z ; x --b--> w --> y --> z. States x=0, y=1, w=2, z=3 (terminal)."""
    P = np.zeros((4, N_ACTIONS, 4))
    P[0, 0, 1] = P[0, 1, 2] = 1.0
    P[2, :, 1] = P[1, :, 3] = P[3, :, 3] = 1.0
    return TabEnv(P, np.array([0.0, 0.0, c, 1.0]), np.zeros(4), np.arange(4),
                  np.array([False, False, False, True]), 0, 4, 4, "thm7_counterexample")


def env_sha256(env: TabEnv) -> str:
    """sha256 of the canonical float64 bytes of the env definition."""
    h = hashlib.sha256()
    for a in (env.P, env.r_mean, env.r_std):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    for a in (env.obs, env.terminal):
        h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
    h.update(np.array([env.s0, env.L, env.n_obs], dtype=np.int64).tobytes())
    return h.hexdigest()


def noise_free_return_range(env: TabEnv) -> tuple[float, float, float]:
    """(min, max) undiscounted return from s0 with mean rewards over all reachable paths, and the largest
    summed reward-noise std along any path (A-BINS)."""
    lo, hi, sd = np.inf, -np.inf, 0.0

    def rec(s, z, var, depth):
        nonlocal lo, hi, sd
        z, var = z + env.r_mean[s], var + env.r_std[s] ** 2
        if env.terminal[s] or depth == env.L - 1:
            lo, hi, sd = min(lo, z), max(hi, z), max(sd, np.sqrt(var))
            return
        for s2 in np.nonzero(env.P[s].sum(0) > 0)[0]:
            rec(int(s2), z, var, depth + 1)

    rec(env.s0, 0.0, 0.0, 0)
    return float(lo), float(hi), float(sd)


def bin_centres(env: TabEnv, n_bins: int, widen: float = 2.0) -> np.ndarray:
    """A-BINS: centres linspace(z_min0 - widen*sd, z_max0 + widen*sd, n_bins); nearest-centre assignment
    (official code). ``widen=0`` is the noise-free-range sensitivity."""
    lo, hi, sd = noise_free_return_range(env)
    return np.linspace(lo - widen * sd, hi + widen * sd, n_bins)


# ---------------------------------------------------------------- episode sampler (pure, jittable)
class Episode(NamedTuple):
    state: jnp.ndarray     # [L] true state ids (padded with 0)
    obs: jnp.ndarray       # [L] observation ids
    action: jnp.ndarray    # [L] action (0 at terminal / padding)
    reward: jnp.ndarray    # [L] collected reward (0 at padding)
    valid: jnp.ndarray     # [L] step exists
    decision: jnp.ndarray  # [L] step is a non-terminal state with an action (an update source)


def env_arrays(env: TabEnv) -> dict:
    return dict(P=jnp.asarray(env.P), r_mean=jnp.asarray(env.r_mean), r_std=jnp.asarray(env.r_std),
                obs=jnp.asarray(env.obs), terminal=jnp.asarray(env.terminal))


def sample_episode(ea: dict, s0: int, L: int, probs_fn, key) -> Episode:
    """Roll one episode. ``probs_fn(obs_id) -> [A]`` action probabilities. Common random numbers: the step-t
    uniforms/normals depend only on (key, t), not on the policy or the state (paired design).
    Key use: fold_in(key, 0) env transitions, fold_in(key, 1) reward noise, fold_in(key, 2) actions."""
    k_env, k_noise, k_act = (jax.random.fold_in(key, i) for i in range(3))
    u_env = jax.random.uniform(k_env, (L,), dtype=jnp.float64)
    z_noise = jax.random.normal(k_noise, (L,), dtype=jnp.float64)
    u_act = jax.random.uniform(k_act, (L,), dtype=jnp.float64)

    def step(carry, t):
        s, done = carry
        valid = ~done
        o = ea["obs"][s]
        term = ea["terminal"][s]
        r = jnp.where(valid, ea["r_mean"][s] + ea["r_std"][s] * z_noise[t], 0.0)
        p = probs_fn(o)
        a = jnp.minimum(jnp.sum(u_act[t] > jnp.cumsum(p)[:-1]), p.shape[0] - 1)  # inverse CDF
        ps = ea["P"][s, a]
        s2 = jnp.minimum(jnp.sum(u_env[t] > jnp.cumsum(ps)[:-1]), ps.shape[0] - 1)
        decision = valid & ~term
        out = (jnp.where(valid, s, 0), jnp.where(valid, o, 0), jnp.where(decision, a, 0), r, valid, decision)
        return (jnp.where(decision, s2, s), done | term), out

    _, (s, o, a, r, v, d) = jax.lax.scan(step, (jnp.asarray(s0), jnp.asarray(False)), jnp.arange(L))
    return Episode(s, o, a, r, v, d)

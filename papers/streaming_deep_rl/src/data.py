"""Environments / data streams. There is no training dataset in this paper except ETTm2 (E8); the "data" are the
agent's own streams. Every env here is a pure (reset, step) pair with NO auto-reset and NO time-limit wrapper
of its own, so the agent loop sees the true terminal observation (the official wrappers normalize both the
terminal obs and the following reset obs, D10).

* MinAtar (E3/E9): gymnax 1.0.0 ``<Game>-MinAtar`` ports (JAX) -- a PROXY for MinAtar 1.0.15 ():
  - minimal action sets (gymnax default ``use_minimal_action_set=True``; MinAtar v1 registration),
  - sticky actions p = 0.1 handled by the agent loop (``sticky``): MinAtar 1.0.15 environment.act draws the coin
    before every act, repeats the previously EXECUTED action, last_action starts at 0 and is never reset,
  - time limit lifted (MinAtar v1 has none): ``max_steps_in_episode = 1e8``.
* Brax hopper (E6/E7/E2, PROXY): raw Brax env, ``positional`` backend (declared
  before any learning run; every config sets it; the PPO comparison run, whose own code hard-codes ``generalized``, was
  redirected to ``positional``), our own 1000-step truncation (gym-v4 TimeLimit); time
  feature (official AddTimeInfo) is added by the agent loop.
* ETTm2 (E8): official ETTEnvironment semantics: obs = the 7 columns of row t (incl. the CURRENT OT), cumulant =
  min-max-normalized OT of row t+1, episode = one pass, done at t = len-1. Paper §4.5 instead says the observation
  holds the 6 load features plus the OT of the PREVIOUS step ; we follow the code that produced
  Fig. 10. In train.run_td the transition S_k -> S_{k+1} (S_k built from rows <= k, incl. OT_k) carries cumulant
  OT_{k+2}, so v(S_k) targets OT_{k+2} + gamma OT_{k+3} + ... -- identical to the official stream (same-init
  differential).
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

MINATAR_GAMES = ("Breakout", "SpaceInvaders", "Asterix", "Seaquest")
STICKY_P = 0.1                     # official code (MinAtar 1.0.15 default sticky_action_prob)
NO_TIME_LIMIT = 100_000_000        # ASSUMED (MinAtar v1 has no time limit; gymnax default 1000 lifted)


class Env(NamedTuple):
    reset: Callable          # key -> (state, obs)
    step: Callable           # (key, state, action) -> (state, obs, reward, terminated)
    obs_shape: tuple
    n_actions: int           # discrete: |A|; continuous: action dim
    discrete: bool
    name: str


def minatar_env(game: str) -> Env:
    import gymnax
    env, params = gymnax.make(f"{game}-MinAtar")
    params = params.replace(max_steps_in_episode=NO_TIME_LIMIT)

    def reset(key):
        obs, st = env.reset_env(key, params)
        return st, obs.astype(jnp.float32)

    def step(key, st, a):
        obs, st2, r, done, _ = env.step_env(key, st, a, params)
        return st2, obs.astype(jnp.float32), jnp.asarray(r, jnp.float32), jnp.asarray(done, bool)

    obs_shape = tuple(env.observation_space(params).shape)
    return Env(reset, step, obs_shape, int(env.num_actions), True, f"{game}-MinAtar-gymnax-sticky")


def sticky(key, last_exec, a, p=STICKY_P):
    """MinAtar 1.0.15 Environment.act: with prob p execute the previously executed action instead of a."""
    rep = jax.random.uniform(key) < p
    return jnp.where(rep, last_exec, a)


BRAX_BACKEND = "positional"    #  (all configs set it explicitly; was a stale "generalized")
# Backends that FAILED the passive-policy proxy gate (scripts/hopper_passive_gate.py).
# positional: zero torque stands for all 1000 steps (return 1013) where Hopper-v4 falls at 155 steps.
PROXY_GATE_FAILED = frozenset({"positional"})
EPISODE_LIMIT = 1000           # gym-v4 max_episode_steps (official AddTimeInfo time_limit)


def brax_env(name: str = "hopper", backend: str = BRAX_BACKEND) -> Env:
    from brax import envs as benvs
    env = benvs.get_environment(name, backend=backend)

    def canon(s):
        return s.replace(reward=jnp.asarray(s.reward, jnp.float32), done=jnp.asarray(s.done, jnp.float32))

    def reset(key):
        s = canon(env.reset(key))
        return s, s.obs

    def step(key, s, a):
        s2 = canon(env.step(s, a))
        return s2, s2.obs, s2.reward, s2.done > 0.5

    return Env(reset, step, (env.observation_size,), env.action_size, False, f"brax-{name}-{backend}")


# ------------------------------------------------------------------ ETTm2 (E8)
ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTm2.csv"
# downloaded on first use; set ETT_CSV to use another copy
ETT_CACHE = os.environ.get("ETT_CSV", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   "external", "data", "ETTm2.csv"))
ETT_SHA256 = "db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd"  # fetched 2026-10-15, 69 680 rows


def load_ett(path: str = ETT_CACHE) -> dict[str, Any]:
    """Returns obs (N, 7) float64 raw columns HUFL..OT, cumulant (N,) min-max normalized OT, provenance."""
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(ETT_URL, path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    raw = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=range(1, 8), dtype=np.float64)
    ot = raw[:, -1]
    cum = (ot - ot.min()) / (ot.max() - ot.min())
    return dict(obs=raw, cumulant=cum, sha256=sha, rows=int(raw.shape[0]), sha_ok=sha == ETT_SHA256)

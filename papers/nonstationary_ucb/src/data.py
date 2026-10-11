"""Piecewise-stationary and periodic Bernoulli environments (there is no dataset in this paper).

An environment is a closed-form mean schedule p_t(i), t = 1..T (1-indexed rounds), stored as a
float64 array ``means[t-1, i]``. Rewards are X_t(i) = 1{u_t(i) < p_t(i)} with u drawn inside the scan from
the environment PRNG branch (common random numbers across policies, paper footnote 1).

Environments (README, sources.yaml):
  * ``abrupt``   : piecewise-constant segments separated by breakpoints (E1, E1-text, E1b, X1 family).
  * ``periodic`` : p_t(1) = base + amp cos(2 pi periods t / T), p_t(2) = base (E2, E2b).
  * ``constant`` : stationary means (Toy-check degenerate toy G1-d uses means (1, 0)).
"""
from __future__ import annotations

import hashlib
import math
from typing import NamedTuple, Sequence

import numpy as np


class Env(NamedTuple):
    name: str
    K: int
    T: int
    means: np.ndarray        # (T, K) float64, row t-1 = round t
    resets: np.ndarray       # (T,) bool, True at round t if the means changed between t-1 and t (abrupt only)
    breakpoints: tuple       # rounds at which a new segment starts (abrupt only; () otherwise)


def abrupt(T: int, segments: Sequence[Sequence[float]], breakpoints: Sequence[int], name: str = "abrupt") -> Env:
    """Segment j covers rounds [bp_{j-1}, bp_j) with bp_0 = 1 and bp_last = T + 1 (paper §5: '3000 <= t < 5000')."""
    segments = np.asarray(segments, dtype=np.float64)
    bps = list(breakpoints)
    if len(segments) != len(bps) + 1:
        raise ValueError(f"need len(segments) == len(breakpoints)+1, got {len(segments)} and {len(bps)}")
    if bps != sorted(bps) or (bps and (bps[0] < 2 or bps[-1] > T)):
        raise ValueError(f"breakpoints must be increasing in [2, T], got {bps}")
    t = np.arange(1, T + 1)
    seg = np.searchsorted(np.asarray(bps), t, side="right")  # number of breakpoints <= t
    means = segments[seg]
    resets = np.zeros(T, dtype=bool)
    for b in bps:
        resets[b - 1] = True
    return Env(name, segments.shape[1], T, means, resets, tuple(bps))


def periodic(T: int, base: float = 0.5, amp: float = 0.4, periods: float = 3.0, name: str = "periodic") -> Env:
    """E2: p_t(1) = 0.5 + 0.4 cos(6 pi t / T) (3 periods), p_t(2) = 0.5 (paper §5)."""
    t = np.arange(1, T + 1, dtype=np.float64)
    p1 = base + amp * np.cos(2.0 * math.pi * periods * t / T)
    means = np.stack([p1, np.full(T, base)], axis=1)
    return Env(name, 2, T, means, np.zeros(T, dtype=bool), ())


def constant(T: int, mu: Sequence[float], name: str = "constant") -> Env:
    means = np.tile(np.asarray(mu, dtype=np.float64)[None, :], (T, 1))
    return Env(name, len(mu), T, means, np.zeros(T, dtype=bool), ())


def evenly_spaced_breakpoints(T: int, upsilon: int) -> list[int]:
    """Upsilon breakpoints at round(k T / (Upsilon + 1)), k = 1..Upsilon."""
    return [int(round(k * T / (upsilon + 1))) for k in range(1, upsilon + 1)]


def alternating(T: int, upsilon: int, base: Sequence[float], swapped: Sequence[float], name: str = "x1") -> Env:
    """X1 family: segments alternate base, swapped, base, ... over Upsilon evenly spaced breakpoints."""
    segs = [base if j % 2 == 0 else swapped for j in range(upsilon + 1)]
    return abrupt(T, segs, evenly_spaced_breakpoints(T, upsilon), name=name)


def from_config(env_cfg: dict, name: str) -> Env:
    """Build an environment from a config's ``env`` block (see src/configs/*.yaml)."""
    kind = env_cfg["kind"]
    T = int(env_cfg["T"])
    if kind == "abrupt":
        return abrupt(T, env_cfg["segments"], env_cfg["breakpoints"], name=name)
    if kind == "periodic":
        return periodic(T, env_cfg.get("base", 0.5), env_cfg.get("amp", 0.4), env_cfg.get("periods", 3.0), name=name)
    if kind == "constant":
        return constant(T, env_cfg["mu"], name=name)
    if kind == "alternating":
        return alternating(T, int(env_cfg["upsilon"]), env_cfg["base"], env_cfg["swapped"], name=name)
    raise ValueError(f"unknown env kind {kind!r}")


# ---------------------------------------------------------------- self-tests and fingerprints
def self_test(env: Env) -> dict:
    """Oracle quantities: best-arm counts, number of best-arm switches, regret of every constant-arm policy."""
    mu = env.means
    best = mu.max(1)
    star = mu.argmax(1)
    ties = int((np.sort(mu, 1)[:, -1] == np.sort(mu, 1)[:, -2]).sum()) if env.K > 1 else 0
    return {
        "best_arm_counts": np.bincount(star, minlength=env.K).tolist(),
        "best_arm_switches": int((star[1:] != star[:-1]).sum()),
        "n_mean_changes": int((np.abs(np.diff(mu, axis=0)).max(1) > 0).sum()),
        "constant_arm_regret": (best[:, None] - mu).sum(0).tolist(),
        "n_tied_steps": ties,
        "breakpoints": list(env.breakpoints),
    }


def means_sha256(env: Env) -> str:
    return hashlib.sha256(np.ascontiguousarray(env.means, dtype=np.float64).tobytes()).hexdigest()

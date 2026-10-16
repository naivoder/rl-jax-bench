"""Data for the statistics: normalisation constants (from sources.yaml, never retyped), the Dopamine ALE per-run
scores (the only public partial mirror of the withdrawn gs://rl-benchmark-data, the README, the E6 own-compute pool
(runs/<tag>/<env>/<arm>/seed_<n>/metrics.json), and synthetic generators with known ground truth for toy checks.

This paper has no training dataset; every analysis is on final per-run scores. numpy only (no jax import), so the
loaders can be used both by the float64 statistics processes and by report scripts.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = os.path.join(REPO, "src", "configs", "sources.yaml")
DATA = os.environ.get("DATA_ROOT", os.path.join(REPO, "external", "data"))  # large inputs, see README

# Dopamine baselines (paper A.1 names Dopamine as the source of DQN / DQN(Adam) / C51 / IQN / Rainbow); pinned commit.
DOPAMINE_COMMIT = "5873f5494ee0c2d7c016d0ab2ad530354fec59d0"  # github.com/google/dopamine master on 2026-10-16
DOPAMINE_URL = "https://raw.githubusercontent.com/google/dopamine/{sha}/baselines/atari/data/{game}.json"
# : Dopamine agent label -> the paper's Fig 9 name
DOPAMINE_AGENTS = {"DQN": "DQN (Nature)", "DQN (Adam + MSE in JAX)": "DQN (Adam)", "C51": "C51", "IQN": "IQN",
                   "Rainbow": "Rainbow"}


def sources() -> dict:
    with open(SOURCES) as f:
        return yaml.safe_load(f)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ normalisation
def normalise(raw: np.ndarray, games: list[str], low: dict, high: dict) -> np.ndarray:
    """(score - low) / (high - low) per task column (paper §2). raw (..., M) with columns in `games` order."""
    lo = np.array([low[g] for g in games], dtype=np.float64)
    hi = np.array([high[g] for g in games], dtype=np.float64)
    return (np.asarray(raw, np.float64) - lo) / (hi - lo)


def ale_games() -> list[str]:
    """The paper's 55 ALE games = the official human table keys (Surround/Defender removed; R13)."""
    return list(sources()["normalisation"]["ale_human"]["value"].keys())


# ------------------------------------------------------------------------------------------------ Dopamine ALE
def fetch_dopamine(out_dir: str | None = None, games: list[str] | None = None) -> dict:
    """Download baselines/atari/data/<game>.json at the pinned commit; returns {game: {path, sha256, url}}."""
    out_dir = out_dir or os.path.join(DATA, "dopamine")
    os.makedirs(out_dir, exist_ok=True)
    prov = {}
    for g in games or ale_games():
        name = g.lower()
        url = DOPAMINE_URL.format(sha=DOPAMINE_COMMIT, game=name)
        path = os.path.join(out_dir, f"{name}.json")
        if not os.path.exists(path):
            with urllib.request.urlopen(url, timeout=60) as r, open(path + ".tmp", "wb") as f:
                f.write(r.read())
            os.replace(path + ".tmp", path)
        prov[g] = dict(path=path, sha256=sha256(path), url=url)
    return prov


def parse_dopamine_game(path: str) -> dict[str, np.ndarray]:
    """{agent: (runs, iterations)} from one game file. Run identity: rows of an agent are consecutive
    blocks whose Iteration restarts at 0; every block must be 0..I-1 in order, else ValueError."""
    with open(path) as f:
        rows = json.load(f)
    by = {}
    for r in rows:
        by.setdefault(r["Agent"], []).append((int(r["Iteration"]), float(r["Value"])))
    out = {}
    for agent, seq in by.items():
        runs, cur = [], []
        for it, v in seq:
            if it == 0 and cur:
                runs.append(cur)
                cur = []
            if it != len(cur):
                raise ValueError(f"{path} {agent}: iteration {it} out of order at position {len(cur)}")
            cur.append(v)
        runs.append(cur)
        lens = {len(r) for r in runs}
        if len(lens) != 1:
            raise ValueError(f"{path} {agent}: unequal run lengths {lens}")
        out[agent] = np.array(runs, dtype=np.float64)
    return out


def load_dopamine_ale(final_only: bool = True, normalised: bool = True, fetch: bool = True):
    """Return (scores, games, provenance): scores[paper_agent] = (5 runs, 55 games) final-iteration scores
    (or (5, 55, I) per-iteration curves), human-normalised with the official table."""
    games = ale_games()
    prov = fetch_dopamine(games=games) if fetch else {}
    src = sources()["normalisation"]
    per_game = {g: parse_dopamine_game(os.path.join(DATA, "dopamine", f"{g.lower()}.json")) for g in games}
    scores = {}
    for d_agent, p_agent in DOPAMINE_AGENTS.items():
        cube = np.stack([per_game[g][d_agent] for g in games], axis=1)  # (runs, games, iterations)
        if final_only:
            cube = cube[:, :, -1]
        if normalised:
            lo, hi = src["ale_random"]["value"], src["ale_human"]["value"]
            cube = np.moveaxis(normalise(np.moveaxis(cube, 1, -1), games, lo, hi), -1, 1)  # games axis last, back
        scores[p_agent] = cube
    return scores, games, prov


# ------------------------------------------------------------------------------------------------ E6 pool
E6_BOUNDS = {"CartPole-v1": (0.0, 500.0), "Acrobot-v1": (-500.0, 0.0)}  #  (sources.yaml e6.normalise)


def load_e6_pool(root: str, envs=("CartPole-v1", "Acrobot-v1"), arms=("separate", "shared", "advnorm_batch"),
                 key: str = "eval_sampled_mean", normalised: bool = True):
    """{arm: (seeds, tasks)} from runs/<tag>/<env>/<arm>/seed_<n>/metrics.json (src/train.py). Seeds present for
    every env of an arm are used, in ascending order; returns (pool, seeds_per_arm)."""
    pool, seeds_used = {}, {}
    for arm in arms:
        per_env = {}
        for env in envs:
            d = os.path.join(root, env, arm)
            per_env[env] = {}
            if os.path.isdir(d):
                for s in os.listdir(d):
                    p = os.path.join(d, s, "metrics.json")
                    if s.startswith("seed_") and os.path.exists(p):
                        with open(p) as f:
                            per_env[env][int(s[5:])] = json.load(f)["summary"][key]
        common = sorted(set.intersection(*(set(v) for v in per_env.values()))) if per_env else []
        if not common:
            continue
        raw = np.array([[per_env[e][s] for e in envs] for s in common], dtype=np.float64)
        if normalised:
            raw = np.stack([(raw[:, j] - E6_BOUNDS[e][0]) / (E6_BOUNDS[e][1] - E6_BOUNDS[e][0]) for j, e in enumerate(envs)], 1)
        pool[arm], seeds_used[arm] = raw, common
    return pool, seeds_used


# ------------------------------------------------------------------------------------------------ synthetic
def gaussian_design(rng: np.random.Generator, mu: np.ndarray, sigma: np.ndarray, n: int) -> np.ndarray:
    """x[n, m] = mu_m + sigma_m * eps the README."""
    return mu[None, :] + sigma[None, :] * rng.standard_normal((n, len(mu)))

"""EXP-P95(a): the paper's statistic (conditional 95th percentile + binomial CI, paper §2 footnote 4) on the
authors' own raw study data, and how it behaves when fewer configurations are sampled (README, README).

Data: mujoco123_data.json (sha256 verified first) via the converted pickle (scripts/official/convert_data.py), experiment
`final_stability` (group F, Table 5). Score of a configuration = last entry of `_eval_`.

For each choice in group F (C64 input_normalization, C66 reward(=value) normalization, C67 normalize_advantages,
C68 gradient_clipping, learning_rate) x 5 envs:
  full data: per-value q95 (np.percentile linear), official CI, value ranking.
  sub-samples: N in {32, 64, 128, 256} configurations drawn WITHOUT replacement from the env's rows (all values
  pooled, as a smaller random study would), R = 2000 sub-samples, rng seed fixed. Report
    (i)  P(sub-sample ranks the values exactly as the full data) and P(same best value);
    (ii) empirical coverage of the official CI (ranks lo+1, hi+1, clamped) and of the l/h CI for the FULL-DATA
         q95 of each value, pooled over values and sub-samples, against
    (iii) the exact coverage. Sampling is without replacement from a finite population of n_pop rows per value, so
         K = #{sub-sample <= xi} given n_v is hypergeometric(n_pop, M, n_v), M = #{pop <= xi}. The binomial
         (infinite-population) value of README is reported next to it.
  Judgment: empirical vs exact coverage MATCH if within 3 Monte-Carlo SE (README item 4).

Usage: uv run python scripts/p95_subsample.py  (CPU; writes runs/p95/p95_subsample.json)
"""
import hashlib
import json
import os
import pickle
import sys

import numpy as np
from scipy.stats import binom, hypergeom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval import coverage_exact, p95, percentile_ci  # noqa: E402

import os  # noqa: E402
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official"))  # authors' checkout, see README
DATA = os.path.join(OFFICIAL_ROOT, "data")
SHA = "ce5f423c1d693ca69c00365045287ef9513621f367012c58b2ad201571203d24"
P = "_gin.study_design.choice_value_"
ENVS = ["Hopper-v1", "Humanoid-v1", "Walker2d-v1", "HalfCheetah-v1", "Ant-v1"]
CHOICES = ["input_normalization", "reward_normalization", "normalize_advantages", "gradient_clipping", "learning_rate"]
NS = [32, 64, 128, 256]
R = 2000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "p95")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def ranks(lo_hi_n, n, conv):
    lo, hi = binom.interval(0.95, n, 0.95)
    lo, hi = int(lo), int(hi)
    if conv == "official":
        return min(lo + 1, n), min(hi + 1, n)
    return max(lo, 1), min(hi, n)


def exact_cov_hyper(n_pop, M, n, conv):
    """P(X_(l) <= xi <= X_(h)) when n of n_pop are drawn without replacement and M population values are <= xi
    (xi strictly between order statistics, so X_(l) <= xi <=> K >= l; X_(h) >= xi <=> K <= h-1)."""
    if n == 0:
        return np.nan
    l, h = ranks(None, n, conv)
    rv = hypergeom(n_pop, M, n)
    return float(rv.cdf(h - 1) - rv.cdf(l - 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    got = sha256(os.path.join(DATA, "mujoco123_data.json"))
    assert got == SHA, got
    print(f"[INFO] raw data sha256 verified {got[:12]}…", flush=True)
    D = pickle.load(open(os.path.join(DATA, "mujoco123.pkl"), "rb"))
    name = {m["title"].split(": ")[1].split(" ")[0]: k for k, (df, m) in D.items()}
    df, _ = D[name["final_stability"]]
    df = df.copy()
    df["score"] = df["_eval_"].map(lambda d: d[max(d)])
    rng = np.random.default_rng(20210101)
    res = {"sha256": got, "R": R, "NS": NS, "cells": []}
    agg = {(N, c): {"hit": 0, "tot": 0, "exact_h": 0.0, "exact_b": 0.0} for N in NS for c in ("official", "lh")}
    rank_agg = {N: [] for N in NS}
    best_agg = {N: [] for N in NS}
    for ch in CHOICES:
        col = P + ch
        for env in ENVS:
            e = df[df.env_name == env]
            vals = sorted(e[col].astype(str).unique())
            v_arr = e[col].astype(str).values
            s_arr = e["score"].values.astype(float)
            full = {}
            for v in vals:
                x = s_arr[v_arr == v]
                q = p95(x)
                lo, hi = percentile_ci(x, "official")
                full[v] = {"n": int(len(x)), "q95": q, "ci": [lo, hi], "M": int(np.sum(x <= q))}
            order = sorted(vals, key=lambda v: -full[v]["q95"])
            cell = {"choice": ch, "env": env, "full": full, "order": order, "sub": {}}
            for N in NS:
                same_order, same_best = 0, 0
                hits = {"official": 0, "lh": 0}
                tot = 0
                ex_h = {"official": 0.0, "lh": 0.0}
                ex_b = {"official": 0.0, "lh": 0.0}
                for r in range(R):
                    idx = rng.choice(len(s_arr), size=N, replace=False)
                    sv, ss = v_arr[idx], s_arr[idx]
                    q_sub = {}
                    for v in vals:
                        x = ss[sv == v]
                        n_v = len(x)
                        if n_v == 0:
                            q_sub[v] = -np.inf
                            continue
                        q_sub[v] = p95(x)
                        tot += 1
                        for conv in ("official", "lh"):
                            lo, hi = percentile_ci(x, conv)
                            hits[conv] += int(lo <= full[v]["q95"] <= hi)
                            ex_h[conv] += exact_cov_hyper(full[v]["n"], full[v]["M"], n_v, conv)
                            ex_b[conv] += coverage_exact(n_v, conv)
                    o = sorted(vals, key=lambda v: -q_sub[v])
                    same_order += int(o == order)
                    same_best += int(o[0] == order[0])
                sub = {"p_same_order": same_order / R, "p_same_best": same_best / R, "n_value_draws": tot}
                for conv in ("official", "lh"):
                    emp = hits[conv] / tot
                    sub[f"cov_{conv}"] = emp
                    sub[f"cov_{conv}_mcse"] = float(np.sqrt(emp * (1 - emp) / tot))
                    sub[f"cov_{conv}_exact_hyper"] = ex_h[conv] / tot
                    sub[f"cov_{conv}_exact_binom"] = ex_b[conv] / tot
                    a = agg[(N, conv)]
                    a["hit"] += hits[conv]; a["tot"] += tot; a["exact_h"] += ex_h[conv]; a["exact_b"] += ex_b[conv]
                rank_agg[N].append(sub["p_same_order"])
                best_agg[N].append(sub["p_same_best"])
                cell["sub"][N] = sub
                print(f"[METRIC] p95sub choice={ch} env={env} N={N} k={len(vals)} p_order={sub['p_same_order']:.3f} "
                      f"p_best={sub['p_same_best']:.3f} cov_off={sub['cov_official']:.3f}"
                      f"(exact {sub['cov_official_exact_hyper']:.3f}) cov_lh={sub['cov_lh']:.3f}"
                      f"(exact {sub['cov_lh_exact_hyper']:.3f})", flush=True)
            res["cells"].append(cell)
    res["pooled"] = {}
    for N in NS:
        d = {"p_same_order_mean": float(np.mean(rank_agg[N])), "p_same_best_mean": float(np.mean(best_agg[N]))}
        for conv in ("official", "lh"):
            a = agg[(N, conv)]
            emp = a["hit"] / a["tot"]
            d[conv] = {"emp": emp, "mcse": float(np.sqrt(emp * (1 - emp) / a["tot"])), "exact_hyper": a["exact_h"] / a["tot"],
                       "exact_binom": a["exact_b"] / a["tot"], "draws": a["tot"]}
        res["pooled"][N] = d
        print(f"[METRIC] p95pooled N={N} " + json.dumps(d), flush=True)
    with open(os.path.join(OUT, "p95_subsample.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("[INFO] done", flush=True)


if __name__ == "__main__":
    main()

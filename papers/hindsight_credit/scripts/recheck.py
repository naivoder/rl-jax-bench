"""Recheck re-runs (2026-10-11): the pre-declared sensitivities (see README).

    uv run python -m scripts.recheck [family|pgpaper|e1start|all]
        -> runs/recheck/{family,pgpaper,e1start}.json

family : primary family P1, P3-P10 under primary / S-HLR / S-HFORM / S-HPRIOR, seeds 0, 1, 3 (as before) and 4, 5 (fresh);
         Holm over the 9 claims per (variant, seed). PG at its seed-999-tuned lr (runs/full/pg_tuning.json).
pgpaper: descriptive. PG at policy lr 0.05 and no critic vs the primary HCA arms on P7-P10, plus the rmse of every PG
         setting's mean / std curve against the paper's exact Fig. 5 vector curves.
e1start: E1 with theta0[chain, long] = +1 (V0 = -1.819); HCA with h0 = zero (A-INIT) and h0 = policy (labelled, no paper
         support); PG re-tuned on seed 999 over the App.-E grid under this start.
"""
from __future__ import annotations

import json
import os
import sys
import time

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src  # noqa: E402,F401
import numpy as np  # noqa: E402
from scipy import stats as st  # noqa: E402
from rl_jax_bench.stats import holm  # noqa: E402

from scripts.full import TUNING_SEED, run_arm, tuned_lrs  # noqa: E402
from src.train import load_config, method_lrs, settings  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "recheck")
CFG = load_config()
TL = tuned_lrs()
SEEDS = (0, 1, 3, 4, 5)
FRESH = (4, 5)
CLAIMS = {"P1": ("E1", 0, "state_hca", "final"), "P3": ("E3", 0, "state_hca", "final"),
          "P4": ("E3", 0, "return_hca", "final"), "P5": ("E4", 0, "state_hca", "avg"),
          "P6": ("E5", 5, "state_hca", "avg"), "P7": ("E6", 0, "state_hca", "final"),
          "P8": ("E6", 0, "return_hca", "final"), "P9": ("E7", 0, "return_hca", "final"),
          "P10": ("E8", 0, "return_hca", "avg")}
H = CFG["hca"]
BASE = np.array([H["policy_lr"], H["hindsight_lr"], H["value_lr"], H["reward_model_lr"]])
# variant -> (runner kwargs, lrs for the HCA arm)
VARIANTS = {"primary": ({}, BASE)}
for ah in (0.1, 0.2, 0.8, 1.6):
    VARIANTS[f"hlr{ah}"] = ({}, BASE * np.array([1, ah / H["hindsight_lr"], 1, 1]))
VARIANTS["swapped_a0.4_ah0.3"] = ({}, np.array([0.4, 0.3, H["value_lr"], H["reward_model_lr"]]))
VARIANTS["counts"] = ({"h_est": "counts"}, BASE)
VARIANTS["ema"] = ({"h_est": "ema"}, BASE)
VARIANTS["residual"] = ({"h_param": "residual"}, BASE)


def stat(c, s):
    return c[:, -1] if s == "final" else c.mean(1)


def verdict(d, p_holm):
    if p_holm < 0.05:
        return "REPRODUCED" if d > 0 else "REVERSED"
    return "NOT REPRODUCED (n.s.)"


def compare(x, pg):
    d = x - pg
    return dict(arm=float(x.mean()), arm_se=float(x.std(ddof=1) / np.sqrt(len(x))), pg=float(pg.mean()),
                diff=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), wins=int((d > 0).sum()),
                n=int(len(d)), p=float(st.ttest_1samp(d, 0).pvalue),
                p_wilcoxon=float(st.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)


def pg_stat(cl, seed, lrs=None):
    exp, si, _, s = CLAIMS[cl]
    kw = settings(CFG["experiments"][exp])[si]
    lrs = method_lrs(CFG, "pg", TL[exp][si]) if lrs is None else lrs
    return stat(run_arm(CFG, exp, kw, "pg", seed, 100, lrs, si), s)


def cmd_family():
    res = {"seeds": SEEDS, "fresh_seeds": FRESH, "variants": {k: dict(kwargs=v[0], lrs=v[1].tolist()) for k, v in VARIANTS.items()},
           "claims": CLAIMS, "results": {}}
    t0 = time.time()
    arrays = {}
    for seed in SEEDS:
        pgs = {cl: pg_stat(cl, seed) for cl in CLAIMS}
        for cl, v in pgs.items():
            arrays[f"s{seed}_{cl}_pg"] = v
        for vn, (kw, lrs) in VARIANTS.items():
            rows = {}
            for cl, (exp, si, arm, s) in CLAIMS.items():
                x = stat(run_arm(CFG, exp, settings(CFG["experiments"][exp])[si], arm, seed, 100, lrs, si, **kw), s)
                arrays[f"s{seed}_{cl}_{vn}"] = x
                rows[cl] = compare(x, pgs[cl])
            hp = holm({cl: r["p"] for cl, r in rows.items()})
            for cl, r in rows.items():
                r["p_holm"] = hp[cl]["p_holm"]
                r["verdict"] = verdict(r["diff"], r["p_holm"])
            res["results"].setdefault(vn, {})[seed] = rows
            print(f"seed {seed} {vn:20s} " + " ".join(f"{cl}:{r['diff']:+.3f}{'*' if r['p_holm'] < .05 else ''}"
                                                      for cl, r in rows.items()), f"{time.time() - t0:.0f}s", flush=True)
    res["wall_s"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, "family.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "family_per_run.npz"), **arrays)


def _band(b):
    b = np.array(b); xs = np.round(b[:, 0] * (10 if max(b[:, 0]) < 1 else 1)).astype(int); up, lo = {}, {}
    for x, y in zip(xs, b[:, 1]):
        up[x] = max(up.get(x, -9), y); lo[x] = min(lo.get(x, 9), y)
    k = sorted(up)
    return np.array([up[i] for i in k]), np.array([lo[i] for i in k])


def cmd_pgpaper():
    V = json.load(open(os.path.join(ROOT, "data", "fig5_vector.json")))
    paper_lrs = np.array([0.05, 0.0, 0.0, 0.0])
    res = {"pg_paper_lrs": paper_lrs.tolist(), "note": "descriptive only: lr chosen by matching the paper's PG curve",
           "fit": {}, "claims": {}}
    for exp, panel in (("E6", "left"), ("E7", "middle")):
        pm = np.array(V[panel]["pg_mean"])[:, 1]; u, l = _band(V[panel]["pg_band"]); psd = (u - l) / 2
        kw = CFG["experiments"][exp]["env_kwargs"]
        for name, lrs in [("paper_matched_lr0.05_v0", paper_lrs)] + [(f"grid_lr{lr}", method_lrs(CFG, "pg", lr))
                                                                     for lr in CFG["pg"]["lr_grid"]]:
            c = run_arm(CFG, exp, kw, "pg", 0, 100, lrs, 0)[:, :100]
            m, s = c.mean(0), c.std(0, ddof=1)
            res["fit"][f"{exp}_{name}"] = dict(rmse_mean=float(np.sqrt(np.mean((m - pm) ** 2))),
                                               rmse_sd=float(np.sqrt(np.mean((s - psd[:100]) ** 2))),
                                               final=float(c[:, -1].mean()), paper_final=float(pm[-1]))
            print(exp, name, {k: round(v, 4) for k, v in res["fit"][f"{exp}_{name}"].items()}, flush=True)
    for seed in SEEDS:
        rows = {}
        for cl in ("P7", "P8", "P9", "P10"):
            exp, si, arm, s = CLAIMS[cl]
            kw = settings(CFG["experiments"][exp])[si]
            x = stat(run_arm(CFG, exp, kw, arm, seed, 100, BASE, si), s)
            rows[cl] = compare(x, pg_stat(cl, seed, paper_lrs))
        res["claims"][seed] = rows
        print("seed", seed, {cl: (round(r["diff"], 4), f"{r['p']:.1e}") for cl, r in rows.items()}, flush=True)
    json.dump(res, open(os.path.join(OUT, "pgpaper.json"), "w"), indent=1)


def cmd_e1start():
    ec = CFG["experiments"]["E1"]
    kw = ec["env_kwargs"]
    th = np.zeros((7, 2)); th[:5, 1] = 1.0  # shortcut n=5: chain obs 0..4, action 1 = long; G/D rows unused
    from src.train import make_env
    assert make_env(ec, kw).n_obs == 7
    theta0 = tuple(map(tuple, th))
    scores = {lr: float(run_arm(CFG, "E1", kw, "pg", TUNING_SEED, 100, method_lrs(CFG, "pg", lr), 0,
                                theta0=theta0).mean()) for lr in CFG["pg"]["lr_grid"]}
    best = max(scores, key=scores.get)
    res = {"theta0_chain_long": 1.0, "pg_tuning_seed": TUNING_SEED, "pg_scores": scores, "pg_best_lr": best,
           "paper": {"state_hca": [-0.629, -0.142, -0.096], "return_hca": [-0.817, -0.50, -0.429],
                     "pg": [-0.879, -0.338, -0.179]}, "seeds": {}}
    print("PG tuning under biased start", scores, "best", best, flush=True)
    for seed in SEEDS:
        pgc = run_arm(CFG, "E1", kw, "pg", seed, 100, method_lrs(CFG, "pg", best), 0, theta0=theta0)
        rec = {"pg_curve_pts": [float(pgc[:, e].mean()) for e in (0, 20, 50, 100)]}
        rows = {}
        for h0 in ("zero", "policy"):
            for arm in ("state_hca", "return_hca"):
                c = run_arm(CFG, "E1", kw, arm, seed, 100, BASE, 0, theta0=theta0, h0=h0)
                rec[f"{arm}_h{h0}_curve_pts"] = [float(c[:, e].mean()) for e in (0, 20, 50, 100)]
                if arm == "state_hca":
                    rows[f"P1_h{h0}"] = compare(c[:, -1], pgc[:, -1])
        hp = holm({k: r["p"] for k, r in rows.items()})
        for k, r in rows.items():
            r["p_holm"] = hp[k]["p_holm"]; r["verdict"] = verdict(r["diff"], r["p_holm"])
        rec["P1"] = rows
        res["seeds"][seed] = rec
        print("seed", seed, {k: (round(r["diff"], 4), r["verdict"]) for k, r in rows.items()}, rec["pg_curve_pts"], flush=True)
    json.dump(res, open(os.path.join(OUT, "e1start.json"), "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("family", cmd_family), ("pgpaper", cmd_pgpaper), ("e1start", cmd_e1start)):
        if what in (name, "all"):
            fn()

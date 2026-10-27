"""Three-way table: torch reference (DI-engine loss + original MinAtar) vs the JAX implementation at the same windows.

    python scripts/official/analyze_ref.py   # writes runs/ref/analysis.json   (JAX venv; numpy only)
Windows are on the 200 k-step reference horizon: EARLY = first 25 k steps, MID = 100-115 k, FINAL = last 15 k (185-200 k).
JAX numbers come from the first five Breakout seeds (500 k steps, same bins; the first 200 k steps are the same horizon, nothing is
extrapolated).  Mean +- standard error over seeds (reference n = 3, JAX n = 5)."""
import glob, json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = 5000
W = {"EARLY": (0, 25000), "MID": (100000, 115000), "FINAL": (185000, 200000)}


def ms(x):
    x = np.asarray(x, float)
    return (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan"), len(x))


def ref_windows(path):
    d = json.load(open(path))
    eps = d["episodes"]                         # (end step, return)
    out = {}
    for k, (lo, hi) in W.items():
        xs = [r for (t, r) in eps if lo <= t < hi]
        out[k] = float(np.mean(xs)) if xs else float("nan")
    out["greedy"] = d.get("greedy_final_mean"); out["greedy_pre"] = d.get("greedy_after_pretrain"); out["capped"] = d["n_capped_episodes"]
    return out


def jax_windows(metrics_glob, arm):
    rows = []
    for p in sorted(glob.glob(metrics_glob)):
        c = json.load(open(p))["Breakout/subopt"][arm]
        bs, bc = np.asarray(c["bin_sum"]), np.asarray(c["bin_cnt"])
        r = {}
        for k, (lo, hi) in W.items():
            r[k] = float(bs[lo // BIN:hi // BIN].sum() / max(bc[lo // BIN:hi // BIN].sum(), 1))
        rows.append(r)
    return rows


res = {}
groups = {}
for p in sorted(glob.glob(os.path.join(REPO, "runs/ref/*_s[0-9].json"))):
    name = os.path.basename(p).rsplit("_s", 1)[0]
    groups.setdefault(name, []).append(ref_windows(p))
for name, rows in groups.items():
    res[f"torch_ref/{name}"] = {k: ms([r[k] for r in rows if r[k] is not None and not np.isnan(r[k])]) for k in ("EARLY", "MID", "FINAL", "greedy", "greedy_pre", "capped")}
JAX = {"DQfD (eps 0.01)": ("runs/breakout_first5/seed_*/metrics.json", "DQfD"), "NoMargin": ("runs/breakout_first5/seed_*/metrics.json", "NoMargin"),
       "PDD": ("runs/breakout_first5/seed_*/metrics.json", "PDD"),
       "DQfD eps 0.1 (P1)": ("runs/breakout_probe/seed_*/metrics.json", "DQfD+eps0=0.1+eps1=0.1")}
for name, (g, arm) in JAX.items():
    rows = jax_windows(os.path.join(REPO, g), arm)
    res[f"jax/{name}"] = {k: ms([r[k] for r in rows]) for k in W}
json.dump(res, open(os.path.join(REPO, "runs/ref/analysis.json"), "w"), indent=1)
for k, v in res.items():
    print(f"{k:42s}", "  ".join(f"{m}={a[0]:.3f}+-{a[1]:.3f}(n={a[2]})" for m, a in v.items() if a[2]))

"""Paired comparison of two arms by run (per-run mean squared OSPA): usage paired.py tagA:algoA tagB:algoB
Prints RMS of each, paired t on per-run mean d2 (A - B), p, wins of A, bootstrap CI of the RMS ratio.  Descriptive unless in a pre-declared family."""
import glob, json, os, sys
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
def d2(tag, algo):
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "..", "runs", tag, "seed_*", "metrics.json")), key=lambda p: int(p.split("seed_")[-1].split("/")[0]))]
    return np.array([np.mean(r["algos"][algo]["ospa2"]) for r in recs])
a, b = (s.split(":") for s in sys.argv[1:3])
x, y = d2(*a), d2(*b)
t, p = stats.ttest_rel(x, y)
rng = np.random.default_rng(0)
bs = [np.sqrt(x[i].mean()) / np.sqrt(y[i].mean()) for i in (rng.integers(0, len(x), len(x)) for _ in range(10000))]
print(f"{a[1]} RMS {np.sqrt(x.mean()):.4f}  {b[1]} RMS {np.sqrt(y.mean()):.4f}  t={t:.3f} p={p:.3g}  A better in {(x < y).sum()}/{len(x)}  ratio CI95 {np.percentile(bs, 2.5):.3f}..{np.percentile(bs, 97.5):.3f}")

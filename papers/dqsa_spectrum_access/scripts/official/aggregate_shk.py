"""Aggregate shkrwnd reference runs per arm: mean +- SE over seeds. Usage (repo environment): uv run python scripts/official/aggregate_shk.py"""
import collections, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import analyze_shk as A
import glob
root = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "official")
arms = collections.defaultdict(list)
for p in sorted(glob.glob(os.path.join(root, "shk_*", "trace.npz"))):
    name = os.path.basename(os.path.dirname(p))
    m = re.match(r"shk_(.*?)N(\d+)K(\d+)_s(\d+)", name)
    arms[(m.group(1) or "default_", int(m.group(2)), int(m.group(3)))].append(A.one(p))
def ms(v): v = np.array(v, float); return f"{v.mean():.3f} +- {(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('nan')):.3f}"
print("| arm | N | K | seeds | util first 5k | util 10-20k | util 40-50k | util last 5k | collision last 5k | idle last 5k | strict-Pareto windows (last 5k) | pattern windows | per-seed util last 5k |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for (tag, N, K), rs in sorted(arms.items()):
    pat = [r.get("pattern_E7_windows", r.get("pattern_E6_windows")) for r in rs]
    pat = "-" if pat[0] is None else ms(pat)
    print(f"| {tag or 'default_'} | {N} | {K} | {len(rs)} | {ms([r['util_first5k'] for r in rs])} | {ms([r['util_blocks'][1] for r in rs])} | {ms([r['util_blocks'][4] for r in rs])} | {ms([r['util_last5k'] for r in rs])} | {ms([r['coll_last5k'] for r in rs])} | {ms([r['idle_last5k'] for r in rs])} | {ms([r['pareto_strict_windows'] for r in rs])} | {pat} | {', '.join(f'{r['util_last5k']:.2f}' for r in rs)} |")

"""Budget rule, as declared before the runs (see README).  Reads the monitor records of two runs (E0 and E7, each ONE vmapped group of the
SAME 10 selection seeds 100..109, flown to 40k iterations with sched_iters 10000) and prints GO/NO-GO per candidate budget R.
    python scripts/budget_rule.py runs/small_g10/E0_40k runs/small_g10/E7_40k
Declared rule:
  * per-seed 'reaches strict Pareto at R'  :=  median over the monitor checkpoints at (R-2000, R-1000, R) of the seed's pareto_strict
    (fraction of the 64 MONITOR worlds, world offset >= 100000, never the reporting worlds) >= 0.5
  * budget R is GO for an experiment iff >= 50 % of the 10 seeds reach it (>= 5 of 10)
  * primary budget = the smallest R in {10000, 20000, 40000} that is GO for BOTH E0 and E7; if none, the primary stays 10000 and the shortfall is reported as
    'not recovered by budget up to 40k' (it is still conditional on).
"""
import glob
import json
import os
import sys

import numpy as np

BUDGETS = (10000, 20000, 40000)
THRESH = 0.5


def seed_reads(run_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "seed_*", "metrics.json"))):
        m = json.load(open(f))
        mon = {x["it"]: x["pareto_strict"] for x in m["monitor"]}
        out[m["seed"]] = mon
    return out


def reaches(mon, R):
    pts = [mon.get(R - 2000), mon.get(R - 1000), mon.get(R)]
    if any(p is None for p in pts):
        return None
    return float(np.median(pts)) >= THRESH


res = {}
for d in sys.argv[1:]:
    reads = seed_reads(d)
    exp = os.path.basename(d.rstrip("/")).split("_")[0]
    res[exp] = {R: [reaches(m, R) for m in reads.values()] for R in BUDGETS}
    print(exp, "group seeds", list(reads))
    for R in BUDGETS:
        v = res[exp][R]
        if any(x is None for x in v):
            print(f"  R={R}: monitor records missing")
        else:
            print(f"  R={R}: {sum(v)}/{len(v)} seeds reach strict Pareto -> {'GO' if sum(v) >= 0.5 * len(v) else 'NO-GO'}")
primary = None
for R in BUDGETS:
    ok = all(all(x is not None for x in res[e][R]) and sum(res[e][R]) >= 0.5 * len(res[e][R]) for e in ("E0", "E7") if e in res)
    if ok and all(e in res for e in ("E0", "E7")):
        primary = R
        break
print("PRIMARY BUDGET:", primary if primary else "stays 10000 (not recovered by budget up to 40k; conditional on)")

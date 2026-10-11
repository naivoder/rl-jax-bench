"""review (post-hoc, labelled): per-run distributions at every Figure-1 / slide read-off time.

    uv run python scripts/perrun_curves.py     -> runs/full/perrun_at.npz

The headline arms store only the per-run FINAL regret, so the intermediate read-offs (E1 R(3000), R(5000); E1b R(6000),
R(10000)) have no per-run distribution. This re-runs the headline arms on the SAME keys (seeds 0-4 x 2000; asserted
bit-identical at T to runs/full/<exp>) keeping per-run curves, and stores the per-run regret at each read-off time.
scripts/report.py reads the file to give every verdict cell a per-run ("one realisation") reading next to the
mean-based judgment.
"""
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
import src  # noqa: E402,F401
from src import train  # noqa: E402
from src.common import check_device  # noqa: E402

check_device()
out = {}
for cname, d, times in (("e1", "e1", (3000, 5000)), ("padb_e1", "padb_e1", (3000, 5000)),
                        ("e1b", "e1b", (6000, 10000)), ("padb_e1b", "padb_e1b", (6000, 10000))):
    cfg = yaml.safe_load(open(os.path.join(ROOT, "src", "configs", f"{cname}.yaml")))
    pols = [p for p in cfg["policies"] if p != "oracle_ucb1"]
    recs = train.run_experiment_seeds(cfg, [0, 1, 2, 3, 4], 2000, policies=pols, keep_curves=True)
    grid = np.asarray(recs[0]["t_grid"])
    for p in pols:
        C = np.concatenate([r["_curves"][p] for r in recs])
        ref = np.concatenate([json.load(open(os.path.join(ROOT, f"runs/full/{d}/seed_{s}/metrics.json")))["policies"][p]["final_regret"]
                              for s in range(5)])
        assert np.allclose(C[:, -1], ref, atol=1e-5), f"{cname}/{p}: re-run must reproduce the stored headline arm"
        for t in times:
            out[f"{cname.replace('padb_', '')}|{p}|{t}"] = C[:, int(np.flatnonzero(grid == t)[0])]
        print(cname, p, {t: float(out[f"{cname.replace('padb_', '')}|{p}|{t}"].mean()) for t in times}, flush=True)
np.savez_compressed(os.path.join(ROOT, "runs", "full", "perrun_at.npz"), **out)

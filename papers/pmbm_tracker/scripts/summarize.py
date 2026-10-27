"""Summarise runs/<tag>/seed_*/metrics.json: RMS OSPA (Table II aggregation) +- SE per algorithm, overflow check."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import eval as EV

def load(tag, root=None):
    root = root or os.path.join(os.path.dirname(__file__), "..", "runs")
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, tag, "seed_*", "metrics.json")))]
    return recs

def summarize(tag, root=None, key="ospa2"):
    recs = load(tag, root)
    names = sorted({a for r in recs for a in r["algos"]})
    out = {}
    for a in names:
        d2 = np.array([r["algos"][a][key] for r in recs if a in r["algos"]])
        cell = EV.table_cell(d2, n_boot=2000)
        out[a] = (cell, d2)
    ovf = np.array([r["diag"]["pmbm"]["ovf"] for r in recs if "pmbm" in r["diag"]])
    return recs, out, ovf

if __name__ == "__main__":
    for tag in sys.argv[1:]:
        recs, out, ovf = summarize(tag)
        print(f"== {tag}: n_runs={len(recs)}")
        for a, (cell, d2) in out.items():
            print(f"  {a:12s} {cell}")
        if len(ovf):
            print("  overflow totals", dict(zip(("pois_evict","slot_full","cap_rows_dropped","kbest_ovf","meas_ignored","child_trunc"), ovf.sum(0).tolist())))

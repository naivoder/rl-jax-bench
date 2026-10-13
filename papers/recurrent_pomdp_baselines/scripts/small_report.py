"""Small-stage report: our 4-seed results per cell vs the authors' logs over the same window and the official-code CPU
reference runs; pre-declared PASS criteria (the README); grad-norm diagnostics; per-seed aggregate
runs/seed_<n>/metrics.json.  Writes runs/small/report.json."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_jax_bench.stats import mean_se, paired_compare  # noqa: E402

from src.eval import author_final, final_value, load_run, read_progress  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G2 = os.path.join(REPO, "runs", "small")
OFF = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official")), "repo", "logs", "pomdp", "Pendulum", "V")
CELLS = {
    "P_sac": dict(sep="P_sac_separate", sh="P_sac_shared", window=(16000, 20000),
                  ref_sep="pomdp/Pendulum/V/sac_lstm/len--1/*/*/*/oar/*/progress.csv",
                  ref_sh="pomdp/Pendulum/V/sac_lstm_shared/len--1/*/*/*/oar/*/progress.csv",
                  sep_floor=-360.0, sh_ceiling=-215.0),
    "P_td3": dict(sep="P_td3_separate", sh="P_td3_shared", window=(16000, 20000),
                  ref_sep="pomdp/Pendulum/V/td3_lstm/len--1/*/*/*/oar/*/progress.csv",
                  ref_sh="pomdp/Pendulum/V/td3_lstm_shared/len--1/*/*/*/oar/*/progress.csv",
                  sep_floor=-360.0, sh_ceiling=-770.0),
    "S_td3": dict(sep="S_td3_separate", sh="S_td3_shared", window=(120000, 150000),
                  ref_sep="meta/PointRobotSparse-v0/td3_lstm/len-64/*/*/*/or/*/progress.csv",
                  ref_sh="meta/PointRobotSparse-v0/td3_lstm_shared/len-64/*/*/*/or/*/progress.csv",
                  sep_floor=35.0, sh_ceiling=10.0),
}


def finals(run, variant):
    d = load_run(os.path.join(G2, run)).get(variant, {})
    return {s: final_value(d[s]) for s in sorted(d)}, d


def diag_medians(d, keys):
    out = {}
    for k in keys:
        v = [np.median([t[k] for t in m["train"] if k in t]) for m in d.values() if any(k in t for t in m["train"])]
        if v:
            out[k] = float(np.median(v))
    return out


def official_ref(arch_dir, window):
    out = []
    for f in sorted(glob.glob(os.path.join(OFF, arch_dir, "**", "progress.csv"), recursive=True)):
        c = read_progress(f)
        if len(c) == 0:
            continue
        sel = (c[:, 0] >= window[0]) & (c[:, 0] <= window[1])
        out.append({"file": f, "max_env_steps": float(c[:, 0].max()),
                    "window_mean": float(c[sel, 1].mean()) if sel.any() else None,
                    "curve": c.tolist()})
    return out


def main():
    rep, ok = {}, True
    for cell, c in CELLS.items():
        sep, dsep = finals(c["sep"], "separate")
        r = {"ours_separate": list(sep.values())}
        if not sep:
            rep[cell] = {"missing": True}
            ok = False
            continue
        r["ours_separate_mean_se"] = mean_se(list(sep.values()))
        r["ref_separate"] = author_final(c["ref_sep"], window=c["window"])
        r["ref_separate_mean_se"] = mean_se(r["ref_separate"])
        r["ref_shared"] = author_final(c["ref_sh"], window=c["window"])
        r["ref_shared_mean_se"] = mean_se(r["ref_shared"])
        r["diag_separate"] = diag_medians(dsep, ["off_q_rnn", "off_pi_rnn", "clean_q_rnn", "clean_pi_rnn",
                                                 "clean_pi_into_q_rnn"])
        variants = ["official", "noleak", "detach"] if cell.startswith("P") else ["official"]
        crit = {"sep_floor": bool(np.mean(list(sep.values())) >= c["sep_floor"])}
        for v in variants:
            sh, dsh = finals(c["sh"], v)
            if not sh:
                continue
            seeds = sorted(set(sep) & set(sh))
            r[f"ours_shared_{v}"] = list(sh.values())
            r[f"ours_shared_{v}_mean_se"] = mean_se(list(sh.values()))
            r[f"paired_sep_minus_{v}"] = paired_compare([sep[s] for s in seeds], [sh[s] for s in seeds])
            r[f"diag_shared_{v}"] = diag_medians(dsh, ["off_rnn", "clean_q_rnn", "clean_pi_rnn",
                                                       "clean_pi_into_q_heads"])
            if v == "official":
                crit["direction_sep_gt_shared"] = bool(np.mean(list(sep.values())) > np.mean(list(sh.values())))
                crit["shared_order_of_magnitude"] = bool(np.mean(list(sh.values())) <= c["sh_ceiling"])
        if cell == "P_sac":
            r["official_code_ref_separate"] = [{k: v for k, v in x.items() if k != "curve"}
                                               for x in official_ref("sac_lstm", c["window"])]
            r["official_code_ref_shared"] = [{k: v for k, v in x.items() if k != "curve"}
                                             for x in official_ref("sac_lstm_shared", c["window"])]
        r["criteria"] = crit
        ok &= all(crit.values())
        rep[cell] = r
    rep["small_pass"] = bool(ok)
    with open(os.path.join(G2, "report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    # per-seed aggregate in the required layout runs/seed_<n>/metrics.json (small stage values; full stage extends it)
    for s in range(4):
        agg = {"seed": s, "stage": "small", "cells": {}}
        for cell, c in CELLS.items():
            for run, vs in ((c["sep"], ["separate"]), (c["sh"], ["official", "noleak", "detach"])):
                for v in vs:
                    fs, _ = finals(run, v)
                    if s in fs:
                        agg["cells"][f"{cell}/{v}"] = fs[s]
        os.makedirs(os.path.join(REPO, "runs", f"seed_{s}"), exist_ok=True)
        with open(os.path.join(REPO, "runs", f"seed_{s}", "metrics.json"), "w") as f:
            json.dump(agg, f, indent=1)
    print(json.dumps({k: (v.get("criteria") if isinstance(v, dict) else v) for k, v in rep.items()}, indent=1))
    for cell in CELLS:
        r = rep.get(cell, {})
        if "ours_separate_mean_se" in r:
            print(cell, "sep", np.round(r["ours_separate_mean_se"], 1), "ref", np.round(r["ref_separate_mean_se"], 1),
                  "| shared", {v: np.round(r.get(f"ours_shared_{v}_mean_se", [np.nan]), 1).tolist()
                               for v in ("official", "noleak", "detach")}, "ref", np.round(r["ref_shared_mean_se"], 1))


if __name__ == "__main__":
    main()

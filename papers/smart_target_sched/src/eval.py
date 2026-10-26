"""Evaluation: aggregate runs/<tag>/seed_<r>/metrics.json into the paper's tables and the pre-declared statistics.

    python src/eval.py --tag small [--md out.md] [--json out.json]

Per cell: mean +- SE over replications (replication = the seed unit), paired differences against the baselines (paired t, 95 % t- and
bootstrap CIs through rl_jax_bench.stats), Holm over the declared primary family (Whittle/MP vs myopic and vs
TEV in every Table I-IV cell), judgment vs the paper (MATCH <= 5 % rel., CLOSE <= 20 %, else DIVERGENT; gaps in percentage points).
The paper numbers below are transcribed from Tables I-IV (arXiv:2312.07858v1); the Fig 5/6 references are read from the authors' vector fixture.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np  # noqa: E402
from rl_jax_bench.stats import mean_se, paired_compare, holm  # noqa: E402

# ---- paper tables  (paper Tables I-IV): cell -> K -> (index policy, myopic, TEV/TEC)
PAPER = {
    "t1_q2":      {1: (823.19, 868.71, 871.19), 2: (400.53, 405.85, 406.17), 3: (284.65, 293.80, 293.72)},
    "t1_q29":     {1: (961.25, 993.93, 1009.15), 2: (458.49, 464.43, 465.12), 3: (319.84, 326.04, 334.06)},
    "t2_q2":      {1: (750.91, 790.61, 790.40), 2: (377.36, 381.92, 384.06), 3: (268.30, 275.81, 275.75)},
    "t2_q29":     {1: (817.23, 849.91, 861.88), 2: (406.26, 409.88, 410.56), 3: (285.67, 296.35, 296.19)},
    "t3_q2":      {1: (1554.35, 1614.41, 1622.97), 2: (772.19, 807.35, 808.89), 3: (547.38, 567.14, 567.94)},
    "t3_q2345":   {1: (1664.83, 1731.69, 1733.95), 2: (821.05, 859.02, 860.42), 3: (581.47, 605.75, 605.73)},
    "t4_reckless": {1: (4364.18, 4480.28, 4436.27), 2: (1142.87, 1153.19, 1153.43), 3: (610.10, 613.15, 633.80)},
    "t4_cautious": {1: (3468.00, 3584.63, 3534.40), 2: (902.06, 931.63, 917.27), 3: (492.69, 500.40, 504.08)},
    "t4_mixed":   {1: (6777.79, 7014.92, 6879.65), 2: (1824.24, 1895.25, 1860.21), 3: (990.12, 1022.45, 1040.56)},
}
POL_NAMES = ("whittle", "myopic", "tev")
# ---- paper Fig 5/6 reference values: the AUTHORS' exact vector-figure points (tests/fixtures/paper_fig_curves.npz, built by scripts/extract_fig_vectors.py),
# not by-eye readings.  Curve order in every Bound_gap figure: 0 = Whittle/MP index, 1 = myopic, 2 = TEV/TEC.  Raises when the fixture is missing.
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "paper_fig_curves.npz")
FIG_KEYS = {"fig5a": "Bound_gap_rec", "fig5b": "Bound_gap_cau", "fig5c": "Bound_gap_rec_cau", "fig6": "Bound_gap_rec_cau_ratio"}


def paper_gap(fig: str, K: int) -> dict:
    """Authors' plotted gap (%) of {whittle, myopic, tev} at pool size K of figure `fig` (exact vector point; K must be on their grid)."""
    z = np.load(FIXTURE)                                     # raises FileNotFoundError if the fixture is absent: no literal fall-back
    key = FIG_KEYS[fig]
    x = z[f"{key}__0__x"]
    i = int(np.argmin(np.abs(x - K)))
    if abs(x[i] - K) > 0.05:
        raise KeyError(f"{fig}: K={K} is not on the authors' grid {np.round(x, 2).tolist()}")
    return {pol: float(z[f"{key}__{j}__y"][i]) for j, pol in enumerate(POL_NAMES)}


def judge(ours: float, paper: float) -> str:
    rel = abs(ours - paper) / abs(paper)
    return "MATCH" if rel <= 0.05 else ("CLOSE" if rel <= 0.20 else "DIVERGENT")


def load(tag_dir: str):
    """-> (cells, meta): cells[cid][field][policy] = np.array over replications; field in J / n_pcli1_viol / pmax / V_D / lam_star / pvis."""
    files = sorted(glob.glob(os.path.join(tag_dir, "seed_*", "metrics.json")), key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]))
    cells: dict = {}
    meta = None
    for f in files:
        m = json.load(open(f))
        meta = meta or m.get("meta")
        for cid, flds in m["cells"].items():
            for fld, d in flds.items():
                for pol, v in d.items():
                    cells.setdefault(cid, {}).setdefault(fld, {}).setdefault(pol, []).append(v)
    for cid in cells:
        for fld in cells[cid]:
            for pol in cells[cid][fld]:
                cells[cid][fld][pol] = np.array(cells[cid][fld][pol])
    return cells, meta


def table_rows(cells, variant="primary"):
    rows = []
    for name, ks in PAPER.items():
        for K, pv in ks.items():
            cid = f"{name}|K={K}|{variant}"
            if cid not in cells:
                continue
            J = cells[cid]["J"]
            row = {"cell": cid, "scenario": name, "K": K, "n": len(J["whittle"])}
            for pol, ppaper in zip(POL_NAMES, pv):
                m, se = mean_se(J[pol])
                row[pol] = {"mean": m, "se": se, "paper": ppaper, "rel": (m - ppaper) / ppaper, "judge": judge(m, ppaper)}
            row["improvement_over_best_baseline_pct"] = 100 * (min(row["myopic"]["mean"], row["tev"]["mean"]) - row["whittle"]["mean"]) / min(row["myopic"]["mean"], row["tev"]["mean"])
            row["paper_improvement_pct"] = 100 * (min(pv[1], pv[2]) - pv[0]) / min(pv[1], pv[2])
            # improvement with its SE (delta method for the ratio of means, paired by replication) and its grade against the paper's
            bp = "myopic" if row["myopic"]["mean"] <= row["tev"]["mean"] else "tev"
            w_, b_ = np.asarray(J["whittle"], float), np.asarray(J[bp], float)
            R = (b_.mean() - w_.mean()) / b_.mean()
            se_R = float(np.std((b_ - w_) - R * b_, ddof=1) / (np.sqrt(len(w_)) * b_.mean()))
            row["improvement_se_pct"] = 100 * se_R
            row["improvement_z_vs_paper"] = (row["improvement_over_best_baseline_pct"] - row["paper_improvement_pct"]) / (100 * se_R)
            row["improvement_within_2se"] = bool(abs(row["improvement_z_vs_paper"]) <= 2.0)          # our SE only (strict)
            row["improvement_z2_vs_paper"] = row["improvement_z_vs_paper"] / np.sqrt(2.0)             # + the paper's own Nmc = 100 noise, ASSUMED equal to ours (its SE is not reported)
            row["improvement_within_2se_both"] = bool(abs(row["improvement_z2_vs_paper"]) <= 2.0)
            row["order_ok"] = bool(row["whittle"]["mean"] < min(row["myopic"]["mean"], row["tev"]["mean"]))
            row["paper_order_ok"] = bool(pv[0] < min(pv[1], pv[2]))
            row["extras"] = {p: {"mean": mean_se(J[p])[0], "se": mean_se(J[p])[1]} for p in J if p not in POL_NAMES}
            rows.append(row)
    return rows


def paired_family(cells, variant="primary", base="whittle", others=("myopic", "tev"), names=None):
    """Whittle/MP vs each baseline in every cell: paired_compare(a=base, b=other) (diff < 0 = base cheaper = better) + Holm."""
    comps, pv = {}, {}
    for cid, c in cells.items():
        if not cid.endswith("|" + variant) or (names and cid.split("|")[0] not in names):
            continue
        for o in others:
            if o in c["J"] and base in c["J"]:
                r = paired_compare(c["J"][base], c["J"][o])
                key = f"{cid}|{base}-vs-{o}"
                comps[key] = r
                pv[key] = r["t_p"]
    h = holm(pv) if pv else {}
    for k in comps:
        comps[k]["p_holm"] = h[k]["p_holm"]
        comps[k]["reproduced"] = bool(comps[k]["diff_mean"] < 0 and h[k]["p_holm"] < 0.05)
    return comps


def fig_gap_rows(cells, name_prefix, variant="primary"):
    rows = []
    for cid, c in sorted(cells.items(), key=lambda kv: (kv[0].split("|")[0], int(kv[0].split("K=")[1].split("|")[0]))):
        if not cid.startswith(name_prefix + "|") or not cid.endswith("|" + variant) or "V_D" not in c:
            continue
        VD = c["V_D"]["lb"]
        row = {"cell": cid, "n": len(VD), "V_D_mean": float(VD.mean()), "pvis_max": float(c["pvis"]["lb"].max())}
        for pol in POL_NAMES:
            if pol in c["J"]:
                row[pol + "_gap_pct"] = 100 * (float(c["J"][pol].mean()) - float(VD.mean())) / float(VD.mean())          # ratio of means (primary)
                row[pol + "_gap_pct_meanof"] = float(np.mean(100 * (c["J"][pol] - VD) / VD))                               # mean of per-replication gaps
        rows.append(row)
    return rows


def fmt_table(rows) -> str:
    out = ["| cell | n | Whittle/MP (ours, paper, rel, judge) | myopic | TEV/TEC | impr. % (ours +- SE / paper, z) | order |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda p: f"{r[p]['mean']:.2f} +- {r[p]['se']:.2f} ({r[p]['paper']:.2f}, {100 * r[p]['rel']:+.2f} %, {r[p]['judge']})"  # noqa: E731
        out.append(f"| {r['cell']} | {r['n']} | {f('whittle')} | {f('myopic')} | {f('tev')} | {r['improvement_over_best_baseline_pct']:.2f} +- {r['improvement_se_pct']:.2f} / {r['paper_improvement_pct']:.2f} (z {r['improvement_z_vs_paper']:+.1f}; z/sqrt2 {r['improvement_z2_vs_paper']:+.1f}) | {'ok' if r['order_ok'] == r['paper_order_ok'] else 'DIFFERS'} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs-root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
    ap.add_argument("--variant", default="primary")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    cells, meta = load(os.path.join(a.runs_root, a.tag))
    rows = table_rows(cells, a.variant)
    fam = paired_family(cells, a.variant, names=set(PAPER))
    md = [f"tag {a.tag}, variant {a.variant}, replications per cell: {sorted({r['n'] for r in rows})}", "", fmt_table(rows), ""]
    md.append(f"per-cell improvement over the best baseline within 2 SE of the paper's: {sum(r['improvement_within_2se'] for r in rows)}/{len(rows)} (our SE only), {sum(r['improvement_within_2se_both'] for r in rows)}/{len(rows)} (our and the paper's SE, equal and independent)")
    md.append(f"index-arm cost per cell, |z| <= 2 against the paper (per cell, n = {len(rows)}): {sum(abs(r['whittle']['mean'] - r['whittle']['paper']) / r['whittle']['se'] <= 2 for r in rows)}/{len(rows)} (our SE only), {sum(abs(r['whittle']['mean'] - r['whittle']['paper']) / (np.sqrt(2) * r['whittle']['se']) <= 2 for r in rows)}/{len(rows)} (with the paper's equal SE)")
    nrep = sum(v["reproduced"] for v in fam.values())
    md.append(f"paired family (Whittle/MP vs myopic and TEV, {len(fam)} tests, Holm): {nrep}/{len(fam)} reproduced (diff < 0 and p_holm < 0.05)")
    for p in ("fig5a", "fig5b", "fig5c", "fig6"):
        g = fig_gap_rows(cells, p, a.variant)
        for r in g:
            md.append(f"{r['cell']} (n={r['n']}): V_D={r['V_D_mean']:.2f}, gaps % " + ", ".join(f"{pol}={r.get(pol + '_gap_pct', float('nan')):.2f}" for pol in POL_NAMES) + f" | max visited state {r['pvis_max']:.1f}")
    text = "\n".join(md)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")
    if a.json:
        json.dump({"rows": rows, "family": fam, "meta": meta}, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

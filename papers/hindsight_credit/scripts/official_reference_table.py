"""Three-way table: paper read-off / the official snippet inside the numpy reference driver (100 runs, its own
numpy RNG, seed 0) / ours at small-scale scale (20 runs, seed 2). Pure post-processing of saved JSON (no jax).

    uv run python scripts/official_reference_table.py

Inputs : runs/official/ref/E*.json (ref_driver.py independent / fig3r), runs/small/summary.json,
         runs/official/ours_E2_official_immediate.json, runs/official/lockstep.json (E3 Monte-Carlo State HCA).
Output : runs/official/reference_table.json, and a markdown table on stdout.
z = (official - ours) / sqrt(se_off^2 + se_ours^2): independent seeds, so a two-sample (Welch-type) z, descriptive.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "runs", *p)  # noqa: E731
PAPER = {"E1": (-0.10, -0.43), "E3": (0.68, 0.83), "E4": (0.84, 0.80), "E6": (1.845, 1.77), "E7": (1.495, 1.765)}
PAPER_SWEEP = {"E5": {0: (0.79, 0.80), 2: (0.78, 0.72), 5: (0.66, 0.47)}, "E8": {0: (1.58, 1.97), 2: (1.535, 1.68), 4: (1.515, 1.52)}}
E2_PAPER = {"state_hca": [0.45, 0.62, 0.84, 1.05, 1.56, 1.96, 2.75], "return_hca": [0.21, 0.41, 0.65, 0.89, 1.42, 1.90, 2.78]}


def mse(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def judge(ref, paper, spread):
    d = abs(ref - paper)
    if d <= max(0.05 * abs(paper), spread):
        return "MATCH"
    return "CLOSE" if d <= 0.2 * abs(paper) else "DIVERGENT"


def main():
    g2 = json.load(open(R("small", "summary.json")))["experiments"]
    lock = json.load(open(R("official", "lockstep.json")))["cases"]
    rows = []
    for exp in ("E1", "E3", "E4", "E5", "E6", "E7", "E8"):
        ref = json.load(open(R("official", "ref", f"{exp}.json")))
        stat = "avg" if exp in ("E5", "E8") else "final"
        pts = PAPER_SWEEP.get(exp, {0: PAPER.get(exp)})
        for si, paper in pts.items():
            g2row = g2[exp][si]  # small-scale summary rows are in sweep order (every sweep point has a paper value)
            assert all(abs(float(g2row["setting"][k]) - float(v)) < 1e-9 for k, v in ref["settings"][si]["env_kwargs"].items()
                       if not isinstance(v, (bool, str))), (exp, si)
            for mi, m in enumerate(("state_hca", "return_hca")):
                om, ose = mse(ref["settings"][si]["methods"][m][stat])
                um, use = g2row["arms"][m]["mean"], g2row["arms"][m]["se"]
                note = ""
                if exp == "E3" and m == "state_hca":
                    note = f"ours small-scale is 3-step; ours MC (lockstep, seed 2) = {lock['E3_s0_state_hca']['ours_final_mean']:.3f}"
                rows.append({"exp": exp, "setting": ref["settings"][si]["env_kwargs"], "stat": stat, "method": m,
                             "paper": paper[mi], "official_mean": om, "official_se": ose, "ours_mean": um, "ours_se": use,
                             "z_official_vs_ours": (om - um) / max(np.hypot(ose, use), 1e-12),
                             "official_vs_paper": judge(om, paper[mi], 2 * ose), "note": note})
    e2o = json.load(open(R("official", "ref", "E2.json")))
    e2u = json.load(open(R("official", "ours_E2_official_immediate.json")))
    for i, (po, pu) in enumerate(zip(e2o["points"], e2u["points"])):
        for m in ("state_hca", "return_hca"):
            om, ose = mse(po[m])
            um, use = pu[m]
            rows.append({"exp": "E2", "setting": {"p_long": round(po["p_long"], 4), "oracle": po["oracle"]}, "stat": "online",
                         "method": m, "paper": E2_PAPER[m][i], "official_mean": om, "official_se": ose, "ours_mean": um,
                         "ours_se": use, "z_official_vs_ours": (om - um) / max(np.hypot(ose, use), 1e-12),
                         "official_vs_paper": judge(om, E2_PAPER[m][i], 2 * ose), "note": "immediate = official j=0 term"})
    json.dump({"rows": rows}, open(R("official", "reference_table.json"), "w"), indent=1)
    print("| Exp | setting | stat | arm | paper | official snippet (100 runs) | ours small-scale (20 runs) | z | official vs paper |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        s = ",".join(f"{k}={v}" for k, v in r["setting"].items() if k in ("sigma", "eps", "p_long"))
        print(f"| {r['exp']} | {s} | {r['stat']} | {r['method']} | {r['paper']} | {r['official_mean']:.3f} ± {r['official_se']:.3f} | "
              f"{r['ours_mean']:.3f} ± {r['ours_se']:.3f} | {r['z_official_vs_ours']:+.2f} | {r['official_vs_paper']} |")


if __name__ == "__main__":
    main()

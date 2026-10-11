"""Recheck tables from runs/recheck/{family.json, family_per_run.npz,
pgpaper.json, e1start.json}. Pure post-processing.

    uv run python scripts/recheck_report.py -> runs/recheck/tables.md + revised.json
"""
import json
import os
import sys

import numpy as np
from scipy import stats as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "runs", "recheck")
F = json.load(open(os.path.join(D, "family.json")))
A = np.load(os.path.join(D, "family_per_run.npz"))
PG = json.load(open(os.path.join(D, "pgpaper.json")))
E1 = json.load(open(os.path.join(D, "e1start.json")))
SEEDS = ["0", "1", "3", "4", "5"]
CLAIMS = list(F["claims"])
READINGS = ["swapped_a0.4_ah0.3", "counts", "ema", "residual"]
SWEEP = ["hlr0.1", "hlr0.2", "primary", "hlr0.8", "hlr1.6"]
AB = {"REPRODUCED": "REP", "REVERSED": "REV", "NOT REPRODUCED (n.s.)": "n.s."}
R = F["results"]


def v(vn, seed, cl):
    return R[vn][seed][cl]["verdict"]


def tally(vn, cl):
    return " / ".join(AB[v(vn, s, cl)] for s in SEEDS)


out, revised = [], {}
out.append("### Before / after per claim (seed 0 decisive; HCA arm stat, 100 paired runs; Δ = after − before on the same run keys)")
out.append("| claim | reading | HCA before (mean ± SE) | HCA after | Δ ± SE (paired t p) | diff vs PG before → after | Holm verdict before → after (seed 0) | verdicts after, seeds 0/1/3/4/5 |")
out.append("|---|---|---|---|---|---|---|---|")
for cl in CLAIMS:
    for vn in READINGS:
        b, a = A[f"s0_{cl}_primary"], A[f"s0_{cl}_{vn}"]
        d = a - b
        p = st.ttest_1samp(d, 0).pvalue if np.any(d != 0) else 1.0
        rb, ra = R["primary"]["0"][cl], R[vn]["0"][cl]
        out.append(f"| {cl} | {vn} | {b.mean():.3f} ± {b.std(ddof=1)/10:.3f} | {a.mean():.3f} ± {a.std(ddof=1)/10:.3f} | "
                   f"{d.mean():+.3f} ± {d.std(ddof=1)/10:.3f} (p {p:.1e}) | {rb['diff']:+.4f} ± {rb['se']:.4f} → "
                   f"{ra['diff']:+.4f} ± {ra['se']:.4f} | {AB[rb['verdict']]} → {AB[ra['verdict']]} (p_Holm {ra['p_holm']:.1e}) | {tally(vn, cl)} |")
out.append("")
out.append("### Revision rule applied (seed-0 verdict differs from primary AND the same verdict on both fresh seeds 4, 5)")
out.append("| claim | primary verdicts 0/1/3/4/5 | reading(s) that move it (seed 0, 4, 5 agree) | revised judgment |")
out.append("|---|---|---|---|")
for cl in CLAIMS:
    prim = v("primary", "0", cl)
    movers = [vn for vn in READINGS if v(vn, "0", cl) != prim and v(vn, "4", cl) == v(vn, "0", cl) == v(vn, "5", cl)]
    if movers:
        alt = sorted({AB[v(vn, '0', cl)] for vn in movers})
        rj = f"READING-DEPENDENT: {AB[prim]} at A-LR/A-INIT/logit-SGD h; {'/'.join(alt)} under {', '.join(movers)}"
    else:
        rj = f"{AB[prim]} (unchanged by the rule)"
    revised[cl] = dict(primary=prim, movers=movers, revised=rj, primary_tally=tally("primary", cl),
                       reading_tallies={vn: tally(vn, cl) for vn in READINGS})
    out.append(f"| {cl} | {tally('primary', cl)} | {', '.join(movers) or '—'} | {rj} |")
out.append("")
out.append("### S-HLR robustness (α_h sweep; verdicts on seeds 0/1/3/4/5; seed-0 diff vs PG)")
out.append("| claim | " + " | ".join(f"α_h {x.replace('hlr', '') if x != 'primary' else '0.4 (primary)'}" for x in SWEEP) + " |")
out.append("|---|" + "---|" * len(SWEEP))
for cl in CLAIMS:
    out.append(f"| {cl} | " + " | ".join(f"{R[x]['0'][cl]['diff']:+.3f}; {tally(x, cl)}" for x in SWEEP) + " |")
out.append("")
out.append("### S-PGPAPER (descriptive only): PG at lr 0.05, no critic (fits the paper's Fig. 5 PG curve) vs primary HCA arms")
out.append("| fit to paper Fig. 5 PG | rmse mean curve | rmse std curve | final (ours) | final (paper) |")
out.append("|---|---|---|---|---|")
for k, r in PG["fit"].items():
    out.append(f"| {k} | {r['rmse_mean']:.4f} | {r['rmse_sd']:.4f} | {r['final']:.4f} | {r['paper_final']:.4f} |")
out.append("")
out.append("| claim | seed 0 diff ± SE (p) | seeds 1/3/4/5 diff (p) | sign |")
out.append("|---|---|---|---|")
for cl in ("P7", "P8", "P9", "P10"):
    r0 = PG["claims"]["0"][cl]
    rest = ", ".join(f"{PG['claims'][s][cl]['diff']:+.3f} ({PG['claims'][s][cl]['p']:.0e})" for s in SEEDS[1:])
    out.append(f"| {cl} | {r0['diff']:+.4f} ± {r0['se']:.4f} ({r0['p']:.1e}) | {rest} | {'+' if r0['diff'] > 0 else '−'} on "
               f"{sum((PG['claims'][s][cl]['diff'] > 0) == (r0['diff'] > 0) for s in SEEDS)}/5 |")
out.append("")
out.append(f"### S-E1START: θ0[chain, long] = +1 (V0 = −1.819); PG re-tuned on seed 999 → lr {E1['pg_best_lr']} "
           f"(scores {', '.join(f'{k}: {v:.3f}' for k, v in E1['pg_scores'].items())})")
out.append("| arm | ep0 | ep20 | ep50 | ep100 (seed 0) | paper ep20 / ep50 / ep100 |")
out.append("|---|---|---|---|---|---|")
s0 = E1["seeds"]["0"]
for key, lab, pk in (("pg_curve_pts", f"PG lr {E1['pg_best_lr']}", "pg"), ("state_hca_hzero_curve_pts", "State, h0 = uniform (A-INIT)", "state_hca"),
                     ("state_hca_hpolicy_curve_pts", "State, h0 = π (unsupported)", "state_hca"),
                     ("return_hca_hzero_curve_pts", "Return, h0 = uniform", "return_hca"),
                     ("return_hca_hpolicy_curve_pts", "Return, h0 = π (unsupported)", "return_hca")):
    out.append(f"| {lab} | " + " | ".join(f"{x:+.3f}" for x in s0[key]) + " | " + " / ".join(f"{x:+.3f}" for x in E1["paper"][pk]) + " |")
out.append("")
out.append("| P1 (State − PG, ep100) | seed 0 diff ± SE (Holm verdict) | verdicts 0/1/3/4/5 |")
out.append("|---|---|---|")
for h in ("hzero", "hpolicy"):
    r0 = E1["seeds"]["0"]["P1"][f"P1_{h}"]
    out.append(f"| h0 = {'uniform' if h == 'hzero' else 'π'} | {r0['diff']:+.4f} ± {r0['se']:.4f} ({AB[r0['verdict']]}) | "
               + " / ".join(AB[E1['seeds'][s]['P1'][f'P1_{h}']['verdict']] for s in SEEDS) + " |")
txt = "\n".join(out)
open(os.path.join(D, "tables.md"), "w").write(txt + "\n")
json.dump(revised, open(os.path.join(D, "revised.json"), "w"), indent=1)
print(txt)

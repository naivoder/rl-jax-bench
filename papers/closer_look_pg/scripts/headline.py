"""Compact full-scale headline table (markdown) from runs/full/full_report.json, with the authors'-code column."""
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(REPO, "runs", "full", "full_report.json")))
ARMS = ["hopper_ppo", "hopper_trpo", "walker2d_ppo", "walker2d_trpo"]


def f(x, nd=3):
    if x is None:
        return "–"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) < 0.01 and x != 0:
        return f"{x:.2g}"
    return f"{x:.{nd}f}"


def ms(v, nd=3):
    v = np.asarray(v, float)
    return f"{f(v.mean(), nd)} ± {f(v.std(ddof=1) / np.sqrt(len(v)), nd)} ({len(v)})"


def lab5(r):
    """Pre-declared E5 rule (pre-declared): >= 5 HOLDS, 2-5 PARTIAL, < 2 FAILS."""
    return "HOLDS" if r >= 5 else ("PARTIAL" if r >= 2 else "FAILS")


def d(o, p):
    return "–" if p in (None, 0) else f"{100 * (np.mean(o) - p) / abs(p):+.0f}%"


rows = ["| Metric | Paper | Authors' code | Ours (mean ± SE, n) | Δ% | Judgment | Notes |", "|---|---|---|---|---|---|---|"]
for a in ARMS:
    e = R["E0"][a]
    rows.append(f"| E0 final reward {a} (its 468–487) | {f(e['paper'][0])} ({f(e['paper'][1])}–{f(e['paper'][2])}) | "
                f"{f(e['official'][0])} ± {f(e['official'][1])} (3) | {ms(e['per_seed'], 0)} | {d(e['per_seed'], e['paper'][0])} | "
                f"{e['judgment']} | proxy env; spread rule |")
for a in ARMS:
    for it, n, met in ((0, 2000, "to_true"), (150, 2000, "to_true"), (450, 2000, "to_true"), (450, 2000, "pairwise"),
                       (450, 10000, "to_true"), (450, 100000, "to_true")):
        k = f"{a}/{it}/{n}/{met}"
        if k not in R["E1"]:
            continue
        e = R["E1"][k]
        off = e["official"]
        rows.append(f"| E1 {met}(n={n:g}) {a} it {it} | {f(e['paper'][0])} ({f(e['paper'][1])}..{f(e['paper'][2])}){' interp.' if n == 2000 else ''} | "
                    f"{'–' if off is None else f'{f(off[0])} ± {f(off[1])} (3; ref 1e5)'} | {ms(e['per_seed'])} | {d(e['per_seed'], e['paper'][0])} | "
                    f"{e['judgment']} | {'3 seeds' if n == 100000 else 'ref 1e6'} |")
for a in ARMS:
    for key, lab in (("heldout_returns_mre", "Returns-MRE"), ("heldout_gae_mre_code", "GAE-MRE (code form)"),
                     ("heldout_gae_mre_true", "GAE-MRE (|·|/|·|)")):
        e = R["E2"][f"{a}/{key}/450"]
        pp = e["paper"]
        if key == "heldout_gae_mre_true":
            pp = R["E2"][f"{a}/heldout_gae_mre_code/450"]["paper"]       # H2: the paper's curve read as the dimensionless form
        med = float(np.median(e["per_seed"]))
        j = e["judgment"]
        if key == "heldout_gae_mre_true":
            j = ("MATCH" if (abs(med - pp[0]) / pp[0] <= .05 or pp[1] <= med <= pp[2]) else
                 "CLOSE" if abs(med - pp[0]) / pp[0] <= .2 else "DIVERGENT") + " (H2 reading)"
        note = "median of 5 (paper: median of 5)" + ("; TRPO code-form not reproducible by authors' code" if
                                                       (key == "heldout_gae_mre_code" and "trpo" in a) else "")
        rows.append(f"| E2 held-out {lab} {a} it 450 | {f(pp[0])} [{f(pp[1])}, {f(pp[2])}] | {f(e['official_median'])} | "
                    f"median {f(med)}; {ms(e['per_seed'])} | {d([med], pp[0])} | {j} | {note} |")
for env in ("hopper", "walker2d"):
    for it in (150, 300, 450):
        for kind in ("zero", "zero_raw", "gae", "vstar"):
            k = f"{env}_ppo/{it}/10000/{kind}"
            if k not in R["E3"]:
                continue
            e = R["E3"][k]
            lab = {"zero": "zero (batch-mean centred)", "zero_raw": "zero (un-centred; post-review)"}.get(kind, kind)
            ja = e.get("judgment_amended_A3", e["judgment"])
            rows.append(f"| E3 Fig 4 pairwise(1e4) {env} it {it} {lab} | {f(e['paper'])} | – | {ms(e['per_seed'])} | {d(e['per_seed'], e['paper'])} | "
                        f"{e['judgment']}{'' if ja == e['judgment'] else f' (A3: {ja})'} | PPO (algorithm not stated in the paper; assumed); V* on 1e6 pairs; declared rule |")
for env in ("hopper", "walker2d"):
    e = R["E5"][env]
    rows.append(f"| E5 reward ratio PPO / no-baseline {env} | ≈10× ('almost an order of magnitude') | – | {e['ratio']:.2f}× "
                f"({ms(e['ppo'], 0)} / {ms(e['nobaseline'], 0)}) | – | {lab5(e['ratio'])} | "
                f"pre-declared arm keeps adv-norm centring |")
dg = R["E5_diag"]
for k, v in dg.items():
    rows.append(f"| E5 diag hopper: {k} | ≈10× | – | {v['ratio']:.2f}× ({ms(v['per_seed'], 0)}) | – | "
                f"{lab5(v['ratio'])} | 3 seeds, diagnostic |")
for env in ("hopper", "walker2d"):
    e = R.get("literal_no_baseline", {}).get(f"P4_nocenter {env}")
    if e:
        rows.append(f"| E5 reward ratio PPO / no-baseline un-centred {env} (post-review) | ≈10× | – | {e['ratio']:.2f}× "
                    f"({ms(R['E5'][env]['ppo'], 0)} / {ms(e['per_seed'], 0)}) | – | {lab5(e['ratio'])} | 5 seeds, labelled post-review |")
print("\n".join(rows))

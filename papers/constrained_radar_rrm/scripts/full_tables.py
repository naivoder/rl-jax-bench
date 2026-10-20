"""Render runs/<tag>/report.json as the markdown tables of results/tables.md (stdout). Usage: full_tables.py [--tag=full_v2]

The re-evaluation flags: learner rows carry 'k/5 catastrophic' (seed mean U < -1e6; such means are numerically irreproducible
magnitudes); calibrated quantities are flagged: Table VI under A-04a is the A-04a anchor
itself; every baseline U level depends on SNR0, which is fitted to the Fig. 8 single-target baseline curve, so baseline
levels are 'anchor-dependent', not independent tests. G's heuristic reference is printed when it is EA."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_report import PAPER_TINIT, PAPER_U, judge  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--tag=")), "full_v2")
R = json.load(open(os.path.join(REPO, "runs", TAG, "report.json")))


def f(x, nd=0):
    if x is None:
        return "-"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "nan"
    if abs(x) >= 1e6 or (abs(x) < 1e-2 and x != 0):
        return f"{x:.3g}"
    return f"{x:,.{nd}f}"


def pm(d, nd=0):
    return f"{f(d['mean'], nd)} ± {f(d['se'], nd)} (n={d['n']})"


CASE_NAME = dict(case1="I", case2="II", case3f="III (omega 0.5)", case3s="III (omega 0.005)", case4="IV")


def table7(ck):
    print(f"\n#### Table VII, calibration {ck} (replica / case schedule, k_test; baselines: mean ± SE over the 5 "
          f"pairing groups of 20 draws; learners: over 5 seeds)\n")
    print("| Metric | Paper | Ours (mean ± SE, n) | Δ% | Judgment | Notes |")
    print("|---|---|---|---|---|---|")
    for case in ["case1", "case2", "case3f", "case3s", "case4"]:
        tab = R["cases"].get(f"{case}|{ck}")
        if not tab:
            continue
        for arm, d in tab.items():
            base = arm.split("[")[0]
            if "[" in arm:
                var = arm.split("[")[1].split("]")[0]
                if var not in ("A04", "A04a"):
                    continue
            pu = PAPER_U[case].get(base)
            u = d["U"]
            if pu is None:
                j, rel = "-", float("nan")
            else:
                j, rel = judge(u["mean"], pu)
            note = []
            if "n_catastrophic" in d:
                note.append(f"{d['n_catastrophic']}/5 seeds catastrophic (< -1e6)")
            if "G" in d:
                ref = "" if d["G"].get("heuristic", "DA") == "DA" else " (vs EA: DA mean is below EA here)"
                note.append(f"G = {f(d['G']['mean'], 2)} [{f(d['G']['t95'][0], 2)}, {f(d['G']['t95'][1], 2)}]{ref}")
                note.append(f"sat {f(d['sat']['mean'], 3)}")
                note.append(f"share {f(d['share']['mean'], 2)}")
            if "zero-shot" in arm:
                note.append("secondary")
            if "[" in arm:
                note.insert(0, f"median {f(float(np.median(u['per_seed'])))}")
            if base == "oa_myopic" or base == "oa_blind":
                note.append("labelled addition" if base == "oa_myopic" else
                            ("OA without region knowledge" if case == "case4" else "OA without L_bm knowledge (= full-run OA)"))
            if base in ("ea", "da", "oa", "oa_myopic", "oa_blind"):
                note.append("level anchor-dependent (SNR0 fitted to Fig. 8)")
            print(f"| U, Case {CASE_NAME[case]}, {arm} | {f(pu) if pu else '-'} | {pm(u)} | "
                  f"{f(100 * rel, 1) if np.isfinite(rel) else '-'} | {j} | {'; '.join(note)} |")


def table6(ck):
    tab = R["cases"].get(f"case1|{ck}")
    if not tab:
        return
    print(f"\n#### Table VI (Case I T_init, slots), calibration {ck}\n")
    print("| Metric | Paper | Ours (mean ± SE, n) | Δ% | Judgment | Notes |")
    print("|---|---|---|---|---|---|")
    for arm, d in tab.items():
        base = arm.split("[")[0]
        if "[" in arm and arm.split("[")[1].split("]")[0] not in ("A04", "A04a"):
            continue
        p = PAPER_TINIT.get(base)
        j, rel = judge(d["tinit"]["mean"], p) if p else ("-", float("nan"))
        flag = ""
        if ck == "A04a" and "[" not in arm:
            flag = "; CALIBRATED (A-04a): S_ref fitted to this table's pooled EA/DA/OA value, not a test"
        print(f"| T_init (slots), {arm} | {p if p else '-'} | {pm(d['tinit'], 1)} | {f(100 * rel, 1) if np.isfinite(rel) else '-'} | "
              f"{j}{' (calibrated)' if flag else ''} | scan {f(d['scan']['mean'])}{flag} |")


def family(ck):
    fam = R["families"].get(ck)
    if not fam:
        return
    print(f"\n#### Family H1-H9, calibration {ck} ({'PRIMARY' if ck == 'A04' else 'sensitivity'}; paired t over 5 "
          f"seed-level pairs, Holm over 9)\n")
    print("| id | learner (mean ± SE) | reference (mean ± SE) | diff [t95] | boot95 | rel | p | p_Holm | verdict |")
    print("|---|---|---|---|---|---|---|---|---|")
    for h, d in fam.items():
        print(f"| {h} | {pm(d['a'], 1 if h == 'H7' else 0)} | {pm(d['b'], 1 if h == 'H7' else 0)} | "
              f"{f(d['diff_mean'], 1)} [{f(d['t95'][0], 1)}, {f(d['t95'][1], 1)}] | "
              f"[{f(d['boot95'][0], 1)}, {f(d['boot95'][1], 1)}] | {f(100 * d['rel_improvement'], 1)} % | "
              f"{f(d['p_test'], 4)} | {f(d['p_holm'], 4)} | {d['verdict']} |")


def preds(ck):
    P = R["preds"].get(ck)
    if not P:
        return
    print(f"\n#### Predictions P1-P6, calibration {ck}\n")
    print(json.dumps(P, indent=1, default=str))


def main():
    for ck in ("A04", "A04a"):
        table7(ck)
        table6(ck)
        family(ck)
    for ck in ("A04", "A04a"):
        print(f"\n#### Comparable claims, {ck}\n")
        for k, v in R["comparable"].get(ck, {}).items():
            if isinstance(v, str):
                print(f"- {k}: {v}")
            elif "verdict" in v:
                print(f"- {k}: rel {f(100 * v['rel'], 1)} %, boot95 [{f(100 * v['rel_boot95'][0], 1)}, "
                      f"{f(100 * v['rel_boot95'][1], 1)}] % -> {v['verdict']}")
            else:
                print(f"- {k}: diff {f(v['diff_mean'])} t95 [{f(v['t95'][0])}, {f(v['t95'][1])}] p {f(v['t_p'], 4)}")
        preds(ck)
    print("\n#### Ablation / debug\n")
    print(json.dumps({k: {v: {kk: (vv if not isinstance(vv, dict) or "mean" not in vv else
                                   f"{f(vv['mean'], 3)} ± {f(vv['se'], 3)}")
                              for kk, vv in d.items() if kk not in ("U_vs_ref", "sat_vs_A04", "U_vs_A04")}
                          if isinstance(d, dict) else d for v, d in ab.items()} for k, ab in
                      list(R["ablation"].items()) + [("debug_" + k, v) for k, v in R["debug"].items()]},
                     indent=1, default=str))
    print("\n#### Curves (A-18 plateau, lambda)\n")
    for run, d in R.get("curves", {}).items():
        for v, x in d.items():
            print(f"- {run}[{v}]: plateau {x['n_plateau']}/{x['n']}, windup fixed (max chunk-mean lambda < 1e6 in all"
                  f" seeds) = {x['windup_fixed']}, max chunk-mean lambda per seed "
                  f"{[f(l['lam_chunkmean_max']) for l in x['lanes']]}")
    print("\n#### C8\n")
    for k, v in R["c5c7c8"].get("C8", {}).items():
        print(f"- {k}: mean {f(v['mean'], 3)} t95 [{f(v['t95'][0], 3)}, {f(v['t95'][1], 3)}] -> {v['sign']}")


if __name__ == "__main__":
    main()

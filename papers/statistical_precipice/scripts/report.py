"""Render the full run markdown tables the README run) from runs/full/{ale,e6}.json. Prints to stdout.

    python scripts/report.py [ale|e6|all]
"""
import json
import math
import os
import sys

import numpy as np
from scipy import stats as st

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G3 = os.path.join(REPO, "runs", "full")
NAMES = {"median": "median", "iqm": "IQM", "mean": "mean", "og": "OG", "dp25": "DP-25", "dp50": "DP-50", "superhuman": "P(X>1)"}


def f(v, d=3):
    return f"{v:.{d}f}"


def verdict(rel, inside=False):
    if rel <= 0.05:
        return "MATCH"
    if inside:
        return "MATCH (in paper CI)"
    return "CLOSE" if rel <= 0.20 else "**DIVERGENT**"


def ale():
    """the review pass: no '<=' clipped convention (M2: the axes-edge lower end is a true value, compared, with
    the read tolerance); every row carries its provenance: rows involving DQN (Nature) / DQN (Adam)
    are DATA DIVERGENT (input data differs from the paper's), the numeric rule is still printed for them."""
    d = json.load(open(os.path.join(G3, "ale.json")))
    L = []
    L.append("| Metric | Paper (PDF vector read) | Ours: point [95 % CI] (n = 5 runs × 55 games) | Δ% | Numeric rule | Judgment | Notes: authors' code (rliable@3ccd9f4, same data) |")
    L.append("|---|---|---|---|---|---|---|")
    cnt, cnt_est, cnt_dd = {}, {}, {}

    def add(label, r, d_=3, endpoint=False):
        pc = r["paper_ci"]
        tol = r.get("read_tol", 0.0)
        pcs = f"[{f(pc[0], d_)}, {f(pc[1], d_)}]"
        if endpoint:   # A.20: the claim is the interval; judge the worse endpoint (both ends compared)
            rel = max(r["lo_rel"], r["hi_rel"])
            v = verdict(rel)
            note = f"authors [{f(r['authors_ci'][0], d_)}, {f(r['authors_ci'][1], d_)}]; endpoints Δ {100*r['lo_rel']:.1f} % / {100*r['hi_rel']:.1f} %"
            lab, ours = f"A.20 {label} interval", f"[{f(r['ours_ci'][0], d_)}, {f(r['ours_ci'][1], d_)}]"
        else:
            inside = pc[0] - tol <= r["ours"] <= pc[1] + tol
            v = verdict(r["rel"], inside)
            au = r["authors"]
            note = (f"authors {f(au, d_)} [{f(r['authors_ci'][0], d_)}, {f(r['authors_ci'][1], d_)}]" if isinstance(au, float) else str(au))
            lab, ours = label, f"{f(r['ours'], d_)} [{f(r['ours_ci'][0], d_)}, {f(r['ours_ci'][1], d_)}]"
            pcs = f"{f(r['paper'], d_)} {pcs}"
        dd = r.get("provenance") == "data_divergent"
        judg = "DATA DIVERGENT" if dd else v
        L.append(f"| {lab} | {pcs} | {ours} | {100*rel if endpoint else 100*r['rel']:.1f} | {v} | {judg} | {note} |")
        k = v.split()[0].strip('*')
        cnt[k] = cnt.get(k, 0) + 1
        tgt = cnt_dd if dd else cnt_est
        tgt[k] = tgt.get(k, 0) + 1

    for k, r in d["fig9"].items():
        a, s = k.split("|")
        add(f"Fig 9 {a} {NAMES[s]}", r)
    for k, r in d["a20"].items():
        a, s = k.split("|")
        add(f"{a} {NAMES[s]}", r, endpoint=True)
    for k, r in d["a28"].items():
        a, s = k.split("|")
        add(f"A.28 {a} {NAMES[s]}", r)
    for k, r in d["a27"].items():
        add(f"A.27 P({k})", r)
    for k, r in d["fig11a_studentized"].items():
        m, s = k.split("|")
        r = dict(r, authors="= Colab `mean_CI` (E0 parity 2.1e-12); D6-corrected CI [" + f(r["d6_ci"][0]) + ", " + f(r["d6_ci"][1]) + f"], df {r['d6_df']:.1f}")
        add(f"Fig 11a DMC {m} {s} mean (studentized, n = {r['n_runs']})", r)
    print("\n".join(L))
    print(f"\nNumeric rule, all rows: {cnt}")
    print(f"Estimator rows (C51 / IQN / Rainbow / DMC): {cnt_est}")
    print(f"DATA DIVERGENT rows (involve DQN (Nature) or DQN (Adam)), numeric rule shown for information: {cnt_dd}")


def e6(name="e6.json"):
    d = json.load(open(os.path.join(G3, name)))
    if name != "e6.json":
        base = json.load(open(os.path.join(G3, "e6.json")))
        d = dict(base, **{k: v for k, v in d.items() if k in ("fewrun", "coverage", "counts")})
    print(json.dumps({k: d[k] for k in ("counts", "seeds", "failed_runs", "raw_mean_se", "raw_greedy_mean_se") if k in d}, indent=1))
    if "primary" in d:
        print("\n| H | comparison | IQM diff [95 % CI] | P(sep > other) [CI] | task-mean score sep / other (mean ± SE, n) | mean diff [CI] | Welch t, p | Holm p | decision |")
        print("|---|---|---|---|---|---|---|---|---|")
        for h, r in d["primary"].items():
            print(f"| {h} | {r['comparison']} | {r['iqm_diff']:+.4f} [{r['iqm_diff_ci'][0]:+.4f}, {r['iqm_diff_ci'][1]:+.4f}] | {r['poi']:.3f} [{r['poi_ci'][0]:.3f}, {r['poi_ci'][1]:.3f}] | "
                  f"{r['taskmean_sep'][0]:.4f} ± {r['taskmean_sep'][1]:.4f} / {r['taskmean_other'][0]:.4f} ± {r['taskmean_other'][1]:.4f} (n = {r['n'][0]}, {r['n'][1]}) | "
                  f"{r['mean_diff']:+.4f} [{r['mean_diff_ci'][0]:+.4f}, {r['mean_diff_ci'][1]:+.4f}] | t = {r['welch_t']:.2f}, p = {r['welch_p']:.2g} | {r['p_holm']:.2g} | {r['decision']} |")
    if "fewrun" in d:
        print("\n| comparison | n | point has pool sign | CI excl. 0, pool sign | CI excl. 0, wrong sign | Welch p<.05 | CI covers pool diff | mean CI width | SD of point |")
        print("|---|---|---|---|---|---|---|---|---|")
        for k, r in d["fewrun"].items():
            c, n = k.split("/")
            print(f"| {c} (pool {r['pool_iqm_diff']:+.4f}) | {n[1:]} | {r['point_same_sign']:.3f} | {r['ci_excludes0_same_sign']:.3f} | {r['ci_excludes0_wrong_sign']:.3f} | {r['welch_p_lt_05']:.3f} | {r['ci_covers_pool_diff']:.3f} | {r['mean_ci_width']:.4f} | {r['point_sd']:.4f} |")
    if "coverage" in d:
        print("\n| arm / stat | K | truth | percentile | basic | bc | bca | width (pct) | D10 prediction |")
        print("|---|---|---|---|---|---|---|---|---|")
        for k, r in d["coverage"].items():
            arm, s, K = k.split("/")
            Kn = int(K[1:])
            pred = st.t.cdf(1.959963984540054 * math.sqrt((Kn - 1) / Kn), 2 * (Kn - 1)) * 2 - 1
            print(f"| {arm} / {s} | {Kn}{'*' if r['finite_pool_caveat'] else ''} | {r['truth']:.4f} | " + " | ".join(
                f"{r[m]['coverage']:.3f} ± {r[m]['se']:.3f}" + (f" ({r[m]['undefined']} undef.)" if r[m]['undefined'] else "") for m in ("percentile", "basic", "bc", "bca"))
                + f" | {r['percentile']['width']:.4f} | {pred:.3f} |")


def sens():
    """the review pass labelled sensitivities (runs/full/e6_sens.json) beside the full-run result (runs/full/e6.json, same keys)."""
    d = json.load(open(os.path.join(G3, "e6_sens.json")))
    base = json.load(open(os.path.join(G3, "e6.json")))
    if "iqm_mode_a_effective_trim" in d:
        tr = d["iqm_mode_a_effective_trim"]
        Ns = sorted({int(k[1:]) for r in tr.values() for k in r})
        print("\n| pool (cut per end / K) | " + " | ".join(f"N = {n}" for n in Ns) + " |")
        print("|---|" + "---|" * len(Ns))
        for lab, r in tr.items():
            print(f"| {lab} | " + " | ".join((lambda v: f"{v['cut_each_end']}/{v['K']} = {100*v['trim_fraction']:.1f} %")(r[f"N{n}"]) if f"N{n}" in r else "—" for n in Ns) + " |")
    if "paired" in d:
        print("\n| eval | H | r(task-mean) | per-task r | paired t, p | Wilcoxon p | Welch p (declared) | IQM diff [paired 95 % CI] |")
        print("|---|---|---|---|---|---|---|---|")
        for k, r in d["paired"].items():
            ev, h = k.split("/")
            print(f"| {ev} | {h} | {r['pearson_r_taskmean']:+.3f} (p {r['pearson_p']:.2f}) | {', '.join('n/a' if v != v else f'{v:+.2f}' for v in r['pearson_r_per_task'])} | "
                  f"t = {r['paired_t']:.2f}, p = {r['paired_p']:.2g} | {r['wilcoxon_p']:.2g} | {r['welch_p']:.2g} | {r['iqm_diff']:+.4f} [{r['paired_iqm_diff_ci'][0]:+.4f}, {r['paired_iqm_diff_ci'][1]:+.4f}] |")
    if "greedy" in d:
        print("\n| H (greedy eval) | IQM sep / other | IQM diff [95 % CI] | P(sep > other) [CI] | Welch t, p |")
        print("|---|---|---|---|---|")
        for h, r in d["greedy"].items():
            print(f"| {h} {r['comparison']} | {r['iqm_by_arm'][0]:.4f} / {r['iqm_by_arm'][1]:.4f} | {r['iqm_diff']:+.4f} [{r['iqm_diff_ci'][0]:+.4f}, {r['iqm_diff_ci'][1]:+.4f}] | "
                  f"{r['poi']:.3f} [{r['poi_ci'][0]:.3f}, {r['poi_ci'][1]:.3f}] | t = {r['welch_t']:.2f}, p = {r['welch_p']:.2g} |")
        fr_rows = [(f"greedy {h}", n, v) for h, r in d["greedy"].items() for n, v in r["fewrun"].items()]
    else:
        fr_rows = []
    if "fewrun_with_replacement" in d:
        fr_rows += [(f"with repl. {k.split('/')[0]}", k.split("/")[1], v) for k, v in d["fewrun_with_replacement"].items()]
    if fr_rows:
        print("\n| variant | n | point has pool sign | CI excl. 0, pool sign | CI excl. 0, wrong sign | Welch p<.05 | CI covers pool diff | mean CI width |")
        print("|---|---|---|---|---|---|---|---|")
        for lab, n, v in fr_rows:
            se = v["mc_se"]
            print(f"| {lab} (pool {v['pool_iqm_diff']:+.4f}) | {n[1:]} | {v['point_same_sign']:.3f} | {v['ci_excludes0_same_sign']:.3f} ± {se['ci_excludes0_same_sign']:.3f} | "
                  f"{v['ci_excludes0_wrong_sign']:.3f} ± {se['ci_excludes0_wrong_sign']:.3f} | {v['welch_p_lt_05']:.3f} | {v['ci_covers_pool_diff']:.3f} ± {se['ci_covers_pool_diff']:.3f} | {v['mean_ci_width']:.4f} |")
    if "iqm_mode_c" in d:
        ic = d["iqm_mode_c"]
        print("\n| few-run (no repl.) | n | mode a: excl. 0 pool / wrong sign, covers | mode c: excl. 0 pool / wrong sign, covers |")
        print("|---|---|---|---|")
        for k, v in ic["fewrun"].items():
            a_ = base["fewrun"][k]
            print(f"| {k.split('/')[0]} | {k.split('/')[1][1:]} | {a_['ci_excludes0_same_sign']:.3f} / {a_['ci_excludes0_wrong_sign']:.3f}, {a_['ci_covers_pool_diff']:.3f} | "
                  f"{v['ci_excludes0_same_sign']:.3f} / {v['ci_excludes0_wrong_sign']:.3f}, {v['ci_covers_pool_diff']:.3f} |")
        print("\n| coverage | K | truth a / c | percentile a / c | basic a / c | bc a / c | bca a / c |")
        print("|---|---|---|---|---|---|---|")
        for k, v in ic["coverage"].items():
            arm, _, K = k.split("/")
            a_ = base["coverage"][f"{arm}/iqm/{K}"]
            print(f"| {arm} IQM | {K[1:]} | {a_['truth']:.4f} / {v['truth']:.4f} | " + " | ".join(f"{a_[m]['coverage']:.3f} / {v[m]['coverage']:.3f}" for m in ("percentile", "basic", "bc", "bca")) + " |")
        print("\n| sampling (bias vs pool, 100,000 subsamples) | N | IQM mode a bias ± SE | IQM mode c bias ± SE | pooled median bias |")
        print("|---|---|---|---|---|")
        for k, v in ic["sampling"].items():
            arm, _, N = k.split("/")
            a_ = base["sampling"][f"{arm}/iqm/{N}"]
            m_ = base["sampling"][f"{arm}/pooled_median/{N}"]
            print(f"| {arm} | {N[1:]} | {a_['bias']:+.4f} ± {a_['bias_se']:.4f} | {v['bias']:+.4f} ± {v['bias_se']:.4f} | {m_['bias']:+.4f} |")
        print("\n| lift 95 % interval | N | ℓ = 0.10: mode a / mode c | ℓ = 0.25: mode a / mode c |")
        print("|---|---|---|---|")
        for k, v in ic["lift"].items():
            arm, _, N = k.split("/")
            a_ = base["lift"][f"{arm}/iqm/{N}"]
            fm = lambda q: f"[{q[0]:+.3f}, {q[1]:+.3f}]"  # noqa: E731
            print(f"| {arm} | {N[1:]} | {fm(a_['0.10'])} / {fm(v['0.10'])} | {fm(a_['0.25'])} / {fm(v['0.25'])} |")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("ale", "all"):
        ale()
    if what in ("e6", "all"):
        e6()
    if what == "e6_declared":
        e6("e6_declared.json")
    if what in ("sens", "all") and os.path.exists(os.path.join(G3, "e6_sens.json")):
        sens()

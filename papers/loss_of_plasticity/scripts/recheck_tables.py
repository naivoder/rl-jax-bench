"""recheck tables -> runs/recheck_tables.json.

    uv run python -m scripts.recheck_tables [d12] [act] [b10pop]

d12    /: Fig. D.12 under the standard and the paper-era AdamGnT, on both stream readings (flip-one 3 M:
        full_scr_d12 [CPU] vs recheck_d12_fo_pe [CPU, 4 blocks merged]; re-sample 1 M: full_scr_d12_rs [GPU] vs recheck_d12_rs_pe
        [GPU]); per-cell paired diff (paper-era - standard, same streams; two-sided, descriptive), judgments under the
        full-scale rule and the strict 5 % rule, P1-P6 (one-sided, Holm over 6) per reading.
act    : S-ACT CBP arms + linear control on never-used streams 2000-2199; the three pre-declared statistics.
b10pop : Fig. B.10 first bins, 100-stream values vs the population probes (strict 5 % counts).
"""
import json
import os
import sys

import numpy as np

from rl_jax_bench.stats import holm, mean_se, paired_compare  # noqa: E402

from scripts.full_tables import d12_table, one_sided  # noqa: E402
from scripts.full_extra_tables import B10_FIRST_DIGITISED, counts, strict  # noqa: E402
from src.eval import RUNS, load  # noqa: E402

OUT = os.path.join(RUNS, "recheck_tables.json")
SA = os.path.join(RUNS, "recheck")


def per_seed(tag, arm):
    r = load(tag)
    ss = sorted(r)
    return ss, np.asarray([r[s]["arms"][arm]["mse_whole_run"] for s in ss])


def part_d12(out):
    res = {}
    for reading, std_tag, pe_tag in (("flip_one_3M", "full_scr_d12", "recheck_d12_fo_pe"), ("resample_1M", "full_scr_d12_rs", "recheck_d12_rs_pe")):
        if not load(pe_tag):
            print("missing", pe_tag)
            continue
        rows_s, tests_s = d12_table(std_tag, {})
        rows_p, tests_p = d12_table(pe_tag, {})
        hs = holm({k: one_sided(v, False) for k, v in tests_s.items()})
        hp = holm({k: one_sided(v, False) for k, v in tests_p.items()})
        rows = {}
        for a in rows_s:
            ss, xs = per_seed(std_tag, a)
            sp, xp = per_seed(pe_tag, a)
            assert ss == sp, (a, ss[:3], sp[:3])
            pc = paired_compare(xp, xs)
            p = rows_s[a]["paper"]
            rows[a] = {"paper": p, "standard": rows_s[a]["ours"], "paper_era": rows_p[a]["ours"],
                       "diff": [pc["diff_mean"], pc["diff_se"]], "p_two": pc["t_p"], "n": len(ss),
                       "judge_standard": rows_s[a]["judge"], "judge_paper_era": rows_p[a]["judge"],
                       "strict_standard": strict(rows_s[a]["ours"][0], p) if p else rows_s[a]["judge"],
                       "strict_paper_era": strict(rows_p[a]["ours"][0], p) if p else rows_p[a]["judge"],
                       "dpct_standard": None if p is None else 100 * (rows_s[a]["ours"][0] - p) / p,
                       "dpct_paper_era": None if p is None else 100 * (rows_p[a]["ours"][0] - p) / p}
        res[reading] = {"std_tag": std_tag, "pe_tag": pe_tag, "rows": rows,
                        "counts": {k: counts([r[f"judge_{k}"] for r in rows.values()]) for k in ("standard", "paper_era")},
                        "counts_strict": {k: counts([r[f"strict_{k}"] for r in rows.values()]) for k in ("standard", "paper_era")},
                        "tests_paper_era": tests_p, "holm_paper_era": hp, "tests_standard": tests_s, "holm_standard": hs}
        print(f"== {reading}: counts {res[reading]['counts']} strict {res[reading]['counts_strict']}")
        for a, r in rows.items():
            print(f"{a:36s} paper {r['paper']}  std {r['standard'][0]:.3f}±{r['standard'][1]:.3f} {r['judge_standard']:9s}"
                  f" pe {r['paper_era'][0]:.3f}±{r['paper_era'][1]:.3f} {r['judge_paper_era']:9s}"
                  f" diff {r['diff'][0]:+.4f}±{r['diff'][1]:.4f} p {r['p_two']:.2g}")
        for k, v in tests_p.items():
            print(f"  PE {k:40s} {v['diff_mean']:+.4f}±{v['diff_se']:.4f} holm1s {hp[k]['p_holm']:.2g} n {v['n']}")
    out["d12"] = res


# early bins fixed before launch from the fresh-probe selection on streams 100-199)
EARLY_BIN = {"tanh_cbp_0.0001": 8, "sigmoid_cbp_1e-05": 3, "elu_cbp_1e-05": 2, "relu_cbp_0.0001": 2,
             "leaky_relu_cbp_0.0001": 9, "swish_cbp_0.0001": 2, "linear_0.003": 2}


def bins40(pp):
    P = pp.shape[1] // 4 * 4
    return pp[:, :P].reshape(len(pp), -1, 4).mean(-1)


def part_act(out):
    R = {}
    for tag in ("recheck_act_conf", "recheck_act_conf_lin"):
        for s, d in load(tag).items():
            R.setdefault(s, {}).update(d["arms"])
    ss = sorted(R)
    stats = {}
    S_free = {}
    for a in EARLY_BIN:
        pp = np.asarray([R[s][a]["mse_per_10k"] for s in ss])
        b = bins40(pp)
        late = pp[:, -25:].mean(1)
        S_free[a] = late - b[:, 5:25].mean(1)
        unb = late - b[:, EARLY_BIN[a]]
        stats[a] = {"n": len(ss), "last250k": mean_se(late), "S_free": mean_se(S_free[a]), "S_unb": mean_se(unb),
                    "p_free_1s": one_sided(paired_compare(late, b[:, 5:25].mean(1)), True),
                    "p_unb_1s": one_sided(paired_compare(late, b[:, EARLY_BIN[a]]), True)}
    cbp = [a for a in EARLY_BIN if a != "linear_0.003"]
    for a in cbp:
        pc = paired_compare(S_free[a], S_free["linear_0.003"])
        stats[a]["S_linctl"] = [pc["diff_mean"], pc["diff_se"]]
        stats[a]["p_linctl_1s"] = one_sided(pc, True)
    for key in ("free", "unb", "linctl"):
        h = holm({a: stats[a][f"p_{key}_1s"] for a in cbp})
        for a in cbp:
            stats[a][f"holm_{key}"] = h[a]["p_holm"]
    for a in stats:   # one-sided 95 % upper bounds (normal approx.; n = 200)
        for key in ("S_free", "S_unb", "S_linctl"):
            if key in stats[a]:
                m, se = stats[a][key]
                stats[a][key + "_ub95"] = m + 1.645 * se
    out["act"] = {"seeds": [ss[0], ss[-1]], "n": len(ss), "early_bin": EARLY_BIN, "arms": stats}
    for a, v in stats.items():
        line = f"{a:24s} last250k {v['last250k'][0]:.3f}  S_free {v['S_free'][0]:+.4f}±{v['S_free'][1]:.4f} (ub {v['S_free_ub95']:+.3f}, p1 {v['p_free_1s']:.2g})" \
               f"  S_unb {v['S_unb'][0]:+.4f}±{v['S_unb'][1]:.4f} (ub {v['S_unb_ub95']:+.3f}, p1 {v['p_unb_1s']:.2g})"
        if "S_linctl" in v:
            line += f"  linctl {v['S_linctl'][0]:+.4f}±{v['S_linctl'][1]:.4f} (ub {v['S_linctl_ub95']:+.3f}, holm {v['holm_linctl']:.2g}; holm free {v['holm_free']:.2g} unb {v['holm_unb']:.2g})"
        print(line)


def part_b10pop(out):
    """First 40k bin vs the digitised paper points: 100 shared streams (full_scr_b10) vs the population (seeds 100-1999,
    CPU probe ENV-2) and the 1000-stream GPU probe (M6, streams 100-1099)."""
    env = json.load(open(os.path.join(SA, "environment", "first_bin_fresh_summary.json")))["arms"]
    m6 = load("recheck/measurement/b10_firstbin_s100_1099")
    rows = {}
    for a, p in B10_FIRST_DIGITISED.items():
        v6 = np.asarray([np.mean(m6[s]["arms"][a]["mse_per_10k"][:4]) for s in sorted(m6)])
        e = env[a]
        rows[a] = {"paper": p, "streams_0_99": e["probe_0_99"], "pop_100_1999": [e["pop"], e["pop_se"]],
                   "m6_100_1099": mean_se(v6),
                   "dpct_0_99": 100 * (e["probe_0_99"] - p) / p, "dpct_pop": 100 * (e["pop"] - p) / p,
                   "dpct_m6": 100 * (v6.mean() - p) / p,
                   "strict_0_99": strict(e["probe_0_99"], p), "strict_pop": strict(e["pop"], p),
                   "strict_m6": strict(float(v6.mean()), p)}
    out["b10pop"] = {"rows": rows, **{f"counts_{k}": counts([r[f"strict_{k}"] for r in rows.values()])
                                      for k in ("0_99", "pop", "m6")},
                     **{f"mean_dpct_{k}": float(np.mean([r[f"dpct_{k}"] for r in rows.values()])) for k in ("0_99", "pop", "m6")}}
    for a, r in rows.items():
        print(f"{a:17s} paper {r['paper']:.3f} 0-99 {r['streams_0_99']:.3f} ({r['dpct_0_99']:+.1f}%) pop {r['pop_100_1999'][0]:.3f}"
              f" ({r['dpct_pop']:+.1f}% {r['strict_pop']}) m6 {r['m6_100_1099'][0]:.3f} ({r['dpct_m6']:+.1f}% {r['strict_m6']})")
    print({k: v for k, v in out["b10pop"].items() if k != "rows"})


PARTS = {"d12": part_d12, "act": part_act, "b10pop": part_b10pop}


def main(argv):
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for p in (argv or list(PARTS)):
        PARTS[p](out)
    json.dump(out, open(OUT, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1:])

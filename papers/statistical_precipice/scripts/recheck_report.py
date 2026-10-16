"""Render the recheck before/after tables from runs/full/recheck_*.json. No computation."""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G3 = os.path.join(REPO, "runs", "full")


def J(n):
    p = os.path.join(G3, n)
    return json.load(open(p)) if os.path.exists(p) else None


def theory():
    d, e = J("recheck_theory.json"), J("recheck_theory_eyeballed.json")
    print("\n**T1 (width exponent)**\n\n| fit | pre-fix (eyeballed Fig 6) | after (vector Fig 6) | SE from law 0.449, after | SE from 0.5, after |\n|---|---|---|---|---|")
    for s in ("iqm", "median"):
        a, b = e["T1"][f"paper_{s}"], d["T1"][f"paper_{s}"]
        print(f"| paper {s} percentile width | {a['beta']:.3f} ± {a['se']:.3f} | {b['beta']:.3f} ± {b['se']:.3f} | {b['se_from_law']:+.2f} | {b['se_from_half']:+.2f} |")
    print("\n**T2 (paper Fig 6 IQM percentile coverage vs D10)**\n\n| N | paper eyeballed | paper vector | D10 (pre-fix) | D10 + FPC (R = 200) | gap pre-fix (eyeballed − D10), pp | gap after (vector − D10 FPC), pp | FPC target |\n|---|---|---|---|---|---|---|---|")
    for a, b in zip(e["T2"]["paper"], d["T2"]["paper"]):
        print(f"| {b['N']} | {100 * a['paper_iqm_pct']:.1f} | {100 * b['paper_iqm_pct']:.1f} | {100 * b['d10']:.2f} | {100 * b['d10_fpc']:.2f} | "
              f"{a['gap_d10_pp']:+.2f} | {b['gap_fpc_pp']:+.2f} | {100 * b['fpc_target']:.2f} |")
    print("\n**T2 / §3.5 prediction column on E6 (M = 2, R = 100), shared IQM percentile**\n\n| K | ours (declared run) | D10 (pre-fix) | D10 + FPC | gap pre-fix, pp | gap after, pp | FPC target |\n|---|---|---|---|---|---|---|")
    for r in d["T2"]["e6_shared_iqm"]:
        print(f"| {r['K']} | {100 * r['cov']:.1f} | {100 * r['d10']:.1f} | {100 * r['d10_fpc']:.1f} | {r['gap_d10_pp']:+.2f} | {r['gap_fpc_pp']:+.2f} | {100 * r['fpc_target']:.1f} |")
    print("\n**M3 ranking (closest method to the reference), K ∈ {10, 20}**\n\n| cell | percentile | basic | bc | bca | FPC target | closest to 0.95 | closest to FPC target | percentile behind closest (FPC), pp |\n|---|---|---|---|---|---|---|---|---|")
    for r in d["M3"]["cells"]:
        c = r["cov"]
        print(f"| {r['cell']} | {c['percentile']:.3f} | {c['basic']:.3f} | {c['bc']:.3f} | {c['bca']:.3f} | {r['fpc_target']:.3f} | {r['closest_to_95']} | {r['closest_to_fpc']} | {r['pct_gap_to_closest_fpc_pp']:.2f} |")
    print(f"\npercentile closest: {d['M3']['pct_closest_95']} / 12 (to 0.95), {d['M3']['pct_closest_fpc']} / 12 (to FPC target)")
    print("\nPaper's own Fig 6 under the same ranking (vector read): " + "; ".join(
        f"{r['stat']} N={r['N']}: closest {r['closest_to_fpc']}" for r in d["M3"]["paper_fig6_ranking"]))
    print("\n**§3.4 'CI covers pool diff' vs FPC**\n\n| pair | n | covers (declared) | D10 + FPC prediction | FPC target |\n|---|---|---|---|---|")
    for r in d["fewrun_cover"]:
        print(f"| {r['pair']} | {r['n']} | {r['covers']:.3f} | {r['d10_fpc']:.3f} | {r['fpc_target']:.3f} |")
    print("\n**T3, E6 side (IQM mode c, one fresh key set, 100,000 subsamples)**\n\n| arm | bias N = 3, 5, 10, 25, 50 | κ pre-fix (mixed keys, OLS SE) | κ after (± OLS ⊕ MC) | a/N + c/N² χ² (3 dof), p | consistent (rule) |\n|---|---|---|---|---|---|")
    for arm, r in d["T3_e6"].items():
        f = r["a_over_N_plus_c_over_N2"]
        print(f"| {arm} | {', '.join(f'{b:+.5f}' for b in r['bias'])} | {r['legacy_fit']['kappa']:.3f} ± {r['legacy_fit']['se']:.3f} | "
              f"{r['kappa']:.3f} ± {r['kappa_se_total']:.3f} | {f['chi2']:.1f}, {f['p']:.4f} | {'yes' if r['consistent_O1N_leading'] else 'no'} |")
    print("\n**T3, paper side (Fig A.17)**\n\n| algorithm | E[IQM_N] − E[IQM_100], N = 3, 5, 10, 25, 50 (×1e-4) | read σ (1 px) | κ free [95 % CI] | κ = 1 fit, max \\|resid\\| (px) | strictly decreasing | rule |\n|---|---|---|---|---|---|---|")
    for nm, r in d["T3_paper_a17"].items():
        print(f"| {nm} | {', '.join(f'{1e4 * b:+.1f}' for b in r['bias_vs_N100'])} | {r['read_sigma']:.1e} | {r['kappa_free']:.2f} "
              f"[{r['kappa_ci95'][0]:.2f}, {r['kappa_ci95'][1]:.2f}] | {r['kappa1_fit']['max_abs_resid_px']:.1f} | {r['strictly_decreasing']} | {'met' if r['rule_strict'] else 'not met'} |")
    g = d["toys_d3"]
    print("\n**toy checks D3 (average-score profile)**\n\n| N | max bias vs own target (τ ≥ 0.03 from every μ_m) | gap to run-score tail (different estimand) |\n|---|---|---|")
    for r in g["avg_rows"]:
        print(f"| {r['N']} | {r['max_abs_bias_vs_own_target']:.4f} | {r['avg_gap_to_run_tail']:.4f} |")
    print(f"\nrun-score half: max |z| {g['max_abs_z_run']:.2f}, variance ratio {g['var_ratio_range'][0]:.3f}–{g['var_ratio_range'][1]:.3f}; "
          f"pre-fix criterion {g['legacy_pass']}, new criterion {g['pass_']}")
    ale = J("ale.json")["fig10_iqm_curve"]
    print("\n**Fig 10 right (three-way: paper vector read / authors' pipeline on our hardware / ours)**\n\n| agent | frame (M) | paper point [band] | authors' code point [band] | ours point [band] | \\|Δ\\| point / worse band end | judgment |\n|---|---|---|---|---|---|---|")
    for r in d["fig10_right"]:
        if r["agent"] not in ("C51", "IQN", "Rainbow"):
            continue
        a = ale[r["agent"]]
        i = [1, 10, 25, 50, 75, 100, 125, 150, 175, 200].index(r["frame_paper"])
        fo = f"{r['frame_paper']}" if r["frame_paper"] <= 175 else "200 (ours 199)"
        print(f"| {r['agent']} | {fo} | {r['paper']:.4f} [{r['paper_band'][0]:.4f}, {r['paper_band'][1]:.4f}] | "
              f"{a['authors_point'][i]:.4f} [{a['authors_ci'][0][i]:.4f}, {a['authors_ci'][1][i]:.4f}] | {r['ours']:.4f} [{r['ours_band'][0]:.4f}, {r['ours_band'][1]:.4f}] | "
              f"{r['abs_diff']:.4f} / {max(r['band_abs_diff']):.4f} | {r['point_judgment']} / band {r['band_judgment']} |")


def rs():
    d = J("recheck_rs.json")
    if d is None:
        return
    print("\n** before / after per arm (normalised score, sampled eval, seeds 1–100 paired)**\n\n| arm / task | before (defaults) mean ± SE | after (reward scaling) mean ± SE | Δ ± SE | paired t p | Wilcoxon p |\n|---|---|---|---|---|---|")
    for k, v in d["paired_before_after_normalised"].items():
        pp = "n/a" if v["paired_p"] is None else f"{v['paired_p']:.2g}"
        wp = "n/a" if v["wilcoxon_p"] is None else f"{v['wilcoxon_p']:.2g}"
        print(f"| {k} | {v['before'][0]:.4f} ± {v['before'][1]:.4f} | {v['after'][0]:.4f} ± {v['after'][1]:.4f} | {v['delta']:+.4f} ± {v['delta_se']:.4f} | {pp} | {wp} |")
    print("\n** primary family under each pool**\n\n| pool | eval | H | IQM sep / other | IQM diff [95 % CI] | P(sep > other) [CI] | Welch p | Holm p | decision rule |\n|---|---|---|---|---|---|---|---|---|")
    for k, fam in d["families"].items():
        tag, ev = k.split("/")
        for h, f in fam.items():
            print(f"| {tag} | {ev.replace('eval_', '').replace('_mean', '')} | {h} | {f['iqm_sep']:.4f} / {f['iqm_other']:.4f} | {f['iqm_diff']:+.4f} "
                  f"[{f['iqm_diff_ci'][0]:+.4f}, {f['iqm_diff_ci'][1]:+.4f}] | {f['poi']:.3f} [{f['poi_ci'][0]:.3f}, {f['poi_ci'][1]:.3f}] | "
                  f"{f['welch_p']:.2g} | {f['p_holm']:.2g} | {f['decision']} |")
    print("\n** score shape (raw, sampled eval)**\n\n| pool | arm | CartPole mean ± SE (runs at 500) | greedy | Acrobot mean ± SE (runs ≤ −400) | greedy | Acrobot skew |\n|---|---|---|---|---|---|---|")
    for tag, arms in d["describe"].items():
        for a, t in arms.items():
            c, ac = t["CartPole-v1"], t["Acrobot-v1"]
            print(f"| {tag} | {a} | {c['sampled_mean']:.2f} ± {c['sampled_se']:.2f} ({c['n_at_500']}) | {c['greedy_mean']:.2f} | "
                  f"{ac['sampled_mean']:.2f} ± {ac['sampled_se']:.2f} ({ac['n_at_or_below_m400']}) | {ac['greedy_mean']:.2f} | {ac['skew']:+.2f} |")


def semisynth():
    d = J("recheck_semisynth.json")
    if d is None:
        return
    print(f"\n** semi-synthetic M = 26 pool (R = {d['R']}, {d['sets']} sets × {d['inner']} reps, device {d['device']})**\n\n| noise / stat / K | percentile | basic | bc | bca | FPC target | closest | percentile behind closest, pp |\n|---|---|---|---|---|---|---|---|")
    for k, c in d["cells"].items():
        v = c["coverage"]
        print(f"| {k} | {v['percentile']:.3f} | {v['basic']:.3f} | {v['bc']:.3f} | {v['bca']:.3f} | {c['fpc_target']:.3f} | {c['closest']} | {c['pct_gap_pp']:.2f} |")
    print(f"\nrule: {d['rule']}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["theory", "rs", "semisynth"]
    for w in which:
        globals()[w]()

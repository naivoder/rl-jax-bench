"""Full-scale report (tests fixed before the runs; described in the README). Reads runs/<tag>/summary.json of the
full-scale grids and writes runs/report/<env>.json plus a printed table.

    report.py mg  [tag=full_mg]
    report.py lh  [tag=full_lh]
    report.py pd  [tag=full_pd]      (alpha sweep + T-alpha fit; the draws part is scripts/rr_report.py)
    report.py pd_seeds                 (E1 paired tests on the preliminary PD seed runs)

Primary families: one-sided paired t (by seed) + 10 000-sample bootstrap CI of the paired difference, Holm within the
family; claim iff Holm p < 0.05 and the oriented bootstrap CI excludes 0."""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval import OFFICIAL_HP_RUNS, judge, mean_se, official_tsv  # noqa: E402
from rl_jax_bench.stats import boot_ci, holm  # noqa: E402

OUT = "runs/report"
LH_TSV = os.path.join(OFFICIAL_HP_RUNS, "lighthouse", "pairwise_imitation_results")
LH_MAP = {"bc": "bc", "ppo": "ppo", "bc_then_ppo": "bc_then_ppo", "advisor": "advisor_ppo", "dagger": "dagger",
          "dagger_prefix": "dagger"}


def runs_of(tag):
    return json.load(open(os.path.join("runs", tag, "summary.json")))["runs"]


def one_sided(a, b, lower=False):
    """Paired one-sided test that a beats b (a > b, or a < b if lower). Returns oriented diff (a - b) stats."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (b - a) if lower else (a - b)          # oriented: > 0 means a is better
    if np.all(d == d[0]):
        p = 0.0 if d[0] > 0 else 1.0
    else:
        t = st.ttest_1samp(d, 0.0)
        p = t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2
    return dict(diff_raw_mean=float((a - b).mean()), diff_oriented_mean=float(d.mean()),
                diff_se=float(d.std(ddof=1) / np.sqrt(len(d))), boot95_oriented=boot_ci(d), p_one_sided=float(p),
                wins=int((d > 0).sum()), ties=int((d == 0).sum()), n=len(d))


def family(tests):
    H = holm({k: v["p_one_sided"] for k, v in tests.items()})
    for k, v in tests.items():
        v["p_holm"] = H[k]["p_holm"]
        v["claim"] = bool(v["p_holm"] < 0.05 and v["boot95_oriented"][0] > 0)
    return tests


def per_seed(rows, metric):
    rows = sorted(rows, key=lambda r: r["seed"])
    return [r["seed"] for r in rows], np.array([r["final"][metric] for r in rows])


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    json.dump(obj, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1, default=float)


# ------------------------------------------------------------------ MiniGrid LC Once
def report_mg(tag="full_mg", out_name="mg"):
    R = runs_of(tag)
    seed_runs = [r for r in R if r.get("draw") is None]
    arms = []
    for r in seed_runs:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    tsv = official_tsv("AskForHelpLavaCrossingOnce")
    band = {}
    for r in tsv:
        if 3e-4 <= r["lr"] <= 3e-3:
            band.setdefault(r["exp_type"], []).append(r["reward"][-1])
    name = {"ppo": "ppo", "bc": "bc", "bc_tf1": "bc_teacher_forcing", "bc_ppo": "bc_with_ppo",
            "bc_then_ppo": "bc_then_ppo", "advisor": "advisor"}
    out = {"arms": {}, "family_E3": {}, "draws": {}}
    vals = {}
    for a in arms:
        rows = [r for r in seed_runs if r["arm"] == a]
        seeds, v = per_seed(rows, "ret")
        vals[a] = (seeds, v)
        _, succ = per_seed(rows, "success")
        _, tog = per_seed(rows, "toggled")
        o = np.array(band[name[a]])
        q1, med, q3 = np.percentile(o, [25, 50, 75])
        mu, se = mean_se(v)
        # learning curve from the per-run files (mid checkpoints: 200 episodes; final: 1000)
        curves = []
        for r in sorted(rows, key=lambda r: r["seed"]):
            sub = f"lr{r['lr']:g}" + (f"_a{r['alpha']:g}" if "advisor" in a else "")
            m = json.load(open(os.path.join("runs", tag, r["cell"], a, sub, f"seed_{r['seed']}", "metrics.json")))
            curves.append({k: e["ret"] for k, e in m["evals"].items()})
        steps = sorted(curves[0], key=int)
        curve = {s: mean_se([c[s] for c in curves]) for s in steps}
        out["arms"][a] = dict(seeds=seeds, ret=v.round(4).tolist(), mean=mu, se=se, success=float(succ.mean()),
                              toggled=float(tog.mean()), official_band_mean=float(o.mean()),
                              official_band_se=float(o.std(ddof=1) / np.sqrt(len(o))),
                              official_band_iqr=[float(q1), float(q3)], official_n=len(o),
                              judgment=judge(mu, float(o.mean()), (float(q1), float(q3))), curve=curve)
        print(f"MG {a:12s} ret {mu:+.3f}+-{se:.3f} {v.round(3).tolist()} succ {succ.mean():.3f} toggled {tog.mean():.2f}"
              f" | official band {o.mean():+.3f} IQR [{q1:.3f},{q3:.3f}] -> {out['arms'][a]['judgment']}"
              f" | curve " + " ".join(f"{int(s)//1000}k:{curve[s][0]:.3f}" for s in steps))
    fa, va = vals["advisor"]
    tests = {}
    for b in ("ppo", "bc", "bc_tf1", "bc_ppo", "bc_then_ppo"):
        if b in vals:
            assert vals[b][0] == fa
            tests[b] = one_sided(va, vals[b][1])
    out["family_E3"] = family(tests)
    for b, t in tests.items():
        print(f"  E3 ADV vs {b:12s} diff {t['diff_raw_mean']:+.3f}+-{t['diff_se']:.3f} boot {np.round(t['boot95_oriented'], 3).tolist()}"
              f" p1 {t['p_one_sided']:.2e} holm {t['p_holm']:.2e} wins {t['wins']}/{t['n']} -> {'CLAIM' if t['claim'] else 'no claim'}")
    # ADV at the authors' draws
    ref = {a["draw"]: a["reward"][-1] for a in json.load(open("runs/official_ref/authors_code_runs.json"))
           if a["task"] == "AskForHelpLavaCrossingOnce" and a["exp"] == "advisor" and a["draw"] is not None}
    th = {int(r["seed"]): r["reward"][-1] for r in tsv if r["exp_type"] == "advisor"}
    for r in sorted([r for r in R if r.get("draw") is not None], key=lambda r: r["draw"]):
        out["draws"][r["draw"]] = dict(lr=r["lr"], alpha=r["alpha"], ours=r["final"]["ret"],
                                       ours_success=r["final"]["success"], tsv=th[r["draw"]],
                                       authors_code_our_hw=ref.get(r["draw"]))
    if out["draws"]:
        o = np.array([d["ours"] for d in out["draws"].values()])
        t = np.array([d["tsv"] for d in out["draws"].values()])
        c = np.array([d["authors_code_our_hw"] for d in out["draws"].values()])
        out["draws_summary"] = dict(ours=mean_se(o), tsv=mean_se(t), authors_code=mean_se(c),
                                    judgment_vs_tsv=judge(float(o.mean()), float(t.mean())),
                                    judgment_vs_authors_code=judge(float(o.mean()), float(c.mean())))
        for d, x in out["draws"].items():
            print(f"  draw {d}: lr {x['lr']:.3g} alpha {x['alpha']:g} ours {x['ours']:+.3f} | TSV {x['tsv']:+.3f} | their code {x['authors_code_our_hw']:+.3f}")
        print("  draws mean:", out["draws_summary"])
    dump(out_name, out)


# ------------------------------------------------------------------ 2D-Lighthouse
def lh_official(arm, i, j):
    rows = list(csv.DictReader(open(os.path.join(LH_TSV, f"{LH_MAP[arm]}__2_15.tsv")), delimiter="\t"))
    out = []
    for r in rows:
        if float(r["view_radius"]) != i:
            continue
        # PPO ignores the teacher; the authors' PPO TSV repeats the SAME 25 runs under every expert radius (identical
        # values), so take one copy (expert radius = i) to keep n = 25 distinct runs.
        jj = i if arm == "ppo" else j
        if r["expert_view_radius"] == "" or float(r["expert_view_radius"]) != jj:
            continue
        out.append((float(r["avg_ep_length"]), float(r["reached_near_optimal"])))
    return np.array(out)


def report_lh(tag="full_lh_og", dagger_tag="full_lh_dagger_og", out_name="lh"):
    R = runs_of(tag)
    if dagger_tag and os.path.exists(os.path.join("runs", dagger_tag, "summary.json")):
        # DAgger re-run after the per-stage early-stopping fix (commit 70d0da7); the pre-fix DAgger runs in <tag> are
        # kept on disk and reported separately as 'dagger_prefix'
        pre = [dict(r, arm="dagger_prefix") for r in R if r["arm"] == "dagger"]
        R = [r for r in R if r["arm"] != "dagger"] + runs_of(dagger_tag) + pre
    by = defaultdict(list)
    for r in R:
        by[(r["cell"], r["arm"], r["alpha"] if r["arm"] == "advisor" else None)].append(r)
    out = {"cells": {}, "family_E2": {}, "profile_i1": {}, "table_A1": {}}
    for (cell, arm, alpha), rows in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0)):
        i, j = (int(x[1:]) for x in cell.split("_"))
        seeds, v = per_seed(rows, "len")
        _, no = per_seed(rows, "near_opt_1.1")
        key = f"{cell}/{arm}" + (f"_a{alpha:g}" if arm == "advisor" else "")
        comp_arm = "bc" if (arm == "advisor" and alpha == 0) else ("dagger" if arm == "dagger_prefix" else arm)
        o = lh_official(comp_arm, i, j)
        q1, med, q3 = np.percentile(o[:, 0], [25, 50, 75])
        om = float(np.median(v))
        mu, se = mean_se(v)
        out["cells"][key] = dict(i=i, j=j, arm=arm, alpha=alpha, seeds=seeds, len=v.round(2).tolist(), median=om,
                                 mean=mu, se=se, near_opt_1_1=float(no.mean()), opt=float(rows[0]["opt_len"]),
                                 comparator=f"TSV {LH_MAP[comp_arm]} ({i},{j if comp_arm != 'ppo' else i})", official_median=float(med),
                                 official_iqr=[float(q1), float(q3)], official_n=len(o),
                                 official_near_opt_1_1=float(o[:, 1].mean()),
                                 judgment=judge(om, float(med), (float(q1), float(q3))))
        print(f"LH {key:24s} median {om:7.1f} mean {mu:7.1f}+-{se:5.1f} near-opt {no.mean():.2f} | "
              f"{out['cells'][key]['comparator']} median {med:7.1f} IQR [{q1:6.1f},{q3:6.1f}] n={len(o)} "
              f"near-opt {o[:, 1].mean():.2f} -> {out['cells'][key]['judgment']}")
    tests = {}
    for cell in ("i1_j15", "i3_j15", "i5_j9", "i5_j15"):
        a = out["cells"][f"{cell}/advisor_a20"]
        for b in ("bc", "bc_then_ppo"):
            bb = out["cells"][f"{cell}/{b}"]
            assert a["seeds"] == bb["seeds"]
            tests[f"{cell}:ADV<{b}"] = one_sided(a["len"], bb["len"], lower=True)
    out["family_E2"] = family(tests)
    for k, t in tests.items():
        print(f"  E2 {k:22s} len diff (ADV - X) {t['diff_raw_mean']:+8.1f}+-{t['diff_se']:.1f} boot(oriented) "
              f"{np.round(t['boot95_oriented'], 1).tolist()} p1 {t['p_one_sided']:.2e} holm {t['p_holm']:.2e} "
              f"wins {t['wins']}/{t['n']} -> {'CLAIM' if t['claim'] else 'no claim'}")
    # E2-profile: BC median length vs j at i = 1
    js = [j for j in range(1, 16, 2) if f"i1_j{j}/bc" in out["cells"]]
    ours = np.array([out["cells"][f"i1_j{j}/bc"]["median"] for j in js])
    theirs = np.array([out["cells"][f"i1_j{j}/bc"]["official_median"] for j in js])
    rho = st.spearmanr(ours, theirs)
    out["profile_i1"] = dict(j=js, ours_median=ours.tolist(), tsv_median=theirs.tolist(), spearman=float(rho.statistic),
                             spearman_p=float(rho.pvalue), argmax_ours=js[int(ours.argmax())],
                             argmax_tsv=js[int(theirs.argmax())],
                             judgments=[out["cells"][f"i1_j{j}/bc"]["judgment"] for j in js])
    print("  E2-profile i=1 BC:", out["profile_i1"])
    for i in (1, 3, 5):
        out["table_A1"][i] = {a: out["cells"].get(f"i{i}_j{i}/{a}", {}).get("near_opt_1_1")
                              for a in ("advisor_a20", "bc", "dagger", "dagger_prefix")}
        out["table_A1"][i].update({f"tsv_{a}": out["cells"].get(f"i{i}_j{i}/{a}", {}).get("official_near_opt_1_1")
                                   for a in ("advisor_a20", "bc", "dagger")})
    print("  Table A.1 analogue (fraction of runs with test length < 1.1 x optimal):", out["table_A1"])
    out["tags"] = [tag, dagger_tag]
    dump(out_name, out)


# ------------------------------------------------------------------ Poisoned Doors alpha sweep (T-alpha + ablation)
def report_pd(tag="full_pd", out_name="pd_alpha"):
    R = [r for r in runs_of(tag) if r.get("draw") is None]
    out = {"alpha": {}}
    pred = {0.0: -2 / 3}
    for a in (1.0, 2.0, 4.0, 8.0):
        pred[a] = 1 - 1 / (3 ** a - 1)
    for a in sorted({r["alpha"] for r in R}):
        rows = [r for r in R if r["alpha"] == a]
        seeds, v = per_seed(rows, "ret")
        _, lk = per_seed(rows, "chose_locked")
        _, op = per_seed(rows, "opened_lock")
        mu, se = mean_se(v)
        out["alpha"][a] = dict(seeds=seeds, ret=v.round(4).tolist(), mean=mu, se=se, chose_locked=float(lk.mean()),
                               opened_lock=float(op.mean()), predicted_exact_flow=pred.get(a))
        print(f"PD ADV alpha {a:4g}: ret {mu:+.4f}+-{se:.4f} P(locked) {lk.mean():.4f} P(opened) {op.mean():.4f}"
              f" | exact-flow prediction {pred.get(a)}")
    al = np.array([a for a in (1.0, 2.0, 4.0) if a in out["alpha"]])
    gap = np.array([max(1 - out["alpha"][a]["mean"], 1e-6) for a in al])
    if len(al) >= 2:
        slope, icpt = np.polyfit(al, np.log(gap), 1)
        out["fit"] = dict(alphas=al.tolist(), one_minus_R=gap.tolist(), slope=float(-slope), intercept=float(icpt),
                          ln3=float(np.log(3)), predicted_one_minus_R=[1 / (3 ** a - 1) for a in al])
        print("  T-alpha fit: -d ln(1-R)/d alpha =", round(-slope, 3), "vs ln 3 =", round(np.log(3), 3), out["fit"])
    m = [out["alpha"][a]["mean"] for a in sorted(out["alpha"])]
    out["ordering_holds"] = bool(all(x < y + 1e-9 for x, y in zip(m[:-1], m[1:])))
    dump(out_name, out)


def report_pd_seeds(tag="prelim_pd", out_name="pd_seeds"):
    rep = json.load(open(os.path.join("runs", tag, "report.json")))
    f = rep["pd_privileged/advisor"]
    tests = {}
    for b in ("ppo", "bc", "dagger", "bc_tf1", "bc_ppo", "bc_then_ppo"):
        row = rep[f"pd_privileged/{b}"]
        assert row["seeds"] == f["seeds"]
        tests[b] = one_sided(f["ret"]["per_seed"], row["ret"]["per_seed"])
    family(tests)
    for b, t in tests.items():
        print(f"E1 ADV vs {b:12s} diff {t['diff_raw_mean']:+.3f}+-{t['diff_se']:.3f} boot {np.round(t['boot95_oriented'], 3).tolist()}"
              f" p1 {t['p_one_sided']:.2e} holm {t['p_holm']:.2e} wins {t['wins']}/{t['n']} -> {'CLAIM' if t['claim'] else 'no claim'}")
    dump(out_name, dict(source=f"runs/{tag}/report.json", family_E1=tests))


if __name__ == "__main__":
    what = sys.argv[1]
    args = sys.argv[2:]
    {"mg": report_mg, "lh": report_lh, "pd": report_pd, "pd_seeds": report_pd_seeds}[what](*args)
    # balanced-goal LH numbers (before the official test-goal list): report.py lh full_lh full_lh_dagger lh_balanced

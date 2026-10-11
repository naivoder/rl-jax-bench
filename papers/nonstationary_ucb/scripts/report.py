"""Full-scale analysis: verdict tables, the pre-declared test families (src/configs/full_plan.yaml), ablations, X1-X4.

    uv run python scripts/report.py      -> runs/full/report.json, runs/full/report_tables.txt

Written and committed before the first Full-scale run (see README). Unit of every
claim test = seed mean (n = 5); one-sided paired t (scipy ttest_rel, alternative from the plan) + paired bootstrap
CI (rl_jax_bench.stats.paired_compare); Holm within each family. Per-replication tests are secondary.
"""
from __future__ import annotations

import glob
import json
import math
import warnings
import os
import sys

import numpy as np
import yaml
from scipy import stats as st

warnings.filterwarnings("ignore", category=RuntimeWarning)  # wilcoxon on constant-sign n=5 diffs

from rl_jax_bench.stats import boot_ci, holm, paired_compare  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
from src import verdicts as V  # noqa: E402  (MATCH/CLOSE/DIVERGENT, per-run reading, log-log slope)
from src import theory as th  # noqa: E402  (numpy only)

G3 = os.environ.get("FULL_RUNS", os.path.join(ROOT, "runs", "full"))  # override only for dry runs
PLAN = yaml.safe_load(open(os.path.join(ROOT, "src", "configs", "full_plan.yaml")))
# post-hoc follow-up (see README): POST-HOC additions, labelled; the pre-declared PLAN is unchanged.
RPLAN = yaml.safe_load(open(os.path.join(ROOT, "src", "configs", "posthoc_plan.yaml")))
_PR = os.path.join(G3, "perrun_at.npz")
PERRUN = dict(np.load(_PR)) if os.path.exists(_PR) else {}  # per-run regret at intermediate read-offs
OUT_LINES: list[str] = []
OUT_SUFFIX = ""


def p(line=""):
    OUT_LINES.append(line)
    print(line)


# ---------------------------------------------------------------- loading
def load_dir(d):
    recs = [json.load(open(f)) for f in glob.glob(os.path.join(d, "seed_*", "metrics.json"))]
    return sorted(recs, key=lambda r: r["seed"])


EXPS = {  # experiment -> [(run dir, arm prefix)]; all dirs of one experiment share exp_id/cfg_idx (CRN)
    "e1": [("e1", ""), ("padb_e1", ""), ("abl_e1", ""), ("tie_e1", "tie_"), ("abl2_e1", "")],
    "e2": [("e2", ""), ("padb_e2", ""), ("abl_e2", ""), ("tie_e2", "tie_"), ("abl2_e2", "")],
    "e1b": [("e1b", ""), ("padb_e1b", "")],
    "e2b": [("e2b", ""), ("padb_e2b", "")],
    "e1_text": [("e1_text", "")],
    "sweep_e1": [("sweep_e1", "")],
    "sweep_e2": [("sweep_e2", "")],
    "tuned_e1": [("tuned_e1", "")],
    "tuned_e2": [("tuned_e2", "")],
    "sweepj_e1": [("sweepj_e1", ""), ("sweepjx_e1", "")],   # (post-hoc) + small-xi extension
    "sweepj_e2": [("sweepj_e2", ""), ("sweepjx_e2", "")],
    "tunedj_e1": [("tunedj_e1", "")],
    "tunedj_e2": [("tunedj_e2", "")],
}


def load_exp(name):
    """-> {seeds, arms: {arm: {seed_means (5,), reps (5, R), arm1 (5,), at: {t: (5,)}, rec}}, integrity}"""
    arms, seeds, fps, shas, crn, bad, cfg = {}, None, None, set(), True, False, None
    for d, pre in EXPS[name]:
        recs = load_dir(os.path.join(G3, d))
        if not recs:
            continue
        s = [r["seed"] for r in recs]
        f = [r["env"]["uniforms_fp_sha256"] for r in recs]
        if seeds is None:
            seeds, fps, cfg = s, f, recs[0]["config"]
        assert s == seeds and f == fps, f"{name}/{d}: seeds or environment fingerprints differ from the first dir"
        shas |= {r["env"]["means_sha256"] for r in recs}
        crn &= all(r["crn_identical"] for r in recs)
        for a in recs[0]["policies"]:
            P = [r["policies"][a] for r in recs]
            bad |= any(q["any_bad"] for q in P)
            arms[pre + a] = {"seed_means": np.array([q["final_regret_mean"] for q in P]),
                             "reps": np.array([q["final_regret"] for q in P], dtype=float),
                             "arm1": np.array([q["final_arm1_mean"] for q in P]),
                             "arm1_reps": np.array([q["final_arm1"] for q in P], dtype=float),
                             "at": {k: np.array([q["regret_at_mean"][k] for q in P]) for k in P[0]["regret_at_mean"]},
                             "params": P[0]["params"], "algo": P[0]["algo"], "label": P[0]["label"],
                             "run_s": P[0]["run_s"], "compile_s": P[0]["compile_s"],
                             "thr": P[0]["throughput_steps_reps_per_s"], "ntilde": np.array([q["ntilde_mean"] for q in P])}
    if seeds is None:
        return None
    return {"seeds": seeds, "arms": arms, "cfg": cfg, "fps": fps, "means_sha": sorted(shas), "crn": crn, "any_bad": bad}


def mse(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


# ---------------------------------------------------------------- verdicts
def judge(ours, paper, prec):
    vc = PLAN["verdict"]
    return V.judge(ours, paper, prec, match=vc["match"], close=vc["close"])


def judge_cat(r):
    return V.judge_category(r, PLAN["verdict"]["match"], PLAN["verdict"]["close"])


def perrun_reading(dist, pv, prec):
    """review (post-hoc): where the read-off falls in OUR per-run distribution (one realisation)."""
    return V.perrun_reading(dist, pv, prec, band=RPLAN["perrun_band"])


def verdict_rows(E, readoff, arms_map=None, exp=None):
    """Rows for every (metric, arm) with a read-off. arms_map maps a read-off arm key to our arm key."""
    rows = []
    prec = readoff.get("precision", 10)
    prec_pulls = readoff.get("precision_pulls", 100)  # recheck RC1: vector read-offs carry their own pull precision
    metrics = [("final_regret", "final R_T", None)] + [(k, f"R({k.split('_t')[1]})", k.split("_t")[1])
                                                        for k in readoff if k.startswith("regret_t")]
    metrics += [("arm1_pulls_T", "arm-1 pulls at T", "arm1")] if "arm1_pulls_T" in readoff else []
    for key, label, at in metrics:
        for arm_ro, pv in readoff[key].items():
            arm = (arms_map or {}).get(arm_ro, arm_ro)
            if arm not in E["arms"]:
                continue
            A = E["arms"][arm]
            if at is None:
                vals, reps = A["seed_means"], A["reps"].ravel()
            elif at == "arm1":
                vals, reps = A["arm1"], A["arm1_reps"].ravel()
            else:
                vals, reps = A["at"][at], PERRUN.get(f"{exp}|{arm}|{at}")
            m, se = mse(vals)
            pr = prec_pulls if at == "arm1" else prec
            j, lim = judge(m, pv, pr)
            se_run = None if reps is None else float(reps.std(ddof=1) / math.sqrt(reps.size))
            # (post-hoc): does the mean-based judgment flip within +-2 run-level SE (Monte-Carlo noise)?
            k = RPLAN["mc_boundary_se"]
            mc_lim = None if se_run is None else {judge_cat(abs(m + d - pv) / pv) for d in (-k * se_run, 0.0, k * se_run)} != {j}
            pr_read = None if reps is None else perrun_reading(reps, pv, pr)
            rows.append({"metric": label, "arm": arm, "paper": pv, "mean": m, "se": se, "se_run": se_run, "n": int(A["reps"].size),
                         "rel": (m - pv) / pv, "judgment": j, "precision_limited": lim, "mc_limited": mc_lim, "precision": pr,
                         "paper_pct_in_runs": None if reps is None else float((reps < pv).mean() * 100), "perrun": pr_read})
    return rows


def print_verdicts(title, rows):
    p(f"\n### {title}\n")
    p("| Metric | Arm | Paper | Ours: mean ± SE over n = 5 seed means [run-level SE, n = 10⁴ runs] | Δ% | Judgment (mean) | "
      "Per-run reading (post-hoc, ): paper value vs our 5–95 % run band | Notes |")
    p("|---|---|---|---|---|---|---|---|")
    for r in rows:
        note = f"read-off ±{r['precision']}"
        if r["precision_limited"]:
            note += "; precision-limited"
        if r["mc_limited"]:
            note += "; **MC-limited (judgment flips within ±2 run-level SE)**"
        srun = "" if r["se_run"] is None else f" [{r['se_run']:.1f}]"
        q = r["perrun"]
        pread = "n/a" if q is None else (f"{q['reading']} ({q['pct']:.1f}th pct; band {q['band'][0]:.0f}–{q['band'][1]:.0f}"
                                         + ("; precision-limited" if q["precision_limited"] else "") + ")")
        p(f"| {r['metric']} | {r['arm']} | {r['paper']} | {r['mean']:.1f} ± {r['se']:.1f}{srun} | "
          f"{100 * r['rel']:+.1f} | {r['judgment']} | {pread} | {note} |")


# ---------------------------------------------------------------- tests
def one_test(E, arm, ref, direction):
    a, b = E["arms"][arm]["seed_means"], E["arms"][ref]["seed_means"]
    pc = paired_compare(a, b, n_boot=PLAN["n_boot"])
    t = st.ttest_rel(a, b, alternative=direction)
    rep = paired_compare(E["arms"][arm]["reps"].ravel(), E["arms"][ref]["reps"].ravel(), n_boot=1000)
    trep = st.ttest_rel(E["arms"][arm]["reps"].ravel(), E["arms"][ref]["reps"].ravel(), alternative=direction)
    return {"arm": arm, "ref": ref, "direction": direction, "diff_mean": pc["diff_mean"], "diff_se": pc["diff_se"],
            "boot95": pc["boot95"], "t": float(t.statistic), "p_one_sided": float(t.pvalue), "rel": pc["rel_improvement"],
            "rel_boot95": pc["rel_boot95"], "per_seed_diff": pc["diff_per_seed"],
            "secondary_per_rep": {"diff_mean": rep["diff_mean"], "diff_se": rep["diff_se"], "p_one_sided": float(trep.pvalue),
                                  "n": rep["n"]}}


def run_family(name, exps, plan=None):
    plan = plan or PLAN
    tests = {}
    for e, arm, ref, direction in plan["families"][name]:
        if exps.get(e) is None or arm not in exps[e]["arms"] or ref not in exps[e]["arms"]:
            continue
        tests[f"{e}:{arm}<{ref}" if direction == "less" else f"{e}:{arm}>{ref}"] = one_test(exps[e], arm, ref, direction)
    if not tests:
        return None
    adj = holm({k: v["p_one_sided"] for k, v in tests.items()})
    for k, v in tests.items():
        v["p_holm"] = adj[k]["p_holm"]
        lo, hi = v["boot95"]
        ci_ok = hi < 0 if v["direction"] == "less" else lo > 0
        v["holds"] = bool(v["p_holm"] < PLAN["alpha"] and ci_ok)
    p(f"\n#### Family {name} ({len(tests)} tests, Holm within this family only, one-sided, unit = seed mean, n = 5"
      + (", POST-HOC (post-hoc follow-up)" if plan is not PLAN else "") + ")\n")
    p("| Claim | diff (mean ± SE) | boot 95 % CI | rel. | t | p (1-sided) | p Holm | HOLDS | per-rep diff ± SE (n), p |")
    p("|---|---|---|---|---|---|---|---|---|")
    for k, v in tests.items():
        r = v["secondary_per_rep"]
        p(f"| {k} | {v['diff_mean']:+.1f} ± {v['diff_se']:.1f} | [{v['boot95'][0]:+.1f}, {v['boot95'][1]:+.1f}] | "
          f"{100 * v['rel']:+.1f} % | {v['t']:.2f} | {v['p_one_sided']:.2g} | {v['p_holm']:.2g} | "
          f"{'yes' if v['holds'] else 'NO'} | {r['diff_mean']:+.1f} ± {r['diff_se']:.1f} ({r['n']}), {r['p_one_sided']:.2g} |")
    return tests


def descriptive(exps):
    out = {}
    p("\n#### Descriptive comparisons (two-sided, no claim)\n")
    p("| Comparison | diff (mean ± SE) | boot 95 % CI | rel. | t-test p (2-sided) |")
    p("|---|---|---|---|---|")
    for e, a, b in PLAN["descriptive"]:
        if exps.get(e) is None or a not in exps[e]["arms"] or b not in exps[e]["arms"]:
            continue
        pc = paired_compare(exps[e]["arms"][a]["seed_means"], exps[e]["arms"][b]["seed_means"], n_boot=PLAN["n_boot"])
        out[f"{e}:{a}-{b}"] = pc
        p(f"| {e}: {a} − {b} | {pc['diff_mean']:+.1f} ± {pc['diff_se']:.1f} | [{pc['boot95'][0]:+.1f}, {pc['boot95'][1]:+.1f}] | "
          f"{100 * pc['rel_improvement']:+.1f} % | {pc['t_p']:.2g} |")
    return out


def ablation_table(E, pairs, title):
    p(f"\n#### {title}\n")
    p("| Variant | vs | variant regret (mean ± SE) | headline regret | paired diff ± SE (seed) | Δ% | arm-1 pulls |")
    p("|---|---|---|---|---|---|---|")
    out = {}
    for v, ref in pairs:
        if v not in E["arms"] or ref not in E["arms"]:
            continue
        a, b = E["arms"][v]["seed_means"], E["arms"][ref]["seed_means"]
        pc = paired_compare(a, b, n_boot=PLAN["n_boot"])
        out[v] = {"ref": ref, "mean": mse(a), "ref_mean": mse(b), "diff": [pc["diff_mean"], pc["diff_se"]], "boot95": pc["boot95"],
                  "rel": pc["rel_improvement"], "arm1": float(E["arms"][v]["arm1"].mean())}
        p(f"| {v} | {ref} | {mse(a)[0]:.1f} ± {mse(a)[1]:.1f} | {mse(b)[0]:.1f} | {pc['diff_mean']:+.1f} ± {pc['diff_se']:.1f} "
          f"[{pc['boot95'][0]:+.1f}, {pc['boot95'][1]:+.1f}] | {100 * pc['rel_improvement']:+.1f} | {E['arms'][v]['arm1'].mean():.0f} |")
    return out


# ---------------------------------------------------------------- X3 sweep
def grid_edges(E, keys, best):
    """review which hyperparameters of the argmin sit at the edge of the grid (true optimum may lie outside)."""
    P = {k: E["arms"][k]["params"] for k in keys}
    out = []
    for h in ("xi", "gamma", "tau", "g", "alpha"):
        vals = sorted({round(float(v[h]), 12) for v in P.values() if h in v})
        if len(vals) < 2 or h not in P[best]:
            continue
        b = round(float(P[best][h]), 12)
        if b == vals[0]:
            out.append(f"{h} at grid min")
        elif b == vals[-1]:
            out.append(f"{h} at grid max")
    return out


def x3(E, paper_keys):
    out = {}
    p("\n| Arm | configs | grid argmin | argmin regret (mean ± SE) | paper setting regret | paper / argmin | roughly optimal (≤ 10 %) | argmin at a grid edge? |")
    p("|---|---|---|---|---|---|---|---|")
    for algo, pk in paper_keys.items():
        keys = [k for k, v in E["arms"].items() if v["algo"] == algo]
        means = {k: E["arms"][k]["seed_means"].mean() for k in keys}
        best = min(means, key=means.get)
        pm = means[pk]
        ok = pm <= (1 + PLAN["x3_roughly_optimal"]) * means[best]
        out[algo] = {"n_configs": len(keys), "argmin": best, "argmin_params": E["arms"][best]["params"],
                     "argmin_mean_se": mse(E["arms"][best]["seed_means"]), "paper_key": pk, "paper_mean": float(pm),
                     "ratio": float(pm / means[best]), "roughly_optimal": bool(ok),
                     "grid": {k: [float(means[k]), E["arms"][k]["params"]] for k in keys},
                     "run_s_total": float(sum(E["arms"][k]["run_s"] for k in keys)), "edges": grid_edges(E, keys, best)}
        m, se = mse(E["arms"][best]["seed_means"])
        p(f"| {algo} | {len(keys)} | {best} | {m:.1f} ± {se:.1f} | {pm:.1f} | {pm / means[best]:.3f} | {'yes' if ok else 'NO'} | "
          f"{'; '.join(out[algo]['edges']) or 'no'} |")
    return out


# ---------------------------------------------------------------- X1 / X2
def load_x1(grid):
    pts = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "runs", "x1", grid, "T*_U*"))):
        recs = load_dir(d)
        if len(recs) < 5:
            continue
        T, U = recs[0]["T"], recs[0]["config"]["env"]["upsilon"]
        pts[(T, U)] = recs
    return dict(sorted(pts.items()))


def slope_boot(xs, seed_means, n_boot=10_000, rng_seed=0):
    """OLS slope of log(mean over seeds) on log x; CI by resampling seeds (same resample at every point)."""
    return V.loglog_slope_boot(xs, seed_means, n_boot=n_boot, rng_seed=rng_seed)


def x1_report():
    ext = yaml.safe_load(open(os.path.join(ROOT, "src", "configs", "extensions.yaml")))["x1"]
    band = ext["expected_slope"]
    out = {}
    for grid, key, bname in (("a", "T", "T_fixed_upsilon"), ("b", "U", "upsilon_fixed_T"), ("c025", "T", "c025"), ("c05", "T", "c05")):
        pts = load_x1(grid)
        if len(pts) < 3:
            continue
        xs = [T if key == "T" else U for (T, U) in pts]
        res = {"points": [list(k) for k in pts], "policies": {}}
        p(f"\n#### X1 grid {grid} (x = {key}; {len(pts)} points; 5 seeds × {pts[next(iter(pts))][0]['reps']} reps)\n")
        p("| Policy | " + " | ".join(f"{key}={x:g}" for x in xs) + " | slope (all) [95 % CI] | slope (T ≥ 1e4) | band | in band |")
        p("|---|" + "---|" * len(xs) + "---|---|---|---|")
        for pol in ("ducb", "swucb", "ucb1", "exp3s"):
            sm = [[r["policies"][pol]["final_regret_mean"] for r in recs] for recs in pts.values()]
            est, ci = slope_boot(xs, sm)
            sub = [i for i, (T, U) in enumerate(pts) if T >= 10_000]
            est2 = slope_boot([xs[i] for i in sub], [sm[i] for i in sub], n_boot=2000)[0] if (key == "T" and len(sub) >= 3) else None
            b = band[bname] if pol in ("ducb", "swucb") else ([band["ucb1_T_min"], 1.5] if (pol == "ucb1" and key == "T") else None)
            inb = None if b is None else bool(b[0] <= est <= b[1])
            means = [float(np.mean(s)) for s in sm]
            norm = None
            if key == "T":
                last = list(pts)[-3:]
                if pol == "swucb":
                    norm = [float(np.mean([r["policies"][pol]["final_regret_mean"] for r in pts[k]]) / math.sqrt(k[0] * k[1] * math.log(k[0])))
                            for k in last]
                if pol == "ducb":
                    norm = [float(np.mean([r["policies"][pol]["final_regret_mean"] for r in pts[k]]) / (math.sqrt(k[0] * k[1]) * math.log(k[0])))
                            for k in last]
            res["policies"][pol] = {"means": means, "seed_means": sm, "slope": est, "slope_ci": ci, "slope_T_ge_1e4": est2,
                                    "band": b, "in_band": inb, "normalised_last3": norm,
                                    "params": [pts[k][0]["policies"][pol]["params"] for k in pts],
                                    "run_s": [pts[k][0]["policies"][pol]["run_s"] for k in pts]}
            p(f"| {pol} | " + " | ".join(f"{m:.0f}" for m in means) + f" | {est:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
              f"{'' if est2 is None else f'{est2:.3f}'} | {'' if b is None else b} | {'' if inb is None else ('yes' if inb else 'NO')} |")
        for pol in ("swucb", "ducb"):
            if res["policies"][pol]["normalised_last3"]:
                p(f"- {pol} normalised regret over the last 3 points "
                  f"({'R/√(TΥ log T)' if pol == 'swucb' else 'R/(√(TΥ) log T)'}): "
                  + ", ".join(f"{v:.4f}" for v in res["policies"][pol]["normalised_last3"]))
        out[grid] = res
    return out


def x2_report():
    """Theorem 1 / Theorem 7 RHS vs empirical E N~_T(i) (xi = 0.6) on X1 grids a and b."""
    ext = yaml.safe_load(open(os.path.join(ROOT, "src", "configs", "extensions.yaml")))["x1"]
    base, sw = np.array(ext["base"]), np.array(ext["swapped"])
    gaps = []
    for i in range(3):  # Delta mu_T(i) = min over steps where i is suboptimal of mu* - mu_i
        g = [m.max() - m[i] for m in (base, sw) if m.max() - m[i] > 0]
        gaps.append(min(g))
    out, rows = {}, []
    for grid in ("a", "b"):
        for (T, U), recs in load_x1(grid).items():
            st_ = recs[0]["env"]["self_test"]
            subopt = [T - c for c in st_["best_arm_counts"]]
            for pol, fn in (("ducb", "thm1"), ("swucb", "thm7")):
                prm = recs[0]["policies"][pol]["params"]
                emp = np.mean([r["policies"][pol]["ntilde_mean"] for r in recs], 0)
                for i in range(3):
                    if fn == "thm1":
                        bnd = th.thm1_bound(prm["gamma"], T, U, prm["xi"], gaps[i], 3)
                    else:
                        bnd = th.thm7_bound(prm["tau"], T, U, prm["xi"], gaps[i])
                    vac = bnd >= subopt[i]
                    rows.append({"grid": grid, "T": T, "U": U, "policy": pol, "arm": i + 1, "Delta": gaps[i],
                                 "empirical": float(emp[i]), "bound": float(bnd), "vacuous": bool(vac),
                                 "violated": bool(emp[i] > bnd), "ratio": float(bnd / max(emp[i], 1e-9)), "subopt_steps": subopt[i]})
    out["rows"] = rows
    nv = [r for r in rows if not r["vacuous"]]
    out["n_points"], out["n_nonvacuous"] = len(rows), len(nv)
    out["n_violations"] = sum(r["violated"] for r in rows)
    out["PASS"] = out["n_violations"] == 0
    p(f"\nX2: {len(rows)} (grid point, policy, arm) cells; non-vacuous {len(nv)}; violations {out['n_violations']}.")
    if nv:
        p("\n| grid | T | Υ | policy | arm | Δ | empirical E Ñ | bound | bound / empirical |")
        p("|---|---|---|---|---|---|---|---|---|")
        for r in nv:
            p(f"| {r['grid']} | {r['T']} | {r['U']} | {r['policy']} | {r['arm']} | {r['Delta']:.1f} | {r['empirical']:.1f} | "
              f"{r['bound']:.0f} | {r['ratio']:.1f} |")
    return out


def x1_local_slopes():
    out = {}
    for d in ("x1/a", "x1/c025", "x1/c05", "x1_widegap/a"):
        pts = {}
        for q in glob.glob(os.path.join(ROOT, "runs", d, "T*_U*")):
            recs = load_dir(q)
            if len(recs) == 5:
                pts[recs[0]["T"]] = recs
        if len(pts) < 3:
            continue
        Ts = sorted(pts)
        out[d] = {}
        for pol in ("ducb", "swucb"):
            y = [np.mean([r["policies"][pol]["final_regret_mean"] for r in pts[T]]) for T in Ts]
            loc = [math.log(y[i + 1] / y[i]) / math.log(Ts[i + 1] / Ts[i]) for i in range(len(Ts) - 1)]
            out[d][pol] = {"T": Ts, "local": loc}
            p(f"- {d} {pol}: T {Ts[0]:g}..{Ts[-1]:g}, local slopes " + ", ".join(f"{v:.2f}" for v in loc))
    return out


def x2_widegap():
    """review Theorem 1 / Theorem 7 on the wide-gap DIAGNOSTIC instance (A-G3-2; Delta = 0.8 for every arm when
    suboptimal), where the D-UCB bound is non-vacuous at large T. Same rule as X2 (vacuous = bound >= suboptimal steps)."""
    rows = []
    for q in glob.glob(os.path.join(ROOT, "runs", "x1_widegap", "a", "T*_U*")):
        recs = load_dir(q)
        if len(recs) < 5:
            continue
        T, U = recs[0]["T"], recs[0]["config"]["env"]["upsilon"]
        subopt = [T - c for c in recs[0]["env"]["self_test"]["best_arm_counts"]]
        for pol, fn in (("ducb", "thm1"), ("swucb", "thm7")):
            prm = recs[0]["policies"][pol]["params"]
            emp = np.mean([r["policies"][pol]["ntilde_mean"] for r in recs], 0)
            for i in range(3):
                bnd = th.thm1_bound(prm["gamma"], T, U, prm["xi"], 0.8, 3) if fn == "thm1" else th.thm7_bound(prm["tau"], T, U, prm["xi"], 0.8)
                rows.append({"T": T, "U": U, "policy": pol, "arm": i + 1, "empirical": float(emp[i]), "bound": float(bnd),
                             "subopt_steps": subopt[i], "vacuous": bool(bnd >= subopt[i]), "violated": bool(emp[i] > bnd),
                             "ratio": float(bnd / max(emp[i], 1e-9))})
    rows.sort(key=lambda r: (r["policy"], r["T"], r["arm"]))
    nv = [r for r in rows if not r["vacuous"]]
    p(f"\nX2 wide-gap: {len(rows)} cells; non-vacuous {len(nv)} (D-UCB {sum(r['policy'] == 'ducb' for r in nv)}); "
      f"violations {sum(r['violated'] for r in rows)}.")
    p("\n| T | policy | arm | empirical E Ñ | bound | suboptimal steps | vacuous | bound / empirical |")
    p("|---|---|---|---|---|---|---|---|")
    for r in rows:
        p(f"| {r['T']} | {r['policy']} | {r['arm']} | {r['empirical']:.1f} | {r['bound']:.0f} | {r['subopt_steps']} | "
          f"{'yes' if r['vacuous'] else 'no'} | {r['ratio']:.1f} |")
    return {"rows": rows, "n_nonvacuous": len(nv), "n_violations": sum(r["violated"] for r in rows)}


# ---------------------------------------------------------------- X4
def x4_report():
    ext = yaml.safe_load(open(os.path.join(ROOT, "src", "configs", "extensions.yaml")))["x4"]
    C, Cp = ext["C_paper"], ext["C_rederived"]
    files = sorted(glob.glob(os.path.join(ROOT, "runs", "x4", "T*.json")), key=lambda f: int(os.path.basename(f)[1:-5]))
    out = {}
    if not files:
        return out
    p("\n| T | policy | E N_T(2) | precond. 20.98 ≤ EN ≤ 0.738T | E R | d* | E* R(d*) | max_d E* R (d) | E R·E* R(d*)/T | ≥ C? | ≥ C′? | max{E R, max E* R} / √(C T) | / √(C′ T) | UCB-1: max E* R/(T/log T) |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in files:
        r = json.load(open(f))
        T = r["T"]
        out[T] = {}
        for pol, v in r["policies"].items():
            ER, EN, ds = v["stationary"]["mean"], v["stationary"]["EN2"], v["dstar"]
            mix = {int(k): m["mean"] for k, m in v["mixture"].items()}
            Es = mix[ds]
            dmax = max(mix, key=mix.get)
            prod = ER * Es / T
            mx = max(ER, mix[dmax])
            pre = 64 / (9 * r["alpha_kl"]) <= EN <= T / (4 * r["alpha_kl"])
            r16 = mix[dmax] / (T / math.log(T)) if pol == "ucb1" else None
            out[T][pol] = {"EN2": EN, "precondition": bool(pre), "ER": ER, "dstar": ds, "EstarR_dstar": Es, "dmax": dmax,
                           "EstarR_max": mix[dmax], "prod_over_T": prod, "ge_C": bool(prod >= C), "ge_Cp": bool(prod >= Cp),
                           "cor14_ratio_C": mx / math.sqrt(C * T), "cor14_ratio_Cp": mx / math.sqrt(Cp * T), "remark16": r16,
                           "mixture": mix}
            p(f"| {T} | {pol} | {EN:.1f} | {'yes' if pre else 'no'} | {ER:.1f} | {ds} | {Es:.1f} | {mix[dmax]:.1f} ({dmax}) | {prod:.3f} | "
              f"{'yes' if prod >= C else 'NO'} | {'yes' if prod >= Cp else 'NO'} | {mx / math.sqrt(C * T):.2f} | {mx / math.sqrt(Cp * T):.2f} | "
              f"{'' if r16 is None else f'{r16:.4f}'} |")
    return out


# ---------------------------------------------------------------- read-offs (recheck RC1)
READOFF_MODE = "vector"  # set by --readoff; "eye" reproduces the pre-recheck tables


def readoffs(e):
    """Read-offs come from src/configs/<e>.yaml (not from the run records, whose config copy predates RC1).

    vector (default): Figure-1 values extracted from the PDF's vector paths; eye: the pre-fix eye reads
    (``readoff_eye``). Configs without ``readoff_eye`` (the ALT slides, bitmap only) use ``readoff`` in both modes.
    """
    cfg = yaml.safe_load(open(os.path.join(ROOT, "src", "configs", f"{e}.yaml")))
    if READOFF_MODE == "eye" and "readoff_eye" in cfg:
        return cfg["readoff_eye"]
    return cfg["readoff"]


# ---------------------------------------------------------------- main
def main():
    exps = {k: load_exp(k) for k in EXPS}
    rep = {"integrity": {}, "verdicts": {}, "families": {}, "ablations": {}}
    for k, E in exps.items():
        if E is not None:
            rep["integrity"][k] = {"seeds": E["seeds"], "crn_identical": E["crn"], "any_bad": E["any_bad"],
                                   "means_sha256": E["means_sha"], "uniforms_fp": E["fps"]}
    p("## Integrity\n")
    for k, v in rep["integrity"].items():
        p(f"- {k}: seeds {v['seeds']}, CRN identical {v['crn_identical']}, any non-finite {v['any_bad']}, "
          f"means sha {[s[:12] for s in v['means_sha256']]}, distinct env fingerprints {len(set(v['uniforms_fp'])) == len(v['uniforms_fp'])}")

    ro = {e: readoffs(e) for e in ("e1", "e2", "e1b", "e2b") if exps.get(e)}
    for e in ("e1", "e2"):
        if exps.get(e):
            rows = verdict_rows(exps[e], ro[e], exp=e)
            rows += [dict(r, arm="ducb_padB (secondary arm)") for r in verdict_rows(exps[e], ro[e], {"ducb": "ducb_padB"}, exp=e)
                     if r["arm"] == "ducb_padB"] if "ducb_padB" in exps[e]["arms"] else []
            rep["verdicts"][e] = rows
            print_verdicts(f"{e.upper()} (paper Figure 1, T = 1e4)", rows)
    for e in ("e1b", "e2b"):
        if exps.get(e):
            rows = verdict_rows(exps[e], ro[e], exp=e)
            if "ducb_padB" in exps[e]["arms"]:
                rows += [dict(r, arm="ducb_padB (secondary arm)") for r in verdict_rows(exps[e], ro[e], {"ducb": "ducb_padB"}, exp=e)
                         if r["arm"] == "ducb_padB"]
            rep["verdicts"][e] = rows
            print_verdicts(f"{e.upper()} (ALT slides, T = 2e4; secondary, never pooled)", rows)
    if exps.get("e1"):
        E = exps["e1"]
        r3 = ro["e1"]["regret_t3000"]
        p(f"\nE1 R(3000) ordering (paper C3: UCB-1 initially best; read-offs UCB-1 {r3['ucb1']:g}, SW {r3['swucb']:g}, "
          f"EXP3.S {r3['exp3s']:g}, D {r3['ducb']:g}): "
          + ", ".join(f"{a} {E['arms'][a]['at']['3000'].mean():.1f}" for a in ("ucb1", "swucb", "exp3s", "ducb", "ducb_padB", "oracle_ucb1")
                      if a in E["arms"]))
        if "oracle_ucb1" in E["arms"]:
            m, se = mse(E["arms"]["oracle_ucb1"]["seed_means"])
            p(f"\nORACLE-RESTART-UCB1 (EXTENSION reference, E1): {m:.1f} ± {se:.1f}")
    for fam in ("P", "S1", "S2", "S3", "T"):
        res = run_family(fam, exps)
        if res:
            rep["families"][fam] = res
    p("\nNote: Holm runs within each family, never across families; the bootstrap over n = 5 seed means "
      "has only 126 distinct resamples, so its percentile CIs are coarse and somewhat too narrow.")
    for fam in ("Tj", "S2B"):  # POST-HOC (post-hoc follow-up)
        res = run_family(fam, exps, RPLAN)
        if res:
            rep["families"][fam] = res
    rep["descriptive"] = descriptive(exps)
    for e in ("e1", "e2"):
        if exps.get(e):
            pairs = [("ducb_nodisc", "ducb"), ("swucb_nowin", "swucb"), ("swucb_tau3684", "swucb"), ("ducb_xi06", "ducb"),
                     ("swucb_xi06", "swucb"), ("exp3s_S3", "exp3s"), ("exp3s_S6", "exp3s"), ("ducb_lit", "ducb"),
                     ("ducb_padB", "ducb"), ("oracle_ucb1", "ucb1")] + [(f"tie_{a}", a) for a in ("ucb1", "exp3s", "ducb", "swucb")]
            pairs += [("exp3s_S7", "exp3s"), ("ducb_nodisc_padB", "ducb_padB"), ("ducb_nodisc_padB", "ucb1"),
                      ("ducb_nodisc", "ucb1")]  # post-hoc follow-up
            rep["ablations"][e] = ablation_table(exps[e], pairs, f"Ablations on {e.upper()} (CRN-paired with the headline arm)")
    if exps.get("e1_text"):
        E = exps["e1_text"]
        p("\nE1-text (ABLATION, paper §5 text means): " + ", ".join(f"{a} {mse(v['seed_means'])[0]:.1f} ± {mse(v['seed_means'])[1]:.1f}"
                                                                  for a, v in E["arms"].items()))
        rep["e1_text"] = {a: mse(v["seed_means"]) for a, v in E["arms"].items()}
    rep["x3"] = {}
    for e in ("sweep_e1", "sweep_e2"):
        if exps.get(e):
            p(f"\n#### X3 {e}")
            rep["x3"][e] = x3(exps[e], {"ucb1": "ucb1_xi0.5", "ducb": "ducb_c4", "swucb": "swucb_c4", "exp3s": "exp3s_S2"})
    for e in ("sweepj_e1", "sweepj_e2"):  # (post-hoc): joint grids
        if exps.get(e):
            p(f"\n#### X3 {e} (JOINT grids, post-hoc, )")
            rep["x3"][e] = x3(exps[e], RPLAN["x3j_paper_keys"])
    p("\n### X1")
    rep["x1"] = x1_report()
    p("\n### X1 local slopes (log-log slope between consecutive T points)")
    rep["x1_local"] = x1_local_slopes()
    p("\n### X2")
    rep["x2"] = x2_report() if rep["x1"] else None
    p("\n### X2 on the wide-gap instance (post-hoc)")
    rep["x2_widegap"] = x2_widegap()
    p("\n### X4")
    rep["x4"] = x4_report()
    os.makedirs(G3, exist_ok=True)
    with open(os.path.join(G3, f"report{OUT_SUFFIX}.json"), "w") as f:
        json.dump(rep, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(os.path.join(G3, f"report_tables{OUT_SUFFIX}.txt"), "w") as f:
        f.write("\n".join(OUT_LINES) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--readoff", choices=("vector", "eye"), default="vector",
                    help="Figure-1 read-offs: vector-extracted (recheck RC1, default) or the pre-fix eye reads")
    ap.add_argument("--out-suffix", default="", help="write report<suffix>.json / report_tables<suffix>.txt")
    a = ap.parse_args()
    READOFF_MODE = a.readoff
    OUT_SUFFIX = a.out_suffix
    main()

"""full-scale aggregation: per-arm mean +- SE over seeds, paired t-tests + 10^4-resample bootstrap CIs, relative
improvements, heuristic comparison, readings at the paper's x positions, ablations, probe ratios, tuning.

    python scripts/report.py  # writes runs/full/report.json
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

from rl_jax_bench.stats import boot_ci, holm, paired_compare  # noqa: E402  (shared seed-level statistics)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = {"maml": -39.0, "multi": -41.0, "state": -63.0, "heuristic": -54.0}  # Fig. 5 left, read by eye (+-2)


def load(root):
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "*", "seed_*", "metrics.json"))):
        arm = p.split(os.sep)[-3]
        seed = int(p.split(os.sep)[-2][5:])
        with open(p) as f:
            m = json.load(f)
        if m["final"]["it"] != m["config"]["iterations"]:
            continue  # incomplete run
        out.setdefault(arm, {})[seed] = m
    return out


def last5(m):
    return m["final"]["test_mean_last5"]


def at_it(m, it):
    for e in m["evals"]:
        if e["it"] == it:
            return e["test_mean"]
    return None


def paired(a_runs, b_runs, f=last5):
    common = sorted(set(a_runs) & set(b_runs))
    a = np.array([f(a_runs[s]) for s in common])
    b = np.array([f(b_runs[s]) for s in common])
    # t-interval of the paired difference included (: percentile bootstraps under-cover at n = 5)
    return {"seeds": common, **paired_compare(a, b)}


def _mean_or_none(xs):
    return float(np.mean(xs)) if xs and all(x is not None for x in xs) else None


def arm_summary(runs):
    seeds = sorted(runs)
    v = np.array([last5(runs[s]) for s in seeds])
    ep_std = np.array([np.mean([e["test_std"] for e in runs[s]["evals"][-5:]]) for s in seeds])
    hist = {s: runs[s]["history"] for s in seeds}
    # In-batch Var(PG) / baseline MSE are NOT comparable across arms: multi (4x4) and maml (1x16)
    # batches share inputs, so their within-batch spread excludes the input-driven part.  Only none/state (16 fresh
    # inputs x 1 rollout) are reported; cross-arm variance claims use the matched-policy probe only.
    arm_name = runs[seeds[0]]["arm"]
    fresh = arm_name in ("none", "state")
    vpg = [float(np.nanmean([r["var_pg"] for r in hist[s][-500:]])) for s in seeds] if fresh else None
    bmse = [float(np.mean([r["baseline_mse"] for r in hist[s][-500:]])) for s in seeds]
    ent = [float(np.mean([r["entropy"] for r in hist[s][-50:]])) for s in seeds]
    tim = [runs[s]["timing"] for s in seeds]
    return {"seeds": seeds, "n": len(seeds), "last5": v.round(3).tolist(), "mean": float(v.mean()),
            "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
            "boot95_mean": boot_ci(v), "per_episode_std_last5": float(ep_std.mean()),
            # None when a run did not evaluate there (eval_start, e.g. the post-review confirmation runs)
            "at_it250_mean": _mean_or_none([at_it(runs[s], 250) for s in seeds]),
            "at_it1250_mean": _mean_or_none([at_it(runs[s], 1250) for s in seeds]),
            "greedy_last": float(np.mean([runs[s]["evals"][-1]["greedy_mean"] for s in seeds])),
            "var_pg_last500_per_seed__none_state_only": vpg,
            "baseline_mse_in_batch_last500_per_seed__NOT_comparable_across_arms": bmse, "entropy_last50": ent,
            "steady_iters_per_s": float(np.mean([t["steady_iters_per_s"] for t in tim])),
            "steady_env_steps_per_s": float(np.mean([t["steady_env_steps_per_s"] for t in tim])),
            "first_compile_s": float(np.mean([t["first_compile_s"] for t in tim])),
            "flops_per_train_step": tim[0].get("flops_per_train_step"),
            "wall_s_total_mean": float(np.mean([t.get("wall_s_total") or np.nan for t in tim])),
            "git_sha": sorted({runs[s]["git_sha"] for s in seeds}),
            "test_sha256": sorted({str(runs[s].get("test_sha256")) for s in seeds})}


def summarize_probe(pr):
    out = {}
    for arm, seeds in pr.items():
        for part in ("final_policy", "fixed_set", "init_policy"):
            for s, res in seeds.items():
                if part not in res:
                    continue
                for b, v in res[part].items():
                    if not isinstance(v, dict):
                        continue
                    d = out.setdefault(f"{arm}/{part}", {}).setdefault(b, {"vs_state": [], "vs_none": [], "coord": []})
                    d["vs_state"].append(v["var_ratio_vs_state"])
                    d["vs_none"].append(v["var_pg"] / res[part]["none"]["var_pg"])
                    d["coord"].append(v["median_coord_var_ratio_vs_state"])
                    d.setdefault("bias_frac_z3", []).append(v.get("bias_frac_abs_z_gt_3"))
    summ = {}
    for k, dd in out.items():
        summ[k] = {}
        for b, d in dd.items():
            row = {}
            for nm in ("vs_state", "vs_none", "coord"):
                v = np.array(d[nm])
                lv = np.log(v)
                t = stats.ttest_1samp(lv, 0.0) if len(lv) > 1 and np.std(lv) > 0 else None
                row[nm] = {"ratios": np.round(v, 4).tolist(), "geo_mean": float(np.exp(lv.mean())),
                           "geo_boot95": [float(np.exp(c)) for c in boot_ci(lv)] if len(lv) > 1 else None,
                           "t_p_log_ratio_eq_0": float(t.pvalue) if t is not None else None,
                           "n_below_1": int((v < 1).sum()), "n": len(v)}
            bz = [x for x in d["bias_frac_z3"] if x is not None]
            row["bias_frac_abs_z_gt_3_mean"] = float(np.mean(bz)) if bz else None
            summ[k][b] = row
    return summ


def main():
    os.chdir(ROOT)
    rep = {"paper_fig5_left": PAPER}
    g3 = load("runs/full")
    rep["arms"] = {a: arm_summary(r) for a, r in g3.items()}
    heur = json.load(open("runs/small/heuristics.json"))
    rep["heuristics"] = heur
    spt = heur["spt"]["mean"]
    rep["paired_vs_state"] = {a: paired(g3[a], g3["state"]) for a in g3 if a != "state" and "state" in g3}
    if "maml" in g3 and "multi" in g3:
        rep["maml_vs_multi"] = paired(g3["maml"], g3["multi"])
    for a in ("none",):
        if a in g3:
            rep[f"paired_vs_{a}"] = {b: paired(g3[b], g3[a]) for b in g3 if b != a}
    # vs heuristic: the heuristic is a constant on the identical test set -> one-sample tests on (arm - spt)
    rep["vs_heuristic_spt"] = {}
    for a, r in g3.items():
        d = np.array([last5(r[s]) - spt for s in sorted(r)])
        t = stats.ttest_1samp(d, 0.0)
        rep["vs_heuristic_spt"][a] = {"diff_mean": float(d.mean()), "t_p": float(t.pvalue), "boot95": boot_ci(d)}
    # ablations
    abl = {}
    for root, name, ref in (("runs/full_ablation", "multi_shuffled", "multi"),
                            ("runs/full_ablation_noadapt", "maml", "maml")):
        r = load(root).get(name)
        if r and ref in g3:
            key = name if root.endswith("ablation") else "maml_noadapt"
            abl[key] = {"summary": arm_summary(r), "vs_" + ref: paired(r, g3[ref]),
                        "vs_state": paired(r, g3["state"]) if "state" in g3 else None}
    rep["ablations"] = abl
    # probe (matched-policy gradient variance): merge every probe file
    # MAML probes at the training k (meta_k = 16) are in probe_maml_k16.json and are loaded LAST so they
    # supersede the k = 8 MAML entries of the older files; the k = 8 summaries are kept under *_maml_k8_superseded.
    def merge(files):
        out = {}
        for pp in files:
            for arm, seeds in json.load(open(pp)).items():
                out.setdefault(arm, {}).update(seeds)
        return out

    def k16_last(files):
        return [f for f in files if not f.endswith("probe_maml_k16.json")] + [f for f in files if f.endswith("probe_maml_k16.json")]

    old = sorted(glob.glob("runs/full/probe_*.json"))
    pr = merge(k16_last(old + sorted(glob.glob("runs/full_ablation/probe*.json"))))
    pn = "runs/full_ablation_noadapt/probe.json"
    if os.path.exists(pn):
        for arm, seeds in json.load(open(pn)).items():
            pr.setdefault("maml_noadapt", {}).update(seeds)
    if pr:
        rep["probe"] = summarize_probe(pr)
    k8 = merge([f for f in old if not f.endswith("probe_maml_k16.json")])
    if "maml" in k8:
        rep["probe_maml_k8_superseded"] = summarize_probe({"maml": k8["maml"]})
    told = sorted(glob.glob("runs/full_tuned/probe_*.json"))
    prt = merge(k16_last(told))
    if prt:
        rep["probe_tuned"] = summarize_probe(prt)
    k8t = merge([f for f in told if not f.endswith("probe_maml_k16.json")])
    if "maml" in k8t:
        rep["probe_tuned_maml_k8_superseded"] = summarize_probe({"maml": k8t["maml"]})
    # tuning
    tune = {}
    for p in sorted(glob.glob("runs/tune/*/*/seed_*/metrics.json")):
        parts = p.split(os.sep)
        cfgname, arm, seed = parts[-4], parts[-3], int(parts[-2][5:])
        m = json.load(open(p))
        if m["final"]["it"] != m["config"]["iterations"]:
            continue
        tune.setdefault(arm, {}).setdefault(cfgname, {})[seed] = last5(m)
    sel = {}
    for arm, d in tune.items():
        means = {c: float(np.mean(list(v.values()))) for c, v in d.items() if len(v) == 2}
        if means:
            best = max(means, key=means.get)
            sel[arm] = {"grid_val_last5_mean": means, "selected": best, "selected_val": means[best],
                        "n_configs": len(means)}
    rep["tuning"] = sel
    tuned = load("runs/full_tuned") if os.path.isdir("runs/full_tuned") else {}
    if tuned:
        rep["tuned_arms"] = {a: arm_summary(r) for a, r in tuned.items()}
        tabl = {}
        for root, name, ref, key in (("runs/full_tuned_ablation", "multi_shuffled", "multi", "multi_shuffled"),
                                     ("runs/full_tuned_ablation_noadapt", "maml", "maml", "maml_noadapt")):
            r = load(root).get(name) if os.path.isdir(root) else None
            if r and ref in tuned:
                tabl[key] = {"summary": arm_summary(r), "vs_" + ref: paired(r, tuned[ref]),
                             "vs_state": paired(r, tuned["state"]) if "state" in tuned else None}
        rep["tuned_ablations"] = tabl
        if "state" in tuned:
            rep["tuned_paired_vs_state"] = {a: paired(tuned[a], tuned["state"]) for a in tuned if a != "state"}
    #: Holm over the four primary arm-vs-state tests (paper + tuned config; the tuned pair is post hoc)
    fam = {}
    for cfgname, key in (("paper", "paired_vs_state"), ("tuned", "tuned_paired_vs_state")):
        for a in ("multi", "maml"):
            if a in rep.get(key, {}):
                fam[f"{cfgname}/{a}"] = rep[key][a]["t_p"]
    rep["holm_primary4"] = holm(fam)
    #: pre-declared confirmation of the tuned config on fresh seeds 10-19
    conf = load("runs/full_confirm") if os.path.isdir("runs/full_confirm") else {}
    if conf:
        rep["confirm_arms"] = {a: arm_summary(r) for a, r in conf.items()}
        if "state" in conf:
            rep["confirm_paired_vs_state"] = {a: paired(conf[a], conf["state"]) for a in conf if a != "state"}
            rep["confirm_holm"] = holm({a: rep["confirm_paired_vs_state"][a]["t_p"]
                                        for a in ("multi", "maml") if a in rep["confirm_paired_vs_state"]})
        if "maml" in conf and "multi" in conf:
            rep["confirm_maml_vs_multi"] = paired(conf["maml"], conf["multi"])
        # tuned config pooled over seeds 0-19 (descriptive only; the confirmatory test is seeds 10-19 alone)
        if tuned:
            pooled = {a: {**tuned[a], **conf[a]} for a in conf if a in tuned}
            if "state" in pooled:
                rep["tuned_pooled20_paired_vs_state"] = {a: paired(pooled[a], pooled["state"]) for a in pooled
                                                         if a != "state"}
    # observation sensitivity (queued + in-service work), tuned config, seeds 0-2, paired with full_tuned
    sens = load("runs/sens_obs_work") if os.path.isdir("runs/sens_obs_work") else {}
    if sens:
        rep["sens_obs_work"] = {a: {"summary": arm_summary(r),
                                    "vs_work_queued_same_arm": paired(r, tuned[a]) if a in tuned else None}
                                for a, r in sens.items()}
        if "state" in sens:
            rep["sens_obs_work"]["paired_vs_state"] = {a: paired(sens[a], sens["state"]) for a in sens if a != "state"}
    with open("runs/full/report.json", "w") as f:
        json.dump(rep, f, indent=1)
    # console summary
    for a, s in rep["arms"].items():
        print(f"{a:16s} n={s['n']:2d} last5 {s['mean']:8.2f} +- {s['se'] or 0:5.2f}  @250 {s['at_it250_mean'] or 0:8.1f} "
              f"@1250 {s['at_it1250_mean'] or 0:8.1f}  ent {np.mean(s['entropy_last50']):.3f}  ep-std {s['per_episode_std_last5']:.1f}")
    for a, p in rep["paired_vs_state"].items():
        print(f"  {a} - state: {p['diff_mean']:+.2f} +- {p['diff_se']:.2f}  t p={p['t_p']:.3g}  boot {np.round(p['boot95'], 2)}"
              f"  rel {p['rel_improvement']:+.3f} {np.round(p['rel_boot95'], 3)}  wins {p['wins']}/{p['n']}")
    for k, v in rep["ablations"].items():
        print(k, {kk: (round(vv["diff_mean"], 2), round(vv["t_p"], 4), np.round(vv["boot95"], 2).tolist())
                  for kk, vv in v.items() if kk.startswith("vs_") and vv}, "mean", round(v["summary"]["mean"], 2))
    for key in ("probe", "probe_tuned"):
        for k, d in rep.get(key, {}).items():
            print(k, {b: (round(v["vs_state"]["geo_mean"], 3), round(v["coord"]["geo_mean"], 3),
                          f"{v['vs_state']['n_below_1']}/{v['vs_state']['n']}") for b, v in d.items()})
    if sel:
        print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    sys.exit(main())

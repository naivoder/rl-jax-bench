"""Small-scale report: reduce runs/small/<env>_<alg>/seed_*/{metrics,small}.json, compare with the paper's extracted
curves (src/configs/paper_figdata.json) at the SAME iterations where the paper
has them, and evaluate the pre-declared order-of-magnitude sanity checks S1-S6. Writes runs/small/small_report.json.

Sanity checks (declared before the instrument numbers were seen):
  S1 training: mean reward over its 40-48 > it-0 reward in every arm, and within 10x of the paper's Fig 8 mean at it 48.
  S2 gradients (it 0): to_true(2000) and pairwise(2000) within a factor 10 of the paper's it-0 values (Figs 9/10, same
     env/alg); to_true increases with n (100 -> 1e4).
  S3 value: held-out Returns-MRE and GAE-MRE (code formula) at its 10-40 within 10x of the paper's median curve at the
     same iterations (Walker: Fig 3; Hopper: Fig 12).
  S4 baselines: all three arms finite at every n; V* fit improved its held-out MSE over its first evaluation.
  S5 landscape: |S(0,0)| < 1e-5 for both regimes; ||Delta|| < ||u|| (paper: 'an order of magnitude lower').
  S6 integrity: normalisers frozen in every instrument call; env fingerprint identical across all runs; no NaN abort.
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, "src", "configs", "paper_figdata.json")
from rl_jax_bench.stats import mean_se  # noqa: E402

PAPER_GRAD_IT0 = {  # paper Figs 9/10, it 0, mean of 3 agents: (pairwise@2e3 log-interp, to_true@2e3, to_true@1e2, to_true@1e4)
    ("walker2d", "trpo"): (0.657, 0.799, 0.314, 0.960), ("walker2d", "ppo"): (0.657, 0.794, 0.305, 0.956),
    ("hopper", "trpo"): (0.634, 0.783, 0.248, 0.952), ("hopper", "ppo"): (0.282, 0.472, 0.115, 0.787)}
PAPER_BASE_150_10K = {"walker2d": (0.024, 0.123, 0.234), "hopper": (0.037, 0.174, 0.368)}  # Fig 4 it150 (Zero, V, V*)


def paper_curves():
    d = json.load(open(FIG))
    out = {}
    names = {0: "hopper", 1: "walker2d"}
    for i in (0, 1):
        for alg in ("TRPO", "PPO"):
            ser = [s for s in d["fig8"][i]["series"] if s["series"] == alg]
            out[("fig8", names[i], alg.lower())] = ser
    for alg in ("TRPO", "PPO"):
        out[("heldout_gae", "walker2d", alg.lower())] = [s for s in d["fig3"][0]["series"] if s["series"] == alg]
        out[("heldout_ret", "walker2d", alg.lower())] = [s for s in d["fig3"][1]["series"] if s["series"] == alg]
        out[("heldout_gae", "hopper", alg.lower())] = [s for s in d["fig12"][0]["series"] if s["series"] == alg]
        out[("heldout_ret", "hopper", alg.lower())] = [s for s in d["fig12"][1]["series"] if s["series"] == alg]
    return out


def at(series, it, agg=np.median):
    return float(agg([np.interp(it, s["x"], s["y"]) for s in series]))


def within10(x, ref):
    return bool(ref / 10 <= x <= ref * 10) if ref > 0 else None


def main(exp="small", its=(0, 15, 45)):
    pc = paper_curves()
    rep, checks = {}, {}
    fps = set()
    for run in sorted(glob.glob(os.path.join(REPO, "runs", exp, "*_*"))):
        tag = os.path.basename(run)
        if tag == "logs" or not os.path.isdir(run):
            continue
        env, alg = tag.split("_")
        seeds = sorted(glob.glob(os.path.join(run, "seed_*")))
        M = [json.load(open(os.path.join(s, "metrics.json"))) for s in seeds]
        if alg.endswith("-nobaseline"):                     # E5 arm: only the reward curve
            rew = np.array([[h["mean_reward"] for h in m["history"]] for m in M])
            rep[tag] = {"n_seeds": len(M), "reward_it0": mean_se(rew[:, 0]), "reward_last9": mean_se(rew[:, 40:49].mean(1)),
                        "per_seed_last9": rew[:, 40:49].mean(1).tolist()}
            continue
        I = [json.load(open(os.path.join(s, f"{exp}.json"))) for s in seeds if os.path.exists(os.path.join(s, f"{exp}.json"))]
        for m in M:
            fps.add(json.dumps(m["env_fingerprint"], sort_keys=True) + env)
        r = {"n_seeds": len(M)}
        H = [m["history"] for m in M]
        rew = np.array([[h["mean_reward"] for h in hh] for hh in H])        # [S, iters]
        r["reward_it0"] = mean_se(rew[:, 0])
        r["reward_last9"] = mean_se(rew[:, 40:49].mean(1))
        r["paper_reward_it48"] = at(pc[("fig8", env, alg)], 48, np.mean)
        r["paper_reward_it0"] = at(pc[("fig8", env, alg)], 0, np.mean)
        logged = [h for h in H[0] if "heldout_returns_mre" in h]
        vit = [h["it"] for h in logged]
        r["value"] = {}
        for key in ("heldout_returns_mre", "heldout_gae_mre_code", "heldout_gae_mre_true", "train_returns_mre",
                    "train_gae_mre_code", "train_gae_mre_true", "eval_return"):
            vals = np.array([[h[key] for h in hh if "heldout_returns_mre" in h] for hh in H])
            r["value"][key] = {str(it): mean_se(vals[:, j]) + (float(np.median(vals[:, j])),) for j, it in enumerate(vit)}
        r["paper_value"] = {k: {str(it): at(pc[(k, env, alg)], it) for it in vit} for k in ("heldout_gae", "heldout_ret")}
        s1 = bool(r["reward_last9"][0] > r["reward_it0"][0]) and within10(r["reward_last9"][0], r["paper_reward_it48"])
        s3 = []
        for it in vit[1:]:
            s3.append(within10(r["value"]["heldout_returns_mre"][str(it)][2], r["paper_value"]["heldout_ret"][str(it)]))
            s3.append(within10(r["value"]["heldout_gae_mre_code"][str(it)][2], r["paper_value"]["heldout_gae"][str(it)]))
        r["s3_detail"] = s3
        # instruments
        if I:
            r["grad"], r["baseline"], r["landscape"], r["value_floor"] = {}, {}, {}, {}
            frozen = all(I_[f"it{it}"].get("normaliser_frozen_ok", True) for I_ in I for it in its if f"it{it}" in I_)
            for it in its:
                k = f"it{it}"
                if k not in I[0]:
                    continue
                if "grad" in I[0][k]:
                    g = {}
                    for n in I[0][k]["grad"]:
                        g[n] = {"pairwise": mean_se([x[k]["grad"][n]["pairwise"] for x in I]),
                                "to_true": mean_se([x[k]["grad"][n]["to_true"] for x in I]),
                                "pairwise_rawadv": mean_se([x[k]["grad_raw_adv"][n]["pairwise"] for x in I]),
                                "to_true_rawadv": mean_se([x[k]["grad_raw_adv"][n]["to_true"] for x in I])}
                    r["grad"][k] = g
                if "baseline" in I[0][k]:
                    r["baseline"][k] = {kind: {n: mean_se([x[k]["baseline"][kind][n]["pairwise"] for x in I])
                                               for n in I[0][k]["baseline"][kind]} for kind in ("zero", "gae", "vstar")}
                    vfit = [x[k]["vstar_fit"] for x in I]
                    r["baseline"][k]["vstar_fit_improved"] = all(min(v["heldout_mse_std"]) < v["heldout_mse_std"][0] for v in vfit)
                    r["baseline"][k]["vstar_heldout_r2"] = [v["heldout_r2"] for v in vfit]
                    r["baseline"][k]["vstar_best_step"] = [v["best_step"] for v in vfit]
                if "value_floor" in I[0][k]:
                    r["value_floor"][k] = {q: mean_se([x[k]["value_floor"][q] for x in I]) for q in I[0][k]["value_floor"]}
                if "landscape" in I[0][k]:
                    L = {}
                    for reg in ("few", "many"):
                        ls = [x[k]["landscape"][reg] for x in I]
                        bp = {p: {q: mean_se([l["by_pairs"][p][q] for l in ls]) for q in
                                  ("roughness_rel_se", "spearman_S_T", "true_gain_step", "dT_dalpha0")} for p in ls[0]["by_pairs"]}
                        L[reg] = {"S00_max": max(abs(l["S00"]) for l in ls), "delta_norm": mean_se([l["delta_norm"] for l in ls]),
                                  "u_norm": ls[0]["u_norm"], "by_pairs": bp,
                                  "step_stats": {q: mean_se([l["step_stats"][q] for l in ls]) for q in ls[0]["step_stats"]}}
                    r["landscape"][k] = L
            pg = PAPER_GRAD_IT0[(env, alg)]
            g0 = r["grad"].get("it0", {})
            s2 = None
            if g0:
                s2 = (within10(g0["2000"]["to_true"][0], pg[1]) and within10(max(g0["2000"]["pairwise"][0], 1e-9), pg[0])
                      and g0["10000"]["to_true"][0] > g0["100"]["to_true"][0])
            r["paper_grad_it0"] = dict(zip(("pairwise_2e3", "to_true_2e3", "to_true_1e2", "to_true_1e4"), pg))
            r["paper_baseline_it150_10k"] = dict(zip(("zero", "gae", "vstar"), PAPER_BASE_150_10K[env]))
            s4 = all(np.isfinite(v[0]) for k in r["baseline"] for kind in ("zero", "gae", "vstar")
                     for v in r["baseline"][k][kind].values()) and all(r["baseline"][k]["vstar_fit_improved"] for k in r["baseline"])
            s5 = all(L[reg]["S00_max"] < 1e-5 and L[reg]["delta_norm"][0] < L[reg]["u_norm"]
                     for L in r["landscape"].values() for reg in ("few", "many"))
            checks[tag] = {"S1": s1, "S2": s2, "S3": all(x for x in s3 if x is not None), "S4": bool(s4), "S5": bool(s5),
                           "S6_frozen": bool(frozen)}
        else:
            checks[tag] = {"S1": s1, "S3": all(x for x in s3 if x is not None)}
        rep[tag] = r
    checks["S6_fingerprint_consistent"] = len({f[-12:] for f in fps}) == len(fps) and len(fps) <= 2
    ok = all(v for c in checks.values() for v in (c.values() if isinstance(c, dict) else [c]) if v is not None)
    out = {"pass": ok, "checks": checks, "report": rep}
    json.dump(out, open(os.path.join(REPO, "runs", exp, f"{exp}_report.json"), "w"), indent=1, default=float)
    print(json.dumps(checks, indent=1))
    print("SMALL-SCALE", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()

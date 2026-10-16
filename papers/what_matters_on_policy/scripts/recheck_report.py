"""Recheck fix-stage report (README 'Recheck (2026-10-15)'; pre-declaration).

(1)  lr horizon on the halfcheetah PROXY, n = 5 paired seeds 0-4 (same keys as full-scale):
    pre-fix  = full-scale lanes (tag full; lr -> 0 over 2M frames),
    post-fix = REC:= P1 lane X5_rec_lr_to0_1M, A5a:= P1 lane X6_lr1em4_to0_1M (configs identical to REC/A5a after
               the fix), A2b/A4/A7:= the recheck GPU run (tag recheck_lrfix).
    Per arm: ratio of means to its own REC, paired diff +- SE, t95 of rho, paired t p, Holm over the re-run family,
    the three full-scale readings (pre-declared  rule, generic rule, interval reading) via scripts/report.py.
(2)  faithful env (gymnasium HalfCheetah-v4, 1M, 10 eval episodes,  horizon for homogeneity): the authors'
    modules vs our learner, seeds 0-9 for REC/A7 (5-9 fresh) and 0-4 for A2b/A5a; per code and pooled; the code x arm
    interaction (Welch on the per-seed paired differences) is the pre-declared test of H0 "one algorithm".

  uv run python scripts/recheck_report.py  -> runs/recheck/fix/recheck_report.json + stdout
"""
import glob
import json
import math
import os
import sys

import numpy as np
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_jax_bench.stats import holm  # noqa: E402
from report import judge, judge_ci, judge_generic, paper_ratio, parse_paper  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OO = os.path.join(REPO, "runs", "recheck", "official-oracle")
OUT = os.path.join(REPO, "runs", "recheck", "fix")
PAPER_BOOT = {"A2b_initstd_1": (0.52, 0.61), "A4_lambda_0p95": (0.61, 1.13), "A5a_lr_1em4": (0.73, 0.83),
              "A7_valuenorm_none": (0.65, 0.88)}          # README full-scale table, raw-row bootstrap 
SEEDS = [0, 1, 2, 3, 4]


def lanes(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        for l in json.load(open(f))["lanes"]:
            out.setdefault(l["arm"], {})[l["seed"]] = (float("nan") if l["nonfinite"] else l["score"], l)
    return out


def contrast(x, r):
    x, r = np.asarray(x, float), np.asarray(r, float)
    d = x - r
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    tq = st.t.ppf(0.975, n - 1)
    return {"n": n, "arm_mean": float(x.mean()), "arm_se": float(x.std(ddof=1) / math.sqrt(n)),
            "rec_mean": float(r.mean()), "rec_se": float(r.std(ddof=1) / math.sqrt(n)),
            "rho": float(x.mean() / r.mean()), "diff": float(d.mean()), "diff_se": float(se),
            "rho_t95": [float(1 + (d.mean() - tq * se) / r.mean()), float(1 + (d.mean() + tq * se) / r.mean())],
            "p": float(st.ttest_rel(x, r).pvalue), "wins": f"{int((d > 0).sum())}/{n}"}


def proxy_part(P):
    pre = lanes(os.path.join(REPO, "runs", "full", "halfcheetah", "seed_*", "metrics.json"))
    p1 = lanes(os.path.join(REPO, "runs", "recheck", "learner-strength", "p1", "seed_*", "metrics.json"))
    fix = lanes(os.path.join(REPO, "runs", "recheck_lrfix", "halfcheetah", "seed_*", "metrics.json"))
    post = {"REC": p1["X5_rec_lr_to0_1M"], "A5a_lr_1em4": p1["X6_lr1em4_to0_1M"]}
    post.update({a: fix[a] for a in ("A2b_initstd_1", "A4_lambda_0p95", "A7_valuenorm_none") if a in fix})
    # the post-fix lr horizon really is the training length on every post-fix lane
    for a, d in post.items():
        for s, (_, l) in d.items():
            assert l["resolved_config"].get("lr_horizon_frames") in (1001472, None), (a, s)
    res = {"pre": {}, "post": {}, "rec_pre_vs_post": contrast([post["REC"][s][0] for s in SEEDS],
                                                               [pre["REC"][s][0] for s in SEEDS])}
    for tag, src in (("pre", pre), ("post", post)):
        arms = [a for a in ("A2b_initstd_1", "A4_lambda_0p95", "A5a_lr_1em4", "A7_valuenorm_none") if a in src]
        pv = {}
        for a in arms:
            c = contrast([src[a][s][0] for s in SEEDS], [src["REC"][s][0] for s in SEEDS])
            res[tag][a] = c
            pv[a] = c["p"]
        hp = holm(pv)
        for a in arms:
            c = res[tag][a]
            rp, spread, _, _ = paper_ratio(P, a, "halfcheetah")
            c["holm_p_refamily"] = float(hp[a]["p_holm"])
            c["paper_rho"], c["paper_spread"] = float(rp), [float(spread[0]), float(spread[1])]
            c["delta_pct"] = float(100 * (c["rho"] - rp) / rp)
            c["P"] = judge(c["rho"], rp, spread, hp[a]["p_holm"])
            c["G"] = judge_generic(c["rho"], rp, spread)
            c["I"] = judge_ci(c["rho_t95"], rp, PAPER_BOOT[a])
            if tag == "post":
                pre_c = res["pre"][a]
                c["arm_post_minus_pre"] = contrast([src[a][s][0] for s in SEEDS], [pre[a][s][0] for s in SEEDS])
                c["rho_change"] = float(c["rho"] - pre_c["rho"])
    # lr actually logged by the post-fix GPU lanes: first and last chunk mean
    lr_log = {}
    for a in ("A2b_initstd_1", "A4_lambda_0p95", "A7_valuenorm_none"):
        if a in fix:
            dg = fix[a][0][1]["diag"]
            if "lr" in dg[0]:
                lr_log[a] = [float(dg[0]["lr"]), float(dg[-1]["lr"])]
    res["post_lr_chunk_mean_first_last"] = lr_log
    return res


def v4_part():
    MAP_OFF = {"rec_pre": "REC", "rec_pre_std1": "A2b", "rec_pre_lr1em4": "A5a", "rec_pre_vnone": "A7"}
    MAP_OURS = {"REC": "REC", "A2b_initstd_1": "A2b", "A5a_lr_1em4": "A5a", "A7_valuenorm_none": "A7"}

    def load(sub, mapping, key):
        out = {}
        for f in glob.glob(f"{OO}/{sub}/*.json"):
            r = json.load(open(f))
            if r.get("overlap") or not r["curve"] or r["curve"][-1][0] < 1_000_000:
                continue
            if sub == "ours" and r.get("lr_horizon_frames", 2000000) != 2000000:
                continue                       #  is homogeneous in the  horizon
            arm = mapping.get(r[key])
            if arm:
                out.setdefault(arm, {})[r["seed"]] = float(np.mean([c[1] for c in r["curve"][1:]]))
        return out

    codes = {"official": load("official", MAP_OFF, "cfg"), "ours": load("ours", MAP_OURS, "arm")}
    res = {"per_code": {}, "pooled": {}, "interaction": {}, "rec_level": {}}
    rng = np.random.default_rng(0)
    for arm in ("A2b", "A5a", "A7"):
        diffs, per = {}, {}
        for code, d in codes.items():
            common = sorted(set(d.get(arm, {})) & set(d.get("REC", {})))
            if len(common) < 2:
                continue
            x, r = [d[arm][s] for s in common], [d["REC"][s] for s in common]
            c = contrast(x, r)
            c["seeds"] = common
            fresh = [s for s in common if s >= 5]
            if len(fresh) >= 2:
                c["fresh_seeds_5_9"] = contrast([d[arm][s] for s in fresh], [d["REC"][s] for s in fresh])
            old = [s for s in common if s < 5]
            if len(old) >= 2:
                c["seeds_0_4"] = contrast([d[arm][s] for s in old], [d["REC"][s] for s in old])
            res["per_code"].setdefault(arm, {})[code] = c
            diffs[code] = np.array(x) - np.array(r)
            per[code] = (np.array(x), np.array(r))
        if len(per) == 2:
            a, b = diffs["ours"], diffs["official"]
            w = st.ttest_ind(a, b, equal_var=False)
            res["interaction"][arm] = {"ours_diff": float(a.mean()), "official_diff": float(b.mean()),
                                       "welch_t": float(w.statistic), "p": float(w.pvalue), "n": [len(a), len(b)]}
            X = np.concatenate([per[c][0] for c in per])
            R = np.concatenate([per[c][1] for c in per])
            bs = []
            for _ in range(10000):         # resample seeds within each code (stratified)
                ix = np.concatenate([rng.integers(0, len(per[c][0]), len(per[c][0])) + (0 if c == "official" else
                                     len(per["official"][0])) for c in ("official", "ours")])
                bs.append(X[ix].mean() / R[ix].mean())
            res["pooled"][arm] = {"n": len(X), "rho": float(X.mean() / R.mean()),
                                  "boot95": [float(v) for v in np.percentile(bs, [2.5, 97.5])]}
    o, t = codes["official"].get("REC", {}), codes["ours"].get("REC", {})
    if len(o) > 1 and len(t) > 1:
        a, b = np.array(list(t.values())), np.array(list(o.values()))
        res["rec_level"] = {"ours_mean": float(a.mean()), "ours_se": float(a.std(ddof=1) / math.sqrt(len(a))),
                            "official_mean": float(b.mean()), "official_se": float(b.std(ddof=1) / math.sqrt(len(b))),
                            "n": [len(a), len(b)], "welch_p": float(st.ttest_ind(a, b, equal_var=False).pvalue)}
    res["scores"] = {c: {arm: {str(s): round(v, 1) for s, v in sorted(d[arm].items())} for arm in d}
                     for c, d in codes.items()}
    # SA-T1b: post-fix (lr -> 0 at 1M) contrasts of our learner on v4: REC:= X5, A5a:= X6 (oracle-lens runs),
    # A2b/A7:= ours_fix/ (default horizon after); seeds 0-2
    post = {}
    for f in glob.glob(f"{OO}/ours/X*_s*.json") + glob.glob(f"{OO}/ours_fix/*.json"):
        r = json.load(open(f))
        if not r["curve"] or r["curve"][-1][0] < 1_000_000:
            continue
        hz = r.get("lr_horizon_frames") or r["resolved"].get("lr_horizon_frames")
        arm = {"X5_rec_lr_to0_1M": "REC", "X6_lr1em4_to0_1M": "A5a", "A2b_initstd_1": "A2b",
               "A7_valuenorm_none": "A7"}.get(r["arm"])
        if arm and (r["arm"].startswith("X") or hz == 1001472):
            post.setdefault(arm, {})[r["seed"]] = float(np.mean([c[1] for c in r["curve"][1:]]))
    res["post_fix_ours"] = {}
    for arm in ("A2b", "A5a", "A7"):
        common = sorted(set(post.get(arm, {})) & set(post.get("REC", {})))
        if len(common) >= 2:
            c = contrast([post[arm][s] for s in common], [post["REC"][s] for s in common])
            c["seeds"] = common
            pre_c = sorted(set(codes["ours"].get(arm, {})) & set(codes["ours"]["REC"]) & set(common))
            c["pre_fix_same_seeds"] = contrast([codes["ours"][arm][s] for s in pre_c],
                                               [codes["ours"]["REC"][s] for s in pre_c])
            res["post_fix_ours"][arm] = c
    return res


V4ARM = {"A2b": "A2b_initstd_1", "A5a": "A5a_lr_1em4", "A7": "A7_valuenorm_none"}


def judge_v4(P, v4):
    """The three full-scale readings applied to the faithful-env (v4) contrasts of each code (Holm over that code's
    v4 family) and to the post-fix rows of ours."""
    for code in ("ours", "official"):
        rows = {a: d[code] for a, d in v4["per_code"].items() if code in d}
        hp = holm({a: c["p"] for a, c in rows.items()})
        for a, c in rows.items():
            rp, spread, _, _ = paper_ratio(P, V4ARM[a], "halfcheetah")
            c["holm_p"] = hp[a]["p_holm"]
            c["paper_rho"], c["delta_pct"] = float(rp), float(100 * (c["rho"] - rp) / rp)
            c["P"] = judge(c["rho"], rp, spread, hp[a]["p_holm"])
            c["G"] = judge_generic(c["rho"], rp, spread)
            c["I"] = judge_ci(c["rho_t95"], rp, PAPER_BOOT[V4ARM[a]])
    rows = v4.get("post_fix_ours", {})
    hp = holm({a: c["p"] for a, c in rows.items()}) if rows else {}
    for a, c in rows.items():
        rp, spread, _, _ = paper_ratio(P, V4ARM[a], "halfcheetah")
        c["paper_rho"], c["delta_pct"] = float(rp), float(100 * (c["rho"] - rp) / rp)
        c["P"] = judge(c["rho"], rp, spread, hp[a]["p_holm"])
        c["G"] = judge_generic(c["rho"], rp, spread)
        c["I"] = judge_ci(c["rho_t95"], rp, PAPER_BOOT[V4ARM[a]])


def main():
    P = parse_paper()
    rep = {"proxy_lr_fix": proxy_part(P), "v4": v4_part()}
    judge_v4(P, rep["v4"])
    os.makedirs(OUT, exist_ok=True)
    json.dump(rep, open(os.path.join(OUT, "recheck_report.json"), "w"), indent=1)
    pr = rep["proxy_lr_fix"]
    c = pr["rec_pre_vs_post"]
    print(f"REC post-fix (X5) {c['arm_mean']:.0f}+-{c['arm_se']:.0f} vs pre {c['rec_mean']:.0f}+-{c['rec_se']:.0f}: "
          f"ratio {c['rho']:.3f} diff {c['diff']:+.0f}+-{c['diff_se']:.0f} p {c['p']:.3f} wins {c['wins']}")
    for tag in ("pre", "post"):
        for a, c in pr[tag].items():
            print(f"{tag:4s} {a:18s} {c['arm_mean']:6.0f}+-{c['arm_se']:4.0f} rho {c['rho']:.3f} "
                  f"t95 [{c['rho_t95'][0]:.3f},{c['rho_t95'][1]:.3f}] diff {c['diff']:+6.0f}+-{c['diff_se']:4.0f} "
                  f"p {c['p']:.3f} holm {c['holm_p_refamily']:.3f} paper {c['paper_rho']:.3f} d% {c['delta_pct']:+.1f} "
                  f"P={c['P'][0]} G={c['G'][0]} I={c['I'][0]}")
    print("lr logged (first/last chunk mean):", pr["post_lr_chunk_mean_first_last"])
    v = rep["v4"]
    for arm, d in v["per_code"].items():
        for code, c in d.items():
            line = (f"v4 {arm} {code:8s} n {c['n']} rho {c['rho']:.3f} t95 [{c['rho_t95'][0]:.3f},{c['rho_t95'][1]:.3f}]"
                    f" diff {c['diff']:+.0f}+-{c['diff_se']:.0f} p {c['p']:.3f} wins {c['wins']}")
            if "fresh_seeds_5_9" in c:
                f = c["fresh_seeds_5_9"]
                line += f" | fresh 5-9: rho {f['rho']:.3f} diff {f['diff']:+.0f}+-{f['diff_se']:.0f} wins {f['wins']}"
            line += f" | d% {c['delta_pct']:+.1f} P={c['P'][0]} G={c['G'][0]} I={c['I'][0]}"
            print(line)
        if arm in v["interaction"]:
            i = v["interaction"][arm]
            print(f"   interaction {arm}: ours {i['ours_diff']:+.0f} vs official {i['official_diff']:+.0f} "
                  f"Welch t {i['welch_t']:.2f} p {i['p']:.4f} n {i['n']}; pooled rho {v['pooled'][arm]['rho']:.3f} "
                  f"boot95 {v['pooled'][arm]['boot95']}")
    print("REC level:", v["rec_level"])
    for arm, c in v.get("post_fix_ours", {}).items():
        q = c["pre_fix_same_seeds"]
        print(f"v4 post-fix ours {arm} n {c['n']} rho {c['rho']:.3f} t95 [{c['rho_t95'][0]:.3f},{c['rho_t95'][1]:.3f}] "
              f"diff {c['diff']:+.0f}+-{c['diff_se']:.0f} | pre-fix same seeds rho {q['rho']:.3f} | "
              f"P={c['P'][0]} G={c['G'][0]} I={c['I'][0]}")


if __name__ == "__main__":
    main()

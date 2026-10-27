"""Metrics and pre-declared statistics from runs/<out>/seed_<n>/metrics.json.

Score definitions:  per seed, per arm, from the exact per-bin sums/counts of END-stamped
episode returns (raw MinAtar reward):
    EARLY = pooled mean return of episodes ending in the first `early_steps` steps
    BEST  = max over windows of `best_steps` steps (stride = bin width) of the pooled mean return in the window
Tests: one-sided paired t (declared direction) + percentile bootstrap CI, Holm across the declared family.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
import numpy as np
from scipy import stats as sps
from rl_jax_bench import stats as S


def buffer_demo_share_chunk_mean(n_demo, step, chunk, cap):
    """Mean over the env steps of the chunk ending at `step` of the buffer's demo share n_demo / (n_demo + min(t, cap)).
    The sampled demo share in the history is a CHUNK average, so the uniform-sampling baseline must be the chunk average too
    (the end-of-chunk value inflated the 10 k point by 1.24x).  `cap` is per.agent_capacity of the config.
    Transitions still pending their n-step window (n_step - 1 = 9) are not subtracted: < 0.05 % of the buffer from step 10 k."""
    t = np.arange(step - chunk + 1, step + 1)
    return float(np.mean(n_demo / (n_demo + np.minimum(t, cap))))


def agent_capacity():
    import yaml
    return yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src/configs/default.yaml")))["per"]["agent_capacity"]


def load(out_dir, game, demonstrator):
    runs = {}
    meta = None
    for p in sorted(glob.glob(os.path.join(out_dir, "seed_*", "metrics.json")), key=lambda x: int(x.split("seed_")[1].split("/")[0])):
        seed = int(p.split("seed_")[1].split("/")[0])
        cell = json.load(open(p)).get(f"{game}/{demonstrator}")
        if cell is None:
            continue
        meta = cell["meta"]
        for arm, r in cell.items():
            if arm != "meta":
                runs.setdefault(arm, {})[seed] = r
    return runs, meta


def window_scores(bin_sum, bin_cnt, bin_w, early_steps, best_steps):
    ne = max(1, int(round(early_steps / bin_w)))
    nw = max(1, int(round(best_steps / bin_w)))
    s, c = np.asarray(bin_sum), np.asarray(bin_cnt)
    early = s[:ne].sum() / max(c[:ne].sum(), 1)
    best, best_at = -np.inf, 0
    cs, cc = np.concatenate([[0], np.cumsum(s)]), np.concatenate([[0], np.cumsum(c)])
    for i in range(0, len(s) - nw + 1):
        n = cc[i + nw] - cc[i]
        if n >= 3:
            m = (cs[i + nw] - cs[i]) / n
            if m > best:
                best, best_at = m, i
    final = s[-nw:].sum() / max(c[-nw:].sum(), 1)
    return float(early), float(best), float(final), best_at * bin_w


def one_sided(p_two, diff):
    return p_two / 2 if diff > 0 else 1 - p_two / 2


def analyse(out_dir, game, demonstrator, early_frac, best_frac, demo_info=None):
    runs, meta = load(out_dir, game, demonstrator)
    steps, bw = meta["steps"], meta["bin_width"]
    early_steps, best_steps = early_frac * steps, best_frac * steps
    info = meta["demonstrator_info"]
    res = dict(game=game, demonstrator=demonstrator, steps=steps, bin_width=bw, early_steps=early_steps, best_steps=best_steps,
               demonstrator_true_mean=info["true_mean"], demonstrator_true_se=info["true_se"],
               demo_best_episode=info["best"], demo_worst_episode=info["worst"], arms={})
    per = {}
    for arm, rr in runs.items():
        seeds = sorted(rr)
        if arm == "IMIT":
            v = np.array([rr[s]["online_return"] for s in seeds])
            res["arms"][arm] = dict(n=len(v), online_return_mean=float(v.mean()), se=S.mean_se(v)[1], per_seed=v.tolist())
            per[arm] = dict(seeds=seeds, early=v, best=v, final=v)
            continue
        sc = np.array([window_scores(rr[s]["bin_sum"], rr[s]["bin_cnt"], bw, early_steps, best_steps) for s in seeds])
        per[arm] = dict(seeds=seeds, early=sc[:, 0], best=sc[:, 1], final=sc[:, 2], best_at=sc[:, 3])
        curve = np.array([np.asarray(rr[s]["bin_sum"]) / np.maximum(np.asarray(rr[s]["bin_cnt"]), 1) for s in seeds])
        res["arms"][arm] = dict(n=len(seeds), early=S.mean_se(sc[:, 0]), best=S.mean_se(sc[:, 1]), final=S.mean_se(sc[:, 2]),
                                best_at_mean=float(sc[:, 3].mean()), curve_mean=curve.mean(0).tolist())
    # --- primary family (sub-optimal demonstrator): H1 (DQfD vs PDD early) and H2b (DQfD best vs demonstrator true mean)
    fam, tests = {}, {}
    if "DQfD" in per and "PDD" in per:
        a, b = per["DQfD"]["early"], per["PDD"]["early"]
        pc = S.paired_compare(a, b)
        tests["H1_early_DQfD_vs_PDD"] = dict(pc, p_one_sided=one_sided(pc["t_p"], pc["diff_mean"]))
        fam["H1"] = tests["H1_early_DQfD_vs_PDD"]["p_one_sided"]
    if "DQfD" in per:
        d = per["DQfD"]["best"] - info["true_mean"]
        t = sps.ttest_1samp(d, 0.0)
        tests["H2b_best_DQfD_vs_demonstrator_mean"] = dict(diff_mean=float(d.mean()), diff_se=S.mean_se(d)[1], t95=S.t_interval(d),
                                                         boot95=S.boot_ci(d), p_one_sided=one_sided(float(t.pvalue), float(d.mean())),
                                                         wins=int((d > 0).sum()), n=len(d),
                                                         bar_se_ok=bool(info["true_se"] < 0.1 * abs(d.mean())) if d.mean() != 0 else False)
        fam["H2b"] = tests["H2b_best_DQfD_vs_demonstrator_mean"]["p_one_sided"]
        tests["H2a_best_vs_best_demo_episode"] = dict(frac_seeds_above=float(np.mean(per["DQfD"]["best"] > info["best"])),
                                                      mean_curve_best=float(np.max(res["arms"]["DQfD"]["curve_mean"])))
    res["holm_family"] = S.holm(fam) if fam else {}
    # --- secondary comparisons (descriptive here; Holm separately)
    sec = {}
    for other in ("IMIT", "NoMargin", "NoNstep", "ADET", "PDDconst", "RBS", "HER"):
        if "DQfD" in per and other in per:
            for metric in ("early", "best", "final"):
                a, b = per["DQfD"][metric], per[other][metric]
                if len(a) == len(b) and len(a) >= 2:
                    pc = S.paired_compare(a, b)
                    sec[f"DQfD_vs_{other}_{metric}"] = dict(diff_mean=pc["diff_mean"], diff_se=pc["diff_se"], boot95=pc["boot95"],
                                                          wins=pc["wins"], n=pc["n"], p_one_sided=one_sided(pc["t_p"], pc["diff_mean"]))
    if "IMIT" in per:
        v = per["IMIT"]["best"]
        sec["IMIT_minus_demonstrator_mean"] = dict(diff_mean=float(v.mean() - info["true_mean"]), se=S.mean_se(v)[1])
    res["secondary"] = sec
    res["tests"] = tests
    # --- catch-up: first bin where PDD's mean curve exceeds DQfD's and stays above 3 bins
    if "DQfD" in res["arms"] and "PDD" in res["arms"]:
        dq, pd = np.array(res["arms"]["DQfD"]["curve_mean"]), np.array(res["arms"]["PDD"]["curve_mean"])
        above = pd > dq
        cu = None
        for i in range(len(above) - 2):
            if above[i:i + 3].all():
                cu = i * bw; break
        res["catch_up_step"] = cu
    # --- up-sample ratio diagnostics
    hp = os.path.join(out_dir, f"history_{game}_{demonstrator}.json")
    if os.path.exists(hp):
        h = json.load(open(hp))
        n_demo = info["n_transitions"]; cap = agent_capacity()   # read from the config
        ups = {}
        for arm in ("DQfD", "HER"):
            if arm in h and h[arm]["interact"]:
                rows = h[arm]["interact"]
                xs = [r["step"] for r in rows]
                chunk = rows[1]["step"] - rows[0]["step"] if len(rows) > 1 else rows[0]["step"]
                ratio = [np.mean(r["demo_share"]) / buffer_demo_share_chunk_mean(n_demo, r["step"], chunk, cap) for r in rows]
                ups[arm] = dict(step=xs[:: max(1, len(xs) // 10)], ratio=ratio[:: max(1, len(ratio) // 10)])
        res["upsample_ratio"] = ups
    return res


def fmt(res):
    L = [f"## {res['game']} / {res['demonstrator']}  (steps {res['steps']:.0f}, early {res['early_steps']:.0f}, best-window {res['best_steps']:.0f})",
         f"demonstrator true mean {res['demonstrator_true_mean']:.3f} +- {res['demonstrator_true_se']:.3f}; best/worst demo episode {res['demo_best_episode']}/{res['demo_worst_episode']}",
         "", "| arm | n | EARLY | BEST | FINAL |", "|---|---|---|---|---|"]
    for arm, r in res["arms"].items():
        if arm == "IMIT":
            L.append(f"| IMIT | {r['n']} | - | {r['online_return_mean']:.3f} +- {r['se']:.3f} (eps=0.01 online) | - |")
        else:
            f = lambda t: f"{t[0]:.3f} +- {t[1]:.3f}" if t[1] is not None else f"{t[0]:.3f}"
            L.append(f"| {arm} | {r['n']} | {f(r['early'])} | {f(r['best'])} | {f(r['final'])} |")
    L.append("")
    for k, v in res["tests"].items():
        L.append(f"- {k}: " + json.dumps({a: b for a, b in v.items() if a not in ('diff_per_seed',)}, default=float))
    L.append("- Holm family: " + json.dumps(res["holm_family"]))
    L.append(f"- catch-up step (PDD > DQfD for 3 bins): {res.get('catch_up_step')}")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--game", required=True); ap.add_argument("--demonstrator", default="subopt")
    ap.add_argument("--early-frac", type=float, default=0.02); ap.add_argument("--best-frac", type=float, default=0.015)
    a = ap.parse_args()
    res = analyse(a.out, a.game, a.demonstrator, a.early_frac, a.best_frac)
    json.dump(res, open(os.path.join(a.out, f"analysis_{a.game}_{a.demonstrator}.json"), "w"), indent=1, default=float)
    print(fmt(res))

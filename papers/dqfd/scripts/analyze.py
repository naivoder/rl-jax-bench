"""Full-scale analysis (tests declared before the runs): merges several runs/<out> directories
per game, computes per-seed EARLY / BEST / FINAL scores from the exact per-bin sums, and the pre-declared tests.

usage: python scripts/analyze.py --out runs/full/analysis.json

Windows (fractions of the interaction budget, bin-aligned): EARLY = first 5 %, BEST = best 3 % window (>= 3 episodes),
FINAL = last 3 %.  A window holding < 3 episodes of a seed is widened (EARLY forward, FINAL backward) bin by bin until it
holds >= 3 episodes (pre-declared fallback; the number of widened windows is reported).
Tests: one-sided paired t (declared direction) + bootstrap CI of the paired difference (rl_jax_bench.stats), Holm per family.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
import numpy as np
from scipy import stats as sps
from rl_jax_bench import stats as S
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES = ["Breakout", "SpaceInvaders", "Asterix", "Seaquest"]
E10 = "+eps0=0.1+eps1=0.1"
DIRS = {   # game -> run directories whose metrics.json hold this game's cell (merged per (arm, seed))
    "Breakout": ["runs/breakout_first5", "runs/breakout_probe", "runs/full/breakout_p1", "runs/full/breakout_p2", "runs/full/breakout_pddconst"],   # + PDDconst seeds 5-9 (verification run)
    "SpaceInvaders": ["runs/full/SpaceInvaders", "runs/full/SpaceInvaders_pddconst"],
    "Asterix": ["runs/full/Asterix", "runs/full/Asterix_pddconst"],
    "Seaquest": ["runs/full/Seaquest", "runs/full/Seaquest_pddconst"],
}


def one_sided(p_two, diff, direction=+1):
    """One-sided p for H: direction * diff > 0."""
    return p_two / 2 if direction * diff > 0 else 1 - p_two / 2


def load_game(game, demonstrator="subopt"):
    arms, meta, hist = {}, None, {}
    for d in DIRS[game]:
        d = os.path.join(REPO, d)
        for p in glob.glob(os.path.join(d, "seed_*", "metrics.json")):
            seed = int(p.split("seed_")[1].split("/")[0])
            cell = json.load(open(p)).get(f"{game}/{demonstrator}")
            if not cell:
                continue
            m = cell["meta"]
            if meta is None or (meta["steps"], meta["bin_width"]) != (m["steps"], m["bin_width"]):
                if meta is not None:
                    raise SystemExit(f"{game}: inconsistent steps/bin width in {d}")
                meta = m
            for arm, r in cell.items():
                if arm != "meta":
                    arms.setdefault(arm, {})[seed] = dict(r, _dir=os.path.relpath(d, REPO))
        hp = os.path.join(d, f"history_{game}_{demonstrator}.json")
        if os.path.exists(hp):
            hist[os.path.relpath(d, REPO)] = json.load(open(hp))
    return arms, meta, hist


def windows(bin_sum, bin_cnt, early_bins, win_bins, min_eps=3):
    s, c = np.asarray(bin_sum, float), np.asarray(bin_cnt, float)
    e = early_bins
    while c[:e].sum() < min_eps and e < len(s):
        e += 1
    f = win_bins
    while c[len(s) - f:].sum() < min_eps and f < len(s):
        f += 1
    early = s[:e].sum() / max(c[:e].sum(), 1)
    final = s[len(s) - f:].sum() / max(c[len(s) - f:].sum(), 1)
    cs, cc = np.concatenate([[0], np.cumsum(s)]), np.concatenate([[0], np.cumsum(c)])
    best, at = -np.inf, 0
    for i in range(0, len(s) - win_bins + 1):
        n = cc[i + win_bins] - cc[i]
        if n >= min_eps:
            m = (cs[i + win_bins] - cs[i]) / n
            if m > best:
                best, at = m, i
    return dict(early=float(early), best=float(best), final=float(final), best_at=at, widened_early=e > early_bins,
                widened_final=f > win_bins)


def per_arm_scores(arms, meta, early_frac, best_frac):
    bw, steps = meta["bin_width"], meta["steps"]
    eb, wb = max(1, int(round(early_frac * steps / bw))), max(1, int(round(best_frac * steps / bw)))
    out = {}
    for arm, rr in arms.items():
        seeds = sorted(rr)
        if arm == "IMIT":
            v = np.array([rr[s]["online_return"] for s in seeds])
            out[arm] = dict(seeds=seeds, early=v, best=v, final=v, widened=0, greedy=None)
            continue
        sc = [windows(rr[s]["bin_sum"], rr[s]["bin_cnt"], eb, wb) for s in seeds]
        gr = [np.mean(rr[s]["final_greedy"]) if "final_greedy" in rr[s] else np.nan for s in seeds]
        out[arm] = dict(seeds=seeds, early=np.array([x["early"] for x in sc]), best=np.array([x["best"] for x in sc]),
                        final=np.array([x["final"] for x in sc]), best_at=np.array([x["best_at"] * bw for x in sc]),
                        widened=int(sum(x["widened_early"] or x["widened_final"] for x in sc)), greedy=np.array(gr),
                        curve=np.array([np.asarray(rr[s]["bin_sum"]) / np.maximum(np.asarray(rr[s]["bin_cnt"]), 1) for s in seeds]),
                        gap=np.array([rr[s].get("final_gap_mean", np.nan) for s in seeds]))
    return out, eb, wb


def pair(per, a, b, metric, direction=+1):
    """Paired (by seed) comparison of arm a against arm b on `metric`; H: direction*(a-b) > 0."""
    if a not in per or b not in per:
        return None
    common = sorted(set(per[a]["seeds"]) & set(per[b]["seeds"]))
    if len(common) < 2:
        return None
    ia = [per[a]["seeds"].index(s) for s in common]; ib = [per[b]["seeds"].index(s) for s in common]
    x, y = per[a][metric][ia], per[b][metric][ib]
    pc = S.paired_compare(x, y)
    return dict(a=a, b=b, metric=metric, n=len(common), diff_mean=pc["diff_mean"], diff_se=pc["diff_se"], t95=pc["t95"], boot95=pc["boot95"],
                wins=pc["wins"] if direction > 0 else len(common) - pc["wins"], p_one_sided=one_sided(pc["t_p"], pc["diff_mean"], direction),
                mean_a=float(x.mean()), mean_b=float(y.mean()))


def vs_const(per, a, metric, const, const_se, direction=+1):
    if a not in per:
        return None
    d = per[a][metric] - const
    d = d[~np.isnan(d)]          # seeds without a final greedy eval (first five Breakout seeds) are dropped for that descriptor
    if len(d) < 2:
        return None
    t = sps.ttest_1samp(d, 0.0)
    m, se = S.mean_se(d)
    return dict(a=a, metric=metric, n=len(d), diff_mean=m, diff_se=se, t95=S.t_interval(d), boot95=S.boot_ci(d),
                wins=int((direction * d > 0).sum()), p_one_sided=one_sided(float(t.pvalue), m, direction),
                bar_se=const_se, bar_se_ok=bool(const_se < 0.1 * abs(m)) if m != 0 else False, mean_a=float(np.nanmean(per[a][metric])))


def catch_up(per, a, b, bw):
    if a not in per or b not in per:
        return None
    ca, cb = per[a]["curve"].mean(0), per[b]["curve"].mean(0)
    above = cb > ca
    for i in range(len(above) - 2):
        if above[i:i + 3].all():
            return int(i * bw)
    return None


from src.eval import buffer_demo_share_chunk_mean, agent_capacity  # noqa: E402


def upsample(hist, meta, game, n_demo):
    """demo share of sampled batches / demo share of the buffer, DQfD, mean over every run dir that holds DQfD."""
    rows = {}
    for d, h in hist.items():
        if "DQfD" not in h or not h["DQfD"].get("interact"):
            continue
        ir = h["DQfD"]["interact"]
        chunk = ir[1]["step"] - ir[0]["step"] if len(ir) > 1 else ir[0]["step"]
        for r in ir:   # chunk-MEAN buffer share (the sampled share is a chunk average), cap from the config
            share = np.mean(r["demo_share"]); step = r["step"]
            rows.setdefault(step, []).append(share / buffer_demo_share_chunk_mean(n_demo, step, chunk, agent_capacity()))
    ks = sorted(rows)
    pick = ks[:: max(1, len(ks) // 8)] + [ks[-1]] if ks else []
    return {int(k): float(np.mean(rows[k])) for k in sorted(set(pick))}


def greedy_bar(game):
    """Demonstrator's own eps = 0 return (scripts/demonstrator_greedy.py): the fair bar for the end-of-run GREEDY descriptor."""
    p = os.path.join(REPO, "runs/full/demonstrator_greedy.json")
    if os.path.exists(p):
        r = json.load(open(p)).get(f"{game}/subopt/eps0.0")
        if r:
            return r["mean"], r["se"]
    return None, None


def analyse_game(game, early_frac=0.05, best_frac=0.03):
    arms, meta, hist = load_game(game)
    if not arms:
        return None
    info = meta["demonstrator_info"]
    per, eb, wb = per_arm_scores(arms, meta, early_frac, best_frac)
    bw = meta["bin_width"]
    tm, tse = info["true_mean"], info["true_se"]
    res = dict(game=game, steps=meta["steps"], bin_width=bw, early_bins=eb, window_bins=wb, demonstrator=dict(true_mean=tm, true_se=tse,
               best_episode=info["best"], worst_episode=info["worst"], n_transitions=info["n_transitions"], n_episodes=info["n_episodes"]),
               arms={})
    for a, p in per.items():
        f = lambda v: [float(np.mean(v)), (S.mean_se(v)[1] if len(v) > 1 else None)]
        res["arms"][a] = dict(n=len(p["seeds"]), seeds=p["seeds"], early=f(p["early"]), best=f(p["best"]), final=f(p["final"]),
                              greedy=(f(p["greedy"][~np.isnan(p["greedy"])]) if p["greedy"] is not None and (~np.isnan(p["greedy"])).any() else None),
                              gap=(f(p["gap"][~np.isnan(p["gap"])]) if "gap" in p and (~np.isnan(p["gap"])).any() else None),
                              widened_windows=p["widened"], per_seed_final=p["final"].tolist(), per_seed_early=p["early"].tolist(),
                              per_seed_best=p["best"].tolist(), dirs=sorted({arms[a][s]["_dir"] for s in p["seeds"]}))
    T = {}
    T["H1_early_DQfD_gt_PDD"] = pair(per, "DQfD", "PDD", "early")
    T["H2b_prime_final_DQfD_gt_demonstrator"] = vs_const(per, "DQfD", "final", tm, tse)
    T["H2b_best_window_biased_DQfD_gt_demonstrator"] = vs_const(per, "DQfD", "best", tm, tse)
    if "DQfD" in per:
        T["H2a_best_gt_best_demo_episode"] = dict(frac_seeds=float(np.mean(per["DQfD"]["best"] > info["best"])),
                                                 mean_curve_best=float(np.max(per["DQfD"]["curve"].mean(0))), best_demo=info["best"])
        gm, gse = greedy_bar(game)
        if not np.isnan(per["DQfD"]["greedy"]).all() and gm is not None:
            T["greedy_final_DQfD_vs_demonstrator_greedy"] = vs_const(dict(DQfD=dict(g=per["DQfD"]["greedy"])), "DQfD", "g", gm, gse)
            T["greedy_final_DQfD_vs_demonstrator_eps001"] = vs_const(dict(DQfD=dict(g=per["DQfD"]["greedy"])), "DQfD", "g", tm, tse)
    if "IMIT" in per:
        T["C6_IMIT_lt_demonstrator"] = vs_const(per, "IMIT", "final", tm, tse, direction=-1)
    S1 = {}
    for x in ("IMIT", "NoMargin", "NoNstep", "ADET"):
        for m in ("early", "final"):
            S1[f"DQfD_gt_{x}_{m}"] = pair(per, "DQfD", x, m)
    for x in ("NoMargin", "NoNstep", "ADET"):   # reverse direction descriptors (does the ablation end ABOVE DQfD?)
        S1[f"{x}_gt_DQfD_final"] = pair(per, x, "DQfD", "final")
    S2 = {}
    d10, p10 = "DQfD" + E10, "PDD+eps1=0.1"
    S2["DQfDe10_final_gt_demonstrator"] = vs_const(per, d10, "final", tm, tse)
    gm, gse = greedy_bar(game)
    if d10 in per and not np.isnan(per[d10]["greedy"]).all() and gm is not None:
        S2["DQfDe10_greedy_gt_demonstrator_greedy"] = vs_const({d10: dict(g=per[d10]["greedy"])}, d10, "g", gm, gse)
    S2["DQfDe10_early_gt_PDDe10"] = pair(per, d10, p10, "early")
    S2["DQfDe10_final_gt_PDDe10"] = pair(per, d10, p10, "final")
    S2["DQfDe10_final_gt_DQfD"] = pair(per, d10, "DQfD", "final")
    S2["DQfDe10_early_gt_PDD"] = pair(per, d10, "PDD", "early")
    S2["NoMarginE10_final_gt_DQfDe10"] = pair(per, "NoMargin" + E10, d10, "final")
    S2["DQfDe10_final_gt_NoMarginE10"] = pair(per, d10, "NoMargin" + E10, "final")
    S2["DQfDe10_early_gt_NoMarginE10"] = pair(per, d10, "NoMargin" + E10, "early")
    res["tests"] = {k: v for k, v in T.items() if v}
    res["secondary_S1"] = {k: v for k, v in S1.items() if v}
    res["secondary_S2"] = {k: v for k, v in S2.items() if v}
    res["catch_up_step_PDD_over_DQfD"] = catch_up(per, "DQfD", "PDD", bw)
    res["catch_up_step_PDDe10_over_DQfDe10"] = catch_up(per, d10, p10, bw)
    res["upsample_ratio_DQfD"] = upsample(hist, meta, game, info["n_transitions"])
    # lr probe (tuning parity): arms 'X+lr=...' compared with the lr 1e-4 reference on the SAME seeds over the first `probe_steps`
    res["lr_probe"] = {}
    return res, per, meta


def lr_probe(game="Breakout"):
    """Tuning parity: per base arm (PDD, DQfDe10) FINAL (last 3 %) at lr 3e-5 / 1e-4 / 3e-4 on the SAME seeds 0-4, paired against
    lr 1e-4; 'matters' (pre-declared rule) iff |diff| > 3 paired SE with 5/5 seeds of the same sign."""
    arms, meta, _ = load_game(game)
    bw = meta["bin_width"]
    nw = max(1, int(round(0.03 * meta["steps"] / bw)))
    fin = lambda name, s: (np.asarray(arms[name][s]["bin_sum"])[-nw:].sum() / max(np.asarray(arms[name][s]["bin_cnt"])[-nw:].sum(), 1))
    out = {}
    for base in ("PDD", "DQfD" + E10):
        row = {}
        for tag, name in (("3e-5", base + "+lr=3e-05"), ("1e-4", base), ("3e-4", base + "+lr=0.0003")):
            if name not in arms:
                continue
            seeds = sorted(set(arms[name]) & set(arms[base]) & set(range(5)))
            x = np.array([fin(name, s) for s in seeds]); ref = np.array([fin(base, s) for s in seeds]); d = x - ref
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None
            row[tag] = dict(seeds=seeds, final_mean=float(x.mean()), final_se=S.mean_se(x)[1], diff_vs_lr1e4=float(d.mean()) if tag != "1e-4" else 0.0,
                            diff_se=se, same_sign=int(max((d > 0).sum(), (d < 0).sum())), n=len(d),
                            matters=bool(tag != "1e-4" and se is not None and abs(d.mean()) > 3 * se and max((d > 0).sum(), (d < 0).sum()) == len(d)))
        out[base] = row
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "runs/full/analysis.json"))
    ap.add_argument("--games", default=",".join(GAMES))
    a = ap.parse_args()
    allres = {}
    for g in a.games.split(","):
        r = analyse_game(g)
        if r:
            allres[g] = r[0]
    fam = {}
    for g, r in allres.items():
        for k in ("H1_early_DQfD_gt_PDD", "H2b_prime_final_DQfD_gt_demonstrator"):
            if k in r["tests"]:
                fam[f"{k}[{g}]"] = r["tests"][k]["p_one_sided"]
    allres["primary_holm"] = S.holm(fam)
    for fname, key in (("S1", "secondary_S1"), ("S2", "secondary_S2")):
        f2 = {}
        for g, r in allres.items():
            if g in GAMES:
                for k, v in r[key].items():
                    f2[f"{k}[{g}]"] = v["p_one_sided"]
        allres[f"{fname}_holm"] = S.holm(f2)
    try:
        allres["lr_probe"] = lr_probe()
    except Exception as e:  # noqa
        allres["lr_probe"] = f"unavailable: {e}"
    json.dump(allres, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({"primary_holm": allres["primary_holm"]}, indent=1))

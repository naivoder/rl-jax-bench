"""Analysis of the pre-declared families: primary Holm family over the games flown (x H1, H2, H3), the secondary
families (src/eval.py), the tolerance/window family on Breakout, the C3-final family, the DESCRIPTIVE combined-policy comparison, crossover steps, and
the ablation differences.  Usage:  python scripts/analyze_amendments.py runs/full Breakout Asterix Seaquest SpaceInvaders  -> runs/full/analysis_a1.json + analysis_a1.md (amendment A1/A2 statistics; the pre-amendment analysis.json files are kept as they were)
All numbers are computed from runs/<dir>/<Game>/seed_<n>/metrics.json; nothing is hand-entered."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import stats as sps
from rl_jax_bench import stats as S
from src import eval as E


def per_seed(runs, arm, key, sl):
    seeds = sorted(runs[arm])
    return np.array([np.mean(np.asarray(runs[arm][s][key], float)[sl]) for s in seeds]), seeds


def first_sustained(a, b, margin=0.0):
    """First evaluation index j such that mean curve a >= b + margin at j and at every later j (sustained crossover); None if never."""
    ok = (np.asarray(a) >= np.asarray(b) + margin)
    for j in range(len(ok)):
        if ok[j:].all():
            return j
    return None


def first_reach(c, level):
    idx = np.nonzero(np.asarray(c) >= level)[0]
    return int(idx[0]) if len(idx) else None


def game_extra(out_dir, game, res):
    runs, meta = E.load(out_dir, game)
    n = meta["n_eval_points"]; steps = meta["steps"]; chunk = steps // n
    ea, fi = E.windows(n)
    mu = meta["guide"]["true_mean"]
    ex = {}
    seeds = sorted(runs["JSRL"])
    # D-comb: combined-policy EARLY of JSRL (ev_thr, current-stage h) vs PDD pi_e-alone EARLY (paired); DESCRIPTIVE
    thr, _ = per_seed(runs, "JSRL", "ev_thr", ea); pdd, _ = per_seed(runs, "PDD", "ev_alone", ea)
    ex["D-comb_early"] = E.paired(thr, pdd)
    # curves
    cur = {a: np.mean([np.asarray(runs[a][s]["ev_alone"], float) for s in sorted(runs[a])], 0) for a in runs}
    ex["crossover_point_JSRL_over_PDD_sustained"] = first_sustained(cur["JSRL"], cur["PDD"])
    ex["crossover_step_JSRL_over_PDD_sustained"] = None if ex["crossover_point_JSRL_over_PDD_sustained"] is None else ex["crossover_point_JSRL_over_PDD_sustained"] * chunk
    for arm in [a for a in ("JSRL", "PDD", "JSRL-Random") if a in cur]:
        j = first_reach(cur[arm][1:], 0.5 * mu)       # first point >= 0.5 mu_g (index in 1..n)
        ex[f"steps_to_half_guide_{arm}"] = None if j is None else (j + 1) * chunk
    # per-seed pi_e-alone points (descriptive): curve means at 5/10/25/50/100 % of the budget
    ex["curve_at_pct"] = {a: {str(p): float(cur[a][int(round(p / 100 * n))]) for p in (0, 5, 10, 25, 50, 75, 100)} for a in cur}
    # EARLY at the paper-like 5 % window (descriptor)
    # steps at which JSRL's curriculum reached h = 0
    if "first_eval_at_h0" in runs["JSRL"][seeds[0]]:
        f = [runs["JSRL"][s]["first_eval_at_h0"] for s in seeds]
        ex["JSRL_first_eval_at_h0"] = f
        ex["JSRL_steps_to_h0_median"] = None if any(x is None for x in f) else float(np.median(f) * chunk)
        ex["JSRL_never_reached_h0"] = int(sum(x is None for x in f))
    return ex, runs, meta


def tolerance_family(runs, meta):
    n = meta["n_eval_points"]; ea, fi = E.windows(n)
    out, ps = {}, {}
    if "PDD" not in runs:
        return out
    variants = [a for a in runs if a.startswith("JSRL+")]
    if not variants:
        return out
    base = {a: (per_seed(runs, a, "ev_alone", ea)[0], per_seed(runs, a, "ev_alone", fi)[0]) for a in ["JSRL", "PDD"] + variants}
    for a in variants:
        t = E.paired(base[a][0], base["PDD"][0]); out[f"V:{a}:EARLY-PDD"] = t; ps[f"V:{a}"] = t["p_one_sided"]
        out[f"desc:{a}-JSRL:EARLY"] = E.paired(base[a][0], base["JSRL"][0])
        out[f"desc:{a}-JSRL:FINAL"] = E.paired(base[a][1], base["JSRL"][1])
        out[f"desc:{a}:FINAL-PDD"] = E.paired(base[a][1], base["PDD"][1])
        out[f"desc:{a}:first_eval_at_h0"] = [runs[a][s].get("first_eval_at_h0") for s in sorted(runs[a])]
    out["holm_V"] = S.holm(ps)
    # C10: FINAL monotone in tolerance (0 < 0.05 < 0.15) and 0.15 vs 0.05 negative with one-sided p < 0.05
    tol = {"0": base.get("JSRL+tol=0"), "0.05": base["JSRL"], "0.15": base.get("JSRL+tol=0.15")}
    if all(v is not None for v in tol.values()):
        m = [float(tol[k][1].mean()) for k in ("0", "0.05", "0.15")]
        t = E.paired(tol["0.15"][1], tol["0.05"][1])
        p_less = t["p_two_sided"] / 2 if t["diff_mean"] < 0 else 1 - t["p_two_sided"] / 2
        out["C10"] = dict(final_means_tol_0_005_015=m, monotone_decreasing=bool(m[0] >= m[1] >= m[2]), d015_vs_005=t, p_one_sided_less=float(p_less),
                          reproduced=bool(m[0] >= m[1] >= m[2] and p_less < 0.05))
    return out


def main():
    out_root, games = sys.argv[1], sys.argv[2:]
    results, extras, tol = [], {}, {}
    for g in games:
        d = os.path.join(out_root, g)
        if not os.path.isdir(d):
            continue
        res = E.analyse(d, g)
        ex, runs, meta = game_extra(d, g, res)
        results.append(res); extras[g] = ex
        if any(a.startswith("JSRL+") for a in runs):
            tol[g] = tolerance_family(runs, meta)
    hp = E.holm_across_games(results)
    hs = E.holm_across_games(results, "strict")          # AMENDMENT A1: the original pi_e-alone H1 + stage-matched statistics
    # C3-final family: JSRL - PDD FINAL, two-sided, Holm over the games
    c3 = {r["game"]: r["secondary"]["D-C3final"] for r in results if "D-C3final" in r["secondary"]}
    c3h = S.holm({g: v["p_two_sided"] for g, v in c3.items()})
    # secondary Holm across games per statistic would be too conservative; the declared secondary family is per game (res['holm_secondary_game'])
    json.dump(dict(games=results, extras=extras, tolerance=tol, holm_primary=hp, holm_strict=hs, c3_final_holm=c3h), open(os.path.join(out_root, "analysis_a1.json"), "w"), indent=1, default=float)
    L = []
    for r in results:
        L.append(E.fmt(r)); L.append("")
        L.append("extras: " + json.dumps(extras[r["game"]], default=float)); L.append("")
    L.append("Holm across the primary family: " + json.dumps(hp)); L.append("Holm across the strict pi_e-alone family: " + json.dumps(hs)); L.append("C3-final Holm over games: " + json.dumps(c3h))
    for g, t in tol.items():
        L.append(f"tolerance family {g}: " + json.dumps(t, default=float))
    open(os.path.join(out_root, "analysis_a1.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

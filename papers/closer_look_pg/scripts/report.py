"""Full-scale report: reduce runs/full/<arm>/seed_*/{metrics,full_gpu,full_e3*,full_g1e5}.json, compare with the paper's
vector-extracted curves (docs paper_extract/paper_figdata.json) and with the authors'-code reference numbers
(runs/oracle/official_reference.json), run the pre-declared primary family P1-P4 (Holm) and the theory checks T1-T3
Writes runs/full/full_report.json and runs/full/full_tables.txt.
"""
import glob
import json
import math
import os
import sys

import numpy as np
from scipy import stats as st

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G3 = os.path.join(REPO, "runs", "full")
FIG = os.path.join(REPO, "src", "configs", "paper_figdata.json")
from rl_jax_bench.stats import holm, mean_se, paired_compare  # noqa: E402

ARMS = [("hopper", "ppo"), ("hopper", "trpo"), ("walker2d", "ppo"), ("walker2d", "trpo")]
if os.environ.get("ARMS_SUBSET"):        # debugging subset
    ARMS = [tuple(a.split("_")) for a in os.environ["ARMS_SUBSET"].split(",")]
SEEDS = [0, 1, 2, 3, 4]
CKPTS = [0, 150, 300, 450]
D = json.load(open(FIG))
OFF = json.load(open(os.path.join(REPO, "runs", "oracle", "official_reference.json")))
OUT, TXT = {}, []


def p(*a):
    TXT.append(" ".join(str(x) for x in a))


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) < 0.01 and x != 0:
        return f"{x:.2g}"
    return f"{x:.{nd}f}"


def ms(x, nd=3):
    m, se = mean_se(x)
    return f"{fmt(m, nd)} ± {fmt(se, nd) if se is not None else '–'} ({len(x)})"


# ------------------------------------------------------------------ paper values
def _interp(s, x, logx=False):
    xs, ys = np.array(s["x"]), np.array(s["y"])
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if logx:
        return float(np.interp(math.log10(x), np.round(np.log10(xs)), ys))
    return float(np.interp(x, xs, ys))


def paper_grad(fig, env, alg, it, n):
    """Figs 9 (pairwise) / 10 (to-true): axes Wlk 0/150/300/450, Hop 0/150/300/450; mean (min, max) of 3 agents."""
    ax = (0 if env == "walker2d" else 4) + CKPTS.index(it)
    v = [_interp(s, n, True) for s in D[fig][ax]["series"] if s["series"] == alg.upper()]
    return float(np.mean(v)), float(min(v)), float(max(v))


def paper_mre(kind, env, alg, it):
    """Held-out MRE: Fig 3 (Walker; ax0 GAE, ax1 Returns), Fig 12 (Hopper: ax0 GAE, ax1 Returns); median [min, max]."""
    fig, ax = ("fig3", 0) if env == "walker2d" else ("fig12", 0)
    ax += 1 if kind == "returns" else 0
    v = [_interp(s, it) for s in D[fig][ax]["series"] if s["series"] == alg.upper()]
    return float(np.median(v)), float(min(v)), float(max(v))


FIG4_COL = {(1.0, 0.0, 0.0): "zero", (0.0, 0.5, 0.0): "gae", (0.0, 0.0, 1.0): "vstar"}


def paper_base(env, it, kind, n):
    ax = (0 if env == "walker2d" else 3) + [150, 300, 450].index(it)
    s = [s for s in D["fig4"][ax]["series"] if FIG4_COL[tuple(s["color"])] == kind][0]
    return _interp(s, n)


def paper_reward(env, alg):
    ax = 0 if env == "hopper" else 1
    S = [s for s in D["fig8"][ax]["series"] if s["series"] == alg.upper()]
    last = [float(np.mean(np.array(s["y"])[np.array(s["x"]) >= max(s["x"]) - 20])) for s in S]
    return float(np.mean(last)), float(min(last)), float(max(last))


def judge(ours, paper, lo=None, hi=None, near_zero_rule=False):
    """Judgment rule (fixed before any run)."""
    if ours is None or paper is None or not np.isfinite(ours):
        return "UNREPRODUCIBLE"
    if near_zero_rule and abs(paper) < 0.05:
        if lo is not None and lo <= ours <= hi:
            return "MATCH"
        if lo is not None and (lo - 0.02 <= ours <= hi + 0.02):
            return "CLOSE"
        return "DIVERGENT"
    rel = (ours - paper) / abs(paper)
    if abs(rel) <= 0.05 or (lo is not None and lo <= ours <= hi):
        return "MATCH"
    if abs(rel) <= 0.20:
        return "CLOSE"
    return "DIVERGENT"


def dpct(ours, paper):
    return "–" if paper in (None, 0) or ours is None else f"{100 * (ours - paper) / abs(paper):+.0f}%"


# ------------------------------------------------------------------ loading
def load(arm, name):
    out = {}
    for s in SEEDS:
        f = os.path.join(G3, arm, f"seed_{s}", f"{name}.json")
        if os.path.exists(f):
            out[s] = json.load(open(f))
    return out


def merged(arm, names):
    res = {}
    for nm in names:
        for s, d in load(arm, nm).items():
            for k, v in d.items():
                if k.startswith("it"):
                    tgt = res.setdefault(s, {}).setdefault(k, {})
                    for blk, val in v.items():          # deep-merge per-n dicts (the 1e5 level comes from its own run)
                        if blk in ("grad", "grad_raw_adv", "baseline") and blk in tgt:
                            tgt[blk].update(val)      # baseline: per-kind merge (the zero_raw arm has its own run)
                        else:
                            tgt[blk] = val
    return res


ALL_ARMS = [("hopper", "ppo"), ("hopper", "trpo"), ("walker2d", "ppo"), ("walker2d", "trpo")]
ENVS = sorted(set(e for e, _ in ARMS))
METRICS = {f"{e}_{a}": load(f"{e}_{a}", "metrics") for e, a in ALL_ARMS}
for e in ("hopper", "walker2d"):
    METRICS[f"{e}_ppo-nobaseline"] = load(f"{e}_ppo-nobaseline", "metrics")
# fullr_* = the cold-review reruns (landscape with the whole-pair-budget true reward; un-centred zero arm).
# They come LAST so that they replace the pre-review landscape; the pre-review landscape is kept in LAND_OLD.
INSTR = {f"{e}_{a}": merged(f"{e}_{a}", ["full_e1a", "full_e1b", "full_land", "full_e3", "full_e3a",
                                                      "full_e3b", "full_g1e5a", "full_g1e5b", "fullr_e3raw",
                                                      "fullr_land"]) for e, a in ARMS}
LAND_OLD = {f"{e}_{a}": merged(f"{e}_{a}", ["full_land"]) for e, a in ARMS}


OFFDIR = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official")), "ref_full")


def official_mre(arm, key, it):
    """Median over the authors'-code 488-iteration runs (3 seeds, Table-1 settings, gymnasium v4) of a held-out MRE."""
    v = []
    for s in (0, 1, 2):
        f = os.path.join(OFFDIR, f"{arm}_s{s}.json")
        if os.path.exists(f):
            h = [r for r in json.load(open(f))["history"] if r.get("it") == it and "heldout" in r]
            if h:
                v.append(h[0]["heldout"][key])
    return float(np.median(v)) if v else None


def hist(arm, key, it):
    return [m["history"][it][key] for s, m in sorted(METRICS[arm].items())]


def final_reward(arm):
    return [float(np.mean([h["mean_reward"] for h in m["history"][468:488]])) for s, m in sorted(METRICS[arm].items())]


# ------------------------------------------------------------------ E0 / E5
p("## E0 reward (Fig 8; mean completed-episode return of the training rollout, its 468-487)")
p("| arm | Paper (5 agents, range) | Authors' code (3 seeds, gymnasium v4) | Ours (mean ± SE, n) | Δ% | Judgment |")
p("|---|---|---|---|---|---|")
OUT["E0"] = {}
for e, a in ARMS:
    arm = f"{e}_{a}"
    fr = final_reward(arm)
    pm, lo, hi = paper_reward(e, a)
    om = OFF[arm]["reward"]["official_final_last20"]
    j = judge(float(np.mean(fr)), pm, lo, hi)
    OUT["E0"][arm] = {"per_seed": fr, "paper": [pm, lo, hi], "official": om, "judgment": j}
    p(f"| {arm} | {pm:.0f} ({lo:.0f}–{hi:.0f}) | {om[0]:.0f} ± {om[1]:.0f} | {ms(fr, 0)} | {dpct(np.mean(fr), pm)} | {j} (proxy env) |")

p("\n## E5 no-baseline PPO (its 468-487) vs PPO; paper: 'almost an order of magnitude worse'")
OUT["E5"] = {}
for e in ("hopper", "walker2d"):
    b, z = final_reward(f"{e}_ppo"), final_reward(f"{e}_ppo-nobaseline")
    ratio = float(np.mean(b) / np.mean(z))
    OUT["E5"][e] = {"ppo": b, "nobaseline": z, "ratio": ratio, "test": paired_compare(b, z)}
    p(f"- {e}: PPO {ms(b, 0)} vs no-baseline {ms(z, 0)}; ratio {ratio:.2f}x")

p("\nE5 divergence diagnostics (Hopper, seeds 0-2, its 468-487; runs/full_diag):")
ppo3 = final_reward("hopper_ppo")[:3]
OUT["E5_diag"] = {}
for tag, path in (("nobaseline (A:=G_t, centred adv-norm; pre-declared arm)", "full/hopper_ppo-nobaseline"),
                  ("nobaseline-nocenter (A:=G_t, scale-only adv-norm)", "full_diag/hopper_ppo-nobaseline-nocenter"),
                  ("zerogae (GAE with V==0, centred)", "full_diag/hopper_ppo-zerogae")):
    v = [float(np.mean([h["mean_reward"] for h in json.load(open(os.path.join(REPO, "runs", path, f"seed_{s}", "metrics.json")))["history"][468:488]]))
         for s in (0, 1, 2)]
    d = paired_compare(ppo3, v)
    OUT["E5_diag"][tag] = {"per_seed": v, "ratio": float(np.mean(ppo3) / np.mean(v)), "test_ppo_minus_arm": d}
    p(f"- {tag}: {ms(v, 0)}; PPO/arm ratio {np.mean(ppo3) / np.mean(v):.2f}x; PPO - arm {d['diff_mean']:.0f} ± {d['diff_se']:.0f}, "
      f"paired p {d['t_p']:.2g} (PPO seeds 0-2: {ms(ppo3, 0)})")

# ------------------------------------------------------------------ E2
p("\n## E2 held-out value MRE (training log; median over 5 seeds [min, max]; paper median [min, max] of 5 agents)")
p("| arm | metric | it | Paper | Authors' code (median, 3 seeds) | Ours median [min,max] | Ours mean ± SE (n) | Δ% (median) | Judgment |")
p("|---|---|---|---|---|---|---|---|---|")
OUT["E2"] = {}
for e, a in ARMS:
    arm = f"{e}_{a}"
    for key, kind in (("heldout_returns_mre", "returns"), ("heldout_gae_mre_code", "gae"), ("heldout_gae_mre_true", None)):
        for it in (0, 150, 450):
            v = hist(arm, key, it)
            med = float(np.median(v))
            pp = paper_mre(kind, e, a, it) if kind else None
            offv = official_mre(arm, key.replace("heldout_", ""), it)
            j = judge(med, pp[0], pp[1], pp[2]) if pp else "–(no paper value)"
            OUT["E2"][f"{arm}/{key}/{it}"] = {"per_seed": v, "paper": pp, "official_median": offv, "judgment": j}
            p(f"| {arm} | {key.replace('heldout_', '')} | {it} | {'–' if not pp else f'{fmt(pp[0])} [{fmt(pp[1])}, {fmt(pp[2])}]'} | "
              f"{fmt(offv)} | {fmt(med)} [{fmt(min(v))}, {fmt(max(v))}] | {ms(v)} | {dpct(med, pp[0]) if pp else '–'} | {j} |")
# whole-curve summary: median over its 50-480 of the per-iteration seed-median (paper: 0.2-0.5 band for Returns-MRE)
p("\nReturns-MRE band check (C6: paper 'about 0.2-0.5', median held-out over its 50-480):")
for e, a in ARMS:
    arm = f"{e}_{a}"
    its = list(range(50, 481, 10))
    med = [float(np.median(hist(arm, "heldout_returns_mre", i))) for i in its]
    tr = [float(np.median(hist(arm, "train_returns_mre", i))) for i in its]
    OUT["E2"][f"{arm}/band"] = {"heldout_median_over_its": float(np.median(med)), "range": [min(med), max(med)],
                                "train_median_over_its": float(np.median(tr))}
    p(f"- {arm}: held-out {np.median(med):.3f} (range of per-iteration medians {min(med):.3f}-{max(med):.3f}); train {np.median(tr):.3f}")

p("\nE2 metric-construction check (scripts/value_mre_variants.py; fresh 10x2048 held-out pairs per seed; median over 5 seeds)")
p("| arm | it | Returns-MRE as logged | complete episodes only | median |V-G|/|G| | states with |G|<0.1 median|G| | their share of the mean |")
p("|---|---|---|---|---|---|---|")
OUT["E2_variants"] = {}
for e, a in ARMS:
    arm = f"{e}_{a}"
    vv = json.load(open(os.path.join(G3, arm, "value_mre_variants.json")))
    for it in (150, 450):
        rows = [vv[f"it{it}/seed{s}"] for s in SEEDS]
        med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
        OUT["E2_variants"][f"{arm}/{it}"] = {"per_seed": rows, "median": med}
        p(f"| {arm} | {it} | {med['mre_logged_def']:.3f} | {med['mre_complete_only']:.3f} | {med['median_rel']:.3f} | "
          f"{100 * med['frac_small_G']:.1f}% | {100 * med['share_of_mean_from_small_G']:.0f}% |")
p("\nValue floor C7 (Returns-MRE / MSE of V_theta vs V* on the same fresh held-out states; mean ± SE over 5 seeds)")
p("| arm | it | Returns-MRE V_θ | Returns-MRE V* | Returns-MSE V_θ | Returns-MSE V* | V* held-out R² |")
p("|---|---|---|---|---|---|---|")
OUT["C7"] = {}
for e, a in ARMS:
    arm = f"{e}_{a}"
    for it in (150, 300, 450):
        F = [INSTR[arm][s][f"it{it}"]["value_floor"] for s in SEEDS]
        r2 = [INSTR[arm][s][f"it{it}"]["vstar_fit"]["heldout_r2"] for s in SEEDS]
        OUT["C7"][f"{arm}/{it}"] = {"floor": F, "r2": r2}
        g = lambda k: [f[k] for f in F]
        p(f"| {arm} | {it} | {ms(g('returns_mre_vtheta'))} | {ms(g('returns_mre_vstar'))} | {ms(g('returns_mse_vtheta'))} | "
          f"{ms(g('returns_mse_vstar'))} | {ms(r2, 2)} |")

# ------------------------------------------------------------------ E1
p("\n## E1 gradient quality (Figs 9/10; ours: reference 1e6 pairs, m=10, 5 seeds for n<=1e4, 3 seeds at 1e5)")
p("| arm | it | n | metric | Paper mean (min..max, 3 agents) | Authors' code (it 0 only; small-scale scale ref 1e5) | Ours (mean ± SE, n) | Δ% | Judgment |")
p("|---|---|---|---|---|---|---|---|---|")
OUT["E1"] = {}
for e, a in ARMS:
    arm = f"{e}_{a}"
    I = INSTR[arm]
    for it in CKPTS:
        for n in (100, 1000, 2000, 10000, 100000):
            for met, fig in (("to_true", "fig10"), ("pairwise", "fig9")):
                v = [I[s][f"it{it}"]["grad"][str(n)][met] for s in SEEDS if f"it{it}" in I.get(s, {})
                     and str(n) in I[s][f"it{it}"].get("grad", {}) and I[s][f"it{it}"]["grad"][str(n)].get(met) is not None]
                if not v:
                    continue
                pm, lo, hi = paper_grad(fig, e, a, it, n)
                og = OFF[arm].get("grad", {}).get(str(it), {}).get(str(n), {}) if it == 0 else {}
                ov = og.get("official_tt" if met == "to_true" else "official_pw")
                j = judge(float(np.mean(v)), pm, lo, hi, near_zero_rule=True)
                OUT["E1"][f"{arm}/{it}/{n}/{met}"] = {"per_seed": v, "paper": [pm, lo, hi], "official": ov, "judgment": j}
                p(f"| {arm} | {it} | {n:g} | {met} | {fmt(pm)} ({fmt(lo)}..{fmt(hi)}){' interp' if n == 2000 else ''} | "
                  f"{'–' if ov is None else f'{fmt(ov[0])} ± {fmt(ov[1])}'} | {ms(v)} | {dpct(np.mean(v), pm)} | {j} |")


def grad_seed(arm, it, n, met="to_true", blk="grad"):
    I = INSTR[arm]
    return [I[s][f"it{it}"][blk][str(n)][met] for s in SEEDS]


# ablations on E1 (secondary)
p("\n## E1 ablations (to_true at n = 2000 / pairwise at n = 2000; mean ± SE over seeds)")
p("| arm | it | headline (normalised Â, fresh-from-reset) | raw A  | pool-subsampled  |")
p("|---|---|---|---|---|")
OUT["E1_ablation"] = {}
for e, a in ARMS:
    arm = f"{e}_{a}"
    for it in CKPTS:
        h = grad_seed(arm, it, 2000)
        r = grad_seed(arm, it, 2000, blk="grad_raw_adv")
        pl = grad_seed(arm, it, 2000, blk="grad_pool")
        hp, plp = grad_seed(arm, it, 2000, "pairwise"), grad_seed(arm, it, 2000, "pairwise", "grad_pool")
        OUT["E1_ablation"][f"{arm}/{it}"] = {"head": h, "raw": r, "pool": pl, "head_pw": hp, "pool_pw": plp}
        p(f"| {arm} | {it} | {ms(h)} / {ms(hp)} | {ms(r)} | {ms(pl)} / {ms(plp)} |")

# ------------------------------------------------------------------ theory T1/T2
p("\n## Theory T1 slope of logit(pairwise) on log n; prediction 1) and T2 to_true residual)")
OUT["T1"], OUT["T2"] = {}, {}
N_REF = 1e6
for blk in ("grad", "grad_pool"):
    slopes_all, resid_all = [], []
    for e, a in ARMS:
        arm = f"{e}_{a}"
        for it in CKPTS:
            sl, rs = [], []
            for s in SEEDS:
                d = INSTR[arm][s][f"it{it}"][blk]
                ns = [n for n in (1000, 2000, 10000, 100000) if str(n) in d]
                pw = np.array([d[str(n)]["pairwise"] for n in ns])
                ok = (pw > 0.005) & (pw < 0.995)
                if ok.sum() >= 2:
                    x, y = np.log(np.array(ns)[ok]), np.log(pw[ok] / (1 - pw[ok]))
                    sl.append(float(np.polyfit(x, y, 1)[0]))
                for n in ns:
                    pwn, tt = d[str(n)]["pairwise"], d[str(n)].get("to_true")
                    if tt is None or not (0.005 < pwn < 0.995):
                        continue
                    nh = n * (1 - pwn) / pwn                       # n_half from
                    pred = math.sqrt(pwn) * math.sqrt(N_REF / (N_REF + nh))
                    rs.append(tt - pred)
            OUT["T1"][f"{blk}/{arm}/{it}"] = sl
            OUT["T2"][f"{blk}/{arm}/{it}"] = rs
            slopes_all += sl
            resid_all += rs
    OUT["T1"][f"{blk}/all"] = [float(np.mean(slopes_all)), float(np.std(slopes_all, ddof=1) / np.sqrt(len(slopes_all))), len(slopes_all)]
    OUT["T2"][f"{blk}/all"] = [float(np.mean(resid_all)), float(np.std(resid_all, ddof=1) / np.sqrt(len(resid_all))), len(resid_all)]
    p(f"- {blk}: slope {OUT['T1'][blk + '/all'][0]:.3f} ± {OUT['T1'][blk + '/all'][1]:.3f} (n={len(slopes_all)} seed-cells); "
      f"to_true - prediction {OUT['T2'][blk + '/all'][0]:+.3f} ± {OUT['T2'][blk + '/all'][1]:.3f} (n={len(resid_all)})")
p("| arm | it | slope fresh (mean ± SE) | slope pool | resid fresh | resid pool |")
p("|---|---|---|---|---|---|")
for e, a in ARMS:
    for it in CKPTS:
        k = f"{e}_{a}/{it}"
        p(f"| {e}_{a} | {it} | {ms(OUT['T1']['grad/' + k])} | {ms(OUT['T1']['grad_pool/' + k])} | "
          f"{ms(OUT['T2']['grad/' + k])} | {ms(OUT['T2']['grad_pool/' + k])} |")

# C3 extrapolation to 1e6 with the model fitted at the largest measured n
p("\n## C3: to_true at n = 1e6 (NOT measured; extrapolation from n_half fitted at the largest n, reference 1e6)")
OUT["C3"] = {}
for e, a in ARMS:
    arm = f"{e}_{a}"
    row = []
    for it in CKPTS:
        ex = []
        for s in SEEDS:
            d = INSTR[arm][s][f"it{it}"]["grad"]
            nmax = max(int(n) for n in d if 0.005 < d[n]["pairwise"] < 0.995) if any(0.005 < d[n]["pairwise"] < 0.995 for n in d) else None
            if nmax is None:
                continue
            pw = d[str(nmax)]["pairwise"]
            nh = nmax * (1 - pw) / pw
            ex.append(math.sqrt(1e6 / (1e6 + nh)) * math.sqrt(N_REF / (N_REF + nh)))
        pm = paper_grad("fig10", e, a, it, 1e6)[0]
        OUT["C3"][f"{arm}/{it}"] = {"extrap": ex, "paper": pm}
        row.append(f"it{it}: {ms(ex)} (paper {pm:.3f})")
    p(f"- {arm}: " + "; ".join(row))

# ------------------------------------------------------------------ E3
p("\n## E3 baselines (Fig 4; pairwise cosine, mean ± SE over 5 seeds; V* fit on 1e6 pairs)")
p("| env | alg | it | n | baseline | Paper (Fig 4; PPO assumed) | Ours (mean ± SE, n) | Δ% | Judgment |")
p("|---|---|---|---|---|---|---|---|---|")
OUT["E3"] = {}
for e, a in ARMS:
    arm = f"{e}_{a}"
    for it in (150, 300, 450):
        for n in (2000, 10000):
            for kind in ("zero", "zero_raw", "gae", "vstar", "zero_gae"):
                v = [INSTR[arm][s][f"it{it}"]["baseline"][kind][str(n)]["pairwise"] for s in SEEDS
                     if kind in INSTR[arm].get(s, {}).get(f"it{it}", {}).get("baseline", {})]
                if not v:
                    continue
                pk = "zero" if kind == "zero_raw" else kind          # Fig 4 'zero' is compared with both zero arms
                pm = paper_base(e, it, pk, n) if (a == "ppo" and kind != "zero_gae") else None
                # DECLARED rule (fixed before any run): near 0 (paper < 0.05) only the spread criterion;
                # Fig 4 has no spread, so the spread is the paper's point: MATCH only if equal, CLOSE within ±0.02.
                j = judge(float(np.mean(v)), pm, pm, pm, near_zero_rule=True) if pm is not None else "–"
                # AMENDMENT A3 (coded in 44a9de3 before the E3 data, but never declared):
                # near 0, MATCH within ±0.02 and CLOSE within ±0.04 absolute. Reported next to the declared verdict.
                ja = j
                if pm is not None and abs(pm) < 0.05:
                    ja = "MATCH" if abs(np.mean(v) - pm) <= 0.02 else ("CLOSE" if abs(np.mean(v) - pm) <= 0.04 else "DIVERGENT")
                OUT["E3"][f"{arm}/{it}/{n}/{kind}"] = {"per_seed": v, "paper": pm, "judgment": j, "judgment_amended_A3": ja}
                p(f"| {e} | {a} | {it} | {n} | {kind} | {fmt(pm)} | {ms(v)} | {dpct(np.mean(v), pm) if pm else '–'} | {j}"
                  f"{'' if ja == j else f' (A3: {ja})'} |")
from collections import Counter  # noqa: E402
for lab, key in (("declared rule", "judgment"), ("amendment A3", "judgment_amended_A3")):
    for kinds in (("zero", "gae", "vstar"), ("zero_raw",)):
        c = Counter(r[key] for k, r in OUT["E3"].items() if r["paper"] is not None and k.split("/")[-1] in kinds)
        OUT.setdefault("E3_tally", {})[f"{lab}/{'+'.join(kinds)}"] = dict(c)
        p(f"- E3 tally ({lab}; kinds {'/'.join(kinds)}; PPO cells with a Fig-4 value): MATCH {c['MATCH']} / CLOSE {c['CLOSE']} / DIVERGENT {c['DIVERGENT']}")
p("\nVariance-reduction ratios trΣ_b/trΣ_0 at n = 1e4 (T3): from cosines | direct n·tr(Ĉov)/same for Zero")
OUT["T3"] = {}
for e, a in ARMS:
    arm = f"{e}_{a}"
    for it in (150, 300, 450):
        rows = []
        for kind in ("gae", "vstar"):
            c15, dr = [], []
            for s in SEEDS:
                b = INSTR[arm][s][f"it{it}"]["baseline"]
                cb, c0 = b[kind]["10000"]["pairwise"], b["zero"]["10000"]["pairwise"]
                if 0 < cb < 1 and 0 < c0 < 1:
                    c15.append(((1 - cb) / cb) / ((1 - c0) / c0))
                # direct: relative noise tr(Cov)/||mean||^2 of the 10 estimates (scale-free; advantages are normalised)
                rb = b[kind]["10000"]["trace_cov"] / b[kind]["10000"]["mean_norm2"]
                r0 = b["zero"]["10000"]["trace_cov"] / b["zero"]["10000"]["mean_norm2"]
                dr.append(rb / r0)
            OUT["T3"][f"{arm}/{it}/{kind}"] = {"d15": c15, "direct": dr}
            rows.append(f"{kind}:  {ms(c15)} | direct {ms(dr)}")
        p(f"- {arm} it{it}: " + "; ".join(rows))

# ------------------------------------------------------------------ E4
# Review the true reward now uses EVERY episode completed within the pair budget (official-code estimator);
# the pre-review estimator (first 2 episodes per 2000-step actor) is in 'by_pairs_firstk' / LAND_OLD. for PPO,
# Spearman is also given against the per-minibatch-normalised surrogate S_mb (the objective PPO actually ascends).
p("\n## E4 landscapes (seed 0; 7x7 on [0,3]^2; true reward = mean of ALL episodes completed within the pair budget)")
p("| arm | it | regime | ‖Δ‖/‖u‖ | episodes/point @2e3 / 2e4 / 1e5 (median) | pairs used @1e5 (median) | rough/SE @2e3 → @2e4 → @1e5 | "
  "abs. roughness @2e3 → @1e5 (fall; noise-only fall) | Spearman(S,T) @1e5 | Spearman(S_mb,T) @1e5 | T(1,0)−T(0,0) ± SE @1e5 | S(1,0) | max|S(0,0)| |")
p("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
OUT["E4"], OUT["E4_old"] = {}, {}
for e, a in ARMS:
    arm = f"{e}_{a}"
    for it in CKPTS:
        L = INSTR[arm].get(0, {}).get(f"it{it}", {}).get("landscape")
        if not L or "by_pairs_firstk" not in L.get("few", {}):
            continue
        for reg in ("few", "many"):
            r = L[reg]
            bp = r["by_pairs"]
            top = bp[max(bp, key=int)]
            rough = [bp[k]["roughness_rel_se"] for k in sorted(bp, key=int)]
            nep = [float(np.median(bp[k]["n_ep"])) for k in sorted(bp, key=int)]
            used = float(np.median(top["pairs_used"]))
            ks_ = sorted(bp, key=int)
            ra = [bp[k]["roughness"] for k in ks_]
            # pure noise: roughness ∝ SE ∝ 1/sqrt(episodes) -> expected fall sqrt(n_ep(1e5)/n_ep(2e3))
            nfall = math.sqrt(np.mean(bp[ks_[-1]]["n_ep"]) / max(np.mean(bp[ks_[0]]["n_ep"]), 1))
            OUT["E4"][f"{arm}/{it}/{reg}"] = {"delta_over_u": r["delta_norm"] / r["u_norm"], "rough": rough,
                                              "n_ep_median": nep, "pairs_used_median_1e5": used,
                                              "rough_abs": ra, "rough_fall": ra[0] / ra[-1], "noise_fall": nfall,
                                              "spearman": top["spearman_S_T"], "spearman_Smb": top.get("spearman_Smb_T"),
                                              "gain": top["true_gain_step"],
                                              "gain_se": top["true_gain_se"], "surr_gain": top["surr_gain_step"],
                                              "surr_gain_mb": top.get("surr_gain_step_mb"),
                                              "S00": r["S00"], "T_range": [float(np.min(top["T"])), float(np.max(top["T"]))]}
            smb = top.get("spearman_Smb_T")
            p(f"| {arm} | {it} | {reg} | {r['delta_norm'] / r['u_norm']:.3f} | {' / '.join(f'{x:.0f}' for x in nep)} | {used:.0f} | "
              f"{' → '.join(f'{x:.2f}' for x in rough)} | {ra[0]:.3g} → {ra[-1]:.3g} ({ra[0] / ra[-1]:.1f}×; {nfall:.1f}×) | {top['spearman_S_T']:+.2f} | {'–' if smb is None else f'{smb:+.2f}'} | "
              f"{top['true_gain_step']:+.1f} ± {top['true_gain_se']:.1f} | {top['surr_gain_step']:+.3g} | {abs(r['S00']):.1g} |")
            # pre-review numbers (first 2 episodes per actor) for the strike-through comparison
            Lo = LAND_OLD[arm].get(0, {}).get(f"it{it}", {}).get("landscape")
            if Lo:
                bo = Lo[reg]["by_pairs"]
                to = bo[max(bo, key=int)]
                OUT["E4_old"][f"{arm}/{it}/{reg}"] = {"rough": [bo[k]["roughness_rel_se"] for k in sorted(bo, key=int)],
                                                      "rough_abs": [bo[k]["roughness"] for k in sorted(bo, key=int)],
                                                      "n_ep_median": [float(np.median(bo[k]["n_ep"])) for k in sorted(bo, key=int)],
                                                      "spearman": to["spearman_S_T"], "gain": to["true_gain_step"],
                                                      "gain_se": to["true_gain_se"],
                                                      "delta_same": bool(abs(Lo[reg]["delta_norm"] - r["delta_norm"]) < 1e-6 * max(1, r["delta_norm"]))}
p("\nE4 before the review (first 2 episodes per 2000-step actor): rough/SE, Spearman, gain")
for k, o in OUT["E4_old"].items():
    p(f"- {k}: episodes/point {' / '.join(f'{x:.0f}' for x in o['n_ep_median'])}; rough/SE {' → '.join(f'{x:.2f}' for x in o['rough'])}; "
      f"abs. rough fall 2e3→1e5 {o['rough_abs'][0] / o['rough_abs'][-1]:.1f}×; "
      f"Spearman {o['spearman']:+.2f}; gain {o['gain']:+.1f} ± {o['gain_se']:.1f}; same step Δ as the rerun: {o['delta_same']}")

# ------------------------------------------------------------------ primary family
fam, det = {}, {}
for e, a in ARMS:
    arm = f"{e}_{a}"
    d = paired_compare(grad_seed(arm, 0, 2000), grad_seed(arm, 450, 2000))
    fam[f"P1 {arm}"], det[f"P1 {arm}"] = d["t_p"], d
    r = hist(arm, "heldout_returns_mre", 450)
    g = hist(arm, "heldout_gae_mre_true", 450)
    d = paired_compare(r, g)
    fam[f"P2 {arm}"], det[f"P2 {arm}"] = d["t_p"], d
for e in ENVS:
    arm = f"{e}_ppo"
    avg = lambda kind: [float(np.mean([INSTR[arm][s][f"it{it}"]["baseline"][kind]["10000"]["pairwise"] for it in (150, 300, 450)]))
                        for s in SEEDS]
    d = paired_compare(avg("vstar"), avg("gae"))
    fam[f"P3a {e} V*>Vθ"], det[f"P3a {e} V*>Vθ"] = d["t_p"], d
    d = paired_compare(avg("gae"), avg("zero"))
    fam[f"P3b {e} Vθ>Zero"], det[f"P3b {e} Vθ>Zero"] = d["t_p"], d
    d = OUT["E5"][e]["test"]
    fam[f"P4 {e}"], det[f"P4 {e}"] = d["t_p"], d
H = holm(fam)
p("\n## Primary family P1-P4 (two-sided paired t across seeds; Holm over 14; HOLDS = p_holm < 0.05 and right sign)")
p("| test | mean diff ± SE | 95% t-CI | bootstrap 95% CI | wins/n | p | p_holm | verdict |")
p("|---|---|---|---|---|---|---|---|")
OUT["family"] = {}
for k in fam:
    d = det[k]
    sign_ok = d["diff_mean"] > 0
    v = "HOLDS" if (H[k]["p_holm"] < 0.05 and sign_ok) else ("PARTIAL" if sign_ok else "FAILS")
    OUT["family"][k] = {**d, **H[k], "verdict": v}
    p(f"| {k} | {fmt(d['diff_mean'])} ± {fmt(d['diff_se'])} | [{fmt(d['t95'][0])}, {fmt(d['t95'][1])}] | "
      f"[{fmt(d['boot95'][0])}, {fmt(d['boot95'][1])}] | {d['wins']}/{d['n']} | {d['t_p']:.2g} | {H[k]['p_holm']:.2g} | {v} |")

# ------------------------------------------------------------------ post-review supplements (NOT in the family)
p("\n## Post-review supplements (descriptive, outside the pre-declared family)")
# the two prose counts
c1 = [float(np.mean(grad_seed(f"{e}_{a}", it, 2000))) for e, a in ARMS for it in (150, 300, 450)]
OUT["recount"] = {"C1_le_0.20": int(sum(x <= 0.20 for x in c1)), "C1_cells": len(c1)}
dd = []
for e, a in ARMS:
    for it in (150, 300, 450):
        F = OUT["C7"][f"{e}_{a}/{it}"]["floor"]
        dd.append((f"{e}_{a}/{it}", abs(np.mean([f["returns_mre_vtheta"] for f in F]) - np.mean([f["returns_mre_vstar"] for f in F]))))
OUT["recount"]["C7_within_0.016"] = int(sum(x <= 0.016 for _, x in dd))
OUT["recount"]["C7_exceptions"] = {k: float(x) for k, x in dd if x > 0.016}
p(f"- C1 cells (it>0) with to_true(2e3) <= 0.20: {OUT['recount']['C1_le_0.20']} of {len(c1)}; "
  f"C7 cells with |MRE V_θ − V*| <= 0.016: {OUT['recount']['C7_within_0.016']} of {len(dd)}; exceptions "
  + ", ".join(f"{k} ({x:.4f})" for k, x in OUT['recount']['C7_exceptions'].items()))

# T2 residual with cluster SEs (the 221 residuals are nested: n-levels share estimates, cells share agents)
for blk in ("grad",):
    by_agent, by_seed = {}, {}
    for e, a in ARMS:
        for it in CKPTS:
            res = OUT["T2"][f"{blk}/{e}_{a}/{it}"]
        # recompute per (arm, seed) cluster
        for sd in SEEDS:
            rs = []
            for it in CKPTS:
                d = INSTR[f"{e}_{a}"][sd][f"it{it}"][blk]
                for n in (1000, 2000, 10000, 100000):
                    if str(n) not in d:
                        continue
                    pwn, tt = d[str(n)]["pairwise"], d[str(n)].get("to_true")
                    if tt is None or not (0.005 < pwn < 0.995):
                        continue
                    nh = n * (1 - pwn) / pwn
                    rs.append(tt - math.sqrt(pwn) * math.sqrt(N_REF / (N_REF + nh)))
            by_agent[f"{e}_{a}/{sd}"] = float(np.mean(rs))
            by_seed.setdefault(sd, []).extend(rs)
    ca = list(by_agent.values())
    cs = [float(np.mean(v)) for v in by_seed.values()]
    OUT["T2_cluster"] = {"agent_cluster": [float(np.mean(ca)), float(np.std(ca, ddof=1) / np.sqrt(len(ca))), len(ca)],
                         "seed_cluster": [float(np.mean(cs)), float(np.std(cs, ddof=1) / np.sqrt(len(cs))), len(cs)]}
    p(f"- T2 residual, cluster SEs: by (arm, seed) agent {ca and np.mean(ca):+.4f} ± {np.std(ca, ddof=1) / np.sqrt(len(ca)):.4f} "
      f"(n={len(ca)} clusters); by seed {np.mean(cs):+.4f} ± {np.std(cs, ddof=1) / np.sqrt(len(cs)):.4f} (n={len(cs)}); "
      f"naive (pooled) {OUT['T2']['grad/all'][0]:+.4f} ± {OUT['T2']['grad/all'][1]:.4f} (n={OUT['T2']['grad/all'][2]})")

# P1 under the pool-subsample scheme (valid at n <= 2e3)
pf, pdt = {}, {}
for e, a in ARMS:
    arm = f"{e}_{a}"
    for met in ("to_true", "pairwise"):
        d = paired_compare(grad_seed(arm, 0, 2000, met, "grad_pool"), grad_seed(arm, 450, 2000, met, "grad_pool"))
        pf[f"P1pool {met} {arm}"], pdt[f"P1pool {met} {arm}"] = d["t_p"], d
Hp = holm(pf)
OUT["F3_P1_pool"] = {k: {**pdt[k], **Hp[k]} for k in pf}
p("\nP1 (it 0 vs it 450 at n = 2000) under the POOL-subsample scheme, Holm over these 8")
p("| test | it 0 (pool) | it 450 (pool) | diff ± SE | wins | p | p_holm |")
p("|---|---|---|---|---|---|---|")
for k in pf:
    met, arm = k.split()[1], k.split()[2]
    d = pdt[k]
    p(f"| {k} | {ms(grad_seed(arm, 0, 2000, met, 'grad_pool'))} | {ms(grad_seed(arm, 450, 2000, met, 'grad_pool'))} | "
      f"{fmt(d['diff_mean'])} ± {fmt(d['diff_se'])} | {d['wins']}/{d['n']} | {d['t_p']:.2g} | {Hp[k]['p_holm']:.2g} |")

# episode lengths at the instrument checkpoints (pairs -> trajectories)
p("\nF4: training-rollout mean episode length at the checkpoints (mean ± SE over 5 seeds) and trajectories per n = 2000 / 1e4")
p("| arm | it 0 | it 150 | it 300 | it 450 |")
p("|---|---|---|---|---|")
OUT["F4_ep_len"] = {}
for e, a in ARMS:
    arm = f"{e}_{a}"
    cells = []
    for it in CKPTS:
        v = hist(arm, "ep_len_mean", it)
        OUT["F4_ep_len"][f"{arm}/{it}"] = v
        m_ = float(np.mean(v))
        cells.append(f"{ms(v, 0)}; {2000 / m_:.1f} / {10000 / m_:.0f} traj")
    p(f"| {arm} | " + " | ".join(cells) + " |")

# power of the n = 5 paired t-tests
p("\nF6: power of a two-sided paired t-test with n = 5 seeds (df 4) at standardised effect d = mean diff / SD of diffs")
OUT["F6_power"] = {}
for alpha, lab in ((0.05, "unadjusted α = 0.05"), (0.05 / 14, "Holm worst case α = 0.05/14")):
    tc = st.t.ppf(1 - alpha / 2, 4)
    row = []
    for dz in (0.5, 1.0, 1.5, 2.0, 3.0):
        nc = dz * math.sqrt(5)
        pw = float(1 - st.nct.cdf(tc, 4, nc) + st.nct.cdf(-tc, 4, nc))
        OUT["F6_power"][f"{alpha:.4f}/{dz}"] = pw
        row.append(f"d={dz}: {pw:.2f}")
    # minimum detectable d at 80 % power
    lo_, hi_ = 0.1, 20.0
    for _ in range(60):
        mid = (lo_ + hi_) / 2
        nc = mid * math.sqrt(5)
        pw = 1 - st.nct.cdf(tc, 4, nc) + st.nct.cdf(-tc, 4, nc)
        lo_, hi_ = (mid, hi_) if pw < 0.8 else (lo_, mid)
    OUT["F6_power"][f"{alpha:.4f}/mdd80"] = hi_
    p(f"- {lab}: " + "; ".join(row) + f"; 80 % power needs d ≥ {hi_:.2f}")
for k, v in OUT["family"].items():
    d_ = abs(v["diff_mean"]) / (v["diff_se"] * math.sqrt(v["n"])) if v["diff_se"] else float("inf")
    OUT["family"][k]["d_z"] = float(d_)
p("- observed d_z in the family: " + "; ".join(f"{k} {OUT['family'][k]['d_z']:.1f}" for k in OUT["family"]))

# P3b against the UN-centred zero arm; P4 against the no-centring no-baseline arm (5 seeds, both envs)
p("\nF2: the literal 'no baseline' arms (un-centred), paired against the pre-declared ones")
OUT["literal_no_baseline"] = {}
for e in ENVS:
    arm = f"{e}_ppo"
    if all("zero_raw" in INSTR[arm][s][f"it{it}"].get("baseline", {}) for s in SEEDS for it in (150, 300, 450)):
        avg = lambda kind: [float(np.mean([INSTR[arm][s][f"it{it}"]["baseline"][kind]["10000"]["pairwise"]
                                           for it in (150, 300, 450)])) for s in SEEDS]
        d = paired_compare(avg("gae"), avg("zero_raw"))
        d2 = paired_compare(avg("zero"), avg("zero_raw"))
        OUT["literal_no_baseline"][f"P3b_raw {e}"] = d
        OUT["literal_no_baseline"][f"zero_minus_raw {e}"] = d2
        p(f"- P3b' {e}: V_θ − Zero(un-centred) {fmt(d['diff_mean'])} ± {fmt(d['diff_se'])} ({d['wins']}/{d['n']}, p {d['t_p']:.2g}); "
          f"Zero(centred) − Zero(un-centred) {fmt(d2['diff_mean'])} ± {fmt(d2['diff_se'])} (p {d2['t_p']:.2g}); "
          f"Zero(un-centred) mean over its {ms(avg('zero_raw'))}")
    fdir = os.path.join(REPO, "runs", "full_extra", f"{e}_ppo-nobaseline-nocenter")
    if all(os.path.exists(os.path.join(fdir, f"seed_{s}", "metrics.json")) for s in SEEDS):
        z = [float(np.mean([h["mean_reward"] for h in json.load(open(os.path.join(fdir, f"seed_{s}", "metrics.json")))["history"][468:488]]))
             for s in SEEDS]
        b = final_reward(f"{e}_ppo")
        d = paired_compare(b, z)
        ratio = float(np.mean(b) / np.mean(z))
        OUT["literal_no_baseline"][f"P4_nocenter {e}"] = {**d, "ratio": ratio, "per_seed": z}
        p(f"- P4' {e}: PPO {ms(b, 0)} vs no-baseline un-centred {ms(z, 0)}; ratio {ratio:.2f}x; "
          f"diff {fmt(d['diff_mean'])} ± {fmt(d['diff_se'])} ({d['wins']}/{d['n']}, p {d['t_p']:.2g})")

# follow-up: time-aware / wider V* refit at it 450 (scripts/vstar_time.py; held-out states re-drawn, paired within run)
vt_files = sorted(glob.glob(os.path.join(REPO, "runs", "full_extra", "vstar_time_*_it450.json")))
if vt_files:
    p("\nRT-C: V* refit on the same 1e6 pairs with t/1000 as an extra input (vstar_t) or [256,256] (vstar_wide); it 450; "
      "Returns-MSE on fresh held-out states (mean ± SE over seeds); 'complete' = states in complete episodes, 'all' = the C7 mask")
    p("| arm | mask | V_θ | V* (as full-scale) | V*(s,t) | V* wide | V*(s,t) / V_θ | V_θ − V*(s,t) paired | R² V* / V*(s,t) / wide |")
    p("|---|---|---|---|---|---|---|---|---|")
    OUT["RT_C_vstar_time"] = {}
    for f in vt_files:
        d = json.load(open(f))
        arm = os.path.basename(d["run"].rstrip("/"))
        ss = sorted(k for k in d if k.startswith("seed"))
        for mask in ("complete", "all"):
            g = lambda nm: [d[k][f"{mask}/mse/{nm}"] for k in ss]
            r2 = lambda nm: [d[k][f"r2/{nm}"] for k in ss]
            pc = paired_compare(g("vtheta"), g("vstar_t")) if len(ss) > 1 else None
            ratio = [a_ / b_ for a_, b_ in zip(g("vstar_t"), g("vtheta"))]
            OUT["RT_C_vstar_time"][f"{arm}/{mask}"] = {k: g(k) for k in ("vtheta", "vstar", "vstar_t", "vstar_wide")} | {
                "ratio_t_over_theta": ratio, "test": pc, "r2": {k: r2(k) for k in ("vstar", "vstar_t", "vstar_wide")}}
            p(f"| {arm} | {mask} | {ms(g('vtheta'))} | {ms(g('vstar'))} | {ms(g('vstar_t'))} | {ms(g('vstar_wide'))} | {ms(ratio, 3)} | "
              f"{'–' if pc is None else '%s (%d/%d, p %.2g)' % (fmt(pc['diff_mean']), pc['wins'], pc['n'], pc['t_p'])} | "
              f"{np.mean(r2('vstar')):.2f} / {np.mean(r2('vstar_t')):.2f} / {np.mean(r2('vstar_wide')):.2f} |")

# sensitivity: the family with P3b / P4 evaluated on the literal no-baseline arms (labelled; not the pre-declared family)
if all(f"P3b_raw {e}" in OUT["literal_no_baseline"] and f"P4_nocenter {e}" in OUT["literal_no_baseline"] for e in ENVS):
    fam2, det2 = dict(fam), dict(det)
    for e in ENVS:
        fam2[f"P3b {e} Vθ>Zero"], det2[f"P3b {e} Vθ>Zero"] = OUT["literal_no_baseline"][f"P3b_raw {e}"]["t_p"], OUT["literal_no_baseline"][f"P3b_raw {e}"]
        fam2[f"P4 {e}"], det2[f"P4 {e}"] = OUT["literal_no_baseline"][f"P4_nocenter {e}"]["t_p"], OUT["literal_no_baseline"][f"P4_nocenter {e}"]
    H2 = holm(fam2)
    OUT["family_literal_no_baseline"] = {}
    p("\nSensitivity (labelled, post-review): Holm over 14 with P3b on the un-centred Zero arm and P4 on the un-centred no-baseline arm")
    p("| test | mean diff ± SE | wins/n | p | p_holm | verdict |")
    p("|---|---|---|---|---|---|")
    for k in fam2:
        d = det2[k]
        v = "HOLDS" if (H2[k]["p_holm"] < 0.05 and d["diff_mean"] > 0) else ("PARTIAL" if d["diff_mean"] > 0 else "FAILS")
        OUT["family_literal_no_baseline"][k] = {"p_holm": H2[k]["p_holm"], "verdict": v}
        if k.startswith(("P3b", "P4")):
            p(f"| {k} (literal) | {fmt(d['diff_mean'])} ± {fmt(d['diff_se'])} | {d['wins']}/{d['n']} | {d['t_p']:.2g} | {H2[k]['p_holm']:.2g} | {v} |")
    c = Counter(v["verdict"] for v in OUT["family_literal_no_baseline"].values())
    p(f"- whole family under this substitution: {c['HOLDS']} HOLDS / {c['PARTIAL']} PARTIAL / {c['FAILS']} FAILS")

json.dump(OUT, open(os.path.join(G3, "full_report.json"), "w"), indent=1, default=float)
open(os.path.join(G3, "full_tables.txt"), "w").write("\n".join(TXT) + "\n")
print("\n".join(TXT))

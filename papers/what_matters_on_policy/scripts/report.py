"""full-scale report (README 'full-scale'; pre-declared family in README 'full-scale — pre-declaration', amendment).

Primary: 9 paired contrasts arm - REC on halfcheetah PROXY (tag full, seeds 0-4): paired t, paired bootstrap 95 %
CI, Holm across the 9; TOST (-10 % of the REC mean) for A3a/A6/A8; judgment vs the paper's conditional-95th ratio and
its spread under THREE readings, each labelled: the pre-declared  rule (DIVERGENT only through a declared clause;
UNCLASSIFIED / CONFLICT otherwise), the task's generic rule, and a post-hoc interval reading (our t95 of rho vs
the paper's point, the paper's bootstrap CI vs our point). Secondary (exploratory): BASE vs REC (Welch, small-scale
study-era BASE lanes), the small-scale halfcheetah lanes side by side (NOT pooled: different log-prob semantics),
small-scale walker2d/hopper rows QUARANTINED with no judgment, eval-noise variance share. Non-finite lanes score NaN
and are excluded with a report.

Usage: uv run python scripts/report.py  -> runs/full/full_report.json + markdown on stdout
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np
from scipy import stats as st

from rl_jax_bench.stats import holm, paired_compare  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_TXT = os.path.join(REPO, "src", "configs", "paper_raw_p95.txt")
PKL = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official")), "data", "mujoco123.pkl")
ENVMAP = {"halfcheetah": "HalfCheetah-v1", "walker2d": "Walker2d-v1", "hopper": "Hopper-v1"}
# arm -> ((exp, choice, arm value), (exp, choice, REC value)) in paper_raw_p95.txt
PAPER_MAP = {
    "A1_obsnorm_none": (("final_stability", "input_normalization", "None"),
                        ("final_stability", "input_normalization", "Avg (comp=False)")),
    "A2a_lastlayer_1": (("final_arch2", "last_kernel_init_policy_scaling", "1.0"),
                        ("final_arch2", "last_kernel_init_policy_scaling", "0.01")),
    "A2b_initstd_1": (("final_arch2", "initial_action_std", "1.0"), ("final_arch2", "initial_action_std", "0.5")),
    "A3a_shared_bc1": (("final_arch", "sub_policy_and_value_function_network_shared_baseline_cost", "1.0"),
                       ("final_arch", "policy_and_value_function_network", "separate")),
    "A4_lambda_0p95": (("final_advantages", "sub_advantage_estimator_gae_gae_lambda", "0.95"),
                       ("final_advantages", "sub_advantage_estimator_gae_gae_lambda", "0.9")),
    "A5a_lr_1em4": (("final_optimize", "sub_optimizer_adam_learning_rate", "0.0001"),
                    ("final_optimize", "sub_optimizer_adam_learning_rate", "0.0003")),
    "A5b_lr_1em3": (("final_optimize", "sub_optimizer_adam_learning_rate", "0.001"),
                    ("final_optimize", "sub_optimizer_adam_learning_rate", "0.0003")),
    "A6_entropy_1em3": (("final_regularizer", "sub_regularization_penalty_entropy_coefficient", "0.001"),
                        ("final_regularizer", "policy_regularization", "no regularization")),
    "A7_valuenorm_none": (("final_stability", "reward_normalization", "None"),
                          ("final_stability", "reward_normalization", "Popart Avg (comp=False)")),
    "A8_advnorm_true": (("final_stability", "normalize_advantages", "True"),
                        ("final_stability", "normalize_advantages", "False")),
}
CLAIM = {"A1_obsnorm_none": "Q1", "A2a_lastlayer_1": "Q2a", "A2b_initstd_1": "Q2b", "A3a_shared_bc1": "Q3",
         "A4_lambda_0p95": "Q4", "A5a_lr_1em4": "Q5", "A5b_lr_1em3": "Q5", "A6_entropy_1em3": "Q6",
         "A7_valuenorm_none": "Q7", "A8_advnorm_true": "Q8"}
TOST_ARMS = {"A3a_shared_bc1", "A6_entropy_1em3", "A8_advnorm_true"}
PRIMARY = ["A1_obsnorm_none", "A2a_lastlayer_1", "A2b_initstd_1", "A3a_shared_bc1", "A4_lambda_0p95", "A5a_lr_1em4",
           "A6_entropy_1em3", "A7_valuenorm_none", "A8_advnorm_true"]
MARGIN = 0.10


def parse_paper():
    out, key = {}, None
    for ln in open(PAPER_TXT):
        m = re.match(r"## (\S+) :: (\S+)", ln)
        if m:
            key = (m.group(1), m.group(2))
            continue
        m = re.match(r"\s+(\S+-v1)\s+(.*)", ln)
        if m and key:
            for v, q, lo, hi, n in re.findall(r"'([^']+)':(\d+)\[(\d+),(\d+)\]n(\d+)", m.group(2)):
                out.setdefault(key, {}).setdefault(m.group(1), {})[v] = tuple(map(float, (q, lo, hi, n)))
    return out


def paper_ratio(P, arm, env):
    (ea, ca, va), (er, cr, vr) = PAPER_MAP[arm]
    qa, la, ha, na = P[(ea, ca)][ENVMAP[env]][va]
    qr, lr, hr, nr = P[(er, cr)][ENVMAP[env]][vr]
    return qa / qr, (la / hr, ha / lr), (qa, la, ha, int(na)), (qr, lr, hr, int(nr))


def load(tag, env):
    lanes = []
    for f in sorted(glob.glob(os.path.join(REPO, "runs", tag, env, "seed_*", "metrics.json"))):
        lanes += json.load(open(f))["lanes"]
    return lanes


def tost(d, m):
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    p1 = 1 - st.t.cdf((d.mean() + m) / se, n - 1)     # H0: diff <= -m
    p2 = st.t.cdf((d.mean() - m) / se, n - 1)         # H0: diff >= +m
    return float(max(p1, p2))


def clamped(n):
    """Official CI: hi = binom.interval(0.95, n, 0.95)[1] used as a 0-based index, clamped to n - 1."""
    return bool(st.binom.interval(0.95, int(n), 0.95)[1] > int(n) - 1)


def judge(rho_o, rho_p, spread, p_holm):
    """PRE-DECLARED rule (see README). Can return DIVERGENT only through a declared clause
    : gap > 20 %, a Holm-significant opposite sign, or paper ~ 0 with ours Holm-significant beyond +-10 %.
    A cell that meets no clause is UNCLASSIFIED. A cell that meets both the MATCH clause (inside the spread) and
    the gap > 20 % DIVERGENT clause is reported as CONFLICT (the declaration gives no precedence)."""
    rel = abs(rho_o - rho_p) / rho_p
    sp = 0 if abs(rho_p - 1) < 0.05 else int(np.sign(rho_p - 1))
    so = int(np.sign(rho_o - 1))
    agree = abs(rho_o - 1) <= MARGIN if sp == 0 else so == sp
    sig = p_holm is not None and p_holm < 0.05
    if sig and sp != 0 and so == -sp:
        return "DIVERGENT", "Holm-significant opposite sign"
    if sig and sp == 0 and abs(rho_o - 1) > MARGIN:
        return "DIVERGENT", "paper ~ 0, ours Holm-significant beyond +-10 %"
    inside = spread[0] <= rho_o <= spread[1]
    if (rel <= 0.05 or inside) and agree:
        if rel > 0.20:
            return "CONFLICT", "MATCH clause (inside spread, signs agree) and DIVERGENT clause (gap > 20 %) both hold"
        return "MATCH", ("inside paper spread" if inside else "within 5 %")
    if rel <= 0.20 and agree:
        return "CLOSE", "within 20 %, same sign"
    if rel > 0.20:
        return "DIVERGENT", "gap > 20 %"
    return "UNCLASSIFIED", "no clause applies: gap <= 20 %, point-estimate signs differ, not Holm-significant"


def judge_generic(rho_o, rho_p, spread):
    """The task's generic rule: MATCH = within 5 % or inside the paper's reported spread; CLOSE <= 20 %;
    DIVERGENT > 20 % (no sign condition)."""
    rel = abs(rho_o - rho_p) / rho_p
    inside = spread[0] <= rho_o <= spread[1]
    if rel <= 0.05 or inside:
        return "MATCH", ("inside paper spread" if inside else "within 5 %") + ("; gap > 20 %" if rel > 0.20 else "")
    return ("CLOSE", "within 20 %") if rel <= 0.20 else ("DIVERGENT", "gap > 20 %, outside spread")


def judge_ci(rho_t95, rho_p, paper_boot):
    """Interval reading (added after the run, NOT pre-declared): our t95 interval of rho against the
    paper's point ratio, and the paper's bootstrap interval against our point."""
    lo, hi = rho_t95
    ours_excl_paper = not (lo <= rho_p <= hi)
    ours_excl_1 = hi < 1 or lo > 1
    sp = 0 if abs(rho_p - 1) < 0.05 else int(np.sign(rho_p - 1))
    if ours_excl_1 and not ours_excl_paper and sp != 0 and np.sign((lo + hi) / 2 - 1) == sp:
        return "REPRODUCED-DIRECTION", "our t95 excludes 1 on the paper's side and contains its point"
    if ours_excl_paper:
        return "INCONSISTENT", "our t95 excludes the paper's point ratio"
    return "INCONCLUSIVE", "our t95 contains both 1 and the paper's point ratio"


def paper_boot(B=3000, seed=1):
    """Bootstrap 95 % CI of the paper's q95 ratio from the authors' raw rows (HalfCheetah-v1), as in
    scripts/review/paper_ratio_boot.py. Returns {arm: (point, lo, hi, n_arm, n_rec)}."""
    import pickle
    P = "_gin.study_design.choice_value_"
    Dd = pickle.load(open(PKL, "rb"))
    name = {m["title"].split(": ")[1].split(" ")[0]: k for k, (df, m) in Dd.items()}
    rng = np.random.default_rng(seed)

    def get(ex, col, val):
        df, _ = Dd[name[ex]]
        s = df[df.env_name == "HalfCheetah-v1"]
        return s[s[P + col].astype(str) == f"'{val}'"]["_eval_"].map(lambda d: d[max(d)]).values.astype(float)

    out = {}
    for arm, ((ea, ca, va), (er, cr, vr)) in PAPER_MAP.items():
        a, r = get(ea, ca, va), get(er, cr, vr)
        bs = [np.percentile(a[rng.integers(0, len(a), len(a))], 95) / np.percentile(r[rng.integers(0, len(r), len(r))], 95)
              for _ in range(B)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[arm] = (float(np.percentile(a, 95) / np.percentile(r, 95)), float(lo), float(hi), len(a), len(r))
    return out


def score_of(l):
    """A lane's score; NaN for a lane with a skipped non-finite update (src/train.py, README dtype policy)."""
    return float("nan") if l["nonfinite"] else l["score"]


def main():
    P = parse_paper()
    PB = paper_boot()
    rep = {"primary": [], "secondary": {}}
    L = load("full", "halfcheetah")
    sc = {}
    for l in L:
        sc.setdefault(l["arm"], {})[l["seed"]] = score_of(l)
    nonfinite = [(l["arm"], l["seed"]) for l in L if l["nonfinite"]]
    rec = sc["REC"]
    seeds = sorted(rec)
    rec_v = np.array([rec[s] for s in seeds])
    rec_m = rec_v.mean()
    rows, pv = [], {}
    for a in PRIMARY:
        if a not in sc:
            continue
        x = np.array([sc[a][s] for s in seeds])
        ok = np.isfinite(x) & np.isfinite(rec_v)          # exclusions reported, never imputed
        pc = paired_compare(x[ok], rec_v[ok])
        pv[a] = pc["t_p"]
        rows.append({"arm": a, "n": int(ok.sum()), "excluded_seeds": [s for s, o in zip(seeds, ok) if not o],
                     "mean": float(x[ok].mean()), "se": float(x[ok].std(ddof=1) / math.sqrt(ok.sum())),
                     "rho": float(x[ok].mean() / rec_v[ok].mean()), **pc})
    H = holm(pv)
    for r in rows:
        a = r["arm"]
        r["p_holm"] = H[a]["p_holm"]
        rp, spread, qa, qr = paper_ratio(P, a, "halfcheetah")
        bp, blo, bhi, _, _ = PB[a]
        assert abs(bp - rp) < 2e-3 * rp, (a, bp, rp)      # raw rows reproduce paper_raw_p95.txt
        rho_t95 = [1.0 + r["rel_t95"][0], 1.0 + r["rel_t95"][1]]   # rho = 1 + mean(d)/mean(REC)
        r.update({"rho_paper": rp, "paper_spread": spread, "paper_arm_q95": qa, "paper_rec_q95": qr,
                  "paper_boot95": [blo, bhi], "paper_ci_clamped": {"arm": clamped(qa[3]), "rec": clamped(qr[3])},
                  "rho_t95": rho_t95, "delta_pct": 100 * (r["rho"] - rp) / rp,
                  "paper_boot_excludes_ours": not (blo <= r["rho"] <= bhi)})
        if a in TOST_ARMS:
            r["tost_p"] = tost(r["diff_per_seed"], MARGIN * rec_m)
        r["judgment"], r["why"] = judge(r["rho"], rp, spread, r["p_holm"])
        r["judgment_generic"], r["why_generic"] = judge_generic(r["rho"], rp, spread)
        if r["judgment_generic"] == "MATCH" and "spread" in r["why_generic"] and any(r["paper_ci_clamped"].values()):
            r["why_generic"] += "; paper CI clamped at the max (not counted as reproduction)"
        r["judgment_ci"], r["why_ci"] = judge_ci(rho_t95, rp, (blo, bhi))
        r["claim"] = CLAIM[a]
    rep["primary"] = rows
    rep["rec"] = {"n": len(rec_v), "mean": float(rec_m), "se": float(rec_v.std(ddof=1) / math.sqrt(len(rec_v))),
                  "per_seed": rec_v.tolist(), "median": float(np.median(rec_v))}
    rep["nonfinite_lanes"] = nonfinite
    tally = lambda key: {j: sorted(r["arm"] for r in rows if r[key] == j) for j in sorted({r[key] for r in rows})}
    rep["tally"] = {"predeclared": tally("judgment"), "generic": tally("judgment_generic"), "ci": tally("judgment_ci")}
    # eval-noise vs seed-noise (: variance share, not SE ratio)
    ep_se = [math.sqrt(np.mean(np.square(l["curve_se_over_episodes"][1:])) / 10) for l in L]
    sd = float(rec_v.std(ddof=1))
    rep["eval_noise"] = {"median_score_se_from_episodes": float(np.median(ep_se)),
                         "max_score_se_from_episodes": float(np.max(ep_se)), "rec_seed_sd": sd,
                         "variance_share_median": float((np.median(ep_se) / sd) ** 2)}
    # secondary 1: BASE vs REC (Welch; small-scale study-era BASE lanes = X2_base_rawlogp, seeds 1000-1002)
    G2 = load("small", "halfcheetah")
    base = np.array([score_of(l) for l in G2 if l["arm"] == "X2_base_rawlogp"])
    assert np.isfinite(base).all()
    w = st.ttest_ind(base, rec_v, equal_var=False)
    rep["secondary"]["base_vs_rec"] = {"base_mean": float(base.mean()), "base_se": float(base.std(ddof=1) / math.sqrt(len(base))),
                                       "n_base": len(base), "rho": float(base.mean() / rec_m), "welch_t": float(w.statistic),
                                       "welch_p": float(w.pvalue)}
    # secondary 2: the small-scale halfcheetah lanes (seeds 1000-1002, released-HEAD semantics), reported SIDE BY SIDE,
    # never pooled with the study-era full-scale lanes 
    g2 = {}
    for l in G2:
        g2.setdefault(l["arm"], {})[l["seed"]] = score_of(l)
    side = []
    for r in rows:
        a = r["arm"]
        k2 = sorted(g2["REC"])
        d = np.array([g2[a][s] - g2["REC"][s] for s in k2])
        side.append({"arm": a, "n": len(d), "semantics": "released-HEAD (X4-like)", "diff_mean": float(np.nanmean(d)),
                     "diff_se": float(np.nanstd(d, ddof=1) / math.sqrt(np.isfinite(d).sum())),
                     "rho": float(np.nanmean([g2[a][s] for s in k2]) / np.mean([g2["REC"][s] for s in k2]))})
    rep["secondary"]["small_halfcheetah_side_by_side"] = side
    # secondary 3: small-scale walker2d/hopper -- QUARANTINED: n = 3, released-HEAD semantics, which the study never
    # used (README). Descriptive ratios only; NO judgment is emitted for them.
    ind = []
    for env in ["walker2d", "hopper"]:
        G = load("small", env)
        s = {}
        for l in G:
            s.setdefault(l["arm"], {})[l["seed"]] = score_of(l)
        rm = np.mean(list(s["REC"].values()))
        for a in PRIMARY + ["A5b_lr_1em3"]:
            x = np.array([s[a][k] for k in sorted(s["REC"])])
            d = x - np.array([s["REC"][k] for k in sorted(s["REC"])])
            rp, spread, _, _ = paper_ratio(P, a, env)
            ind.append({"env": env, "arm": a, "rho": float(np.nanmean(x) / rm), "diff": float(np.nanmean(d)),
                        "n_finite": int(np.isfinite(x).sum()), "rho_paper": rp, "spread": spread,
                        "status": "QUARANTINED (released-HEAD semantics, n = 3; no judgment)"})
    rep["secondary"]["small_quarantined"] = ind
    with open(os.path.join(REPO, "runs", "full", "full_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    # markdown
    print(f"REC: {rec_m:.0f} ± {rep['rec']['se']:.0f} (n={len(rec_v)}; per seed {np.round(rec_v).astype(int).tolist()})")
    print("| arm (claim) | paper ρ [spread] {boot95} clamp | ours ρ (mean ± SE, n) t95 | diff ± SE; t p; Holm p | Δ% | pre-declared | generic | interval reading |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        tp = f"; TOST p {r['tost_p']:.3f}" if "tost_p" in r else ""
        cl = "/".join(k for k, v in r["paper_ci_clamped"].items() if v) or "-"
        print(f"| {r['arm']} ({r['claim']}) | {r['rho_paper']:.3f} [{r['paper_spread'][0]:.2f}, {r['paper_spread'][1]:.2f}] "
              f"{{{r['paper_boot95'][0]:.2f}, {r['paper_boot95'][1]:.2f}}} {cl} | {r['rho']:.3f} ({r['mean']:.0f} ± {r['se']:.0f}, {r['n']}) "
              f"[{r['rho_t95'][0]:.3f}, {r['rho_t95'][1]:.3f}] | {r['diff_mean']:+.0f} ± {r['diff_se']:.0f}; {r['t_p']:.4f}; "
              f"{r['p_holm']:.4f}{tp} | {r['delta_pct']:+.1f} | {r['judgment']} ({r['why']}) | {r['judgment_generic']} "
              f"({r['why_generic']}) | {r['judgment_ci']}{'; paper boot excludes ours' if r['paper_boot_excludes_ours'] else ''} |")
    print("tally:", json.dumps(rep["tally"]))
    print(json.dumps(rep["secondary"]["base_vs_rec"], default=float))
    for r in side:
        print(f"[small-hc side-by-side, HEAD semantics, n=3] {r['arm']} rho={r['rho']:.3f} diff={r['diff_mean']:+.0f}±{r['diff_se']:.0f}")
    for r in ind:
        print(f"[quarantined] {r['env']} {r['arm']} rho={r['rho']:.3f} paper={r['rho_paper']:.3f} n_finite={r['n_finite']}")
    print("eval noise:", rep["eval_noise"], "nonfinite:", nonfinite)


if __name__ == "__main__":
    main()

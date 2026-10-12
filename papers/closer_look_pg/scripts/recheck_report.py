"""Recheck (2026-10-12) reducer: every before/after and labelled statistic, pre-declared before the reruns. Reads the full-scale report (runs/full/full_report.json), the per-seed full-scale files and the
Recheck outputs under runs/recheck/; writes runs/recheck/recheck_report.json and recheck_tables.txt.
Sections whose inputs are missing print 'pending' instead of failing.
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
SA = os.path.join(REPO, "runs", "recheck")
OFFDIR = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official")), "ref_full")
FIG = os.path.join(REPO, "src", "configs", "paper_figdata.json")
from rl_jax_bench.stats import holm, mean_se, paired_compare  # noqa: E402

R = json.load(open(os.path.join(G3, "full_report.json")))
D = json.load(open(FIG))
SEEDS = [0, 1, 2, 3, 4]
ARMS = ["hopper_ppo", "hopper_trpo", "walker2d_ppo", "walker2d_trpo"]
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
    x = [v for v in x if v is not None]
    if not x:
        return "–"
    m, se = mean_se(x)
    return f"{fmt(m, nd)} ± {fmt(se, nd) if se is not None else '–'} ({len(x)})"


def judge(ours, paper, lo=None, hi=None, near_zero_rule=False):
    """The declared judgment rule (copied from scripts/report.py)."""
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


def dpct(o, pp):
    return "–" if pp in (None, 0) or o is None else f"{100 * (o - pp) / abs(pp):+.0f}%"


def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = st.ttest_ind(a, b, equal_var=False)
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"diff": float(a.mean() - b.mean()), "se": se, "z": float((a.mean() - b.mean()) / se), "p": float(t.pvalue),
            "n": [len(a), len(b)]}


# ====================================================================== E0 ( smoothing, gym, fix)
def metrics(arm, s):
    return json.load(open(os.path.join(G3, arm, f"seed_{s}", "metrics.json")))["history"]


p("## E0 final reward: raw (declared) / trailing-100 mean  / it-450 agents on MJX and gymnasium v4  / "
  "post-step-kinematics env ")
p("| arm | paper mean (agent range) | authors' code (gym v4) | ours raw its 468–487 | ours trailing-100 at 487 | ours "
  "EMA 0.02 | it-450 agents, MJX legacy (64 ep) | same agents, MJX post-step env | same agents, gym v4 (20 ep) | "
  "judgments raw / smoothed / gym |")
p("|---|---|---|---|---|---|---|---|---|---|")
GYM = jload(os.path.join(SA, "environment", "e0_gym_eval.json")) or {}
FIX = jload(os.path.join(SA, "env_fix", "env_fix_eval.json")) or {}
OUT["E0"] = {}
for arm in ARMS:
    E0 = R["E0"][arm]
    pm, plo, phi = E0["paper"]
    raw = E0["per_seed"]
    tr100, ema = [], []
    for s in SEEDS:
        r = np.array([h["mean_reward"] for h in metrics(arm, s)])
        tr100.append(float(r[388:488].mean()))
        e = r[0]
        for x in r[1:]:
            e = 0.98 * e + 0.02 * x
        ema.append(float(e))
    gym = [GYM[f"{arm}_s{s}_it450"]["ret_mean"] for s in SEEDS if f"{arm}_s{s}_it450" in GYM]
    fx = FIX.get(f"{arm}/450")
    mjx = fx["legacy_return"] if fx else []
    post = fx["post_step_return"] if fx else []
    js = [judge(float(np.mean(v)), pm, plo, phi) if v else "pending" for v in (raw, tr100, gym)]
    off = E0.get("official")
    OUT["E0"][arm] = {"paper": E0["paper"], "raw": raw, "trail100": tr100, "ema02": ema, "gym_it450": gym,
                      "mjx_it450": mjx, "mjx_post_step_it450": post,
                      "mjx_minus_gym": paired_compare(mjx, gym) if len(gym) == 5 and len(mjx) == 5 else None,
                      "post_minus_legacy": fx["diff_post_minus_legacy"] if fx else None,
                      "judgments": dict(zip(("raw", "trail100", "gym"), js)),
                      "dpct": {k: dpct(float(np.mean(v)), pm) if v else None
                               for k, v in (("raw", raw), ("trail100", tr100), ("ema02", ema), ("gym", gym))}}
    p(f"| {arm} | {fmt(pm)} ({fmt(plo)}–{fmt(phi)}) | {fmt(off[0])} ± {fmt(off[1])} (3) | {ms(raw)} {dpct(np.mean(raw), pm)} | "
      f"{ms(tr100)} {dpct(np.mean(tr100), pm)} | {ms(ema)} {dpct(np.mean(ema), pm)} | {ms(mjx)} | {ms(post)} | "
      f"{ms(gym)} {dpct(np.mean(gym), pm) if gym else ''} | {' / '.join(js)} |")
for arm in ARMS:
    o = OUT["E0"][arm]
    if o["mjx_minus_gym"]:
        d = o["mjx_minus_gym"]
        p(f"- {arm}: MJX − gym on the same it-450 agents {fmt(d['diff_mean'])} ± {fmt(d['diff_se'])} "
          f"(MJX higher {d['wins']}/5, p {d['t_p']:.2g})")
    if o["post_minus_legacy"]:
        d = o["post_minus_legacy"]
        p(f"- {arm}: post-step − legacy env, same agents and keys: {fmt(d['diff_mean'])} ± {fmt(d['diff_se'])} "
          f"({d['wins']}/5 higher, p {d['t_p']:.2g})")


# ====================================================================== E2 window
def paper_mre_series(env, alg, kind):
    fig, ax = ("fig3", 0) if env == "walker2d" else ("fig12", 0)
    ax += 1 if kind == "returns" else 0
    return [s for s in D[fig][ax]["series"] if s["series"] == alg.upper()]


WIN = list(range(400, 490, 10))
p("\n## E2 held-out MRE on the window its 400–480 (median over the window per agent, then median over agents)")
p("| cell | paper it 450 | paper window [agent range] | authors' code window (3) | ours it 450 (declared) | ours window "
  "[seed range] | declared judgment | window judgment | counted? |")
p("|---|---|---|---|---|---|---|---|---|")
OUT["E2_window"] = {}
tally = {"declared": {}, "window": {}, "declared_counted": {}, "window_counted": {}}
for arm in ARMS:
    env, alg = arm.split("_")
    for key, kind, lab in (("heldout_returns_mre", "returns", "Returns-MRE"), ("heldout_gae_mre_code", "gae", "GAE-MRE code"),
                           ("heldout_gae_mre_true", "gae", "GAE-MRE |.|/|.| (H2)")):
        ser = paper_mre_series(env, alg, kind)
        pw = [float(np.median([np.interp(i, s["x"], s["y"]) for i in WIN])) for s in ser]
        p450 = float(np.median([np.interp(450, s["x"], s["y"]) for s in ser]))
        ow = []
        for s in SEEDS:
            h = {r["it"]: r[key] for r in metrics(arm, s) if r.get(key) is not None}
            ow.append(float(np.median([h[i] for i in WIN])))
        offk = key.replace("heldout_", "")
        aw = []
        for s in (0, 1, 2):
            f = os.path.join(OFFDIR, f"{arm}_s{s}.json")
            if os.path.exists(f):
                h = {r["it"]: r["heldout"][offk] for r in json.load(open(f))["history"] if "heldout" in r}
                aw.append(float(np.median([h[i] for i in WIN if i in h])))
        c = R["E2"][f"{arm}/{key}/450"]
        jd = c["judgment"]
        if "true" in key:     # the headline's "H2 reading": |V-y|/|y| judged against the paper's GAE-MRE curve
            p450s = [float(np.interp(450, s["x"], s["y"])) for s in ser]
            jd = judge(float(np.median(c["per_seed"])), p450, min(p450s), max(p450s))
        jw = judge(float(np.median(ow)), float(np.median(pw)), min(pw), max(pw))
        counted = "no (H2 reading)" if "true" in key else "yes"
        OUT["E2_window"][f"{arm}/{key}"] = {"paper_450": p450, "paper_window": pw, "ours_window": ow, "authors_window": aw,
                                            "judgment_declared": jd, "judgment_window": jw}
        for tk, j in (("declared", jd), ("window", jw)):
            tally[tk][j] = tally[tk].get(j, 0) + 1
            if "true" not in key:
                tally[f"{tk}_counted"][j] = tally[f"{tk}_counted"].get(j, 0) + 1
        p(f"| {arm} {lab} | {fmt(p450)} | {fmt(float(np.median(pw)))} [{fmt(min(pw))}, {fmt(max(pw))}] | "
          f"{fmt(float(np.median(aw))) if aw else '–'} | {fmt(float(np.median(c['per_seed'])))} | {fmt(float(np.median(ow)))} "
          f"[{fmt(min(ow))}, {fmt(max(ow))}] | {jd} | {jw} | {counted} |")
OUT["E2_tally_it450_cells"] = tally
p(f"- tallies over these 12 it-450 cells: declared {tally['declared']}, window {tally['window']}; "
  f"without the H2 rows : declared {tally['declared_counted']}, window {tally['window_counted']}")

# ====================================================================== E1
REL = {}
for arm in ARMS:      # its 150/300/450 (seeds 0-2) and the it-0 control (seed 0) are separate files (CPU placement)
    parts = [jload(os.path.join(SA, "ref_reliability", f"{arm}{t}.json")) for t in ("", "_it0", "_it150", "_it300", "_it450")]
    parts = [x for x in parts if x]
    REL[arm] = {"cells": {k: v for x in parts for k, v in x["cells"].items()}} if parts else None
p("\n## E1 split-half reference reliability : r_ref = √(2ρ/(1+ρ)), ρ = cos of two disjoint 5e5-pair halves")
p("| arm | it 0 | it 150 | it 300 | it 450 |")
p("|---|---|---|---|---|")
OUT["ref_reliability"] = {}
for arm in ARMS:
    row = []
    for it in (0, 150, 300, 450):
        if REL[arm] is None:
            row.append("pending")
            continue
        c = [REL[arm]["cells"].get(f"it{it}/seed{s}") for s in SEEDS]
        c = [x for x in c if x]
        if not c:
            row.append("–")
            continue
        rr = [x["r_ref"] for x in c]
        rho = [x["rho_split_half"] for x in c]
        sv = [x["cos_full_vs_saved"] for x in c if x["cos_full_vs_saved"] is not None]
        OUT["ref_reliability"][f"{arm}/{it}"] = {"r_ref": rr, "rho": rho, "cos_full_vs_saved": sv,
                                                 "measurable_mean": bool(np.mean(rr) >= 0.5)}
        row.append(f"r_ref {fmt(float(np.mean(rr)))} [{fmt(min(rr))}, {fmt(max(rr))}]; ρ {fmt(float(np.mean(rho)))}; "
                   f"cos(re-collected ref, saved full-scale ref) {fmt(float(np.mean(sv))) if sv else '–'}")
    p(f"| {arm} | " + " | ".join(row) + " |")

p("\n## E1 cells re-judged ( like-for-like 2e3; noise-aware; reference reliability)")
p("| cell | paper | ours measured (declared) | ours log-interp 1e3→1e4 | declared | interp judgment | noise-aware | ref r_ref |")
p("|---|---|---|---|---|---|---|---|")
OUT["E1"] = {}
cnt = {"declared": {}, "interp2e3": {}, "noise_aware": {}, "not_measurable": 0}
for k, c in R["E1"].items():
    arm, it, n, met = k.split("/")
    ours = c["per_seed"]
    paper = c["paper"]
    if paper is None or ours is None:
        continue
    pm, plo, phi = paper
    m = float(np.mean(ours))
    sd = float(np.std(ours, ddof=1))
    se = sd / math.sqrt(len(ours))
    na = abs(m - pm) <= 2 * math.sqrt(se ** 2 + sd ** 2 / 3)
    rec = {"declared": c["judgment"], "noise_aware": "within noise" if na else "outside noise"}
    interp = None
    if n == "2000":
        a = R["E1"].get(f"{arm}/{it}/1000/{met}")
        b = R["E1"].get(f"{arm}/{it}/10000/{met}")
        if a and b:
            interp = [x + (y - x) * math.log10(2) for x, y in zip(a["per_seed"], b["per_seed"])]
            rec["interp2e3"] = judge(float(np.mean(interp)), pm, plo, phi, near_zero_rule=True)
            cnt["interp2e3"][rec["interp2e3"]] = cnt["interp2e3"].get(rec["interp2e3"], 0) + 1
    rr = OUT["ref_reliability"].get(f"{arm}/{it}")
    if met == "to_true" and rr is not None and not rr["measurable_mean"] and int(n) <= 100000:
        rec["ref"] = "not measurable at 1e6 (r_ref < 0.5)"
        cnt["not_measurable"] += 1
    cnt["declared"][c["judgment"]] = cnt["declared"].get(c["judgment"], 0) + 1
    cnt["noise_aware"][rec["noise_aware"]] = cnt["noise_aware"].get(rec["noise_aware"], 0) + 1
    OUT["E1"][k] = {**rec, "ours_mean": m, "ours_se": se, "ours_sd": sd, "paper": paper,
                    "ours_interp_mean": None if interp is None else float(np.mean(interp))}
    if n == "2000" or (met == "to_true" and it in ("0", "450")):
        p(f"| {k} | {fmt(pm)} ({fmt(plo)}..{fmt(phi)}) | {fmt(m)} ± {fmt(se)} | "
          f"{'–' if interp is None else fmt(float(np.mean(interp)))} {'' if interp is None else dpct(float(np.mean(interp)), pm)} | "
          f"{c['judgment']} | {rec.get('interp2e3', '–')} | {rec['noise_aware']} | "
          f"{'–' if rr is None or met != 'to_true' else fmt(float(np.mean(rr['r_ref'])))} |")
OUT["E1_tallies"] = cnt
p(f"- tallies over all {sum(cnt['declared'].values())} E1 cells: declared {cnt['declared']}; noise-aware "
  f"{cnt['noise_aware']}; 2e3 cells with ours interpolated {cnt['interp2e3']}; to_true cells not measurable at 1e6: "
  f"{cnt['not_measurable']}")
# cross-tab declared DIVERGENT -> noise-aware
xt = {}
for k, v in OUT["E1"].items():
    xt.setdefault(v["declared"], {}).setdefault(v["noise_aware"], 0)
    xt[v["declared"]][v["noise_aware"]] += 1
OUT["E1_crosstab"] = xt
p(f"- declared × noise-aware: {xt}")


# ====================================================================== E3
def fig4(env, it, kind, n=10000):
    ax = (0 if env == "walker2d" else 3) + [150, 300, 450].index(it)
    col = {(1.0, 0.0, 0.0): "zero", (0.0, 0.5, 0.0): "gae", (0.0, 0.0, 1.0): "vstar"}
    s = [s for s in D["fig4"][ax]["series"] if col[tuple(s["color"])] == kind][0]
    xs, ys = np.array(s["x"]), np.array(s["y"])
    o = np.argsort(xs)
    return float(np.interp(n, xs[o], ys[o]))


KINDS = ["gae", "vstar", "vstar_t", "mc_vtheta", "mc_vstar", "mc_vstar_t"]
p("\n## E3 (Fig 4) with the recheck kinds: pairwise(1e4), mean ± SE over seeds (full-scale keys)")
p("| arm | it | paper V_θ | paper V* | V_θ (GAE) | V* (GAE) | V*(s,t) (GAE) | G − V_θ | G − V* | G − V*(s,t) | "
  "R² V* / V*(s,t) | repro of full-scale gae / vstar (max abs) | paper V_θ / V* inside our seed range? |")
p("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
OUT["E3"] = {}
E3OK = True
for arm in ("hopper_ppo", "walker2d_ppo", "hopper_trpo", "walker2d_trpo"):
    env = arm.split("_")[0]
    for it in (150, 300, 450):
        per = {k: [] for k in KINDS}
        r2, r2t, rep = [], [], {"gae": [], "vstar": []}
        for s in SEEDS:
            f = os.path.join(G3, arm, f"seed_{s}", "sa_e3.json")
            d = jload(f)
            if d is None or f"it{it}" not in d:
                continue
            b = d[f"it{it}"]["baseline"]
            for k in KINDS:
                per[k].append(b[k]["10000"]["pairwise"])
            r2.append(d[f"it{it}"]["vstar_fit"]["heldout_r2"])
            r2t.append(d[f"it{it}"]["vstar_t_fit"]["heldout_r2"])
            g3 = R["E3"].get(f"{arm}/{it}/10000/gae")
            if g3:
                rep["gae"].append(abs(b["gae"]["10000"]["pairwise"] - g3["per_seed"][s]))
                rep["vstar"].append(abs(b["vstar"]["10000"]["pairwise"] - R["E3"][f"{arm}/{it}/10000/vstar"]["per_seed"][s]))
        if not per["gae"]:
            if arm.endswith("ppo"):
                E3OK = False
                p(f"| {arm} | {it} | pending |")
            continue
        if arm.endswith("ppo") and len(per["gae"]) < 5:
            E3OK = False
        pv = fig4(env, it, "gae") if arm.endswith("ppo") else None
        ps = fig4(env, it, "vstar") if arm.endswith("ppo") else None
        inside = (None if pv is None else
                  f"{'yes' if min(per['gae']) <= pv <= max(per['gae']) else 'no'} / "
                  f"{'yes' if min(per['vstar']) <= ps <= max(per['vstar']) else 'no'} (V*(s,t): "
                  f"{'yes' if min(per['vstar_t']) <= ps <= max(per['vstar_t']) else 'no'})")
        OUT["E3"][f"{arm}/{it}"] = {"per_seed": per, "r2_vstar": r2, "r2_vstar_t": r2t, "paper_vtheta": pv,
                                    "paper_vstar": ps, "repro_maxabs": {k: max(v) if v else None for k, v in rep.items()},
                                    "n": len(per["gae"])}
        p(f"| {arm} | {it} | {fmt(pv)} | {fmt(ps)} | " + " | ".join(ms(per[k]) for k in KINDS) +
          f" | {fmt(float(np.mean(r2)))} / {fmt(float(np.mean(r2t)))} | "
          f"{fmt(max(rep['gae'])) if rep['gae'] else '–'} / {fmt(max(rep['vstar'])) if rep['vstar'] else '–'} | {inside or '–'} |")

# P3 sensitivity: replace V* by V*(s,t) (and the GAE arms by G - V) in the 14-test family
if E3OK:
    fam = {k: v["t_p"] for k, v in R["family"].items()}
    det = {}
    for env in ("hopper", "walker2d"):
        arm = f"{env}_ppo"
        avg = lambda kind: [float(np.mean([OUT["E3"][f"{arm}/{it}"]["per_seed"][kind][s] for it in (150, 300, 450)]))
                            for s in SEEDS]
        for lab, a, b in ((f"P3a' {env} V*(s,t)>Vθ", "vstar_t", "gae"), (f"P3a'' {env} (G−V*)>(G−Vθ)", "mc_vstar", "mc_vtheta"),
                          (f" {env} (G−Vθ) − GAE(Vθ)", "mc_vtheta", "gae"), (f" {env} (G−V*) − GAE(V*)", "mc_vstar", "vstar"),
                          (f" {env} V*(s,t) − V*", "vstar_t", "vstar")):
            det[lab] = paired_compare(avg(a), avg(b))
    for env in ("hopper", "walker2d"):
        f2 = dict(fam)
        f2[f"P3a {env} V*>Vθ"] = det[f"P3a' {env} V*(s,t)>Vθ"]["t_p"]
        H = holm(f2)
        det[f"P3a' {env} V*(s,t)>Vθ"]["p_holm_in_family"] = H[f"P3a {env} V*>Vθ"]["p_holm"]
    OUT["E3_tests"] = det
    p("\n## E3 paired tests over seeds (pairwise(1e4) averaged over its 150/300/450; labelled sensitivities, not the family)")
    p("| test | mean diff ± SE | wins/n | p | p_holm (P3a substituted into the 14-test family) | verdict |")
    p("|---|---|---|---|---|---|")
    for k, d in det.items():
        ph = d.get("p_holm_in_family")
        v = ("HOLDS" if (ph is not None and ph < 0.05 and d["diff_mean"] > 0) else
             ("PARTIAL" if d["diff_mean"] > 0 else "FAILS")) if ph is not None else "–"
        p(f"| {k} | {fmt(d['diff_mean'])} ± {fmt(d['diff_se'])} | {d['wins']}/{d['n']} | {d['t_p']:.2g} | "
          f"{'–' if ph is None else f'{ph:.2g}'} | {v} |")

# ====================================================================== E4
p("\n## E4 rerun (GPU, seed 0): estimators, unclipped surface, step-axis statistics, un-annealed PPO step")
p("| arm | it | regime | ‖Δ‖/‖u‖ (‖u‖=2) / (unit u) | T(1,0)−T(0,0) all | start | firstk | review all | mean(start−all) on β=0 (SE units) | "
  "Spearman S–T grid / step (start) | S_unclip grid / step | surr gain clipped / unclipped |")
p("|---|---|---|---|---|---|---|---|---|---|---|---|")
OUT["E4"] = {}


def e4row(arm, it, reg, d, old, tag):
    L = d[f"it{it}"]["landscape"][reg]
    A_ = L["by_pairs"]["100000"]
    S_ = L.get("by_pairs_start", {}).get("100000")
    K_ = L["by_pairs_firstk"]["100000"]
    T, Ts, Tk = np.array(A_["T"]), np.array(S_["T"]) if S_ else None, np.array(K_["T"])
    se, ses = np.array(A_["T_se"]), np.array(S_["T_se"]) if S_ else None
    g = lambda X, SE: (float(X[2, 0] - X[0, 0]), float(math.hypot(SE[2, 0], SE[0, 0])))
    ga, gs, gk = g(T, se), (g(Ts, ses) if S_ else None), g(Tk, np.array(K_["T_se"]))
    bias = float(np.mean((Ts[:, 0] - T[:, 0]) / np.hypot(ses[:, 0], se[:, 0]))) if S_ else None
    oldg = None
    if old is not None:
        Lo = old[f"it{it}"]["landscape"][reg]["by_pairs"]["100000"]
        To, so = np.array(Lo["T"]), np.array(Lo["T_se"])
        oldg = g(To, so)
    rec = {"delta_norm": L["delta_norm"], "u_norm": L["u_norm"], "gain_all": ga, "gain_start": gs, "gain_firstk": gk,
           "gain_review_all": oldg, "start_minus_all_se_units_beta0": bias,
           "spearman_grid_start": S_["spearman_S_T"] if S_ else None,
           "spearman_step_start": S_["spearman_S_T_step"] if S_ else None,
           "spearman_unclip_grid_start": S_.get("spearman_Sunclip_T") if S_ else None,
           "spearman_unclip_step_start": S_.get("spearman_Sunclip_T_step") if S_ else None,
           "surr_gain_clip": A_["surr_gain_step"], "surr_gain_unclip": A_.get("surr_gain_step_unclip"),
           "spearman_grid_all": A_["spearman_S_T"], "spearman_step_all": A_.get("spearman_S_T_step"),
           "T_beta0_all": T[:, 0].tolist(), "T_beta0_start": None if Ts is None else Ts[:, 0].tolist()}
    OUT["E4"][f"{arm}/{it}/{reg}{tag}"] = rec
    gs_ = lambda x: "–" if x is None else f"{fmt(x[0])} ± {fmt(x[1])}"
    p(f"| {arm}{tag} | {it} | {reg} | {fmt(L['delta_norm'] / L['u_norm'])} / {fmt(L['delta_norm'])} | {gs_(ga)} | {gs_(gs)} | "
      f"{gs_(gk)} | {gs_(oldg)} | {fmt(bias)} | {fmt(rec['spearman_grid_start'])} / {fmt(rec['spearman_step_start'])} | "
      f"{fmt(rec['spearman_unclip_grid_start'])} / {fmt(rec['spearman_unclip_step_start'])} | "
      f"{fmt(rec['surr_gain_clip'])} / {fmt(rec['surr_gain_unclip'])} |")


for arm in ARMS:
    d = jload(os.path.join(G3, arm, "seed_0", "sa_land.json"))
    old = jload(os.path.join(G3, arm, "seed_0", "fullr_land.json"))
    if d is None:
        p(f"| {arm} | pending |")
        continue
    for it in sorted([int(k[2:]) for k in d if k.startswith("it")]):
        for reg in ("few", "many"):
            e4row(arm, it, reg, d, old, "")
    dn = jload(os.path.join(G3, arm, "seed_0", "sa_land_na.json"))
    if dn is not None:
        for it in sorted([int(k[2:]) for k in dn if k.startswith("it")]):
            for reg in ("few", "many"):
                e4row(arm, it, reg, dn, None, " (no anneal)")

# C9c recount: significant (|z| > 2) decreases/increases along the many-sample step, per estimator
if OUT["E4"]:
    c9 = {}
    for k, v in OUT["E4"].items():
        arm, it, reg = k.split("/")
        if not reg.startswith("many"):
            continue
        for est in ("gain_all", "gain_start"):
            g = v[est]
            if g is None:
                continue
            z = g[0] / g[1] if g[1] > 0 else 0.0
            key = f"{arm.split('_')[1]}{' no-anneal' if 'no anneal' in reg else ''}/{est}"
            c9.setdefault(key, {"dec": 0, "inc": 0, "ns": 0, "cells": 0})
            c9[key]["cells"] += 1
            c9[key]["dec" if z < -2 else ("inc" if z > 2 else "ns")] += 1
    OUT["C9c_recount"] = c9
    p(f"- C9c recount (many-sample, |z| > 2): {c9}")

# ====================================================================== fresh seeds + official-on-MJX
p("\n## Hopper TRPO: fresh seeds 5–9 (ours, MJX, unchanged code) and official-on-MJX seeds 0–4, instrument "
  "(m = 10, reference 1e5)")
fresh = {}
for f in sorted(glob.glob(os.path.join(SA, "fresh", "hopper_trpo", "seed_*", "rtb.json"))):
    d = json.load(open(f))
    fresh[d["seed"]] = d
orig = {}
for s in SEEDS:
    d = jload(os.path.join(REPO, "runs", "review", "ref1e5", "hopper_trpo", f"seed_{s}.json"))
    if d:
        orig[s] = d
offm = {}
for f in sorted(glob.glob(os.path.join(SA, "official-oracle", "onmjx", "hopper_trpo_s*.json"))):
    d = json.load(open(f))
    if d.get("cells"):
        offm[d["seed"]] = d
OUT["late_trpo_to_true"] = {"fresh_seeds": sorted(fresh), "orig_seeds": sorted(orig), "official_mjx_seeds": sorted(offm)}


def wtxt(w):
    return f"{fmt(w['diff'])} ± {fmt(w['se'])} (p {w['p']:.2g})"


def off_cell(d, it, n, met):
    c = d["cells"].get(str(it)) or d["cells"].get(f"it{it}")
    if c is None:
        return None
    for k in (str(n), f"n{n}"):
        if k in c:
            return c[k][met] if isinstance(c[k], dict) else None
    return c.get(f"{met}/{n}")


p("| it | n | metric | ours seeds 0–4 (full-scale, ref 1e5) | ours fresh seeds 5–9 | ours all 10 | official on MJX | "
  "Welch ours-10 vs official | paper |")
p("|---|---|---|---|---|---|---|---|---|")
for it in (150, 300, 450):
    for n in (2000, 10000):
        for met in ("to_true", "pairwise"):
            a = [orig[s][f"it{it}"]["grad"][str(n)][met] for s in sorted(orig)]
            b = [fresh[s][f"it{it}"]["grad"][str(n)][met] for s in sorted(fresh) if f"it{it}" in fresh[s]]
            o = [off_cell(offm[s], it, n, met) for s in sorted(offm)]
            o = [x for x in o if x is not None]
            w = welch(a + b, o) if len(a + b) >= 2 and len(o) >= 2 else None
            pc = R["E1"].get(f"hopper_trpo/{it}/{n}/{met}", {}).get("paper")
            OUT["late_trpo_to_true"][f"{it}/{n}/{met}"] = {"ours_orig": a, "ours_fresh": b, "official_mjx": o, "welch": w}
            p(f"| {it} | {n} | {met} | {ms(a)} | {ms(b)} | {ms(a + b)} | {ms(o)} | "
              f"{'–' if w is None else wtxt(w)} | "
              f"{'–' if not pc else fmt(pc[0])} |")
# early-reward lag with fresh seeds
fm = {}
for f in sorted(glob.glob(os.path.join(SA, "fresh", "hopper_trpo", "seed_*", "metrics.json"))):
    d = json.load(open(f))
    fm[d["seed"]] = [h["mean_reward"] for h in d["history"]]
if fm:
    p("\n- (Hopper TRPO mean training reward per window; ours = full-scale seeds 0–4 + fresh 5–9; official = "
      "MJX seeds 0–4):")
    for lo, hi in ((40, 49), (60, 70), (100, 110), (140, 151), (290, 301), (430, 451)):
        a = [float(np.mean([h["mean_reward"] for h in metrics("hopper_trpo", s)[lo:hi]])) for s in SEEDS]
        b = [float(np.mean(v[lo:hi])) for v in fm.values() if len(v) >= hi]
        o = [float(np.mean([h["mean_reward"] for h in d["history"][lo:hi]])) for d in offm.values()
             if len(d.get("history", [])) >= hi]
        w = welch(a + b, o) if len(o) >= 2 else None
        OUT.setdefault("early_trpo_lag", {})[f"{lo}-{hi - 1}"] = {"ours_orig": a, "ours_fresh": b, "official_mjx": o, "welch": w}
        p(f"  - its {lo}–{hi - 1}: ours 0–4 {ms(a, 0)}, fresh 5–9 {ms(b, 0)}, official-MJX {ms(o, 0)}"
          + ("" if w is None else f"; ours-10 − official {fmt(w['diff'])} ± {fmt(w['se'])} (Welch p {w['p']:.2g})"))

# ====================================================================== lockstep, env parity
p("\n## Lockstep multi-iteration differential (10 iterations, our batches replayed through the official primitives)")
for f in sorted(glob.glob(os.path.join(SA, "lockstep", "*.json"))):
    d = json.load(open(f))
    rows = d["rows"]
    first_bad = next((r["it"] for r in rows if not r["ok"]), None)
    OUT.setdefault("lockstep", {})[os.path.basename(f)[:-5]] = {
        "strict_pass": d["pass"], "within_float_chaos": d.get("pass_within_float_chaos"),
        "first_iter_over_1e-4": first_bad,
        "pi_rel": [r["pi_max_rel"] for r in rows], "pi_control_rel": [r["pi_control_max_rel"] for r in rows],
        "v_rel": [r["v_max_rel"] for r in rows], "v_control_rel": [r["v_control_max_rel"] for r in rows]}
    p(f"- {os.path.basename(f)[:-5]}: strict 1e-4 {'PASS' if d['pass'] else 'FAIL'} (first iteration over: {first_bad}); "
      f"policy rel. diff it 0–4 max {max(r['pi_max_rel'] for r in rows[:5]):.1e}; it 9 {rows[-1]['pi_max_rel']:.1e} vs "
      f"ours-vs-ours control {rows[-1]['pi_control_max_rel']:.1e}; value net max {max(r['v_max_rel'] for r in rows):.1e} "
      f"(control {max(r['v_control_max_rel'] for r in rows):.1e}); within float chaos at every iteration: "
      f"{d.get('pass_within_float_chaos')}")
for k in ("legacy", "post_step"):
    d = jload(os.path.join(SA, "env_fix", f"env_parity_{k}.json"))
    if d:
        for e, r in d["rows"].items():
            OUT.setdefault("env_parity", {})[f"{k}/{e}"] = r
            p(f"- env parity {k} {e}: definitions {'PASS' if r['definition_pass'] else 'FAIL'}; one-step dynamics reward "
              f"error median {fmt(r['dyn_reward_abs_err_median'])} (p95 {fmt(r['dyn_reward_abs_err_p95'])}); next-obs rel. "
              f"error median {fmt(r['dyn_obs1_rel_err_median'])}; done mismatches {r['dyn_done_mismatches']}")

os.makedirs(SA, exist_ok=True)
json.dump(OUT, open(os.path.join(SA, "recheck_report.json"), "w"), indent=1, default=float)
open(os.path.join(SA, "recheck_tables.txt"), "w").write("\n".join(TXT) + "\n")
print("\n".join(TXT))

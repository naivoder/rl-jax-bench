"""Full-scale report: end-points at our final step vs the vector-digitized paper curves, the
pre-declared Holm family, secondary comparisons, ObGD-vs-Adam overshoot diagnostics, the no-ObGD lr selection on
SELECT seeds, and authors'-code references. Usage: python scripts/report.py [tag]  -> runs/<tag>/report.json"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from rl_jax_bench import stats as st  # noqa: E402

from src import eval as ev  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "full"
root = os.path.join(REPO, "runs", tag)
PAPER = json.load(open(os.path.join(REPO, "src", "configs", "paper_digitized.json")))["figures"]
SELECT = {10001, 10002, 10003}
rep = {"tag": tag}


def load(exp):
    out = {}
    for p in glob.glob(os.path.join(root, exp, "*", "seed_*", "metrics.json")):
        r = json.load(open(p))
        out.setdefault(r["arm"], {})[r["seed"]] = r
    return out


def endpoint(r, stride=None, x=None):
    """Per-run end-point. Default: the run's last bin at its own stride (metrics.json summary). With (stride, x):
    the mean return of episodes ending in (x - stride, x] (official plot.py bin semantics), used to compare with a
    figure whose grid stride differs from the run's chunk stride (: Fig. 4 is binned at 125k).
    A diverged lane is NaN here : it is counted separately and the arm-level paper rule applies
    (Fig. 2: an algorithm with any diverged run is shown as diverged/0; Fig. 4/7: its mean curve stops)."""
    if r.get("diverged_by_step") is not None:
        return float("nan")
    if stride is None:
        return float(r["summary"]["final_bin"])
    e = np.asarray(r["episodes"], float).reshape(-1, 3)
    m = (e[:, 0] > x - stride) & (e[:, 0] <= x)
    return float(e[m, 1].mean()) if m.any() else float("nan")


def arm_stats(runs, seeds, stride=None, x=None):
    v = np.array([endpoint(runs[s], stride, x) for s in seeds])
    ndiv = int(sum(runs[s].get("diverged_by_step") is not None for s in seeds))
    first = np.array([runs[s]["summary"]["bins"][0] for s in seeds], float)
    base = dict(n=len(v), per_seed=v.round(3).tolist(), diverged=ndiv, first_bin_mean=float(np.nanmean(first)))
    if ndiv:   # arm-level rule : no mean is formed over a mix of returns and failure indicators
        return dict(base, mean=None, se=None, ci90=None, arm_rule=f"diverged ({ndiv}/{len(v)} lanes)")
    m, se = st.mean_se(v)
    n = len(v)
    from scipy import stats as sst
    h = float(sst.t.ppf(0.95, n - 1) * se) if se else None
    return dict(base, mean=m, se=se, ci90=[m - h, m + h] if h else None, arm_rule="ok")


def paired_ok(a_runs, b_runs, seeds, stride=None, x=None):
    """Paired compare only when neither arm has a diverged lane (); else a note."""
    if any(r.get("diverged_by_step") is not None for rr in (a_runs, b_runs) for r in (rr[q] for q in seeds)):
        return dict(note="not tested: an arm has diverged lanes (arm-level rule)")
    return st.paired_compare([endpoint(a_runs[q], stride, x) for q in seeds], [endpoint(b_runs[q], stride, x) for q in seeds])


def diag_summary(runs, seeds, kind):
    """Lane diagnostics summed over chunks: overshoot rate, bound-active fraction, max P1 excess."""
    out = {}
    for s in seeds:
        r = runs[s]
        cols = r["diag_cols"]
        d = np.array(r["diag"], float)
        if d.size == 0:
            continue
        g = {c: d[:, i] for i, c in enumerate(cols)}
        n = g["n"].sum()
        act = g["n_active"].sum() if "n_active" in g else g.get("n_active_v", np.zeros(1)).sum()
        out[s] = dict(overshoot_rate=float(g["n_overshoot"].sum() / max(n, 1)),
                      bound_active=float(act / max(n, 1)), max_viol=float(np.nanmax(g["max_viol"])))
    keys = ["overshoot_rate", "bound_active", "max_viol"]
    return {k: (float(np.mean([o[k] for o in out.values()])) if k != "max_viol" else
                float(np.max([o[k] for o in out.values()]))) for k in keys} if out else {}


def fig_stride(fig, panel, series):
    """The figure's bin stride in steps (src.eval.figure_stride; 125k on Fig. 4, 50k on Fig. 7/8)."""
    return ev.figure_stride(PAPER[fig][panel][series]["points"])


def paper_at(fig, panel, series, x):
    """Paper value AT a figure grid point (src.eval.grid_value; never interpolated, ); None off-grid."""
    e = PAPER[fig][panel].get(series)
    if e is None:
        return None
    if float(x) > e["x_end"] + 1e-6:
        return dict(value=None, note=f"paper curve ends at {e['x_end']:.2f}M (mean over runs non-finite: diverged)")
    v = ev.grid_value(e["points"], x)
    strd = fig_stride(fig, panel, series)
    if v is None:
        return dict(value=None, note=f"{x}M is not a grid point of {fig} (stride {strd})")
    return dict(value=v, band90=e["band"].get(str(float(x))), stride=strd)


def judge(ours, paper, band):
    if paper is None or ours is None:
        return None, None
    d = (ours - paper) / abs(paper)
    inside = band is not None and band[0] <= ours <= band[1]
    j = "MATCH" if (abs(d) <= 0.05 or inside) else ("CLOSE" if abs(d) <= 0.20 else "DIVERGENT")
    return float(d), j


def one_sided(pc, direction=+1):
    """One-sided p in the stated direction from the two-sided paired t."""
    t, p = pc["t"], pc["t_p"]
    return p / 2 if np.sign(t) == direction else 1 - p / 2


family, fam_dir, fam_ci = {}, {}, {}

# ------------------------------------------------------------------ E3 MinAtar
# Figure 4 is binned at 125k (5M / 40). Every Fig.-4 comparison re-bins OUR raw episodes at the figure's
# stride and reads the paper AT its grid points (0.75 / 0.875 / 1.0M); the pre-declared 50k last-bin end-point is
# kept alongside (`*_50k`) for the record. H1 is tested on the figure-grid end-point (both pass).
FIG4_GRID = (0.75, 0.875, 1.0)


def _judged(rr, seeds, fig, game, series, stride, xx):
    s_ = arm_stats(rr, seeds, stride, xx * 1e6)
    pa = paper_at(fig, game, series, xx)
    out = dict(mean=s_["mean"], se=s_["se"], n=s_["n"], diverged=s_["diverged"], per_seed=s_["per_seed"])
    if pa and pa.get("value") is not None and s_["mean"] is not None:
        out["paper"] = pa
        out["delta_rel"], out["judgment"] = judge(s_["mean"], pa["value"], pa.get("band90"))
    return out


for game in ("Breakout", "SpaceInvaders", "Asterix"):
    runs = load(f"full_e3_{game}")
    if not runs:
        continue
    any_r = next(iter(next(iter(runs.values())).values()))
    T = int(any_r["config"]["total_steps"])
    x = T / 1e6
    s4 = fig_stride("fig4", game, "stream_q")
    e = {"T": T, "schedule_steps": int(any_r["config"].get("schedule_steps", T)), "fig4_stride": s4,
         "run_bin_stride": int(any_r["config"].get("bin_stride", 50_000)), "arms": {}}
    for a, rr in runs.items():
        seeds = sorted(rr)
        s_ = arm_stats(rr, seeds, s4, T)                    # end-point = the Fig.-4 bin ending at T
        s_["diag"] = diag_summary(rr, seeds, "q")
        s_["endpoint_50k"] = {k: v for k, v in arm_stats(rr, seeds).items() if k in ("mean", "se", "n")}
        ser = {"stream_q_eps05": "stream_q"}.get(a, a)
        if a in ("stream_q", "classic_q", "dqn1"):
            s_["grid"] = {str(xx): _judged(rr, seeds, "fig4", game, ser, s4, xx) for xx in FIG4_GRID if xx <= x}
            pa = paper_at("fig4", game, ser, x)
            if pa and pa.get("value") is not None and s_["mean"] is not None:
                s_["paper"] = pa
                s_["delta_rel"], s_["judgment"] = judge(s_["mean"], pa["value"], pa.get("band90"))
        e["arms"][a] = s_
    common = sorted(set(runs["stream_q"]) & set(runs["classic_q"]))
    pc = paired_ok(runs["stream_q"], runs["classic_q"], common, s4, T)
    e["stream_vs_classic"] = pc
    e["stream_vs_classic_50k"] = paired_ok(runs["stream_q"], runs["classic_q"], common)
    e["paper_stream_minus_classic_at_T"] = (paper_at("fig4", game, "stream_q", x)["value"]
                                            - paper_at("fig4", game, "classic_q", x)["value"])
    pdir = np.sign(e["paper_stream_minus_classic_at_T"])
    e["paper_direction_at_T"] = "stream > classic" if pdir > 0 else "classic > stream"
    if game in ("Breakout", "SpaceInvaders") and "t_p" in pc:
        h = "H1" if game == "Breakout" else "H2"
        family[h] = pc["t_p"]                      # two-sided (declared before the run)
        fam_dir[h] = (np.sign(pc["diff_mean"]) == pdir)
        fam_ci[h] = pc["boot95"][0] > 0 if pdir > 0 else pc["boot95"][1] < 0
    for other in ("dqn1", "stream_q_eps05"):
        if other in runs:
            cs = sorted(set(runs["stream_q"]) & set(runs[other]))
            e[f"stream_vs_{other}"] = paired_ok(runs["stream_q"], runs[other], cs, s4, T)
    rep[f"e3_{game}"] = e

# ------------------------------------------------------------------ E3 secondary arms on the common 8e5 checkpoint view
# 0.8M is NOT a Fig.-4 grid point ; the last full Fig.-4 bin inside the 8e5 view is (0.625M, 0.75M].
runs = load("full_e3ck800k_Breakout")
if runs:
    s4 = fig_stride("fig4", "Breakout", "stream_q")
    xv = 0.75
    e = {"T": 800000, "compared_at": xv, "fig4_stride": s4,
         "note": "all 80 lanes at the 8e5 checkpoint (run killed while writing the 1M files)", "arms": {}}
    for a, rr in runs.items():
        seeds = sorted(rr)
        ser = {"stream_q_eps05": "stream_q"}.get(a, a)
        s_ = _judged(rr, seeds, "fig4", "Breakout", ser, s4, xv)
        s_["endpoint_50k_at_0.8M"] = {k: v for k, v in arm_stats(rr, seeds).items() if k in ("mean", "se", "n")}
        e["arms"][a] = s_
    for other in ("dqn1", "stream_q_eps05", "classic_q"):
        cs = sorted(set(runs["stream_q"]) & set(runs[other]))
        e[f"stream_vs_{other}"] = paired_ok(runs["stream_q"], runs[other], cs, s4, xv * 1e6)
    rep["e3_Breakout_800k"] = e

# ------------------------------------------------------------------ E6/E7 hopper
runs = load("full_e6_hopper")
if runs:
    any_r = next(iter(runs["stream_ac"].values()))
    T = int(any_r["config"]["total_steps"])
    x = T / 1e6
    #  check: the run's end-point stride must equal the figure grid stride it is compared with (Fig. 7/8: 50k)
    for fg in ("fig7", "fig8"):
        assert fig_stride(fg, "Hopper", "stream_ac") == int(any_r["config"]["bin_stride"]), fg
    rep_seeds = sorted(runs["stream_ac"])
    lrs = sorted(a for a in runs if a.startswith("no_obgd@"))
    sel = {a: float(np.mean([endpoint(runs[a][s]) for s in SELECT if s in runs[a]])) for a in lrs}
    chosen = max(sel, key=sel.get) if sel else None
    e = {"T": T, "select_seed_endpoints": sel, "no_obgd_chosen": chosen, "arms": {}}
    names = {"stream_ac": "stream_ac", "no_norm": "no_norm", "no_ln": "no_ln", "no_sparse": "no_sparse",
             "classic_ac": "classic_ac", "stream_ac0": "stream_ac0", "classic_ac0": "classic_ac0"}
    for a, rr in runs.items():
        s_ = arm_stats(rr, rep_seeds)
        s_["diag"] = diag_summary(rr, rep_seeds, "ac")
        key = "no_obgd" if a == chosen else names.get(a)
        if key:
            fig = "fig8" if key in ("stream_ac0", "classic_ac0") else "fig7"
            pa = paper_at(fig, "Hopper", key, x)
            if pa:
                s_["paper"] = pa
                if pa.get("value") is not None:
                    s_["delta_rel"], _ = judge(s_["mean"], pa["value"], pa.get("band90"))
        e["arms"][a] = s_
    tests = {"H3": chosen, "H4": "no_norm", "H5": "no_ln", "H6": "no_sparse", "H7": "classic_ac", "H8": "stream_ac0"}
    e["tests"] = {}
    for h, a in tests.items():
        pc = paired_ok(runs["stream_ac"], runs[a], rep_seeds)
        if "t_p" not in pc:
            e["tests"][h] = dict(arm=a, **pc)
            continue
        pc["p_one_sided"] = one_sided(pc, +1)
        e["tests"][h] = dict(arm=a, **pc)
        family[h] = pc["p_one_sided"]
        fam_dir[h] = pc["diff_mean"] > 0
        fam_ci[h] = pc["boot95"][0] > 0
    # secondary: ObGD removal the largest drop (paired: no-ObGD vs each other ablated arm, lower is a bigger drop)
    e["obgd_largest_drop"] = {a: paired_ok(runs[a], runs[chosen], rep_seeds) for a in ("no_norm", "no_ln", "no_sparse")}
    e["classic_improve"] = {a: dict(first_bin=e["arms"][a]["first_bin_mean"], last=e["arms"][a]["mean"])
                            for a in ("classic_ac", "classic_ac0")}
    e["lr_sensitivity_reported_seeds"] = {a: e["arms"][a]["mean"] for a in lrs}
    rep["e6_hopper"] = e
    # E2 (secondary, struck with the proxy): PPO (papers/ppo_implementation_details code) on the same proxy, outputs under runs/ppo/, same 50k-bin end-point (episodes ending in (T-50k, T])
    pp = sorted(glob.glob(os.path.join(REPO, "runs", "ppo", tag, "brax-hopper", "default", "seed_*",
                                       "metrics.json")))
    if pp:
        ppo = {}
        for q in pp:
            r = json.load(open(q))
            ea = np.array(r["episodes"], float)
            Tp = int(r["per_update_global_step"][-1])
            m = (ea[:, 0] > T - 50_000) & (ea[:, 0] <= T)
            ppo[r["seed"]] = dict(endpoint=float(ea[m, 1].mean()), last_step=Tp, n_ep=int(m.sum()))
        ss = sorted(set(ppo) & set(rep_seeds))
        v = [ppo[q]["endpoint"] for q in ss]
        mm, se = st.mean_se(v)
        e2 = dict(ppo_per_seed=ppo, ppo_mean=mm, ppo_se=se,
                  stream_minus_ppo=st.paired_compare([endpoint(runs["stream_ac"][q]) for q in ss], v))
        rep["e2_ppo"] = e2

# ------------------------------------------------------------------ E8
# The primary MSE compares v(S_{j+1}) with G[j+1] (keys without suffix); '_official' keys are the official
# plotting alignment (pred[j] vs G[j]) that small-scale / full-scale reported. The review re-run (runs/review/e8_ETTm2, same
# 60-lane program => bitwise-identical trajectories) carries both; it is used when present and its
# official-aligned values are checked against the full-scale record.
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official"))  # authors' clone + venv (scripts/official/setup.sh)
OFF_TD = os.path.join(OFFICIAL_ROOT, "runs/td_ett/ref_seed*.npz")
review_e8 = os.path.join(REPO, "runs", "review")
runs = load("e8_ETTm2")
if os.path.isdir(os.path.join(review_e8, "e8_ETTm2")):
    old_runs, root_saved = runs, root
    root = review_e8
    runs = load("e8_ETTm2")
    root = root_saved
    same = [abs(runs[a][q]["summary"]["mse_last_official"] - old_runs[a][q]["summary"]["mse_last"]) for a in old_runs
            for q in old_runs[a]] if old_runs else []
    rep["e8_rerun_identity_max_abs_diff_vs_full"] = float(max(same)) if same else None
if runs:
    seeds = sorted(runs["stream_td"])
    g = lambda a, k: [runs[a][s]["summary"][k] for s in seeds]  # noqa: E731
    keys = [k for k in ("mse_first", "mse_last", "mse_last_excl_tail", "mse_first_official", "mse_last_official")
            if k in runs["stream_td"][seeds[0]]["summary"]]
    e = {k: {a: dict(zip(("mean", "se"), st.mean_se(g(a, k)))) for a in runs} for k in keys}
    pc = st.paired_compare(g("classic_td", "mse_last"), g("stream_td", "mse_last"))   # >0 = stream better
    pc["p_one_sided"] = one_sided(pc, +1)
    e["H9"] = pc
    if "mse_last_official" in keys:
        pco = st.paired_compare(g("classic_td", "mse_last_official"), g("stream_td", "mse_last_official"))
        pco["p_one_sided"] = one_sided(pco, +1)
        e["H9_official_alignment"] = pco
    e["n"] = len(seeds)
    family["H9"] = pc["p_one_sided"]
    fam_dir["H9"] = pc["diff_mean"] > 0
    fam_ci["H9"] = pc["boot95"][0] > 0
    # authors' code, same summary function on their saved per-step predictions (both alignments)
    refs = []
    for f in sorted(glob.glob(OFF_TD)):
        z = np.load(f)
        refs.append(ev.ett_summary(z["pred"], ev.discounted_returns(z["cum"], 0.99), z["cum"]))
    if refs:
        e["authors"] = {k: dict(zip(("mean", "se"), st.mean_se([r[k] for r in refs])), per_seed=[r[k] for r in refs])
                        for k in ("mse_first", "mse_last", "mse_last_excl_tail", "mse_first_official",
                                  "mse_last_official")}
    rep["e8"] = e

# ------------------------------------------------------------------ Holm
hh = st.holm(family) if family else {}
rep["family"] = {h: dict(**hh[h], direction_as_paper=bool(fam_dir[h]), boot_ci_excludes_0=bool(fam_ci[h]),
                         pass_=bool(hh[h]["p_holm"] < 0.05 and fam_dir[h] and fam_ci[h])) for h in sorted(hh)}
rep["family_size"] = len(family)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


json.dump(_clean(rep), open(os.path.join(root, "report.json"), "w"), indent=1)
for h, v in rep["family"].items():
    print(h, v)
print("[INFO] wrote", os.path.join(root, "report.json"))

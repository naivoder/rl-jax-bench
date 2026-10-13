"""Aggregate per-run metrics (runs/<tag>/<env>/<method>/seed_<n>/metrics.json) into summaries, compare with the
paper's plot read-offs and with official-code reference curves; mirror every run into
runs/seed_<n>/metrics.json (the required per-seed layout, keyed "<tag>/<env>/<method>").

    python src/eval.py --tag small
No JAX needed (numpy/scipy only).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

from rl_jax_bench.stats import holm, mean_se, paired_compare  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL_RUNS = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official")), "runs")
METHODS = ["a2c_asym_hs", "a2c_asym_s", "a2c", "a2c_react_2", "a2c_react_4"]

# paper Fig. 3 read-offs: final value (last ~5 % of training) and read precision
PAPER_FINAL = {
    "heavenhell_3": dict(prec=0.02, a2c_asym_hs=1.00, a2c_asym_s=0.16, a2c=0.10, a2c_react_2=0.00, a2c_react_4=0.00),
    "heavenhell_4": dict(prec=0.02, a2c_asym_hs=0.79, a2c_asym_s=0.04, a2c=0.01, a2c_react_2=-0.01, a2c_react_4=0.01),
    "shopping_5": dict(prec=3.0, a2c_asym_hs=3.0, a2c_asym_s=-6.0, a2c=3.0, a2c_react_2=-17.0, a2c_react_4=2.0),
    "shopping_6": dict(prec=3.0, a2c_asym_hs=2.0, a2c_asym_s=-7.0, a2c=2.0, a2c_react_2=-100.0, a2c_react_4=-83.0),
    "car_flag": dict(prec=0.02, a2c_asym_hs=0.45, a2c_asym_s=0.46, a2c=0.01, a2c_react_2=-0.01, a2c_react_4=0.01),
    "cleaner": dict(prec=1.5, a2c_asym_hs=68.5, a2c_asym_s=49.0, a2c=69.0, a2c_react_2=43.0, a2c_react_4=46.0),
}
# paper s.e.m. band at the end of training where the plot shows one wider than the read precision
PAPER_SPREAD = {("heavenhell_4", "a2c_asym_hs"): 0.09, ("heavenhell_3", "a2c"): 0.05}
# small-scale proxy (HH-3 stopped at 1M): Fig. 3a read at 1M (500-dpi render, x 67 px/M, y 445 px/unit; steep part of the
# asym-hs curve, so the read is coarse: +-0.10). Others are flat near their 1M values.
PAPER_AT_1M_HH3 = dict(prec=0.10, a2c_asym_hs=0.40, a2c_asym_s=0.07, a2c=0.00, a2c_react_2=0.00, a2c_react_4=0.00)


def load_cell(tag, env, method):
    recs = []
    for p in sorted(glob.glob(os.path.join(REPO, "runs", tag, env, method, "seed_*", "metrics.json")),
                    key=lambda s: int(re.search(r"seed_(\d+)", s).group(1))):
        recs.append(json.load(open(p)))
    return recs


def first_crossing(t, y, thr, mode=None):
    """First t at which the polyline (t, y) reaches thr. linear interpolation between the
    vertices on either side of the crossing, the same estimator for the paper's 101-vertex curve and for our curves
    (the pre-fix version returned the first VERTEX >= thr, which biases a coarse curve late by up to one grid
    interval). UA_CROSSING=vertex (or mode='vertex') reproduces the pre-fix estimator."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    idx = np.nonzero(y >= thr)[0]
    if not len(idx):
        return float("nan")
    i = int(idx[0])
    if (mode or os.environ.get("UA_CROSSING", "interp")) == "vertex" or i == 0:
        return float(t[i])
    return float(t[i - 1] + (thr - y[i - 1]) / (y[i] - y[i - 1]) * (t[i] - t[i - 1]))


def judge(ours, paper, prec, spread=0.0):
    """judgment rule: tol_abs = max(spread, prec) (the paper's s.e.m. half-width or the read precision, whichever
    is larger -- NOT their sum). MATCH if within 5 % relative or |ours - paper| <= tol_abs; CLOSE if <= 20 % relative;
    else DIVERGENT. Near-zero paper values (|paper| < 5 prec) use the absolute rule only: MATCH <= tol_abs,
    CLOSE <= 4 tol_abs. The spread is the paper's 20-run band, our n is 5 (stated with every verdict)."""
    tol_abs = max(spread, prec)
    if abs(paper) < 5 * prec:
        d = abs(ours - paper)
        return "MATCH" if d <= tol_abs else ("CLOSE" if d <= 4 * tol_abs else "DIVERGENT")
    rel = abs(ours - paper) / abs(paper)
    if rel <= 0.05 or abs(ours - paper) <= tol_abs:
        return "MATCH"
    return "CLOSE" if rel <= 0.20 else "DIVERGENT"


def summarize(tag, env, thresholds=()):
    out = {}
    for m in METHODS:
        recs = load_cell(tag, env, m)
        if not recs:
            continue
        fin = [r["final"] for r in recs]
        auc = [r["auc"] for r in recs]
        fm, fse = mean_se(fin)
        am, ase = mean_se(auc)
        row = dict(n=len(recs), final_mean=fm, final_se=fse, auc_mean=am, auc_se=ase, final_per_run=fin,
                   auc_per_run=auc, t_env_final=[r["t_env_final"] for r in recs],
                   n_log_points=[r["n_log_points"] for r in recs])
        for thr in thresholds:
            tc = [first_crossing(r["curve"]["t"], r["curve"]["avg100"], thr) for r in recs]
            row[f"t_first_avg100_ge_{thr}"] = tc
        tim = recs[0].get("timing") or {}
        row["timing"] = {k: tim.get(k) for k in ("wall_s", "first_chunk_s", "steady_s", "env_steps_per_s")}
        if recs[0]["curve"].get("probe") and recs[0].get("probe_names"):
            pr = np.array([np.asarray(r["curve"]["probe"])[-10:].mean(0) for r in recs])
            row["probe_final_mean"] = pr.mean(0).tolist()
            row["probe_final_se"] = (pr.std(0, ddof=1) / np.sqrt(len(pr))).tolist() if len(pr) > 1 else None
        # mean curve on a common grid (for plots / reports)
        grid = np.linspace(0, max(r["t_env_final"] for r in recs), 101)
        curves = np.array([np.interp(grid, r["curve"]["t"], r["curve"]["avg100"]) for r in recs])
        row["curve_grid"] = grid.tolist()
        row["curve_mean"] = curves.mean(0).tolist()
        row["curve_se"] = (curves.std(0, ddof=1) / np.sqrt(len(recs))).tolist() if len(recs) > 1 else None
        out[m] = row
    return out


def paired(summary, a, b, key="final_per_run"):
    if a not in summary or b not in summary:
        return None
    from scipy import stats as sst
    x, y = summary[a][key], summary[b][key]
    n = min(len(x), len(y))
    res = paired_compare(x[:n], y[:n])
    res["welch_p"] = float(sst.ttest_ind(x, y, equal_var=False).pvalue)   # pairing by index is nominal
    return res


def official_curves(pattern):
    """Parse WANDB json lines of official runs (scripts/official/official_main_shim.py) -> list of (t, avg100) arrays."""
    res = {}
    for p in sorted(glob.glob(os.path.join(OFFICIAL_RUNS, pattern))):
        t, y = [], []
        for line in open(p, errors="ignore"):
            if line.startswith("WANDB ") and "performance/avg100_behavior_mean_return" in line:
                d = json.loads(line[6:])
                t.append(d["simulation_timesteps"])
                y.append(d["performance/avg100_behavior_mean_return"])
        if t:
            res[os.path.basename(p)] = dict(t=t, avg100=y, t_last=t[-1], last10=float(np.mean(y[-10:])))
    return res


def mirror_per_seed(tag):
    """runs/seed_<n>/metrics.json: {"<tag>/<env>/<method>": record} for every run index n."""
    by_seed = {}
    for p in glob.glob(os.path.join(REPO, "runs", tag, "*", "*", "seed_*", "metrics.json")):
        r = json.load(open(p))
        by_seed.setdefault(r["run_idx"], {})[f"{tag}/{r['env']}/{r['method']}"] = r
    for n, recs in by_seed.items():
        path = os.path.join(REPO, "runs", f"seed_{n}", "metrics.json")
        cur = json.load(open(path)) if os.path.exists(path) else {}
        cur.update(recs)
        tmp = path + ".tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(cur, f)
        os.replace(tmp, path)


def small_report():
    rep = {}
    s5 = summarize("small", "shopping_5", thresholds=(-10, -5))
    h3 = summarize("small", "heavenhell_3", thresholds=(0.5,))
    rep["shopping_5"] = {m: {k: v for k, v in r.items() if not k.startswith("curve")} for m, r in s5.items()}
    rep["heavenhell_3_1M"] = {m: {k: v for k, v in r.items() if not k.startswith("curve")} for m, r in h3.items()}
    rep["judgments"] = {}
    for m, r in s5.items():
        p = PAPER_FINAL["shopping_5"]
        rep["judgments"][f"shopping_5/{m}"] = dict(ours=r["final_mean"], se=r["final_se"], paper=p[m],
                                                   judgment=judge(r["final_mean"], p[m], p["prec"]))
    for m, r in h3.items():
        # the 1M read-off sits on the steepest part of Fig. 3a; judge against the vector-extracted
        # paper curve averaged over the same window as our `final` (last 10 log points), as in full-scale. The coarse
        # read-off judgment is kept as `judgment_readoff` (superseded).
        p = PAPER_AT_1M_HH3
        recs = load_cell("small", "heavenhell_3", m)
        w0 = float(np.mean([r_["curve"]["t"][-10] for r_ in recs]))
        w1 = float(np.mean([r_["curve"]["t"][-1] for r_ in recs]))
        pm, ph = paper_window("heavenhell_3", m, w0, w1)
        rep["judgments"][f"heavenhell_3@1M/{m}"] = dict(
            ours=r["final_mean"], se=r["final_se"], n=r["n"], paper=pm, paper_sem_half=ph, window=[w0, w1],
            judgment=judge(r["final_mean"], pm, PREC_EXACT["heavenhell_3"], ph or 0.0),
            paper_readoff=p[m], judgment_readoff=judge(r["final_mean"], p[m], p["prec"]))
    tests = {}
    for env, s in (("shopping_5", s5), ("heavenhell_3", h3)):
        for b in ("a2c_asym_s", "a2c", "a2c_react_2", "a2c_react_4"):
            pc = paired(s, "a2c_asym_hs", b)
            if pc is not None:
                tests[f"{env}: hs - {b} (final)"] = pc
        pc = paired(s, "a2c_asym_hs", "a2c", key="auc_per_run")
        if pc is not None:
            tests[f"{env}: hs - a2c (auc)"] = pc
    adj = holm({k: v["t_p"] for k, v in tests.items()}) if tests else {}
    for k, v in tests.items():
        v["p_holm_exploratory"] = adj[k]["p_holm"]
    rep["paired_tests_exploratory"] = tests
    rep["official_reference"] = official_curves("shopping5_*.log")
    rep["official_reference"].update(official_curves("heavenhell3_*.log"))
    return rep, s5, h3


def small_markdown(rep) -> str:
    L = ["| setting | arm | ours final (mean ± s.e., n) | paper | judgment | auc | first t with avg100 ≥ thr (mean over runs reaching it) |",
         "|---|---|---|---|---|---|---|"]
    for env, key, thr in (("Shopping-5 @2M", "shopping_5", -10), ("HH-3 @1M", "heavenhell_3_1M", 0.5)):
        for m, r in rep[key].items():
            j = rep["judgments"][("shopping_5/" if key == "shopping_5" else "heavenhell_3@1M/") + m]
            tc = np.asarray(r.get(f"t_first_avg100_ge_{thr}", []), float)
            reach = tc[np.isfinite(tc)]
            tcs = f"{reach.mean() / 1e6:.2f}M ({len(reach)}/{len(tc)} runs)" if len(reach) else f"never (0/{len(tc)})"
            pap = (f"{j['paper']:+.3f} ± {j['paper_sem_half'] or 0:.3f} (vector curve; ~~read-off {j['paper_readoff']:+g}: "
                   f"{j['judgment_readoff']}~~)" if "paper_readoff" in j else f"{j['paper']:+g}")
            L.append(f"| {env} | {m} | {r['final_mean']:+.3f} ± {r['final_se'] or 0:.3f} ({r['n']}) | {pap} | "
                     f"{j['judgment']} | {r['auc_mean']:+.3f} | thr {thr}: {tcs} |")
    return "\n".join(L)


def value_at(rec, t0, t1):
    """Mean avg100 over a run's log points with t in (t0, t1] (official-code comparison at a fixed step)."""
    t, y = np.asarray(rec["curve"]["t"]), np.asarray(rec["curve"]["avg100"])
    sel = (t > t0) & (t <= t1)
    return float(y[sel].mean()) if sel.any() else float("nan")


def window_steps(tag, env, method):
    """W = 10 log intervals (T_max / n_log each) of a cell: the span our `final` (last 10 log points) covers."""
    c = json.load(open(os.path.join(REPO, "runs", tag, env, method, "config.json")))["cfg"]
    return 10.0 * c["max_env_steps"] / c["n_log"]


# authors'-code reference logs for the speed table (n = 3 at the paper settings; react-4: recheck)
OFFICIAL_SPEED = {("shopping_5", "a2c_asym_hs"): "shopping5_asym-a2c_s*.log",
                  ("shopping_5", "a2c_asym_s"): "shopping5_asym-a2c-state_s*.log",
                  ("shopping_5", "a2c"): "shopping5_a2c_s*.log",
                  ("shopping_5", "a2c_react_4"): "recheck_shopping5_react4_s*.log",
                  ("heavenhell_3", "a2c_asym_hs"): "heavenhell3_asym-a2c_s*.log"}

# Cleaner collapse test windows, fixed from the paper's Fig. 3f vector curve before any Cleaner run
CLEANER_EARLY, CLEANER_LATE = (3.0e6, 3.5e6), (3.8e6, 4.0e6)


def _safe_paired(a, b):
    try:
        return paired_compare(list(a), list(b))
    except ValueError as e:          # e.g. all paired differences exactly 0 (same state => same V(s))
        return dict(diff_mean=float(np.mean(np.asarray(a) - np.asarray(b))), error=str(e))


# scripts/paper_curves.py (exact PDF vectors). axis maps calibrated on the plot gridlines.
# UA_PAPER_CURVES=src/configs/paper_fig3_curves_labelcal.json reproduces the pre-fix (tick-label-center) reductions.
PAPER_CURVES = os.environ.get("UA_PAPER_CURVES") or os.path.join(REPO, "src", "configs", "paper_fig3_curves.json")
if not os.path.isabs(PAPER_CURVES):
    PAPER_CURVES = os.path.join(REPO, PAPER_CURVES)


# judgment precision floor for the vector-extracted paper values. HH / Car-Flag: the absolute floor
# 0.02 for near-zero values. Shopping 0.35 and Cleaner 0.6 are STATED CONVENTIONS fixed before full-scale (recheck:
# the "0.5 pt of line width" rationale gives ~1.1 for Shopping, not 0.35, and the gridline-calibrated extraction is
# accurate to ~0.03 pt = 0.07 units; the 0.35 convention is kept, not loosened, so no verdict moves through it)
PREC_EXACT = {"heavenhell_3": 0.02, "heavenhell_4": 0.02, "car_flag": 0.02, "shopping_5": 0.35, "shopping_6": 0.35,
              "cleaner": 0.6}


def paper_window(env, m, t0, t1):
    """Paper Fig. 3 mean curve averaged over [t0, t1] and the mean s.e.m. half-width there (vector-extracted)."""
    pc = json.load(open(PAPER_CURVES)).get(env, {}).get(m)
    if not pc or "t" not in pc:     # never fall back to a literal read-off
        raise KeyError(f"no vector-extracted paper curve for {env}/{m} in {PAPER_CURVES}")
    return curve_window(pc, t0, t1)


PAPER_FIG4 = os.path.join(REPO, "src", "configs", "paper_fig4_curves.json")   # scripts/paper_curves.py --fig4


def curve_window(pc, t0, t1):
    """Mean of an extracted paper curve over [t0, t1] and the mean half-width of its band polygon there."""
    t, y = np.asarray(pc["t"]), np.asarray(pc["mean"])
    g = np.linspace(t0, min(t1, t.max()), 50)
    mean = float(np.interp(g, t, y).mean())
    half = None
    poly = np.asarray(pc.get("band_poly", []))
    if len(poly) > 4:
        k = int(np.argmax(poly[:, 0]))          # fill_between: one edge forward to max t, the other edge back
        e1, e2 = poly[1:k + 1], poly[k + 1:]     # poly[0] is the closing vertical segment's start
        e1, e2 = e1[np.argsort(e1[:, 0])], e2[np.argsort(e2[:, 0])]
        half = float(np.mean(np.abs(np.interp(g, e1[:, 0], e1[:, 1]) - np.interp(g, e2[:, 0], e2[:, 1]))) / 2)
    return mean, half


def full_report():
    """full-scale: per-cell judgments vs the vector-extracted paper curves, the primary and secondary test families (pre-registered,
    results/full_preregistration.json), official-code reference values, Figure-4 probe structure, speed, ablation A1.

    Truncated cells (stopped at the budget): all runs of a cell share one wall clock, so runs that
    learned (shorter episodes) reach fewer env steps. The PRIMARY per-cell value is therefore the COMMON-STEP value:
    each run's mean avg100 over (t_c - W, t_c], t_c = the cell's smallest stop step, W = 10 log intervals, against the
    paper curve over the same window. Two per-run-window variants are kept as a sensitivity table. Every between-arm
    test uses the common window of BOTH arms (t_c = smallest stop over the two arms' runs)."""
    from scipy import stats as sst
    cells = {"heavenhell_3": "full", "heavenhell_4": "full", "shopping_5": "small", "shopping_6": "full",
             "car_flag": "full", "cleaner": "full"}
    thr = {"heavenhell_3": (0.5, 0.9), "heavenhell_4": (0.5,), "shopping_5": (-10, -5), "shopping_6": (-10, -5),
           "car_flag": (0.3,), "cleaner": (40, 60)}
    S, rep = {}, dict(cells={}, judgments={}, sensitivity={}, primary_tests={}, secondary_tests={}, exploratory={})
    for env, tag in cells.items():
        S[env] = summarize(tag, env, thresholds=thr[env])
        rep["cells"][env] = {m: {k: v for k, v in r.items() if not k.startswith("curve")} for m, r in S[env].items()}
        for m, r in S[env].items():
            recs = load_cell(tag, env, m)
            prec = PREC_EXACT[env]
            trunc = any(r_.get("truncated") for r_ in recs)
            W = window_steps(tag, env, m)
            # (A) per-run window: mean of per-run finals vs paper over the MEAN window (the earlier primary)
            w0 = float(np.mean([r_["curve"]["t"][-10] for r_ in recs]))
            w1 = float(np.mean([r_["curve"]["t"][-1] for r_ in recs]))
            pA, hA = paper_window(env, m, w0, w1 + (0 if trunc else 1e9))
            jA = judge(r["final_mean"], pA, prec, hA or 0.0)
            # informativeness: would the non-learning start-of-training policy also MATCH?
            base = float(np.mean([np.mean(r_["curve"]["avg100"][:5]) for r_ in recs]))
            if trunc:
                # (B) paper averaged over each run's own window, then over runs
                pw = [paper_window(env, m, r_["curve"]["t"][-10], r_["curve"]["t"][-1]) for r_ in recs]
                pB, hB = float(np.mean([p[0] for p in pw])), float(np.mean([p[1] or 0.0 for p in pw]))
                jB = judge(r["final_mean"], pB, prec, hB)
                # (C) common step: all runs over (t_c - W, t_c], t_c = smallest stop in the cell  -> PRIMARY
                tc = float(min(r_["curve"]["t"][-1] for r_ in recs))
                ours = [value_at(r_, tc - W, tc) for r_ in recs]
                om, ose = mean_se(ours)
                pm, ph = paper_window(env, m, tc - W, tc)
                win = [tc - W, tc]
                rep["sensitivity"][f"{env}/{m}"] = dict(
                    A_mean_window=dict(ours=r["final_mean"], paper=pA, judgment=jA, window=[w0, w1]),
                    B_per_run_window=dict(ours=r["final_mean"], paper=pB, judgment=jB),
                    C_common_step=dict(ours=om, se=ose, paper=pm, judgment=judge(om, pm, prec, ph or 0.0), window=win),
                    t_stop_per_run=[r_["curve"]["t"][-1] for r_ in recs], final_per_run=r["final_per_run"],
                    spearman_tstop_final=float(sst.spearmanr([r_["curve"]["t"][-1] for r_ in recs],
                                                             r["final_per_run"]).statistic))
            else:
                ours, (om, ose), pm, ph, win = r["final_per_run"], (r["final_mean"], r["final_se"]), pA, hA, [w0, w1]
            j = judge(om, pm, prec, ph or 0.0)
            jbase = judge(base, pm, prec, ph or 0.0)
            rel = (om - pm) / abs(pm) * 100 if abs(pm) >= 5 * prec else None
            rep["judgments"][f"{env}/{m}"] = dict(
                ours=om, se=ose, n=r["n"], ours_per_run=ours, paper=pm, paper_sem_half=ph,
                paper_final_T=PAPER_FINAL[env][m], window=win, common_step=trunc, truncated=trunc, delta_pct=rel,
                judgment=j, data_tag=tag, start_of_training_avg100=base,
                informative=not (j == "MATCH" and jbase == "MATCH"),
                z_vs_paper=float((om - pm) / np.hypot(ose or 0.0, ph or 0.0)) if (ose or ph) else None)

    def add(fam, name, env, a, b, key="final"):
        """Between-arm test on per-run scalars at a COMMON step for both arms. Pre-registered test:
        paired t by run_idx; the pairing is nominal (no common random numbers across arms), so Welch is reported
        alongside and Holm-adjusted separately."""
        tag = cells[env]
        ra, rb = load_cell(tag, env, a), load_cell(tag, env, b)
        if not ra or not rb:
            return
        if key == "auc":
            if any(r_.get("truncated") for r_ in ra + rb):
                tc = min(r_["curve"]["t"][-1] for r_ in ra + rb)   # AUC over the common steps t <= t_c
                f = lambda r_: float(np.mean([y for t, y in zip(r_["curve"]["t"], r_["curve"]["avg100"]) if t <= tc]))  # noqa: E731
                name += f" (t <= {tc / 1e6:.2f}M, common window)"
            else:
                f = lambda r_: r_["auc"]  # noqa: E731
        elif any(r_.get("truncated") for r_ in ra + rb):
            tc = min(r_["curve"]["t"][-1] for r_ in ra + rb)
            W = window_steps(tag, env, a)
            f = lambda r_: value_at(r_, tc - W, tc)  # noqa: E731
            name = name.replace(" final", "") + f" common step t∈({(tc - W) / 1e6:.2f}M, {tc / 1e6:.2f}M]"
        else:
            f = lambda r_: r_["final"]  # noqa: E731
        x, y = [f(r_) for r_ in ra], [f(r_) for r_ in rb]
        res = paired_compare(x, y)
        res.update(welch_p=float(sst.ttest_ind(x, y, equal_var=False).pvalue), a=x, b=y)
        fam[name] = res
    P_TESTS, S_TESTS = rep["primary_tests"], rep["secondary_tests"]
    add(P_TESTS, "HH-3 hs-A2C final", "heavenhell_3", "a2c_asym_hs", "a2c")
    add(P_TESTS, "HH-3 hs-asym_s final", "heavenhell_3", "a2c_asym_hs", "a2c_asym_s")
    add(P_TESTS, "HH-4 hs-A2C final", "heavenhell_4", "a2c_asym_hs", "a2c")
    add(P_TESTS, "HH-4 hs-asym_s final", "heavenhell_4", "a2c_asym_hs", "a2c_asym_s")
    add(P_TESTS, "Shopping-5 hs-asym_s final (small data)", "shopping_5", "a2c_asym_hs", "a2c_asym_s")
    add(P_TESTS, "Shopping-6 hs-asym_s final", "shopping_6", "a2c_asym_hs", "a2c_asym_s")
    add(P_TESTS, "Shopping-5 hs-A2C auc (small data)", "shopping_5", "a2c_asym_hs", "a2c", "auc")
    add(P_TESTS, "Shopping-6 hs-A2C auc", "shopping_6", "a2c_asym_hs", "a2c", "auc")
    # the as-launched items 1-4, 6 on unmatched truncated finals (superseded by the common-step versions; record only)
    rep["exploratory"]["F1_unmatched_finals_superseded"] = {
        k: paired(S[e], "a2c_asym_hs", b) for k, e, b in (
            ("HH-3 hs-A2C final", "heavenhell_3", "a2c"), ("HH-3 hs-asym_s final", "heavenhell_3", "a2c_asym_s"),
            ("HH-4 hs-A2C final", "heavenhell_4", "a2c"), ("HH-4 hs-asym_s final", "heavenhell_4", "a2c_asym_s"),
            ("Shopping-6 hs-asym_s final", "shopping_6", "a2c_asym_s")) if e in S}
    add(S_TESTS, "Car-Flag hs-A2C final", "car_flag", "a2c_asym_hs", "a2c")
    add(S_TESTS, "Car-Flag hs-asym_s final", "car_flag", "a2c_asym_hs", "a2c_asym_s")
    add(S_TESTS, "Cleaner hs-asym_s final", "cleaner", "a2c_asym_hs", "a2c_asym_s")
    # Cleaner collapse: the old peak(max over 200 points) - final test is biased upward on any noisy
    # curve. Replacement, declared before any Cleaner run: drop = mean over a fixed early window (3.0, 3.5]M (the paper's
    # asym-s mean curve peaks at 3.28M, 60.1, Fig. 3f vector extraction) minus mean over (3.8, 4.0]M; tested as
    # drop(asym-s) - drop(asym-hs), the matched control (paper asym-hs is flat there: 68.2 -> 68.6).
    ra, rb = load_cell("full", "cleaner", "a2c_asym_s"), load_cell("full", "cleaner", "a2c_asym_hs")
    if ra and rb:
        drop = lambda r_: value_at(r_, *CLEANER_EARLY) - value_at(r_, *CLEANER_LATE)  # noqa: E731
        x, y = [drop(r_) for r_ in ra], [drop(r_) for r_ in rb]
        res = paired_compare(x, y)
        res.update(welch_p=float(sst.ttest_ind(x, y, equal_var=False).pvalue), a=x, b=y)
        S_TESTS["Cleaner drop(asym_s) - drop(hs), early (3.0,3.5]M minus late (3.8,4.0]M"] = res
    for fam in (P_TESTS, S_TESTS):
        if fam:
            adj = holm({k: v["t_p"] for k, v in fam.items()})
            adjw = holm({k: v["welch_p"] for k, v in fam.items()})
            for k in fam:
                fam[k]["p_holm"] = adj[k]["p_holm"]
                fam[k]["p_holm_welch"] = adjw[k]["p_holm"]
    ex = rep["exploratory"]
    # official-code reference (n = 3, CPU, 2M runs; final_2M = mean of the last 10 of 200 log points =
    # t in (1.9, 2.0]M) next to ours over the same window. Welch: two implementations, unpaired. A high p is NOT
    # evidence of equivalence (n = 3 vs 5, underpowered).
    off = json.load(open(os.path.join(REPO, "runs", "official_reference", "summary.json")))
    ex["official_vs_ours"] = {}
    for env in ("heavenhell_3", "shopping_5"):
        for m in ("a2c_asym_hs", "a2c_asym_s", "a2c"):
            o = off.get(f"paper/{env}/{m}", {}).get("final_2M")
            recs = load_cell(cells[env], env, m)
            if not (o and recs):
                continue
            ours = [value_at(r_, 1.9e6, 2.0e6) for r_ in recs] if env == "heavenhell_3" else [r_["final"] for r_ in recs]
            om, ose = mean_se(ours)
            ex["official_vs_ours"][f"{env}/{m}@2M"] = dict(
                official=o, ours_mean=om, ours_se=ose, ours_n=len(ours), ours_per_run=ours, window=[1.9e6, 2.0e6],
                welch_p=float(sst.ttest_ind(ours, o["per_run"], equal_var=False).pvalue))
    # authors' code HH-3 asym-s to 4.5M (5 seeds, CPU; scripts/official/run_official.sh, logs
    # long_heavenhell3_asym-a2c-state_s<n>.log) vs ours at the same steps
    lc = official_curves("long_heavenhell3_asym-a2c-state_s*.log")
    recs = load_cell("full", "heavenhell_3", "a2c_asym_s")
    if lc and recs:
        tc = float(min(r_["curve"]["t"][-1] for r_ in recs))
        W = window_steps("full", "heavenhell_3", "a2c_asym_s")
        long = {}
        for lab, (t0, t1) in (("1M", (0.9e6, 1.0e6)), ("2M", (1.9e6, 2.0e6)), ("3M", (2.9e6, 3.0e6)),
                              ("4M", (3.9e6, 4.0e6)), ("ours_common_window", (tc - W, tc))):
            oo = [value_at(dict(curve=c), t0, t1) for c in lc.values() if c["t_last"] >= t1]
            ours = [value_at(r_, t0, t1) for r_ in recs]
            if len(oo) < 2:
                continue
            pm, ph = paper_window("heavenhell_3", "a2c_asym_s", t0, t1)
            long[lab] = dict(window=[t0, t1], official_per_run=oo, official=mean_se(oo), ours_per_run=ours,
                             ours=mean_se(ours), paper=pm, paper_sem_half=ph,
                             welch_p=float(sst.ttest_ind(ours, oo, equal_var=False).pvalue),
                             official_runs_below_0p05=int(np.sum(np.asarray(oo) < 0.05)),
                             ours_runs_below_0p05=int(np.sum(np.asarray(ours) < 0.05)))
        ex["official_long_hh3_asym_s"] = dict(runs=sorted(lc), t_last=[c["t_last"] for c in lc.values()], at=long)
    # Figure 4 structure at the full-scale STOP (HH-4 runs stopped at 3.3-6.1M; the paper's Fig. 4 spans 0-10M, so this is
    # NOT the paper's 10M figure): s: (a)=(c), (b)=(d); h: (a)=(b); hs: priest-visited above not-visited
    if "heavenhell_4" in S:
        fig4 = {}
        for m in ("a2c_asym_hs", "a2c_asym_s", "a2c"):
            recs = load_cell("full", "heavenhell_4", m)
            if not recs:
                continue
            pr = np.array([np.asarray(r_["curve"]["probe"])[-10:].mean(0) for r_ in recs])
            fig4[m] = dict(mean=pr.mean(0).tolist(), se=(pr.std(0, ddof=1) / np.sqrt(len(pr))).tolist(),
                           per_run=pr.tolist(), t_stop_per_run=[r_["curve"]["t"][-1] for r_ in recs],
                           a_eq_c=float(np.max(np.abs(pr[:, 0] - pr[:, 2]))), b_eq_d=float(np.max(np.abs(pr[:, 1] - pr[:, 3]))),
                           a_eq_b=float(np.max(np.abs(pr[:, 0] - pr[:, 1]))),
                           priest_minus_nopriest=_safe_paired((pr[:, 2] + pr[:, 3]) / 2, (pr[:, 0] + pr[:, 1]) / 2))
        ex["fig4_hh4_at_stop"] = fig4
        # Fig. 4 extracted from the PDF vectors (gridline-calibrated), compared at the SAME
        # step: each arm's common step (t_c - W, t_c], t_c = the arm's smallest stop step, W = 10 log intervals
        p4 = json.load(open(PAPER_FIG4))
        fig4m = {}
        for m in ("a2c_asym_hs", "a2c_asym_s", "a2c"):
            recs = load_cell("full", "heavenhell_4", m)
            if not recs:
                continue
            tc = float(min(r_["curve"]["t"][-1] for r_ in recs))
            W = window_steps("full", "heavenhell_4", m)

            def probe_at(r_, j):
                t_, pr_ = np.asarray(r_["curve"]["t"]), np.asarray(r_["curve"]["probe"])
                sel = (t_ > tc - W) & (t_ <= tc)
                return float(pr_[sel, j].mean())
            ours = np.array([[probe_at(r_, j) for j in range(4)] for r_ in recs])
            rows = {}
            for j, k in enumerate("abcd"):
                pm, ph = curve_window(p4[k][m], tc - W, tc)
                om, ose = mean_se(ours[:, j].tolist())
                rows[p4["_names"][k]] = dict(ours=om, se=ose, ours_per_run=ours[:, j].tolist(), paper=pm,
                                             paper_sem_half=ph, judgment=judge(om, pm, PREC_EXACT["heavenhell_4"], ph or 0.0),
                                             z=float((om - pm) / np.hypot(ose or 0.0, ph or 0.0)) if (ose or ph) else None)
            pp = [curve_window(p4[k][m], tc - W, tc)[0] for k in "abcd"]
            p10 = [curve_window(p4[k][m], 9.5e6, 10e6)[0] for k in "abcd"]
            gap = (ours[:, 2] + ours[:, 3]) / 2 - (ours[:, 0] + ours[:, 1]) / 2
            fig4m[m] = dict(window=[tc - W, tc], probes=rows, paper_at_10M=p10,
                            priest_minus_nopriest=dict(ours_per_run=gap.tolist(), ours=mean_se(gap.tolist()),
                                                       paper=(pp[2] + pp[3]) / 2 - (pp[0] + pp[1]) / 2))
        ex["fig4_hh4_matched_step"] = fig4m
    # small-scale vs full-scale HH-3 (same key tree): avg100 near 1M
    g2, g3 = load_cell("small", "heavenhell_3", "a2c_asym_hs"), load_cell("full", "heavenhell_3", "a2c_asym_hs")
    if g2 and g3:
        ex["hh3_small_vs_full_at_1M"] = dict(small=[value_at(r, 0.99e6, 1.0e6) for r in g2],
                                             full=[value_at(r, 0.95e6, 1.0e6) for r in g3])
    # speed (asym-hs time-to-0.9 on HH-3; Shopping time-to-reach -5): first step at which the
    # MEAN curve (ours: 5 runs on a common grid; paper: vector-extracted mean curve) reaches the threshold
    pcur = json.load(open(PAPER_CURVES))
    speed = {}
    for env, thr, T in (("heavenhell_3", 0.9, 2.1e6), ("heavenhell_3", 0.5, 2.1e6), ("shopping_5", -5, 2e6),
                        ("shopping_6", -5, 2e6)):
        for m in METHODS:
            recs = load_cell(cells[env], env, m)
            if not recs or m not in pcur.get(env, {}):
                continue
            g = np.linspace(0, min(T, min(r["curve"]["t"][-1] for r in recs)), 400)
            mc = np.mean([np.interp(g, r["curve"]["t"], r["curve"]["avg100"]) for r in recs], 0)
            ours = first_crossing(g, mc, thr)
            paper = first_crossing(np.asarray(pcur[env][m]["t"]), pcur[env][m]["mean"], thr)
            per = [first_crossing(r["curve"]["t"][1:], r["curve"]["avg100"][1:], thr) for r in recs]  # skip t=0 point
            if np.isnan(ours) and np.isnan(paper):
                continue
            rel = (ours - paper) / paper * 100 if np.isfinite(ours) and np.isfinite(paper) else None
            jd = ("MATCH" if abs(rel) <= 5 else "CLOSE" if abs(rel) <= 20 else "DIVERGENT") if rel is not None else "n/a"
            # the authors' code (unmodified main_a2c.py, Table-2 settings, CPU), whose
            # x-axis is simulation_timesteps exactly as ours; same mean-curve + crossing estimator as ours
            oc = official_curves(OFFICIAL_SPEED[(env, m)]) if (env, m) in OFFICIAL_SPEED else {}
            off = dict(official_mean_curve=None, official_n=0, official_per_run=None, delta_vs_official_pct=None)
            if oc:
                go = np.linspace(0, min(T, min(c["t_last"] for c in oc.values())), 400)
                omc = np.mean([np.interp(go, c["t"], c["avg100"]) for c in oc.values()], 0)
                ocr = first_crossing(go, omc, thr)
                off = dict(official_mean_curve=ocr, official_n=len(oc),
                           official_per_run=[first_crossing(c["t"][1:], c["avg100"][1:], thr) for c in oc.values()],
                           delta_vs_official_pct=(ours - ocr) / ocr * 100 if np.isfinite(ours) and np.isfinite(ocr) else None,
                           paper_vs_official_pct=(paper - ocr) / ocr * 100 if np.isfinite(paper) and np.isfinite(ocr) else None)
            speed[f"{env}/{m}/thr{thr}"] = dict(ours_mean_curve=ours, paper_mean_curve=paper, per_run=per,
                                                delta_pct=rel, judgment=jd, **off)
    rep["speed"] = speed
    # Ablation A1 (pre-declared): Shopping-5 asym-s with MC advantages (lambda = 1) vs
    # the small-scale TD(0) asym-s runs (same run_idx keys), Holm over {final, auc}
    mc = load_cell("ablation", "shopping_5", "a2c_asym_s")
    if len(mc) == 5 and all(r["t_env_final"] >= 2_000_000 for r in mc):
        td = load_cell("small", "shopping_5", "a2c_asym_s")
        hs = load_cell("small", "shopping_5", "a2c_asym_hs")
        A1 = {"Shopping-5 asym_s MC - asym_s TD0 final": paired_compare([r["final"] for r in mc], [r["final"] for r in td]),
              "Shopping-5 asym_s MC - asym_s TD0 auc": paired_compare([r["auc"] for r in mc], [r["auc"] for r in td])}
        adj = holm({k: v["t_p"] for k, v in A1.items()})
        for k in A1:
            A1[k]["p_holm"] = adj[k]["p_holm"]
        rep["A1"] = A1
        rep["exploratory"]["A1_mc_vs_hs_td0_final"] = paired_compare([r["final"] for r in mc], [r["final"] for r in hs])
        rep["exploratory"]["A1_cells"] = dict(mc_final=[r["final"] for r in mc], mc_auc=[r["auc"] for r in mc],
                                              td_final=[r["final"] for r in td], td_auc=[r["auc"] for r in td],
                                              mc_t_ge_m5=[first_crossing(r["curve"]["t"], r["curve"]["avg100"], -5) for r in mc],
                                              td_t_ge_m5=[first_crossing(r["curve"]["t"], r["curve"]["avg100"], -5) for r in td])
    return rep, S


SA = "recheck"   # recheck re-runs


def _cellsum(vals):
    v = [x for x in vals if np.isfinite(x)]
    m, se = mean_se(v) if len(v) > 1 else ((v[0], None) if v else (float("nan"), None))
    return dict(mean=m, se=se, n=len(v), per_run=list(vals), solved_gt_0p5=int(np.sum(np.asarray(v) > 0.5)))


def _welch(a, b):
    from scipy import stats as sst
    a, b = [x for x in a if np.isfinite(x)], [x for x in b if np.isfinite(x)]
    return float(sst.ttest_ind(a, b, equal_var=False).pvalue) if len(a) > 1 and len(b) > 1 else None


def recheck_report():
    """Recheck re-runs (all CPU, full-f32 matmuls the matmul-precision fix), reduced at fixed, pre-declared step windows.
    * HH-4 asym-hs fresh seed 2, n = 10 (20 planned)  vs paper / TF32 learner-strength probe (seed 1, n = 20, GPU) /
      authors' code (n = 5).
    * full-scale key trees (seed 0, n = 5) re-trained in f32: HH-4 asym-hs and HH-3 asym-s, vs the stored full-scale runs.
    * HH-3 asym-s fresh seed 1, n = 10 (20 planned)  vs paper / authors' code (4.5M, n = 5).
    * authors' code Shopping-5 react-4 seeds 0-2 vs ours (small-scale) and the paper.
    * HH-3 asym-s seed 0 with lambda_final_factor 0.01 (code-history sensitivity, diagnostic) vs the f32 HH-3 re-run."""
    out = {}
    g3 = json.load(open(os.path.join(REPO, "runs", "full", "summary.json")))["report"]["judgments"]
    c4 = tuple(g3["heavenhell_4/a2c_asym_hs"]["window"])     # the full-scale common windows, exactly 
    c3 = tuple(g3["heavenhell_3/a2c_asym_s"]["window"])
    hh = {
        "heavenhell_4": dict(method="a2c_asym_hs", windows=[(0.5e6, 1.0e6), (1.5e6, 2.0e6), (2.31e6, 2.81e6),
                                                            (2.5e6, 3.0e6), c4], common=c4,
                             sources=dict(f32_fresh=(f"{SA}/hh4_hs_fp32_s2",),
                                          tf32_probe_n20=("recheck/learner_strength/hh4_td0_s1",),
                                          f32_full_seeds_n5=(f"{SA}/hh4_hs_fp32_s0",),
                                          tf32_full_n5=("full",)),
                             official="recheck_heavenhell4_asym-a2c_s*.log", primary="f32_fresh"),
        "heavenhell_3": dict(method="a2c_asym_s", windows=[(0.9e6, 1.0e6), (1.9e6, 2.0e6), (2.9e6, 3.0e6),
                                                           c3, (3.9e6, 4.0e6)], common=c3,
                             sources=dict(f32_fresh=(f"{SA}/hh3_s_fp32_s1",),
                                          f32_full_seeds_n5=(f"{SA}/hh3_s_fp32_s0",),
                                          tf32_full_n5=("full",),
                                          f32_lamfinal001_full_seeds_n5=(f"{SA}/hh3_s_lamfin001_s0",)),
                             official="long_heavenhell3_asym-a2c-state_s*.log", primary="f32_fresh"),
    }
    for env, d in hh.items():
        m = d["method"]
        recs = {k: load_cell(v[0], env, m) for k, v in d["sources"].items()}
        recs = {k: v for k, v in recs.items() if v}
        oc = official_curves(d["official"])
        rows = {}
        for (t0, t1) in d["windows"]:
            pm, ph = paper_window(env, m, t0, t1)
            row = dict(window=[t0, t1], paper=pm, paper_sem_half=ph)
            vals = {}
            for k, rr in recs.items():
                v = [value_at(r_, t0, t1) if r_["curve"]["t"][-1] >= t1 - 1 else float("nan") for r_ in rr]
                vals[k] = v
                c = _cellsum(v)
                if c["n"] > 1:
                    c["judgment"] = judge(c["mean"], pm, PREC_EXACT[env], ph or 0.0)
                    c["z_vs_paper"] = float((c["mean"] - pm) / np.hypot(c["se"] or 0.0, ph or 0.0))
                    c["delta_pct"] = (c["mean"] - pm) / abs(pm) * 100 if abs(pm) >= 5 * PREC_EXACT[env] else None
                row[k] = c
            ov = [value_at(dict(curve=c), t0, t1) for c in oc.values() if c["t_last"] >= t1]
            row["authors_code"] = _cellsum(ov)
            vals["authors_code"] = ov
            row["welch"] = {f"{a} vs {b}": _welch(vals[a], vals[b]) for a, b in (
                ("f32_fresh", "tf32_probe_n20"), ("f32_fresh", "authors_code"),
                ("f32_full_seeds_n5", "tf32_full_n5"), ("f32_fresh", "tf32_full_n5"),
                ("f32_lamfinal001_full_seeds_n5", "f32_full_seeds_n5"))
                if a in vals and b in vals}
            # TF32 question on pooled runs, f32 (fresh + full-scale keys) vs TF32
            f32p = [x for k in ("f32_fresh", "f32_full_seeds_n5") if k in vals for x in vals[k]]
            tfp = [x for k in ("tf32_probe_n20", "tf32_full_n5") if k in vals for x in vals[k]]
            row["pooled_f32"], row["pooled_tf32"] = _cellsum(f32p), _cellsum(tfp)
            row["welch"]["pooled f32 vs pooled tf32"] = _welch(f32p, tfp)
            for a, b in (("f32_full_seeds_n5", "tf32_full_n5"), ("f32_lamfinal001_full_seeds_n5", "f32_full_seeds_n5")):
                if a in vals and b in vals and all(np.isfinite(vals[a] + vals[b])) and len(vals[a]) == len(vals[b]):
                    row[f"paired {a} - {b}"] = _safe_paired(vals[a], vals[b])
            rows[f"({t0 / 1e6:.2f},{t1 / 1e6:.2f}]M"] = row
        # stuck runs: never leave the random policy (mean avg100 over [2M, end of the common window] < 0.05)
        stuck = {k: int(np.sum([value_at(r_, 2.0e6, d["common"][1]) < 0.05 for r_ in rr])) for k, rr in recs.items()}
        stuck["authors_code"] = int(np.sum([value_at(dict(curve=c), 2.0e6, d["common"][1]) < 0.05 for c in oc.values()
                                            if c["t_last"] >= d["common"][1]]))
        out[f"{env}/{m}"] = dict(windows=rows, stuck_runs=stuck, n={k: len(v) for k, v in recs.items()},
                                 n_authors_code=len(oc), common_window=list(d["common"]), primary=d["primary"],
                                 t_last_min={k: float(min(r_["curve"]["t"][-1] for r_ in rr)) for k, rr in recs.items()})
    # HH-3 asym-s stuck-at-uniform fraction, ours vs the authors' code. Stuck = mean avg100
    # over (1.5, 2.0]M < 0.05 (same classification as the [2M, 4.16M] window on all 30 of our runs, checked beforehand)
    from scipy import stats as sst
    ours = [r_ for tag in (f"{SA}/hh3_s_fp32_s1", f"{SA}/hh3_s_fp32_s0", "full")
            for r_ in load_cell(tag, "heavenhell_3", "a2c_asym_s")]
    offc = {}
    for pat in ("stuck_heavenhell3_asym-a2c-state_s*.log", "long_heavenhell3_asym-a2c-state_s*.log",
                "heavenhell3_asym-a2c-state_s*.log"):
        offc.update({k: c for k, c in official_curves(pat).items() if c["t_last"] >= 1.9e6})   # 200 logs: last ≈ 1.99M
    if ours and offc:
        ov = lambda c, a, b: value_at(dict(curve=c), a, b)  # noqa: E731
        o_st = [value_at(r_, 1.5e6, 2.0e6) < 0.05 for r_ in ours]
        a_st = [ov(c, 1.5e6, 2.0e6) < 0.05 for c in offc.values()]
        o2, a2 = [value_at(r_, 1.9e6, 2.0e6) for r_ in ours], [ov(c, 1.9e6, 2.0e6) for c in offc.values()]
        pm, ph = paper_window("heavenhell_3", "a2c_asym_s", 1.9e6, 2.0e6)
        out["heavenhell_3/a2c_asym_s/stuck_rate"] = dict(
            criterion="mean avg100 over (1.5, 2.0]M < 0.05", ours_stuck=int(sum(o_st)), ours_n=len(o_st),
            authors_stuck=int(sum(a_st)), authors_n=len(a_st), authors_runs=sorted(offc),
            fisher_p=float(sst.fisher_exact([[sum(o_st), len(o_st) - sum(o_st)], [sum(a_st), len(a_st) - sum(a_st)]]).pvalue),
            ours_1p9_2p0=_cellsum(o2), authors_1p9_2p0=_cellsum(a2), welch_1p9_2p0=_welch(o2, a2), paper_1p9_2p0=pm,
            paper_sem_half=ph,
            ours_nonstuck_1p9_2p0=_cellsum([x for x, st in zip(o2, o_st) if not st]),
            authors_nonstuck_1p9_2p0=_cellsum([x for x, st in zip(a2, a_st) if not st]))
    # Shopping-5 react-4 finals (last 10 of 200 log points): authors' code vs ours (small-scale) vs paper
    oc = official_curves("recheck_shopping5_react4_s*.log")
    ours = load_cell("small", "shopping_5", "a2c_react_4")
    if oc and ours:
        of = [c["last10"] for c in oc.values() if c["t_last"] >= 1.99e6]
        w0 = float(np.mean([r_["curve"]["t"][-10] for r_ in ours]))
        pm, ph = paper_window("shopping_5", "a2c_react_4", w0, 2.0e6 + 1e9)
        out["shopping_5/a2c_react_4"] = dict(authors_code=_cellsum(of), ours=_cellsum([r_["final"] for r_ in ours]),
                                             paper=pm, paper_sem_half=ph, welch_ours_vs_authors=_welch([r_["final"] for r_ in ours], of),
                                             authors_judgment_vs_paper=judge(float(np.mean(of)), pm, PREC_EXACT["shopping_5"], ph or 0.0) if of else None,
                                             runs=sorted(oc))
    return out


def recheck_markdown(rep) -> str:
    L = []
    f = lambda c: "—" if not c or not c.get("n") else (f"{c['mean']:+.3f} ± {c['se']:.3f} (n={c['n']}, {c['solved_gt_0p5']} > 0.5)"  # noqa: E731
                                                       if c.get("se") is not None else f"{c['mean']:+.3f} (n=1)")
    for cell, v in rep.items():
        if "windows" not in v:
            continue
        keys = [k for k in ("f32_fresh", "tf32_probe_n20", "f32_full_seeds_n5", "tf32_full_n5",
                            "f32_lamfinal001_full_seeds_n5") if k in v["n"]] + ["pooled_f32", "pooled_tf32"]
        L += [f"\n{cell}: stuck runs (mean avg100 over [2M, {v['common_window'][1] / 1e6:.2f}M] < 0.05): {v['stuck_runs']}",
              "| window | paper | authors' code | " + " | ".join(keys) + " | judgment (primary) | Welch p |",
              "|---|---|---|" + "---|" * len(keys) + "---|---|"]
        for w, r in v["windows"].items():
            pr = r.get(v["primary"], {})
            L.append(f"| {w} | {r['paper']:+.3f} ± {r['paper_sem_half'] or 0:.3f} | {f(r['authors_code'])} | "
                     + " | ".join(f(r.get(k)) for k in keys) + f" | {pr.get('judgment', '—')} (z {pr.get('z_vs_paper', float('nan')):+.2f}) | "
                     + "; ".join(f"{k}: {p:.2g}" for k, p in r["welch"].items() if p is not None) + " |")
    return "\n".join(L)


ENV_NAME = {"heavenhell_3": "HH-3", "heavenhell_4": "HH-4", "shopping_5": "Shopping-5", "shopping_6": "Shopping-6",
            "car_flag": "Car-Flag", "cleaner": "Cleaner"}
def _ms(x, se):
    return f"{x:+.3f} ± {se or 0:.3f}"


def full_markdown(rep) -> str:
    L = ["| Metric | Paper (20 runs, mean ± s.e.m.) | Ours (mean ± SE, n) | Δ% | Judgment | Informative? | Notes |",
         "|---|---|---|---|---|---|---|"]
    off = rep["exploratory"].get("official_vs_ours", {})
    for k, j in rep["judgments"].items():
        env, m = k.split("/")
        w0, w1 = j["window"]
        when = (f"avg100 at the COMMON step t∈({w0 / 1e6:.2f}M, {w1 / 1e6:.2f}M] (stopped early, budget)" if j["common_step"]
                else f"avg100 over t∈[{w0 / 1e6:.2f}M, {w1 / 1e6:.2f}M] (final, T)")
        paper = f"{j['paper']:+.3f}" + (f" ± {j['paper_sem_half']:.3f}" if j["paper_sem_half"] is not None else "")
        d = f"{j['delta_pct']:+.0f}%" if j["delta_pct"] is not None else "abs rule"
        inf = "yes" if j["informative"] else f"NO (start-of-training {j['start_of_training_avg100']:+.3f} also MATCHes)"
        note = [f"z vs paper {j['z_vs_paper']:+.2f}" if j["z_vs_paper"] is not None else "",
                f"paper @T: {j['paper_final_T']:+g}", f"data: {j['data_tag']}"]
        o = off.get(f"{env}/{m}@2M")
        if o:   # built from the computed comparison, never a literal
            note.append(f"@2M t∈(1.9,2.0]M: authors' code {_ms(o['official']['mean'], o['official']['se'])} "
                        f"(n={o['official']['n']}) vs ours {_ms(o['ours_mean'], o['ours_se'])} (n={o['ours_n']}), "
                        f"Welch p {o['welch_p']:.2f} (no difference detected; underpowered, not equivalence)")
        L.append(f"| {ENV_NAME[env]} {m} {when} | {paper} | {_ms(j['ours'], j['se'])} ({j['n']}) | {d} | "
                 f"{j['judgment']} | {inf} | {'; '.join(x for x in note if x)} |")
    return "\n".join(L)


def sensitivity_markdown(sens: dict) -> str:
    L = ["| Cell | t_stop per run (M) | Spearman(t_stop, final) | (A) per-run finals vs paper over mean window | "
         "(B) per-run finals vs paper over each run's window | (C) common step (primary) |", "|---|---|---|---|---|---|"]
    for k, v in sens.items():
        env, m = k.split("/")
        f = lambda d: f"{d['ours']:+.3f} vs {d['paper']:+.3f}: {d['judgment']}"  # noqa: E731
        c = v["C_common_step"]
        L.append(f"| {ENV_NAME[env]} {m} | {', '.join(f'{t / 1e6:.2f}' for t in v['t_stop_per_run'])} | "
                 f"{v['spearman_tstop_final']:+.2f} | {f(v['A_mean_window'])} | {f(v['B_per_run_window'])} | "
                 f"({c['window'][0] / 1e6:.2f}, {c['window'][1] / 1e6:.2f}]M {f(c)} |")
    return "\n".join(L)


def speed_markdown(sp: dict) -> str:
    L = ["| Metric (first t where the mean curve reaches thr) | Paper | Authors' code (mean curve, n) | Ours (mean curve; per-run) "
         "| Δ% vs paper | Judgment | Δ% vs authors' code |", "|---|---|---|---|---|---|---|"]
    for k, v in sp.items():
        env, m, thr = k.split("/")
        f = lambda x: "never" if x is None or not np.isfinite(x) else f"{x / 1e6:.3f}M"  # noqa: E731
        per = ", ".join(f(x) for x in v["per_run"])
        d = f"{v['delta_pct']:+.1f}%" if v["delta_pct"] is not None else "—"
        o = f"{f(v['official_mean_curve'])} (n={v['official_n']})" if v.get("official_n") else "—"
        do = f"{v['delta_vs_official_pct']:+.1f}%" if v.get("delta_vs_official_pct") is not None else "—"
        L.append(f"| {ENV_NAME[env]} {m} {thr} | {f(v['paper_mean_curve'])} | {o} | {f(v['ours_mean_curve'])} ({per}) | {d} | "
                 f"{v['judgment']} | {do} |")
    return "\n".join(L)


def tests_markdown(fam: dict) -> str:
    L = ["| Test | mean diff | paired-t 95% CI | bootstrap 95% CI | wins/n | p (paired t) | p Holm | p (Welch) | p Holm (Welch) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, v in fam.items():
        L.append(f"| {k} | {v['diff_mean']:+.3f} | [{v['t95'][0]:+.3f}, {v['t95'][1]:+.3f}] | "
                 f"[{v['boot95'][0]:+.3f}, {v['boot95'][1]:+.3f}] | {v['wins']}/{v['n']} | {v['t_p']:.2g} | {v['p_holm']:.2g} | "
                 + (f"{v['welch_p']:.2g} | {v['p_holm_welch']:.2g} |" if "p_holm_welch" in v else "— | — (CRN-paired) |"))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="small")
    a = ap.parse_args()
    mirror_per_seed(a.tag)
    if a.tag == "small":
        rep, s5, h3 = small_report()
        out = os.path.join(REPO, "runs", "small", "summary.json")
        with open(out, "w") as f:
            json.dump(dict(report=rep, curves=dict(shopping_5={m: {k: r[k] for k in ("curve_grid", "curve_mean", "curve_se")}
                                                               for m, r in s5.items()},
                                                   heavenhell_3={m: {k: r[k] for k in ("curve_grid", "curve_mean", "curve_se")}
                                                                 for m, r in h3.items()})), f, indent=1)
        print(small_markdown(rep))
        for k, v in rep["judgments"].items():
            print(f"[METRIC] {k}: ours={v['ours']:.3f}+-{v['se'] or 0:.3f} paper={v['paper']} -> {v['judgment']}")
        print(f"[INFO] wrote {out}")
    if a.tag == "ablation":
        pass
    if a.tag == "recheck":
        rep = recheck_report()
        out = os.path.join(REPO, "runs", SA, "summary.json")
        with open(out, "w") as f:
            json.dump(rep, f, indent=1)
        md = recheck_markdown(rep)
        with open(os.path.join(REPO, "runs", SA, "tables.md"), "w") as f:
            f.write(md + "\n")
        print(md)
        print(f"[INFO] wrote {out}")
    if a.tag == "full":
        rep, S = full_report()
        out = os.path.join(REPO, "runs", "full", "summary.json")
        with open(out, "w") as f:
            json.dump(dict(report=rep, curves={e: {m: {k: r[k] for k in ("curve_grid", "curve_mean", "curve_se")}
                                                   for m, r in s.items()} for e, s in S.items()}), f, indent=1)
        for k, v in rep["judgments"].items():
            print(f"[METRIC] {k}: ours={v['ours']:.3f}+-{v['se'] or 0:.3f} (n={v['n']}) paper={v['paper']} -> {v['judgment']}")
        for fam in ("primary_tests", "secondary_tests"):
            for k, v in rep[fam].items():
                print(f"[METRIC] {fam} {k}: diff={v['diff_mean']:+.4f} CI={v.get('boot95')} p={v['t_p']:.3g} "
                      f"p_holm={v['p_holm']:.3g}")
        with open(os.path.join(REPO, "runs", "full", "tables.md"), "w") as f:
            f.write(full_markdown(rep) + "\n\nSENSITIVITY (truncated cells)\n" + sensitivity_markdown(rep["sensitivity"]) + "\n\nSPEED\n" + speed_markdown(rep["speed"]) + "\n\nPRIMARY TESTS\n" + tests_markdown(rep["primary_tests"]) + "\n\nSECONDARY TESTS\n" + tests_markdown(rep["secondary_tests"])
                    + ("\n\nA1\n" + tests_markdown(rep["A1"]) if rep.get("A1") else "") + "\n")
        print(f"[INFO] wrote {out}")


if __name__ == "__main__":
    main()

"""full run, Atari 200M (E3) completion, DM Control studentized rows (E4) and the paper's qualitative ALE claims, on the
public Dopamine 5-run data, beside the authors'-code numbers (data/official/reference_ale.npz, rliable@3ccd9f4 run in
its own venv) and the paper's values read EXACTLY from the PDF's vector drawings (data/paper_readoff/readoff.json,
scripts/paper/vector_readoff.py). Writes runs/full/ale.json.

    JAX_PLATFORMS=cpu python scripts/full_ale.py [--out ...]

Key tree: branch_key(0, EXP_IDS['e3'], alg, STAT) with STAT ids below (logged in the JSON); E4 has no RNG.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO, "external", "data"))  # large inputs, see README
sys.path.insert(0, REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from rl_jax_bench.jaxlog import Timer, check_device, info, metric, write_json  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
from src import eval as E  # noqa: E402

STAT = dict(fig9=1, a20=2, a28=3, a27=4, profiles=5, fig10=6, q1_diff=7, sub_m1=8, sub_m2=9)
ALGS = ["DQN (Nature)", "DQN (Adam)", "C51", "IQN", "Rainbow"]
A27_ALIAS = {"D41 (1Dture)": "DQN (Nature)", "D41 (AdDP)": "DQN (Adam)", "C51": "C51", "5(0": "REM", "I41": "IQN",
             "5DiQERw": "Rainbow", "0-I41": "M-IQN", "DreDPerV2": "DreamerV2"}
A27_PANELS = ["DQN (Nature)", "DQN (Adam)", "C51", "REM", "IQN", "Rainbow", "M-IQN", "DreamerV2"]
A28_ROWS = ["DreamerV2", "M-IQN", "IQN", "Rainbow", "REM", "C51", "DQN (Adam)", "DQN (Nature)"]  # top -> bottom


DQN_AGENTS = ("DQN (Nature)", "DQN (Adam)")


def judge(ours, paper, spread=None, tol=0.0):
    """Shared-context rule: MATCH within 5 % relative or inside the paper's drawn 95 % CI (closed, widened by the read
    tolerance tol = the panel's gridline-fit residual; M2); CLOSE <= 20 %; else DIVERGENT.
    Returns (verdict, rel, reason)."""
    rel = abs(ours - paper) / max(abs(paper), 1e-12)
    if rel <= 0.05:
        return "MATCH", rel, "<=5%"
    if spread is not None and spread[0] - tol <= ours <= spread[1] + tol:
        return "MATCH", rel, "inside paper CI"
    return ("CLOSE" if rel <= 0.20 else "DIVERGENT"), rel, ""


def provenance(label):
    """Rows that involve DQN (Nature) or DQN (Adam) are DATA DIVERGENT -- the public Dopamine mirror
    demonstrably differs from the paper's input for those agents (4 / 7 point statistics of each outside the paper's
    own 95 % CI while the estimator equals the authors' code; runs/full/data_provenance.json), so they are not
    estimator results. Returns 'data_divergent' or 'estimator'."""
    return "data_divergent" if any(a in label for a in DQN_AGENTS) else "estimator"


def readoff():
    R = json.load(open(os.path.join(REPO, "data", "paper_readoff", "readoff.json")))
    out = {}
    for p in R["fig9"]:
        nm = {"Median": "median", "IQM": "iqm", "Mean": "mean", "Optimality Gap": "og"}[p["title"]]
        for r in p["rows"]:
            out.setdefault("fig9", {}).setdefault(r["label"], {})[nm] = r
    for p in R["a20"]:
        nm = {"Median": "median", "IQM": "iqm", "Mean": "mean", "Optimality Gap": "og"}[p["title"]]
        for r in p["rows"]:
            out.setdefault("a20", {}).setdefault(r["label"], {})[nm] = r
    for p in R["a28"]:
        nm = {"Difficulty Progress (25%)": "dp25", "Difficulty Progress (50%)": "dp50",
              "Superhuman Probability": "superhuman"}[p["title"]]
        rows = sorted(p["rows"], key=lambda r: r["y"])
        assert len(rows) == 8
        for lab, r in zip(A28_ROWS, rows):   # the bottom label is mis-assigned by the y match; order is unambiguous
            out.setdefault("a28", {}).setdefault(lab, {})[nm] = r
    for x, p in zip(A27_PANELS, sorted(R["a27"], key=lambda q: (round(q["axes"][1]), q["axes"][0]))):
        for r in p["rows"]:
            out.setdefault("a27", {})[f"{x} > {A27_ALIAS[r['label']]}"] = r
    fig11 = [p for p in R["fig11a"] if p["title"] and "steps" in p["title"] and len(p["rows"]) == 9]
    for p in fig11:
        for r in p["rows"]:
            out.setdefault("fig11a", {}).setdefault(p["title"].split()[0], {})[r["label"]] = r
    return out


def row(ours, ci, paper_r, oracle=None, oracle_ci=None, label=""):
    pt = paper_r["point"]
    v, rel, why = judge(ours, pt, (paper_r["lo"], paper_r["hi"]), paper_r.get("tol", 0.0))
    return dict(paper=pt, paper_ci=[paper_r["lo"], paper_r["hi"]], paper_ci_at_axes_edge=[paper_r["at_edge_lo"], paper_r["at_edge_hi"]],
                read_tol=paper_r.get("tol", 0.0), ours=ours, ours_ci=ci, authors=oracle, authors_ci=oracle_ci, rel=rel,
                verdict=v, reason=why, provenance=provenance(label))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "full", "ale.json"))
    args = ap.parse_args()
    check_device()
    T = Timer()
    src = D.sources()
    B_AGG = src["bootstrap"]["reps_aggregate"]["value"]
    B_PROF = src["bootstrap"]["reps_profile"]["value"]
    B_POI = src["bootstrap"]["reps_poi"]["value"]
    key = A.branch_key(0, A.EXP_IDS["e3"])
    O = np.load(os.path.join(DATA_ROOT, "official", "reference_ale.npz"))
    oalg = list(O["algs"])
    P = readoff()
    final, games, _ = D.load_dopamine_ale(final_only=True, fetch=False)
    cube, _, _ = D.load_dopamine_ale(final_only=False, fetch=False)
    for a in ALGS:  # the inputs are the authors' normalised matrices bit for bit (the official-code check)
        assert np.array_equal(final[a], O[f"norm_{oalg.index(a)}"]), a
    out = dict(stat_ids=STAT, exp=A.EXP_IDS["e3"], master_seed=0, reps=dict(agg=B_AGG, profile=B_PROF, poi=B_POI),
               paper_source="PDF vector drawings (scripts/paper/vector_readoff.py); fit residual <= 1.2e-4 data units")

    # ---------------------------------------------------------------- Fig 9 and A.20 (task + run bootstrap)
    for name, tb, sid in (("fig9", False, STAT["fig9"]), ("a20", True, STAT["a20"])):
        tab = E.aggregate_table(jax.random.fold_in(key, sid), final, B_AGG, task_bootstrap=tb)
        rows = {}
        for a in ALGS:
            j = oalg.index(a)
            ok = O[f"fig9_s0_{j}_ci"] if name == "fig9" else O[f"a20_{j}_ci"]
            for i, s in enumerate(A.AGG_NAMES):
                pt, lo, hi = tab[a][s]
                pr = P[name][a][s]
                r = row(pt, [lo, hi], pr, float(O[f"fig9_s0_{j}_pt"][i]), [float(ok[0, i]), float(ok[1, i])], label=a)
                if name == "a20":   # the point is Fig 9's; the A.20 claim is the INTERVAL: judge endpoints
                    r["lo_rel"] = abs(lo - pr["lo"]) / abs(pr["lo"])
                    r["hi_rel"] = abs(hi - pr["hi"]) / abs(pr["hi"])
                    r["width_ratio_ours_over_paper"] = (hi - lo) / (pr["hi"] - pr["lo"])
                    r["width_ratio_vs_fig9"] = None
                rows[f"{a}|{s}"] = r
        out[name] = rows
        metric(name, seconds=T.lap(), verdicts={v: sum(r["verdict"] == v for r in rows.values()) for v in ("MATCH", "CLOSE", "DIVERGENT")})
    for k, r in out["a20"].items():
        f9 = out["fig9"][k]
        r["width_ratio_vs_fig9"] = (r["ours_ci"][1] - r["ours_ci"][0]) / (f9["ours_ci"][1] - f9["ours_ci"][0])

    # ---------------------------------------------------------------- A.28 DP-25 / DP-50 / superhuman
    alt = lambda x: jnp.stack([A.difficulty_progress(x, 0.25), A.difficulty_progress(x, 0.5), A.superhuman_prob(x)])  # noqa: E731
    rows = {}
    for j0, a in enumerate(ALGS):
        b, ci, _ = A.interval_estimate(jax.random.fold_in(jax.random.fold_in(key, STAT["a28"]), j0), (jnp.asarray(final[a]),), alt, B_AGG)
        j = oalg.index(a)
        for i, s in enumerate(("dp25", "dp50", "superhuman")):
            rows[f"{a}|{s}"] = row(float(b[i]), [float(ci[0, i]), float(ci[1, i])], P["a28"][a][s],
                                   float(O[f"a25_{j}_pt"][i]), [float(O[f"a25_{j}_ci"][0, i]), float(O[f"a25_{j}_ci"][1, i])], label=a)
    out["a28"] = rows
    metric("a28", seconds=T.lap())

    # ---------------------------------------------------------------- A.27 P(X > Y), 20 ordered pairs
    pairs = [(x, y) for x in ALGS for y in ALGS if x != y]
    tab = E.poi_table(jax.random.fold_in(key, STAT["a27"]), final, pairs, B_POI)
    onames = list(O["poi_names"])
    rows = {}
    for x, y in pairs:
        k = f"{x} > {y}"
        pt, lo, hi = tab[k]
        ko = onames.index(f"{x}|{y}")
        rows[k] = row(pt, [lo, hi], P["a27"][k], float(O["poi_pt"][ko]), [float(v) for v in O["poi_ci"][ko]], label=k)
        rows[k]["abs_diff"] = abs(pt - P["a27"][k]["point"])
    out["a27"] = rows
    metric("a27", seconds=T.lap())

    # ---------------------------------------------------------------- Fig 10 profiles, IQM vs frames, A.24
    taus = jnp.linspace(0.0, 8.0, 81)
    prof = {}
    for j0, a in enumerate(ALGS):
        b, ci, _ = A.interval_estimate(jax.random.fold_in(jax.random.fold_in(key, STAT["profiles"]), j0),
                                       (jnp.asarray(final[a]),), lambda x: A.score_profile(x, taus), B_PROF)
        j = oalg.index(a)
        prof[a] = dict(point=np.asarray(b), ci=np.asarray(ci), max_abs_diff_points_vs_authors=float(np.abs(np.asarray(b) - O[f"prof_{j}"]).max()))
    d = prof["IQN"]["point"] - prof["Rainbow"]["point"]
    t = np.asarray(taus)
    sign_changes = [float(t[i + 1]) for i in range(len(t) - 1) if np.sign(d[i]) != np.sign(d[i + 1]) and d[i] != 0 and d[i + 1] != 0]
    ov = lambda a, b: float(np.mean((prof[a]["ci"][0] <= prof[b]["ci"][1]) & (prof[b]["ci"][0] <= prof[a]["ci"][1])))  # noqa: E731
    out["fig10"] = dict(
        iqn_minus_rainbow_mean_below_2=float(d[t < 2].mean()), iqn_minus_rainbow_mean_from_2=float(d[t >= 2].mean()),
        iqn_minus_rainbow_sign_changes_at_tau=sign_changes,
        band_overlap_fraction={"DQN (Adam) vs C51": ov("DQN (Adam)", "C51"), "IQN vs Rainbow": ov("IQN", "Rainbow"),
                               "DQN (Nature) vs DQN (Adam)": ov("DQN (Nature)", "DQN (Adam)")},
        profile_points_max_abs_diff_vs_authors=max(p["max_abs_diff_points_vs_authors"] for p in prof.values()))
    fr = np.asarray(O["fig10_frames_idx"])
    iqm_curve = {}
    for j0, a in enumerate(ALGS):
        x = jnp.asarray(cube[a][:, :, fr])
        b, ci, _ = A.interval_estimate(jax.random.fold_in(jax.random.fold_in(key, STAT["fig10"]), j0), (x,),
                                       lambda s: jax.vmap(lambda k: A.agg_iqm(s[:, :, k]))(jnp.arange(s.shape[2])), B_PROF)
        j = oalg.index(a)
        iqm_curve[a] = dict(frames_M=(fr + 1).tolist(), point=np.asarray(b).tolist(), ci=np.asarray(ci).tolist(),
                            authors_point=O[f"fig10_{j}_pt"].tolist(), authors_ci=O[f"fig10_{j}_ci"].tolist(),
                            max_abs_diff_points=float(np.abs(np.asarray(b) - O[f"fig10_{j}_pt"]).max()))
    out["fig10_iqm_curve"] = iqm_curve
    gam = np.geomspace(0.25, 4.0, 9)
    out["a24_og_over_gamma"] = {a: {f"{g:.3f}": float(A.agg_og(jnp.asarray(final[a]), float(g)) / g) for g in gam} for a in ALGS}
    metric("fig10", seconds=T.lap(), **{k: v for k, v in out["fig10"].items() if not isinstance(v, (dict, list))})

    # ---------------------------------------------------------------- qualitative claims Q1-Q3
    f9 = out["fig9"]
    ov_ci = lambda a, b, s: f9[f"{a}|{s}"]["ours_ci"][0] <= f9[f"{b}|{s}"]["ours_ci"][1] and f9[f"{b}|{s}"]["ours_ci"][0] <= f9[f"{a}|{s}"]["ours_ci"][1]  # noqa: E731
    q1 = {s: dict(ours_overlap=bool(ov_ci("DQN (Adam)", "C51", s)),
                  paper_overlap=bool(P["fig9"]["DQN (Adam)"][s]["lo"] <= P["fig9"]["C51"][s]["hi"] and P["fig9"]["C51"][s]["lo"] <= P["fig9"]["DQN (Adam)"][s]["hi"]))
          for s in A.AGG_NAMES}
    diff = lambda x, y: A.aggregates4(x) - A.aggregates4(y)  # noqa: E731
    b, ci, _ = A.interval_estimate(jax.random.fold_in(key, STAT["q1_diff"]), (jnp.asarray(final["C51"]), jnp.asarray(final["DQN (Adam)"])), diff, B_AGG)
    q1["C51_minus_DQN_Adam"] = {s: [float(b[i]), float(ci[0, i]), float(ci[1, i])] for i, s in enumerate(A.AGG_NAMES)}
    q1["P(C51 > DQN (Adam))"] = out["a27"]["C51 > DQN (Adam)"]["ours"], out["a27"]["C51 > DQN (Adam)"]["ours_ci"]
    q1["profile_band_overlap_fraction"] = out["fig10"]["band_overlap_fraction"]["DQN (Adam) vs C51"]
    order = lambda s, src_, rev: [a for a in sorted(ALGS, key=lambda a: src_(a, s), reverse=rev)]  # noqa: E731
    o_ours = lambda a, s: f9[f"{a}|{s}"]["ours"]  # noqa: E731
    o_pap = lambda a, s: P["fig9"][a][s]["point"]  # noqa: E731
    out["claims"] = dict(
        Q1_dqn_adam_vs_c51=q1,
        Q2_order=dict(ours={s: order(s, o_ours, s != "og") for s in A.AGG_NAMES},
                      paper={s: order(s, o_pap, s != "og") for s in A.AGG_NAMES}),
        Q4_iqn_rainbow_cross=out["fig10"]["iqn_minus_rainbow_sign_changes_at_tau"])
    q2 = out["claims"]["Q2_order"]
    out["claims"]["Q2_median_vs_iqm_order_differs"] = dict(ours=q2["ours"]["median"] != q2["ours"]["iqm"], paper=q2["paper"]["median"] != q2["paper"]["iqm"])
    out["claims"]["Q3_og_order_differs_from_iqm"] = dict(ours=q2["ours"]["og"] != q2["ours"]["iqm"], paper=q2["paper"]["og"] != q2["paper"]["iqm"])

    # ---------------------------------------------------------------- E1-sub on the ALE pool (N <= 5)
    sub = {}
    for j0, a in enumerate(ALGS):
        pool = jnp.asarray(final[a])
        k = jax.random.fold_in(jax.random.fold_in(key, STAT["sub_m2"]), j0)
        dm = np.asarray(A.sampling_distribution(k, pool, A.agg_median, 3, 100_000))[:, 0]
        di = np.asarray(A.sampling_distribution(jax.random.fold_in(k, 1), pool, A.agg_iqm, 3, 100_000))[:, 0]
        tm, ti = float(A.agg_median(pool)), float(A.agg_iqm(pool))
        sub[a] = dict(median_bias_N3=float(dm.mean() - tm), median_bias_se=float(dm.std() / np.sqrt(len(dm))),
                      iqm_bias_N3=float(di.mean() - ti), iqm_bias_se=float(di.std() / np.sqrt(len(di))),
                      median_sd_N3=float(dm.std()), iqm_sd_N3=float(di.std()),
                      median_ci_width_N5=f9[f"{a}|median"]["ours_ci"][1] - f9[f"{a}|median"]["ours_ci"][0],
                      iqm_ci_width_N5=f9[f"{a}|iqm"]["ours_ci"][1] - f9[f"{a}|iqm"]["ours_ci"][0])
    out["e1sub_ale"] = dict(per_alg=sub, reps=100_000, n=3, mode="per task with replacement from the 5-run pool (m-out-of-n, Q7)",
                            M1_iqm_narrower_all=all(v["iqm_ci_width_N5"] < v["median_ci_width_N5"] for v in sub.values()),
                            M2_iqm_less_biased_all=all(abs(v["iqm_bias_N3"]) < abs(v["median_bias_N3"]) for v in sub.values()),
                            median_bias_signs=[int(np.sign(v["median_bias_N3"])) for v in sub.values()])
    metric("e1sub_ale", seconds=T.lap(), M1=out["e1sub_ale"]["M1_iqm_narrower_all"], M2=out["e1sub_ale"]["M2_iqm_less_biased_all"])

    # ---------------------------------------------------------------- E4 DMC studentized rows (Fig 11a)
    n = src["dmc"]["studentized_n"]["value"]
    dmc = {}
    for steps in ("100k", "500k"):
        mu = src["normalisation"][f"dmc_reported_means_{steps}"]["value"]
        sd = src["normalisation"][f"dmc_reported_stds_{steps}"]["value"]
        for meth in ("CURL", "SUNRISE", "CURL-D2RL"):
            m, ci = A.studentized_ci_official(mu[meth], sd[meth], n[meth])
            m6, ci6, df6 = A.studentized_ci_d6(mu[meth], sd[meth], n[meth])
            r = row(m, ci, P["fig11a"][steps][meth], label=meth)
            r.update(d6_ci=ci6, d6_df=df6, n_runs=n[meth], lo_rel=abs(ci[0] - P["fig11a"][steps][meth]["lo"]) / P["fig11a"][steps][meth]["lo"],
                     hi_rel=abs(ci[1] - P["fig11a"][steps][meth]["hi"]) / P["fig11a"][steps][meth]["hi"],
                     authors="= ours (E0: studentized rows vs Colab mean_CI, max 2.1e-12)")
            dmc[f"{meth}|{steps}"] = r
    out["fig11a_studentized"] = dmc
    out["wall_seconds"] = T.total() if hasattr(T, "total") else None
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_json(args.out, out)
    info("done", out=args.out)


if __name__ == "__main__":
    main()

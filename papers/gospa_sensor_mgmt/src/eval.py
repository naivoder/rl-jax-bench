"""eval.py -- aggregation, grading and the pre-declared statistical family (README 'Results') (no jax import; numpy/scipy only).

Table III statistic G = (1/K) sum_k sqrt( (1/R) sum_m d^2_{k,m} )  (mean over steps of the per-step RMS; official utils.
calculate_average_algorithm_GOSPA_error; matches the digitized Fig. 5).  Secondary G_alt = sqrt(mean_{k,m} d^2) ([26] eq. 48).
Per-run scalar for paired tests: gbar_m = (1/K) sum_k d_{k,m} (mean GOSPA, not squared); the seed (= MC run m) is the unit.
Paired tests: paired t + paired bootstrap CI (rl_jax_bench.stats), Holm over the 33-test primary family, per configuration.
Grading of Table III cells (CFG-O): MATCH rel <= 5 %, CLOSE <= 20 %, DIVERGENT otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from src import model as M  # noqa: E402  (numpy only)
from rl_jax_bench.stats import paired_compare, holm, boot_ci, mean_se  # noqa: E402

DEFAULT_OUT = os.environ.get("GOSPA_RUNS", "runs")


def load_cell(root: str, tag: str, cfg_name: str, alg: str, runs: int):
    """-> {clutter(str): dict of arrays [R, K, ...]} for all runs 0..runs-1 (raises if a seed file/cell is missing)."""
    out = {}
    safe = cfg_name.replace("+", "_").replace("=", "-").replace(",", "_")
    for m in range(runs):
        shard = os.path.join(root, tag, f"seed_{m}", f"metrics__{safe}__{alg}.json")
        if os.path.exists(shard):
            with open(shard) as f:
                d = json.load(f)
        else:                                               # merged file written by scripts/merge_metrics.py
            with open(os.path.join(root, tag, f"seed_{m}", "metrics.json")) as f:
                d = json.load(f)[cfg_name][alg]
        for lam, cell in d.items():
            if lam == "_meta":
                continue
            o = out.setdefault(lam, {})
            for k, v in cell.items():
                if k == "counters":
                    o.setdefault("counters", []).append([v[n] for n in sorted(v) if n != "r_gt_1"])   # r_gt_1 is informational, kept apart
                    if "r_gt_1" in v:
                        o.setdefault("r_gt_1", []).append(v["r_gt_1"])
                elif k.startswith("mean_"):
                    o.setdefault(k, []).append(v)
                else:
                    o.setdefault(k, []).append(v)
    for lam, o in out.items():
        for k in list(o):
            o[k] = np.asarray(o[k], dtype=float)
    return out


def table3_stat(gsq):
    """gsq [R, K] -> (G, G_alt)."""
    gsq = np.asarray(gsq)
    return float(np.mean(np.sqrt(gsq.mean(axis=0)))), float(np.sqrt(gsq.mean()))


def run_scalar(gsq):
    """per-run mean GOSPA [R]."""
    return np.sqrt(np.asarray(gsq)).mean(axis=1)


def boot_G(gsq, n=10000, seed=0):
    """run-bootstrap 95 % CI of G (resample whole runs)."""
    gsq = np.asarray(gsq)
    rng = np.random.default_rng(seed)
    R = gsq.shape[0]
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, R, R)
        vals[i] = np.mean(np.sqrt(gsq[idx].mean(axis=0)))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def grade(ours, paper):
    rel = abs(ours - paper) / paper
    return rel, ("MATCH" if rel <= 0.05 else "CLOSE" if rel <= 0.20 else "DIVERGENT")


def counters_ok(cell):
    """invalidating counters (all but plan_nan_steps) must all be zero."""
    names = sorted(["birth_overflow", "meas_overflow", "est_overflow", "root_overflow", "plan_overflow", "nonfinite_filter",
                    "lbp_nonconverged", "plan_nan_steps"])
    c = cell["counters"]
    inval = {n: int(c[:, i].sum()) for i, n in enumerate(names) if n != "plan_nan_steps" and c[:, i].sum() > 0}
    return inval


def build_table(root, tag, cfg_name, algs, clutter, runs):
    T = {}
    for a in algs:
        cell = load_cell(root, tag, cfg_name, a, runs)
        T[a] = {}
        for lam in clutter:
            c = cell[str(lam)]
            G, Galt = table3_stat(c["gospa_sq"])
            T[a][lam] = dict(G=G, G_alt=Galt, ci=boot_G(c["gospa_sq"], 10000), gbar=run_scalar(c["gospa_sq"]), cell=c,
                             invalid=counters_ok(c), n=int(c["gospa_sq"].shape[0]))
    return T


def families(T, clutter):
    """The 33 pre-declared paired tests (H1 12, H2 9, H3 12).  Direction: first arm should be LOWER (better) than the second."""
    tests = {}
    gv = ["MCTS1-GD", "MCTS2-GD", "MCTS3-GD", "MCTS4-GD"]
    for lam in clutter:
        for v in gv:                                                           # H1: non-myopic GD < myopic GD
            tests[f"H1|{v}<Myopic-GD|{lam}"] = (T[v][lam]["gbar"], T["Myopic-GD"][lam]["gbar"])
        for v in ("MCTS1-GD", "MCTS2-GD", "MCTS4-GD"):                        # H2: MCTS3 < the others
            tests[f"H2|MCTS3-GD<{v}|{lam}"] = (T["MCTS3-GD"][lam]["gbar"], T[v][lam]["gbar"])
        for i in range(1, 5):                                                  # H3: GD < KLD
            tests[f"H3|MCTS{i}-GD<MCTS{i}-KLD|{lam}"] = (T[f"MCTS{i}-GD"][lam]["gbar"], T[f"MCTS{i}-KLD"][lam]["gbar"])
    res, pv = {}, {}
    for k, (a, b) in tests.items():
        d = np.asarray(a) - np.asarray(b)
        if np.allclose(d, 0):
            res[k] = dict(diff_mean=0.0, t_p=1.0, wins=0, n=len(d), boot95=[0.0, 0.0], identical=True)
        else:
            res[k] = paired_compare(a, b, n_boot=10000)
        pv[k] = res[k]["t_p"] if np.isfinite(res[k]["t_p"]) else 1.0
    h = holm(pv)
    for k in res:
        res[k]["p_holm"] = h[k]["p_holm"]
        res[k]["better_first"] = res[k]["diff_mean"] < 0
        res[k]["sig"] = bool(h[k]["p_holm"] < 0.05)
    return res


def verdicts(fam, T, clutter):
    """Apply the pre-declared H1-H3 / C4 rules (README 'Results')."""
    out = {}
    h1 = [v for k, v in fam.items() if k.startswith("H1")]
    n_sig_dir = sum(v["sig"] and v["better_first"] for v in h1)
    n_sig_against = sum(v["sig"] and not v["better_first"] for v in h1)
    out["H1"] = dict(sig_in_direction=n_sig_dir, of=len(h1), sig_against=n_sig_against,
                     verdict="REPRODUCED" if (n_sig_dir >= 11 and n_sig_against == 0) else "PARTIAL" if n_sig_dir >= 6 else "NOT")
    h2 = [v for k, v in fam.items() if k.startswith("H2")]
    lowest = 0
    for lam in clutter:
        g = {v: T[v][lam]["G"] for v in ("MCTS1-GD", "MCTS2-GD", "MCTS3-GD", "MCTS4-GD")}
        lowest += int(min(g, key=g.get) == "MCTS3-GD")
    n2 = sum(v["sig"] and v["better_first"] for v in h2)
    out["H2"] = dict(lowest_columns=lowest, sig_in_direction=n2, of=len(h2),
                     verdict="REPRODUCED" if (lowest == 3 and n2 >= 7) else "PARTIAL" if (lowest >= 2 or n2 >= 4) else "NOT")
    h3 = [(k, v) for k, v in fam.items() if k.startswith("H3")]
    pt = sum(v["diff_mean"] < 0 for _, v in h3)
    n3 = sum(v["sig"] and v["better_first"] for _, v in h3)
    n3a = sum(v["sig"] and not v["better_first"] for _, v in h3)
    out["H3"] = dict(point_direction_GD=pt, sig_GD=n3, sig_KLD=n3a, of=len(h3),
                     verdict="REPRODUCED" if (pt >= 9 and n3 >= 6 and n3a == 0) else "PARTIAL" if pt >= 7 else "NOT")
    rng_ok = {}
    for a in T:
        if a in ("RANDOM", "HOLD"):
            continue
        gs = [T[a][lam]["G"] for lam in clutter]
        rng_ok[a] = max(gs) - min(gs)
    out["C4"] = dict(range_by_alg={a: round(v, 3) for a, v in rng_ok.items()}, max_range=max(rng_ok.values()),
                     verdict="REPRODUCED" if max(rng_ok.values()) <= 5.0 else "NOT")
    return out


def c5_check(T, lam):
    """C5: myopic planners stuck above the obstacle (both sensors y > 90, |x| < 45 at the end) in >= 90 % runs; MCTS3-GD at t = 50
    has both sensors within 60 m of the origin in >= 90 % of runs."""
    res = {}
    for a in ("Myopic-GD", "Myopic-KLD"):
        sp = T[a][lam]["cell"]["spos"]                      # [R, K, 2, 2]
        fin = sp[:, -1]
        stuck = ((fin[:, :, 1] > 90) & (np.abs(fin[:, :, 0]) < 45)).all(axis=1)
        res[a] = float(stuck.mean())
    sp = T["MCTS3-GD"][lam]["cell"]["spos"]
    t50 = sp[:, min(50, sp.shape[1] - 1)]
    near = (np.sqrt((t50 ** 2).sum(-1)) < 60).all(axis=1)
    res["MCTS3-GD_near_origin_t50"] = float(near.mean())
    res["verdict"] = "REPRODUCED" if (res["Myopic-GD"] >= 0.9 and res["Myopic-KLD"] >= 0.9 and res["MCTS3-GD_near_origin_t50"] >= 0.9) else "NOT"
    return res


def fig5_curves(cell):
    """per-step RMS of total / loc / missed / false (each sqrt of the run-mean) from one cell dict."""
    return {k: np.sqrt(np.asarray(cell[k]).mean(axis=0)) for k in ("gospa_sq", "loc", "missed", "false")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cfg", default="CFG-O")
    ap.add_argument("--runs", type=int, required=True)
    ap.add_argument("--clutter", nargs="+", type=float, default=list(M.CLUTTER_RATES))
    ap.add_argument("--algs", nargs="+", default=M.ALG_ORDER)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    T = build_table(a.out, a.tag, a.cfg, a.algs, a.clutter, a.runs)
    print(f"Table III statistic G (time-mean of per-step RMS), {a.cfg}, {a.runs} runs, tag {a.tag}")
    print("%-12s" % "alg" + "".join(" | clutter %-5s G   [95%% CI]      paper  rel   grade" % c for c in a.clutter))
    for alg in a.algs:
        row = "%-12s" % alg
        for lam in a.clutter:
            t = T[alg][lam]
            p = M.PAPER_TABLE3.get(alg)
            if p is not None and lam in M.CLUTTER_RATES:
                pp = p[M.CLUTTER_RATES.index(lam)]
                rel, g = grade(t["G"], pp)
                row += " | %7.2f [%6.2f,%6.2f] %6.2f %5.1f%% %s" % (t["G"], t["ci"][0], t["ci"][1], pp, 100 * rel, g)
            else:
                row += " | %7.2f [%6.2f,%6.2f]" % (t["G"], t["ci"][0], t["ci"][1])
            if t["invalid"]:
                row += " INVALID%s" % t["invalid"]
        print(row)
    summary = dict(cfg=a.cfg, runs=a.runs, G={al: {str(l): T[al][l]["G"] for l in a.clutter} for al in a.algs})
    need = {"Myopic-GD", "MCTS1-GD", "MCTS2-GD", "MCTS3-GD", "MCTS4-GD", "MCTS1-KLD", "MCTS2-KLD", "MCTS3-KLD", "MCTS4-KLD"}
    if need <= set(a.algs):
        fam = families(T, a.clutter)
        v = verdicts(fam, T, a.clutter)
        summary["families"] = {k: {kk: vv for kk, vv in d.items() if kk != "diff_per_seed"} for k, d in fam.items()}
        summary["verdicts"] = v
        print(json.dumps(v, indent=1))
    if "Myopic-GD" in a.algs and "Myopic-KLD" in a.algs and "MCTS3-GD" in a.algs:
        summary["C5"] = c5_check(T, a.clutter[min(1, len(a.clutter) - 1)])
        print("C5", summary["C5"])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(summary, f, indent=1, default=float)


if __name__ == "__main__":
    main()

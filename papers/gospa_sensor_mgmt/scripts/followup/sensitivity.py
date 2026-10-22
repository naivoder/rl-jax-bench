"""Follow-up sensitivity report (CFG-P2 vs one-at-a-time flips, 50 runs, same CRN streams).

For every flip in tag p2_sens: Table III statistic G [95 % run-bootstrap CI] and the paired difference of the per-run mean GOSPA gbar against the
CFG-P2 baseline cell of the same algorithm (tag p2), paired t + bootstrap CI, Holm over all flips x clutter rates.  Then H3 (GD < KLD) conditional
on the KLD form / on each shared flip, and the trace-coordinate consistency table.  Output: runs/report/followup_sensitivity.json + printed markdown.
Usage: python scripts/followup/sensitivity.py"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from src import eval as E  # noqa: E402
from rl_jax_bench.stats import paired_compare, holm  # noqa: E402

ROOT = os.environ.get("GOSPA_RUNS", "runs")
LAMS = ["0.1", "1.0", "2.0"]
OUT = {}


def cell(tag, cfg, alg):
    for n in (50, 25):                                  # some flips were flown with 25 runs (same keys = the first 25 runs of the 50-run baseline)
        try:
            return E.load_cell(ROOT, tag, cfg, alg, n)
        except (FileNotFoundError, KeyError):
            pass
    return None


def G(c, lam, key="gospa_sq"):
    return E.table3_stat(c[lam][key])[0]


def gbar(c, lam, key="gospa_sq"):
    return E.run_scalar(c[lam][key])


def pc(a, b):
    n = min(len(a), len(b))                             # paired on the common first n runs
    a, b = np.asarray(a)[:n], np.asarray(b)[:n]
    d = a - b
    if np.allclose(d, 0):
        return dict(diff_mean=0.0, t_p=1.0, wins=0, boot95=[0.0, 0.0], identical=True)
    return paired_compare(a, b, n_boot=10000)


FLIPS = [  # (flip id, label, cfg name in p2_sens, algs)
    ("kld_form", "kld_mode=info_gain", "CFG-P2+kld_mode=info_gain", ["Myopic-KLD", "MCTS1-KLD", "MCTS2-KLD", "MCTS3-KLD", "MCTS4-KLD"]),
    ("budget", "budget_mode=iter", "CFG-P2+budget_mode=iter", ["MCTS3-GD", "MCTS4-GD", "MCTS3-KLD", "MCTS4-KLD"]),
    ("trace", "plan_trace=pos", "CFG-P2+plan_trace=pos", ["Myopic-GD", "MCTS1-GD", "MCTS3-GD"]),
    ("estimator", "est_rule=gamma", "CFG-P2+est_rule=gamma", ["Myopic-GD", "MCTS3-GD", "MCTS3-KLD"]),
    ("merge", "merge_each=true", "CFG-P2+merge_each=true", ["Myopic-GD", "MCTS3-GD", "MCTS3-KLD"]),
    ("uct_eps", "uct_eps=300", "CFG-P2+uct_eps=300", ["MCTS3-GD", "MCTS3-KLD"]),
]
rows, pv = [], {}
cells = {}
for fid, label, cfgn, algs in FLIPS:
    for a in algs:
        c, b = cell("p2_sens", cfgn, a), cell("p2", "CFG-P2", a)
        if c is None or b is None:
            print("missing", cfgn, a); continue
        cells[(label, a)] = c
        for lam in LAMS:
            r = pc(gbar(c, lam), gbar(b, lam))
            k = f"{label}|{a}|{lam}"
            pv[k] = r["t_p"] if np.isfinite(r["t_p"]) else 1.0
            n = c[lam]["gospa_sq"].shape[0]
            rows.append(dict(key=k, flip_id=fid, flip=label, alg=a, lam=lam, n=n, G=G(c, lam), ci=E.boot_G(c[lam]["gospa_sq"], 10000), G_base=E.table3_stat(b[lam]["gospa_sq"][:n])[0],
                             diff=r["diff_mean"], boot95=r["boot95"], p=pv[k], wins=r.get("wins"), inv=E.counters_ok(c[lam]),
                             r_gt_1=int(np.sum(c[lam].get("r_gt_1", [0])))))
h = holm(pv)
for r in rows:
    r["p_holm"] = h[r["key"]]["p_holm"]
OUT["flips"] = rows
print("| flip id | flip | algorithm | lambda_C | runs | G flip [95 % CI] | G CFG-P2 (same runs) | paired diff of gbar (flip - base) [boot 95 %] | p Holm | sig |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print("| %s | %s | %s | %s | %d | %.2f [%.2f, %.2f] | %.2f | %+.2f [%+.2f, %+.2f] | %.3g | %s |" % (
        r["flip_id"], r["flip"], r["alg"], r["lam"], r["n"], r["G"], r["ci"][0], r["ci"][1], r["G_base"], r["diff"], r["boot95"][0], r["boot95"][1],
        r["p_holm"], "yes" if r["p_holm"] < 0.05 else "no"))

# ---- H3 conditional on the KLD form / shared flips
print("\nH3 (GD < KLD) conditional on the KLD form")
H3 = {}
def h3(label, gd_get, kd_get, variants):
    res, p = {}, {}
    for v in variants:
        for lam in LAMS:
            gd, kd = gd_get(v), kd_get(v)
            if gd is None or kd is None:
                continue
            r = pc(gbar(gd, lam), gbar(kd, lam))
            k = f"{v}|{lam}"
            n = min(gd[lam]["gospa_sq"].shape[0], kd[lam]["gospa_sq"].shape[0])
            res[k] = dict(n=n, G_gd=E.table3_stat(gd[lam]["gospa_sq"][:n])[0], G_kld=E.table3_stat(kd[lam]["gospa_sq"][:n])[0], diff=r["diff_mean"], boot95=r["boot95"], p=r["t_p"], wins_gd=int(50 - r.get("wins", 0)) if r.get("wins") is not None else None)
            p[k] = r["t_p"] if np.isfinite(r["t_p"]) else 1.0
    hh = holm(p)
    for k in res:
        res[k]["p_holm"] = hh[k]["p_holm"]
    H3[label] = res
    nsig_gd = sum(v["p_holm"] < 0.05 and v["diff"] < 0 for v in res.values())
    nsig_k = sum(v["p_holm"] < 0.05 and v["diff"] > 0 for v in res.values())
    ndir = sum(v["diff"] < 0 for v in res.values())
    print(f"{label}: {len(res)} cells, GD better in {ndir}, Holm-significant GD {nsig_gd}, KLD {nsig_k}")
    for k, v in res.items():
        print("  %s (n=%d) G_GD %.2f G_KLD %.2f diff %+.2f [%+.2f, %+.2f] p_holm %.3g" % (k, v["n"], v["G_gd"], v["G_kld"], v["diff"], v["boot95"][0], v["boot95"][1], v["p_holm"]))
    return res
vs = ["MCTS1", "MCTS2", "MCTS3", "MCTS4"]
h3("bernoulli (CFG-P2 default)", lambda v: cell("p2", "CFG-P2", v + "-GD"), lambda v: cell("p2", "CFG-P2", v + "-KLD"), vs)
h3("info_gain KLD", lambda v: cell("p2", "CFG-P2", v + "-GD"), lambda v: cell("p2_sens", "CFG-P2+kld_mode=info_gain", v + "-KLD"), vs)
h3("CFG-P v1 (archived, bernoulli KL on the position block)", lambda v: cell("e1", "CFG-P", v + "-GD"), lambda v: cell("e1", "CFG-P", v + "-KLD"), vs)
for lab, cfgn, tag in (("budget_mode=iter (both arms)", "CFG-P2+budget_mode=iter", "p2_sens"), ("est_rule=gamma (both arms)", "CFG-P2+est_rule=gamma", "p2_sens"),
                       ("merge_each (both arms)", "CFG-P2+merge_each=true", "p2_sens"), ("uct_eps=300 (both arms)", "CFG-P2+uct_eps=300", "p2_sens")):
    h3(lab, lambda v, c=cfgn: cell("p2_sens", c, v + "-GD"), lambda v, c=cfgn: cell("p2_sens", c, v + "-KLD"), ["MCTS3"] + (["MCTS4"] if "budget" in lab else []))
OUT["H3"] = H3

# ---- trace-coordinate consistency table (G on the 4-D metric and on the position metric)
print("\nplanner trace x metric coordinates")
TRACE = {}
for a in ("Myopic-GD", "MCTS1-GD", "MCTS3-GD", "MCTS4-GD"):
    v1, p2, pp = cell("e1", "CFG-P", a), cell("p2", "CFG-P2", a), cell("p2_sens", "CFG-P2+plan_trace=pos", a)
    for lam in LAMS:
        row = {}
        if v1: row["v1 pos-trace, 4-D metric"] = G(v1, lam); row["v1 pos-trace, pos metric"] = G(v1, lam, "gospa_pos_sq")
        if p2: row["P2 full-trace, 4-D metric"] = G(p2, lam); row["P2 full-trace, pos metric"] = G(p2, lam, "gospa_pos_sq")
        if pp: row["P2 pos-trace (nodes budget), 4-D metric"] = G(pp, lam); row["P2 pos-trace (nodes budget), pos metric"] = G(pp, lam, "gospa_pos_sq")
        TRACE[f"{a}|{lam}"] = row
        print(a, lam, {k: round(v, 2) for k, v in row.items()})
OUT["trace_consistency"] = TRACE
json.dump(OUT, open(os.path.join(HERE, "..", "..", "runs", "report", "followup_sensitivity.json"), "w"), indent=1, default=float)

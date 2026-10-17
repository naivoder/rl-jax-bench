"""Recheck (pre-declared): window-matched re-grade of the learner figure rows
against the vector-extracted paper curves, plus the paper run's rank among our seeds and a 95 % single-seed prediction
interval. CPU, reads existing runs only (no learner rerun).
Rule: MATCH iff |rel| <= 5 % or |diff| <= 0.01 + our 95 % t half-width; CLOSE iff |rel| <= 20 %;
else DIVERGENT. The 'stricter of J and J_conv' is kept; J is now compared with the paper's 70-CPI mean and J_conv with its
CPI 11-70 mean (the matched windows)."""
import json, os
import numpy as np
from scipy import stats

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROOT = os.path.join(HERE, "runs")
V3 = json.load(open(os.path.join(HERE, "data", "fig3_vector.json")))   # values extracted from the paper's vector figures
V4 = json.load(open(os.path.join(HERE, "data", "fig4_vector.json")))
SEG = [15, 15, 15, 15, 10]


def per_cpi(run, arm, exp, reading):
    lj = json.load(open(f"{ROOT}/{run}/lanes.json"))
    ids, ntr = [tuple(x) for x in lj["ids"]], lj["config"]["train_cpis"]
    o = np.load(f"{ROOT}/{run}/arm_{arm}.npz")["outs"][:, ntr:, 0] / 1000.0
    sel = [i for i, (e, r, s) in enumerate(ids) if e == exp and r == reading]
    return o[sel]                       # [seeds, 70]


def per_cpi_baseline(run, name, exp, reading):
    lj = json.load(open(f"{ROOT}/{run}/lanes.json"))
    ids = [tuple(x) for x in lj["ids"]]
    o = np.load(f"{ROOT}/{run}/baselines.npz")[f"{name}__outs"][..., 0] / 1000.0
    sel = [i for i, (e, r, s) in enumerate(ids) if e == exp and r == reading]
    return o[sel]


def judge(ours_seeds, paper):
    m, n = ours_seeds.mean(), len(ours_seeds)
    hw = stats.t.ppf(0.975, n - 1) * ours_seeds.std(ddof=1) / np.sqrt(n)
    rel = 100 * (m - paper) / paper
    if abs(rel) <= 5 or abs(m - paper) <= 0.01 + hw:
        lab = "MATCH"
    elif abs(rel) <= 20:
        lab = "CLOSE"
    else:
        lab = "DIVERGENT"
    sd = ours_seeds.std(ddof=1)
    pi = stats.t.ppf(0.975, n - 1) * sd * np.sqrt(1 + 1 / n)
    rank = int((ours_seeds < paper).sum())         # number of our seeds below the paper value
    return dict(ours=round(float(m), 4), paper=round(float(paper), 4), rel_pct=round(float(rel), 2), label=lab,
                seeds_below_paper=f"{rank}/{n}", pred_int95=[round(float(m - pi), 4), round(float(m + pi), 4)],
                paper_inside_pred_int=bool(abs(paper - m) <= pi))


ORDER = {"MATCH": 0, "CLOSE": 1, "DIVERGENT": 2}
rows = []
fig3 = [("L", "mc_p04", "Fig 3L"), ("R", "mc_p02", "Fig 3R")]
runs = {("mc_p04", 100): "full", ("mc_p02", 100): "full", ("mc_p04", 500): "full_500cpi", ("mc_p02", 500): "review_e2"}
for panel, exp, name in fig3:
    for meth, arm in (("DDRQN", "ddrqn"), ("DQN", "dqn")):
        y = np.asarray(V3[panel][meth]["y"])
        pJ, pJc = y.mean(), y[10:].mean()
        for cpis in (100, 500):
            rc = per_cpi(runs[(exp, cpis)], arm, exp, "lin")
            jJ, jJc = judge(rc.mean(1), pJ), judge(rc[:, 10:].mean(1), pJc)
            strict = max((jJ["label"], jJc["label"]), key=ORDER.get)
            rows.append(dict(row=f"{name} {meth} R_lin @{cpis}", J=jJ, J_conv=jJc, judgment=strict))
    # PI (only @100 has a PI with this run's streams; @500 PI from review_pi500)
    y = np.asarray(V3[panel]["PI"]["y"])
    for cpis, run in ((100, "full"), (500, "review_pi500")):
        try:
            rc = per_cpi_baseline(run, "pi", exp, "lin")
        except Exception as ex:          # noqa: BLE001
            continue
        jJ, jJc = judge(rc.mean(1), y.mean()), judge(rc[:, 10:].mean(1), y[10:].mean())
        rows.append(dict(row=f"{name} PI R_lin @{cpis}", J=jJ, J_conv=jJc,
                         judgment=max((jJ["label"], jJc["label"]), key=ORDER.get)))
        if panel == "L":   # the plotted 'PI' is pi* scored under Eq. 3
            rq = per_cpi_baseline(run, "pi", exp, "eq3")
            jJ, jJc = judge(rq.mean(1), y.mean()), judge(rq[:, 10:].mean(1), y[10:].mean())
            rows.append(dict(row=f"{name} PI R_eq3 @{cpis} (the curve is pi* under Eq. 3)", J=jJ, J_conv=jJc,
                             judgment=max((jJ["label"], jJc["label"]), key=ORDER.get)))

# Fig 4 segments (R_lin) @100 (full mc_sched) and @500 (review_e3)
ends = np.cumsum(SEG); starts = ends - np.array(SEG)
for meth, arm in (("DDRQN", "ddrqn"), ("DQN", "dqn")):
    y = np.asarray(V4[meth]["y"])
    for cpis, run in ((100, "full"), (500, "review_e3")):
        rc = per_cpi(run, arm, "mc_sched", "lin")
        for k, (s, e) in enumerate(zip(starts, ends)):
            j = judge(rc[:, s:e].mean(1), y[s:e].mean())
            rows.append(dict(row=f"Fig 4 {meth} R_lin seg{k + 1} @{cpis}", J_seg=j, judgment=j["label"]))

json.dump(rows, open(f"{ROOT}/regrade_figs.json", "w"), indent=1)
for r in rows:
    if "J" in r:
        print(f"{r['row']:<52} J {r['J']['ours']:.4f} vs {r['J']['paper']:.4f} ({r['J']['rel_pct']:+.1f}%) | "
              f"Jc {r['J_conv']['ours']:.4f} vs {r['J_conv']['paper']:.4f} ({r['J_conv']['rel_pct']:+.1f}%) -> {r['judgment']}"
              f" | below {r['J_conv']['seeds_below_paper']} PI95 {r['J_conv']['pred_int95']}")
    else:
        j = r["J_seg"]
        print(f"{r['row']:<52} {j['ours']:.4f} vs {j['paper']:.4f} ({j['rel_pct']:+.1f}%) -> {r['judgment']} | below {j['seeds_below_paper']}")

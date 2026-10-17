"""Grades the conditional T-C audit (the README full-scale addendum): DDRQN and DQN at the paper's 500 training CPIs (E1) vs the
100-CPI full-scale arms, paired by seed on identical evaluation streams. Usage: python scripts/long_train_report.py runs/full_500cpi runs/full"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import eval as EV  # noqa: E402
from rl_jax_bench.stats import mean_se, paired_compare  # noqa: E402

TC, G3 = sys.argv[1], sys.argv[2]
la = json.load(open(os.path.join(TC, "lanes.json")))
ids = [tuple(x) for x in la["ids"]]
g3ids = [tuple(x) for x in json.load(open(os.path.join(G3, "lanes.json")))["ids"]]
bz = np.load(os.path.join(G3, "baselines.npz"))
ntr = la["config"]["train_cpis"]
m500 = {}
for arm in ("dqn", "ddrqn"):
    z = np.load(os.path.join(TC, f"arm_{arm}.npz"))
    o, fp = z["outs"][:, ntr:], z["fps"][:, ntr:]
    ident = all(np.array_equal(fp[i], bz["pistar__fps"][g3ids.index(ids[i])]) for i in range(len(ids)))
    m500[arm] = ({ids[i]: EV.lane_metrics(o[i], [70]) for i in range(len(ids))}, ident)
R = {}
for rd in ("eq3", "lin"):
    seeds = sorted(s for e, r, s in ids if r == rd)
    g = {s: json.load(open(os.path.join(G3, f"seed_{s}", "metrics.json")))["lanes"][f"mc_p04/{rd}"]["metrics"] for s in seeds}
    v = lambda arm, met: np.array([m500[arm][0][("mc_p04", rd, s)][met] for s in seeds])  # noqa: E731
    w = lambda arm, met: np.array([g[s][arm][met] for s in seeds])  # noqa: E731
    d_len = paired_compare(v("ddrqn", "J_conv"), w("ddrqn", "J_conv"), n_boot=10_000)
    p1 = d_len["t_p"] / 2 if d_len["t"] > 0 else 1 - d_len["t_p"] / 2
    d_dqn = paired_compare(v("ddrqn", "J_conv"), v("dqn", "J_conv"), n_boot=10_000)
    d_q = paired_compare(v("dqn", "J_conv"), w("dqn", "J_conv"), n_boot=10_000)
    gap = w("pistar", "J_conv") - v("ddrqn", "J_conv")
    R[rd] = dict(ddrqn500=mean_se(v("ddrqn", "J_conv")), dqn500=mean_se(v("dqn", "J_conv")),
                 ddrqn500_sd=mean_se(v("ddrqn", "sigma_within")), dqn500_sd=mean_se(v("dqn", "sigma_within")),
                 ddrqn500_adapt=mean_se(v("ddrqn", "adapt_pct")), ddrqn500_coll=mean_se(v("ddrqn", "coll_pct")),
                 ddrqn500_minus_100={k: d_len[k] for k in ("diff_mean", "diff_se", "t95", "boot95", "wins", "n")}, p_one=p1,
                 ddrqn500_minus_dqn500={k: d_dqn[k] for k in ("diff_mean", "diff_se", "t95", "boot95", "wins", "n")},
                 dqn500_minus_100={k: d_q[k] for k in ("diff_mean", "t95", "wins")},
                 H_a_gap=mean_se(gap), H_a_upper=float(gap.mean() + 2.262 * gap.std(ddof=1) / np.sqrt(len(gap))),
                 implicated=bool(p1 < 0.05 and d_len["diff_mean"] > 0),
                 explains=bool(p1 < 0.05 and d_len["diff_mean"] > 0 and d_dqn["t95"][1] >= 0))
R["stream_identity"] = {a: m500[a][1] for a in m500}
json.dump(R, open(os.path.join(TC, "long_train_tests.json"), "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))

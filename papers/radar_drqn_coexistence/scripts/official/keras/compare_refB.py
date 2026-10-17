import numpy as np, json, os, sys
PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the paper folder
from scipy import stats
R = os.path.join(PAPER, "runs", "full")
ids = json.load(open(f"{R}/lanes.json"))["ids"]
res = {}
for arm in ("dqn", "ddrqn"):
    G = np.load(f"{R}/arm_{arm}.npz")["outs"]
    ours = []
    for s in range(3):
        rc = G[ids.index(["mc_p04", "eq3", s]), 100:, 0] / 1000
        ours.append(dict(J_conv=rc[10:].mean(), sigma=rc[10:].std(ddof=1)))
    for adam in ("kerasadam", "optaxadam"):
        fs = [f"refB_{arm}_{adam}_s{s}.json" for s in range(3)]
        if not all(os.path.exists(f) for f in fs):
            continue
        K = [json.load(open(f)) for f in fs]
        k = np.array([x["J_conv"] for x in K]); o = np.array([x["J_conv"] for x in ours]); d = k - o
        t = stats.ttest_rel(k, o)
        res[f"{arm}_{adam}"] = dict(keras_J_conv=k.round(4).tolist(), ours_J_conv=o.round(4).tolist(), keras_mean=float(k.mean()),
            ours_mean=float(o.mean()), diff_mean=float(d.mean()), diff_se=float(d.std(ddof=1) / np.sqrt(3)),
            diff_t95=[float(d.mean() - 4.303 * d.std(ddof=1) / np.sqrt(3)), float(d.mean() + 4.303 * d.std(ddof=1) / np.sqrt(3))],
            p_two_sided=float(t.pvalue), keras_sigma_within=[round(x["sigma_within"], 4) for x in K],
            ours_sigma_within=[round(float(x["sigma"]), 4) for x in ours], keras_adapt_pct=[round(x["adapt_pct"], 1) for x in K],
            rule_PARITY_abs_le_0p02=bool(abs(d.mean()) <= 0.02), rule_our_pipeline_suspect=bool(d.mean() >= 0.05 and arm == "ddrqn"))
for arm in ("dqn", "ddrqn"):
    if f"{arm}_kerasadam" in res and f"{arm}_optaxadam" in res:
        a = np.array(res[f"{arm}_kerasadam"]["keras_J_conv"]); b = np.array(res[f"{arm}_optaxadam"]["keras_J_conv"])
        res[f"{arm}_kerasadam_minus_optaxadam_within_keras"] = dict(diff=(a - b).round(4).tolist(), mean=float((a - b).mean()))
print(json.dumps(res, indent=1)); json.dump(res, open("refB_compare.json", "w"), indent=1)

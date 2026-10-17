"""Plain-SGD oracle check (recheck rule 6): Keras TF 2.2 SGD learners vs our recheck dqn_sgd / ddrqn_sgd at E2 (mc_p02, eq3),
seeds 0-2 of the recheck pool, same init + replay indices. PARITY iff |mean diff| <= 0.02 J_conv per arm; implementation suspect iff
Keras and ours disagree on the E2 validity check (upper 95 % t of pi* - J_conv <= 0.02)."""
import json, os, sys
import numpy as np
from scipy import stats
PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the paper folder
sys.path.insert(0, os.path.join(PAPER, "scripts"))
from review_report import arm_metrics, base_metrics  # noqa: E402
R = os.path.join(PAPER, "runs")
O = f"{R}/keras_oracle"
seeds = [0, 1, 2]
pist = np.array([base_metrics(f"{R}/recheck", "pistar")[0][("mc_p02", "eq3", s)]["J_conv"] for s in seeds])
out = {}
for arm, ours_arm in (("dqn", "dqn_sgd"), ("ddrqn", "ddrqn_sgd")):
    k = np.array([json.load(open(f"{O}/sgdE2_{arm}_s{s}.json"))["J_conv"] for s in seeds])
    ka = np.array([json.load(open(f"{O}/sgdE2_{arm}_s{s}.json"))["adapt_pct"] for s in seeds])
    m = arm_metrics(f"{R}/recheck", ours_arm)[0]
    o = np.array([m[("mc_p02", "eq3", s)]["J_conv"] for s in seeds])
    oa = np.array([m[("mc_p02", "eq3", s)]["adapt_pct"] for s in seeds])
    d = k - o
    t = stats.t.ppf(0.975, 2)
    def valid(x):
        g = pist - x
        return bool(g.mean() + t * g.std(ddof=1) / np.sqrt(3) <= 0.02)
    out[arm] = dict(keras_J_conv=k.round(4).tolist(), ours_J_conv=o.round(4).tolist(), keras_adapt=ka.round(1).tolist(),
                    ours_adapt=oa.round(1).tolist(), pistar=pist.round(4).tolist(), diff_mean=float(d.mean()),
                    diff_t95=[float(d.mean() - t * d.std(ddof=1) / np.sqrt(3)), float(d.mean() + t * d.std(ddof=1) / np.sqrt(3))],
                    PARITY=bool(abs(d.mean()) <= 0.02), keras_valid=valid(k), ours_valid=valid(o),
                    implementation_suspect=bool(valid(k) != valid(o)))
json.dump(out, open(f"{O}/sgdE2_compare.json", "w"), indent=1)
print(json.dumps(out, indent=1))

"""Frozen-policy recheck report: OUR frozen policy (Adam lr 0) vs the AUTHORS' frozen StreamAC (official_frozen.json), and
Adam 1e-7 without traces (lam 0) vs with traces, real Hopper-v4 version-matched. Writes
runs/recheck/frozen_policy/frozen_policy_report.json."""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from scipy import stats as st  # noqa: E402
from src import eval as ev  # noqa: E402
from rl_jax_bench.stats import mean_se  # noqa: E402

D = os.path.join(REPO, "runs", "recheck", "frozen_policy")


def bins(arm, seeds, root, total, stride):
    out = {}
    for s in seeds:
        e = np.asarray(json.load(open(os.path.join(root, arm, f"seed_{s}", "metrics.json")))["episodes"], float)
        out[s] = ev.run_bins(e, total, stride)
    return out


lr0_25 = bins("no_obgd@lr=0", range(1, 9), os.path.join(D, "ours_lr0"), 100_000, 25_000)
lr0_50 = bins("no_obgd@lr=0", range(1, 9), os.path.join(D, "ours_lr0"), 100_000, 50_000)
lam0_50 = bins("no_obgd@lr=1e-7@lam=0", (1, 2, 3), os.path.join(D, "ours_lr0"), 100_000, 50_000)
lam8_50 = bins("no_obgd@lr=1e-7", (1, 2, 3), os.path.join(REPO, "runs", "recheck", "no_obgd_real_hopper"), 100_000, 50_000)
off = json.load(open(os.path.join(D, "official_frozen.json")))
off_first = [v["mean"] for v in off.values()]
ours_first = [lr0_25[s][0] for s in range(1, 9)]
w = st.ttest_ind(ours_first, off_first, equal_var=False)
ms_o, se_o = mean_se(ours_first)
ms_a, se_a = mean_se(off_first)
res = dict(
    frozen_first_25k=dict(ours_lr0=[ms_o, se_o], authors=[ms_a, se_a], diff=ms_o - ms_a,
                          two_combined_se=2 * float(np.hypot(se_o, se_a)), welch_p=float(w.pvalue),
                          ours_per_seed=np.round(ours_first, 2).tolist(), authors_per_seed=np.round(off_first, 2).tolist()),
    ours_lr0_25k_curve=[mean_se([lr0_25[s][i] for s in range(1, 9)]) for i in range(4)],
    ours_lr0_50k_curve=[mean_se([lr0_50[s][i] for s in range(1, 9)]) for i in range(2)],
    lam0_vs_lam08_50k={str(s): dict(lam0=np.round(lam0_50[s], 2).tolist(), lam08=np.round(lam8_50[s], 2).tolist())
                       for s in (1, 2, 3)},
    lam0_50k_curve=[mean_se([lam0_50[s][i] for s in (1, 2, 3)]) for i in range(2)],
    lam08_50k_curve=[mean_se([lam8_50[s][i] for s in (1, 2, 3)]) for i in range(2)])
json.dump(res, open(os.path.join(D, "frozen_policy_report.json"), "w"), indent=1)
print(json.dumps(res, indent=1))

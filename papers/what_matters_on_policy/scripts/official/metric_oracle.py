# Metric oracle: the notebook's OWN percentile_ci and plot_top_90 top-set rule, exec'd verbatim from
# seed_rl/mujoco/what_matters_in_on_policy_rl.ipynb (nothing re-typed), applied to the authors' study rows. Dumps the
# per-(experiment, choice, env, value) config-score vectors (x.iloc[-1] of _eval_, sorted as the notebook does) and
# the notebook's (percentile, low, high, careful-flag) for p in {0.9, 0.95, 0.99}, plus the notebook's top-5% set
# value counts per env, so tests/test_official_parity.py can compare src/eval.py on the same vectors.
#   uv run python scripts/official/metric_oracle.py <out.npz>
import json
import pickle
import re
import sys

import numpy as np
from scipy.stats import binom  # noqa: F401  (used by the exec'd notebook code)

import os  # noqa: E402
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' checkout, see README
NB = os.path.join(OFFICIAL_ROOT, "seed_rl", "mujoco", "what_matters_in_on_policy_rl.ipynb")
src = "\n".join("".join(c["source"]) for c in json.load(open(NB))["cells"] if c["cell_type"] == "code")


def grab(name):
    i = src.index(f"def {name}(")
    j = re.search(r"\n(def |\S)", src[i + 1:]).start() + i + 1
    return src[i:j]


ns = {"np": np, "binom": binom}
exec(grab("percentile_ci"), ns)
top_src = grab("plot_top_90")
# the notebook's top-set rule, verbatim line from plot_top_90:
TOP_LINE = "p90 = p90[-((len(p90) - 1) // (100 // (100 - percentile)) + 1):]"
assert TOP_LINE in top_src
percentile_ci = ns["percentile_ci"]

D = pickle.load(open(os.path.join(OFFICIAL_ROOT, "data", "mujoco123.pkl"), "rb"))
name = {m["title"].split(": ")[1].split(" ")[0]: k for k, (df, m) in D.items()}
P = "_gin.study_design.choice_value_"
GROUPS = [("final_stability", "input_normalization"), ("final_stability", "reward_normalization"),
          ("final_stability", "normalize_advantages"), ("final_arch2", "initial_action_std"),
          ("final_advantages", "sub_advantage_estimator_gae_gae_lambda")]
out = {}
for ex, ch in GROUPS:
    df, _ = D[name[ex]]
    for env in ["Hopper-v1", "Walker2d-v1", "HalfCheetah-v1"]:
        e = df[df["env_name"] == env]
        e = e[e[P + ch].notnull()]
        allx = []
        for v, grp in e.groupby(P + ch):
            x = [s.iloc[-1] if hasattr(s, "iloc") else s[max(s)] for s in grp["_eval_"].values]
            x.sort()
            key = f"{ex}|{ch}|{env}|{v}"
            out[f"x/{key}"] = np.array(x, np.float64)
            for p in (0.9, 0.95, 0.99):
                r = percentile_ci(x, p)
                out[f"ci/{key}/{p}"] = np.array([r[0], r[1], r[2], float(r[3])], np.float64)
            allx += [(xx, str(v)) for xx in x]
        # top-5% set of this env (notebook rule with percentile=95), value counts
        p90 = sorted(allx)
        p90 = p90[-((len(p90) - 1) // (100 // (100 - 95)) + 1):]
        vals, cnt = np.unique([v for _, v in p90], return_counts=True)
        out[f"top/{ex}|{ch}|{env}/values"] = vals
        out[f"top/{ex}|{ch}|{env}/counts"] = cnt
        out[f"top/{ex}|{ch}|{env}/all_scores"] = np.array([a for a, _ in allx])
        out[f"top/{ex}|{ch}|{env}/all_values"] = np.array([b for _, b in allx])
np.savez_compressed(sys.argv[1], **out)
print("[INFO] wrote", sys.argv[1], len(out), "arrays")

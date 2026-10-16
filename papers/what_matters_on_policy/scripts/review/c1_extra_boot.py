"""review: bootstrap CIs for the REC-like subsets of A5a and A1 (the review covered A2b/A4/A6/A7/A8)."""
import pickle
import numpy as np
P = "_gin.study_design.choice_value_"
import os  # noqa: E402
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' checkout, see README
D = pickle.load(open(os.path.join(OFFICIAL_ROOT, "data", "mujoco123.pkl"), "rb"))
NAME = {m["title"].split(": ")[1].split(" ")[0]: k for k, (df, m) in D.items()}
q = lambda v: f"'{v}'"
rng = np.random.default_rng(0)


def frame(ex):
    df, _ = D[NAME[ex]]
    df = df.copy()
    df["score"] = df["_eval_"].map(lambda d: d[max(d)])
    return df[df.env_name == "HalfCheetah-v1"]


def boot(a, r, B=10000):
    out = [np.percentile(a[rng.integers(0, len(a), len(a))], 95) / np.percentile(r[rng.integers(0, len(r), len(r))], 95)
           for _ in range(B)]
    return np.percentile(out, [2.5, 97.5])


def sub(df, filters, col, arm, rec):
    s = df
    for c, v in filters.items():
        s = s[s[P + c].astype(str).isin([q(x) for x in v])]
    return (s[s[P + col].astype(str) == q(arm)]["score"].values.astype(float),
            s[s[P + col].astype(str) == q(rec)]["score"].values.astype(float))


fo = frame("final_optimize")
a, r = sub(fo, {"sub_optimizer_adam_momentum": ["0.9"], "learning_rate_decay": ["1.0"],
                "sub_optimizer_adam_epsilon": ["1e-05", "1e-06", "1e-07", "1e-08", "1e-09"]},
           "sub_optimizer_adam_learning_rate", "0.0001", "0.0003")
print("A5a REC-like n", len(a), len(r), "q95", round(np.percentile(a, 95) / np.percentile(r, 95), 3), "boot95", boot(a, r).round(3),
      "median", round(np.median(a) / np.median(r), 3))
fs = frame("final_stability")
F = {"learning_rate": ["0.0003", "0.001"], "reward_normalization": ["Popart Avg (comp=False)"], "normalize_advantages": ["False"],
     "gradient_clipping": ["0.5", "1.0", "2.0"], "ppo_epsilon": ["0.2", "0.3"]}
a, r = sub(fs, F, "input_normalization", "None", "Avg (comp=False)")
print("A1 REC-like n", len(a), len(r), "q95", round(np.percentile(a, 95) / np.percentile(r, 95), 3), "boot95", boot(a, r).round(3))

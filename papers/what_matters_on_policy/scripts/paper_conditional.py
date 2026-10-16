"""Paper-side estimand check for the full-scale cells (DIVERGENT cause list): do the authors' own study
rows show the one-factor effect when the OTHER sampled choices are held at REC-like values?

The paper's per-choice number is a conditional 95th percentile over configurations in which every other sampled
choice is random. Our arms change one choice around REC. This script re-reads the authors' raw rows
(mujoco123_data.json via the converted pickle) and compares the arm value with the REC value on progressively
REC-like subsets: median, mean and q95 of the configuration score (last `_eval_` entry), with n and a
Mann-Whitney p (two-sided; descriptive, no family).

Usage: uv run python scripts/paper_conditional.py  (CPU; writes runs/p95/paper_conditional.json)
"""
import json
import os
import pickle

import numpy as np
from scipy.stats import mannwhitneyu

P = "_gin.study_design.choice_value_"
import os  # noqa: E402
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official"))  # authors' checkout, see README
D = pickle.load(open(os.path.join(OFFICIAL_ROOT, "data", "mujoco123.pkl"), "rb"))
NAME = {m["title"].split(": ")[1].split(" ")[0]: k for k, (df, m) in D.items()}
ENVS = ["HalfCheetah-v1", "Walker2d-v1", "Hopper-v1"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "p95")


def q(v):
    return f"'{v}'"


def frame(ex):
    df, _ = D[NAME[ex]]
    df = df.copy()
    df["score"] = df["_eval_"].map(lambda d: d[max(d)])
    return df


def compare(df, col, arm, rec, filters, env):
    s = df[df.env_name == env]
    for c, vals in filters.items():
        s = s[s[P + c].astype(str).isin([q(v) for v in vals])]
    a = s[s[P + col].astype(str) == q(arm)]["score"].values.astype(float)
    r = s[s[P + col].astype(str) == q(rec)]["score"].values.astype(float)
    if len(a) < 3 or len(r) < 3:
        return {"n_arm": int(len(a)), "n_rec": int(len(r))}
    p = mannwhitneyu(a, r).pvalue
    return {"n_arm": int(len(a)), "n_rec": int(len(r)),
            "median_ratio": float(np.median(a) / np.median(r)), "mean_ratio": float(a.mean() / r.mean()),
            "q95_ratio": float(np.percentile(a, 95) / np.percentile(r, 95)),
            "median_arm": float(np.median(a)), "median_rec": float(np.median(r)), "mw_p": float(p)}


def main():
    os.makedirs(OUT, exist_ok=True)
    res = []
    a2 = frame("final_arch2")
    REC_ARCH = [
        ("none", {}),
        ("tanh", {"action_postprocessing": ["tanh"]}),
        ("tanh+softplus+global", {"action_postprocessing": ["tanh"], "scale_function": ["@tf.nn.softplus"],
                                  "std_independent_of_input": ["True"]}),
        ("+lr3e-4/1e-3", {"action_postprocessing": ["tanh"], "scale_function": ["@tf.nn.softplus"],
                          "std_independent_of_input": ["True"], "learning_rate": ["0.0003", "0.001"]}),
        ("+lastlayer<=0.01", {"action_postprocessing": ["tanh"], "scale_function": ["@tf.nn.softplus"],
                              "std_independent_of_input": ["True"], "learning_rate": ["0.0003", "0.001"],
                              "last_kernel_init_policy_scaling": ["0.01", "0.001"]}),
        ("+tanh-act,depth2", {"action_postprocessing": ["tanh"], "scale_function": ["@tf.nn.softplus"],
                              "std_independent_of_input": ["True"], "learning_rate": ["0.0003", "0.001"],
                              "last_kernel_init_policy_scaling": ["0.01", "0.001"], "activation": ["@tf.nn.tanh"],
                              "policy_mlp_depth": ["2"]}),
    ]
    for arm_name, col, arm, rec in [("A2b_initstd_1", "initial_action_std", "1.0", "0.5"),
                                    ("A2c_initstd_0p1", "initial_action_std", "0.1", "0.5")]:
        for tag, f in REC_ARCH:
            for env in ENVS:
                r = compare(a2, col, arm, rec, f, env)
                res.append({"arm": arm_name, "exp": "final_arch2", "subset": tag, "env": env, **r})
    REC_ARCH_LL = [(t, {k: v for k, v in f.items() if k != "last_kernel_init_policy_scaling"}) for t, f in REC_ARCH]
    REC_ARCH_LL = [(t, dict(f, initial_action_std=["0.5", "1.0"]) if t not in ("none", "tanh") else f)
                   for t, f in REC_ARCH_LL]
    for tag, f in REC_ARCH_LL:
        for env in ENVS:
            r = compare(a2, "last_kernel_init_policy_scaling", "1.0", "0.01", f, env)
            res.append({"arm": "A2a_lastlayer_1", "exp": "final_arch2", "subset": tag, "env": env, **r})
    # group F / G / J cells: progressively REC-like subsets of the experiment's own sampled choices
    LR = ["0.0003", "0.001"]
    fs = frame("final_stability")
    F_REC = {"input_normalization": ["Avg (comp=False)"], "reward_normalization": ["Popart Avg (comp=False)"],
             "normalize_advantages": ["False"], "gradient_clipping": ["0.5", "1.0", "2.0"], "learning_rate": LR,
             "ppo_epsilon": ["0.2", "0.3"]}
    for arm_name, col, arm, rec in [("A1_obsnorm_none", "input_normalization", "None", "Avg (comp=False)"),
                                    ("A7_valuenorm_none", "reward_normalization", "None", "Popart Avg (comp=False)"),
                                    ("A8_advnorm_true", "normalize_advantages", "True", "False")]:
        others = {k: v for k, v in F_REC.items() if k != col}
        steps = [("none", {}), ("lr", {"learning_rate": LR})]
        acc = {"learning_rate": LR}
        for k in [k for k in others if k != "learning_rate"]:
            acc = dict(acc, **{k: others[k]})
            steps.append(("+" + k, dict(acc)))
        for tag, f in steps:
            for env in ENVS:
                res.append({"arm": arm_name, "exp": "final_stability", "subset": tag, "env": env,
                            **compare(fs, col, arm, rec, f, env)})
    fa = frame("final_advantages")
    G_REC = [("none", {}), ("GAE+lr", {"advantage_estimator": ["GAE"], "learning_rate": LR}),
             ("+MSE,no-vclip", {"advantage_estimator": ["GAE"], "learning_rate": LR, "value_loss": ["MSE"],
                                "ppo_style_value_clipping_epsilon": ["None"]})]
    for tag, f in G_REC:
        for env in ENVS:
            res.append({"arm": "A4_lambda_0p95", "exp": "final_advantages", "subset": tag, "env": env,
                        **compare(fa, "sub_advantage_estimator_gae_gae_lambda", "0.95", "0.9", f, env)})
    fo = frame("final_optimize")
    J_REC = [("none", {}), ("momentum0.9", {"sub_optimizer_adam_momentum": ["0.9"]}),
             ("+decay", {"sub_optimizer_adam_momentum": ["0.9"], "learning_rate_decay": ["1.0"]}),
             ("+eps<=1e-5", {"sub_optimizer_adam_momentum": ["0.9"], "learning_rate_decay": ["1.0"],
                             "sub_optimizer_adam_epsilon": ["1e-05", "1e-06", "1e-07", "1e-08", "1e-09"]})]
    for tag, f in J_REC:
        for env in ENVS:
            res.append({"arm": "A5a_lr_1em4", "exp": "final_optimize", "subset": tag, "env": env,
                        **compare(fo, "sub_optimizer_adam_learning_rate", "0.0001", "0.0003", f, env)})
    # A6: entropy penalty 1e-3 vs no regularisation (final_regularizer; other choices at that experiment's base)
    rg = frame("final_regularizer")
    rg_ent = rg.copy()
    pen = rg_ent[P + "sub_policy_regularization_penalty_regularization_penalty"].astype(str)
    coef = rg_ent[P + "sub_regularization_penalty_entropy_coefficient"].astype(str)
    reg = rg_ent[P + "policy_regularization"].astype(str)
    rg_ent["ent_arm"] = np.where(reg == q("no regularization"), "none",
                                 np.where((reg == q("penalty")) & (pen == q("entropy")), "ent_" + coef.str.strip("'"),
                                          "other"))
    for lr in [None, ["0.0003"], ["0.0003", "0.001"]]:
        for env in ENVS:
            s = rg_ent[rg_ent.env_name == env]
            if lr:
                s = s[s[P + "learning_rate"].astype(str).isin([q(v) for v in lr])]
            for c in ["1e-05", "3e-05", "0.0001", "0.0003", "0.001", "0.003"]:
                a = s[s.ent_arm == "ent_" + c]["score"].values.astype(float)
                r = s[s.ent_arm == "none"]["score"].values.astype(float)
                row = {"arm": "A6_entropy", "coef": c, "exp": "final_regularizer", "subset": f"lr={lr}", "env": env,
                       "n_arm": int(len(a)), "n_rec": int(len(r))}
                if len(a) >= 3 and len(r) >= 3:
                    row.update({"median_ratio": float(np.median(a) / np.median(r)),
                                "mean_ratio": float(a.mean() / r.mean()),
                                "q95_ratio": float(np.percentile(a, 95) / np.percentile(r, 95)),
                                "mw_p": float(mannwhitneyu(a, r).pvalue)})
                res.append(row)
    # the fixed (non-sampled) base of final_regularizer and final_arch2: transform / init std / std fn
    base = {}
    for ex in ["final_regularizer", "final_arch2", "final_stability", "final_advantages"]:
        df, _ = D[NAME[ex]]
        keep = {}
        for c in df.columns:
            if c.startswith("_gin.") and not c.startswith(P) and any(
                    k in c for k in ["postprocessor", "initial", "std_fn", "min_std", "Orthogonal.gain", "learning_rate",
                                     "lambda_", "GAE", "epsilon", "normaliz", "PopArt", "entropy"]):
                u = df[c].dropna().astype(str).unique()
                if len(u) == 1:
                    keep[c.replace("_gin.", "")] = u[0]
        base[ex] = keep
    with open(os.path.join(OUT, "paper_conditional.json"), "w") as f:
        json.dump({"rows": res, "fixed_base": base}, f, indent=1)
    for r in res:
        if r["env"] == "HalfCheetah-v1" or r["arm"] == "A6_entropy":
            print("[METRIC] cond " + " ".join(f"{k}={(round(v, 3) if isinstance(v, float) else v)}" for k, v in r.items()))
    for ex, k in base.items():
        print(f"[INFO] fixed base {ex}: {k}")


if __name__ == "__main__":
    main()

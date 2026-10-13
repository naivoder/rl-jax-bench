"""small-scale report: our runs (runs/small, runs/small_lstm) + CleanRL oracle curves -> runs/small/report.json and a
markdown table on stdout. Same summary function (src/eval.summarize) for ours and the oracle."""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src.eval import load_runs, summarize  # noqa: E402
from rl_jax_bench.stats import mean_se  # noqa: E402

OR = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official")), "oracle_runs")
ORACLE = OR + "/small"


def ours(root, env, arm):
    runs = load_runs(os.path.join(REPO, "runs", root), env, arm)
    if not runs:
        return None
    fw = [r["summary"]["final_window"] for r in runs.values()]
    auc = [r["summary"]["auc"] for r in runs.values()]
    pu = lambda k: [float(np.nanmean(r["per_update"][k])) for r in runs.values()]  # noqa: E731
    m, se = mean_se(fw)
    return dict(n=len(fw), final_window=m, se=se, per_seed=fw, auc=mean_se(auc)[0],
                k3=float(np.mean(pu("k3"))), k3_last_mb=float(np.mean(pu("k3_last"))), clipfrac=float(np.mean(pu("clipfrac"))),
                expl_var=float(np.mean(pu("explained_variance"))),
                ratio_dev_first_mb_max=float(max(np.max(r["per_update"]["ratio_dev_first_mb"]) for r in runs.values())),
                sts=[r["summary"]["steps_to_threshold"] for r in runs.values()],
                greedy=float(np.mean([r["summary"]["greedy_eval_mean"] for r in runs.values()])),
                total=runs[next(iter(runs))]["config"]["total_timesteps"])


def oracle(script, env, total, root=ORACLE):
    roots = root.split("+")
    fs = sorted(f for r in roots for f in glob.glob(os.path.join(r, f"{script}__{env}__*.json")))
    if not fs:
        return None
    fw, auc, k3, cf, ev = [], [], [], [], []
    for f in fs:
        d = json.load(open(f))
        e = np.array([[s, r, 1, 0] for s, r in d["episodes"]], float)
        sm = summarize(e, total, env)
        fw.append(sm["final_window"]); auc.append(sm["auc"])
        sc = d.get("scalars", {})
        k3.append(np.mean([v for _, v in sc.get("losses/approx_kl", [[0, np.nan]])]))
        cf.append(np.mean([v for _, v in sc.get("losses/clipfrac", [[0, np.nan]])]))
        ev.append(np.nanmean([v for _, v in sc.get("losses/explained_variance", [[0, np.nan]])]))
    m, se = mean_se(fw)
    return dict(n=len(fw), final_window=m, se=se, per_seed=fw, auc=mean_se(auc)[0], k3_last_mb=float(np.mean(k3)),
                clipfrac=float(np.mean(cf)), expl_var=float(np.mean(ev)))


def main():
    rep = {"ours": {}, "oracle": {}}
    rows = []
    for root, env, arms in [
        ("small", "CartPole-v1", ["separate", "shared", "advnorm_batch", "advnorm_ddof0", "no_vclip", "trunc_bootstrap"]),
        ("small", "Acrobot-v1", ["separate", "shared", "advnorm_batch", "advnorm_ddof0", "no_vclip", "trunc_bootstrap"]),
        ("small", "MountainCar-v0", ["separate", "shared", "advnorm_batch", "advnorm_ddof0", "no_vclip", "trunc_bootstrap"]),
        ("small", "Pendulum-v1", ["default", "no_rscale", "no_vclip", "gymnasium029", "trunc_bootstrap", "advnorm_batch"]),
        ("small_lstm", "CartPole-v1", ["lstm", "lstm_noreset", "lstm_zero_train", "ff_masked", "ff_full"]),
        ("small_lstm_v2", "CartPole-v1", ["lstm", "lstm_noreset", "lstm_zero_train", "ff_masked", "ff_full",
                                          "no_rscale", "gymnasium029"]),
        ("small", "brax-hopper", ["default"]),
    ]:
        for arm in arms:
            r = ours(root, env, arm)
            if r is None:
                continue
            key = f"{root}/{env}/{arm}"
            rep["ours"][key] = r
            rows.append(f"| ours | {env}{' (masked, ' + root + ')' if root.startswith('small_lstm') else ''} | {arm} | {r['n']} | "
                        f"{r['final_window']:.1f} ± {r['se']:.1f} | {r['auc']:.1f} | {r['k3_last_mb']:.2e} | "
                        f"{r['clipfrac']:.3f} | {r['expl_var']:.2f} | {r['ratio_dev_first_mb_max']:.1e} |")
    for script, env, total, root in [
            ("ppo", "CartPole-v1", 100000, ORACLE), ("cleanrl_ppo_shared", "CartPole-v1", 100000, ORACLE),
            ("ppo", "Acrobot-v1", 100000, ORACLE), ("cleanrl_ppo_shared", "Acrobot-v1", 100000, ORACLE),
            ("ppo", "Acrobot-v1", 100000, ORACLE + "+" + OR + "/small_acrobot_extra"),
            ("ppo_continuous_action", "Pendulum-v1", 200000, ORACLE),
            ("baselines_run_separate_networks", "CartPole-v1", 100000, OR + "/small_baselines"),
            ("baselines_run", "CartPole-v1", 100000, OR + "/small_baselines"),
            ("baselines_run_separate_networks", "Acrobot-v1", 100000, OR + "/small_baselines"),
            ("baselines_run", "Acrobot-v1", 100000, OR + "/small_baselines"),
            ("cleanrl_ppo_lstm_maskedcartpole", "CartPole-v1", 100000, OR + "/small_lstm"),
            ("cleanrl_ppo_lstm_maskedcartpole", "CartPole-v1", 300000, OR + "/small_lstm_rscale")]:
        r = oracle(script, env, total, root)
        if r is None:
            continue
        root = "+".join(os.path.basename(x) for x in root.split("+"))
        rep["oracle"][f"{root}/{script}/{env}"] = r
        rows.append(f"| {'baselines' if script.startswith('baselines') else 'CleanRL'} | {env} "
                    f"({root}, {total // 1000}k) | {script} | {r['n']} | {r['final_window']:.1f} ± {r['se']:.1f} | "
                    f"{r['auc']:.1f} | {r['k3_last_mb']:.2e} (last mb) | {r['clipfrac']:.3f} | {r['expl_var']:.2f} | – |")
    print("| impl | env | arm | seeds | final-window return (mean ± s.e.) | AUC | k3 | clipfrac | expl. var | max\\|ρ−1\\| first mb |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    json.dump(rep, open(os.path.join(REPO, "runs", "small", "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

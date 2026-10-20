"""Compare the CleanRL ddpg_continuous_action_jax @ fe8d8a0 reference runs (own venv, gymnasium Pendulum-v1, CPU)
with our DDPG (scripts/oracle_pendulum_ddpg.py, gymnax Pendulum-v1). Metric: mean of the last 10 training
episodic returns per seed (with exploration noise, as CleanRL logs them). Seeds are NOT paired (different env
RNG streams), so the comparison is Welch's t on seed means plus a bootstrap CI of the difference.

Agreement rule (declared before the pf2 / paper-hp CleanRL runs finished;
the pf1 numbers were already visible when it was written): ours AGREES with the reference if the Welch 95 % CI of
(ours - CleanRL) contains 0 OR |diff| <= 25 return units (~15 % of the CleanRL level), and every seed of both
reaches > -400 (Pendulum's 'swinging up' regime; random policy ~ -1200).
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

from rl_jax_bench.stats import boot_ci, mean_se  # noqa: E402

REF = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official")), "ref_runs")
OURS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "oracle", "pendulum")


def cleanrl_last10(tag):
    out = {}
    for p in sorted(glob.glob(f"{REF}/cleanrl_pend_{tag}_s*.log")):
        r = [float(x) for x in re.findall(r"episodic_return=\[(-?[\d.]+)\]", open(p).read())]
        out[int(re.search(r"_s(\d+)\.log", p).group(1))] = float(np.mean(r[-10:])) if len(r) >= 10 else float("nan")
    return out


def main():
    rows = []
    pairs = [("cleanrl hp, policy_frequency 1", "pf1", "cleanrl", "cleanrl_nets"),
             ("cleanrl hp, policy_frequency 1", "pf1", "cleanrl", "paper_nets"),
             ("cleanrl hp (pf 2 = CleanRL default) vs ours (pf 1)", "pf2", "cleanrl", "cleanrl_nets"),
             ("paper Table V hp (lr 2e-4, gamma 0.9, tau 0.01, batch 128)", "paperhp", "paper", "cleanrl_nets"),
             ("paper Table V hp (lr 2e-4, gamma 0.9, tau 0.01, batch 128)", "paperhp", "paper", "paper_nets")]
    for name, tag, hp, arm in pairs:
        ref = cleanrl_last10(tag)
        p = os.path.join(OURS, f"ours_pendulum_{hp}.json")
        ours = {int(k): v["last10"] for k, v in json.load(open(p))["results"][arm].items()}
        a, b = np.array(list(ours.values())), np.array(list(ref.values()))
        d = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        dof = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
        ci = [d - stats.t.ppf(0.975, dof) * se, d + stats.t.ppf(0.975, dof) * se]
        p_w = stats.ttest_ind(a, b, equal_var=False).pvalue
        agree = (ci[0] <= 0 <= ci[1] or abs(d) <= 25) and a.min() > -400 and b.min() > -400
        rows.append(dict(setting=name, ours_arm=arm, ours=mean_se(a), cleanrl=mean_se(b), ours_seeds=ours,
                         cleanrl_seeds=ref, diff=float(d), welch_ci95=[float(c) for c in ci], welch_p=float(p_w),
                         agree=bool(agree)))
        print(f"{name:62s} ours[{arm}] {a.mean():8.1f} +- {mean_se(a)[1]:5.1f} | CleanRL {b.mean():8.1f} +- "
              f"{mean_se(b)[1]:5.1f} | diff {d:+7.1f} CI [{ci[0]:+.1f}, {ci[1]:+.1f}] p={p_w:.2f} agree={agree}")
    json.dump(rows, open(os.path.join(OURS, "compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

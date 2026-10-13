"""Review checks of two flags, the LSTM L3 direction and Pendulum reward scaling, against the authors' code.
Reward scaling: CleanRL ppo_continuous_action.py with NormalizeReward + its clip removed ($OFFICIAL_ROOT/harness/review_ppo_continuous_action_norscale.py,
sha256 f32e2f0e...) vs the unmodified file (oracle_runs/full/pendulum_1m), and ours in gymnasium029 mode with/without
reward scaling. LSTM L3: ours lstm_noreset under gymnasium029 (runs/review_lstm_g029) vs ours gymnasium029 L3-on, and the
CleanRL L3 on/off pair (full-scale). numpy/scipy only."""
import glob, json, os, sys
import numpy as np
from scipy import stats as st
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src.eval import summarize  # noqa: E402
from rl_jax_bench.stats import mean_se, paired_compare  # noqa: E402
OR = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official")), "oracle_runs")

def orc(pat, total=1000000, env="Pendulum-v1"):
    out = {}
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f)); assert d["returncode"] == 0 and d["total_timesteps"] == total
        e = np.array([[s, r, 1, 0] for s, r in d["episodes"]], float)
        out[d["seed"]] = summarize(e, total, env)["final_window"]
    return out

def ours(tag, env, arm, seeds=range(1, 11)):
    return {s: json.load(open(f"{REPO}/runs/{tag}/{env}/{arm}/seed_{s}/metrics.json"))["summary"]["final_window"] for s in seeds}

on, off = orc(OR + "/full/pendulum_1m/ppo_continuous_action__*.json"), orc(OR + "/review/pendulum_1m_norscale/*.json")
s = sorted(on)
print("rscale CleanRL rscale on", mean_se(list(on.values())), "off", mean_se(list(off.values())))
c = paired_compare([on[i] for i in s], [off[i] for i in s]); print("  on-off paired", c["diff_mean"], c["diff_se"], c["t_p"], c["wins"])
print("  Welch", st.ttest_ind(list(on.values()), list(off.values()), equal_var=False))
a, b = ours("full", "Pendulum-v1", "gymnasium029"), ours("review_pend_g029", "Pendulum-v1", "no_rscale")
c = paired_compare([a[i] for i in range(1, 11)], [b[i] for i in range(1, 11)])
print("rscale ours gymnasium029 on", mean_se(list(a.values())), "off", mean_se(list(b.values())), "on-off", c["diff_mean"], c["diff_se"], c["t_p"], c["wins"])
print("  ours off vs CleanRL off", st.ttest_ind(list(b.values()), list(off.values()), equal_var=False))
i4 = json.load(open(f"{REPO}/runs/full/report.json"))["investigation"]["I-4 CleanRL LSTM L3 off - on"]
lon, loff = ours("full_lstm", "CartPole-v1", "gymnasium029"), ours("review_lstm_g029", "CartPole-v1", "lstm_noreset")
c = paired_compare([loff[i] for i in range(1, 11)], [lon[i] for i in range(1, 11)])
print("lstm_l3 ours gymnasium029 L3 off", mean_se(list(loff.values())), "on", mean_se(list(lon.values())), "off-on", c["diff_mean"], c["diff_se"], c["t_p"], c["wins"])
cd = np.array(i4["off"]) - np.array(i4["on"]); od = np.array([loff[i] - lon[i] for i in range(1, 11)])
print("  CleanRL off-on", cd.mean(), cd.std(ddof=1) / np.sqrt(len(cd)), "interaction Welch", st.ttest_ind(od, cd, equal_var=False))
print("  ours on vs CleanRL on", st.ttest_ind(list(lon.values()), i4["on"], equal_var=False))
print("  ours off vs CleanRL off", st.ttest_ind(list(loff.values()), i4["off"], equal_var=False))

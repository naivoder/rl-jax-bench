"""follow-up resolution read-outs (declared before launch; all
exploratory, raw p-values). lstm_recipe: the post's own LSTM recipe on the proxy; lstm_l3_oracle_n10: CleanRL L3 on/off at n=10;
baselines_pendulum: baselines ppo2 MuJoCo treatment on Pendulum-v0 vs our baselines-mode default arm.
Run: uv run python -m scripts.review.resolution_report"""
import json
import os

import numpy as np
from scipy import stats as st

from scripts.full.report import OR, REPO, oracle, ours, welch_equiv
from rl_jax_bench.stats import mean_se, paired_compare

R = os.path.join(OR, "resolution")
G3 = os.path.join(OR, "full")


def ms(x):
    m, se = mean_se(list(x))
    return [float(m), float(se), len(list(x))]


def by_seed(o):
    return dict(zip(o["seeds"], o["fw"]))


def paired(a, b):
    s = sorted(set(a) & set(b))
    c = paired_compare([a[i] for i in s], [b[i] for i in s])
    return dict(diff=float(c["diff_mean"]), se=float(c["diff_se"]), p=float(c["t_p"]), wins=c["wins"], n=len(s))


rep = {}
# ---- lstm_recipe: the post's LSTM recipe (eps 0.1, no reward normalization)
oa = ours("resolution_lstm", "CartPole-v1", "lstm_atari")
oe = ours("resolution_lstm", "CartPole-v1", "lstm_eps01")
ol = ours("full_lstm", "CartPole-v1", "lstm")
oc = oracle(os.path.join(R, "lstm_atari_500k", "*.json"), "CartPole-v1", 500000)
A, E, L = (dict(zip(x["seeds"], x["fw"])) for x in (oa, oe, ol))
rep["lstm_recipe"] = dict(
    ours_lstm_atari=ms(A.values()), ours_lstm_eps01=ms(E.values()), ours_lstm=ms(L.values()),
    cleanrl_lstm_atari=ms(oc["fw"]) if oc else None,
    ours_atari_vs_cleanrl_atari=welch_equiv(list(A.values()), oc["fw"]) if oc else None,
    lstm_atari_minus_lstm=paired(A, L), lstm_eps01_minus_lstm=paired(E, L),
    ratio_dev_max=dict(lstm_atari=oa["ratio_dev_max"], lstm_eps01=oe["ratio_dev_max"]),
    random_policy_floor="~22 (small-scale no_rscale 22.0 +- 0.1)")
# ----: CleanRL L3 on/off at n = 10
con = oracle(os.path.join(G3, "lstm_500k", "*.json") + "+" + os.path.join(R, "lstm_500k_s6_10", "*.json"),
             "CartPole-v1", 500000)
coff = oracle(os.path.join(G3, "lstm_noreset_500k", "*.json") + "+" + os.path.join(R, "lstm_noreset_500k_s6_10", "*.json"),
              "CartPole-v1", 500000)
Con, Coff = by_seed(con), by_seed(coff)
yon = dict(zip(*[ours("full_lstm", "CartPole-v1", "gymnasium029")[k] for k in ("seeds", "fw")]))
yoff = dict(zip(*[ours("review_lstm_g029", "CartPole-v1", "lstm_noreset")[k] for k in ("seeds", "fw")]))
bon = dict(zip(*[ours("full_lstm", "CartPole-v1", "lstm")[k] for k in ("seeds", "fw")]))
boff = dict(zip(*[ours("full_lstm", "CartPole-v1", "lstm_noreset")[k] for k in ("seeds", "fw")]))
cd = np.array([Coff[s] - Con[s] for s in sorted(set(Con) & set(Coff))])
yd = np.array([yoff[s] - yon[s] for s in sorted(yon)])
rep["lstm_l3_oracle_n10"] = dict(
    cleanrl_on=ms(Con.values()), cleanrl_off=ms(Coff.values()), cleanrl_off_minus_on=paired(Coff, Con),
    cleanrl_seeds_on=sorted(Con), cleanrl_seeds_off=sorted(Coff),
    ours_g029_on=ms(yon.values()), ours_g029_off=ms(yoff.values()), ours_g029_off_minus_on=paired(yoff, yon),
    ours_baselines_off_minus_on=paired(boff, bon),
    interaction_welch_p=float(st.ttest_ind(yd, cd, equal_var=False).pvalue),
    interaction_diff=float(yd.mean() - cd.mean()),
    ours_on_vs_cleanrl_on=welch_equiv(list(yon.values()), list(Con.values())),
    ours_off_vs_cleanrl_off=welch_equiv(list(yoff.values()), list(Coff.values())),
    cleanrl_on_first5_vs_last5=[ms([Con[s] for s in range(1, 6)]), ms([Con[s] for s in range(6, 11) if s in Con])])
# ----: baselines ppo2 MuJoCo treatment on Pendulum-v0 vs ours baselines-mode default (gymnax Pendulum-v1)
bl = oracle(os.path.join(R, "baselines_pendulum_1m", "*.json"), "Pendulum-v1", 1000000)
od = ours("full", "Pendulum-v1", "default")
cl = oracle(os.path.join(G3, "pendulum_1m", "ppo_continuous_action__*.json"), "Pendulum-v1", 1000000)
if bl:
    rep["baselines_pendulum"] = dict(baselines=ms(bl["fw"]), baselines_per_seed=bl["fw"], ours_default=ms(od["fw"]),
                      ours_vs_baselines=welch_equiv(od["fw"], bl["fw"]), cleanrl=ms(cl["fw"]),
                      baselines_vs_cleanrl=welch_equiv(bl["fw"], cl["fw"]),
                      wall_s=bl["wall_s"], note="baselines on gym 0.15.7 Pendulum-v0; ours on gymnax Pendulum-v1")
out = os.path.join(REPO, "runs", "resolution_lstm", "resolution_report.json")
json.dump(rep, open(out, "w"), indent=1, default=float)
print(json.dumps(rep, indent=1, default=float))
# ----: TF1 Adam epsilon on Pendulum, paired with full-scale default
at = ours("resolution_adam", "Pendulum-v1", "adam_tf1")
if at and bl:
    AT, OD = dict(zip(at["seeds"], at["fw"])), dict(zip(od["seeds"], od["fw"]))
    rep["I5"] = dict(adam_tf1=ms(AT.values()), adam_tf1_minus_default=paired(AT, OD),
                     adam_tf1_vs_baselines=welch_equiv(list(AT.values()), bl["fw"]))
    json.dump(rep, open(out, "w"), indent=1, default=float)
    print("I5", json.dumps(rep["I5"], default=float))

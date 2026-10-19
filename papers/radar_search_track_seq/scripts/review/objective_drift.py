"""review: is the learners' movement LEARNING (toward their own objective) or drift? Validation-curve change of the
optimised return and of M_cov (step 0 -> end), the final-eval return vs Random, and the pre-clip gradient norm vs
max_grad_norm 0.5 (how much of each update survives the global-norm clip)."""
import json, glob, numpy as np
from scipy import stats as st
G = "runs/full"
b = json.load(open(G + "/baselines.json"))
R = np.array([np.mean(b["arms"]["random"][s]["per_episode"]["ret_flip"]) for s in "01234"])
for a in ["flat", "bigru", "mhsa", "mhsa_nobc", "mhsa_noae", "mhsa_lit"]:
    ms = [json.load(open(f)) for f in sorted(glob.glob(f"{G}/{a}/seed_*/metrics.json"))]
    obj = "ret_lit" if a == "mhsa_lit" else "ret_flip"          # each arm's OWN optimised return
    d = np.array([m["curve"][-1][obj] - m["curve"][0][obj] for m in ms])
    dm = np.array([m["curve"][-1]["m_cov"] - m["curve"][0]["m_cov"] for m in ms])
    x = np.array([m["eval_stochastic"]["ret_flip"] for m in ms])
    g = np.array([m["train_hist"]["gnorm"] for m in ms])
    print(f"{a}: d(own return, validation) {d.mean():+.0f} ± {d.std(ddof=1)/np.sqrt(5):.0f} (neg in {(d<0).sum()}/5, p2 {st.ttest_1samp(d,0).pvalue:.3f}) | "
          f"d(M_cov, validation) {dm.mean():+.1f} ± {dm.std(ddof=1)/np.sqrt(5):.1f} (p2 {st.ttest_1samp(dm,0).pvalue:.2f}) | "
          f"eval ret_flip {x.mean():.0f} vs Random {R.mean():.0f} (Welch p2 {st.ttest_ind(x,R,equal_var=False).pvalue:.3f}) | "
          f"pre-clip gnorm it1 {g[:,0].mean():.0f} -> it7 {g[:,-1].mean():.0f}; surviving fraction 0.5/gnorm {np.mean(0.5/g):.1e}")

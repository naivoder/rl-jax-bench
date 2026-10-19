import json, glob, itertools, numpy as np
from scipy import stats as st
G="runs/full"
L={a:sorted([json.load(open(f)) for f in glob.glob(f"{G}/{a}/seed_*/metrics.json")],key=lambda m:m["seed"]) for a in ["flat","bigru","mhsa","mhsa_nobc","mhsa_noae","mhsa_lit"]}
base=json.load(open(G+"/baselines.json"))
print("== CRN check: are pre-training streams shared across ablation arms at equal seed? (bc_curve equality)")
for s in range(2):
    a=L["mhsa"][s]["pretrain"]; b=L["mhsa_nobc"][s]["pretrain"]; c=L["mhsa_noae"][s]["pretrain"]
    print(s, "ae_final mhsa vs nobc", a["ae_final"], b["ae_final"], "| bc_curve[0] mhsa/nobc/noae", a["bc_curve"][0], b["bc_curve"][0], c["bc_curve"][0])
    print("   first train_hist rew mhsa vs nobc vs noae", L["mhsa"][s]["train_hist"]["rew"][0], L["mhsa_nobc"][s]["train_hist"]["rew"][0], L["mhsa_noae"][s]["train_hist"]["rew"][0])
print("== episode selection in M_cov (episodes with no track excluded) and per-track covariance")
for a,ms in L.items():
    print(a, "no-track eps/seed", [m["eval_stochastic"]["n_episodes_no_track"] for m in ms], "cov_per_track %.1f"%np.mean([m["eval_stochastic"]["cov_per_track"] for m in ms]), "frac_steps_tracked %.4f"%np.mean([m["eval_stochastic"]["frac_steps_tracked"] for m in ms]))
for a in ["random","coverage","static","hand","ceiling"]:
    d=base["arms"][a]; print(a, "no-track eps", [d[s]["n_episodes_no_track"] for s in sorted(d,key=int)], "cov_per_track %.1f"%np.mean([d[s]["cov_per_track"] for s in d]), "frac_steps_tracked %.4f"%np.mean([d[s]["frac_steps_tracked"] for s in d]))
print("== MHSA deterministic collapse: det sr / central1 per seed")
for m in L["mhsa"]:
    print(m["seed"], m["eval_deterministic"]["sr"], m["eval_deterministic"]["central1"], m["eval_deterministic"]["n_init"], m["eval_deterministic"]["mean_tracks"])
print("== pairing fragility: two-sided paired-t p over ALL 120 seed pairings, and Welch")
for h,(ref,k) in {"H1":("flat","m_cov"),"H2":("bigru","m_cov"),"H4":("random","gospa_d"),"H5":("flat","gospa_miss"),"H6":("bigru","gospa_miss")}.items():
    x=np.array([m["eval_stochastic"][k] for m in L["mhsa"]])
    y=np.array([m["eval_stochastic"][k] for m in L[ref]]) if ref in L else np.array([base["arms"]["random"][str(s)][k] for s in range(5)])
    ps=np.array([st.ttest_rel(x,y[list(p)]).pvalue for p in itertools.permutations(range(5))])
    w=st.ttest_ind(x,y,equal_var=False)
    pid=st.ttest_rel(x,y).pvalue
    print(h, f"index-paired p2 {pid:.3f} | over 120 pairings: median p2 {np.median(ps):.3f}, frac p2<0.05 {np.mean(ps<0.05):.2f} | Welch p2 {w.pvalue:.3f} one-sided-less {w.pvalue/2 if w.statistic<0 else 1-w.pvalue/2:.3f}")
print("== GOSPA d ratio grading power: what grade would Random/Static/Coverage get against the MHSA band?")
def grade(v,band):
    a,b,lo,hi=band
    if lo<=v<=hi: return "MATCH"
    near=min((a,b),key=lambda p:abs(v-p)); rel=abs(v-near)/abs(near)
    return ("MATCH" if rel<=.05 else "CLOSE" if rel<=.2 else "DIVERGENT")+f" ({rel*100:.1f}%)"
band=(0.890,0.919,0.871,0.938)
for nm,v in [("random itself",1.0),("static",1.0337),("coverage",1.0012),("flat",0.9964),("mhsa",1.0086)]: print(nm, v, grade(v,band))
band=(0.96,1.04,0.96,1.04)

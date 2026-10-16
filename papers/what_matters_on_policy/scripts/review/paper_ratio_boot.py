import pickle,numpy as np,json
P="_gin.study_design.choice_value_"
import os  # noqa: E402
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' checkout, see README
D=pickle.load(open(os.path.join(OFFICIAL_ROOT, "data", "mujoco123.pkl"),"rb"))
NAME={m["title"].split(": ")[1].split(" ")[0]: k for k,(df,m) in D.items()}
MAP={
 "A1":(("final_stability","input_normalization","None"),("final_stability","input_normalization","Avg (comp=False)")),
 "A2a":(("final_arch2","last_kernel_init_policy_scaling","1.0"),("final_arch2","last_kernel_init_policy_scaling","0.01")),
 "A2b":(("final_arch2","initial_action_std","1.0"),("final_arch2","initial_action_std","0.5")),
 "A3a":(("final_arch","sub_policy_and_value_function_network_shared_baseline_cost","1.0"),("final_arch","policy_and_value_function_network","separate")),
 "A4":(("final_advantages","sub_advantage_estimator_gae_gae_lambda","0.95"),("final_advantages","sub_advantage_estimator_gae_gae_lambda","0.9")),
 "A5a":(("final_optimize","sub_optimizer_adam_learning_rate","0.0001"),("final_optimize","sub_optimizer_adam_learning_rate","0.0003")),
 "A6":(("final_regularizer","sub_regularization_penalty_entropy_coefficient","0.001"),("final_regularizer","policy_regularization","no regularization")),
 "A7":(("final_stability","reward_normalization","None"),("final_stability","reward_normalization","Popart Avg (comp=False)")),
 "A8":(("final_stability","normalize_advantages","True"),("final_stability","normalize_advantages","False"))}
ours={"A1":0.562,"A2a":0.984,"A2b":1.164,"A3a":0.901,"A4":1.009,"A5a":1.038,"A6":0.984,"A7":0.965,"A8":1.019}
rng=np.random.default_rng(1)
def get(ex,col,val):
    df,_=D[NAME[ex]]; s=df[df.env_name=='HalfCheetah-v1']
    v=s[s[P+col].astype(str)==f"'{val}'"]['_eval_'].map(lambda d:d[max(d)]).values.astype(float); return v,s
for k,((ea,ca,va),(er,cr,vr)) in MAP.items():
    a,s=get(ea,ca,va); r,_=get(er,cr,vr)
    rat=np.percentile(a,95)/np.percentile(r,95)
    bs=[np.percentile(a[rng.integers(0,len(a),len(a))],95)/np.percentile(r[rng.integers(0,len(r),len(r))],95) for _ in range(3000)]
    lo,hi=np.percentile(bs,[2.5,97.5])
    extra=''
    if k=='A6':
        pen=s[P+'sub_policy_regularization_penalty_regularization_penalty'].astype(str); coef=s[P+'sub_regularization_penalty_entropy_coefficient'].astype(str); reg=s[P+'policy_regularization'].astype(str)
        a2=s[(reg=="'penalty'")&(pen=="'entropy'")&(coef=="'0.001'")]['_eval_'].map(lambda d:d[max(d)]).values
        extra=f' [A6 entropy-penalty-only n={len(a2)} q95ratio={np.percentile(a2,95)/np.percentile(r,95):.3f}]'
    print(f"{k:4s} n={len(a)}/{len(r)} q95ratio={rat:.3f} boot95=[{lo:.3f},{hi:.3f}] ours={ours[k]} inside={lo<=ours[k]<=hi}{extra}")

import json,pickle,numpy as np
J=json.load(open('runs/p95/paper_conditional.json'))
for r in J['rows']:
    if r['env']=='HalfCheetah-v1' and r['arm'] in ('A7_valuenorm_none','A8_advnorm_true','A4_lambda_0p95','A5a_lr_1em4','A2b_initstd_1','A1_obsnorm_none') or (r['arm']=='A6_entropy' and r['env']=='HalfCheetah-v1' and r.get('coef')=='0.001'):
        print(r['arm'][:6],r['subset'],r.get('coef',''),r['n_arm'],r['n_rec'],'q95',round(r.get('q95_ratio',np.nan),3),'med',round(r.get('median_ratio',np.nan),3))
# bootstrap CI for q95 ratio on the reported 'resolving' subsets
P="_gin.study_design.choice_value_"
import os  # noqa: E402
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' checkout, see README
D=pickle.load(open(os.path.join(OFFICIAL_ROOT, "data", "mujoco123.pkl"),"rb"))
NAME={m["title"].split(": ")[1].split(" ")[0]: k for k,(df,m) in D.items()}
def frame(ex):
    df,_=D[NAME[ex]]; df=df.copy(); df['score']=df['_eval_'].map(lambda d:d[max(d)]); return df[df.env_name=='HalfCheetah-v1']
q=lambda v:f"'{v}'"
rng=np.random.default_rng(0)
def boot(a,r,B=10000):
    out=[]
    for _ in range(B):
        aa=a[rng.integers(0,len(a),len(a))]; rr=r[rng.integers(0,len(r),len(r))]
        out.append(np.percentile(aa,95)/np.percentile(rr,95))
    return np.percentile(out,[2.5,97.5])
def sub(df,filters,col,arm,rec):
    s=df
    for c,v in filters.items(): s=s[s[P+c].astype(str).isin([q(x) for x in v])]
    return s[s[P+col].astype(str)==q(arm)]['score'].values.astype(float), s[s[P+col].astype(str)==q(rec)]['score'].values.astype(float)
fs=frame('final_stability'); LR=["0.0003","0.001"]
F={"learning_rate":LR,"input_normalization":["Avg (comp=False)"],"normalize_advantages":["False"],"gradient_clipping":["0.5","1.0","2.0"],"ppo_epsilon":["0.2","0.3"]}
a,r=sub(fs,F,'reward_normalization','None','Popart Avg (comp=False)'); print('A7 REC-like n',len(a),len(r),'q95 ratio',np.percentile(a,95)/np.percentile(r,95),'boot95',boot(a,r))
a,r=sub(fs,{},'reward_normalization','None','Popart Avg (comp=False)'); print('A7 all n',len(a),len(r),'q95 ratio',np.percentile(a,95)/np.percentile(r,95),'boot95',boot(a,r,2000))
rg=frame('final_regularizer')
reg=rg[P+'policy_regularization'].astype(str); pen=rg[P+'sub_policy_regularization_penalty_regularization_penalty'].astype(str); coef=rg[P+'sub_regularization_penalty_entropy_coefficient'].astype(str)
for lr in [None,["0.0003"],["0.0003","0.001"]]:
    m=np.ones(len(rg),bool) if lr is None else rg[P+'learning_rate'].astype(str).isin([q(v) for v in lr]).values
    a=rg[m&(reg==q('penalty')).values&(pen==q('entropy')).values&(coef==q('0.001')).values]['score'].values.astype(float)
    r=rg[m&(reg==q('no regularization')).values]['score'].values.astype(float)
    if len(a)>=3: print('A6 lr',lr,'n',len(a),len(r),'q95',np.percentile(a,95)/np.percentile(r,95),'boot95',boot(a,r,4000))
fa=frame('final_advantages')
a,r=sub(fa,{"advantage_estimator":["GAE"],"learning_rate":LR,"value_loss":["MSE"],"ppo_style_value_clipping_epsilon":["None"]},'sub_advantage_estimator_gae_gae_lambda','0.95','0.9')
print('A4 REC-like n',len(a),len(r),np.percentile(a,95)/np.percentile(r,95),boot(a,r))
fs2=fs
a,r=sub(fs,{k:v for k,v in F.items() if k!='normalize_advantages'}|{"reward_normalization":["Popart Avg (comp=False)"]},'normalize_advantages','True','False')
print('A8 REC-like n',len(a),len(r),np.percentile(a,95)/np.percentile(r,95),boot(a,r))

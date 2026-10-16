import pickle, numpy as np, pandas as pd
from scipy.stats import binom
import os  # noqa: E402
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' checkout, see README
D=pickle.load(open(os.path.join(OFFICIAL_ROOT, "data", "mujoco123.pkl"),'rb'))
name={m['title'].split(': ')[1].split(' ')[0]:k for k,(df,m) in D.items()}
ENVS=['Hopper-v1','Humanoid-v1','Walker2d-v1','HalfCheetah-v1','Ant-v1']
def sc(d): return d[max(d)]
def p95(x):
    x=np.sort(np.asarray(x)); n=len(x); lo,hi=binom.interval(0.95,n,0.95); hi=min(int(hi),n-1)
    return np.percentile(x,95), x[int(lo)], x[hi], n
def table(ex, col, order=None):
    df,_=D[name[ex]]; df=df.copy(); df['score']=df['_eval_'].map(sc)
    vals=order or sorted(df[col].dropna().astype(str).unique())
    print(f'\n## {ex} :: {col.replace("_gin.study_design.choice_value_","")}')
    for e in ENVS:
        s=df[df.env_name==e]
        row=[]
        for v in vals:
            x=s[s[col].astype(str)==v]['score'].values
            if len(x)==0: row.append(f'{v}:NA'); continue
            p,l,h,n=p95(x); row.append(f'{v}:{p:.0f}[{l:.0f},{h:.0f}]n{n}')
        print(f'  {e:15s} '+'  '.join(row))
def cols(ex):
    df,_=D[name[ex]]; return [c for c in df.columns if 'choice_value' in c]
for ex in ['final_stability','final_arch','final_arch2','final_advantages','final_optimize','final_regularizer']:
    print('\nCOLUMNS', ex, [c.replace('_gin.study_design.choice_value_','') for c in cols(ex)])
P='_gin.study_design.choice_value_'
table('final_stability',P+'input_normalization')
table('final_stability',P+'reward_normalization')
table('final_stability',P+'normalize_advantages')
table('final_stability',P+'gradient_clipping')
table('final_stability',P+'learning_rate')
table('final_arch2',P+'last_kernel_init_policy_scaling')
table('final_arch2',P+'initial_action_std')
table('final_arch',P+'policy_and_value_function_network')
table('final_arch',P+'sub_policy_and_value_function_network_shared_baseline_cost')
table('final_advantages',P+'sub_advantage_estimator_gae_gae_lambda')
table('final_optimize',P+'sub_optimizer_adam_learning_rate')
table('final_regularizer',P+'sub_regularization_penalty_entropy_coefficient')
table('final_regularizer',P+'policy_regularization')
# Table 5 check
df,_=D[name['final_stability']]; df=df.copy(); df['score']=df['_eval_'].map(sc)
print('\nTable5 recompute:')
for e in ENVS:
    x=df[df.env_name==e]['score'].values; print(e, len(x), [round(np.percentile(x,q)) for q in (90,95,99)], round(x.max()))

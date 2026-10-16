import pickle,numpy as np
P="_gin.study_design.choice_value_"
import os  # noqa: E402
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' checkout, see README
D=pickle.load(open(os.path.join(OFFICIAL_ROOT, "data", "mujoco123.pkl"),"rb"))
NAME={m["title"].split(": ")[1].split(" ")[0]: k for k,(df,m) in D.items()}
df,_=D[NAME['final_arch2']]; s=df[df.env_name=='HalfCheetah-v1'].copy(); s['score']=s['_eval_'].map(lambda d:d[max(d)])
f={"action_postprocessing":["tanh"],"scale_function":["@tf.nn.softplus"],"std_independent_of_input":["True"],"learning_rate":["0.0003","0.001"],"last_kernel_init_policy_scaling":["0.01","0.001"]}
for c,v in f.items(): s=s[s[P+c].astype(str).isin([f"'{x}'" for x in v])]
a=s[s[P+'initial_action_std'].astype(str)=="'1.0'"].score.values.astype(float); r=s[s[P+'initial_action_std'].astype(str)=="'0.5'"].score.values.astype(float)
rng=np.random.default_rng(0)
bs=[np.percentile(a[rng.integers(0,len(a),len(a))],95)/np.percentile(r[rng.integers(0,len(r),len(r))],95) for _ in range(10000)]
print(len(a),len(r),np.percentile(a,95)/np.percentile(r,95),np.percentile(bs,[2.5,97.5]), 'frac>=1.164', np.mean(np.array(bs)>=1.164))
print(np.sort(r).round(0)); print(np.sort(a).round(0))

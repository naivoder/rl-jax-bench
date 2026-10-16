import pickle,numpy as np
import os  # noqa: E402
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' checkout, see README
D=pickle.load(open(os.path.join(OFFICIAL_ROOT, "data", "mujoco123.pkl"),"rb"))
name = {m["title"].split(": ")[1].split(" ")[0]: k for k, (df, m) in D.items()}
df,_=D[name["final_time"]]
print([c for c in df.columns if 'eval' in c.lower()][:20])
for i in range(3):
    r=df.iloc[i]; e=r['_eval_']; o=r['_eval_original_avg']
    ks=sorted(e); print(r['env_name'], [round(e[k],1) for k in ks][:12]); ko=sorted(o); ov=[o[k] for k in ko]; print('  orig',[round(v,1) for v in ov]); print('  mean(orig[1:])',np.mean(ov[1:]), 'mean all',np.mean(ov))

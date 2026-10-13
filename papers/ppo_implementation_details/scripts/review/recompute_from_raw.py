import json, glob, os, numpy as np
from scipy import stats as st
R=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs")
def fw_raw(p):
    d=json.load(open(p)); e=np.array(d["episodes"],float); T=d["config"]["total_timesteps"]
    w=e[:,0]>0.9*T
    v=e[w,1].sum()/e[w,2].sum()
    assert abs(v-d["summary"]["final_window"])<1e-9,(p,v,d["summary"]["final_window"])
    return d["seed"],v,d
def arm(tag,env,a):
    out={}
    for p in sorted(glob.glob(f"{R}/{tag}/{env}/{a}/seed_*/metrics.json")):
        s,v,d=fw_raw(p); out[s]=v
    return out
def ms(x): x=np.array(list(x)); return f"{x.mean():.1f} ± {x.std(ddof=1)/np.sqrt(len(x)):.1f} (n={len(x)})"
rows=[("full","CartPole-v1",a) for a in ["separate","shared","advnorm_batch","advnorm_ddof0","no_vclip","trunc_bootstrap"]]
rows+=[("full","Acrobot-v1",a) for a in ["separate","shared","advnorm_batch","advnorm_ddof0","no_vclip","trunc_bootstrap"]]
rows+=[("full","Pendulum-v1",a) for a in ["default","gym021","gymnasium029","trunc_bootstrap","no_rscale","no_vclip","advnorm_batch"]]
rows+=[("full_lstm","CartPole-v1",a) for a in ["lstm","gymnasium029","lstm_noreset","ff_masked","ff_full","lstm_zero_train"]]
A={}
for r in rows:
    A[r]=arm(*r); print(r, ms(A[r].values()), sorted(A[r]))
def pt(a,b):
    s=sorted(set(a)&set(b)); x=np.array([a[i] for i in s]); y=np.array([b[i] for i in s]); d=x-y
    t=st.ttest_rel(x,y); return d.mean(), d.std(ddof=1)/np.sqrt(len(d)), t.pvalue, int((d>0).sum()), len(s)
print("QB cart", pt(A[("full","CartPole-v1","separate")],A[("full","CartPole-v1","shared")]))
print("QB acro", pt(A[("full","Acrobot-v1","separate")],A[("full","Acrobot-v1","shared")]))
AB=[(("full","CartPole-v1","separate"),("full","CartPole-v1","advnorm_batch")),
(("full","Acrobot-v1","separate"),("full","Acrobot-v1","advnorm_batch")),
(("full","CartPole-v1","separate"),("full","CartPole-v1","no_vclip")),
(("full","Acrobot-v1","separate"),("full","Acrobot-v1","no_vclip")),
(("full","Pendulum-v1","default"),("full","Pendulum-v1","no_vclip")),
(("full","Pendulum-v1","default"),("full","Pendulum-v1","no_rscale")),
(("full","CartPole-v1","trunc_bootstrap"),("full","CartPole-v1","separate")),
(("full","Pendulum-v1","trunc_bootstrap"),("full","Pendulum-v1","default")),
(("full_lstm","CartPole-v1","lstm"),("full_lstm","CartPole-v1","lstm_noreset"))]
ps=[]
for i,(a,b) in enumerate(AB):
    r=pt(A[a],A[b]); ps.append(r[2]); print("AB%d"%(i+1), ["%.4g"%v for v in r])
o=np.argsort(ps); m=len(ps); run=0; h={}
for k,i in enumerate(o): run=max(run,min(1,(m-k)*ps[i])); h[i]=run
print("holm", {f"AB{i+1}":round(h[i],5) for i in range(m)})
# 50 seeds
for a in ["separate","shared"]:
    x=list(A[("full","Acrobot-v1",a)].values())+list(arm("full_manyseeds","Acrobot-v1",a).values())
    print("50",a,ms(x),"median",np.median(x),"stuck",sum(np.array(x)<-400))
# per-seed collection consistency
bad=0
for p in sorted(glob.glob(f"{R}/seed_*/metrics.json")):
    d=json.load(open(p))
    print(os.path.basename(os.path.dirname(p)), type(d).__name__, list(d)[:5] if isinstance(d,dict) else len(d)); break

"""Independent recompute: every headline number, paired test and Holm value from raw runs/*/metrics.json and the
authors' progress.csv, with scipy only (independent of src/eval.py and rl_jax_bench.stats).  Usage: python scripts/review_recompute.py"""
import json, glob, os, numpy as np, csv
from scipy import stats as st
R=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
def load(d, v):
    out={}
    for f in sorted(glob.glob(f"{R}/{d}/{v}/seed_*/metrics.json")):
        m=json.load(open(f)); 
        steps=np.array(m["eval_env_steps"]); rets=np.array(m["eval_returns"])
        T=m["total_env_steps_run"]
        sel=steps>=0.8*T
        fm=rets[sel].mean()
        fk=m.get("final_metric_key","final_last20"); assert abs(fm-m[fk])<1e-6, (f, fm, m[fk])
        out[m["seed"]]=dict(fm=fm, n=sel.sum(), lo=steps[sel].min(), hi=steps[sel].max(), T=T, nev=len(steps),
                            steps_unique=len(set(steps.tolist()))==len(steps), sha=m.get("git_sha"), dirty=m.get("git_dirty"), m=m)
    return out
def ms(x): x=np.asarray(x,float); return x.mean(), x.std(ddof=1)/np.sqrt(len(x)), x.std(ddof=1)
cells={"P1sep":("full/P1_sac_separate","separate"),"P1off":("full/P1_sac_shared","official"),"P1det":("full/P1_sac_shared","detach"),
"P3sep":("small/P_td3_separate","separate"),"P3off":("small/P_td3_shared","official"),
"P5sep":("small/S_td3_separate","separate"),"P5off":("small/S_td3_shared","official"),
"G2sacsep":("small/P_sac_separate","separate"),"G2sacoff":("small/P_sac_shared","official"),"G2sacnl":("small/P_sac_shared","noleak"),"G2sacdet":("small/P_sac_shared","detach"),
"L5sep":("full/L5_sac_separate","separate"),"L5off":("full/L5_sac_shared","official")}
D={}
for k,(d,v) in cells.items():
    o=load(d,v); D[k]=o
    vals=[o[s]["fm"] for s in sorted(o)]
    m,se,sd=ms(vals)
    s0=o[sorted(o)[0]]
    print(f"{k:9s} n={len(vals)} mean={m:9.2f} se={se:7.2f} sd={sd:7.2f} window=[{s0['lo']},{s0['hi']}] nwin={s0['n']} T={s0['T']} nev={s0['nev']} uniq={s0['steps_unique']} sha={str(s0['sha'])[:10]} dirty={s0['dirty']} per={np.round(vals,1).tolist()}")
def paired(a,b,name):
    S=sorted(set(D[a])&set(D[b])); x=np.array([D[a][s]["fm"] for s in S]); y=np.array([D[b][s]["fm"] for s in S]); d=x-y
    t=st.ttest_rel(x,y); p1=t.pvalue/2 if t.statistic>0 else 1-t.pvalue/2
    ci=st.t.interval(0.95,len(d)-1,loc=d.mean(),scale=d.std(ddof=1)/np.sqrt(len(d)))
    print(f"{name}: seeds {S} diff {d.mean():.1f} ± {d.std(ddof=1)/np.sqrt(len(d)):.1f} t={t.statistic:.2f} p1={p1:.4g} t95=[{ci[0]:.1f},{ci[1]:.1f}] per={np.round(d,1).tolist()}")
    return p1
p={}
p["P1"]=paired("P1sep","P1off","P1"); p["P3"]=paired("P3sep","P3off","P3"); p["P5"]=paired("P5sep","P5off","P5")
paired("P1det","P1off","E6 det-off"); paired("P1det","P1sep","E6 det-sep"); paired("G2sacnl","G2sacoff","noleak-off20k")
paired("P1sep","L5sep","Len sep200-sep5"); paired("P1off","L5off","Len sh200-sh5"); paired("L5sep","L5off","sepL5-shL5")
def holm(p):
    ks=sorted(p,key=p.get); m=len(ks); out={}; run=0
    for i,k in enumerate(ks): run=max(run,min(1,(m-i)*p[k])); out[k]=run
    return out
print("holm3",holm(p)); print("holm5",holm({**p,"P2":1,"P4":1}))
# authors logs
AL=os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official")), "numeric_results", "extracted", "logs_txt")
def alog(g, window=None):
    out=[]
    for f in sorted(glob.glob(os.path.join(AL,g))):
        rows=[r for r in csv.DictReader(open(f)) if r.get("metrics/return_eval_total") not in (None,"")]
        s=np.array([float(r["z/env_steps"]) for r in rows]); v=np.array([float(r["metrics/return_eval_total"]) for r in rows])
        if window: sel=(s>=window[0])&(s<=window[1])
        else: sel=s>=0.8*s.max()
        out.append((v[sel].mean(), int(s.max()), int(sel.sum()), s[sel].min()))
    return out
for g,w in [("pomdp/Pendulum/V/sac_lstm/len--1/*/*/*/oar/*/progress.csv",None),("pomdp/Pendulum/V/sac_lstm_shared/len--1/*/*/*/oar/*/progress.csv",None),
            ("pomdp/Pendulum/V/sac_lstm/len--1/*/*/*/oar/*/progress.csv",(16000,20000)),("pomdp/Pendulum/V/sac_lstm_shared/len--1/*/*/*/oar/*/progress.csv",(16000,20000)),
            ("pomdp/Pendulum/V/td3_lstm/len--1/*/*/*/oar/*/progress.csv",(16000,20000)),("pomdp/Pendulum/V/td3_lstm_shared/len--1/*/*/*/oar/*/progress.csv",(16000,20000)),
            ("meta/PointRobotSparse-v0/td3_lstm/len-64/*/*/*/or/*/progress.csv",(120000,150000)),("meta/PointRobotSparse-v0/td3_lstm_shared/len-64/*/*/*/or/*/progress.csv",(120000,150000)),
            ("pomdp/Pendulum/V/sac_lstm/len-5/*/*/*/oar/*/progress.csv",None),("pomdp/Pendulum/V/sac_lstm_shared/len-5/*/*/*/oar/*/progress.csv",None)]:
    o=alog(g,w); v=[x[0] for x in o]; m,se,sd=ms(v)
    print(g.split('/')[2:5], w, f"n={len(v)} mean={m:.1f} se={se:.1f} sd={sd:.1f} maxstep={[x[1] for x in o]} nwin={[x[2] for x in o]} lo={[x[3] for x in o]}")

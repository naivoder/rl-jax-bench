import json, glob, os, numpy as np
from scipy import stats as st
R="runs"
G=R+"/full"
base=json.load(open(G+"/baselines.json"))
rep=json.load(open(G+"/report.json"))
def pe_mean(pe,k):
    v=np.asarray(pe[k],float); v=v[np.isfinite(v)]; return v.mean() if v.size else np.nan
vals={}
for arm in ["flat","bigru","mhsa","mhsa_nobc","mhsa_noae","mhsa_lit"]:
    ms=sorted([json.load(open(f)) for f in glob.glob(f"{G}/{arm}/seed_*/metrics.json")],key=lambda m:m["seed"])
    assert [m["seed"] for m in ms]==[0,1,2,3,4],arm
    vals[arm]={}
    for k in ["m_cov","sr","gospa_d","gospa_miss","n_init","mean_tracks"]:
        a=np.array([pe_mean(m["per_episode"],k) for m in ms]); b=np.array([m["eval_stochastic"][k] for m in ms])
        assert np.allclose(a,b,rtol=1e-12,equal_nan=True),(arm,k,a,b)
        vals[arm][k]=a
    # scenario identity, env version
    print(arm, {m["eval_scenarios"]["keys_sha256"][:8] for m in ms}, {m["env_version"]["data_py_sha256"][:8] for m in ms}, {m["env_steps"] for m in ms}, {m["device"] for m in ms}, [len(m["per_episode"]["m_cov"]) for m in ms])
for arm,d in base["arms"].items():
    seeds=sorted(d,key=int)
    vals[arm]={}
    for k in ["m_cov","sr","gospa_d","gospa_miss","n_init","mean_tracks"]:
        a=np.array([pe_mean(d[s]["per_episode"],k) for s in seeds]); b=np.array([d[s][k] for s in seeds])
        assert np.allclose(a,b,rtol=1e-12,equal_nan=True),(arm,k)
        vals[arm][k]=a if len(seeds)>1 else np.repeat(a,5)
print("base keys sha", base["eval_scenarios"]["keys_sha256"][:8])
rnd=vals["random"]
for k in ["m_cov","sr","gospa_d","gospa_miss"]:
    for a in ["flat","bigru","mhsa","coverage","static","hand","ceiling"]:
        r=vals[a][k]/rnd[k]; r=r[np.isfinite(r)]
        print(f"ratio {k}:{a} {r.mean():.4f} ± {r.std(ddof=1)/np.sqrt(len(r)):.4f} n={len(r)}  report={rep['ratios'][f'{k}:{a}']['mean']:.4f}")
H={"H1":("flat","m_cov"),"H2":("bigru","m_cov"),"H3":("coverage","m_cov"),"H4":("random","gospa_d"),"H5":("flat","gospa_miss"),"H6":("bigru","gospa_miss")}
ps={}
for h,(ref,k) in H.items():
    d=vals["mhsa"][k]-vals[ref][k]
    p1=st.t.cdf(d.mean()/(d.std(ddof=1)/np.sqrt(5)),4) if d.std()>0 else np.nan
    ps[h]=p1
    p2=2*min(p1,1-p1)
    print(h, f"diff {d.mean():.4g} se {d.std(ddof=1)/np.sqrt(5):.3g} p1 {p1:.3g} p2 {p2:.3g} wins {(d<0).sum()}  report p1 {rep['primary'][h]['p_one_sided']:.3g} holm {rep['primary'][h]['p_holm']:.3g} {rep['primary'][h]['verdict']}")
order=sorted(ps,key=ps.get); m=len(order); run=0
for i,h in enumerate(order):
    run=max(run,min(1,(m-i)*ps[h])); print("holm",h,f"{run:.3g}")
# seed files
for s in range(5):
    sf=json.load(open(f"{R}/seed_{s}/metrics.json"))
    for a in ["flat","bigru","mhsa","random","coverage","hand"]:
        for k in ["m_cov","gospa_d","gospa_miss","sr"]:
            assert np.isclose(sf["arms"][a][k],vals[a][k][s],equal_nan=True),(s,a,k)
print("seed files consistent")
# absolute
for a in ["random","coverage","static","hand","ceiling","flat","bigru","mhsa"]:
    v=vals[a]["m_cov"]; v=v[np.isfinite(v)]
    print("abs m_cov",a,round(v.mean(),1),"tracks",round(np.nanmean(vals[a]["mean_tracks"]),4),"init",round(np.nanmean(vals[a]["n_init"]),2), "gospa_d %.4g"%np.mean(vals[a]["gospa_d"]))

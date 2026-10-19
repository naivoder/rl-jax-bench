"""SE over seeds conditions on ONE fixed set of 100 scenarios shared by every seed. Two-level bootstrap (scenarios,
same indices for every arm; then seeds) of the graded ratios, to see whether any grade is fragile to scenario sampling."""
import json, glob, numpy as np
G="runs/full"
base=json.load(open(G+"/baselines.json"))
def lane(a,k): return np.array([json.load(open(f))["per_episode"][k] for f in sorted(glob.glob(f"{G}/{a}/seed_*/metrics.json"))],float)
def bl(a,k):
    d=base["arms"][a]; x=np.array([d[s]["per_episode"][k] for s in sorted(d,key=int)],float)
    return np.repeat(x,5,0) if x.shape[0]==1 else x
rng=np.random.default_rng(0); B=4000
for k,arms in [("m_cov",["mhsa","flat","bigru","coverage","hand"]),("sr",["mhsa","flat","coverage"]),("gospa_d",["mhsa"]),("gospa_miss",["mhsa"])]:
    R=bl("random",k)
    for a in arms:
        X=lane(a,k) if a in ("mhsa","flat","bigru") else bl(a,k)
        def stat(si,ei):
            x=np.nanmean(X[si][:,ei],1); r=np.nanmean(R[si][:,ei],1); return np.mean(x/r)
        pt=stat(np.arange(5),np.arange(100))
        bs=[stat(rng.integers(0,5,5),rng.integers(0,100,100)) for _ in range(B)]
        bs_seed_only=[stat(rng.integers(0,5,5),np.arange(100)) for _ in range(B)]
        print(f"{k}:{a} ratio {pt:.4f}  95% CI seeds-only [{np.percentile(bs_seed_only,2.5):.4f},{np.percentile(bs_seed_only,97.5):.4f}]  seeds+scenarios [{np.percentile(bs,2.5):.4f},{np.percentile(bs,97.5):.4f}]")

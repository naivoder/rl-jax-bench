import json, os, numpy as np
R=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs")
def load(tag,env,arm,s): return json.load(open(f"{R}/{tag}/{env}/{arm}/seed_{s}/metrics.json"))
for env,arm,seeds,tag in [("CartPole-v1","separate",range(1,11),"review_rerun"),("Acrobot-v1","separate",range(1,11),"review_rerun"),
                          ("Pendulum-v1","trunc_bootstrap",range(1,11),"review_rerun"),("Pendulum-v1","no_rscale",range(1,11),"review_rerun"),
                          ("Acrobot-v1","separate",[9],"review_s9")]:
    eq_ep=eq_pu=0; maxd=0.0; fwd=0.0
    for s in seeds:
        a=load("full",env,arm,s); b=load(tag,env,arm,s)
        ea,eb=np.array(a["episodes"]),np.array(b["episodes"])
        same_ep = ea.shape==eb.shape and np.array_equal(ea,eb)
        eq_ep+=same_ep
        same=True
        for k in a["per_update"]:
            x,y=np.asarray(a["per_update"][k],float),np.asarray(b["per_update"][k],float)
            if not np.array_equal(x,y,equal_nan=True):
                same=False; maxd=max(maxd,float(np.nanmax(np.abs(x-y))))
        eq_pu+=same
        fwd=max(fwd,abs(a["summary"]["final_window"]-b["summary"]["final_window"]))
    print(tag,env,arm,f"episodes identical {eq_ep}/{len(list(seeds))}; per_update identical {eq_pu}/{len(list(seeds))}; max per_update diff {maxd:.3g}; max |Δ final_window| {fwd:.3g}")

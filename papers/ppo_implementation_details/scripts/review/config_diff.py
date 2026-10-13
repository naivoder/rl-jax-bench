import json,glob,os
R=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs")
def cfg(tag,env,arm,seed=1):
    d=json.load(open(f"{R}/{tag}/{env}/{arm}/seed_{seed}/metrics.json")); return d["config"],d["git"],d["timing"]
def diff(a,b):
    return {k:(a.get(k),b.get(k)) for k in set(a)|set(b) if a.get(k)!=b.get(k)}
for env,base,arms,tag in [("CartPole-v1","separate",["shared","advnorm_batch","advnorm_ddof0","no_vclip","trunc_bootstrap"],"full"),
 ("Acrobot-v1","separate",["shared","advnorm_batch","advnorm_ddof0","no_vclip","trunc_bootstrap"],"full"),
 ("Pendulum-v1","default",["gym021","gymnasium029","trunc_bootstrap","no_rscale","no_vclip","advnorm_batch"],"full"),
 ("CartPole-v1","lstm",["gymnasium029","lstm_noreset","ff_masked","ff_full","lstm_zero_train"],"full_lstm")]:
    b,g,_=cfg(tag,env,base)
    print(env,base,"git",g)
    for a in arms:
        c,g2,t=cfg(tag,env,a); print("  ",a,diff(b,c),"git",g2, "nvmap",t["n_seeds_vmapped"])
for a in ["separate","shared"]:
    b,g,_=cfg("full","Acrobot-v1",a); c,g2,_=cfg("full_manyseeds","Acrobot-v1",a,11)
    print("manyseeds",a,diff(b,c),g,g2)
# git of all full runs
gs=set()
for p in glob.glob(f"{R}/full*/*/*/seed_*/metrics.json"):
    gs.add(json.load(open(p))["git"])
print(gs)

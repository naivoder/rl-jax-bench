"""Review: X3 sampled vs averaged for the index arm on IDENTICAL P0 (primary key tree), i.e. the paired version of the unpaired '3.2 % lower' statement."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import numpy as np, model as M, algorithm as A, data as D
from rl_jax_bench.stats import paired_compare
R=os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'runs'), 'full')
st={}
for r in range(100):
    for cid,c in json.load(open(f'{R}/seed_{r}/metrics.json'))['cells'].items():
        if 'whittle' in c['J']: st.setdefault(cid,[]).append(c['J']['whittle'])
rels=[];unp=[]
for name,scn in D.TABLE_SCENARIOS.items():
    for K in (1,2,3):
        bank=D.build_bank(scn); rk=D.rep_keys(0,D.TABLE_EXP[name[:2]],scn.sid,K,100); p0k,tie,mode=D.split_rep_keys(rk); P0=D.sample_P0(scn,p0k)
        Js=np.asarray(A.simulate(bank,P0,tie,mode,'whittle',K,A.Cfg(sampled=True))['J']); Ja=np.array(st[f'{name}|K={K}|primary'])
        Jx=np.array(st[f'{name}|K={K}|sampled']); pc=paired_compare(Js,Ja); rels.append(100*pc['rel_improvement']); unp.append(100*(Jx.mean()-Ja.mean())/Ja.mean())
        print(f"{name} K={K} paired {rels[-1]:+.2f}% (se {100*pc['diff_se']/Ja.mean():.2f}) | stored unpaired {unp[-1]:+.2f}%",flush=True)
print('mean paired %.2f%% range [%.2f, %.2f]; stored unpaired mean %.2f%% range [%.2f, %.2f]'%(np.mean(rels),min(rels),max(rels),np.mean(unp),min(unp),max(unp)))

"""Exact tabular A2D (toy check (v)) with the projection occupancy uniform-in-time (proj_gamma = 1, default)
vs gamma-discounted (proj_gamma = 0.995, Def. 1). Writes runs/full_extra/exact_occupancy.json."""
import sys, os
os.environ['JAX_PLATFORMS']='cpu'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import exact as X, data as D
import json
out={}
for env in ['TD1','TD','TD2']:
    tab=X.Tab(env)
    for tgt in ['q','mc']:
        for pg in [1.0, D.GAMMA]:
            r=tab.exact_a2d(tgt, iters=600, proj_gamma=pg)
            out[f'{env}/{tgt}/proj_gamma={pg}']=(round(r['ret'],4), round(r['kl'],4))
            print(env,tgt,pg,r['ret'],r['kl'],flush=True)
json.dump(out,open('runs/full_extra/exact_occupancy.json','w'),indent=1)

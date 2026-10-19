"""Recheck probe: the 'never-revisited track' value of the M_cov statistic in OUR scene (the analogue of the
paper's deterministic Static 2630.72 +- 0.0236 / companion Fig. 7d point mass). One track initiated from one detection
(SimpleMeasurementInitiator, as data.initiate_state) and only predicted until the deleter fires (live reading: the
deleting step is excluded; library reading: included). Descriptive; CPU."""
import sys, json
sys.path.insert(0, '.')
import src.common as C
import jax, jax.numpy as jnp, numpy as np
from dataclasses import replace
from src.configs.config import Config
from src import data
base = Config().env
out = {}
for tag, cfg in [('primary_50ms_trace', base), ('50ms_frob', replace(base, del_stat='frob')),
                 ('5ms_trace', replace(base, dt=0.005)), ('5ms_frob', replace(base, dt=0.005, del_stat='frob'))]:
    F, Q = data.cv_FQ(cfg.dt, cfg.q_filt)
    F, Q = np.asarray(F), np.asarray(Q)
    for rng in (10e3, 20e3, 30e3):
        z = jnp.array([0.1, 0.2, rng])
        x0, C0 = data.initiate_state(z, cfg)
        P = np.asarray(C0); norms = [np.linalg.norm(P)]; lib = None
        for k in range(100000):
            P = F @ P @ F.T + Q
            stat = np.trace(P) if cfg.del_stat == 'trace' else np.linalg.norm(P)
            if stat > cfg.del_thr:
                lib = norms + [np.linalg.norm(P)]
                break
            norms.append(np.linalg.norm(P))
        out[f'{tag}_r{int(rng/1e3)}km'] = dict(life_steps=len(norms), coast_mean_live=float(np.mean(norms)),
                                                coast_mean_lib=float(np.mean(lib)), C0_frob=float(norms[0]), last=float(norms[-1]))
        print(tag, int(rng), out[f'{tag}_r{int(rng/1e3)}km'])
import os; os.makedirs('runs/recheck/measurement', exist_ok=True)
json.dump(out, open('runs/recheck/measurement/probe_coast.json', 'w'), indent=1)
# compare to stored full-scale per-episode M_cov distributions
r = json.load(open('runs/full/baselines.json'))
cv = out['primary_50ms_trace_r20km']['coast_mean_live']
for a in ('random', 'coverage', 'hand', 'ceiling'):
    m = np.array(r['arms'][a]['0']['per_episode']['m_cov'], float)
    q = np.nanpercentile(m, [5, 25, 50, 75, 95])
    print(a, 'per-episode M_cov pct 5/25/50/75/95', np.round(q, 0), 'as fraction of coast value', np.round(q / cv, 3),
          'frac within 2% of coast', float(np.mean(np.abs(m / cv - 1) < 0.02)))

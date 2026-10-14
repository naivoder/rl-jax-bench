"""Collect the third batch of official-code reference runs (oracle/ref_run3.py; folders ref3_<tag>_<seed>: TD2 lambda sweep,
TD1 Q / no-Q, TD RL(MDP) stuck-seed check) plus the second-batch RL(MDP) TD runs into runs/official_ref/ref3_summary.json.
Final = mean of the last 5 stochastic evaluations (official results.csv: pomdp_sto for the trainee, mdp_sto for
RL(MDP)); F = mean projection loss over the last 5 logs."""
import csv, glob, json, os, sys
import numpy as np
from rl_jax_bench.stats import mean_se  # noqa: E402

ROOT = os.path.join(os.environ.get('OFFICIAL_ROOT', 'external/official'), 'repo/a2d_gridworld/tests/results/AdaptAsymDagger')
COLS = ['it', 'timesteps', 'mdp_det', 'pomdp_det', 'mixed_det', 'beta', 'proj', 'mdp_sto', 'pomdp_sto', 'mixed_sto']
out = {}
pats = [('ref3_*', None), ('ref_ail_compact_*', 'RL_state'), ('ref2_fixed_ail_compact_*', 'RL_state')]
for pat, only in pats:
    for f in sorted(glob.glob(os.path.join(ROOT, '*', pat, '*', '*', 'results.csv'))):
        parts = f.split('/')
        env, run, mode = parts[-5], parts[-4], parts[-3]
        if only and mode != only:
            continue
        if only and 'TigerDoorEnv-v0' not in env:
            continue
        rows = [r for r in list(csv.reader(open(f)))[1:] if len(r) == len(COLS)]
        if not rows:
            continue
        c = {k: np.array([float(r[i]) for r in rows]) for i, k in enumerate(COLS)}
        if mode == 'RL_state' and not (run.startswith('ref3_rlmdp') or only):
            continue                                       # the untrained expert folder of an A2D run
        key = 'mdp_sto' if mode == 'RL_state' else 'pomdp_sto'
        tag = 'rlmdp' if mode == 'RL_state' else run[len('ref3_'):].rsplit('_', 1)[0]
        rec = {'run': run, 'last_it': float(c['it'][-1]), 'final': float(np.mean(c[key][-5:])),
               'F_last5': float(np.mean(c['proj'][-5:])), 'curve_it': c['it'].tolist(), 'curve': c[key].tolist()}
        out.setdefault(f'{env}/{tag}', []).append(rec)
summ = {}
for k, recs in sorted(out.items()):
    fins = np.array([r['final'] for r in recs])
    summ[k] = {'n': len(fins), 'mean_se': mean_se(fins) if len(fins) > 1 else (float(fins[0]), None),
               'median': float(np.median(fins)), 'finals': fins.round(3).tolist(),
               'F_last5': [round(r['F_last5'], 6) for r in recs], 'last_it': [r['last_it'] for r in recs]}
    print(k, summ[k]['n'], np.round(fins, 2), 'F_last5', summ[k]['F_last5'])
os.makedirs('runs/official_ref', exist_ok=True)
json.dump({'summary': summ, 'runs': out}, open('runs/official_ref/ref3_summary.json', 'w'))

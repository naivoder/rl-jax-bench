"""Collect the second batch of official-code reference runs (oracle/ref_run2.py; folders ref2_<variant>_<mode>_<seed>, plus
the first-batch ref_* runs as variant 'fixed') into runs/official_ref/ref2_summary.json: per (env, variant, arm) the
per-run final = mean of the last 5 stochastic evaluations of the trained policy (official results.csv columns:
iters, timesteps, mdp_det, pomdp_det, mixed_det, beta, projection_loss, mdp_sto, pomdp_sto, mixed_sto)."""
import csv, glob, json, os, sys
import numpy as np
from rl_jax_bench.stats import mean_se  # noqa: E402

ROOT = os.path.join(os.environ.get('OFFICIAL_ROOT', 'external/official'), 'repo/a2d_gridworld/tests/results/AdaptAsymDagger')
ENV = {'MiniGrid-LavaGapS7-v0': 'FL', 'MiniGrid-TigerDoorEnv-v0': 'TD'}
ARM = {'A2D_partial_state': 'a2d_compact', 'AD_partial_state': 'ail_compact', 'RL_state': 'rl_mdp',
       'A2D_observe': 'a2d_image', 'AD_observe': 'ail_image', 'RL_observe': 'rl_image', 'ARL_observe': 'rl_asym'}
COLS = ['it', 'timesteps', 'mdp_det', 'pomdp_det', 'mixed_det', 'beta', 'proj', 'mdp_sto', 'pomdp_sto', 'mixed_sto']
runs = {}
for f in sorted(glob.glob(os.path.join(ROOT, '*', 'ref*', '*', '*', 'results.csv'))):
    parts = f.split('/')
    env, run, mode = ENV[parts[-5]], parts[-4], parts[-3]
    if run.endswith('_98') or run.endswith('_99'):
        continue                                  # cost-timing runs (5 iterations), not reference runs
    if run.startswith('ref2_'):
        variant = run.split('_')[1]
    else:
        variant = 'fixed'
    if mode == 'RL_state' and 'reuse' in run:
        continue                                  # an empty RL_state dir (expert copied, not trained)
    rows = [r for r in list(csv.reader(open(f)))[1:] if len(r) == len(COLS)]
    if not rows:
        continue
    c = {k: np.array([float(r[i]) for r in rows]) for i, k in enumerate(COLS)}
    key = 'mdp_sto' if mode == 'RL_state' else 'pomdp_sto'
    arm = ARM.get(mode, mode)
    rec = {'run': run, 'last_it': float(c['it'][-1]), 'final': float(np.mean(c[key][-5:])),
           'final_det': float(np.mean(c[key.replace('sto', 'det')][-5:])), 'proj5': float(np.mean(c['proj'][-5:])),
           'curve_it': c['it'].tolist(), 'curve': c[key].tolist(), 'proj': c['proj'].tolist()}
    runs.setdefault(env, {}).setdefault(variant, {}).setdefault(arm, []).append(rec)
summary = {}
for env, dv in runs.items():
    for variant, da in dv.items():
        for arm, recs in da.items():
            full = [r for r in recs if r['last_it'] >= 200 or arm.endswith('image')]
            fins = np.array([r['final'] for r in full])
            if not len(fins):
                continue
            m, se = mean_se(fins) if len(fins) > 1 else (float(fins.mean()) if len(fins) else float('nan'), float('nan'))
            summary.setdefault(env, {}).setdefault(variant, {})[arm] = {
                'n': len(full), 'mean': float(m), 'se': float(se), 'finals': fins.tolist(),
                'runs': [r['run'] for r in full], 'last_it': [r['last_it'] for r in full],
                'proj5': [r['proj5'] for r in full]}
            print(f'{env:3s} {variant:8s} {arm:12s} n={len(full)} final {m:8.2f} +- {se:5.2f}  {np.round(fins, 2)}')
os.makedirs('runs/official_ref', exist_ok=True)
json.dump({'summary': summary, 'runs': runs}, open('runs/official_ref/ref2_summary.json', 'w'))

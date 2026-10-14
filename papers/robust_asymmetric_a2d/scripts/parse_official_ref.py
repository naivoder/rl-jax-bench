"""Collect the official-code reference runs (oracle/ref_run.py) into runs/official_ref/ref_curves.json.
results.csv columns (official do_log): iters, timesteps, mdp_det, pomdp_det, mixed_det, beta, projection_loss,
mdp_sto, pomdp_sto, mixed_sto."""
import csv, glob, json, os
ROOT = os.path.join(os.environ.get('OFFICIAL_ROOT', 'external/official'), 'repo/a2d_gridworld/tests/results/AdaptAsymDagger')
ENV = {'MiniGrid-LavaGapS7-v0': 'FL', 'MiniGrid-TigerDoorEnv-v0': 'TD'}
out = {}
for f in sorted(glob.glob(os.path.join(ROOT, '*', 'ref_*', '*', '*', 'results.csv'))):
    parts = f.split('/')
    env, run, mode = ENV[parts[-5]], parts[-4], parts[-3]
    rows = list(csv.reader(open(f)))[1:]
    cols = ['it', 'timesteps', 'mdp_det', 'pomdp_det', 'mixed_det', 'beta', 'proj', 'mdp_sto', 'pomdp_sto', 'mixed_sto']
    cur = {c: [float(r[i]) for r in rows if len(r) == len(cols)] for i, c in enumerate(cols)}
    out.setdefault(env, {}).setdefault(mode, {})[run] = cur
os.makedirs('runs/official_ref', exist_ok=True)
json.dump(out, open('runs/official_ref/ref_curves.json', 'w'))
for env, d in out.items():
    for mode, runs in d.items():
        for run, c in runs.items():
            key = 'mdp_sto' if mode.startswith('RL_state') else 'pomdp_sto'
            last = c[key][-5:]
            print(env, mode, run, 'n_evals', len(c['it']), 'last it', c['it'][-1] if c['it'] else None,
                  'final5 %s %.2f' % (key, sum(last) / max(len(last), 1)), 'proj5 %.4f' % (sum(c['proj'][-5:]) / 5))

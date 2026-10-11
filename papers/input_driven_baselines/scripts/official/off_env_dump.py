"""OFFICIAL side (run with the official venv). Drives the authors' unmodified load_balance env.
Dumps, per seed: their generated jobs, a random action sequence, their obs (training-agent formula),
rewards, done flags, wall times; plus closed-loop official SPT/LeastWork actions and returns.
Usage: off_env_dump.py <out.npz> <mode: paper|default> <n_seeds> <seed0>"""
import sys, os
OFF = os.path.join(os.environ.get('OFFICIAL_ROOT', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'official')), 'input_driven_rl_example')  # the authors' clone
out, mode, n, s0 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rates = '0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05'.split()
argv = ['x', '--num_workers', '10', '--service_rates'] + rates
if mode == 'paper':
    argv += ['--job_distribution', 'pareto', '--job_size_pareto_shape', '1.5', '--job_size_pareto_scale', '100',
             '--job_interval', '55', '--num_stream_jobs', '500']
sys.argv = argv
sys.path.insert(0, OFF)
import numpy as np
from param import args
from load_balance.env import LoadBalanceEnvironment
from load_balance.job_generator import generate_jobs
from load_balance.heuristic_agents import ShortestProcessingTimeAgent, LeastWorkAgent

def obs_of(state):  # verbatim formula of the official training agent / ActorAgent.get_action
    workers, job, _ = state
    inputs = np.zeros([args.num_workers + 1])
    for w in workers:
        inputs[w.worker_id] = min(sum(j.size for j in w.queue) / args.job_size_norm_factor / 5.0, 20.0)
    inputs[-1] = min(job.size / args.job_size_norm_factor, 10.0)
    return inputs

env = LoadBalanceEnvironment()
res = {k: [] for k in ['arr', 'size', 'act', 'obs', 'rew', 'done', 'wall', 'spt_act', 'spt_rew', 'lw_rew',
                       'spt_ret', 'lw_ret']}
for s in range(s0, s0 + n):
    np.random.seed(s); t, sz = generate_jobs(args.num_stream_jobs)
    act = np.random.RandomState(10_000 + s).randint(0, args.num_workers, size=args.num_stream_jobs)
    np.random.seed(s); env.reset()
    st = env.observe(); obs, rew, dn, wall = [], [], [], []
    for a in act:
        obs.append(obs_of(st)); wall.append(st[2])
        st, r, d = env.step(int(a)); rew.append(r); dn.append(d)
        if d: break
    assert len(rew) == len(act) and dn[-1] and not any(dn[:-1]), (s, len(rew))
    res['arr'].append(t); res['size'].append(sz); res['act'].append(act); res['obs'].append(obs)
    res['rew'].append(rew); res['done'].append(dn); res['wall'].append(wall)
    for nm, ag in (('spt', ShortestProcessingTimeAgent()), ('lw', LeastWorkAgent())):
        np.random.seed(s); env.reset(); st = env.observe(); d = False; tot = 0; acts = []; rs = []
        while not d:
            a = ag.get_action(st); acts.append(a); st, r, d = env.step(a); tot += r; rs.append(r)
        res[nm + '_ret'].append(tot); res[nm + '_rew'].append(rs)
        if nm == 'spt': res['spt_act'].append(acts)
np.savez(out, **{k: np.asarray(v) for k, v in res.items()})
print('wrote', out, 'spt mean/1e4', np.mean(res['spt_ret']) / 1e4, 'lw mean/1e4', np.mean(res['lw_ret']) / 1e4)

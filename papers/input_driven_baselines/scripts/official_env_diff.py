"""OUR side: replay the official jobs + actions through src.envs and diff step by step.
Usage: python scripts/official_env_diff.py <official npz>  (CPU JAX)"""
import sys, os, json
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
import numpy as np, jax, jax.numpy as jnp
from src.envs import make_lb_spec, lb_reset, lb_step, lb_obs, lb_heuristic_action
from src.data import LBInputs
from src.train import load_config
d = np.load(sys.argv[1], allow_pickle=True)
cfg = load_config(os.path.join(REPO, 'src/configs/load_balance.yaml'))
n_jobs = d['arr'].shape[1]; cfg['jobs_per_episode'] = n_jobs
spec = make_lb_spec(cfg)
step = jax.jit(lambda s, a, i: lb_step(spec, s, a, i)); obsf = jax.jit(lambda s, i: lb_obs(spec, s, i))
spt = jax.jit(lambda s, i: lb_heuristic_action(spec, 'spt', s, i))
out = {'n_seq': int(d['arr'].shape[0]), 'n_jobs': n_jobs}
rew_mis = obs_mis = done_mis = 0; obs_maxabs = 0.0; first_obs_mis = None; ret_off = []; ret_ours = []
spt_act_mis = 0; spt_ret_ours = []; spt_first = None
for i in range(d['arr'].shape[0]):
    inp = LBInputs(jnp.asarray(d['arr'][i], jnp.int32), jnp.asarray(d['size'][i], jnp.int32))
    s = lb_reset(spec); R = []
    for t, a in enumerate(d['act'][i]):
        o, _ = obsf(s, inp); o = np.asarray(o)
        diff = np.abs(o - d['obs'][i][t]).max()
        if diff > 1e-5:
            obs_mis += 1; first_obs_mis = first_obs_mis or (i, t, o.tolist(), d['obs'][i][t].tolist())
        obs_maxabs = max(obs_maxabs, float(diff))
        s, r, dn = step(s, jnp.int32(a), inp); R.append(float(r))
        done_mis += int(bool(dn) != bool(d['done'][i][t]))
    R = np.array(R); rew_mis += int(np.sum(R != d['rew'][i])); ret_off.append(d['rew'][i].sum()); ret_ours.append(R.sum())
    # closed-loop: our SPT vs official SPT actions on the same inputs
    s = lb_reset(spec); tot = 0.0
    for t in range(n_jobs):
        a = int(spt(s, inp));
        if a != int(d['spt_act'][i][t]):
            spt_act_mis += 1; spt_first = spt_first or (i, t, a, int(d['spt_act'][i][t]))
        s, r, _ = step(s, jnp.int32(a), inp); tot += float(r)
    spt_ret_ours.append(tot)
out.update(reward_mismatches=rew_mis, done_mismatches=done_mis, obs_mismatch_steps=obs_mis, obs_max_abs_diff=obs_maxabs,
           first_obs_mismatch=first_obs_mis, total_steps=int(d['act'].size),
           ret_off_mean_1e4=float(np.mean(ret_off)) / 1e4, ret_ours_mean_1e4=float(np.mean(ret_ours)) / 1e4,
           spt_action_mismatches=spt_act_mis, spt_first_mismatch=spt_first,
           spt_ret_official_1e4=float(np.mean(d['spt_ret'])) / 1e4, spt_ret_ours_1e4=float(np.mean(spt_ret_ours)) / 1e4,
           lw_ret_official_1e4=float(np.mean(d['lw_ret'])) / 1e4)
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(sys.argv[1].replace('.npz', '_diff.json'), 'w'), indent=1, default=str)

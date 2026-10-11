"""Heuristics (ours, plus official-tie-break variants) on OFFICIAL-generated vs OUR-generated inputs; generator stats."""
import sys, os, json
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
import numpy as np, jax, jax.numpy as jnp
from src.envs import make_lb_spec, lb_reset, lb_step, lb_now, lb_queued_work, lb_queue_counts, lb_durations
from src.data import LBInputs, sample_lb_batch, lb_input_kwargs, test_key
from src.train import load_config
cfg = load_config(os.path.join(REPO, 'src/configs/load_balance.yaml'))
spec = make_lb_spec(cfg)
def act(name, s, inp):
    now = lb_now(s, inp); wait = jnp.maximum(s.busy_until - now, 0).astype(jnp.float32); rates = spec.rates
    q = lb_queued_work(spec, s, inp, now)
    if name == 'spt_fast': return jnp.argmin(q / rates - 1e-6 * rates)          # ours
    if name == 'spt_official': return jnp.argmin(q / rates)                      # first index wins ties (np strict <)
    if name == 'leastwork_official': return jnp.argmin(q)
    if name == 'jsq_fast': return jnp.argmin(lb_queue_counts(spec, s, now) - 1e-3 * rates)
    if name == 'jsq_first': return jnp.argmin(lb_queue_counts(spec, s, now).astype(jnp.float32))
    if name == 'eft': return jnp.argmin(wait + lb_durations(spec, inp.sizes[s.k]).astype(jnp.float32) - 1e-6 * rates)
    if name == 'lrt': return jnp.argmin(wait - 1e-6 * rates)
def roll(name):
    def f(inp):
        def body(s, _):
            a = act(name, s, inp).astype(jnp.int32); s2, r, _ = lb_step(spec, s, a, inp); return s2, r
        _, r = jax.lax.scan(body, lb_reset(spec), None, length=spec.n_jobs); return jnp.sum(r) * 1e-4
    return jax.jit(jax.vmap(f))
names = ['spt_fast', 'spt_official', 'leastwork_official', 'jsq_fast', 'jsq_first', 'eft', 'lrt']
sets = {}
for tag, fn in [('official_test_1e8', 'off_env_paper_test.npz'), ('official_seeds_1000', 'off_env_paper_s1000.npz')]:
    d = np.load(fn); sets[tag] = (LBInputs(jnp.asarray(d['arr'], jnp.int32), jnp.asarray(d['size'], jnp.int32)), d)
sets['ours_test'] = (sample_lb_batch(test_key(), 100, **lb_input_kwargs(cfg)), None)
sets['ours_1000'] = (sample_lb_batch(jax.random.PRNGKey(777), 1000, **lb_input_kwargs(cfg)), None)
out = {}
for tag, (inp, d) in sets.items():
    row = {}
    for nm in names:
        r = np.asarray(roll(nm)(inp)); row[nm] = [round(float(r.mean()), 3), round(float(r.std(ddof=1) / np.sqrt(len(r))), 3)]
    if d is not None:
        row['official_env_spt'] = round(float(np.mean(d['spt_ret']) / 1e4), 3)
        row['official_env_leastwork'] = round(float(np.mean(d['lw_ret']) / 1e4), 3)
    a = np.asarray(inp.arrivals); s = np.asarray(inp.sizes); g = np.diff(np.concatenate([np.zeros((a.shape[0], 1)), a], 1), axis=1)
    row['gen'] = dict(gap_mean=round(float(g.mean()), 3), size_mean=round(float(s.mean()), 2), size_median=float(np.median(s)),
                      size_q99=float(np.quantile(s, .99)), size_max=int(s.max()), span_mean=round(float(a[:, -1].mean()), 1),
                      offered_util=round(float(s.sum(1).mean() / a[:, -1].mean() / 6.0), 4))
    out[tag] = row; print(tag, json.dumps(row))
json.dump(out, open('heuristics_cross.json', 'w'), indent=1)

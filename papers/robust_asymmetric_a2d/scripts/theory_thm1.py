"""Empirical test of Theorem 1 (AIL converges to the implicit policy pi_hat(a|b) = E_{d(s|b)}[pi_theta(a|s)]) on
the 20 trained RL(MDP) experts actually used by the full-scale AIL(Compact) arm: for each seed, the exact AIL fixed point
of THAT seed's frozen expert (tabular, src/exact.py, undiscounted visitation = what the AIL sampler draws) vs the
sampled AIL(Compact) final return / clean KL of the same seed. Prediction (Thm 1): per-seed agreement within MC error,
slope ~ 1 across seeds. Writes runs/full/theory_thm1.json. CPU only."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import train as T, exact as E, data as D, model as M   # noqa: E402
import numpy as np   # noqa: E402
import jax, jax.numpy as jnp   # noqa: E402
import scipy.stats as st   # noqa: E402

EXP = sys.argv[1] if len(sys.argv) > 1 else 'full'      # full (float32) | full_f64 (float64 re-run)
out = {}
for env in ['FL', 'TD']:
    tab = E.Tab(env)
    lat = D.Latent(jnp.asarray(tab.pos), jnp.asarray(tab.h), jnp.asarray(tab.il.astype(bool)))
    X = D.state_vec(tab.spec, lat)
    pol = M.make_policy('flat')
    tmpl = pol.init(jax.random.PRNGKey(0), X[0])
    seeds = list(range(20))
    ex = T.load_expert(f'runs/{EXP}/{env}/expert_best.npz', seeds, tmpl)
    rows = []
    for i, sd in enumerate(seeds):
        p = jax.tree_util.tree_map(lambda x: x[i], ex)
        probs = np.asarray(jnp.exp(pol.apply(jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float32), p), X)), np.float64)
        probs /= probs.sum(1, keepdims=True)
        fp = tab.ail_fixed_point(probs, iters=80, gamma_w=1.0)
        _, ret_e = tab.occupancy(tab.behaviour(pi_e=probs, pi_l=None))
        m = json.load(open(f'runs/{EXP}/{env}/ail_compact/seed_{sd}/metrics.json'))
        rows.append({'seed': sd, 'expert_exact_ret': float(ret_e), 'fp_ret_sto': float(fp['ret_sto']),
                     'fp_ret_greedy': float(fp['ret_greedy']), 'fp_kl': float(fp['kl']),
                     'ail_ret_last5': float(np.mean(m['curves']['learner_sto/ret'][-5:])),
                     'ail_kl_last5': float(np.mean(m['curves']['learner_sto/kl_clean'][-5:]))})
    a = np.array([r['ail_ret_last5'] for r in rows]); f = np.array([r['fp_ret_sto'] for r in rows])
    ka = np.array([r['ail_kl_last5'] for r in rows]); kf = np.array([r['fp_kl'] for r in rows])
    d = a - f
    lr = st.linregress(f, a)
    out[env] = {'per_seed': rows, 'ail_minus_fp_mean': float(d.mean()), 'ail_minus_fp_se': float(d.std(ddof=1) / np.sqrt(len(d))),
                'fp_ret_mean': float(f.mean()), 'ail_ret_mean': float(a.mean()),
                'pearson_r': float(st.pearsonr(f, a).statistic), 'slope': float(lr.slope), 'slope_se': float(lr.stderr),
                'kl_ail_mean': float(ka.mean()), 'kl_fp_mean': float(kf.mean()),
                'kl_rel_diff': float((ka.mean() - kf.mean()) / kf.mean()),
                'fp_greedy_mean': float(np.mean([r['fp_ret_greedy'] for r in rows]))}
    print(env, {k: v for k, v in out[env].items() if k != 'per_seed'}, flush=True)
json.dump(out, open(f'runs/full/theory_thm1_{EXP}.json', 'w'), indent=1)

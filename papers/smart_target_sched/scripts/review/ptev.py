"""Review: a cheap no-search heuristic the reproduction did not try: predicted-TEV  index = d * s(phi^0(P)) (uncertainty if left untracked one more slot).
Same keys as runs/full (seed 0) -> paired against the stored whittle/myopic/tev J."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import jax, jax.numpy as jnp, numpy as np
import model as M, algorithm as A, data as D
from rl_jax_bench.stats import paired_compare
_orig = A._index_fn
def ptev(bank, P, cfg=None):
    n = P.shape[0]
    return bank.d * M.stat(M.phi(bank, P, jnp.zeros(n))), jnp.zeros((), jnp.int32)
def ratio(bank, P, cfg=None):   # myopic gain per unit current cost -- another cheap rule
    n = P.shape[0]
    return bank.d * (M.stat(M.phi(bank, P, jnp.zeros(n))) - M.stat(M.phi(bank, P, jnp.ones(n)))) + bank.d * M.stat(P), jnp.zeros((), jnp.int32)
A._index_fn = lambda name: ptev if name == 'ptev' else (ratio if name == 'mytev' else _orig(name))
R = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'runs'), 'full')
stored = {}
for r in range(100):
    for cid, c in json.load(open(f'{R}/seed_{r}/metrics.json'))['cells'].items():
        if cid.endswith('|primary'):
            for p, v in c['J'].items(): stored.setdefault(cid, {}).setdefault(p, []).append(v)
cfg = A.Cfg()
for name, scn in D.TABLE_SCENARIOS.items():
    for K in (1, 2, 3):
        bank = D.build_bank(scn); rk = D.rep_keys(0, D.TABLE_EXP[name[:2]], scn.sid, K, 100)
        p0k, tie, mode = D.split_rep_keys(rk); P0 = D.sample_P0(scn, p0k)
        cid = f'{name}|K={K}|primary'; w = np.array(stored[cid]['whittle'])
        # sanity: the key tree reproduces the stored whittle J for this cell
        out = {}
        for pol in ('ptev', 'mytev'):
            out[pol] = np.asarray(A.simulate(bank, P0, tie, mode, pol, K, cfg)['J'])
        chk = np.asarray(A.simulate(bank, P0, tie, mode, 'tev', K, cfg)['J'])
        assert np.allclose(chk, stored[cid]['tev'], rtol=0, atol=0), cid
        s = ' '.join(f"{pol}: {out[pol].mean():.2f} idx-{pol} {100*paired_compare(w,out[pol])['rel_improvement']:+.2f}% p={paired_compare(w,out[pol])['t_p']:.1e}" for pol in out)
        print(f"{cid} whittle {w.mean():.2f} myopic {np.mean(stored[cid]['myopic']):.2f} | {s}", flush=True)

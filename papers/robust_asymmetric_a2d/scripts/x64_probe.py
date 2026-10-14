"""float64 probe: does float32 training cause the TD 'stuck at a door' seeds (official code is
float64)? Re-runs one arm with jax_enable_x64 and every network parameter / optimizer state cast to float64
(the same path the float64 whole-iteration parity tests exercise), 20 seeds, CPU, into runs/full_x64/.
Usage: JAX_PLATFORMS=cpu python scripts/x64_probe.py ENV ARM[+tag] SEEDS ITERS [key=val ...]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax  # noqa: E402
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402
from src import train as T  # noqa: E402

env, arm = sys.argv[1], sys.argv[2]
n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 20
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 200
f64 = lambda t: jax.tree_util.tree_map(lambda x: x.astype(jnp.float64) if x.dtype == jnp.float32 else x, t)
_init = T.Trainer.init


def init64(self, run_key, expert_params=None):
    ts = _init(self, run_key, expert_params)
    return ts._replace(params=f64(ts.params), opt=f64(ts.opt), best_expert=f64(ts.best_expert),
                       proj_loss=ts.proj_loss.astype(jnp.float64))


T.Trainer.init = init64
kw = dict(T.ARMS[arm.split('+')[0]]); kw.update(env=env, iters=iters)
for kv in sys.argv[5:]:
    k, v = kv.split('=')
    kw[k] = type(getattr(T.Config(), k))(v)
cfg = T.Config(**kw)
T.run('full_x64', arm, cfg, list(range(n_seeds)))

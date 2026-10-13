"""full-scale investigation: is max|rho-1| > 1e-5 at the first minibatch (Pendulum, late training) float32 noise?
Same params, same obs and actions; log pi from the rollout path (one (1, d) forward per step, as PPO.rollout with N=1)
vs the training path (one (64, d) forward per minibatch, as PPO._loss), in float32 (matmul precision 'highest', as in
training) and float64, for several log-std values. Usage: python -m scripts.full.ratio_numerics"""
import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
import json  # noqa: E402
import os  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src import model as mdl  # noqa: E402

net = mdl.SeparateAC(n_out=1, continuous=True, hidden=64)
key = jax.random.PRNGKey(0)
params0 = net.init(key, jnp.zeros((1, 3)))
# make the policy head non-trivial (trained-like magnitude) so mu is O(1)
params0 = jax.tree.map(lambda x: x, params0)
params0["params"]["actor_head"]["kernel"] = jax.random.normal(jax.random.PRNGKey(1), (64, 1)) * 0.5
obs = jax.random.normal(jax.random.PRNGKey(2), (2048, 3)) * 1.5  # normalized-obs scale (clip 10)
perm = jax.random.permutation(jax.random.PRNGKey(3), 2048)[:64]
out = {}
for log_std in (0.0, -2.0, -3.0, -4.0, -4.5):
    row = {}
    for dt in (jnp.float32, jnp.float64):
        p = jax.tree.map(lambda x: x.astype(dt), params0)
        p["params"]["log_std"] = jnp.full((1,), log_std, dt)
        o = obs.astype(dt)
        # rollout path: one (1, 3) forward per step
        dist_roll = jax.lax.map(lambda x: jax.tree.map(lambda y: y[0], net.apply(p, x[None])[0]), o)
        mu, ls = dist_roll
        a = mu + jnp.exp(ls) * jax.random.normal(jax.random.PRNGKey(4), mu.shape, dt)
        lp_roll = mdl.log_prob(dist_roll, a, True)
        # training path: one (64, 3) forward on a shuffled minibatch
        dist_tr, _ = net.apply(p, o[perm])
        lp_tr = mdl.log_prob(dist_tr, a[perm], True)
        row[str(np.dtype(dt))] = float(jnp.max(jnp.abs(jnp.exp(lp_tr - lp_roll[perm]) - 1.0)))
    out[f"log_std={log_std}"] = row
    print("[METRIC]", f"log_std={log_std}", json.dumps(row), flush=True)
out["device"] = str(jax.devices()[0])
p = os.path.join(REPO, "runs", "full", "divergent", "ratio_numerics.json")
os.makedirs(os.path.dirname(p), exist_ok=True)
json.dump(out, open(p, "w"), indent=1)

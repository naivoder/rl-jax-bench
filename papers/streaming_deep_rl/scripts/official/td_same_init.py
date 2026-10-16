"""Differential run, our side: draw our SparseInit for the stream-TD net (seed), write it in torch layout for
td_same_init_official.py, run OUR TDAgent from the same weights for T steps on CPU, save predictions.
Usage: python scripts/official/td_same_init.py <seed> <T> <outdir>"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
from src import agents as ag, data as dt, train as tr  # noqa: E402

seed, T, outdir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
d = dt.load_ett()
X = tr.ema_traces(d["obs"][: T + 1], 0.999).astype(np.float32)
cum = d["cumulant"][2: T + 2].astype(np.float32)
agent = ag.TDAgent(7)
lane = jax.tree.map(lambda x: x[0], tr.stack_lanes("td", [tr.lane_cfg("td", "stream_td")]))
c = agent.init(jax.random.fold_in(jax.random.PRNGKey(0), seed), lane, jnp.asarray(X[0]))
p = c.params
names = {"fc1_v": p["trunk"]["fc1"], "hidden_v": p["trunk"]["fc2"], "fc_v": p["out"]}
np.savez(os.path.join(outdir, f"w_seed{seed}.npz"),
         **{f"{k}.weight": np.asarray(v["kernel"]).T for k, v in names.items()},
         **{f"{k}.bias": np.asarray(v["bias"]) for k, v in names.items()})
c, (pred, cr) = jax.jit(lambda c: jax.lax.scan(lambda cc, x: agent.step(cc, x[0], x[1], x[2], lane), c,
                                               (jnp.asarray(X[1:]), jnp.asarray(cum), jnp.zeros(T, bool))))(c)
np.savez(os.path.join(outdir, f"ours_seed{seed}.npz"), pred=np.asarray(pred, float), cum=np.asarray(cr, float))
print("[INFO] ours saved")

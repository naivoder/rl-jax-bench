"""FLOPs per policy step (one replication) from XLA's cost analysis of the jitted step function.

    uv run python scripts/flops.py     -> runs/full/flops.json
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
import src  # noqa: E402,F401
from src import algorithm as alg  # noqa: E402
from src import train  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

out = {}
for K in (3, 2):
    pcs = {"ucb1": {}, "exp3s": {"S": 2, "alpha": 1e-4}, "ducb": {"gamma": 0.9975}, "swucb": {"tau": 1214}}
    for name, pc in pcs.items():
        params, _, buf = train.policy_params(name, pc, K, 10_000)
        step = alg.make_step(name, K, "random")
        st, ac = alg.init_state(name, K, buf), alg.init_acc(K)
        k = jax.random.PRNGKey(0)
        f = jax.jit(lambda s, a, t, mu, k1, k2: step(s, a, t, mu, jnp.asarray(False), k1, k2, params))
        c = f.lower(st, ac, jnp.int32(5000), jnp.full(K, 0.5), k, k).compile().cost_analysis()
        c = c[0] if isinstance(c, list) else c
        out[f"K{K}_{name}"] = {"flops": float(c.get("flops", -1)), "transcendentals": float(c.get("transcendentals", -1)),
                               "bytes_accessed": float(c.get("bytes accessed", -1))}
        print(K, name, out[f"K{K}_{name}"])
os.makedirs(os.path.join(ROOT, "runs", "full"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "runs", "full", "flops.json"), "w"), indent=1)

"""Reproduce the numeric claims checked in review on CPU.

    JAX_PLATFORMS=cpu python scripts/review_checks.py # writes runs/review_checks.json
"""
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src.data import full_sha, lb_input_kwargs, sample_lb_batch, test_key  # noqa: E402
from src.envs import lb_heuristic_action, lb_queue_counts, lb_reset, lb_step, make_lb_spec  # noqa: E402
from src.train import load_config  # noqa: E402

cfg = load_config("src/configs/load_balance.yaml")
spec = make_lb_spec(cfg)
kw = lb_input_kwargs(cfg)
scale = float(cfg["reward_scale"])


def roll(name, inp):
    def body(st, _):
        now = inp.arrivals[st.k]
        nonempty = jnp.mean((lb_queue_counts(spec, st, now) > 0).astype(jnp.float32))
        a = lb_heuristic_action(spec, name, st, inp)
        busy_fastest = (a == jnp.argmax(spec.rates))
        st2, r, _ = lb_step(spec, st, a, inp)
        return st2, (r, nonempty, busy_fastest)

    _, (r, ne, bf) = jax.lax.scan(body, lb_reset(spec), None, length=spec.n_jobs)
    return jnp.sum(r) * scale, jnp.mean(ne), jnp.mean(bf)


out = {}
for label, key in (("test_set", test_key()), ("reviewer_PRNGKey5", jax.random.PRNGKey(5))):
    inp = sample_lb_batch(key, 100, **kw)
    out[label] = {"sha256": full_sha(inp)[:16]}
    for nm in ("spt", "jsq", "eft", "lrt", "fastest"):
        ret, ne, bf = jax.jit(jax.vmap(lambda i: roll(nm, i)))(inp)
        ret, ne = np.asarray(ret), np.asarray(ne)
        out[label][nm] = {"mean": float(ret.mean()), "se": float(ret.std(ddof=1) / np.sqrt(len(ret))),
                          "frac_nonempty_servers_at_arrivals": float(ne.mean()),
                          "frac_jobs_on_fastest": float(np.asarray(bf).mean())}
        print(label, nm, json.dumps(out[label][nm]), flush=True)
json.dump(out, open("runs/review_checks.json", "w"), indent=1)

# the heuristics on 2,000 further sequences (key 4e8, never used elsewhere) to measure how much the fixed
# 100-sequence test set moves an absolute number (Pareto(1.5) sizes have infinite variance).
big = sample_lb_batch(jax.random.PRNGKey(400_000_000), 2000, **kw)
out["fresh_2000"] = {}
for nm in ("spt", "jsq", "eft", "lrt"):
    ret = np.asarray(jax.jit(jax.vmap(lambda i: roll(nm, i)[0]))(big))
    blocks = ret.reshape(20, 100).mean(1)  # 20 disjoint 100-sequence "test sets"
    out["fresh_2000"][nm] = {"mean": float(ret.mean()), "se": float(ret.std(ddof=1) / np.sqrt(len(ret))),
                             "median": float(np.median(ret)),
                             "spread_of_100seq_set_means_std": float(blocks.std(ddof=1)),
                             "min_max_of_100seq_set_means": [float(blocks.min()), float(blocks.max())]}
    print("fresh_2000", nm, json.dumps(out["fresh_2000"][nm]), flush=True)
json.dump(out, open("runs/review_checks.json", "w"), indent=1)

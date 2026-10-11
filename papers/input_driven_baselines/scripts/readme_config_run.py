"""OUR env + learner at the AUTHORS' README configuration (identical overrides to
the README-config probe), with extra --set overrides (e.g. mv_schedule=official) and a
100-iteration test cadence aligned with the authors' trainer.

    python scripts/readme_config_run.py <arm> <iters> <seed> <tag> [key=value...]
"""
import json
import os
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
os.environ["TMPDIR"] = os.path.join(REPO, ".tmp")
sys.path.insert(0, REPO)
import jax  # noqa: E402,F401
import jax.numpy as jnp  # noqa: E402

import src.data as D  # noqa: E402
import src.train as T  # noqa: E402


def sample_uniform_inputs(key, n_jobs, mean_interarrival, pareto_shape=None, pareto_scale=None):
    k_size, k_gap = jax.random.split(key)
    sizes = jnp.floor(jax.random.uniform(k_size, (n_jobs,), minval=100.0, maxval=1000.0)).astype(jnp.int32)
    gaps = jnp.floor(jax.random.exponential(k_gap, (n_jobs,)) * mean_interarrival).astype(jnp.int32)
    return D.LBInputs(jnp.cumsum(gaps).astype(jnp.int32), sizes)


def sample_uniform_batch(key, n, **kw):
    return jax.vmap(lambda k: sample_uniform_inputs(k, **kw))(jax.random.split(key, n))


T.sample_lb_batch = sample_uniform_batch  # train inputs, fixed multi-value inputs, test inputs
D.sample_lb_batch = sample_uniform_batch

arm, iters, seed, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
extra = {k: yaml.safe_load(v) for k, v in (a.split("=", 1) for a in sys.argv[5:])}
OUT = os.path.join(REPO, "runs/readme_config", tag)
cfg = T.load_config(os.path.join(REPO, "src/configs/load_balance.yaml"))
cfg.update({"jobs_per_episode": 1000, "mean_interarrival": 100.0, "gamma": 1.0, "hidden": [200, 128],
            "activation": "leaky_relu", "lr_actor": 1e-3, "lr_critic": 1e-3, "entropy_init": 1.0,
            "entropy_final": 1e-4, "entropy_decay_iters": 1000, "iterations": iters, "eval_every": 100,
            "pg_var_every": 1000000, "ckpt_minutes": 1000})
cfg.update(extra)
os.makedirs(OUT, exist_ok=True)
blob = T.run(cfg, arm, seed, OUT)
print("DONE", json.dumps({"arm": arm, "seed": seed, "tag": tag, "final": blob["final"]["test_mean_last5"]}))

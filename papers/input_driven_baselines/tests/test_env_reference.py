""": the closed-form JAX simulator equals an independent event-driven numpy simulator
(official semantics) on 20 random (input sequence, action list) pairs at the FULL full-scale configuration
(10 servers, 500 jobs): identical per-step rewards (exact integer times) and identical done steps."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import src  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src.data import lb_input_kwargs, sample_lb_batch  # noqa: E402
from src.envs import lb_actions_rollout, make_lb_spec, reference_event_sim  # noqa: E402
from src.train import load_config  # noqa: E402


def test_closed_form_equals_event_sim_20_pairs():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "src", "configs", "load_balance.yaml"))
    spec = make_lb_spec(cfg)
    inp = sample_lb_batch(jax.random.PRNGKey(12345), 20, **lb_input_kwargs(cfg))
    rng = np.random.default_rng(0)
    roll = jax.jit(lambda a, i: lb_actions_rollout(spec, a, i))
    for i in range(20):
        if i % 2 == 0:  # uniform random actions
            acts = rng.integers(0, spec.n_servers, spec.n_jobs)
        else:  # skewed towards the fast servers (long queues, lots of waiting)
            p = np.arange(1, spec.n_servers + 1, dtype=float) ** 3
            acts = rng.choice(spec.n_servers, spec.n_jobs, p=p / p.sum())
        one = jax.tree_util.tree_map(lambda x: x[i], inp)
        r, d = roll(jnp.asarray(acts, jnp.int32), one)
        rr, dd = reference_event_sim(np.asarray(one.arrivals), np.asarray(one.sizes), acts, spec.rate_num, spec.rate_den)
        np.testing.assert_array_equal(np.asarray(r).astype(np.int64), rr)
        assert (np.asarray(d) == dd).all()

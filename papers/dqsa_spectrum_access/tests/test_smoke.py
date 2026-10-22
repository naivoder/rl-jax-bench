"""Smoke test (CPU, < 30 s): shapes of the env / network / rollout, one training step, one eval, baselines run."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()   # persistent compile cache (a warm cache keeps the smoke test under 30 s)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import algorithm as alg  # noqa: E402
import data as D  # noqa: E402
import eval as E  # noqa: E402
import model as M  # noqa: E402

SPEC = D.EnvSpec(name="t", n_min=3, n_max=5, n_channels=2, k_local=2)
FAD = D.EnvSpec(name="f", n_min=6, n_max=6, n_channels=4, k_local=2, subset="random", capacity="rate", fading=True, pu=True)


def test_env_shapes_and_obs():
    for spec in (SPEC, FAD):
        w = D.sample_world(spec, jax.random.PRNGKey(0))
        s = D.init_state(spec, jax.random.PRNGKey(1))
        x = D.observe(spec, w, s)
        assert x.shape == (spec.n_max, 2 * spec.k_local + 2)
        assert w.subset.shape == (spec.n_max, spec.k_local)
        s2, info = jax.jit(lambda w, s: D.step_local(spec, w, s, jnp.ones((spec.n_max,), jnp.int32), jax.random.PRNGKey(2)))(w, s)
        assert info["success"].shape == (spec.n_max,) and s2.h.shape == s.h.shape


def test_network_dueling_identity():
    net = M.make_net(3, lstm=16, hid=4)
    params = M.init_params(net, jax.random.PRNGKey(0), 6)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 5, 6))
    q, _ = M.unroll(net, params, x, M.zero_carry(net, (4,)))
    assert q.shape == (4, 5, 3)
    h = jax.random.normal(jax.random.PRNGKey(2), (4, 16))
    q = net.apply(params, h, method=M.DQSANet.heads)
    v = net.apply(params, h, method=M.DQSANet.value)
    np.testing.assert_allclose(np.asarray(q.mean(-1)), np.asarray(v), atol=1e-5)   # Eq. 10 identity


def test_one_train_step_and_eval():
    tc = alg.TrainCfg(reward="prop_fair", iterations=10, episodes=2, horizon=6, lstm=16, hid=4)
    fns = alg.make_train_fns(SPEC, tc)
    qs = fns.init(jax.random.PRNGKey(0))
    qs2, met = jax.jit(lambda q: fns.iterate(q, 0, jax.random.PRNGKey(1)))(qs)
    assert int(qs2.n_updates) == 1 and np.isfinite(float(met["loss"]))
    changed = any(not np.allclose(a, b) for a, b in zip(jax.tree.leaves(qs.params), jax.tree.leaves(qs2.params)))
    assert changed
    r = E.eval_learned(SPEC, tc, qs2.params, n_exp=4, T=8, chunk=4)
    assert r["util_t"].shape == (4, 8)
    for name in ("tdma_genie", "dp"):
        r = E.eval_baseline(FAD if name == "dp" else SPEC, name, n_exp=2, T=5, chunk=2)
        assert r["util_t"].shape == (2, 5)

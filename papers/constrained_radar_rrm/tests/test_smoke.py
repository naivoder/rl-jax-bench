"""Smoke test (< 30 s on CPU): env shapes, one slot per policy, one DDPG and one DQL training step.
JAX_PLATFORMS=cpu pytest tests/test_smoke.py"""
import numpy as np

from src import common  # noqa: F401
import jax
import jax.numpy as jnp

from src import data, env as E, eval as ev, train
from src.common import EnvCfg, load_yaml

CONSTS = dict(snr0=jnp.float64(1.86), s_ref=jnp.float64(5.6), delta=jnp.float64(0.9))


def test_env_shapes_and_one_slot():
    y = load_yaml()
    cfg = EnvCfg.from_yaml(y)
    lut = ev.make_lut(cfg)
    scen = data.replica_case1(cfg)
    es = E.reset(cfg)
    v = E.pre(cfg, scen, es)
    assert v.xp.shape == (5, 4) and v.Pp.shape == (5, 4, 4) and v.present.shape == (12,)
    es2, out = E.post(cfg, CONSTS, lut, scen, es, v, jnp.zeros(5), jax.random.PRNGKey(0))
    assert E.obs_ddpg(es2, cfg, 1e-4, 1e-4).shape == (11,)
    assert E.obs_dql(es2, cfg, 1e-4, 1e-4).shape == (5, 12)
    assert float(out["g"]) == -0.9 and float(es2.lam) == 500.0       # no track: all scan, eq. 33
    assert int(out["n_pres"]) == 1 and np.isfinite(float(out["U"]))


def test_train_steps_both_learners():
    y = load_yaml()
    y["ddpg"]["learning_starts"] = 0
    y["dql"]["learning_starts"] = 0
    y["ddpg"]["batch"] = 8
    y["dql"]["batch"] = 8
    cfg = EnvCfg.from_yaml(y)
    for algo in ("ddpg", "dql"):
        init_lane, lane_step, pol, _ = train.build(algo, y, cfg, CONSTS, 64, 3000)
        lane_c = dict(CONSTS, alpha=jnp.float64(5000.0), lam0=jnp.float64(5000.0), rs=jnp.float64(1e-4),
                      cs=jnp.float64(1e-4), ls=jnp.float64(1e-4))         # per-lane constants (full-run variants)
        ln = init_lane(jax.random.PRNGKey(1), lane_c)
        step = jax.jit(lane_step)
        for i in range(3):
            ln, m = step(ln, jnp.int32(i))
        leaves = jax.tree.leaves(train.learner_params(algo, ln))
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)
        tau = pol(train.learner_params(algo, ln), CONSTS, E.pre(cfg, ln.scen, ln.es), ln.es)
        assert tau.shape == (5,)

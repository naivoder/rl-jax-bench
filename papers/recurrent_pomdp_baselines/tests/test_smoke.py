"""<30 s CPU smoke test: build every arch on tiny networks, collect, update (incl. the diagnostic step), evaluate;
check shapes and finiteness; check seed pairing (arms share env / warm-up streams)."""
import jax
import jax.numpy as jnp
import numpy as np

from src import common  # noqa: F401
from src.common import load_config
from src.train import Runner

TINY = ["obs_embed=4", "action_embed=2", "reward_embed=2", "rnn_hidden=8", "dqn_layers=[8]", "policy_layers=[8]",
        "batch_size=4", "num_init_rollouts=2", "rollouts_per_iter=1", "num_eval_episodes=2", "markov_batch_size=8"]


def _run(cfg_name, extra):
    cfg = load_config(cfg_name, TINY + extra)
    R = Runner(cfg)
    R.compile()
    coef = jnp.ones((2, 2))
    ts = jax.jit(jax.vmap(R.init_state))(jnp.arange(2), coef)
    ts, ret = R.j_collect_random(ts)
    ts, ret = R.j_collect(ts)
    ts, inf = R.j_chunk(2)(ts)
    ts, ev = R.j_eval(ts)
    assert ts["buf"]["count"].tolist() == [3, 3] and ts["n_upd"].tolist() == [2, 2]
    for v in jax.tree_util.tree_leaves((inf, ev, ts["learner"]["params"])):
        assert np.all(np.isfinite(np.asarray(v)))
    assert ev["return"].shape == (2,)
    return R, ts, inf


def test_smoke_separate_shared():
    _, ts_sep, inf = _run("pendulum_v.yaml", ["arch=separate", "algo=sac", "sampled_seq_len=16", "env=Pendulum-V"])
    assert "off_q_rnn" in inf and "clean_pi_rnn" in inf
    _, ts_sh, inf = _run("pendulum_v.yaml", ["arch=shared", "algo=td3", "sampled_seq_len=16", "env=Pendulum-V"])
    assert "off_rnn" in inf and "clean_pi_into_q_heads" in inf
    # seed pairing: the warm-up episodes (env + exploration streams) are identical across arms
    np.testing.assert_array_equal(np.asarray(ts_sep["buf"]["obs"][:, :2]), np.asarray(ts_sh["buf"]["obs"][:, :2]))
    np.testing.assert_array_equal(np.asarray(ts_sep["buf"]["act"][:, :2]), np.asarray(ts_sh["buf"]["act"][:, :2]))


def test_markov_semicircle_builds():
    """(not part of the 30 s budget) the Markovian reference arch on the meta env."""
    _run("semicircle.yaml", ["arch=markov", "algo=sac", "num_eval_episodes=2"])

"""Smoke test (<30 s on CPU): shapes + one train step for every arm on a tiny load-balancing env.

    JAX_PLATFORMS=cpu python -m pytest tests/test_smoke.py -q
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import src  # noqa: E402,F401  (bootstrap before jax)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.data import lb_input_kwargs, sample_lb_batch  # noqa: E402
from src.envs import lb_actions_rollout, lb_rollout, make_lb_spec, reference_event_sim  # noqa: E402
from src.train import ARMS, build, load_config, make_eval, make_train_step  # noqa: E402

TINY = ["jobs_per_episode=20", "n_envs=4", "mv_groups=2", "mv_k=2", "meta_k=4", "n_value_nets=3",
        "test_episodes=3", "hidden=[8,4]", "pg_var_every=1"]


def _cfg():
    return load_config(os.path.join(os.path.dirname(__file__), "..", "src", "configs", "load_balance.yaml"), TINY)


def test_env_shapes_and_reference():
    cfg = _cfg()
    spec = make_lb_spec(cfg)
    inp = sample_lb_batch(jax.random.PRNGKey(0), 3, **lb_input_kwargs(cfg))
    assert inp.arrivals.shape == (3, 20) and inp.sizes.shape == (3, 20)
    tr = lb_rollout(spec, lambda o: jnp.zeros(spec.n_servers), jax.tree_util.tree_map(lambda x: x[0], inp),
                    jax.random.PRNGKey(1))
    assert tr.obs.shape == (20, spec.obs_dim) and tr.cobs.shape == (20, spec.critic_dim)
    assert tr.actions.shape == (20,) and tr.rewards.shape == (20,)
    rng = np.random.default_rng(0)
    for i in range(3):
        acts = rng.integers(0, spec.n_servers, 20)
        r, d = lb_actions_rollout(spec, jnp.asarray(acts, jnp.int32), jax.tree_util.tree_map(lambda x: x[i], inp))
        rr, dd = reference_event_sim(np.asarray(inp.arrivals[i]), np.asarray(inp.sizes[i]), acts, spec.rate_num,
                                     spec.rate_den)
        np.testing.assert_allclose(np.asarray(r), rr, rtol=1e-6)
        assert (np.asarray(d) == dd).all()


def test_input_replay_is_exact():
    """Same key -> identical input sequence, regardless of what the policy does with it."""
    cfg = _cfg()
    a = sample_lb_batch(jax.random.PRNGKey(7), 2, **lb_input_kwargs(cfg))
    b = sample_lb_batch(jax.random.PRNGKey(7), 2, **lb_input_kwargs(cfg))
    assert (np.asarray(a.arrivals) == np.asarray(b.arrivals)).all() and (np.asarray(a.sizes) == np.asarray(b.sizes)).all()


@pytest.mark.parametrize("arm", ARMS)
def test_one_train_step(arm):
    cfg = _cfg()
    ts, consts, models = build(cfg, arm, seed=0)
    step = jax.jit(make_train_step(cfg, arm, models))
    actor0 = jax.tree_util.tree_map(lambda x: x.copy(), ts["actor"])
    ts2, m = step(ts, consts)
    assert int(ts2["it"]) == 1
    for k in ("train_return", "baseline_mse", "entropy", "var_pg", "actor_loss"):
        assert np.isfinite(float(m[k])), k
    moved = sum(float(jnp.sum(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(ts2["actor"]),
                                                                 jax.tree_util.tree_leaves(actor0)))
    assert moved > 0
    sto, gre = make_eval(cfg, models)(ts2["actor"], sample_lb_batch(jax.random.PRNGKey(9), 3, **lb_input_kwargs(cfg)),
                                      jax.random.PRNGKey(2))
    assert sto.shape == (3,) and gre.shape == (3,)


def test_multi_value_only_updates_sampled_nets_and_uses_preupdate_params():
    cfg = _cfg()
    ts, consts, models = build(cfg, "multi", seed=0)
    old = jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), ts["critic"])
    ts2, _ = jax.jit(make_train_step(cfg, "multi", models))(ts, consts)
    leaf_old = jax.tree_util.tree_leaves(old)[0]
    leaf_new = np.asarray(jax.tree_util.tree_leaves(ts2["critic"])[0])
    changed = [not np.allclose(leaf_old[i], leaf_new[i]) for i in range(leaf_old.shape[0])]
    assert sum(changed) == cfg["mv_groups"]  # exactly the sampled nets moved


def test_multi_shuffled_reads_a_mismatched_head():
    """Ablation arm: the baseline of a rollout on fixed sequence i comes from net (i+1) mod N, the training
    of net i is unchanged (same nets move, same rollouts as the multi arm)."""
    cfg = _cfg()
    ts_m, consts, models = build(cfg, "multi", seed=0)
    ts_s, _, models_s = build(cfg, "multi_shuffled", seed=0)
    _, m_m = jax.jit(make_train_step(cfg, "multi", models))(ts_m, consts)
    _, m_s = jax.jit(make_train_step(cfg, "multi_shuffled", models_s))(ts_s, consts)
    assert float(m_m["train_return"]) == float(m_s["train_return"])  # identical rollouts
    assert float(m_m["baseline_mse"]) != float(m_s["baseline_mse"])  # different (mismatched) baseline


@pytest.mark.parametrize("arm", ["state", "multi"])
def test_baseline_uses_preupdate_critic(arm):
    """The baseline of iteration t is computed BEFORE the critic step: the critic's learning rate cannot
    change this iteration's baseline (Alg. 2 pre-update phi_i)."""
    out = []
    for lr in (0.0, 0.5):
        cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "src", "configs", "load_balance.yaml"),
                          TINY + [f"lr_critic={lr}"])
        ts, consts, models = build(cfg, arm, seed=0)
        ts2, m = jax.jit(make_train_step(cfg, arm, models))(ts, consts)
        moved = sum(float(jnp.sum(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(ts2["critic"]),
                                                                     jax.tree_util.tree_leaves(build(cfg, arm, 0)[0]["critic"])))
        out.append((float(m["baseline_mse"]), moved))
    assert out[0][0] == out[1][0]  # identical baseline
    assert out[0][1] == 0.0 and out[1][1] > 0.0  # but the critic itself did move at lr 0.5


def test_multi_official_schedule_one_adam_step_per_agent():
    """Recheck: mv_schedule=official -> n_envs agents, net index per agent drawn WITH replacement, one Adam
    step per agent on its net (a net drawn twice takes two steps, an undrawn net does not move)."""
    from src.data import KeyTree
    from src.train import batch_shape

    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "src", "configs", "load_balance.yaml"),
                      TINY + ["mv_schedule=official"])
    assert batch_shape(cfg, "multi") == (cfg["n_envs"], 1)
    for seed in range(4):
        ts, consts, models = build(cfg, "multi", seed=seed)
        old = jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), ts["critic"])
        idx = np.asarray(jax.random.choice(jax.random.fold_in(KeyTree(*consts["keys"]).sched, 0),
                                           cfg["n_value_nets"], (cfg["n_envs"],), replace=True))
        ts2, m = jax.jit(make_train_step(cfg, "multi", models))(ts, consts)
        assert np.isfinite(float(m["baseline_mse"]))
        mult = np.bincount(idx, minlength=cfg["n_value_nets"])
        counts = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(ts2["critic_opt"])
                  if np.asarray(leaf).shape == (cfg["n_value_nets"],) and np.issubdtype(np.asarray(leaf).dtype, np.integer)]
        assert counts and np.array_equal(counts[0], mult), (counts, mult)
        leaf_old = jax.tree_util.tree_leaves(old)[0]
        leaf_new = np.asarray(jax.tree_util.tree_leaves(ts2["critic"])[0])
        for i in range(cfg["n_value_nets"]):
            assert np.allclose(leaf_old[i], leaf_new[i]) == (mult[i] == 0)

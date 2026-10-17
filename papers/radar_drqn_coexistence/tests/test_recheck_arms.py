"""Recheck (2026-10-16) sensitivity arms: plain SGD, the Keras full-target-vector loss scale, the
per-trace update cadence, the DQN output ReLU and the PI strict-improvement guard. Every flag
defaults to the shipped behaviour. CPU, a few seconds."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dataclasses  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import algorithm as alg  # noqa: E402
import baselines as bl  # noqa: E402
import data as D  # noqa: E402
import train as TR  # noqa: E402
from model import arm_config  # noqa: E402

CFG = TR.load_cfg(os.path.join(os.path.dirname(__file__), "..", "src", "configs", "full.yaml"))


def test_shipped_defaults_unchanged():
    for arm in ("dqn", "ddrqn"):
        q = arm_config(arm, CFG)
        assert q.optimizer == "adam" and q.loss_scale == 1.0 and q.update_every == 1 and q.output_relu is False


def test_audit_arms_differ_only_in_their_flag():
    dqn, ddrqn = arm_config("dqn", CFG).__dict__, arm_config("ddrqn", CFG).__dict__
    assert arm_config("dqn_sgd", CFG).__dict__ == dqn | {"optimizer": "sgd"}
    assert arm_config("ddrqn_sgd", CFG).__dict__ == ddrqn | {"optimizer": "sgd"}
    assert arm_config("dqn_sgd_k15", CFG).__dict__ == dqn | {"optimizer": "sgd", "loss_scale": 1 / 15}
    assert arm_config("ddrqn_sgd_k15", CFG).__dict__ == ddrqn | {"optimizer": "sgd", "loss_scale": 1 / 15}
    assert arm_config("ddrqn_upd10", CFG).__dict__ == ddrqn | {"update_every": 10}
    assert arm_config("dqn_orelu", CFG).__dict__ == dqn | {"output_relu": True}


def _batch(key, B=8, L=1):
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(inputs=jax.random.uniform(k1, (B, L + 1, D.OBS_DIM)), actions=jax.random.randint(k2, (B, L), 0, D.N_ACT),
                rewards=jax.random.uniform(k3, (B, L)), discounts=jnp.full((B, L), 0.9))


def test_sgd_step_is_eq9():
    cfg = arm_config("dqn_sgd", CFG)
    st = alg.init(jax.random.PRNGKey(0), cfg)
    batch = _batch(jax.random.PRNGKey(1))
    g = jax.grad(lambda p: alg.sequence_loss(p, st.target_params, batch, cfg)[0])(st.params)
    st2, _ = alg.update(st, batch, cfg)
    for w, w2, gw in zip(jax.tree.leaves(st.params), jax.tree.leaves(st2.params), jax.tree.leaves(g)):
        np.testing.assert_allclose(w2, w - cfg.lr * gw, rtol=1e-6, atol=1e-8)


def test_loss_scale_is_keras_full_vector_mse():
    """Keras model.fit(x, Y_full) with Y_full = prediction except at the taken action: MSE = mean over the batch of the
    mean over the 15 outputs = (1/15) x our taken-action mean."""
    base = arm_config("dqn", CFG)
    k15 = dataclasses.replace(base, loss_scale=1 / 15)
    st = alg.init(jax.random.PRNGKey(2), base)
    batch = _batch(jax.random.PRNGKey(3), B=16)
    l1, aux = alg.sequence_loss(st.params, st.target_params, batch, base)
    l15, _ = alg.sequence_loss(st.params, st.target_params, batch, k15)
    net = alg.make_network(base)
    q, _ = alg.unroll(net, base, st.params, batch["inputs"], alg.initial_carry(base, (16,)))
    q0 = q[:, 0]
    y_full = q0.at[jnp.arange(16), batch["actions"][:, 0]].set(aux["y"][:, 0])
    keras = jnp.mean(jnp.mean((y_full - q0) ** 2, axis=-1))
    np.testing.assert_allclose(l15, l1 / 15, rtol=1e-6)
    np.testing.assert_allclose(l15, keras, rtol=1e-5)


def test_update_every_gates_learning():
    lanes, _ = TR.build_lanes(dict(CFG, seed_root=3, eval_cpis=10), ["mc_p04"], ["eq3"], [0])
    lane = jax.tree.map(lambda x: x[0], lanes)
    for arm, expect in (("ddrqn", (1.0, 1.0)), ("ddrqn_upd10", (0.0, 1.0))):
        qcfg = arm_config(arm, CFG)
        c = TR.init_carry(lane, qcfg)
        flags = []
        for t in (328, 329):     # (t+1) // 10 >= 32 -> can learn; (t+1) % 10 == 0 only at t = 329
            _, out, _, _ = TR.lane_step(c, jnp.int32(t), lane.params.T_train, lane, qcfg, 100_000, True)
            flags.append(float(out[7]))
        assert tuple(flags) == expect, (arm, flags)


def test_pi_strict_improvement_guard():
    """On a model with exact Q ties the guard keeps the incumbent action; without ties both forms agree."""
    rng = np.random.default_rng(0)
    n = 4000
    s = rng.integers(0, 50, n)
    a = rng.integers(0, D.N_ACT, n)
    s2 = rng.integers(0, 50, n)
    r = np.round(rng.random(n), 1)
    old = bl.policy_iteration(s, a, r, s2, 0.9)
    new = bl.policy_iteration(s, a, r, s2, 0.9, strict_improvement=True)
    assert new["iters"] <= old["iters"]
    seen = np.zeros(bl.N_FULL, bool)
    seen[s] = True
    # both are greedy w.r.t. their own V up to ties: their values agree on seen states
    np.testing.assert_allclose(new["V"][seen], old["V"][seen], atol=1e-8)

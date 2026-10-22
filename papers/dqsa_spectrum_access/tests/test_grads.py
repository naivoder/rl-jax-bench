"""Gradient correctness.  DQSA is a gradient method (TD regression on a dueling LSTM), so the test is a float64 finite-difference
check_grads (orders 1-2, forward + reverse) of the full DQSA TD loss (drqn.sequence_loss with the dueling-LSTM unroll hook) on a
toy batch, plus the analytic Jacobians of the dueling aggregation (dueling Jacobian identity) and an independent hand computation of the loss value.
float64 is enabled with a context manager so other tests in the same session keep float32."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()   # persistent compile cache (a warm cache keeps the smoke test under 30 s)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import enable_x64  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from rl_jax_bench import drqn  # noqa: E402
import model as M  # noqa: E402


def _toy(B=3, T=4, D=6, A=3, H=5):
    net = M.make_net(A, lstm=H, hid=4)
    params = M.init_params(net, jax.random.PRNGKey(0), D, jnp.float64)
    params = jax.tree.map(lambda p: p.astype(jnp.float64), params)
    tparams = jax.tree.map(lambda p: p * 0.9 + 0.01, params)
    k = jax.random.split(jax.random.PRNGKey(1), 3)
    batch = dict(inputs=jax.random.normal(k[0], (B, T + 1, D), jnp.float64),
                 actions=jax.random.randint(k[1], (B, T), 0, A),
                 rewards=jax.random.uniform(k[2], (B, T), jnp.float64),
                 discounts=jnp.full((B, T), 0.95, jnp.float64), mask=jnp.ones((B, T), jnp.float64).at[0, -1].set(0.0),
                 carry0=M.zero_carry(net, (B,), jnp.float64))
    cfg = drqn.QConfig(n_actions=A, obs_dim=D, hidden=(), lstm=H, double=True, gamma=0.95)
    return net, params, tparams, batch, cfg


def test_check_grads_dueling_lstm_td_loss():
    with enable_x64():
        net, params, tparams, batch, cfg = _toy()
        uf = M.make_unroll_fn(net)

        def loss(p):
            return drqn.sequence_loss(p, tparams, batch, cfg, uf)[0]

        assert jax.tree.leaves(params)[0].dtype == jnp.float64
        check_grads(loss, (params,), order=2, modes=["rev"], atol=1e-5, rtol=1e-5)
        check_grads(loss, (params,), order=1, modes=["fwd"], atol=1e-5, rtol=1e-5)


def test_loss_value_matches_numpy_by_hand():
    """Independent re-computation of the double-Q TD loss (/scalar TD loss) from the network's own q values with numpy loops."""
    with enable_x64():
        net, params, tparams, batch, cfg = _toy()
        uf = M.make_unroll_fn(net)
        loss, aux = drqn.sequence_loss(params, tparams, batch, cfg, uf)
        q_on = np.asarray(uf(params, batch["inputs"], batch["carry0"])[0])
        q_tg = np.asarray(uf(tparams, batch["inputs"], batch["carry0"])[0])
        act, rew, disc, m = (np.asarray(batch[k]) for k in ("actions", "rewards", "discounts", "mask"))
        tot, cnt = 0.0, 0.0
        for b in range(act.shape[0]):
            for t in range(act.shape[1]):
                a_star = int(np.argmax(q_on[b, t + 1]))
                y = rew[b, t] + disc[b, t] * q_tg[b, t + 1, a_star]
                tot += m[b, t] * (q_on[b, t, act[b, t]] - y) ** 2
                cnt += m[b, t]
        assert abs(float(loss) - tot / cnt) < 1e-10


def test_dueling_jacobian_D4():
    """dQ(a)/dV = 1 and dQ(a)/dA(b) = delta_ab - 1/(K'+1)  (dueling Jacobian identity)."""
    A = 4
    V = jnp.array([0.7])
    Av = jnp.array([0.1, -0.3, 0.5, 0.2])
    f = lambda v, a: v + a - a.mean()
    jV = jax.jacobian(f, 0)(V, Av)[:, 0]
    jA = jax.jacobian(f, 1)(V, Av)
    np.testing.assert_allclose(np.asarray(jV), np.ones(A), atol=1e-7)
    np.testing.assert_allclose(np.asarray(jA), np.eye(A) - 1.0 / A, atol=1e-7)


def test_first_step_full_vector_mse_equals_scalar_td_D6_D7():
    """Full-vector gradient == scalar TD gradient: at theta == theta1 the gradient of the full-vector regression onto Qhat (Q1 with the taken entry
    replaced by y) equals the gradient of the scalar TD loss on the taken action, up to the 1/(K'+1) factor of the mean."""
    with enable_x64():
        net, params, tparams, batch, cfg = _toy()
        uf = M.make_unroll_fn(net)
        B, T = batch["actions"].shape
        A = cfg.n_actions
        q1 = uf(params, batch["inputs"], batch["carry0"])[0]
        qt = uf(tparams, batch["inputs"], batch["carry0"])[0]
        astar = jnp.argmax(q1[:, 1:], -1)
        y = batch["rewards"] + batch["discounts"] * jnp.take_along_axis(qt[:, 1:], astar[..., None], -1)[..., 0]
        onehot = jax.nn.one_hot(batch["actions"], A, dtype=jnp.float64)
        qhat = jax.lax.stop_gradient(q1[:, :-1] * (1 - onehot) + y[..., None] * onehot)
        m = batch["mask"]

        def d6(p):
            q = uf(p, batch["inputs"], batch["carry0"])[0][:, :-1]
            return (m[..., None] * (q - qhat) ** 2).sum() / (m.sum() * A)

        g6 = jax.grad(d6)(params)
        g7 = jax.grad(lambda p: drqn.sequence_loss(p, tparams, batch, cfg, uf)[0])(params)
        for a, b in zip(jax.tree.leaves(g6), jax.tree.leaves(g7)):
            np.testing.assert_allclose(np.asarray(a) * A, np.asarray(b), atol=1e-10, rtol=1e-8)

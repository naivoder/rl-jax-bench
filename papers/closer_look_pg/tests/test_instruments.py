"""Instrument regression tests (CPU, ~1 min): the landscape true reward spends the whole pair budget, the un-centred zero arm really has no baseline, the per-minibatch PPO surface, the eval run-name
parser and the E5 value target."""
import jax
import jax.numpy as jnp
import numpy as np

from src import common  # noqa: F401
from src import algorithm as A
from src import eval as E
from src.configs import default as C
from src.train import batch_from_traj


def test_true_reward_uses_every_completed_episode():
    ctx = E.Ctx("hopper", "ppo")
    ts = ctx.init_state(jax.random.PRNGKey(0))
    norm = jax.tree_util.tree_map(lambda x: x, E.D.norm_init(ctx.spec.obs_dim))
    keys = jax.random.split(jax.random.PRNGKey(5), 1)
    T, pairs = 200, 400                                           # 2 actors x 200 steps; random-init episodes ~20 steps
    out = jax.jit(lambda p, k: E.true_reward(ctx, p, norm, k, pairs, T))(ts.pi[None], keys)
    tr = E.collect(ctx, ts.pi, norm, jax.random.split(keys[0], 2), T)          # the same actors, re-rolled
    done = np.asarray(tr.not_done) < 0.5
    assert np.array_equal(np.asarray(out["all_cnt"])[0], done.sum(1))        # every completed episode counted
    used = np.asarray(out["all_pairs"])[0]
    assert np.all(used <= T) and np.all(used == np.where(done, np.asarray(tr.ep_len), 0).sum(1))
    assert np.all(np.asarray(out["all_cnt"]) > np.asarray(out["k_cnt"]))     # the old estimator kept 1 per actor here
    m, se, c = E.reduce_true(out["all_sum"], out["all_sq"], out["all_cnt"], 2)
    assert abs(float(m[0]) - float(np.asarray(tr.ep_ret)[done].mean())) < 1e-3 * max(1.0, abs(float(m[0])))


def _toy_batch(nets, pi, n=128, seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 3)
    obs = jax.random.normal(k[0], (n, 11)); act = jax.random.normal(k[1], (n, 3))
    adv = 3.0 + jax.random.normal(k[2], (n,))                     # large common offset: a constant baseline matters
    return A.Batch(obs, act, nets.logp(pi, obs, act), adv, adv, jnp.zeros(n), jnp.ones(n))


def test_zero_raw_is_uncentred():
    from src.model import Nets
    cfg = C.algo_config("ppo", "hopper")
    nets = Nets(11, 3, cfg)
    pi, _ = nets.init(jax.random.PRNGKey(1))
    b = _toy_batch(nets, pi)
    g_raw = A.pg_vector(nets, pi, b.obs, b.act, b.adv, normalize=False)
    g_cen = A.pg_vector(nets, pi, b.obs, b.act, b.adv, normalize=True)
    g_manual = A.pg_vector(nets, pi, b.obs, b.act, b.adv - b.adv.mean(), normalize=False)
    cos = lambda x, y: float(jnp.dot(x, y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y)))
    assert cos(g_cen, g_manual) > 0.9999                          # 'zero' = constant batch-mean baseline
    assert cos(g_raw, g_cen) < 0.99                               # 'zero_raw' differs: no baseline at all


def test_per_minibatch_surrogate():
    from src.model import Nets
    cfg = C.algo_config("ppo", "hopper")
    nets = Nets(11, 3, cfg)
    pi, _ = nets.init(jax.random.PRNGKey(2))
    b = _toy_batch(nets, pi)
    d = 0.05 * jax.random.normal(jax.random.PRNGKey(3), pi.shape)
    u = 0.05 * jax.random.normal(jax.random.PRNGKey(4), pi.shape)
    al = jnp.linspace(0.0, 1.0, 3)
    S = A.surrogate_grid(nets, "ppo", 0.2, pi, d, u, al, al, b)
    whole = jnp.arange(128)[None]                                 # one minibatch = the whole batch -> identical surface
    S1 = A.surrogate_grid(nets, "ppo", 0.2, pi, d, u, al, al, b, mb_idx=whole)
    assert np.allclose(np.asarray(S), np.asarray(S1), atol=1e-6)
    idx = A._minibatch_perms(jax.random.PRNGKey(5), 128, 2, 4)
    Smb = A.surrogate_grid(nets, "ppo", 0.2, pi, d, u, al, al, b, mb_idx=idx)
    assert abs(float(Smb[0, 0])) < 1e-6 and not np.allclose(np.asarray(S), np.asarray(Smb))


def test_e5_value_target_is_gae():
    cfg = C.algo_config("ppo", "hopper")
    cfg["adv_baseline"] = "zero"
    from src.model import Nets
    nets = Nets(11, 3, cfg)
    pi, v = nets.init(jax.random.PRNGKey(0))
    T = 32
    traj = E.D.Traj(*(jnp.zeros((T, 11)) if i == 0 else jnp.zeros((T, 3)) if i == 1 else
                      (jnp.ones(T) if i in (3, 4, 5) else jnp.zeros(T)) for i in range(10)))
    b = batch_from_traj(nets, cfg, v, traj)
    adv_gae, ret = A.gae(traj.rew, nets.v_apply(v, traj.obs), traj.not_done, cfg["gamma"], cfg["gae_lambda"])
    assert np.allclose(np.asarray(b.adv), np.asarray(ret))       # policy: A := G_t
    assert np.allclose(np.asarray(b.vadv), np.asarray(adv_gae))  # value target: V + A^GAE


def test_run_name_parser():
    import pytest
    for name, want in (("runs/full/hopper_ppo", ("hopper", "ppo")), ("runs/x/walker2d_ppo-nobaseline/", ("walker2d", "ppo")),
                       ("hopper_trpo", ("hopper", "trpo")), ("hopper_ppo-nobaseline-nocenter", ("hopper", "ppo"))):
        assert E.parse_run_name(name) == want
    with pytest.raises(ValueError):
        E.parse_run_name("runs/hopper_sac")


def test_start_estimator_selection():
    """Recheck the 'start' estimator keeps every episode that started at step <= T - epl (always complete),
    and drops the later-starting ones even when they happened to complete (the length-biased ones)."""
    import jax.numpy as jnp
    from src.eval import start_selected
    # T = 10, epl = 4: episodes [0..2] (len 3), [3..6] (len 4), [7..8] (len 2), [9] truncated
    lens = jnp.asarray([1, 2, 3, 1, 2, 3, 4, 1, 2, 1], jnp.float32)
    nd = jnp.asarray([1, 1, 0, 1, 1, 1, 0, 1, 0, 1], jnp.float32)
    sel = np.asarray(start_selected(nd, lens, 4))
    # ends at t=2 (start 0) and t=6 (start 3 <= 6) kept; end at t=8 starts at 7 > 6 -> dropped
    assert sel.tolist() == [False, False, True, False, False, False, True, False, False, False]

"""Smoke test (<30 s on CPU): shapes + one step for every policy, chunked runner, CRN, orbax resume.

    JAX_PLATFORMS=cpu uv run python -m pytest tests/test_smoke.py -q
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import src  # noqa: E402,F401  (bootstrap before jax)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import algorithm as alg  # noqa: E402
from src import data  # noqa: E402
from src.train import run_policy, run_experiment, run_experiment_seeds  # noqa: E402

K = 3


def _env(T=200):
    return data.abrupt(T, [[.5, .3, .4], [.5, .9, .4], [.5, .3, .4]], [60, 100])


def _params(name):
    return {"ucb1": alg.make_params(), "oracle_ucb1": alg.make_params(), "ducb": alg.make_params(gamma=0.95), "ducb_lit": alg.make_params(gamma=0.95),
            "swucb": alg.make_params(tau=30), "exp3s": alg.make_params(g=0.1, alpha=1e-3)}[name]


@pytest.mark.parametrize("name", alg.POLICIES)
def test_one_step_shapes(name):
    st = alg.init_state(name, K, 32)
    acc = alg.init_acc(K)
    step = alg.make_step(name, K, "random")
    k = jax.random.PRNGKey(0)
    mu = jnp.asarray([.5, .3, .4])
    st2, acc2, a, aux = step(st, acc, jnp.int32(1), mu, jnp.asarray(False), k, k, _params(name))
    assert a.shape == () and a.dtype == jnp.int32 and 0 <= int(a) < K
    assert aux.shape == (K,)
    assert jax.tree_util.tree_structure(st2) == jax.tree_util.tree_structure(st)
    for x, y in zip(jax.tree_util.tree_leaves(st2), jax.tree_util.tree_leaves(st)):
        assert x.shape == y.shape and x.dtype == y.dtype
    assert acc2.regret.dtype == jnp.float64  # 
    if name != "exp3s":
        assert int(a) == 0  # round-robin initialisation plays arm 1 at t = 1


@pytest.mark.parametrize("name", alg.POLICIES)
def test_chunked_runner_shapes_and_chunk_invariance(name):
    env = _env()
    R = 4
    k_env, k_pol = alg.rep_keys(0, 99, 0, alg.POLICY_ID[name], R)
    one = run_policy(name, env, _params(name), 32, k_env, k_pol, tie="random", log_every=10)
    two = run_policy(name, env, _params(name), 32, k_env, k_pol, tie="random", log_every=10, chunk_steps=100)
    assert one["reg_curves"].shape == (R, 20) and one["arm1_curves"].shape == (R, 20)
    np.testing.assert_array_equal(one["reg_curves"], two["reg_curves"])
    np.testing.assert_array_equal(one["acc"].arm1, two["acc"].arm1)
    assert np.all(np.diff(one["reg_curves"], axis=1) >= 0)  # pseudo-regret is non-decreasing
    assert not one["acc"].bad.any()


def test_crn_and_experiment_record(tmp_path):
    cfg = {"name": "smoke", "exp_id": 99, "log_every": 10, "report_t": [100],
           "env": {"kind": "abrupt", "T": 200, "breakpoints": [60, 100],
                   "segments": [[.5, .3, .4], [.5, .9, .4], [.5, .3, .4]]},
           "policies": {"ucb1": {"xi": 0.5}, "exp3s": {"S": 2, "alpha": 0.005}, "ducb": {"xi": 0.5, "gamma": 0.95},
                        "swucb": {"xi": 0.5, "tau": 30}, "oracle_ucb1": {"xi": 0.5}}}
    rec = run_experiment(cfg, 0, 8, out_dir=str(tmp_path))
    assert rec["crn_identical"]
    assert os.path.exists(tmp_path / "seed_0" / "metrics.json")
    assert len(rec["policies"]["swucb"]["final_regret"]) == 8
    assert len(rec["policies"]["ducb"]["curve"]["mean"]) == 21
    rec1 = run_experiment(cfg, 1, 8, policies=["ucb1"])
    assert rec1["env"]["uniforms_fp_sha256"] != rec["env"]["uniforms_fp_sha256"]  # seeds differ


def test_batched_seeds_identical_to_sequential():
    cfg = {"name": "smoke", "exp_id": 99, "log_every": 10,
           "env": {"kind": "abrupt", "T": 200, "breakpoints": [60, 100],
                   "segments": [[.5, .3, .4], [.5, .9, .4], [.5, .3, .4]]},
           "policies": {"exp3s": {"S": 2, "alpha": 0.005}, "swucb": {"xi": 0.5, "tau": 30},
                        "ducb_g1": {"algo": "ducb", "xi": 0.5, "gamma": 1.0}}}
    batched = run_experiment_seeds(cfg, [3, 4], 6)
    for rb in batched:
        rs = run_experiment(cfg, rb["seed"], 6)
        assert rb["env"]["uniforms_fp_sha256"] == rs["env"]["uniforms_fp_sha256"]
        for n in cfg["policies"]:
            np.testing.assert_array_equal(rb["policies"][n]["final_regret"], rs["policies"][n]["final_regret"])
            np.testing.assert_array_equal(rb["policies"][n]["final_arm1"], rs["policies"][n]["final_arm1"])
        assert rb["crn_identical"]


def test_orbax_resume_bit_identical(tmp_path):
    env = _env(400)
    R = 3
    name = "swucb"
    k_env, k_pol = alg.rep_keys(0, 98, 0, alg.POLICY_ID[name], R)
    full = run_policy(name, env, _params(name), 32, k_env, k_pol, tie="random", log_every=10)
    ck = str(tmp_path / "ck")
    # simulate a killed job: run only the first chunk by truncating the env, keeping its checkpoint
    run_policy(name, env, _params(name), 32, k_env, k_pol, tie="random", log_every=10, chunk_steps=100, ckpt_dir=ck)
    import shutil
    import orbax.checkpoint as ocp
    mgr = ocp.CheckpointManager(ck)
    steps = sorted(mgr.all_steps())
    mgr.close()
    for s in steps[1:]:  # drop every checkpoint after round 101 -> resume restarts from t = 101
        shutil.rmtree(os.path.join(ck, str(s)))
    res = run_policy(name, env, _params(name), 32, k_env, k_pol, tie="random", log_every=10, chunk_steps=100, ckpt_dir=ck)
    np.testing.assert_array_equal(full["reg_curves"], res["reg_curves"])
    np.testing.assert_array_equal(full["acc"].regret, res["acc"].regret)


def test_exp3s_eta_mult_default_bit_identical():
    """Recheck RC3: eta_mult = 1 reproduces the pre-RC3 EXP3.S update bit for bit; eta_mult != 1 changes it."""
    import math
    rng = np.random.default_rng(0)

    def old_update(logw, a, x, p, g, alpha, K):  # verbatim pre-RC3 formula
        xhat = jnp.where(jnp.arange(K) == a, x / p[a], 0.0)
        share = jnp.log(math.e * alpha / K) + jax.scipy.special.logsumexp(logw)
        new = jnp.logaddexp(logw + g * xhat / K, share)
        return new - jnp.max(new)

    for _ in range(50):
        logw = jnp.asarray(rng.normal(size=K))
        p = jnp.asarray(rng.dirichlet(np.ones(K)))
        a, x, g = jnp.int32(rng.integers(K)), jnp.float64(rng.integers(2)), jnp.float64(rng.uniform(0.01, 0.5))
        ref = old_update(logw, a, x, p, g, 1e-4, K)
        assert np.array_equal(np.asarray(alg.exp3s_update_logw(logw, a, x, p, g, 1e-4, K)), np.asarray(ref))
        assert np.array_equal(np.asarray(alg.exp3s_update_logw(logw, a, x, p, g, 1e-4, K, jnp.float64(1.0))), np.asarray(ref))
    env = _env(400)
    ke, kp = alg.rep_keys(0, 1, 0, alg.POLICY_ID["exp3s"], 8)
    base = run_policy("exp3s", env, alg.make_params(g=0.1, alpha=1e-3), 1, ke, kp, tie="random", log_every=100)
    same = run_policy("exp3s", env, alg.make_params(g=0.1, alpha=1e-3, eta_mult=1.0), 1, ke, kp, tie="random", log_every=100)
    diff = run_policy("exp3s", env, alg.make_params(g=0.1, alpha=1e-3, eta_mult=4.0), 1, ke, kp, tie="random", log_every=100)
    assert np.array_equal(base["acc"].regret, same["acc"].regret)
    assert not np.array_equal(base["acc"].regret, diff["acc"].regret)

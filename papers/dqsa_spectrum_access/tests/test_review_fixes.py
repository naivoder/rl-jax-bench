"""Tests for the review fixes (CPU, tiny): feasible-matching metric, rate-weighted rewards, previous-slot OCA/DP,
fixed subsets, monitor-world offset, launch provenance and same-group determinism."""
import itertools
import json
import os
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import maximum_bipartite_matching  # noqa: E402

import algorithm as alg  # noqa: E402
import data as D  # noqa: E402
import eval as E  # noqa: E402
import train  # noqa: E402


def _brute_matching(subset, active, free):
    N, Kp = subset.shape
    K = len(free)
    rows, cols = [], []
    for n in range(N):
        if not active[n]:
            continue
        for c in subset[n]:
            if free[c]:
                rows.append(n)
                cols.append(int(c))
    if not rows:
        return 0
    g = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, K))
    m = maximum_bipartite_matching(g, perm_type="column")
    return int((m >= 0).sum())


def test_max_feasible_equals_bipartite_matching():
    rng = np.random.default_rng(0)
    for (N, K, pu) in [(6, 4, False), (6, 4, True), (12, 10, True), (9, 12, True), (20, 10, False), (30, 15, True)]:
        spec = D.EnvSpec(name="m", n_min=N, n_max=N, n_channels=K, k_local=2, subset="random", pu=pu)
        f = jax.jit(lambda w, pf: D.max_feasible(spec, w, pf))
        for i in range(25):
            w = D.sample_world(spec, jax.random.PRNGKey(i + 1000 * N))
            free = rng.random(K) < 0.5 if pu else np.ones(K, bool)
            got = int(f(w, jnp.asarray(free)))
            assert got == _brute_matching(np.asarray(w.subset), np.asarray(w.active), free), (N, K, pu, i)


def test_max_feasible_all_subset():
    spec = D.EnvSpec(name="a", n_min=3, n_max=5, n_channels=2, k_local=2)
    w = D.sample_world(spec, jax.random.PRNGKey(0))
    assert int(D.max_feasible(spec, w, jnp.ones((2,), bool))) == min(int(w.n_active), 2)
    assert int(D.max_feasible(spec, w, jnp.array([True, False]))) == 1


def test_paper_scale_coverage_is_below_one():
    """evidence: with random 2-subsets at N=100, K=50 only ~41 % of worlds cover every channel (code's own sample_world)."""
    spec = D.EnvSpec(name="e2", n_min=100, n_max=100, n_channels=50, k_local=2, subset="random")
    keys = D.world_keys(jax.random.PRNGKey(0), 400)
    sub = np.asarray(jax.vmap(lambda k: D.sample_world(spec, k).subset)(keys))
    cov = np.array([len(np.unique(s)) == 50 for s in sub]).mean()
    assert 0.3 < cov < 0.55, cov
    fixed = D.EnvSpec(name="e2f", n_min=100, n_max=100, n_channels=50, k_local=2, subset="fixed")
    fs = np.asarray(D.sample_world(fixed, jax.random.PRNGKey(5)).subset)
    assert len(np.unique(fs)) == 50 and (fs[:, 0] != fs[:, 1]).all()
    np.testing.assert_array_equal(fs, np.asarray(D.sample_world(fixed, jax.random.PRNGKey(6)).subset))   # episode-independent


def test_rate_rewards_equal_count_rewards_under_constant_capacity():
    rng = np.random.default_rng(1)
    succ = jnp.asarray(rng.random((8, 5)) < 0.4)
    active = jnp.array([True, True, True, True, False])
    rate = (succ & active).astype(jnp.float32)                          # constant capacity: 1 per success
    succ_after = jnp.cumsum((succ & active).astype(jnp.float32), 0)
    for old, new in (("competitive", "rate_comp"), ("sum_rate", "rate_sum"), ("prop_fair", "rate_pf")):
        want = jnp.stack([alg.reward_vec(old, succ[t], active, succ_after[t]) for t in range(8)])
        got = alg.reward_rate(new, rate * 1.0, active)
        if new == "rate_comp":
            got = got * D.C_ERGODIC_MBPS
        if new == "rate_sum":
            got = got * D.C_ERGODIC_MBPS
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-5, err_msg=new)


def test_rate_pf_weights_by_rate():
    rate = jnp.array([[10.0, 0.0], [10.0, 30.0]])                       # two slots, two users
    r = alg.reward_rate("rate_pf", rate, jnp.array([True, True]))
    np.testing.assert_allclose(np.asarray(r[:, 0]), [1.0, 0.5 + 1.0], atol=1e-6)   # slot1: 10/10; slot2: 10/20 + 30/30


def test_oca_dp_previous_slot_information():
    spec = D.EnvSpec(name="f", n_min=8, n_max=8, n_channels=4, k_local=2, subset="random", capacity="rate", fading=True, pu=True)
    for name, kw in (("oca", {}), ("dp", {}), ("dp", dict(p_update=0.3, tau=0.5)), ("oca_sameslot", {}), ("dp_sameslot", {})):
        r = E.eval_baseline(spec, name, n_exp=3, T=6, chunk=3, **kw)
        assert r["util_t"].shape == (3, 6) and np.isfinite(r["util_t"]).all() and np.isfinite(r["feasible_t"]).all()
        assert (r["util_t"] * 4 <= r["feasible_t"] + 1e-6).all()        # successes never exceed the feasible matching size
    # announcement packing: choice = k * nP + j (decodes to a channel < K and a grid index < nP)
    k = alg.dp()(jax.random.PRNGKey(0), spec, D.sample_world(spec, jax.random.PRNGKey(1)), D.init_state(spec, jax.random.PRNGKey(2)), 0)
    assert len(k) == 3 and int(k[2].max()) < 4 * len(alg.P_GRID) and (np.asarray(k[2]) // len(alg.P_GRID) == np.asarray(k[1])).all()


def test_oca_uses_previous_slot_counts():
    spec = D.EnvSpec(name="c", n_min=4, n_max=4, n_channels=1, k_local=1)
    w = D.sample_world(spec, jax.random.PRNGKey(0))
    s = D.init_state(spec, jax.random.PRNGKey(1))
    s = s._replace(choice=jnp.zeros((4,), jnp.int32))                   # all four announced channel 0 last slot -> p = 1/4
    pol = alg.oca()
    tx = np.mean([np.asarray(pol(jax.random.PRNGKey(i), spec, w, s, 0)[0]) for i in range(2000)])
    assert abs(tx - 0.25) < 0.02
    s0 = s._replace(choice=-jnp.ones((4,), jnp.int32))                  # nothing announced yet -> p = 1
    assert np.asarray(pol(jax.random.PRNGKey(0), spec, w, s0, 0)[0]).all()


def test_monitor_worlds_are_disjoint_from_reporting_worlds():
    rep, _ = E.eval_keys(1000, 0, 0)
    mon, _ = E.eval_keys(64, 0, E.MONITOR_WORLD_OFFSET)
    rep_set = {tuple(np.asarray(k).tolist()) for k in rep}
    assert not any(tuple(np.asarray(k).tolist()) in rep_set for k in mon)


COMMON = ["--exp", "E0", "--chunk", "10", "--ckpt-every", "10", "--monitor-every", "1000", "--n-exp", "8", "--lstm", "8", "--sched-iters", "20",
          "--iters", "20"]


def test_same_group_is_bit_identical_and_provenance_recorded():
    with tempfile.TemporaryDirectory() as d:
        train.main(COMMON + ["--seeds", "1", "2", "3", "--out", os.path.join(d, "a")])
        train.main(COMMON + ["--seeds", "1", "2", "3", "--out", os.path.join(d, "b")])
        tag = os.listdir(os.path.join(d, "a"))[0]
        for s in (1, 2, 3):
            a = np.load(os.path.join(d, "a", tag, f"seed_{s}", "params.npz"))
            b = np.load(os.path.join(d, "b", tag, f"seed_{s}", "params.npz"))
            for k in a.files:
                assert np.array_equal(a[k], b[k]), (s, k)               # same group + same code => bit-identical
        m = json.load(open(os.path.join(d, "a", tag, "seed_2", "metrics.json")))
        for key in ("git_commit", "git_dirty", "vmapped_group_seeds", "env_sha", "algo_sha", "eval_sha", "launch_provenance"):
            assert key in m, key
        assert m["vmapped_group_seeds"] == [1, 2, 3] and m["launch_provenance"]["n_vmapped_seeds"] == 3
        assert os.path.exists(os.path.join(d, "a", tag, "launch.json"))


def test_new_arms_train_and_eval_smoke():
    """The declared companion arms (rate-weighted reward on the fixed-subset proxy, CSI-input proxy with PUs, E1K2) run end to end: 2 iterations, tiny net, 4 eval worlds."""
    import yaml
    cfgs = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "src", "configs", "experiments.yaml")))
    for exp, reward in (("E2proxyfixed", "rate_pf"), ("E3proxycsi", "rate_comp"), ("E1K2", "competitive"), ("E2proxy", "rate_sum")):
        spec = D.EnvSpec(**cfgs[exp]["env"])
        if spec.n_max > 12:                                           # shrink the proxies for the CPU smoke test; the code path is the same
            spec = D.EnvSpec(**{**cfgs[exp]["env"], "n_min": 6, "n_max": 6, "n_channels": 4})
        tc = alg.TrainCfg(reward=reward, iterations=4, episodes=2, horizon=5, lstm=8, hid=4)
        fns = alg.make_train_fns(spec, tc)
        qs = fns.init(jax.random.PRNGKey(0))
        qs2, met = jax.jit(lambda q: fns.iterate(q, 0, jax.random.PRNGKey(1)))(qs)
        assert np.isfinite(float(met["loss"])) and int(qs2.n_updates) == 1, exp
        r = E.eval_learned(spec, tc, qs2.params, n_exp=3, T=8, chunk=3)
        assert r["pareto_feas_strict"].shape == (3,) and np.isfinite(r["util_bound_t"]).all(), exp
    # observation capacity block: static for E2*, CSI for *csi
    assert D.EnvSpec(**cfgs["E2"]["env"]).capacity == "const" and D.EnvSpec(**cfgs["E2csi"]["env"]).capacity == "rate"


def test_group_size_changes_only_the_last_bits_early_on():
    """(measured): the same seed in a group of 2 and in a group of 3 differs at the ~1e-7 level after a few iterations (batched-matmul lowering depends on the
    batch shape; E0 at paper size: S = 2 vs 3 differ, S = 3 vs 4 do not; E7: S = 2, 3, 4 identical, S = 1 differs) and the chaotic dynamics can amplify it over 10k iterations.  This test pins the
    EARLY behaviour (rounding-level, not a bug): 20 iterations, tiny net, seeds [1, 2] vs [1, 2, 3] agree to 1e-5.  The bit-for-bit statement that holds is 'same group + same code' (test above)."""
    with tempfile.TemporaryDirectory() as d:
        train.main(COMMON + ["--seeds", "1", "2", "--out", os.path.join(d, "a")])
        train.main(COMMON + ["--seeds", "1", "2", "3", "--out", os.path.join(d, "b")])
        tag = os.listdir(os.path.join(d, "a"))[0]
        for s in (1, 2):
            a = np.load(os.path.join(d, "a", tag, f"seed_{s}", "params.npz"))
            b = np.load(os.path.join(d, "b", tag, f"seed_{s}", "params.npz"))
            for k in a.files:
                np.testing.assert_allclose(a[k], b[k], atol=1e-5, rtol=1e-4, err_msg=f"seed {s} {k}")

"""Differential tests vs the AUTHORS' code (branch 2024 @ 407dca7) on identical inputs/weights.
Fixtures: scripts/official/make_fixtures.py, run in the official venv -> tests/fixtures/oracle_fixtures.npz.
Tolerance 1e-5 (absolute on O(1) quantities, relative for larger). The official optimizer stores traces of the
gradient of -output; ours of +output, so traces compare with a sign flip.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")

from src import agents as ag  # noqa: E402
from src import algorithm as alg  # noqa: E402
from src import model as mdl  # noqa: E402
from src import train as tr  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "oracle_fixtures.npz")
F = np.load(FIX) if os.path.exists(FIX) else None
pytestmark = pytest.mark.skipif(F is None, reason="oracle fixtures not generated")
TOL = 1e-5


def close(a, b, tol=TOL, leaf_scale=False):
    """max |a-b| / max(1, |b|) element-wise, or / max(1, max|b|) per tensor (leaf_scale; float32 rounding of a sum is
    relative to the magnitude of its largest terms -- used for traces, whose entries reach ~1e2 when std is small)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    den = max(1.0, float(np.max(np.abs(b)))) if leaf_scale else np.maximum(1.0, np.abs(b))
    err = np.max(np.abs(a - b) / den)
    assert err <= tol, f"max rel err {err:.3e} > {tol}"
    return err


# ---------------------------------------------------------------- torch -> flax layout maps
def lin(prefix, name):
    return {"kernel": jnp.asarray(F[f"{prefix}/{name}.weight"].T), "bias": jnp.asarray(F[f"{prefix}/{name}.bias"])}


def q_params(prefix):
    W1 = F[f"{prefix}/4.weight"].T.reshape(16, 8, 8, 128).transpose(1, 2, 0, 3).reshape(1024, 128)  # (C,H,W)->(H,W,C)
    return {"conv": {"kernel": jnp.asarray(F[f"{prefix}/0.weight"].transpose(2, 3, 1, 0)),
                     "bias": jnp.asarray(F[f"{prefix}/0.bias"])},
            "fc1": {"kernel": jnp.asarray(W1), "bias": jnp.asarray(F[f"{prefix}/4.bias"])},
            "out": {"kernel": jnp.asarray(F[f"{prefix}/7.weight"].T), "bias": jnp.asarray(F[f"{prefix}/7.bias"])}}


def q_trace(prefix):
    """official trace index order = parameters() order: conv.w, conv.b, fc1.w, fc1.b, out.w, out.b; sign flipped."""
    z = [-F[f"{prefix}/{i}"] for i in range(6)]
    W1 = z[2].T.reshape(16, 8, 8, 128).transpose(1, 2, 0, 3).reshape(1024, 128)
    return {"conv": {"kernel": z[0].transpose(2, 3, 1, 0), "bias": z[1]}, "fc1": {"kernel": W1, "bias": z[3]},
            "out": {"kernel": z[4].T, "bias": z[5]}}


def actor_params(prefix):
    return {"trunk": {"fc1": lin(prefix, "fc_layer"), "fc2": lin(prefix, "hidden_layer")},
            "mu": lin(prefix, "linear_mu"), "std": lin(prefix, "linear_std")}


def critic_params(prefix):
    return {"trunk": {"fc1": lin(prefix, "fc_layer"), "fc2": lin(prefix, "hidden_layer")},
            "out": lin(prefix, "linear_layer")}


def mlp_trace(prefix, names):
    z = [-F[f"{prefix}/{i}"] for i in range(2 * len(names))]
    out = {}
    for j, path in enumerate(names):
        d = out
        for p in path[:-1]:
            d = d.setdefault(p, {})
        d[path[-1]] = {"kernel": z[2 * j].T, "bias": z[2 * j + 1]}
    return out


ACTOR_ORDER = [("trunk", "fc1"), ("trunk", "fc2"), ("mu",), ("std",)]
CRITIC_ORDER = [("trunk", "fc1"), ("trunk", "fc2"), ("out",)]


def tree_close(a, b, tol=TOL, leaf_scale=False):
    errs = jax.tree.map(lambda x, y: close(x, y, tol, leaf_scale), a, b)
    return max(jax.tree.leaves(errs))


# ---------------------------------------------------------------- A. ObGD
def test_obgd_matches_official():
    names = ["0.weight", "0.bias", "1.weight", "1.bias"]
    w = {n: jnp.asarray(F[f"obgd/w0/{n}"]) for n in names}
    z = {n: jnp.zeros_like(w[n]) for n in names}
    for k in range(3):
        g = {n: -jnp.asarray(F[f"obgd/g{k}/{n}"]) for n in names}   # official grads are of -output
        w, z, info = alg.obgd_step(w, z, g, jnp.float32(F[f"obgd/delta{k}"]), 1.0, 2.0, 0.99 * 0.8)
        if bool(F[f"obgd/reset{k}"]):
            z_cmp = {n: jnp.zeros_like(z[n]) for n in names}
        else:
            z_cmp = z
        for i, n in enumerate(names):
            close(w[n], F[f"obgd/w{k + 1}/{n}"])
            close(z_cmp[n], -F[f"obgd/z{k + 1}/{i}"])
        if bool(F[f"obgd/reset{k}"]):
            z = z_cmp


# ---------------------------------------------------------------- B. SparseInit statistics
@pytest.mark.parametrize("name,shape", [("fc", (1024, 128)), ("conv", (3, 3, 4, 16)), ("small", (12, 128))])
def test_sparse_init_matches_official(name, shape):
    from scipy import stats
    w = np.asarray(mdl.sparse_kernel(jax.random.PRNGKey(3), shape)).reshape(-1, shape[-1]).T   # (fan_out, fan_in)
    zeros_ours = (w == 0).sum(1)
    zeros_off = F[f"sparse/{name}/zeros_per_unit"]
    assert np.array_equal(np.unique(zeros_ours), np.unique(zeros_off)), (np.unique(zeros_ours), np.unique(zeros_off))
    nz_ours, nz_off = w[w != 0], F[f"sparse/{name}/nonzeros"]
    bound = 1.0 / np.sqrt(w.shape[1])
    assert np.abs(nz_ours).max() <= bound + 1e-7 and np.abs(nz_off).max() <= bound + 1e-7
    assert stats.ks_2samp(nz_ours, nz_off).pvalue > 1e-3


# ---------------------------------------------------------------- C. forward passes
def test_q_forward_matches_official():
    net = mdl.QNetMinAtar(3)
    p = q_params("q/w0")
    for x, qv in zip(F["q/x"], F["q/qvals"]):
        close(net.apply({"params": p}, jnp.asarray(x), 1.0), qv)


def test_ac_forward_matches_official():
    actor, critic = mdl.GaussianActor(3), mdl.ValueNet()
    pp, pv = actor_params("ac/pi0"), critic_params("ac/v0")
    for s, mu, std, v in zip(F["ac/s"], F["ac/mu"], F["ac/std"], F["ac/v"]):
        m, sd = actor.apply({"params": pp}, jnp.asarray(s), 1.0)
        close(m, mu)
        close(sd, std)
        close(critic.apply({"params": pv}, jnp.asarray(s), 1.0), v[0])


# ---------------------------------------------------------------- D. wrappers (obs norm, ScaleReward, time feature)
def test_wrappers_match_official():
    obs, rew, term = F["wr/obs_in"], F["wr/rew_in"], F["wr/term_in"]
    w_o, w_r, u = alg.welford_init((3,)), alg.welford_init(), jnp.float32(0.0)
    T = 5
    j = 0
    w_o, o = alg.normalize_obs(w_o, jnp.asarray(obs[j], jnp.float32))
    outs = [np.concatenate([np.asarray(o), [-0.5]])]
    t_feat = -0.5
    rs = []
    for k in range(len(rew)):
        j += 1
        w_r, u, r = alg.scale_reward(w_r, u, jnp.float32(rew[k]), jnp.float32(term[k]), 0.99)
        w_o, o = alg.normalize_obs(w_o, jnp.asarray(obs[j], jnp.float32))
        outs.append(np.concatenate([np.asarray(o), [t_feat]]))
        t_feat += 1.0 / T
        rs.append(float(r))
        if term[k]:
            j += 1
            w_o, o = alg.normalize_obs(w_o, jnp.asarray(obs[j], jnp.float32))
            t_feat = -0.5
            outs.append(np.concatenate([np.asarray(o), [t_feat]]))
    close(np.array(rs), F["wr/rew_out"], 1e-5)
    close(np.array(outs), F["wr/obs_out"], 1e-4)   # first rows normalize by var~1e-4 (float32 vs float64)


# ---------------------------------------------------------------- E. full stream-Q updates (3 steps, cut at step 1)
def test_stream_q_update_matches_official():
    from src.data import Env
    env = Env(None, None, (10, 10, 4), 3, True, "fixture")
    agent = ag.QAgent(env, total_steps=1000)
    lane = tr.stack_lanes("q", [tr.lane_cfg("q", "stream_q")])
    lane = jax.tree.map(lambda x: x[0], lane)
    p = q_params("q/w0")
    z = jax.tree.map(jnp.zeros_like, p)
    adam = alg.adam_init(p)
    xs = F["q/x"]
    for k, (a, r, done, ng) in enumerate(F["q/steps"]):
        p, z, adam, _, delta, info, _, _ = agent.learn(p, z, adam, p, jnp.asarray(xs[k]), int(a), jnp.float32(r),
                                                       jnp.asarray(xs[k + 1]), jnp.asarray(bool(done)),
                                                       jnp.asarray(bool(ng)), lane, jnp.int32(k))
        close(delta, F[f"q/delta{k}"])
        tree_close(p, q_params(f"q/w{k + 1}"))
        tree_close(z, q_trace(f"q/z{k + 1}"))


# ---------------------------------------------------------------- F. full stream-AC updates (3 steps, last done)
def test_stream_ac_update_matches_official():
    from src.data import Env
    env = Env(None, None, (11,), 3, False, "fixture")
    agent = ag.ACAgent(env)
    lane = jax.tree.map(lambda x: x[0], tr.stack_lanes("ac", [tr.lane_cfg("ac", "stream_ac")]))
    pp, pv = actor_params("ac/pi0"), critic_params("ac/v0")
    zp, zv = jax.tree.map(jnp.zeros_like, pp), jax.tree.map(jnp.zeros_like, pv)
    ap, av = alg.adam_init(pp), alg.adam_init(pv)
    S, A = F["ac/s"], F["ac/a"]
    for k, (r, done) in enumerate(F["ac/steps"]):
        pp, pv, zp, zv, ap, av, delta, _, _, _ = agent.learn(pp, pv, zp, zv, ap, av, jnp.asarray(S[k]),
                                                             jnp.asarray(A[k]), jnp.float32(r), jnp.asarray(S[k + 1]),
                                                             jnp.asarray(bool(done)), lane)
        close(delta, F[f"ac/delta{k}"])
        tree_close(pp, actor_params(f"ac/pi{k + 1}"))
        tree_close(pv, critic_params(f"ac/v{k + 1}"))
        tree_close(zp, mlp_trace(f"ac/zpi{k + 1}", ACTOR_ORDER), leaf_scale=True)
        tree_close(zv, mlp_trace(f"ac/zv{k + 1}", CRITIC_ORDER), leaf_scale=True)


# ---------------------------------------------------------------- G. LeakyReLU kink: all-zero first observation
def td_params(prefix):
    return {"trunk": {"fc1": lin(prefix, "fc1_v"), "fc2": lin(prefix, "hidden_v")}, "out": lin(prefix, "fc_v")}


def test_stream_td_zero_first_obs_matches_official():
    """Regression for the bug found in small-scale: d leaky_relu / dx at x == 0 must be the slope (torch), not 1."""
    agent = ag.TDAgent(7)
    lane = jax.tree.map(lambda x: x[0], tr.stack_lanes("td", [dict(tr.lane_cfg("td", "stream_td"), norm=0)]))
    p = td_params("td/w0")
    z = jax.tree.map(jnp.zeros_like, p)
    S = F["td/s"]
    c = ag.TDCarry(p, z, alg.adam_init(p), alg.welford_init((7,)), alg.welford_init(), jnp.float32(0.0),
                   jnp.asarray(S[0]), jnp.float32(0.0), jnp.float32(0.0))
    for k, r in enumerate(F["td/r"]):
        c, _ = agent.step(c, jnp.asarray(S[k + 1]), jnp.float32(r), jnp.asarray(False), lane)
        tree_close(c.params, td_params(f"td/w{k + 1}"))


def test_stream_q_zero_obs_matches_official():
    from src.data import Env
    agent = ag.QAgent(Env(None, None, (10, 10, 4), 3, True, "fixture"), total_steps=1000)
    lane = jax.tree.map(lambda x: x[0], tr.stack_lanes("q", [tr.lane_cfg("q", "stream_q")]))
    p = q_params("qz/w0")
    out = agent.learn(p, jax.tree.map(jnp.zeros_like, p), alg.adam_init(p), p, jnp.zeros((10, 10, 4)), 1,
                      jnp.float32(0.5), jnp.asarray(F["q/x"][0]), jnp.asarray(False), jnp.asarray(False), lane,
                      jnp.int32(0))
    tree_close(out[0], q_params("qz/w1"))


# ======================================================================================================================
# H. CLOSED-LOOP differential vs the authors' main loops on the REAL envs (scripts/official/make_closed_loop_fixtures.py)
# Their main (same seeding, wrapper stack, agent) ran 600 steps on MinAtar 1.0.15 Breakout-v1 (stream Q) and on
# gymnasium MuJoCo Hopper-v4 (stream AC); a pass-through recorder logged the RAW env outputs. Here OUR wrappers +
# update (QAgent.observe / ACAgent.observe, the exact code train.py scans) consume the same raw outputs and the same
# actions, and every agent input, epsilon, greedy action, TD error, episode return and parameter snapshot is diffed.
# Tolerance: TOL_CL below; the official code's OWN float32-rounding sensitivity over the same horizon (its replay from
# weights perturbed by 1e-7 relative) is recorded in the fixture and asserted to be the same order as our deviation.
# ======================================================================================================================
FIX_CL = os.path.join(os.path.dirname(__file__), "fixtures", "closed_loop_fixtures.npz")
G = np.load(FIX_CL) if os.path.exists(FIX_CL) else None
needs_cl = pytest.mark.skipif(G is None, reason="closed-loop fixtures not generated")
TOL_CL = 1e-5     # absolute on O(1) TD errors / normalized inputs, relative (max(1,|b|)) on params
SNAPS = (1, 10, 100, 600)


def _with(src, fn, *a):
    global F
    old, F = F, src
    try:
        return fn(*a)
    finally:
        F = old


def _q_loop():
    from src.data import Env
    T = int(G["q/T"])
    agent = ag.QAgent(Env(None, None, (10, 10, 4), 3, True, "fixture"), total_steps=T)
    lane = jax.tree.map(lambda x: x[0], tr.stack_lanes("q", [tr.lane_cfg("q", "stream_q", {"stream_q": {
        "eps_frac": float(G["q/eps_frac"])}})]))
    p0 = _with(G, q_params, "q/w0")
    raw0 = jnp.asarray(G["q/reset0_raw"], jnp.float32)
    obs_w, on = agent._obs_in(lane, alg.welford_init((10, 10, 4)), raw0)
    c = ag.QCarry(p0, alg.tree_zeros(p0), alg.adam_init(p0), p0, obs_w, alg.welford_init(), jnp.float32(0.0),
                  jnp.int32(0), on, jnp.int32(0), jnp.float32(0.0), jnp.int32(0), ag.qdiag_zero())

    @jax.jit
    def go(c, t, a, ng, o2, r, d, rr):
        q_s = agent.q(c.params, c.obs, lane.use_ln)
        new, out, delta = agent.observe(c, t, lane, a, ng, a, (jnp.int32(0), o2, r, d), (jnp.int32(0), rr))
        return new, out, delta, agent.eps(t, lane), jnp.argmax(q_s), c.obs

    rec = dict(delta=[], eps=[], greedy=[], obs=[], ep_ret=[], snaps={})
    for k in range(len(G["q/a"])):
        d = bool(G["q/term"][k] or G["q/trunc"][k])
        c, out, delta, eps, greedy, obs_in = go(c, jnp.int32(k), jnp.int32(G["q/a"][k]), jnp.asarray(bool(G["q/ng"][k])),
                                                jnp.asarray(G["q/obs2_raw"][k], jnp.float32),
                                                jnp.float32(G["q/r_raw"][k]), jnp.asarray(d),
                                                jnp.asarray(G["q/reset_raw"][k], jnp.float32))
        rec["delta"].append(float(delta)); rec["eps"].append(float(eps)); rec["greedy"].append(int(greedy))
        rec["obs"].append(np.asarray(obs_in))
        if d:
            rec["ep_ret"].append(float(out[0]))
        if k + 1 in SNAPS:
            rec["snaps"][k + 1] = c.params
    return rec, c


@pytest.fixture(scope="module")
def q_loop():
    return _q_loop()


@needs_cl
def test_closed_loop_q_inputs_eps_and_greedy(q_loop):
    rec, _ = q_loop
    close(np.array(rec["obs"]), G["q/s_in"], TOL_CL)                    # obs norm on the real Breakout stream
    close(np.array(rec["eps"]), G["q/eps"], 1e-6)                        # schedule at official time_step = t + 1
    a, ng, gr = G["q/a"], G["q/ng"].astype(bool), np.array(rec["greedy"])
    assert np.all(gr[~ng] == a[~ng]), "a greedy/tie step disagrees with our argmax"   # ng False => a == argmax q
    assert np.all(gr[ng] != a[ng]), "a non-greedy step equals our argmax"


@needs_cl
def test_closed_loop_q_td_errors_and_params(q_loop):
    rec, c = q_loop
    err = close(np.array(rec["delta"]), G["q/delta"], TOL_CL)
    pert = np.max(np.abs(G["q/pert_delta"] - G["q/delta"]))
    assert err <= max(1e-6, 30 * pert), (err, pert)       # same order as the official code's own rounding sensitivity
    for t, p in rec["snaps"].items():
        tree_close(p, _with(G, q_params, f"q/w{t}"), TOL_CL)
    tree_close(c.trace, _with(G, q_trace, "q/z600"), TOL_CL, leaf_scale=True)


@needs_cl
def test_closed_loop_q_episode_returns(q_loop):
    rec, _ = q_loop
    assert np.array_equal(np.array(rec["ep_ret"]), G["q/ep_ret"])       # the paper's metric input, exact


def _ac_loop():
    from src.data import Env
    agent = ag.ACAgent(Env(None, None, (11,), 3, False, "fixture"))
    lane = jax.tree.map(lambda x: x[0], tr.stack_lanes("ac", [tr.lane_cfg("ac", "stream_ac")]))
    pp, pv = _with(G, actor_params, "ac/pi0"), _with(G, critic_params, "ac/v0")
    t0 = jnp.float32(-0.5)
    obs_w, on = agent._obs_in(lane, alg.welford_init((11,)), jnp.asarray(G["ac/reset0_raw"], jnp.float32), t0)
    c = ag.ACCarry(pp, pv, alg.tree_zeros(pp), alg.tree_zeros(pv), alg.adam_init(pp), alg.adam_init(pv), obs_w,
                   alg.welford_init(), jnp.float32(0.0), jnp.int32(0), on, t0, jnp.float32(0.0), jnp.int32(0),
                   ag.acdiag_zero())

    @jax.jit
    def go(c, a, o2, r, term, rr):
        new, out, delta = agent.observe(c, lane, a, (jnp.int32(0), o2, r, term), (jnp.int32(0), rr))
        return new, out, delta, c.obs

    rec = dict(delta=[], obs=[], ep_ret=[], snaps={})
    for k in range(len(G["ac/a"])):
        assert not G["ac/trunc"][k]   # 600 steps < the 1000-step limit; our truncation is internal (ep_len)
        c, out, delta, obs_in = go(c, jnp.asarray(G["ac/a"][k]), jnp.asarray(G["ac/obs2_raw"][k], jnp.float32),
                                   jnp.float32(G["ac/r_raw"][k]), jnp.asarray(bool(G["ac/term"][k])),
                                   jnp.asarray(G["ac/reset_raw"][k], jnp.float32))
        rec["delta"].append(float(delta)); rec["obs"].append(np.asarray(obs_in))
        if G["ac/term"][k]:
            rec["ep_ret"].append(float(out[0]))
        if k + 1 in SNAPS:
            rec["snaps"][k + 1] = (c.pi, c.v)
    return rec, c


@pytest.fixture(scope="module")
def ac_loop():
    return _ac_loop()


@needs_cl
def test_closed_loop_ac_inputs(ac_loop):
    rec, _ = ac_loop
    # vs the official wrappers fed the same raw obs rounded to float32 (our env dtype): float32-rounding level
    close(np.array(rec["obs"]), G["ac/f32in_s_in"], 1e-6)
    # vs the official run on float64 MuJoCo obs: dominated by the float32 rounding of the raw obs itself, which moves
    # the OFFICIAL code by the same amount (f32in vs official, recorded in the fixture)
    off = np.max(np.abs(G["ac/f32in_s_in"] - G["ac/s_in"]) / np.maximum(1, np.abs(G["ac/s_in"])))
    assert close(np.array(rec["obs"]), G["ac/s_in"], 1e-4) <= 1.1 * off + 1e-6


@needs_cl
def test_closed_loop_ac_td_errors_and_params(ac_loop):
    rec, c = ac_loop
    err = close(np.array(rec["delta"]), G["ac/delta"], TOL_CL)
    pert = np.max(np.abs(G["ac/pert_delta"] - G["ac/delta"]))
    assert err <= max(1e-6, 30 * pert), (err, pert)
    for t, (pp, pv) in rec["snaps"].items():
        tree_close(pp, _with(G, actor_params, f"ac/pi{t}"), TOL_CL)
        tree_close(pv, _with(G, critic_params, f"ac/v{t}"), TOL_CL)
    tree_close(c.z_pi, _with(G, mlp_trace, "ac/zpi600", ACTOR_ORDER), TOL_CL, leaf_scale=True)
    tree_close(c.z_v, _with(G, mlp_trace, "ac/zv600", CRITIC_ORDER), TOL_CL, leaf_scale=True)
    close(np.array(rec["delta"]), G["ac/f32in_delta"], 2e-6)   # same raw-obs dtype as ours


@needs_cl
def test_closed_loop_ac_episode_returns(ac_loop):
    rec, _ = ac_loop
    close(np.array(rec["ep_ret"]), G["ac/ep_ret"], 1e-5)   # float32 sum of float64 MuJoCo rewards


# ---------------------------------------------------------------- I. the paper's metric: official plot.py vs eval.py
@needs_cl
def test_metric_matches_official_plot():
    from src import eval as ev
    n = G["metric/n"]
    xs, ys = np.split(G["metric/x"], np.cumsum(n)[:-1]), np.split(G["metric/y"], np.cumsum(n)[:-1])
    runs = [np.stack([x, y, np.ones_like(x)], 1) for x, y in zip(xs, ys)]
    steps, m, se, _ = ev.curve(runs, 400_000, 50_000)
    assert np.array_equal(steps, G["metric/steps"])
    np.testing.assert_allclose(m, G["metric/avg"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(se, G["metric/se"], rtol=1e-12, atol=1e-12)

"""Parity with the official MadryLab/implementation-matters@5ee6ecb code, from COMMITTED fixtures (no oracle venv
needed). Fixtures in tests/fixtures/official/ were recorded by:

* primitives_{in,out}.npz -- scripts/oracle_diff.py: a real 2048-step MJX Hopper batch + perturbed params (in), and
  the OFFICIAL PyTorch outputs on that batch (out; oracle_harness.py running unmodified steps.py / torch_utils.py /
  models.py / agent.advantage_and_return). Our side is recomputed here from the inputs alone.
* env_{hopper,walker2d}.npz -- record_env.py on gymnasium MuJoCo v4 (the closest runnable stand-in for the
  official gym *-v2 env): our env's obs/reward/done DEFINITIONS evaluated on gym's states must equal gym's.
* gradstats.npz -- official_grad_ref.py: 10 official-primitive gradient estimates + a 1e5-pair reference, and the
  pairwise / to-true cosines computed with sklearn's cosine_similarity (the function the official logging.py imports).

Tolerances: 1e-5 abs-or-rel for elementwise quantities, 1e-4 for multi-step optimiser updates (fp32 accumulation
over 320 Adam steps), 1e-6 for cosine statistics. ~60 s on CPU (jit of the PPO/value/TRPO updates).
"""
import os

import numpy as np
import pytest

from src import common  # noqa: F401  (bootstrap)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
from src.configs.default import algo_config  # noqa: E402
from src.model import Nets, gauss_kl  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "official")


def close(ours, theirs, tol):
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    err = float(np.max(np.abs(ours - theirs)))
    rel = err / max(float(np.max(np.abs(theirs))), 1e-12)
    assert err <= tol or rel <= tol, f"max_abs={err:.3g} max_rel={rel:.3g} tol={tol}"


@pytest.fixture(scope="module")
def P():
    d = dict(np.load(os.path.join(FIX, "primitives_in.npz")))
    o = dict(np.load(os.path.join(FIX, "primitives_out.npz")))
    cfg = algo_config("trpo", "hopper")
    nets = Nets(d["obs"].shape[1], d["act"].shape[1], cfg)

    def flat(prefix, policy=True):
        tree = (nets.pi_unravel(nets.pi_flat0) if policy else nets.v_unravel(nets.v_flat0))
        tree = jax.tree_util.tree_map(lambda x: x, tree)
        sub = tree["params"]["mean" if policy else "v"]
        for i in range(3):
            sub[f"Dense_{i}"]["kernel"] = jnp.asarray(d[f"{prefix}W{i}"])
            sub[f"Dense_{i}"]["bias"] = jnp.asarray(d[f"{prefix}b{i}"])
        if policy:
            tree["params"]["log_std"] = jnp.asarray(d[f"{prefix}log_std"])
        return ravel_pytree(tree)[0]

    def layers(flat_vec, policy=True):
        t = (nets.pi_unravel if policy else nets.v_unravel)(flat_vec)["params"]
        sub = t["mean" if policy else "v"]
        out = []
        for i in range(3):
            out += [np.asarray(sub[f"Dense_{i}"]["kernel"]), np.asarray(sub[f"Dense_{i}"]["bias"])]
        if policy:
            out.append(np.asarray(t["log_std"]))
        return out

    def theirs(prefix, policy=True):
        out = []
        for i in range(3):
            out += [o[f"{prefix}W{i}"], o[f"{prefix}b{i}"]]
        if policy:
            out.append(o[f"{prefix}log_std"])
        return out

    return dict(d=d, o=o, cfg=cfg, nets=nets, pi=flat("pi_"), pi2=flat("pi2_"), v=flat("v_", False), layers=layers,
                theirs=theirs)


def test_gae_and_returns(P):
    d, o = P["d"], P["o"]
    adv, ret = A.gae(jnp.asarray(d["rew"]), jnp.asarray(d["values"]), jnp.asarray(d["not_done"]),
                     float(d["gamma"]), float(d["lam"]))
    close(adv, o["adv"], 1e-5)
    close(ret, o["ret"], 1e-5)


def test_value_metrics(P):
    d, o = P["d"], P["o"]
    args = [jnp.asarray(d[k]) for k in ("v_new", "values", "adv_ours", "ret_ours", "not_done")]
    for clip in (0, 1):
        v = A.value_errors(*args, bool(clip))
        close(v["gae_mre_code"], o[f"gae_mre_code_clip{clip}"], 1e-5)
    v = A.value_errors(*args, False)
    close(v["gae_mse"], o["gae_mse_clip0"], 1e-5)
    close(v["returns_mre"], o["returns_mre"], 1e-5)
    close(v["returns_mse"], o["returns_mse"], 1e-5)


def test_policy_loglik_kl_surrogate(P):
    d, o, nets, pi = P["d"], P["o"], P["nets"], P["pi"]
    obs, act = jnp.asarray(d["obs"]), jnp.asarray(d["act"])
    close(nets.logp(pi, obs, act), o["logp"], 1e-5)
    m1, l1 = nets.pi_apply(pi, obs)
    m2, l2 = nets.pi_apply(P["pi2"], obs)
    close(gauss_kl(m1, l1, m2, l2), o["kl"], 1e-5)
    adv = jnp.asarray(d["adv_ours"])
    close(A.adv_normalize(adv), o["adv_normalized"], 1e-5)
    lp_old = jnp.asarray(d["logp_old"])
    s, g = jax.value_and_grad(lambda p: jnp.mean(jnp.exp(nets.logp(p, obs, act) - lp_old) * A.adv_normalize(adv)))(pi)
    close(s, o["surrogate"], 1e-5)
    for ours, th in zip(P["layers"](g), P["theirs"]("gpi_")):
        close(ours, th, 1e-4)


def test_cg(P):
    d, o = P["d"], P["o"]
    H = jnp.asarray(d["H"])
    close(A.cg_solve(lambda x: H @ x, jnp.asarray(d["cg_b"]), 10), o["cg_x"], 1e-4)


def _batch(P):
    d = P["d"]
    return A.Batch(*(jnp.asarray(d[k]) for k in ("obs", "act", "logp_old", "adv_ours", "ret_ours", "values",
                                                   "not_done")))


def test_ppo_step(P):
    d, nets, n = P["d"], P["nets"], P["d"]["obs"].shape[0]
    k_ppo = jnp.asarray(d["keys"][0])
    assert np.array_equal(np.asarray(A._minibatch_perms(k_ppo, n, 10, 32)).reshape(10, n), d["ppo_perms"])
    cfg = {**algo_config("ppo", "hopper"), "value_lr": float(d["v_lr"])}
    pi_tx, _ = A.make_optimizers(cfg)
    new, _, _ = A.ppo_policy_update(nets, cfg, P["pi"], pi_tx.init(P["pi"]), pi_tx, _batch(P), float(d["ppo_lr"]),
                                    k_ppo)
    assert float(jnp.linalg.norm(new - P["pi"])) > 0.1          # a non-trivial step
    for ours, th in zip(P["layers"](new), P["theirs"]("ppo_pi_")):
        close(ours, th, 1e-4)


def test_value_step(P):
    d, nets, n = P["d"], P["nets"], P["d"]["obs"].shape[0]
    k_v = jnp.asarray(d["keys"][1])
    assert np.array_equal(np.asarray(A._minibatch_perms(k_v, n, 10, 32)).reshape(10, n), d["v_perms"])
    cfg = {**algo_config("ppo", "hopper"), "value_clipping": bool(d["v_clip"]), "value_lr": float(d["v_lr"])}
    _, v_tx = A.make_optimizers(cfg)
    new, _, _ = A.value_update(nets, cfg, P["v"], v_tx.init(P["v"]), v_tx, _batch(P), float(d["v_lr"]), k_v)
    for ours, th in zip(P["layers"](new, False), P["theirs"]("vstep_v_", False)):
        close(ours, th, 1e-4)


def test_trpo_step(P):
    d, nets, cfg, n = P["d"], P["nets"], P["cfg"], P["d"]["obs"].shape[0]
    assert cfg["kl_delta"] == pytest.approx(float(d["max_kl"])) and cfg["cg_damping"] == pytest.approx(float(d["damping"]))
    k_trpo = jnp.asarray(d["keys"][2])
    assert np.array_equal(np.asarray(jax.random.choice(k_trpo, n, (int(n * 0.1),), replace=False)), d["fisher_subset"])
    new, st = A.trpo_policy_update(nets, cfg, P["pi"], _batch(P), k_trpo)
    assert float(jnp.linalg.norm(new - P["pi"])) > 0.1
    for ours, th in zip(P["layers"](new), P["theirs"]("trpo_pi_")):
        close(ours, th, 1e-4)


def test_normalisers(P):
    d, o, cfg = P["d"], P["o"], P["cfg"]
    rs, rr, acc, zo, ro = D.runstat_init((d["stream_obs"].shape[1],)), D.runstat_init(()), 0.0, [], []
    for x, r, reset in zip(d["stream_obs"], d["stream_rew"], d["stream_reset"]):
        if reset:
            acc = 0.0
        rs, y = D.obs_filter(rs, jnp.asarray(x), 10.0, True)
        acc = acc * cfg["gamma"] + r
        rr = D.runstat_push(rr, jnp.float32(acc))
        ro.append(float(jnp.clip(r / (D.runstat_std(rr) + 1e-8), -10, 10)))
        zo.append(np.asarray(y))
    close(np.asarray(zo), o["stream_obs_norm"], 1e-4)
    close(np.asarray(ro), o["stream_rew_norm"], 1e-4)


@pytest.mark.parametrize("post_step_kinematics", [False, True])
@pytest.mark.parametrize("env_name", ["hopper", "walker2d"])
def test_env_definitions_match_gym(env_name, post_step_kinematics):
    """Our env's obs / reward / done functions on gym v4's states equal gym v4's (physics replaced by the recorded
    next state). The physics itself is a labelled PROXY  and is NOT asserted equal. Both env modes
    (legacy and the post-step kinematics) share these definitions."""
    d = dict(np.load(os.path.join(FIX, f"env_{env_name}.npz")))
    env, _ = D.make_env(env_name, "mjx", post_step_kinematics)
    tmpl = env.reset(jax.random.PRNGKey(0))
    shift = np.zeros(d["qpos"].shape[1], np.float32); shift[1] = 1.25          # gym rootz ref=1.25

    def one(q, qd, a, q1, qd1):
        ps = env.pipeline_init(q, qd)
        st = tmpl.replace(pipeline_state=ps, obs=env._get_obs(ps))
        env_step = env.pipeline_step
        env.pipeline_step = lambda *_: env.pipeline_init(q1, qd1)
        try:
            st1 = env.step(st, a)
        finally:
            env.pipeline_step = env_step
        return st.obs, st1.obs, st1.reward, st1.done

    f = jax.jit(jax.vmap(one))
    o0, o1, r, dn = f(*(jnp.asarray(x, jnp.float32) for x in
                        (d["qpos"] - shift, d["qvel"], d["act"], d["qpos1"] - shift, d["qvel1"])))
    close(o0, d["obs"], 1e-5)
    close(o1, d["obs1"], 1e-5)
    close(r, d["rew"], 1e-4)
    assert np.array_equal(np.asarray(dn) > 0.5, d["term"].astype(bool))
    assert d["term"].sum() > 5                                     # terminations actually exercised


@pytest.mark.skipif(not os.path.exists(os.path.join(FIX, "gradstats.npz")), reason="gradstats fixture absent")
def test_gradient_statistics_match_sklearn():
    """Our cos_stats (pairwise mean over i<j, mean cosine to the reference) on the official-primitive gradient
    estimates equals the statistic computed with sklearn's cosine_similarity in the oracle venv."""
    g = dict(np.load(os.path.join(FIX, "gradstats.npz")))
    for n in g["sizes"]:
        s = A.cos_stats(g[f"G_{n}"], g["g_ref"], n_boot=10)
        assert s["pairwise"] == pytest.approx(float(g[f"pairwise_{n}"]), abs=1e-6)
        assert s["to_true"] == pytest.approx(float(g[f"to_true_{n}"]), abs=1e-6)


@pytest.mark.parametrize("env_name", ["hopper", "walker2d"])
def test_env_post_step_kinematics(env_name):
    """Recheck with post_step_kinematics the forward velocity is (q1[0] - q0[0]) / dt and obs[0] is the
    post-step height q1[1] + 1.25, as in gym's step() (qpos read after do_simulation); the legacy env lags by one
    2 ms substep, which this test also confirms is visible (so the check has power)."""
    errs = {}
    for fix in (False, True):
        env, spec = D.make_env(env_name, "mjx", fix)

        def run(key, T=200):
            def step(st, k):
                a = jax.random.uniform(k, (spec.act_dim,), minval=-1.0, maxval=1.0) * 0.5
                st1 = env.step(st, a)
                q0, q1 = st.pipeline_state.q, st1.pipeline_state.q
                out = (st1.metrics["x_velocity"] - (q1[0] - q0[0]) / env.dt, st1.obs[0] - (q1[1] + 1.25))
                st1 = jax.lax.cond(st1.done > 0.5, lambda: env.reset(jax.random.fold_in(k, 1)), lambda: st1)
                return st1, out
            return jax.lax.scan(step, env.reset(key), jax.random.split(key, T))[1]

        dv, dz = jax.jit(jax.vmap(run))(jax.random.split(jax.random.PRNGKey(0), 2))
        errs[fix] = (float(jnp.max(jnp.abs(dv))), float(jnp.max(jnp.abs(dz))))
    assert errs[True][0] < 1e-5 and errs[True][1] < 1e-6, errs
    assert errs[False][0] > 1e-3, errs          # the legacy lag is real

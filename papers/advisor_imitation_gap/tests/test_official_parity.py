"""Differential tests against the authors' code. Fixtures come from scripts/official/make_fixtures.py
run in the authors'-code venv (allenai/advisor @ 6849755 + AllenAct v0.5.0, torch 1.13.1 CPU; regenerated
bit-identically under AllenAct f00445e4). The fixtures are committed, so these tests need no external code."""
import hashlib
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src import algorithm as A
from src.envs import lighthouse as LH
from src.envs import pd as PD

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name):
    p = os.path.join(FIX, name)
    if not os.path.exists(p):
        pytest.skip(f"fixture {name} missing (sh scripts/official/setup.sh, then scripts/official/make_fixtures.py)")
    return np.load(p, allow_pickle=False)


def test_pd_trace_matches_official():
    f = _load("official_pd_trace.npz")
    assert tuple(f["combination"].tolist()) == PD.CODE
    rows = f["rows"]
    step = jax.jit(PD.step)
    s = None
    mism = 0
    for k, (good, o, a, r, d, e) in enumerate(rows):
        if s is None:
            s = PD.reset(jax.random.PRNGKey(0), good=int(good))
        assert int(PD.obs(s)) == int(o), k
        assert int(PD.teacher_privileged(s)) == int(e), k
        s, rr, dd = step(s, int(a))
        mism += (abs(float(rr) - r) > 1e-9) + (bool(dd) != bool(d))
        if bool(dd):
            s = None
    assert mism == 0


def test_lh_design_matches_patsy():
    f = _load("official_lh_design.npz")
    M = LH.design_matrix()
    assert hashlib.sha256(np.packbits(M).tobytes()).hexdigest() == str(f["sha256"])
    assert np.array_equal(M[::97].astype(np.uint8), f["rows_sample"])


def test_lh_obs_and_teacher_match_official():
    f = _load("official_lh_trace.npz")
    goals, actions, tuples, experts = f["goals"], f["actions"], f["tuples"], f["experts"]
    radii = jnp.arange(1, 16)
    obs_all = jax.jit(jax.vmap(LH.corner_obs, in_axes=(None, 0)))
    teach_all = jax.jit(jax.vmap(LH.teacher, in_axes=(None, 0)))
    step = jax.jit(LH.step)
    bad_o = bad_e = n = 0
    for g, a_seq, tup, exp in zip(goals, actions, tuples, experts):
        s = LH.reset(jax.random.PRNGKey(0), goal=int(g))
        for t in range(len(a_seq)):
            bad_o += int(np.sum(np.asarray(obs_all(s, radii)) != tup[t]))
            bad_e += int(np.sum(np.asarray(teach_all(s, radii)) != exp[t]))
            n += 1
            s = s._replace(t=jnp.zeros((), jnp.int32))  # the fixture walks past MAX_STEPS-free; no termination
            s, _, _ = step(s, int(a_seq[t]))
    assert bad_o == 0 and bad_e == 0, (bad_o, bad_e, n)


@pytest.mark.parametrize("case", [0, 1, 2])
def test_losses_match_official_float64(case):
    f = _load("official_loss.npz")
    jax.config.update("jax_enable_x64", True)
    try:
        c = lambda k: jnp.asarray(f[f"c{case}_{k}"])
        batch = dict(action=c("actions"), logp=c("old_lp_a"), adv_norm=c("adv"), value=c("old_v"), ret=c("rets"),
                     label=c("labels"), label_valid=c("mask").astype(bool))
        clip = float(f[f"c{case}_clip"])
        alpha, beta = float(f[f"c{case}_alpha"]), float(f[f"c{case}_bound"])
        for name, mode in (("advisor", 3), ("ppo", 0), ("imitation", 1)):
            def L(lg, ax, v):
                terms = A.per_step_terms(lg, ax, v, batch, clip)
                return A.loss_from_terms(terms, jnp.int32(mode), alpha, beta)[0]
            val, grads = jax.value_and_grad(L, argnums=(0, 1, 2))(c("logits"), c("aux"), c("vals"))
            np.testing.assert_allclose(float(val), float(f[f"c{case}_{name}_loss"]), rtol=1e-10, atol=1e-12)
            for g, k in zip(grads, ("glogits", "gaux", "gvals")):
                np.testing.assert_allclose(np.asarray(g), f[f"c{case}_{name}_{k}"], rtol=1e-8, atol=1e-12)
    finally:
        jax.config.update("jax_enable_x64", False)


def test_mg_dynamics_obs_teacher_match_official():
    """Imported official layouts: egocentric view (lit and masked), dynamics, reward, done and the teacher's
    shortest-path length are identical along random traces; the teacher's action is one of the shortest-path moves."""
    from src.envs import crossing as MG
    f = _load("official_mg_trace.npz")
    step = jax.jit(MG.step)
    lit = jax.jit(MG.lit_obs)
    ob = jax.jit(MG.obs)
    bad = {"lit": 0, "obs": 0, "pos": 0, "rew": 0, "done": 0, "plen": 0, "teacher_not_shortest": 0}
    n = 0
    for k in range(f["grids"].shape[0]):
        s = MG.reset_from_grid(f["grids"][k].astype(np.int32))
        for t in range(f["actions"].shape[1]):
            if t > 0 and f["dones"][k, t - 1]:
                break
            n += 1
            bad["pos"] += int(not (np.array_equal(np.asarray(s.pos), f["pos"][k, t]) and int(s.dir) == f["dirs"][k, t]))
            bad["lit"] += int(not np.array_equal(np.asarray(lit(s)), f["lit"][k, t]))
            bad["plen"] += int(int(s.dist[s.pos[0], s.pos[1], s.dir]) != int(f["path_len"][k, t]))
            a_t = int(MG.teacher(s))
            s2, _, _ = step(s, a_t)
            d_after = int(s2.dist[s2.pos[0], s2.pos[1], s2.dir]) if not bool(s2.success) else 0
            bad["teacher_not_shortest"] += int(d_after != int(f["path_len"][k, t]) - 1)
            s, r, d = step(s, int(f["actions"][k, t]))
            bad["obs"] += int(not np.array_equal(np.asarray(ob(s)), f["obs"][k, t]))
            bad["rew"] += int(abs(float(r) - f["rewards"][k, t]) > 1e-6)
            bad["done"] += int(bool(d) != bool(f["dones"][k, t]))
    assert all(v == 0 for v in bad.values()), (bad, n)
    assert n > 500


def test_mg_generator_distribution_matches_official():
    """Layouts are not bit-identical (JAX vs numpy RNG); compare start shortest-path length and lava fraction."""
    from scipy import stats
    from src.envs import crossing as MG
    f = _load("official_mg_trace.npz")
    keys = jax.random.split(jax.random.PRNGKey(123), 1000)
    st = jax.jit(jax.vmap(MG.reset))(keys)
    plen = np.asarray(st.dist[:, 1, 1, 0])
    lava = np.asarray((st.grid == MG.LAVA).mean((1, 2)))
    assert stats.ks_2samp(plen, f["stats_len"]).pvalue > 0.01
    assert abs(plen.mean() / f["stats_len"].mean() - 1) < 0.05
    assert abs(lava.mean() / f["stats_lava"].mean() - 1) < 0.05


# ---------------------------------------------------------------- further official-code comparisons
def test_robust_reward_matches_allenact_expected_max():
    """Our RobustReward@K U-statistic vs allenact.utils.misc_utils.expected_max_of_subset_statistic (the authors'
    plotting statistic; it rounds values to 1e-8 and counts ties combinatorially)."""
    from src.eval import robust_reward_at_k
    f = _load("official_rr.npz")
    for v, k, o in zip(f["vals"], f["ks"], f["outs"]):
        v = v[~np.isnan(v)]
        np.testing.assert_allclose(robust_reward_at_k(v, int(k)), o, rtol=0, atol=1e-7)


def test_early_stopping_matches_official_stop_if_near_optimal():
    """Our 2D-LH early-stopping memory vs the authors' StopIfNearOptimal on 60 streams x 40 metric intervals."""
    from src.train import es_update
    f = _load("official_es.npz")
    jax.config.update("jax_enable_x64", True)
    try:
        n_stop = 0
        for s in range(f["counts"].shape[0]):
            st = None
            for t in range(f["counts"].shape[1]):
                st, stop, mean = es_update(st, int(f["counts"][s, t]), float(f["aves"][s, t]), float(f["opts"][s]))
                assert bool(stop) == bool(f["decisions"][s, t]), (s, t)
                np.testing.assert_allclose(mean, f["means"][s, t], rtol=1e-12)
                n_stop += int(stop)
        assert 0 < n_stop < f["decisions"].size   # both outcomes exercised
    finally:
        jax.config.update("jax_enable_x64", False)


def _torch_to_flax(name, sd, n_actions):
    """Map an official torch state_dict (float64 numpy) to our flax params."""
    if name == "lh":
        return {"params": {"kernel": sd["linear.weight"].T, "bias": sd["linear.bias"]}}
    pre = "actor_critic." if name == "mg" else ""
    head = {"kernel": sd[f"{pre}ac_nonrecurrent_head.linear.weight"].T, "bias": sd[f"{pre}ac_nonrecurrent_head.linear.bias"]}
    se = f"{pre}state_encoder."
    core = {"w_ih": sd[se + "rnn.weight_ih_l0"].T, "w_hh": sd[se + "rnn.weight_hh_l0"].T,
            "b_ih": sd[se + "rnn.bias_ih_l0"], "b_hh": sd[se + "rnn.bias_hh_l0"]}
    p = {"core": core, "init_state": sd[se + "init_hidden_state"].reshape(-1), "head": {"head": head}}
    if name == "pd":
        p["embed"] = {"embedding": sd["initial_embedding.weight"]}
    else:
        p["eo"] = {"embedding": sd["object_embedding.weight"]}
        p["ec"] = {"embedding": sd["color_embedding.weight"]}
        p["es"] = {"embedding": sd["state_embedding.weight"]}
    return {"params": p}


@pytest.mark.parametrize("name", ["pd", "mg", "lh"])
def test_forward_and_full_update_match_official(name):
    """Official model forward + one complete AllenAct update (compute_returns, advantage normalisation, 4 epochs x 2
    sampler-half minibatches, AdvisorWeightedStage(PPO) loss, clip_grad_norm_(0.5), Adam, LambdaLR factor, PPO clip
    decay), float64. Every intermediate and the updated parameters must agree with ours."""
    from src import model as M
    from src import train as TR
    from src.envs import lighthouse as LH
    f = _load("official_update.npz")
    jax.config.update("jax_enable_x64", True)
    try:
        g = lambda k: np.asarray(f[f"{name}_{k}"])
        T, B = int(f["T"]), int(f["B"])
        sd0 = {k.split("::", 1)[1]: f[k] for k in f.files if k.startswith(f"{name}_p0::")}
        sd1 = {k.split("::", 1)[1]: f[k] for k in f.files if k.startswith(f"{name}_p1::")}
        nA = {"pd": 7, "mg": 4, "lh": 4}[name]
        if name == "pd":
            net, obs = M.PDNet(), jnp.asarray(g("obs"))
        elif name == "mg":
            no, nc, ns = (int(x) for x in g("ncodes"))
            net, obs = M.MGNet(num_objects=no, num_colors=nc, num_states=ns), jnp.asarray(g("obs").astype(np.int32))
        else:
            net, obs = M.LHNet(), jnp.asarray(LH.design_active_columns()[g("rows")])
        p0 = jax.tree_util.tree_map(jnp.asarray, _torch_to_flax(name, sd0, nA))
        p1_ref = _torch_to_flax(name, sd1, nA)
        masks = g("masks")
        start = jnp.asarray(masks == 0)
        carry0 = (jnp.asarray(f[f"{name}_h0"][0]), jnp.asarray(f[f"{name}_h0"][1])) if name != "lh" else net.initial_carry(B)

        def unroll(p, o, s, c0):
            def body(c, x):
                lg, ax, v, c = net.apply(p, x[0], c, x[1])
                return c, (lg, ax, v)
            return jax.lax.scan(body, c0, (o, s))[1]
        lg, ax, v = unroll(p0, obs, start, carry0)
        # torch Categorical stores NORMALISED logits (log-probabilities)
        np.testing.assert_allclose(np.asarray(A.log_softmax(lg)), g("logits"), rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(np.asarray(A.log_softmax(ax)), g("aux"), rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(np.asarray(v), g("vals"), rtol=1e-9, atol=1e-11)

        vals = g("vals")
        done = jnp.asarray(masks[1:] == 0)
        adv, ret = A.gae(jnp.asarray(g("rewards")), jnp.asarray(vals[:T]), done, jnp.asarray(vals[T]), 0.99, 1.0)
        np.testing.assert_allclose(np.asarray(ret), g("rets"), rtol=1e-12, atol=1e-12)
        nadv = A.normalise(adv)
        np.testing.assert_allclose(np.asarray(nadv), g("nadv"), rtol=1e-10, atol=1e-12)

        cfg = TR.TrainCfg(env=name, n_envs=B, T=T, total_steps=300000)
        # the trainer's OWN schedule and learner: no re-typed formulas or minibatch slicing here)
        clip, lr = TR.schedules(cfg, float(f["lr0"]), 60000, 60000, cfg.total_steps)
        np.testing.assert_allclose(float(clip), float(g("clip")), rtol=1e-12)
        np.testing.assert_allclose(float(lr), float(f["lr0"]) * float(g("lr_factor")), rtol=1e-12)
        roll = dict(obs=obs[:T], start=start[:T], action=jnp.asarray(g("actions")), logp=jnp.asarray(g("old_lp")),
                    value=jnp.asarray(vals[:T]), ret=ret, adv_norm=nadv, label=jnp.asarray(g("labels")),
                    label_valid=jnp.asarray(g("lmask").astype(bool)))
        learn = TR.make_learner(cfg, net)
        opt = TR.make_optimizer(cfg)
        p, _, stats = learn(p0, opt.init(p0), roll, carry0, jnp.asarray(g("order"), jnp.int32), clip, lr,
                            jnp.int32(A.MODE_ID["advisor"]), float(f["alpha"]), 0.0)
        np.testing.assert_allclose(np.asarray(stats["loss"]), g("losses"), rtol=1e-9, atol=1e-11)
        flat_ours = jax.tree_util.tree_leaves_with_path(p)
        ref = dict((jax.tree_util.keystr(k), x) for k, x in jax.tree_util.tree_leaves_with_path(p1_ref))
        for k, x in flat_ours:
            np.testing.assert_allclose(np.asarray(x), ref[jax.tree_util.keystr(k)], rtol=1e-9, atol=1e-11,
                                       err_msg=jax.tree_util.keystr(k))
    finally:
        jax.config.update("jax_enable_x64", False)


def test_mg_teacher_action_equals_official_expert_everywhere():
    """Our teacher's ACTION (not only its path length) equals the official networkx
    expert's at every reachable state (x, y, dir) of 100 official layouts, including its tie-breaking."""
    from src.envs import crossing as MG
    f = _load("official_mg_expert_all.npz")
    grids, states, ex = f["grids"], f["states"], f["experts"]
    reset = jax.jit(MG.reset_from_grid)
    teach = jax.jit(jax.vmap(lambda s0, p, d: MG.teacher(s0._replace(pos=p, dir=d)), in_axes=(None, 0, 0)))
    bad = 0
    for k in range(grids.shape[0]):
        m = states[:, 0] == k
        st = states[m]
        s0 = reset(grids[k].astype(np.int32))
        a = teach(s0, jnp.asarray(st[:, 1:3].astype(np.int32)), jnp.asarray(st[:, 3].astype(np.int32)))
        bad += int(np.sum(np.asarray(a) != ex[m]))
    assert len(ex) > 15000
    assert bad == 0, bad

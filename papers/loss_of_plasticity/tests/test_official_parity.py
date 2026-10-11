"""Parity of our JAX implementation with the OFFICIAL code (github.com/shibhansh/loss-of-plasticity @ a6b7958).

The fixtures in tests/fixtures/official/ were recorded by running the official code unmodified (its own venv, torch
CPU) through scripts/oracle/record_fixtures.py; they hold only the inputs we fed it and its outputs.  Each test feeds
the identical inputs to our code.  Full-size versions of every check: scripts/oracle/diff.py -> runs/oracle_diff.

Tolerances (float32 on both sides; torch vs XLA reduction order):
  teacher-forced CBP step   : identical replaced units; weights / utilities / running means / Adam state <= 2e-5
  SCR target network        : Y and tau bit-identical
  effective rank / dead     : rel. diff < 1e-4 / identical counts
  SGD / L2 / linear traj.   : per-step squared error |diff| <= 1e-4 over 3000 steps (observed <= 7.6e-6)
  Adam traj. (torch, GnT)   : per-step |diff| <= 1e-3 over 3000 steps (Adam is chaotic past ~2e4 steps, see RESULTS)
  MNIST w100 SGD traj.      : median per-step loss |diff| <= 1e-5, mean loss |diff| <= 1e-4 over 1000 steps
  S&P step                  : deterministic part <= 1e-6; noise on every weight AND bias, std within 3 % of sigma,
                              two-sample KS test of our noise vs theirs p > 1e-3
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from scripts.oracle.diff import UTIL_MAP, state_from
from src import algorithm as A
from src import data as D
from src import diagnostics as G
from src.model import ACT_ID, REINIT_GAIN

FX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "official")


@pytest.fixture(autouse=True, scope="module")
def _float32_mode():
    """The oracle is float32 and our runs use JAX defaults; tests/test_grads.py switches x64 on at import time for the
    whole session, which changes jax.random integer draws and dtypes -- so pin x64 off here and restore it after."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


def load(name):
    z = np.load(os.path.join(FX, f"{name}.npz"))
    inp = {k[4:]: z[k] for k in z.files if k.startswith("in__")}
    out = {k[5:]: z[k] for k in z.files if k.startswith("out__")}
    return inp, out


def test_manifest_pins_oracle_commit():
    man = json.load(open(os.path.join(FX, "MANIFEST.json")))
    assert man["oracle_commit"] == "a6b79580d85f3025bdb601566d3627c5f489f13b"
    assert len(man["files"]) == 17


# ------------------------------------------------------------------------------------------- teacher-forced CBP
def _teacher_forced(cfg, o, n_hidden, adam_rule="standard", strict=True):
    """Feed the oracle's pre-step state of every step to our learner_step; compare with its post-step state.
    Returns (worst abs diff over weights / utilities / running means / Adam state, n replacements, steps compared)."""
    st = A.Static(n_in=int(cfg["n_in"]), widths=(int(cfg["width"]),) * n_hidden, n_out=int(cfg.get("n_out", 1)),
                  loss="mse" if str(cfg["kind"]) == "ffnn" else "ce", opt=str(cfg["opt"]), replace_mode="accumulate",
                  k_max=1)
    act = str(cfg["act"])
    hp = A.make_hp(lr=float(cfg["lr"]), rho=float(cfg["rho"]), mt=int(cfg["mt"]), eta=float(cfg["eta"]),
                   util=UTIL_MAP[str(cfg["util_type"])], act=ACT_ID[act], reinit_gain=REINIT_GAIN[act],
                   adam_rule=adam_rule)
    step = jax.jit(lambda s, x, y, k: A.learner_step(st, s, hp, x, y, k))
    rho = float(cfg["rho"])
    n_rep, ties, worst, compared = 0, 0, 0.0, 0
    for t in range(int(o["n_steps"])):
        s = state_from(o, f"t{t}_pre_", n_hidden, st.opt, rho)
        y = jnp.asarray(o[f"t{t}_y"]) if st.loss == "mse" else jnp.asarray(o[f"t{t}_y"]).astype(jnp.int32)
        s2, _ = step(s, jnp.asarray(o[f"t{t}_x"]), y, jax.random.PRNGKey(t))
        ref = state_from(o, f"t{t}_post_", n_hidden, st.opt, rho)
        tie = False
        for l in range(n_hidden):
            rep_o = set(o[f"t{t}_rep{l}"].tolist())
            rep_u = set(np.where(np.asarray(s2["cbp"]["age"][l]) == 0)[0].tolist())
            n_rep += len(rep_o)
            if rep_o != rep_u:
                if strict:
                    # only an exact utility tie (torch.topk vs lax.top_k break it differently) is tolerated
                    vals = {float(o[f"t{t}_post_bcu{l}"][i]) for i in rep_o ^ rep_u}
                    assert len(rep_o) == len(rep_u) and len(vals) == 1, f"step {t} layer {l}: {rep_o} vs {rep_u}"
                ties += 1
                tie = True
            k = s2["params"]["params"][f"layer_{l}"]["kernel"]
            for r in rep_o:   # the re-drawn incoming weights are random on both sides: copy theirs, then compare
                k = k.at[:, r].set(ref["params"]["params"][f"layer_{l}"]["kernel"][:, r])
            s2["params"]["params"][f"layer_{l}"]["kernel"] = k
            for key in ("util", "bcu", "f"):
                worst = max(worst, float(np.abs(np.asarray(s2["cbp"][key][l]) - np.asarray(ref["cbp"][key][l])).max()))
            if rep_o == rep_u and strict:
                assert np.array_equal(np.asarray(s2["cbp"]["age"][l]), np.asarray(ref["cbp"]["age"][l]))
        if tie:
            continue
        compared += 1
        for a, b in zip(jax.tree.leaves(s2["params"]), jax.tree.leaves(ref["params"])):
            worst = max(worst, float(np.abs(np.asarray(a) - np.asarray(b)).max()))
        if st.opt == "adam":
            for a, b in zip(jax.tree.leaves(s2["opt"]), jax.tree.leaves(ref["opt"])):
                worst = max(worst, float(np.abs(np.asarray(a) - np.asarray(b)).max()))
    return worst, n_rep, compared


@pytest.mark.parametrize("name,n_hidden", [(f"gnt_ffnn_{u}", 1) for u in
                                           ("adaptable_contribution", "contribution", "zero_contribution", "weight",
                                            "adaptation")] + [("gnt_deep_relu_sgd", 3)])
def test_cbp_step_teacher_forced(name, n_hidden):
    cfg, o = load(name)
    worst, n_rep, compared = _teacher_forced(cfg, o, n_hidden)
    assert n_rep >= 3, "fixture must exercise replacements"
    # exact ties occur only between dead ReLU units (utility exactly 0); most steps must be fully compared
    assert compared >= 20, compared
    assert worst <= 2e-5, worst


# --------------------------------------------- recheck: the paper-era AdamGnT (official dcc3619, pre-f00ca40)
FXP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "official_pre_f00ca40")


def load_prefix(name):
    z = np.load(os.path.join(FXP, f"{name}.npz"))
    return ({k[4:]: z[k] for k in z.files if k.startswith("in__")},
            {k[5:]: z[k] for k in z.files if k.startswith("out__")})


def test_prefix_manifest_pins_paper_era_adamgnt():
    man = json.load(open(os.path.join(FXP, "MANIFEST.json")))
    # sha256 of `git show dcc3619:lop/utils/AdamGnT.py` (identical in every commit up to 63c35f3)
    assert man["adamgnt_sha256"] == "29ac6816981d48f61b09b03a54d7ce7654dd93dbc69f024476b3519fa1427092"
    assert len(man["files"]) == 3


@pytest.mark.parametrize("name", [f"gnt_ffnn_{u}_prefix" for u in ("adaptable_contribution", "adaptation",
                                                                     "contribution")])
def test_cbp_step_teacher_forced_paper_era(name):
    cfg, o = load_prefix(name)
    worst, n_rep, compared = _teacher_forced(cfg, o, 1, adam_rule="paper_era")
    assert n_rep >= 5, "fixture must exercise replacements (and steps on reset Adam state)"
    assert compared >= 50, compared
    assert worst <= 2e-5, worst


def test_adam_rule_fixtures_discriminate():
    """Each rule fails the other rule's oracle: the fixtures can tell the two AdamGnT versions apart."""
    cfg, o = load_prefix("gnt_ffnn_adaptable_contribution_prefix")
    assert _teacher_forced(cfg, o, 1, adam_rule="standard", strict=False)[0] > 1e-3
    cfg, o = load("gnt_ffnn_adaptable_contribution")
    assert _teacher_forced(cfg, o, 1, adam_rule="paper_era", strict=False)[0] > 1e-3


# ------------------------------------------------------------------------------------------- SCR target network
def test_scr_target_network_exact():
    inp, o = load("ltu")
    spec = D.scr_spec("flip_one")
    for s in range(int(inp["n_seeds"])):
        root = jax.random.PRNGKey(s)
        X, Y, tgt = D.scr_stream(jax.random.fold_in(root, 0), jax.random.fold_in(root, 1), spec, 1)
        assert np.array_equal(np.asarray(tgt["W"]), inp[f"W_{s}"])
        assert np.array_equal(np.asarray(X[:2000]).astype(np.uint8), inp[f"X_{s}"])
        np.testing.assert_array_equal(np.asarray(tgt["tau"]), o[f"tau_{s}"])
        np.testing.assert_array_equal(np.asarray(Y[:2000]), o[f"Y_{s}"])


# ------------------------------------------------------------------------------------------- diagnostics
def test_effective_rank_and_dead_units():
    inp, o = load("diag")
    for i in range(int(inp["n"])):
        m = jnp.asarray(inp[f"m_{i}"])
        er = float(G.effective_rank(m))
        assert abs(er - float(o[f"erank_{i}"])) / float(o[f"erank_{i}"]) < 1e-4
        assert int(G.dead_units(m).sum()) == int(o[f"dead_{i}"])


# ------------------------------------------------------------------------------------------- whole trajectories
def _traj(inp, st, hp, X, Y):
    p = {"params": {f"layer_{i}": {"kernel": jnp.asarray(inp[f"W{i}"].T), "bias": jnp.asarray(inp[f"b{i}"])}
                    for i in range(len(st.widths) + 1)}}
    s = A.init_state(st, p)

    def one(s, xy):
        x, y = xy
        s, info = A.learner_step(st, s, hp, x[None], y[None] if st.loss == "ce" else y[None, None],
                                 jax.random.PRNGKey(0))
        return s, info["loss"]

    return np.asarray(jax.jit(lambda s: jax.lax.scan(one, s, (X, Y)))(s)[1])


@pytest.mark.parametrize("name,tol", [("bp_relu_sgd", 1e-4), ("bp_tanh_sgd", 1e-4), ("l2_relu_sgd", 1e-4),
                                      ("linear_sgd", 1e-4), ("bp_tanh_adam", 1e-3), ("bp_tanh_adamgnt", 1e-3)])
def test_scr_trajectory(name, tol):
    inp, o = load(f"traj_{name}")
    act, opt = str(inp["act"]), str(inp["opt"])
    widths = () if act == "linear" else (5,)
    st = A.Static(n_in=20, widths=widths, n_out=1, opt="adam" if opt.startswith("adam") else opt)
    hp = A.make_hp(lr=float(inp["lr"]), wd=float(inp["wd"]), act=ACT_ID.get(act, 0))
    ours = _traj(inp, st, hp, jnp.asarray(inp["X"], jnp.float32), jnp.asarray(inp["Y"]))
    assert ours.shape == o["errs"].shape == (3000,)
    assert float(np.abs(ours - o["errs"]).max()) <= tol


def test_mnist_trajectory():
    inp, o = load("traj_mnist_w100")
    st = A.Static(n_in=784, widths=(100, 100, 100), n_out=10, loss="ce")
    ours = _traj(inp, st, A.make_hp(lr=0.003), jnp.asarray(inp["X"], jnp.float32) / 255.0,
                 jnp.asarray(inp["Y"]).astype(jnp.int32))
    d = np.abs(ours - o["errs"])
    assert float(np.median(d)) <= 1e-5
    assert abs(float(ours.mean()) - float(o["errs"].mean())) <= 1e-4


# ------------------------------------------------------------------------------------------- Shrink-and-Perturb
def test_shrink_and_perturb_step():
    inp, o_s = load("snp_sigma1e-2")
    _, o_0 = load("snp_sigma0")
    sigma = float(inp["sigma"])
    st = A.Static(n_in=784, widths=(100, 100, 100), n_out=10, loss="ce")
    p = {"params": {f"layer_{i}": {"kernel": jnp.asarray(inp[f"W{i}"].T), "bias": jnp.asarray(inp[f"b{i}"])}
                    for i in range(4)}}
    x, y = jnp.asarray(inp["x"]), jnp.asarray(inp["y"]).astype(jnp.int32)

    def post(sig, key):
        hp = A.make_hp(lr=float(inp["lr"]), wd=float(inp["wd"]), sigma=sig)
        s, _ = A.learner_step(st, A.init_state(st, p), hp, x, y, key)
        return s["params"]["params"]

    p0, ps = post(0.0, jax.random.PRNGKey(0)), post(sigma, jax.random.PRNGKey(1))
    ours, theirs = [], []
    for i in range(4):
        L = f"layer_{i}"
        # deterministic part (SGD + torch-style weight decay on weights and biases) equals theirs
        np.testing.assert_allclose(np.asarray(p0[L]["kernel"]).T, o_0[f"W{i}"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(np.asarray(p0[L]["bias"]), o_0[f"b{i}"], atol=1e-6, rtol=0)
        for k_o, k_u in ((f"W{i}", "kernel"), (f"b{i}", "bias")):
            nt = (o_s[k_o] - o_0[k_o]).ravel()
            nu = (np.asarray(ps[L][k_u]) - np.asarray(p0[L][k_u])).T.ravel()
            assert np.all(nt != 0) and np.all(nu != 0), f"noise missing on {k_o}"
            theirs.append(nt)
            ours.append(nu)
    nt, nu = np.concatenate(theirs), np.concatenate(ours)
    assert abs(nt.std() / sigma - 1) < 0.03 and abs(nu.std() / sigma - 1) < 0.03
    assert abs(nt.mean()) < 5 * sigma / np.sqrt(nt.size) and abs(nu.mean()) < 5 * sigma / np.sqrt(nu.size)
    assert stats.ks_2samp(nt, nu).pvalue > 1e-3

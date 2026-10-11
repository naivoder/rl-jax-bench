"""Config options added in follow-up: noise skip (exact), paper-literal replacement count, paper Eq. 4/8 convention."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src import algorithm as A
from src.model import init_mlp


@pytest.fixture(autouse=True, scope="module")
def _float32_mode():
    """tests/test_grads.py turns x64 on for the whole session at import time; these runs are float32 (as in training)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


def _run(st, hp, n=40, seed=0):
    p = init_mlp(jax.random.PRNGKey(seed), 20, (5,), 1, math.sqrt(2))
    s = A.init_state(st, p)
    xs = jax.random.bernoulli(jax.random.PRNGKey(seed + 1), 0.5, (n, 1, 20)).astype(jnp.float32)
    ys = jax.random.normal(jax.random.PRNGKey(seed + 2), (n, 1, 1))

    def body(s, t):
        s, inf = A.learner_step(st, s, hp, xs[t], ys[t], jax.random.fold_in(jax.random.PRNGKey(9), t))
        return s, (inf["loss"], inf["n_rep"])
    return jax.jit(lambda s: jax.lax.scan(body, s, jnp.arange(n)))(s)


def test_noise_skip_is_exact_for_sigma_zero():
    hp = A.make_hp(lr=0.01, wd=1e-3, rho=0.05, mt=2)
    a, la = _run(A.Static(n_in=20, widths=(5,), n_out=1, replace_mode="bernoulli", perturb=True), hp)
    b, lb = _run(A.Static(n_in=20, widths=(5,), n_out=1, replace_mode="bernoulli", perturb=False), hp)
    for x, y in zip(jax.tree.leaves((a, la)), jax.tree.leaves((b, lb))):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_count_base_layer_vs_eligible():
    """'layer' = rho * n_l (paper Alg. 1) replaces more than 'eligible' while units are immature, equal once mature."""
    hp = A.make_hp(lr=0.01, rho=0.2, mt=3)
    st_e = A.Static(n_in=20, widths=(5,), n_out=1, replace_mode="accumulate", k_max=2)
    st_l = A.Static(n_in=20, widths=(5,), n_out=1, replace_mode="accumulate", k_max=2, count_base="layer")
    _, (_, re) = _run(st_e, hp, n=200)
    _, (_, rl) = _run(st_l, hp, n=200)
    assert int(rl.sum()) > int(re.sum())
    # deterministic accumulator with all units mature: 0.2 * 5 = 1 per step; capped by the eligible count
    assert int(rl.sum()) <= 200 and int(re.sum()) <= 200
    # no replacement before the first unit is mature (age > mt): steps 1..3
    assert int(rl[:3].sum()) == 0 and int(re[:3].sum()) == 0


def test_convention_p_is_lagged_correction():
    """Paper Eq. 4: fhat_t = f_{t-1} / (1 - eta^a_t); official 'C' uses f_t. Check one utility step by hand."""
    hp = A.make_hp(lr=0.01, rho=0.0, eta=0.9)
    h = jnp.asarray([[0.5, 1.0, 0.0, 2.0, 1.5]])
    w_in = jnp.ones((20, 5))
    w_out = jnp.ones((5, 1))
    f = jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    age = jnp.full((5,), 4.0)
    util = jnp.zeros(5)
    for conv in ("C", "P"):
        st = A.Static(n_in=20, widths=(5,), n_out=1, convention=conv)
        _, _, f_new, fhat, age_new = A.utility_step(st, h, w_in, w_out, util, f, age, hp, jax.random.PRNGKey(0))
        bc = 1 - 0.9 ** 5
        f_expect = 0.9 * f + 0.1 * h[0]
        assert np.allclose(np.asarray(f_new), np.asarray(f_expect), atol=1e-6)
        want = (f_expect if conv == "C" else f) / bc
        assert np.allclose(np.asarray(fhat), np.asarray(want), atol=1e-6)


# ------------------------------------------------------------- recheck: AdamGnT rule (official code history)
def test_adam_rule_closed_forms_and_reset():
    """standard: lr*sqrt(1-b2^t)/(1-b1^t)*m/(sqrt(v)+eps); paper_era (official dcc3619): lr*(1-b2^t)*m/(sqrt(v)+eps).
    First step after a reset (t = 1): the paper-era step is (1-b2)/(sqrt(1-b2)/(1-b1)) = sqrt(1-b2)*(1-b1) ~ 0.0032 of
    the standard one."""
    st = A.Static(n_in=1, widths=(), n_out=1, opt="adam")
    p = {"params": {"layer_0": {"kernel": jnp.zeros((1, 1)), "bias": jnp.zeros((1,))}}}
    g = jax.tree.map(lambda x: x + 0.5, p)
    opt = A.init_state(st, p)["opt"]
    std, _ = A.adam_update(st, p, g, opt, A.make_hp(lr=0.01))
    pe, _ = A.adam_update(st, p, g, opt, A.make_hp(lr=0.01, adam_rule="paper_era"))
    d_std = float(std["params"]["layer_0"]["kernel"][0, 0])
    d_pe = float(pe["params"]["layer_0"]["kernel"][0, 0])
    assert abs(d_std + 0.01) < 1e-6                            # standard Adam: first step = -lr * sign(g)
    assert abs(d_pe / d_std - math.sqrt(1 - 0.999) * (1 - 0.9)) < 1e-6


def test_adam_rule_standard_is_the_default_and_rule_agrees_at_large_t():
    assert int(A.make_hp(lr=0.01).adam_rule) == 0
    st = A.Static(n_in=20, widths=(5,), n_out=1, opt="adam", replace_mode="bernoulli")
    a, la = _run(st, A.make_hp(lr=0.01, rho=0.0))
    b, lb = _run(st, A.make_hp(lr=0.01, rho=0.0, adam_rule="standard"))
    for x, y in zip(jax.tree.leaves((a, la)), jax.tree.leaves((b, lb))):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    # the two rules differ by sqrt(1-b2^t)*(1-b1^t) <= 1: the paper-era trajectory takes smaller steps early on
    c, lc = _run(st, A.make_hp(lr=0.01, rho=0.0, adam_rule="paper_era"))
    assert not np.allclose(np.asarray(la[0]), np.asarray(lc[0]))


def test_adam_rule_config_validation():
    from src.train import arm_hps
    with pytest.raises(ValueError):
        arm_hps({"opt": "sgd", "lr": 0.01, "arms": [{"name": "x", "adam_rule": "paper_era"}]})
    with pytest.raises(ValueError):
        arm_hps({"opt": "adam", "lr": 0.01, "arms": [{"name": "x", "adam_rule": "gnt2023"}]})
    (_, hp, _), = arm_hps({"opt": "adam", "lr": 0.01, "arms": [{"name": "x", "adam_rule": "paper_era"}]})
    assert int(hp.adam_rule) == 1

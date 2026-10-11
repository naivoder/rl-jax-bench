"""Toy checks -- exactly computable checks of the CBP mechanics, baselines and diagnostics.

    JAX_PLATFORMS=cpu uv run python -m src.toys      # writes runs/toys/toys.json, prints PASS/FAIL

Every check returns (pass, numbers).  The same functions are asserted by tests/test_toys.py.
"""
from __future__ import annotations

import json
import math
import os
import sys
from math import comb

import jax
import jax.numpy as jnp
import numpy as np

from . import algorithm as A
from . import data as D
from . import diagnostics as G
from .model import init_mlp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-5

# hand-computed toy network 2 -> 3 (ReLU) -> 1, alpha = 0, eta = 0.5, m = 1, biases 0
W0 = [[1.0, -1.0, 0.5], [0.5, 1.0, -2.0]]
W1 = [[2.0], [0.1], [-1.0]]
XS = [[1.0, 1.0], [2.0, 0.0], [0.0, 1.0]]
# expected bias-corrected utility (u-hat) per step, sum-reduction (paper Eq. 6) -- hand-computed tables
EXPECTED = {
    "P": {"bcu": [[0, 0, 0], [1.333333, 0, 0], [1.333333, 0, 0.228571]],
          "fhat": [[0, 0, 0], [1, 0, 0], [1.571429, 0, 0.571429]],
          "util": [[1, 0, 0], [1.166667, 0, 0.2], [1.297619, 0.025, 0.214286]]},
    "E": {"bcu": [[2, 0, 0], [1.111111, 0, 0.266667], [1.492063, 0.028571, 0.266667]],
          "fhat": [[0, 0, 0], [1.5, 0, 0], [1.833333, 0, 0.666667]],
          "util": [[1, 0, 0], [0.833333, 0, 0.2], [1.305556, 0.025, 0.233333]]},
    "C": {"bcu": [[0, 0, 0], [0.148148, 0, 0.088889], [0.498866, 0.012245, 0.103401]],
          "fhat": [[1.5, 0, 0], [1.833333, 0, 0.666667], [1.071429, 0.571429, 0.285714]],
          "util": [[0, 0, 0], [0.111111, 0, 0.066667], [0.436508, 0.010714, 0.090476]]},
}


def toy_params(bias_out=0.0):
    return {"params": {"layer_0": {"kernel": jnp.asarray(W0, jnp.float32), "bias": jnp.zeros(3)},
                       "layer_1": {"kernel": jnp.asarray(W1, jnp.float32), "bias": jnp.full((1,), bias_out)}}}


def toy_static(conv="C", red="sum", opt="sgd"):
    return A.Static(n_in=2, widths=(3,), n_out=1, convention=conv, weight_reduction=red,
                    replace_mode="accumulate", opt=opt)


def check_toy_tables():
    """Hand-computed utility tables under conventions P, E, C; unit 1 selected at t = 3 in all three."""
    res, ok = {}, True
    hp = A.make_hp(lr=0.0, rho=1.0 / 6.0, mt=1, eta=0.5)   # rho * 3 eligible accumulates to exactly 1 at t = 3
    for conv in ("P", "E", "C"):
        st = toy_static(conv)
        s = A.init_state(st, toy_params())
        rows = {"bcu": [], "fhat": [], "util": []}
        sel = None
        for t, x in enumerate(XS):
            h = jax.nn.relu(jnp.asarray([x]) @ s["params"]["params"]["layer_0"]["kernel"])
            u, bcu, f, fhat, age = A.utility_step(st, h, s["params"]["params"]["layer_0"]["kernel"],
                                                  s["params"]["params"]["layer_1"]["kernel"], s["cbp"]["util"][0],
                                                  s["cbp"]["f"][0], s["cbp"]["age"][0], hp, jax.random.PRNGKey(0))
            rows["bcu"].append(np.asarray(bcu).tolist())
            rows["fhat"].append(np.asarray(fhat).tolist())
            rows["util"].append(np.asarray(u).tolist())
            s2, inf = A.learner_step(st, s, hp, jnp.asarray([x]), jnp.zeros((1, 1)), jax.random.PRNGKey(t))
            if int(inf["n_rep"][0]) > 0:
                sel = int(np.argmax(np.asarray(s2["params"]["params"]["layer_1"]["kernel"][:, 0]) == 0.0))
                assert t == 2, "replacement must happen at t = 3"
            s = s2
        err = max(float(np.max(np.abs(np.asarray(rows[k]) - np.asarray(EXPECTED[conv][k])))) for k in rows)
        good = err < TOL and sel == 1
        ok &= good
        res[conv] = {"max_abs_err": err, "selected_unit": sel, "pass": bool(good), **rows}
    # mean reduction (official code) scales y by n_in / n_out = 2 for this toy; ranking unchanged
    st = toy_static("C", "mean")
    s = A.init_state(st, toy_params())
    for t, x in enumerate(XS[:2]):
        s, _ = A.learner_step(st, s, hp, jnp.asarray([x]), jnp.zeros((1, 1)), jax.random.PRNGKey(t))
    ratio = np.asarray(s["cbp"]["bcu"][0]) / np.maximum(np.asarray(EXPECTED["C"]["bcu"][1]), 1e-12)
    good = bool(abs(ratio[0] - 2.0) < 1e-5 and abs(ratio[2] - 2.0) < 1e-5)
    res["mean_reduction_ratio"] = {"ratio_units_0_2": [float(ratio[0]), float(ratio[2])], "pass": good}
    return ok and good, res


def check_newborn():
    """Function preservation / gradient flow of a re-initialised unit."""
    st = toy_static("C", "mean")
    hp = A.make_hp(lr=0.0, rho=1.0 / 6.0, mt=1, eta=0.5)
    s = A.init_state(st, toy_params())
    x3 = jnp.asarray([XS[2]])
    out = lambda p: float(A.loss_and_out(st, p, x3, jnp.zeros((1, 1)), 0)[1][0][0, 0])
    before = out(s["params"])
    for t, x in enumerate(XS):
        s, _ = A.learner_step(st, s, hp, jnp.asarray([x]), jnp.zeros((1, 1)), jax.random.PRNGKey(t))
    after = out(s["params"])
    P = s["params"]["params"]
    w_out_row = float(np.abs(np.asarray(P["layer_1"]["kernel"][1])).max())
    new_in = np.asarray(P["layer_0"]["kernel"][:, 1])
    bound = math.sqrt(2.0) * math.sqrt(3.0 / 2.0)
    # gradient of the next step w.r.t. the newborn unit's incoming weights is exactly zero
    g = jax.grad(lambda p: A.loss_and_out(st, p, jnp.asarray([[1.0, 2.0]]), jnp.asarray([[3.0]]), 0)[0])(s["params"])
    g_in = float(np.abs(np.asarray(g["params"]["layer_0"]["kernel"][:, 1])).max())
    g_out = float(np.abs(np.asarray(g["params"]["layer_1"]["kernel"][1])).max())
    ok = (abs(before - 1.1) < 1e-6 and abs(after - 1.0) < 1e-6 and w_out_row == 0.0 and g_in == 0.0
          and np.all(np.abs(new_in) <= bound) and not np.allclose(new_in, [-1.0, 1.0]))
    return bool(ok), {"out_x3_before": before, "out_x3_after": after, "outgoing_row_max_abs": w_out_row,
                      "new_incoming": new_in.tolist(), "reinit_bound": bound, "grad_in_newborn_max_abs": g_in,
                      "grad_out_newborn_max_abs": g_out}


def check_bias_transfer():
    """Paper §6 bias transfer: with a constant input fhat = h exactly (convention C), so the replacement leaves the
    output unchanged when the transfer is on; when off (official code), the output changes by -h_r w_r."""
    res = {}
    x = jnp.asarray([[1.0, 2.0]])
    for bt in (1.0, 0.0):
        st = toy_static("C", "mean")
        hp = A.make_hp(lr=0.0, rho=0.25, mt=1, eta=0.99, bias_transfer=bt)
        s = A.init_state(st, toy_params())
        out0 = float(A.loss_and_out(st, s["params"], x, jnp.zeros((1, 1)), 0)[1][0][0, 0])
        h0 = np.asarray(jax.nn.relu(x @ s["params"]["params"]["layer_0"]["kernel"]))[0]
        nrep = 0
        for t in range(4):
            s, inf = A.learner_step(st, s, hp, x, jnp.zeros((1, 1)), jax.random.PRNGKey(t))
            nrep += int(inf["n_rep"][0])
        out1 = float(A.loss_and_out(st, s["params"], x, jnp.zeros((1, 1)), 0)[1][0][0, 0])
        res[f"bias_transfer_{int(bt)}"] = {"out_before": out0, "out_after": out1, "n_replaced": nrep, "h": h0.tolist()}
    ok = (abs(res["bias_transfer_1"]["out_after"] - res["bias_transfer_1"]["out_before"]) < 1e-5
          and res["bias_transfer_1"]["n_replaced"] >= 1
          and abs(res["bias_transfer_0"]["out_after"] - res["bias_transfer_0"]["out_before"]) > 1e-3)
    return bool(ok), res


def check_bias_correction_constants():
    """eta = 0.99, constant h = 1: P gives 0, 0.502513, 0.994260 at ages 1, 2, 101; C gives exactly 1."""
    out = {}
    for conv in ("P", "C", "E"):
        st = A.Static(n_in=1, widths=(1,), n_out=1, convention=conv)
        hp = A.make_hp(lr=0.0, eta=0.99)
        f = jnp.zeros(1)
        u = jnp.zeros(1)
        age = jnp.zeros(1)
        vals = {}
        for a in range(1, 102):
            u, bcu, f, fhat, age = A.utility_step(st, jnp.ones((1, 1)), jnp.ones((1, 1)), jnp.ones((1, 1)), u, f, age,
                                                  hp, jax.random.PRNGKey(0))
            if a in (1, 2, 101):
                vals[a] = float(fhat[0])
        out[conv] = vals
    ok = (abs(out["P"][1]) < 1e-7 and abs(out["P"][2] - 0.502513) < 1e-6 and abs(out["P"][101] - 0.994260) < 1e-6
          and all(abs(v - 1.0) < 1e-5 for v in out["C"].values())
          and abs(out["E"][2] - 1.0) < 1e-5 and abs(out["E"][101] - 1.0) < 1e-5)
    return bool(ok), out


def check_replacement_counts():
    """Accumulator: exactly one replacement every 5 steps at n_eligible = 2000, rho = 1e-4; Bernoulli frequency
    within 3 s.e. of rho * n_eligible; units with age <= m are never selected."""
    st = A.Static(n_in=1, widths=(2000,), n_out=1, replace_mode="accumulate")
    hp = A.make_hp(lr=0.0, rho=1e-4, mt=100)
    age = jnp.full((2000,), 500.0)
    bcu = jnp.arange(2000, dtype=jnp.float32)
    acc = jnp.zeros(())
    ks = []
    for t in range(1000):
        idx, act, acc, _ = A.select_units(st, bcu, age, acc, hp, jax.random.PRNGKey(t))
        ks.append(int(act.sum()))
    ks = np.asarray(ks)
    acc_ok = ks.sum() == 200 and all(ks[4::5] == 1) and ks.sum() == ks[4::5].sum()
    # Bernoulli on 5 units, rho = 1e-2 -> p = 0.05
    stb = A.Static(n_in=1, widths=(5,), n_out=1, replace_mode="bernoulli")
    hpb = A.make_hp(lr=0.0, rho=1e-2, mt=10)
    ageb = jnp.asarray([5.0, 11.0, 20.0, 3.0, 50.0])     # units 0 and 3 immature
    bcub = jnp.asarray([0.0, 1.0, 2.0, -1.0, 3.0])       # immature units have the smallest utilities
    sel_fn = jax.jit(jax.vmap(lambda k: A.select_units(stb, bcub, ageb, jnp.zeros(()), hpb, k)))
    idx, act, _, _ = sel_fn(jax.random.split(jax.random.PRNGKey(1), 100_000))
    idx, act = np.asarray(idx)[:, 0], np.asarray(act)[:, 0]
    p_hat = act.mean()
    p = 1e-2 * 3
    se = math.sqrt(p * (1 - p) / 100_000)
    chosen = set(idx[act].tolist())
    bern_ok = abs(p_hat - p) < 3 * se and chosen == {1}
    return bool(acc_ok and bern_ok), {"accumulator_total_in_1000_steps": int(ks.sum()),
                                      "accumulator_every_5th": bool(all(ks[4::5] == 1)),
                                      "bernoulli_rate": float(p_hat), "bernoulli_expected": p, "bernoulli_se": se,
                                      "units_chosen": sorted(chosen)}


def check_adam_reset():
    """After a CBP reset the first Adam update of a reset weight is -lr * g / (|g| + eps) (per-element step = 1)."""
    st = toy_static("C", "mean", opt="adam")
    hp = A.make_hp(lr=0.01, rho=1.0 / 6.0, mt=1, eta=0.5)
    s = A.init_state(st, toy_params())
    y = jnp.asarray([[0.3]])
    for t, x in enumerate(XS):
        s, inf = A.learner_step(st, s, hp, jnp.asarray([x]), y, jax.random.PRNGKey(t))
    r = int(np.argmax(np.asarray(s["params"]["params"]["layer_1"]["kernel"][:, 0]) == 0.0))
    t_cnt = np.asarray(s["opt"]["t"]["params"]["layer_1"]["kernel"][r, 0])
    p0 = s["params"]
    x = jnp.sign(p0["params"]["layer_0"]["kernel"][:, r])[None, :]   # makes the newborn unit active (h_r > 0)
    g = jax.grad(lambda p: A.loss_and_out(st, p, x, y, 0)[0])(p0)
    hp_noreplace = hp._replace(rho=jnp.asarray(0.0))
    s2, _ = A.learner_step(st, s, hp_noreplace, x, y, jax.random.PRNGKey(9))
    d = float(s2["params"]["params"]["layer_1"]["kernel"][r, 0] - p0["params"]["layer_1"]["kernel"][r, 0])
    gg = float(g["params"]["layer_1"]["kernel"][r, 0])
    expect = -0.01 * gg / (abs(gg) + 1e-8 / math.sqrt(1 - 0.999))   # AdamGnT eps placement
    ok = t_cnt == 0 and abs(d - expect) < 1e-7 and abs(gg) > 0 and int(inf["n_rep"][0]) == 1
    return bool(ok), {"unit": r, "step_counter_after_reset": float(t_cnt), "delta": d, "expected": expect, "grad": gg}


def check_diagnostics():
    I = jnp.eye(50) * 3.0
    r1 = jnp.outer(jnp.arange(1.0, 101.0), jnp.arange(1.0, 11.0))
    m31 = jnp.asarray([[3.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    rng = np.random.default_rng(0)
    Phi = jnp.asarray(rng.normal(size=(200, 20)), jnp.float32)
    vals = {"erank_I50": float(G.effective_rank(I)), "erank_rank1": float(G.effective_rank(r1)),
            "erank_sv31": float(G.effective_rank(m31)),
            "scale_invariance_diff": float(abs(G.effective_rank(Phi) - G.effective_rank(7.3 * Phi)))}
    h = jnp.asarray(np.c_[np.zeros(100), rng.uniform(0, 1, 100), np.r_[np.zeros(99), 1e-3], np.full(100, 2.0)])
    dead = np.asarray(G.dead_units(h)).tolist()
    vals["dead_mask"] = dead
    # analytic mean |w| at init
    k = jax.random.PRNGKey(0)
    p2000 = init_mlp(k, 784, (2000, 2000, 2000), 10, math.sqrt(2.0))
    p100 = init_mlp(k, 784, (100, 100, 100), 10, math.sqrt(2.0))
    st2000 = A.Static(n_in=784, widths=(2000, 2000, 2000), n_out=10)
    st100 = A.Static(n_in=784, widths=(100, 100, 100), n_out=10)
    vals["wmag_init_w2000"] = float(A.mean_abs_weight(st2000, p2000))
    vals["wmag_init_w100"] = float(A.mean_abs_weight(st100, p100))
    ok = (abs(vals["erank_I50"] - 50) < 1e-3 and abs(vals["erank_rank1"] - 1) < 1e-3
          and abs(vals["erank_sv31"] - 1.754765) < 1e-5 and vals["scale_invariance_diff"] < 1e-3
          and dead == [True, False, False, False]
          and abs(vals["wmag_init_w2000"] - 0.030044) < 2e-4 and abs(vals["wmag_init_w100"] - 0.060014) < 1e-3)
    return bool(ok), vals


def _p_binom_ge(n, k):
    return sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n


def check_scr_target_closed_form(n_mc=400_000):
    """LTU firing probability under i.i.d. uniform bits vs closed form.

    Official net: fire iff matches + 1.5 b > 14.7 -> b=+1: >= 14 of 20 match, b=-1: >= 17 of 20.
    Paper-text net (22 bits incl. constant 1): fire iff >= 16 of 22 match -> average 0.026239."""
    res, ok = {}, True
    for paper_text in (False, True):
        spec = D.scr_spec("flip_one", paper_text)
        tgt = D.scr_target(jax.random.PRNGKey(3), spec)
        n_rand_cols = spec.n_in - (1 if paper_text else 0)
        X = jax.random.bernoulli(jax.random.PRNGKey(4), 0.5, (n_mc, n_rand_cols)).astype(jnp.float32)
        if paper_text:
            X = jnp.concatenate([X, jnp.ones((n_mc, 1))], axis=1)
        _, feats = D.scr_target_apply(tgt, X)
        rate = np.asarray(feats.mean(0))
        if paper_text:
            cpos = np.asarray(tgt["W"][:, -1]) > 0
            closed = np.where(cpos, _p_binom_ge(21, 15), _p_binom_ge(21, 16))
        else:
            bpos = np.asarray(tgt["b"]) > 0
            closed = np.where(bpos, _p_binom_ge(20, 14), _p_binom_ge(20, 17))
        se = np.sqrt(closed * (1 - closed) / n_mc)
        z = np.abs(rate - closed) / np.maximum(se, 1e-12)
        good = bool(z.max() < 4.5)  # 100 units, Bonferroni-ish 4.5 sigma
        ok &= good
        res["paper_text" if paper_text else "official"] = {
            "mean_rate_mc": float(rate.mean()), "mean_rate_closed": float(closed.mean()),
            "max_z": float(z.max()), "pass": good}
    res["paper_text_average_closed_form_uniform_constant_sign"] = 0.5 * (_p_binom_ge(21, 15) + _p_binom_ge(21, 16))
    return ok, res


CHECKS = [check_toy_tables, check_newborn, check_bias_transfer, check_bias_correction_constants,
          check_replacement_counts, check_adam_reset, check_diagnostics, check_scr_target_closed_form]


def main():
    results, all_ok = {}, True
    for fn in CHECKS:
        ok, nums = fn()
        all_ok &= ok
        results[fn.__name__] = {"pass": bool(ok), "numbers": nums}
        print(f"[METRIC] toys {fn.__name__} pass={ok}", flush=True)
    results["TOYS"] = "PASS" if all_ok else "FAIL"
    out = os.path.join(REPO, "runs", "toys")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "toys.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"[INFO] TOYS {results['TOYS']}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

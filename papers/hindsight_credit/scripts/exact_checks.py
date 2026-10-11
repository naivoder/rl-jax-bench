"""Exact checks: toy correctness against exact enumeration.

    uv run python -m scripts.exact_checks [--out runs/exact_checks/exact_checks.json]

Checks (each PASS/FAIL with numbers):
  G1.1 env oracles: Fig. 3-right oracle Q(s0,short)-Q(s0,long), uniform-policy start values, bandit optimum.
  G1.2 exact hindsight (enumeration) equals the closed-form hindsight distributions.
  G1.3 Thm 1 / occupancy identity: E[Qhat(x0,a)] with exact h equals the DP Q (enumeration) and the MC mean of
       1e5 sampled rollouts is within 4 SE of it.
  G1.4 State HCA is unbiased: E[dtheta] over all trajectories with the exact pair-hindsight equals
       jax.grad of the DP value (<= 1e-8) on Shortcut, Bandit (observed), Delayed Effect (MC; both aliasings),
       both variants; T-step form on the fully observed Shortcut (T = 2, 3) with exact V; MC mean of dtheta
       within 4 SE.
  G1.5 T-step bias: App. F (aliased Delayed Effect, exact h -> zero gradient at x0) and the Thm-7
       counterexample (our estimator's expectation vs an independent numpy brute force and vs the truth).
  G1.6 Return HCA: unbiased on the bandit with continuous exact h_z (quadrature); Shortcut support-violation
       bias 0.2664 vs 0.4315 (hand derivation); binned-bandit bias table (hand derivation).
  G1.7 learned tabular h (our training code, frozen policy) converges to the exact fixed point within 4 SE.
"""
from __future__ import annotations

import argparse
import os

import src  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from src import data as D
from src.algorithm import counterfactual_q, return_hca_grads, state_hca_grads
from src.eval import (dp_q, dp_values, enumerate_trajectories, exact_hk, exact_pair_hindsight,
                      exact_return_hindsight, start_value, to_jnp_episode, trajectories_to_batch,
                      true_policy_gradient)
from src.model import Params, init_params, policy
from src.train import make_runner
from rl_jax_bench.jaxlog import Timer, check_device, info, metric, write_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = D.N_ACTIONS
RESULTS: dict = {}
NOISY_SE: dict = {}  # binomial SE of the Monte-Carlo h_z fixed point (noisy-return envs only)


def record(name, passed, **kv):
    RESULTS[name] = {"pass": bool(passed), **kv}
    metric(name, passed=bool(passed), **{k: v for k, v in kv.items() if isinstance(v, (int, float))})


def table_from(hstar: dict, shape) -> np.ndarray:
    """Dense hindsight-probability table (uniform where undefined) -> logits log h (−inf allowed)."""
    t = np.full(shape + (A,), 1.0 / A)
    for k, v in hstar.items():
        t[k] = v
    with np.errstate(divide="ignore"):
        return np.log(t)


def theta_for(env, pi_obs):
    with np.errstate(divide="ignore"):
        return jnp.asarray(np.log(pi_obs))


def expected(fn, env, pi_obs, params, seg_len=None):
    """Exact expectation over all trajectories of fn(params, episode) (mean rewards)."""
    probs, batch = trajectories_to_batch(env, enumerate_trajectories(env, pi_obs))
    out = jax.vmap(lambda e: fn(params, e))(to_jnp_episode(batch))
    return jax.tree_util.tree_map(lambda x: jnp.tensordot(jnp.asarray(probs), x, axes=1), out)


def sampled(fn, env, params, n, key):
    ea = D.env_arrays(env)
    pi = policy(params.theta)
    eps = jax.vmap(lambda k: D.sample_episode(ea, env.s0, env.L, lambda o: pi[o], k))(jax.random.split(key, n))
    return jax.vmap(lambda e: fn(params, e))(eps)


def rand_pi(env, key, scale=0.7):
    th = scale * jax.random.normal(key, (env.n_obs, A))
    return np.asarray(policy(th))


# ---------------------------------------------------------------- G1.1
def g1_env_oracles():
    env = D.shortcut()
    ea = D.env_arrays(env)
    target = [1.0878, 1.2507, 1.4601, 1.7262, 2.0593, 2.4697, 2.9676]
    got = []
    for p in np.linspace(0.1, 0.9, 7):
        th = jnp.tile(jnp.array([0.0, np.log(p / (1 - p))]), (env.n_obs, 1))
        q = dp_q(ea, th, env.L)[0]
        got.append(float(q[0] - q[1]))
    err = max(abs(a - b) for a, b in zip(got, target))
    record("G1.1a_fig3r_oracle", err < 5e-5, max_abs_err_vs_derivation=err, values=got)
    starts = {"shortcut": (D.shortcut(), -0.9630944), "delayed_effect_n5": (D.delayed_effect(5), 0.0),
              "delayed_effect_n3_s2": (D.delayed_effect(3, 2.0), 0.0), "bandit": (D.ambiguous_bandit(0.1), 1.5)}
    vals = {k: float(start_value(D.env_arrays(e), jnp.zeros((e.n_obs, A)), e.s0, e.L)) for k, (e, _) in starts.items()}
    err = max(abs(vals[k] - v) for k, (_, v) in starts.items())
    record("G1.1b_uniform_start_values", err < 1e-6, values=vals, max_abs_err=err)
    opt = {}
    for eps in (0.0, 0.1, 0.2, 0.3, 0.4):
        e = D.ambiguous_bandit(eps, 0.5, True)
        th = jnp.zeros((e.n_obs, A)).at[0].set(jnp.array([-50.0, 50.0]))
        opt[eps] = float(start_value(D.env_arrays(e), th, e.s0, e.L))
    err = max(abs(opt[e] - (2 - e)) for e in opt)
    record("G1.1c_bandit_optimum_2_minus_eps", err < 1e-9, values=opt, max_abs_err=err)
    sc = D.shortcut()
    th = jnp.zeros((sc.n_obs, A)).at[0].set(jnp.array([50.0, -50.0]))
    v = float(start_value(D.env_arrays(sc), th, sc.s0, sc.L))
    record("G1.1d_shortcut_optimum", abs(v + 0.1) < 1e-9, value=v)


# ---------------------------------------------------------------- G1.2
def g1_hindsight_closed_forms():
    env = D.shortcut()
    pi = np.full((env.n_obs, A), 0.5)
    hp, _ = exact_pair_hindsight(env, pi)
    hk = exact_hk(env, pi)
    G, Dd = 5, 6  # goal and absorbing state ids for n = 5
    got = {"h_occ(short|s0,G)": hp[(0, G)][0], "h_occ(short|s0,D)": hp[(0, Dd)][0], "h_occ(short|s0,s1)": hp[(0, 1)][0],
           "h_1(short|s0,G)": hk[(1, G)][0], "h_2(short|s0,G)": hk[(2, G)][0]}
    target = {"h_occ(short|s0,G)": 0.5477539, "h_occ(short|s0,D)": 0.2801699, "h_occ(short|s0,s1)": 0.0,
              "h_1(short|s0,G)": 1.0, "h_2(short|s0,G)": 0.0}
    err = max(abs(got[k] - target[k]) for k in got)
    # the j = i pair: h(.|x,x) = pi(.|x) for states visited at one depth only
    err_ii = max(abs(hp[(s, s)] - pi[s]).max() for s in range(5))
    record("G1.2a_shortcut_hindsight", err < 5e-8 and err_ii < 1e-12, values=got, max_abs_err=err, h_ii_minus_pi=err_ii)
    out = {}
    for eps in (0.1, 0.0, 0.2):
        e = D.ambiguous_bandit(eps, 1.5)
        hb, _ = exact_pair_hindsight(e, np.full((e.n_obs, A), 0.5))
        h = np.array([hb[(0, 1)], hb[(0, 2)]]).T  # [a, s]
        target = np.array([[1 - eps, eps], [eps, 1 - eps]])
        out[eps] = float(np.abs(h - target).max())
        eh = D.ambiguous_bandit(eps, 1.5, True)
        hh, _ = exact_pair_hindsight(eh, np.full((eh.n_obs, A), 0.5))
        out[f"hidden_{eps}"] = float(np.abs(hh[(0, 1)] - 0.5).max())  # aliased successor -> h = pi
    record("G1.2b_bandit_hindsight", max(out.values()) < 1e-12, max_abs_err=max(out.values()), per_case=out)


# ---------------------------------------------------------------- G1.3
def g1_thm1_identity(key):
    rows = {}
    ok = True
    for p_long in (0.5, 0.9, 0.1):
        env = D.shortcut()
        pi = np.tile([1 - p_long, p_long], (env.n_obs, 1))
        hp, _ = exact_pair_hindsight(env, pi)
        hz, _ = exact_return_hindsight(env, pi, D.bin_centres(env, 10))
        centres = jnp.asarray(D.bin_centres(env, 10))
        params = init_params(env.n_obs, 10)._replace(
            theta=theta_for(env, pi), psi=jnp.asarray(table_from(hp, (env.n_obs, env.n_obs))),
            phi=jnp.asarray(table_from(hz, (env.n_obs, 10))))
        q_true = np.asarray(dp_q(D.env_arrays(env), params.theta, env.L)[0])
        for m, imm in (("state_hca", "exact"), ("return_hca", "exact")):
            f = lambda p, e: counterfactual_q(m, p, e, 1.0, centres, imm)  # noqa: E731
            q_exp = np.asarray(expected(f, env, pi, params))
            qs = np.asarray(sampled(f, env, params, 100_000, jax.random.fold_in(key, int(p_long * 100))))
            mc, se = qs.mean(0), qs.std(0, ddof=1) / np.sqrt(len(qs))
            z = np.abs(mc - q_true) / se
            ok &= bool(np.abs(q_exp - q_true).max() < 1e-9 and z.max() < 4)
            rows[f"{m}_p{p_long}"] = {"q_true": q_true, "q_exact_expectation": q_exp, "q_mc": mc, "mc_se": se,
                                      "mc_z": z}
    record("G1.3_thm1_thm5_identity", ok, cases=rows)


# ---------------------------------------------------------------- G1.4
def state_grad_case(env, pi, seg_len=None, variant="official", V=None):
    hp, _ = exact_pair_hindsight(env, pi, seg_len)
    th = theta_for(env, pi)
    params = init_params(env.n_obs, 2)._replace(theta=th, psi=jnp.asarray(table_from(hp, (env.n_obs, env.n_obs))))
    if V is not None:
        params = params._replace(V=V)
    if variant == "paper":  # exact immediate-reward model r(o, a) = r_mean (state reward)
        r_o = np.zeros(env.n_obs)
        r_o[env.obs] = env.r_mean
        params = params._replace(rhat=jnp.tile(jnp.asarray(r_o)[:, None], (1, A)))
    f = lambda p, e: state_hca_grads(p, e, seg_len, 1.0, variant).theta  # noqa: E731
    g_exp = np.asarray(expected(f, env, pi, params))
    g_true = np.asarray(true_policy_gradient(D.env_arrays(env), th, env.s0, env.L))
    return params, f, g_exp, g_true


def g1_state_pg_unbiased(key):
    cases = {}
    ok = True
    envs = {"shortcut": D.shortcut(), "bandit_observed_e0.1": D.ambiguous_bandit(0.1, 1.5),
            "delayed_effect_n3_s2": D.delayed_effect(3, 2.0), "delayed_effect_n5": D.delayed_effect(5, 0.0),
            "delayed_effect_n3_aliased": D.delayed_effect(3, 2.0, "shared")}
    for i, (name, env) in enumerate(envs.items()):
        pi = rand_pi(env, jax.random.fold_in(key, i))
        for variant in ("official", "paper"):
            params, f, g_exp, g_true = state_grad_case(env, pi, None, variant)
            err = float(np.abs(g_exp - g_true).max())
            rec = {"max_abs_err": err, "grad_true_s0": g_true[env.obs[env.s0]], "grad_exp_s0": g_exp[env.obs[env.s0]]}
            if variant == "official":
                gs = np.asarray(sampled(f, env, params, 100_000, jax.random.fold_in(key, 50 + i)))
                mc, se = gs.mean(0), gs.std(0, ddof=1) / np.sqrt(len(gs))
                m = se > 0
                rec["mc_max_z"] = float((np.abs(mc - g_exp)[m] / se[m]).max()) if m.any() else 0.0
                ok &= rec["mc_max_z"] < 4.5  # many components tested at once; 4 SE per component, max over ~20
            ok &= err < 1e-8
            cases[f"{name}_{variant}"] = rec
    # T-step bootstrapped form on the fully observed Shortcut with the exact V
    env = D.shortcut()
    pi = rand_pi(env, jax.random.fold_in(key, 99))
    Vtrue = dp_values(D.env_arrays(env), theta_for(env, pi), env.L)
    for T in (2, 3):
        _, _, g_exp, g_true = state_grad_case(env, pi, T, "official", V=Vtrue)
        err = float(np.abs(g_exp - g_true).max())
        ok &= err < 1e-8
        cases[f"shortcut_T{T}_exactV"] = {"max_abs_err": err}
    record("G1.4_state_hca_pg_unbiased", ok, cases=cases)


# ---------------------------------------------------------------- G1.5
def g1_tstep_bias():
    # (a) App. F: aliased Delayed Effect, 3-step segments, exact h and the obs-level exact V -> no signal at x0
    env = D.delayed_effect(5, 0.0)
    pi = np.full((env.n_obs, A), 0.5)
    pi[0] = [0.6, 0.4]
    # obs-level V: E[V(X) | obs] under pi (chain obs c_j: 2*pi(a0)-1; F+/F-: +-1)
    ea = D.env_arrays(env)
    Vs = np.asarray(dp_values(ea, theta_for(env, pi), env.L))
    w = np.zeros(env.n_obs)
    Vo = np.zeros(env.n_obs)
    occ = np.zeros(env.P.shape[0])
    for p, xs, _ in enumerate_trajectories(env, pi):
        for s in xs:
            occ[s] += p
    for s in range(env.P.shape[0]):
        Vo[env.obs[s]] += occ[s] * Vs[s]
        w[env.obs[s]] += occ[s]
    Vo = np.where(w > 0, Vo / np.maximum(w, 1e-300), 0.0)
    _, _, g_exp, g_true = state_grad_case(env, pi, 3, "official", V=jnp.asarray(Vo))
    record("G1.5a_appF_aliased_bootstrap_no_signal", abs(g_exp[0]).max() < 1e-12 and abs(g_true[0]).max() > 0.1,
           grad_exp_x0=g_exp[0], grad_true_x0=g_true[0])
    # (b) Thm-7 counterexample with our segment pairs (T = 2, gamma = 1)
    env = D.thm7_counterexample(0.5)
    pi = np.full((env.n_obs, A), 0.5)
    Vtrue = dp_values(D.env_arrays(env), jnp.zeros((env.n_obs, A)), env.L)
    _, _, g_exp, g_true = state_grad_case(env, pi, 2, "official", V=Vtrue)
    # independent numpy brute force of the same estimator: Qhat(b) = sum_{j in window} h/pi R_j + h/pi V(X_T)
    hp, _ = exact_pair_hindsight(env, pi, 2)
    Vn = np.asarray(Vtrue)
    qhat = np.zeros(A)
    for p, xs, acts in enumerate_trajectories(env, pi):
        for b in range(A):
            q = env.r_mean[xs[0]]
            for j in range(1, min(3, len(xs))):
                hb = hp[(xs[0], xs[j])][b] / pi[0, b]
                is_boot = j == 2 and not env.terminal[xs[j]]
                q += hb * (Vn[xs[j]] if is_boot else env.r_mean[xs[j]])
            qhat[b] += p * q
    q_true = np.asarray(dp_q(D.env_arrays(env), jnp.zeros((env.n_obs, A)), env.L)[0])
    g_np = pi[0] * (qhat - pi[0] @ qhat)
    record("G1.5b_thm7_counterexample_bias", np.abs(g_np - g_exp[0]).max() < 1e-12 and np.abs(g_exp[0] - g_true[0]).max() > 0.1,
           qhat_ours=qhat, q_true=q_true, grad_exp_x0=g_exp[0], grad_numpy_x0=g_np, grad_true_x0=g_true[0])


# ---------------------------------------------------------------- G1.6
def bandit_return_case(eps, sigma, binned_centres=None, n_grid=40001):
    """Expected Return-HCA update on the bandit at uniform logits, with the exact (continuous or binned) h_z,
    by quadrature over the reward noise; our return_hca_grads with an h_z override."""
    env = D.ambiguous_bandit(eps, sigma)
    mu = np.array([1.0, 2.0])
    pi = np.array([0.5, 0.5])
    P = np.array([[1 - eps, eps], [eps, 1 - eps]])
    zs = np.linspace(mu.min() - 8 * sigma, mu.max() + 8 * sigma, n_grid)
    dz = zs[1] - zs[0]
    f = np.stack([norm.pdf(zs, mu[s], sigma) for s in range(2)])        # f[s, z]
    fa = P @ f                                                          # density of Z given a
    fz = pi @ fa
    if binned_centres is None:
        hz = pi[:, None] * fa / fz[None, :]                              # exact continuous h_z(a|z)
    else:
        idx = np.argmin(np.abs(binned_centres[None, :] - zs[:, None]), 1)
        hb = np.stack([np.array([(pi[a] * fa[a, idx == b]).sum() for b in range(len(binned_centres))])
                       / np.array([fz[idx == b].sum() for b in range(len(binned_centres))]) for a in range(2)])
        hz = hb[:, idx]
    n = 2 * 2 * n_grid
    st = np.zeros((n, 2), np.int64)
    ac = np.zeros((n, 2), np.int64)
    rw = np.zeros((n, 2))
    wts = np.zeros(n)
    hov = np.full((n, 2, A), 0.5)
    k = 0
    for a in range(2):
        for s in range(2):
            sl = slice(k, k + n_grid)
            st[sl, 1], ac[sl, 0], rw[sl, 1] = s + 1, a, zs
            wts[sl] = pi[a] * P[a, s] * norm.pdf(zs, mu[s], sigma) * dz
            hov[sl, 0, :] = hz.T
            k += n_grid
    batch = D.Episode(jnp.asarray(st), jnp.asarray(env.obs[st]), jnp.asarray(ac), jnp.asarray(rw),
                      jnp.ones((n, 2), bool), jnp.asarray(np.stack([np.ones(n, bool), np.zeros(n, bool)], 1)))
    params = init_params(env.n_obs, 1)
    g = jax.vmap(lambda e, h: return_hca_grads(params, e, 1.0, jnp.zeros(1), hz_override=h).theta)(batch, jnp.asarray(hov))
    g_exp = np.asarray(jnp.tensordot(jnp.asarray(wts), g, axes=1))[0]
    g_true = np.asarray(true_policy_gradient(D.env_arrays(env), params.theta, 0, env.L))[0]
    return g_exp, g_true


def g1_return_hca():
    cases = {}
    ok = True
    for eps, sigma in ((0.1, 1.5), (0.0, 0.5), (0.2, 0.5)):
        g_exp, g_true = bandit_return_case(eps, sigma)
        err = float(np.abs(g_exp - g_true).max())
        ok &= err < 1e-6
        cases[f"bandit_continuous_e{eps}_s{sigma}"] = {"grad_exp": g_exp, "grad_true": g_true, "max_abs_err": err}
    record("G1.6a_return_hca_unbiased_bandit_continuous", ok, cases=cases)
    # binned bias table (hand derivation: uniform bins on [mu_min-3s, mu_max+3s], outer bins open
    #  == nearest-midpoint centres)
    target = {(1.5, 0.1, 3): 0.1412, (1.5, 0.1, 10): 0.1931, (0.5, 0.0, 3): 0.1660, (0.5, 0.0, 10): 0.2377,
              (0.5, 0.2, 3): 0.1081, (0.5, 0.2, 10): 0.1460}
    rows, ok = {}, True
    for (sigma, eps, nb), tval in target.items():
        lo, hi = 1.0 - 3 * sigma, 2.0 + 3 * sigma
        centres = lo + (np.arange(nb) + 0.5) * (hi - lo) / nb
        g_exp, g_true = bandit_return_case(eps, sigma, centres, n_grid=200001)
        rows[f"s{sigma}_e{eps}_nb{nb}"] = {"grad_a2": float(g_exp[1]), "derivation": tval, "true": float(g_true[1])}
        ok &= abs(g_exp[1] - tval) < 2e-4
    record("G1.6b_binned_bias_table", ok, rows=rows)
    # Shortcut support violation (uniform policy): s0 'short' logit
    env = D.shortcut()
    pi = np.full((env.n_obs, A), 0.5)
    centres = D.bin_centres(env, 10)
    hz, _ = exact_return_hindsight(env, pi, centres)
    params = init_params(env.n_obs, 10)._replace(phi=jnp.asarray(table_from(hz, (env.n_obs, 10))))
    f = lambda p, e: return_hca_grads(p, e, 1.0, jnp.asarray(centres)).theta  # noqa: E731
    g_exp = np.asarray(expected(f, env, pi, params))
    g_true = np.asarray(true_policy_gradient(D.env_arrays(env), params.theta, 0, env.L))
    Z = np.array(sorted({round(float(sum(env.r_mean[xs])), 9) for _, xs, _ in enumerate_trajectories(env, pi)}))
    distinct = len({int(np.argmin(np.abs(centres - z))) for z in Z}) == len(Z)
    record("G1.6c_shortcut_support_violation_bias",
           abs(g_exp[0, 0] - 0.2664) < 5e-5 and abs(g_true[0, 0] - 0.4315) < 5e-5 and distinct,
           grad_short_s0_ours=float(g_exp[0, 0]), grad_short_s0_true=float(g_true[0, 0]),
           derivation=(0.2664, 0.4315), each_return_own_bin=distinct, grad_all_states_ours=g_exp[:, 0],
           grad_all_states_true=g_true[:, 0])


# ---------------------------------------------------------------- G1.7
def g1_hindsight_fixed_point():
    """G1.7a: the exact expected hindsight update of OUR training code vanishes at the enumerated h*
    (state pairs and return bins), i.e. the cross-entropy target of the implementation is h*."""
    out, ok = {}, True
    for name, env, pl in (("shortcut_p0.5", D.shortcut(), 0.5), ("shortcut_p0.8", D.shortcut(), 0.8),
                          ("bandit_observed", D.ambiguous_bandit(0.1, 1.5), 0.5),
                          ("delayed_effect_n5", D.delayed_effect(5, 0.0), 0.3)):
        pi = np.tile([1 - pl, pl], (env.n_obs, 1))
        hp, _ = exact_pair_hindsight(env, pi)
        centres = D.bin_centres(env, 10)
        hz, _ = exact_return_hindsight(env, pi, centres)
        params = init_params(env.n_obs, 10)._replace(theta=theta_for(env, pi),
                                                     psi=jnp.asarray(table_from(hp, (env.n_obs, env.n_obs))),
                                                     phi=jnp.asarray(table_from(hz, (env.n_obs, 10))))
        ds = np.asarray(expected(lambda p, e: state_hca_grads(p, e, None, 1.0).psi, env, pi, params))
        dr = np.asarray(expected(lambda p, e: return_hca_grads(p, e, 1.0, jnp.asarray(centres)).phi, env, pi, params))
        # perturbed tables must NOT be fixed points (the test can fail)
        pert = params._replace(psi=params.psi.at[..., 0].add(0.3), phi=params.phi.at[..., 0].add(0.3))
        dsp = np.asarray(expected(lambda p, e: state_hca_grads(p, e, None, 1.0).psi, env, pi, pert))
        e_s, e_r = float(np.abs(ds).max()), float(np.abs(dr).max())
        out[name] = {"max_abs_E_dpsi_at_hstar": e_s, "max_abs_E_dphi_at_hstar": e_r,
                     "max_abs_E_dpsi_perturbed_x0": float(np.abs(dsp[0]).max())}
        ok &= e_s < 1e-12 and e_r < 1e-12 and out[name]["max_abs_E_dpsi_perturbed_x0"] > 1e-3
    record("G1.7a_hindsight_fixed_point", ok, cases=out)


def g1_learned_h(key, n_runs=256, episodes=40000, lr_h=0.4, tau=1000.0, min_freq=0.02):
    """G1.7b: our own training loop (frozen uniform policy, Robbins-Monro hindsight lr 0.4/(1+e/1000), 40k episodes) learns h;
    the run-mean final tables match the exact fixed point within 4.5 SE on every interior cell
    (h* in [0.02, 0.98]) with >= ``min_freq`` expected training samples per episode. Boundary/rare cells are
    reported (their convergence is O(1/(alpha n)) in logit space and is covered by G1.7a)."""
    out, ok = {}, True
    cases = [("shortcut_state", D.shortcut(), "state_hca", 10), ("shortcut_return", D.shortcut(), "return_hca", 10),
             ("bandit_observed_state", D.ambiguous_bandit(0.1, 1.5), "state_hca", 10),
             ("bandit_observed_return", D.ambiguous_bandit(0.1, 1.5), "return_hca", 10),
             ("delayed_effect_n3_s2_return", D.delayed_effect(3, 2.0), "return_hca", 3)]
    for ci, (name, env, m, nb) in enumerate(cases):
        runner = make_runner(env, m, episodes=episodes, seg_len=None, n_bins=nb, bin_widen=2.0, gamma=1.0,
                             freeze_policy=True, h_lr_decay_tau=tau)
        res = runner(jax.random.split(jax.random.fold_in(key, ci), n_runs), jnp.array([0.0, lr_h, 0.0, 0.0]))
        pi = np.full((env.n_obs, A), 0.5)
        centres = D.bin_centres(env, nb)
        if m == "state_hca":
            hstar, freq = exact_pair_hindsight(env, pi)
            learned = np.asarray(jax.nn.softmax(res["params"].psi, -1))
        else:
            hstar, freq = (exact_return_hindsight(env, pi, centres) if env.r_std.max() == 0
                           else noisy_return_hindsight(env, centres))
            learned = np.asarray(jax.nn.softmax(res["params"].phi, -1))
        tested, skipped = [], []
        for k, v in hstar.items():
            col = learned[(slice(None),) + k][:, 0]
            mu, se = float(col.mean()), float(col.std(ddof=1) / np.sqrt(n_runs))
            se = float(np.sqrt(se ** 2 + (NOISY_SE.get(k, 0.0) if m == "return_hca" else 0.0) ** 2))
            row = {"key": list(map(int, k)), "exact": float(v[0]), "learned_mean": mu, "se": se,
                   "freq_per_episode": float(freq[k]), "z": abs(mu - v[0]) / max(se, 1e-12)}
            (tested if (0.02 <= v[0] <= 0.98 and freq[k] >= min_freq) else skipped).append(row)
        maxz = max((r["z"] for r in tested), default=0.0)
        ok &= maxz < 4.5 and len(tested) > 0
        out[name] = {"max_z": maxz, "n_tested": len(tested), "tested": tested,
                     "skipped_boundary_or_rare": skipped,
                     "max_abs_err_tested": max((abs(r["learned_mean"] - r["exact"]) for r in tested), default=0.0)}
    record("G1.7b_learned_h_converges", ok, n_runs=n_runs, episodes=episodes, lr_h=lr_h, tau=tau, cases=out)


def noisy_return_hindsight(env, centres, n_mc=2_000_000, seed=7):
    """h_z fixed point for envs with Gaussian reward noise, by a huge Monte-Carlo sample (uniform policy).
    Exact up to O(1/sqrt(n_mc)); used only where enumeration cannot integrate the noise."""
    params = init_params(env.n_obs, len(centres))
    ea = D.env_arrays(env)
    pi = policy(params.theta)
    from src.algorithm import mc_returns, return_bins
    def one(k):
        e = D.sample_episode(ea, env.s0, env.L, lambda o: pi[o], k)
        b = return_bins(mc_returns(e, 1.0), jnp.asarray(centres))
        return e.obs, b, e.action, e.decision
    o, b, a, d = jax.vmap(one)(jax.random.split(jax.random.PRNGKey(seed), n_mc))
    o, b, a, d = (np.asarray(x).ravel() for x in (o, b, a, d))
    out, freq = {}, {}
    for ox, bb in set(zip(o[d], b[d])):
        m = d & (o == ox) & (b == bb)
        freq[(int(ox), int(bb))] = float(m.sum() / n_mc)
        cnt = np.bincount(a[m], minlength=A).astype(float)
        out[(int(ox), int(bb))] = cnt / cnt.sum()
        NOISY_SE[(int(ox), int(bb))] = float(np.sqrt(cnt[0] * cnt[1] / cnt.sum() ** 3))
    return out, freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "exact_checks", "exact_checks.json"))
    args = ap.parse_args()
    check_device()
    key = jax.random.PRNGKey(1)  # exact-check key; disjoint from experiment keys (PRNGKey(seed) fold_in E-id)
    t = Timer()
    g1_env_oracles()
    g1_hindsight_closed_forms()
    g1_thm1_identity(jax.random.fold_in(key, 3))
    g1_state_pg_unbiased(jax.random.fold_in(key, 4))
    g1_tstep_bias()
    g1_return_hca()
    g1_hindsight_fixed_point()
    g1_learned_h(jax.random.fold_in(key, 7))
    RESULTS["all_pass"] = all(v["pass"] for v in RESULTS.values() if isinstance(v, dict))
    RESULTS["wall_s"] = t.lap()
    write_json(args.out, RESULTS)
    info("exact_checks", all_pass=RESULTS["all_pass"], wall_s=RESULTS["wall_s"], out=args.out)
    for k, v in RESULTS.items():
        if isinstance(v, dict):
            print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}")


if __name__ == "__main__":
    main()

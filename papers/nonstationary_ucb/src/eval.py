"""Gate checks and run summaries.

    uv run python -m src.eval --toys --out runs/toys          # Toy-check (hand derivation toys)
    uv run python -m src.eval --small runs/small --out runs/small/summary.json

Toy-check is a set of exact / closed-form / Monte-Carlo checks of the implementation (there are no gradients in
this paper). The same check functions are called by tests/test_grads.py with smaller Monte-Carlo sizes.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

from . import common
from .common import check_device, info, metric, warn, write_json
from . import algorithm as alg
from . import data
from . import theory as th

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

F64 = jnp.float64


# ---------------------------------------------------------------- helpers
def feed_history(name, K, history, params, buf_len=8):
    """Fold a fixed (action, reward) history (1-indexed arms) into a policy's statistics; return the state."""
    st = alg.init_state(name, K, buf_len)
    for t, (a, x) in enumerate(history, start=1):
        st = alg.update(name, st, jnp.asarray(a - 1, jnp.int32), jnp.asarray(x, F64), jnp.asarray(t, jnp.int32),
                        params, None, K)
    return st


def index_after(name, K, history, params, buf_len=8):
    st = feed_history(name, K, history, params, buf_len)
    t = len(history) + 1
    if name == "ucb1":
        return np.asarray(alg.ucb1_index(st.n, st.s, t, params.xi, params.B))
    if name == "ducb":
        return np.asarray(alg.ducb_index(st.n, st.s, params.xi, params.B))
    if name == "swucb":
        return np.asarray(alg.swucb_index(st.n, st.s, t, params.tau, params.xi, params.B))
    raise ValueError(name)


def reference_trajectory(name, mu, T, *, xi=0.5, B=1.0, gamma=1.0, tau=1, X=None, U_pol=None, g=0.0, alpha=0.0):
    """Pure-Python float64 reference, written straight from paper Algorithms 1-3 and Auer et al. Fig. 6
    (hand derivation), independently of src/algorithm.py. Lowest-index ties.

    ``X`` (T, K) is the reward table X_t(i) (default: deterministic, X = 1{mu >= 1}); ``U_pol`` (T,) are the
    EXP3.S draw uniforms. Returns (actions 0-indexed, number of exact ties met after initialisation).
    """
    K = len(mu) if X is None else X.shape[1]
    n, s = [0.0] * K, [0.0] * K
    w = [1.0] * K
    win: list[tuple[int, float]] = []
    acts, ties = [], 0
    for t in range(1, T + 1):
        if name == "exp3s":
            tot = sum(w)
            p = [(1 - g) * wi / tot + g / K for wi in w]
            c, a = 0.0, K - 1
            for i in range(K):
                c += p[i]
                if U_pol[t - 1] * sum(p) <= c:
                    a = i
                    break
        elif t <= K:
            a = t - 1
        else:
            idx = []
            for i in range(K):
                if name == "ucb1":
                    v = s[i] / n[i] + B * math.sqrt(xi * math.log(t) / n[i])
                elif name == "ducb":
                    v = (s[i] / n[i] + 2 * B * math.sqrt(xi * math.log(sum(n)) / n[i])) if n[i] > 0 else math.inf
                elif name == "swucb":
                    v = (s[i] / n[i] + B * math.sqrt(xi * math.log(min(t, tau)) / n[i])) if n[i] > 0 else math.inf
                idx.append(v)
            m = max(idx)
            ties += sum(1 for v in idx if v == m) > 1
            a = idx.index(m)
        x = (1.0 if mu[a] >= 1 else 0.0) if X is None else float(X[t - 1, a])
        acts.append(a)
        if name == "ucb1":
            n[a] += 1
            s[a] += x
        elif name == "ducb":
            n = [gamma * v for v in n]
            s = [gamma * v for v in s]
            n[a] += 1
            s[a] += x
        elif name == "swucb":
            win.append((a, x))
            n[a] += 1
            s[a] += x
            if len(win) > tau:
                oa, ox = win.pop(0)
                n[oa] -= 1
                s[oa] -= ox
        elif name == "exp3s":
            tot = sum(w)
            xhat = x / p[a]
            w = [wi * (math.exp(g * xhat / K) if i == a else 1.0) + math.e * alpha / K * tot for i, wi in enumerate(w)]
            mx = max(w)
            w = [wi / mx for wi in w]
    return np.asarray(acts), ties


def check_stochastic_reference(T=10_000, seed=3, r=0):
    """The JAX scan replays the SAME reward table as the pure-Python reference on one E1 and one E2
    replication (lowest-index ties; EXP3.S with the same draw uniforms): action sequences must be identical."""
    out = {}
    envs = {"e1": (data.abrupt(T, [[.5, .3, .4], [.5, .9, .4], [.5, .3, .4]], [3 * T // 10, T // 2]), 1),
            "e2": (data.periodic(T), 2)}
    for ename, (env, exp_id) in envs.items():
        K = env.K
        means, resets = alg.as_device_env(env.means, env.resets)
        ts = jnp.arange(1, T + 1, dtype=jnp.int32)
        cases = {"ucb1": dict(xi=0.5), "ducb": dict(xi=0.5, gamma=alg.paper_gamma(T)),
                 "swucb": dict(xi=0.5, tau=alg.paper_tau(T)),
                 "exp3s": dict(g=alg.exp3s_g(K, 2, T), alpha=1.0 / T)}
        for name, kw in cases.items():
            k_env, k_pol = alg.rep_keys(seed, exp_id, 0, alg.POLICY_ID[name], r + 1)
            U = np.asarray(jax.vmap(lambda t: alg.env_uniforms(k_env[r], t, K))(ts))
            X = (U < env.means).astype(np.float64)
            Up = np.asarray(jax.vmap(lambda t: jax.random.uniform(jax.random.fold_in(k_pol[r], t), (), F64))(ts))
            acts, aux, acc, _ = alg.trajectory(name, K, "lowest", 2048, alg.make_params(**kw), means, resets, k_env[r], k_pol[r])
            acts = np.asarray(acts)
            ref, ties = reference_trajectory(name, None, T, X=X, U_pol=Up, **kw)
            same = bool(np.array_equal(acts, ref))
            gap = (env.means.max(1) - env.means[np.arange(T), ref]).sum()
            out[f"{ename}_{name}"] = {"identical": same, "first_mismatch": -1 if same else int(np.argmax(acts != ref)),
                                      "ref_exact_ties": ties, "regret_jax": float(acc.regret), "regret_ref": float(gap),
                                      "arm1_plays": int((acts == 0).sum())}
    out["pass"] = all(v["identical"] and abs(v["regret_jax"] - v["regret_ref"]) < 1e-6 for v in out.values())
    return out


# ---------------------------------------------------------------- Toy-check checks (each returns a dict with "pass")
def check_g1a_indices(tol=1e-6):
    hist = [(1, 1.0), (2, 0.0), (1, 0.0)]  # hand derivation
    exp = {"ducb": [1.146248, 1.496149], "swucb": [0.588705, 0.588705], "ucb1": [1.088705, 0.832555]}
    got = {"ducb": index_after("ducb", 2, hist, alg.make_params(xi=0.5, gamma=0.5)),
           "swucb": index_after("swucb", 2, hist, alg.make_params(xi=0.5, tau=2)),
           "ucb1": index_after("ucb1", 2, hist, alg.make_params(xi=0.5))}
    err = {k: float(np.max(np.abs(got[k] - np.asarray(exp[k])))) for k in exp}
    st = feed_history("ducb", 2, hist, alg.make_params(gamma=0.5))
    stats = {"ducb_N": np.asarray(st.n).tolist(), "ducb_S": np.asarray(st.s).tolist(), "ducb_n_total": float(st.n.sum())}
    ok = all(e < tol for e in err.values()) and np.allclose(stats["ducb_N"], [1.25, 0.5]) and abs(stats["ducb_n_total"] - 1.75) < 1e-12
    return {"pass": bool(ok), "got": {k: v.tolist() for k, v in got.items()}, "expected": exp, "max_abs_err": err, **stats,
            "exact_tie_swucb": bool(got["swucb"][0] == got["swucb"][1])}


def check_g1a_tiebreak(n_keys=10_000):
    idx = jnp.asarray([0.588705, 0.588705], F64)
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(123), i))(jnp.arange(n_keys))
    acts = np.asarray(jax.vmap(lambda k: alg.argmax_tiebreak(idx, k, "random"))(keys))
    f = float((acts == 0).mean())
    se = math.sqrt(0.25 / n_keys)
    low = int(alg.argmax_tiebreak(idx, keys[0], "lowest"))
    # the full SW-UCB policy at round 4 of the G1-a history must also randomise
    st = feed_history("swucb", 2, [(1, 1.0), (2, 0.0), (1, 0.0)], alg.make_params(tau=2))
    pol = np.asarray(jax.vmap(lambda k: alg.select("swucb", st, jnp.asarray(4, jnp.int32), alg.make_params(tau=2), k, 2, "random")[0])(keys))
    f_pol = float((pol == 0).mean())
    ok = abs(f - 0.5) <= 4 * se and abs(f_pol - 0.5) <= 4 * se and low == 0
    return {"pass": bool(ok), "freq_arm1": f, "freq_arm1_policy_select": f_pol, "se": se, "lowest_mode_arm": low + 1}


def check_recursions(n_hist=1000, max_t=200, seed=0):
    """D-3 recursion == D-2 direct sum (discounted) and windowed recursion == direct window sum; D-4 identity.

    Each history has random length t_len < max_t, random gamma in (0.5, 0.999), tau in [1, 60), uniform
    actions and continuous rewards in [0, 1). The recursions run vmapped over histories (one compile); the
    state after t_len plays is compared with the direct sums computed in numpy.
    """
    rng = np.random.default_rng(seed)
    K = 3
    acts = rng.integers(0, K, (n_hist, max_t))
    xs = rng.random((n_hist, max_t))
    gammas = rng.uniform(0.5, 0.999, n_hist)
    taus = rng.integers(1, 60, n_hist)
    t_lens = rng.integers(1, max_t, n_hist)

    def run(a, x, gamma, tau):
        pd, pw = alg.make_params(gamma=gamma), alg.make_params(tau=tau)

        def body(c, z):
            sd, sw = c
            ai, xi, t = z
            sd = alg.update("ducb", sd, ai, xi, t, pd, None, K)
            sw = alg.update("swucb", sw, ai, xi, t, pw, None, K)
            return (sd, sw), (sd.n, sd.s, sw.n, sw.s)

        z = (a, x, jnp.arange(1, max_t + 1, dtype=jnp.int32))
        _, out = jax.lax.scan(body, (alg.init_state("ducb", K), alg.init_state("swucb", K, 64)), z)
        return out

    nd, sd, nw, sw = (np.asarray(v) for v in jax.jit(jax.vmap(run))(
        jnp.asarray(acts, jnp.int32), jnp.asarray(xs, F64), jnp.asarray(gammas, F64), jnp.asarray(taus, jnp.int32)))
    errs_d, errs_w, errs_n = [], [], []
    for h in range(n_hist):
        L, g, tau = int(t_lens[h]), float(gammas[h]), int(taus[h])
        a, x = acts[h, :L], xs[h, :L]
        w = g ** (L - np.arange(1, L + 1))
        Nd = np.array([(w * (a == i)).sum() for i in range(K)])
        Sd = np.array([(w * (a == i) * x).sum() for i in range(K)])
        lo = max(0, L - tau)
        Nw = np.array([(a[lo:] == i).sum() for i in range(K)])
        Sw = np.array([((a[lo:] == i) * x[lo:]).sum() for i in range(K)])
        errs_d.append(max(np.abs(nd[h, L - 1] - Nd).max(), np.abs(sd[h, L - 1] - Sd).max()))
        errs_w.append(max(np.abs(nw[h, L - 1] - Nw).max(), np.abs(sw[h, L - 1] - Sw).max()))
        errs_n.append(abs(nd[h, L - 1].sum() - th.n_disc(g, L)))
    out = {"max_err_discounted": float(max(errs_d)), "max_err_window": float(max(errs_w)),
           "max_err_ntotal_D4": float(max(errs_n)), "n_histories": n_hist}
    out["pass"] = bool(out["max_err_discounted"] < 1e-12 and out["max_err_window"] < 1e-12 and out["max_err_ntotal_D4"] < 1e-10)
    return out


def check_g1b_variance(n_draws=1_000_000, seed=1):
    """D-6: plays of arm 1 at s = 1, 3 (t = 3), gamma = 1/2, X ~ Ber(1/2): Var X̄(gamma,1) = 0.17 exactly."""
    p = alg.make_params(gamma=0.5)
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    x1 = (jax.random.uniform(k1, (n_draws,), F64) < 0.5).astype(F64)
    x3 = (jax.random.uniform(k2, (n_draws,), F64) < 0.5).astype(F64)

    def xbar(a, b):
        st = alg.init_state("ducb", 2)
        st = alg.update("ducb", st, jnp.int32(0), a, jnp.int32(1), p, None, 2)
        st = alg.update("ducb", st, jnp.int32(1), jnp.float64(0.0), jnp.int32(2), p, None, 2)
        st = alg.update("ducb", st, jnp.int32(0), b, jnp.int32(3), p, None, 2)
        return st.s[0] / st.n[0]

    xb = np.asarray(jax.jit(jax.vmap(xbar))(x1, x3))
    v = float(xb.var(ddof=1))
    # SE of a sample variance: sqrt((m4 - v^2)/n)
    m4 = float(((xb - xb.mean()) ** 4).mean())
    se = math.sqrt(max(m4 - v ** 2, 0.0) / n_draws)
    exact = 0.25 * (1 / 16 + 1) / 1.25 ** 2
    return {"pass": bool(abs(v - exact) <= 4 * se and abs(xb.mean() - 0.5) <= 4 * math.sqrt(v / n_draws)),
            "var_mc": v, "var_exact": exact, "se": se, "mean_mc": float(xb.mean()), "n": n_draws}


def check_g1c_constants():
    c = {}
    c["eta_identity"] = {xi: 2 * xi * (1 - th.eta_of_xi(xi) ** 2 / 16) for xi in (0.6, 1.0, 2.0)}
    grid = np.linspace(1e-4, 20, 200_001)
    c["ineq16_min_gap"] = float(th.ineq16_gap(grid).min())
    c["remark20"] = th.remark20_constant(0.3)
    c["B_gamma_limit"] = th.B_gamma_limit(0.6, 0.1)
    c["B_gamma_at_1e-6"] = th.B_gamma(1 - 1e-6, 10 ** 12, 0.6, 0.1)
    c["C_gamma_at_1e-6"] = th.C_gamma(1 - 1e-6, 0.6, 3)
    c["C_tau_limit"] = th.C_tau_limit(0.6, 0.1)
    c["C_tau_at_1e8"] = th.C_tau(10 ** 8, 10 ** 12, 0.6, 0.1)
    c["exp3s_g_K3_S2_T1e4"] = alg.exp3s_g(3, 2, 10 ** 4)
    c["exp3s_g_K2_S2_T1e4"] = alg.exp3s_g(2, 2, 10 ** 4)
    c["paper_gamma_T1e4"] = alg.paper_gamma(10 ** 4)
    c["paper_tau_T1e4"] = alg.paper_tau(10 ** 4)
    c["paper_gamma_T2e4"] = alg.paper_gamma(2 * 10 ** 4)
    c["paper_tau_T2e4"] = alg.paper_tau(2 * 10 ** 4)
    c["D_gamma_xi05"] = th.D_gamma(0.9975, 0.5, 3)
    c["thm1_bound_E1"] = th.thm1_bound(0.9975, 10 ** 4, 2, 0.6, 0.1, 3)
    c["thm7_bound_E1"] = th.thm7_bound(1214, 10 ** 4, 2, 0.6, 0.1)
    lb = th.lower_bound_constants(0.5, 0.3, 0.7)
    c["lb_C_paper"], c["lb_C_rederived"] = lb["C_paper"], lb["C_rederived"]
    checks = {
        "eta_identity": all(abs(v - 1) < 1e-12 for v in c["eta_identity"].values()),
        "ineq16": c["ineq16_min_gap"] >= -1e-15,
        "remark20": abs(c["remark20"] - 1.99142) < 5e-6,
        "B_gamma_limit": abs(c["B_gamma_limit"] - 2612.82) < 0.01 and abs(c["B_gamma_at_1e-6"] - 2612.99) < 0.01,
        "C_gamma_to_1": abs(c["C_gamma_at_1e-6"] - 1.030) < 1e-3,
        "C_tau_limit": abs(c["C_tau_limit"] - 242.07) < 0.01 and abs(c["C_tau_at_1e8"] - 242.17) < 0.01,
        "exp3s_g": abs(c["exp3s_g_K3_S2_T1e4"] - 0.0638305) < 1e-7 and abs(c["exp3s_g_K2_S2_T1e4"] - 0.0512039) < 1e-7,
        "gamma_tau": abs(c["paper_gamma_T1e4"] - 0.9975) < 1e-15 and c["paper_tau_T1e4"] == 1214
        and abs(c["paper_gamma_T2e4"] - 0.9982322) < 1e-7 and c["paper_tau_T2e4"] == 1780,
        "D_gamma": round(c["D_gamma_xi05"]) == 2634,
        "thm_bounds_E1": round(c["thm1_bound_E1"]) == 397041 and round(c["thm7_bound_E1"]) == 17950,
        "lb_constants": abs(c["lb_C_paper"] - 0.139878) < 1e-6 and abs(c["lb_C_rederived"] - 0.034970) < 1e-6,
    }
    return {"pass": all(checks.values()), "checks": checks, "values": c}


def check_g1d_deterministic(T=2000, tie="lowest"):
    """Deterministic arms (1, 0): JAX scan == pure-Python reference action sequences; regret == #arm-2 plays."""
    env = data.constant(T, [1.0, 0.0])
    means, resets = alg.as_device_env(env.means, env.resets)
    k_env, k_pol = alg.rep_keys(0, 0, 0, 0, 1)
    cases = {"ucb1": dict(xi=0.5), "ducb": dict(xi=0.5, gamma=0.99), "swucb": dict(xi=0.5, tau=50)}
    out = {}
    for name, kw in cases.items():
        acts, aux, acc, _ = alg.trajectory(name, 2, tie, 64, alg.make_params(**kw), means, resets, k_env[0], k_pol[0])
        acts = np.asarray(acts)
        ref, ties = reference_trajectory(name, [1.0, 0.0], T, **kw)
        out[name] = {"identical": bool(np.array_equal(acts, ref)), "first_mismatch": int(np.argmax(acts != ref)) if not np.array_equal(acts, ref) else -1,
                     "arm2_plays": int((acts == 1).sum()), "regret": float(acc.regret),
                     "regret_equals_arm2_plays": bool(float(acc.regret) == float((acts == 1).sum())), "ref_exact_ties": ties}
    out["pass"] = all(v["identical"] and v["regret_equals_arm2_plays"] for v in out.values())
    return out


def check_exp3s_logspace(steps=50, K=3, seed=0):
    """log-space EXP3.S == naive weights on a short trajectory; probs sum to 1 and are >= g/K."""
    rng = np.random.default_rng(seed)
    g, alpha = 0.2, 1e-2
    logw = jnp.zeros(K, F64)
    w = np.ones(K)
    max_err, min_p, max_sum_err = 0.0, 1.0, 0.0
    for _ in range(steps):
        p_j = np.asarray(alg.exp3s_probs(logw, g, K))
        p_n = (1 - g) * w / w.sum() + g / K
        max_err = max(max_err, float(np.abs(p_j - p_n).max()))
        min_p = min(min_p, float(p_j.min()))
        max_sum_err = max(max_sum_err, abs(float(p_j.sum()) - 1))
        a = int(rng.choice(K, p=p_n))
        x = float(rng.random())
        logw = alg.exp3s_update_logw(logw, jnp.int32(a), jnp.float64(x), jnp.asarray(p_j), g, alpha, K)
        xhat = np.zeros(K)
        xhat[a] = x / p_n[a]
        w = w * np.exp(g * xhat / K) + math.e * alpha / K * w.sum()
    return {"pass": bool(max_err < 1e-10 and min_p >= g / K - 1e-15 and max_sum_err < 1e-12),
            "max_abs_prob_err": max_err, "min_prob": min_p, "g_over_K": g / K, "max_sum_err": max_sum_err}


def check_g1e_thm18(n_paths=1_000_000, gamma=0.99, mu=0.3, t=500, deltas=(0.5, 1.0, 1.5, 2.0), seed=7, chunk=250_000):
    """Theorem 18 Monte Carlo (hand derivation): eps_s = 1{X_{s-1} = 1 or s odd} (predictable; X_0 := 0)."""
    @jax.jit
    def stat(key):
        def body(c, s):
            S, M, N2, xprev = c
            x = (jax.random.uniform(jax.random.fold_in(key, s), xprev.shape, F64) < mu).astype(F64)
            eps = ((xprev == 1.0) | (s % 2 == 1)).astype(F64)  # uses only X_{s-1} and s: predictable
            return (gamma * S + eps * x, gamma * M + eps * mu, gamma ** 2 * N2 + eps, x), None
        z = jnp.zeros(chunk, F64)
        (S, M, N2, _), _ = jax.lax.scan(body, (z, z, z, z), jnp.arange(1, t + 1))
        return (S - M) / jnp.sqrt(jnp.where(N2 > 0, N2, 1.0)), N2 > 0

    vals = []
    for c in range(n_paths // chunk):
        v, ok = stat(jax.random.fold_in(jax.random.PRNGKey(seed), c))
        vals.append(np.asarray(v)[np.asarray(ok)])
    v = np.concatenate(vals)
    n_t = th.n_disc(gamma, t)
    bound = th.thm18_bound(np.asarray(deltas), n_t)
    rows = []
    for d, b in zip(deltas, bound):
        p = float((v > d).mean())
        se = math.sqrt(max(p * (1 - p), 1.0 / len(v)) / len(v))
        rows.append({"delta": d, "empirical": p, "se": se, "bound": float(b), "holds": bool(p <= b + 3 * se),
                     "bound_over_empirical": float(b / p) if p > 0 else None})
    return {"pass": all(r["holds"] for r in rows), "rows": rows, "n_paths": int(len(v)), "n_t": n_t}


def toys(out_dir: str, full: bool = True) -> dict:
    check_device()
    res = {}
    for name, fn in [("G1a_indices", check_g1a_indices), ("G1a_tiebreak", check_g1a_tiebreak),
                     ("D3_recursions", check_recursions), ("G1b_variance", check_g1b_variance),
                     ("G1c_constants", check_g1c_constants), ("G1d_deterministic_lowest", check_g1d_deterministic),
                     ("G1d_deterministic_random", lambda: check_g1d_deterministic(tie="random")),
                     ("stochastic_reference", check_stochastic_reference),
                     ("EXP3S_logspace", check_exp3s_logspace), ("G1e_thm18", check_g1e_thm18)]:
        tm = common.Timer()
        r = fn()
        r["wall_s"] = tm.lap()
        res[name] = r
        metric("toys check", check=name, ok=r["pass"], wall_s=r["wall_s"])
    env_e1 = data.self_test(data.abrupt(10_000, [[.5, .3, .4], [.5, .9, .4], [.5, .3, .4]], [3000, 5000]))
    env_e2 = data.self_test(data.periodic(10_000))
    res["env_selftests"] = {"e1": env_e1, "e2": env_e2,
                            "pass": env_e1["best_arm_counts"][0] == 8000 and env_e1["best_arm_switches"] == 2
                            and env_e2["best_arm_counts"][0] == 4999 and abs(env_e2["constant_arm_regret"][1] - 1273.2395) < 1e-3}
    res["PASS"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
    res["git_sha"], res["git_dirty"], res["device"] = common.git_sha(), common.git_dirty(), str(jax.devices()[0])
    write_json(os.path.join(out_dir, "metrics.json"), res)
    info("TOY CHECKS", PASS=res["PASS"])
    return res


# ---------------------------------------------------------------- Small-scale summary
def summarize_runs(run_dir: str) -> dict:
    """Seed-level summary of runs/<..>/seed_*/metrics.json: mean +- SE over seed means, ratio to read-offs."""
    files = sorted(glob.glob(os.path.join(run_dir, "seed_*", "metrics.json")))
    recs = [json.load(open(f)) for f in files]
    if not recs:
        raise FileNotFoundError(f"no seed_*/metrics.json under {run_dir}")
    cfg = recs[0]["config"]
    ro = cfg.get("readoff", {})
    out = {"experiment": recs[0]["experiment"], "seeds": [r["seed"] for r in recs], "reps_per_seed": recs[0]["reps"],
           "crn_identical_all": all(r["crn_identical"] for r in recs),
           "any_bad": any(p["any_bad"] for r in recs for p in r["policies"].values()),
           "means_sha256": sorted({r["env"]["means_sha256"] for r in recs}),
           "uniform_fps_distinct_across_seeds": len({r["env"]["uniforms_fp_sha256"] for r in recs}) == len(recs),
           "env_self_test": recs[0]["env"]["self_test"], "policies": {}}
    for name in recs[0]["policies"]:
        seed_means = np.array([r["policies"][name]["final_regret_mean"] for r in recs])
        arm1 = np.array([r["policies"][name]["final_arm1_mean"] for r in recs])
        allreps = np.concatenate([r["policies"][name]["final_regret"] for r in recs])
        m = float(seed_means.mean())
        se = float(seed_means.std(ddof=1) / math.sqrt(len(recs))) if len(recs) > 1 else None
        d = {"final_regret_mean": m, "final_regret_se_seeds": se, "final_regret_median_all": float(np.median(allreps)),
             "p05_p95_all": [float(np.percentile(allreps, 5)), float(np.percentile(allreps, 95))],
             "final_arm1_mean": float(arm1.mean()),
             "regret_at_mean": {k: float(np.mean([r["policies"][name]["regret_at_mean"][k] for r in recs]))
                                for k in recs[0]["policies"][name]["regret_at_mean"]},
             "throughput_steps_reps_per_s": float(np.mean([r["policies"][name]["throughput_steps_reps_per_s"] for r in recs])),
             "compile_s_first": recs[0]["policies"][name]["compile_s"]}
        paper = ro.get("final_regret", {}).get(name)
        if paper is not None:
            d["paper_final_regret"] = paper
            d["ratio_to_paper"] = m / paper
            d["rel_diff"] = (m - paper) / paper
            d["within_order_of_magnitude"] = bool(0.1 <= m / paper <= 10)
            d["paper_value_percentile_in_runs"] = float((allreps < paper).mean() * 100)
        pa = ro.get("arm1_pulls_T", {}).get(name)
        if pa is not None:
            d["paper_arm1_pulls_T"] = pa
        out["policies"][name] = d
    return out


def small(run_root: str, out_path: str) -> dict:
    res = {"experiments": {}}
    for sub in sorted(os.listdir(run_root)):
        p = os.path.join(run_root, sub)
        if os.path.isdir(p) and glob.glob(os.path.join(p, "seed_*", "metrics.json")):
            res["experiments"][sub] = summarize_runs(p)
    checks = {}
    for e, s in res["experiments"].items():
        pol = s["policies"]
        checks[f"{e}_integrity"] = s["crn_identical_all"] and not s["any_bad"] and s["uniform_fps_distinct_across_seeds"]
        checks[f"{e}_order_of_magnitude"] = all(v.get("within_order_of_magnitude", True) for v in pol.values())
        checks[f"{e}_direction_DUCB_SWUCB_below_UCB1_EXP3S"] = all(
            pol[a]["final_regret_mean"] < pol[b]["final_regret_mean"] for a in ("ducb", "swucb") for b in ("ucb1", "exp3s"))
    res["checks"] = checks
    res["PASS"] = all(v for k, v in checks.items() if not k.endswith("DUCB_SWUCB_below_UCB1_EXP3S")) and all(checks.values())
    write_json(out_path, res)
    for k, v in checks.items():
        metric("small check", check=k, ok=v)
    info("SMALL SCALE", PASS=res["PASS"])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--toys", action="store_true")
    ap.add_argument("--small", default=None, help="run root with <exp>/seed_*/metrics.json")
    ap.add_argument("--summarize", default=None, help="one experiment dir with seed_*/metrics.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.toys:
        toys(a.out)
    if a.small:
        small(a.small, a.out)
    if a.summarize:
        write_json(a.out, summarize_runs(a.summarize))


if __name__ == "__main__":
    main()

"""Toy checks (toy problems + exactly computable properties). Each check returns a dict with the measured
numbers, the expected numbers and a boolean `pass`. Used by tests/test_toys.py and scripts/toy_checks.py.
float64 where the toy is exact (jax.experimental.enable_x64); float32 where the production dtype is the point.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
def enable_x64():
    """float64 context (jax 0.11: jax.enable_x64(True))."""
    return jax.enable_x64(True)

from . import algorithm as alg
from . import model as mdl


def _close(a, b, tol):
    return bool(np.all(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)) <= tol))


# ------------------------------------------------------------------ ObGD on the linear toy
def obgd_toy() -> dict:
    with enable_x64():
        g, lam, a, kappa = 0.9, 0.8, 1.0, 2.0
        x0, x1 = jnp.array([1.0, 0.5]), jnp.array([0.5, 1.0])
        v = lambda w, x: w @ x  # noqa: E731
        w = jnp.zeros(2)
        z = jnp.zeros(2)
        rec = {}
        # step 0: x0 -r=1-> x1
        d0 = 1.0 + g * v(w, x1) - v(w, x0)
        z = alg.trace_accumulate(z, x0, g * lam)
        w1, i0 = alg.obgd_apply(w, z, d0, a, kappa)
        dplus0 = 1.0 + g * v(w1, x1) - v(w1, x0)
        rec["step0"] = dict(delta=float(d0), z_l1=float(i0.z_l1), M=float(i0.M), alpha_eff=float(i0.step),
                            w=np.asarray(w1).tolist(), delta_plus=float(dplus0), xi=float((d0 - dplus0) / d0),
                            dw_l1=float(i0.dw_l1))
        # step 1: x1 -r=0-> terminal
        d1 = 0.0 - v(w1, x1)
        z = alg.trace_accumulate(z, x1, g * lam)
        w2, i1 = alg.obgd_apply(w1, z, d1, a, kappa)
        dplus1 = 0.0 - v(w2, x1)
        rec["step1"] = dict(delta=float(d1), z=np.asarray(z).tolist(), z_l1=float(i1.z_l1), M=float(i1.M),
                            alpha_eff=float(i1.step), w=np.asarray(w2).tolist(), delta_plus=float(dplus1),
                            xi=float((d1 - dplus1) / d1), dw_l1=float(i1.dw_l1))
        # inactive regime
        wi, ii = alg.obgd_apply(jnp.zeros(2), jnp.array([0.05, 0.05]), 0.2, a, kappa)
        rec["inactive"] = dict(M=float(ii.M), alpha_eff=float(ii.step), dw=np.asarray(wi).tolist())
    exp = {"step0": dict(delta=1.0, z_l1=1.5, M=3.0, alpha_eff=1 / 3, w=[1 / 3, 1 / 6], delta_plus=0.8833333333333333,
                         xi=0.11666666666666667, dw_l1=0.5),
           "step1": dict(delta=-1 / 3, z=[1.22, 1.36], z_l1=2.58, M=5.16, alpha_eff=0.1937984496124031,
                         w=[0.2545219638242894, 0.0788113695090439], delta_plus=-0.2060723514211886,
                         xi=0.3817829457364341, dw_l1=1 / 6),
           "inactive": dict(M=0.2, alpha_eff=1.0, dw=[0.01, 0.01])}
    ok = all(_close(rec[s][k], exp[s][k], 1e-9) for s in exp for k in exp[s])
    bound_ok = rec["step0"]["xi"] <= 1 / kappa and rec["step1"]["xi"] <= 1 / kappa  #  holds on this toy
    return dict(measured=rec, expected=exp, tol=1e-9, xi_le_1_over_kappa=bound_ok, **{"pass": ok and bound_ok})


# ------------------------------------------------------------------ G1.3 P1: ||dw||_1 <= min(|delta|,1)/kappa
def obgd_l1_property(n: int = 20000, seed: int = 0) -> dict:
    k = jax.random.PRNGKey(seed)
    kz, kd, ks, kk = jax.random.split(k, 4)
    z = jax.random.normal(kz, (n, 37)) * jnp.exp(jax.random.uniform(ks, (n, 1), minval=-8, maxval=6))
    d = jax.random.normal(kd, (n,)) * jnp.exp(jax.random.uniform(kk, (n,), minval=-6, maxval=6))
    kappa = 2.0

    def one(zz, dd):
        _, info = alg.obgd_apply(jnp.zeros(37), zz, dd, 1.0, kappa)
        return info

    info = jax.vmap(one)(z, d)
    bound = jnp.minimum(jnp.abs(d), 1.0) / kappa
    rel = (info.dw_l1 - bound) / jnp.maximum(bound, 1e-30)
    active = info.M > 1
    eq_err = jnp.max(jnp.where(active, jnp.abs(rel), 0.0))
    ok = bool(jnp.max(rel) <= 1e-5) and bool(eq_err <= 1e-5)
    return dict(n=n, frac_active=float(active.mean()), max_rel_excess=float(jnp.max(rel)),
                max_rel_err_equality_when_active=float(eq_err), **{"pass": ok})


# ------------------------------------------------------------------ Welford / ScaleReward toy
def welford_toy() -> dict:
    w = alg.welford_init()
    rows = []
    for x in (1.0, 2.0, 4.0):
        w = alg.welford_update(w, x)
        rows.append([int(w.count), float(w.mean), float(w.p), float(alg.welford_var(w))])
    exp = [[1, 1.0, 0.0, 1.0], [2, 1.5, 0.5, 0.5], [3, 2.3333333, 4.6666667, 2.3333333]]
    wr, u = alg.welford_init(), jnp.float32(0.0)
    us, vs, rs = [], [], []
    for r, t in zip((1.0, 0.0, 2.0, 1.0), (0, 0, 1, 0)):
        wr, u, rsc = alg.scale_reward(wr, u, jnp.float32(r), jnp.float32(t), 0.99)
        us.append(float(u))
        vs.append(float(alg.welford_var(wr)))
        rs.append(float(rsc))
    exp_u, exp_v = [1.0, 0.99, 2.0, 2.98], [1.0, 5e-5, 0.3367, 0.9050917]
    exp_r = [r / np.sqrt(v + 1e-8) for r, v in zip((1.0, 0.0, 2.0, 1.0), exp_v)]
    ok = _close(rows, exp, 1e-6) and _close(us, exp_u, 1e-6) and _close(vs, exp_v, 2e-6) and \
        np.allclose(rs, exp_r, rtol=1e-4)
    return dict(welford=rows, welford_expected=exp, u=us, u_expected=exp_u, var_u=vs, var_u_expected=exp_v,
                r_scaled=rs, r_scaled_expected=exp_r, **{"pass": bool(ok)})


# ------------------------------------------------------------------ G1.5 float32 Kahan Welford over 5e6 samples
def welford_precision(n: int = 5_000_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x = np.stack([(rng.random(n) < 0.05).astype(np.float64),                 # sparse binary MinAtar-like channel
                  rng.standard_normal(n) * 3.0 + 50.0,                        # large offset (ETT-like)
                  np.cumsum(rng.standard_normal(n)) * 1e-2 + 1.0,              # drifting
                  rng.standard_normal(n) * 1e-3], 1)                          # tiny scale
    w = jax.lax.scan(lambda w, xi: (alg.welford_update(w, xi), None), alg.welford_init((4,)),
                     jnp.asarray(x, jnp.float32))[0]
    m64, v64 = x.mean(0), x.var(0, ddof=1)
    rel_mean = np.abs(np.asarray(w.mean) - m64) / np.maximum(np.abs(m64), 1e-12)
    rel_var = np.abs(np.asarray(alg.welford_var(w)) - v64) / v64
    # the same without compensation, to show what the Kahan terms buy
    def plain(wp, xi):
        m, p, c = wp
        c = c + 1
        d = xi - m
        m2 = m + d / c
        return (m2, p + d * (xi - m2), c), None
    (pm, pp, pc), _ = jax.lax.scan(plain, (jnp.asarray(x[0], jnp.float32), jnp.zeros(4), jnp.float32(1.0)),
                                   jnp.asarray(x[1:], jnp.float32))
    rel_var_plain = np.abs(np.asarray(pp / (pc - 1)) - v64) / v64
    return dict(n=n, rel_err_mean=rel_mean.tolist(), rel_err_var=rel_var.tolist(),
                rel_err_var_uncompensated=rel_var_plain.tolist(), tol=1e-3,
                **{"pass": bool(rel_var.max() < 1e-3 and rel_mean.max() < 1e-3)})


# ------------------------------------------------------------------ G1.6 SparseInit exact counts
def sparse_init_exact() -> dict:
    out, ok = {}, True
    nets = {"q": (mdl.QNetMinAtar(3), (10, 10, 4)), "actor": (mdl.GaussianActor(3), (12,)),
            "critic": (mdl.ValueNet(), (12,))}
    for name, (net, shp) in nets.items():
        p = mdl.init_params(jax.random.PRNGKey(1), mdl.template(net, shp), mdl.INIT_SPARSE)
        for path, layer in mdl._layer_items(p):
            k = np.asarray(layer["kernel"])
            fan_in = int(np.prod(k.shape[:-1]))
            kk = k.reshape(fan_in, -1)
            nz = (kk != 0).sum(0)
            want = fan_in - int(np.ceil(0.9 * fan_in))
            b = 1 / np.sqrt(fan_in)
            good = bool(np.all(nz == want)) and bool(np.abs(kk).max() <= b + 1e-7) and \
                bool(np.all(np.asarray(layer["bias"]) == 0))
            out[f"{name}/{'/'.join(path)}"] = dict(fan_in=fan_in, nonzeros_per_unit=sorted(set(nz.tolist())),
                                                   expected=want, max_abs=float(np.abs(kk).max()), bound=b, ok=good)
            ok &= good
    return dict(layers=out, **{"pass": ok})


# ------------------------------------------------------------------ G1.7 trace identity (frozen weights)
def trace_identity(T: int = 12) -> dict:
    with enable_x64():
        net = mdl.QNetMinAtar(3)
        p = mdl.init_params(jax.random.PRNGKey(2), mdl.template(net, (10, 10, 4)), mdl.INIT_SPARSE)
        p = jax.tree.map(lambda x: x.astype(jnp.float64), p)
        xs = jax.random.normal(jax.random.PRNGKey(3), (T, 10, 10, 4), jnp.float64)
        acts = jax.random.randint(jax.random.PRNGKey(4), (T,), 0, 3)
        gl = 0.99 * 0.8
        grads = [jax.grad(lambda pp: net.apply({"params": pp}, xs[t], 1.0)[acts[t]])(p) for t in range(T)]
        z = alg.tree_zeros(p)
        for g in grads:
            z = alg.trace_accumulate(z, g, gl)
        closed = alg.tree_zeros(p)
        for t, g in enumerate(grads):
            closed = alg.tree_axpy(gl ** (T - 1 - t), g, closed)
        err = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree.leaves(z), jax.tree.leaves(closed)))
    return dict(T=T, max_abs_err=err, **{"pass": err < 1e-10})


# ------------------------------------------------------------------ G1.8 Watkins cut (both variants)
def watkins_cut() -> dict:
    from .agents import QAgent
    from .data import Env
    from . import train as tr
    agent = QAgent(Env(None, None, (10, 10, 4), 3, True, "toy"), 1000)
    res = {}
    x64 = enable_x64()
    x64.__enter__()
    xs = jax.random.normal(jax.random.PRNGKey(5), (3, 10, 10, 4), jnp.float64)
    for variant, wp in (("official", 0), ("paper", 1)):
        lane = jax.tree.map(lambda x: x[0], tr.stack_lanes("q", [dict(tr.lane_cfg("q", "stream_q"), watkins_paper=wp)]))
        p = mdl.init_params(jax.random.PRNGKey(6), agent.tmpl, mdl.INIT_SPARSE)
        p = jax.tree.map(lambda a: a.astype(jnp.float64), p)
        z = alg.tree_zeros(p)
        ad = alg.adam_init(p)
        # step 0 greedy, step 1 non-greedy
        p1, z1, ad, _, _, _, _, _ = agent.learn(p, z, ad, p, xs[0], 0, jnp.float32(0.3), xs[1], jnp.asarray(False),
                                                jnp.asarray(False), lane, jnp.int32(0))
        g1 = jax.grad(lambda pp: agent.q(pp, xs[1], 1.0)[2])(p1)
        p2, z2, _, _, d2, info, _, _ = agent.learn(p1, z1, ad, p1, xs[1], 2, jnp.float32(-0.2), xs[2],
                                                   jnp.asarray(False), jnp.asarray(True), lane, jnp.int32(1))
        used = jax.tree.map(lambda a, b: (a - b) / (info.step * d2), p2, p1)   # the trace the update used
        z_after_l1 = float(alg.tree_l1(z2))
        if variant == "official":
            gl = 0.99 * lane.lam   # lane values are float32 (0.8 -> 0.800000012)
            want_used = jax.tree.map(lambda a, b: gl * a + b, z1, g1)
            want_after = 0.0
        else:
            want_used = g1
            want_after = float(alg.tree_l1(g1))
        err = max(float(jnp.max(jnp.abs(a - b)) / (1 + float(jnp.max(jnp.abs(b)))))
                  for a, b in zip(jax.tree.leaves(used), jax.tree.leaves(want_used)))
        res[variant] = dict(used_trace_rel_err=err, trace_l1_after=z_after_l1, expected_l1_after=want_after,
                            ok=bool(err < 1e-9 and abs(z_after_l1 - want_after) <= 1e-9 * max(1, want_after)))
    x64.__exit__(None, None, None)
    return dict(**res, **{"pass": res["official"]["ok"] and res["paper"]["ok"]})


# ------------------------------------------------------------------ G1.9 TD(lambda) on a stochastic chain
def td_chain(n_ep: int = 20000, n_seeds: int = 32, p: float = 0.5, gamma: float = 0.9) -> dict:
    """s0 --(p: r=0)--> s1 --(r=1)--> T ;  s0 --(1-p: r=1)--> T.  Tabular features. v1 = 1, v0 = p*gamma + (1-p).
    All |delta| < 1 near the fixed point, so ObGD's step is a deterministic function of the path (no noise-
    dependent weighting) and the iterate average converges to the TD fixed point = v_pi."""
    v_true = np.array([p * gamma + (1 - p), 1.0])
    lam = 0.8
    x = jnp.eye(2)

    def episode(carry, key, mode):
        w, = carry
        go = jax.random.uniform(key) < p
        z = x[0]
        d0 = jnp.where(go, gamma * w[1], 1.0) - w[0]
        if mode == "obgd":
            w, _ = alg.obgd_apply(w, z, d0, 1.0, 2.0)
        else:
            w = w + 0.01 * d0 * z
        z = gamma * lam * z + x[1]
        d1 = 1.0 - w[1]
        if mode == "obgd":
            w2, _ = alg.obgd_apply(w, z, d1, 1.0, 2.0)
        else:
            w2 = w + 0.01 * d1 * z
        w = jnp.where(go, w2, w)
        return (w,), w

    out = {}
    for mode in ("td_alpha0.01", "obgd"):
        m = "obgd" if mode == "obgd" else "td"

        def run(key):
            _, ws = jax.lax.scan(lambda c, k: episode(c, k, m), (jnp.zeros(2),), jax.random.split(key, n_ep))
            return ws[n_ep // 2:].mean(0)

        avg = np.asarray(jax.vmap(run)(jax.random.split(jax.random.PRNGKey(7), n_seeds)))
        mean, se = avg.mean(0), avg.std(0, ddof=1) / np.sqrt(n_seeds)
        z = np.abs(mean - v_true) / np.maximum(se, 1e-12)
        out[mode] = dict(mean=mean.tolist(), se=se.tolist(), v_true=v_true.tolist(), z=z.tolist(),
                         ok=bool(np.all((z < 4) | (np.abs(mean - v_true) < 2e-3))))
    return dict(**out, **{"pass": all(v["ok"] for v in out.values())})


ALL = dict(obgd_toy=obgd_toy, obgd_l1_property=obgd_l1_property, welford_toy=welford_toy,
           welford_precision=welford_precision, sparse_init_exact=sparse_init_exact, trace_identity=trace_identity,
           watkins_cut=watkins_cut, td_chain=td_chain)

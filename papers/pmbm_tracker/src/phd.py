"""GM-PHD (Vo & Ma 2006) and GM-CPHD (Vo, Vo, Cantoni 2007) baselines, fixed-shape JAX (jit / vmap over runs / lax.scan over steps).

Semantics follow the official code (github.com/Agarciafernandez/MTT "PHD filter/": GMPHD_filter.m, GMPHD_filter_update.m,
GMPHD_filter_pruning.m, GMPHD_filter_prediction.m, GMPHD_estimation2.m, GMCPHD_filter.m, GMCPHD_filter_update.m,
GMCPHD_estimation.m, TPHD filter/CPHD_cardinality_pred.m, Compute_esf.m)

Per step (all in the official order): birth component appended to the first free slot (weight birth_rate; at the first step
weight lambda0 if cfg['lambda0'] is set -- official code, sensitivity row -- else birth_rate, i.e. the paper's "predict of the
empty set then update") -> update with ALL measurements (no gating) -> estimate (after update, before pruning) ->
prune (w > prune), greedy max-weight merge (Mahalanobis of the candidate's own covariance < merge), cap `max_comp` ->
predict (w *= p_s, KF predict).

Fixed shapes: C = max_comp + 1 slots (the extra one holds the birth component), N = C (1 + M) updated components. Padding is
carried by zero weights; measurement padding by zvalid. The CPHD elementary symmetric functions and upsilon terms are evaluated in
the log domain (log-esf recursion with logaddexp).

Entry points (single run; vmap over runs):
  run_gmphd(z, zvalid, cfg, pd=None, lambda_c=None, E=16, return_info=False) -> (est_mask bool[K,E], est_x f64[K,E,4])
  run_gmcphd(...) same
  make_phd_batch(cfg, kind, E=16) -> jitted f(z[R,K,M,2], zvalid[R,K,M], pd, lambda_c) -> (est_mask[R,K,E], est_x[R,K,E,4], info)
  ref_run_phd(...) : independent dense float64 numpy reference (unit-test cross-check only).
Component cap: cfg['phd']['max_comp'] = 30 as in the official code, INCLUDING its off-by-one (at most 29 components survive a capped
prune, so 30 enter the update with the birth component); cfg['phd_cap_offbyone'] = False gives a true 30-component cap.
cfg['phd_estimator'] = 'est1' selects the original Vo-Ma extraction (official GMPHD_estimation.m: component i reported round(w_i) times)
instead of the default GMPHD_estimation2; diagnostic for the Table II PHD column, see the smoke script.
pd / lambda_c default to cfg values and may be traced (so one compile serves all 15 Table II settings).
Estimates larger than E are truncated to E (info['n_hat'] keeps the raw count).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import logsumexp as _phd_lse
from scipy.special import gammaln as _phd_gammaln_np

jax.config.update("jax_enable_x64", True)

_PHD_NEG = -jnp.inf


# ------------------------------------------------------------------------------------------------ helpers
def _phd_consts(cfg, kind):
    s = cfg["phd"] if kind == "phd" else cfg["cphd"]
    c = dict(
        F=jnp.asarray(cfg["F"], jnp.float64), Q=jnp.asarray(cfg["Q"], jnp.float64), H=jnp.asarray(cfg["H"], jnp.float64),
        R=jnp.asarray(cfg["R"], jnp.float64), bm=jnp.asarray(cfg["birth_mean"], jnp.float64),
        bP=jnp.asarray(cfg["birth_cov"], jnp.float64), bw=float(cfg["birth_rate"]), ps=float(cfg["p_s"]),
        area=float(cfg["area"]), lambda0=cfg.get("lambda0"), prune=float(s["prune"]), merge=float(s["merge"]),
        Cmax=int(s["max_comp"]), offbyone=bool(cfg.get("phd_cap_offbyone", True)),
    )
    if kind == "cphd":
        c["Nmax"] = int(s["n_card_max"])
    return c


def _phd_inv2(S):
    """Batched 2x2 inverse and log-determinant."""
    a, b, c, d = S[..., 0, 0], S[..., 0, 1], S[..., 1, 0], S[..., 1, 1]
    det = a * d - b * c
    inv = jnp.stack([jnp.stack([d, -b], -1), jnp.stack([-c, a], -1)], -2) / det[..., None, None]
    return inv, jnp.log(det)


def _phd_update_parts(w, m, P, z, zv, c):
    """KF quantities for all slots against all measurements (official GMPHD_filter_update.m / GMCPHD_filter_update.m)."""
    H, R = c["H"], c["R"]
    S = jnp.einsum("ij,cjk,lk->cil", H, P, H) + R
    S = 0.5 * (S + jnp.swapaxes(S, -1, -2))
    Sinv, logdet = _phd_inv2(S)
    Kg = jnp.einsum("cij,kj,ckl->cil", P, H, Sinv)
    Pu = P - jnp.einsum("cij,jk,ckl->cil", Kg, H, P)
    Pu = 0.5 * (Pu + jnp.swapaxes(Pu, -1, -2))
    zp = m @ H.T
    dz = z[None, :, :] - zp[:, None, :]  # [C,M,2]
    maha = jnp.einsum("cmi,cij,cmj->cm", dz, Sinv, dz)
    g = jnp.exp(-0.5 * maha - 0.5 * logdet[:, None] - jnp.log(2.0 * jnp.pi))  # N(z; Hm, S), nz = 2
    g = jnp.where(zv[None, :], g, 0.0)
    mdet = m[:, None, :] + jnp.einsum("cij,cmj->cmi", Kg, dz)  # [C,M,4]
    return g, mdet, Pu


def _phd_flatten(w_miss, w_det, m, mdet, P, Pu):
    """Official layout: missed components first (slot order), then measurement-major detections."""
    C, M = w_det.shape
    wU = jnp.concatenate([w_miss, w_det.T.reshape(-1)])
    mU = jnp.concatenate([m, jnp.transpose(mdet, (1, 0, 2)).reshape(M * C, -1)])
    PU = jnp.concatenate([P, jnp.broadcast_to(Pu[None], (M,) + Pu.shape).reshape(M * C, *Pu.shape[1:])])
    return wU, mU, PU


def _phd_prune_merge(wU, mU, PU, c, Pinv_base):
    """GMPHD_filter_pruning.m: greedy max-weight merging, Mahalanobis (candidate's own covariance) < merge, <= Cmax outputs."""
    Cmax, prune, merge = c["Cmax"], c["prune"], c["merge"]
    PinvU = Pinv_base
    cand0 = wU > prune

    def body(cand, _):
        i = jnp.argmax(jnp.where(cand, wU, -jnp.inf))
        anyc = cand[i]
        dm = mU - mU[i][None, :]
        d = jnp.einsum("ni,nij,nj->n", dm, PinvU, dm)
        sel = cand & (d < merge) & anyc
        ws = jnp.where(sel, wU, 0.0)
        W = ws.sum()
        Ws = jnp.where(anyc, W, 1.0)
        mo = (ws[:, None] * mU).sum(0) / Ws
        Po = (ws[:, None, None] * (PU + mU[:, :, None] * mU[:, None, :])).sum(0) / Ws - mo[:, None] * mo[None, :]
        Po = 0.5 * (Po + Po.T)
        mo = jnp.where(anyc, mo, c["bm"])
        Po = jnp.where(anyc, Po, c["bP"])
        return cand & ~sel, (W, mo, Po)

    cand_left, (Wo, mo, Po) = lax.scan(body, cand0, None, length=Cmax)
    n_trunc = jnp.sum(cand_left) + (Wo[Cmax - 1] > 0)
    if c["offbyone"]:
        # OFFICIAL off-by-one: the Octave loop breaks at index_o == Ncom_max BEFORE incrementing, and Ncom_o = index_o - 1, so when the
        # cap is reached the 30th merged component is written but never marked active -> at most Ncom_max - 1 = 29 survive.
        Wo = Wo.at[Cmax - 1].set(0.0)
    return Wo, mo, Po, n_trunc


def _phd_card_tables(Nmax, ps, bw):
    n = np.arange(Nmax + 1)
    B = np.zeros((Nmax + 1, Nmax + 1))  # B[j, p] = C(p,j) ps^j (1-ps)^(p-j)
    for j in range(Nmax + 1):
        for p in range(j, Nmax + 1):
            B[j, p] = np.exp(_phd_gammaln_np(p + 1) - _phd_gammaln_np(j + 1) - _phd_gammaln_np(p - j + 1) + j * np.log(ps) + (p - j) * np.log1p(-ps))
    birth = np.exp(-bw + n * np.log(bw) - _phd_gammaln_np(n + 1))  # Poisson(sum birth weights), truncated at Nmax (not renormalised)
    conv = np.zeros((Nmax + 1, Nmax + 1))  # pred[n] = sum_i surv[i] birth[n-i]
    for nn in range(Nmax + 1):
        for i in range(nn + 1):
            conv[nn, i] = birth[nn - i]
    lf0 = np.full((Nmax + 1, Nmax + 1), -np.inf)  # log n!/(n-j)!  (j <= n)
    lf1 = np.full((Nmax + 1, Nmax + 1), -np.inf)  # log n!/(n-j-1)! (j <= n-1)
    for nn in range(Nmax + 1):
        for j in range(Nmax + 1):
            if j <= nn:
                lf0[nn, j] = _phd_gammaln_np(nn + 1) - _phd_gammaln_np(nn - j + 1)
            if j <= nn - 1:
                lf1[nn, j] = _phd_gammaln_np(nn + 1) - _phd_gammaln_np(nn - j)
    return jnp.asarray(B), jnp.asarray(conv), jnp.asarray(lf0), jnp.asarray(lf1)


def _phd_log_esf(logL, Nmax):
    """log of the elementary symmetric functions e_0..e_Nmax of exp(logL) (log-domain recursion; -inf entries are inert)."""
    e0 = jnp.full((Nmax + 1,), -jnp.inf).at[0].set(0.0)

    def step(e, l):
        sh = jnp.concatenate([jnp.full((1,), -jnp.inf), e[:-1]])
        return jnp.logaddexp(e, sh + l), None

    e, _ = lax.scan(step, e0, logL)
    return e


def _phd_cphd_weights(w, g, zv, pd, lam, card_pred, c, tabs):
    """GMCPHD_filter_update.m weights and cardinality update, log domain. Returns (w_miss[C], w_det[C,M], card_u)."""
    Nmax, area = c["Nmax"], c["area"]
    _, _, lf0, lf1 = tabs
    jj = jnp.arange(Nmax + 1)
    nn = jnp.arange(Nmax + 1)
    W = w.sum()
    Mv = zv.sum().astype(jnp.float64)
    Lam = pd * (w[:, None] * g).sum(0) * area  # [M]
    logL = jnp.where(zv & (Lam > 0), jnp.log(jnp.where(Lam > 0, Lam, 1.0)), -jnp.inf)
    esf = _phd_log_esf(logL, Nmax)  # [Nmax+1]
    M = zv.shape[0]
    logLm = jnp.where(jnp.eye(M, dtype=bool), -jnp.inf, logL[None, :])
    esf_m = jax.vmap(_phd_log_esf, in_axes=(0, None))(logLm, Nmax)  # [M, Nmax+1]
    loglam = jnp.log(lam)
    l1p = jnp.log1p(-pd)
    logW = jnp.log(W)
    nj = nn[:, None] - jj[None, :]
    t0 = -lam + (Mv - jj)[None, :] * loglam + lf0 + nj * l1p - jj[None, :] * logW + esf[None, :]
    base1 = -lam + (Mv - jj)[None, :] * loglam + lf1 + (nj - 1) * l1p - (jj[None, :] + 1) * logW
    lu0 = _phd_lse(t0, axis=1)
    lu1 = _phd_lse(base1 + esf[None, :], axis=1)
    base1m = base1 - loglam  # one fewer measurement: (M-1-j) log(lambda)
    lu1m = _phd_lse(base1m[None, :, :] + esf_m[:, None, :], axis=2)  # [M, n]
    logcard = jnp.log(card_pred)
    den = _phd_lse(lu0 + logcard)
    card_u = jnp.exp(lu0 + logcard - den)
    f_miss = jnp.exp(_phd_lse(lu1 + logcard) - den)
    f_det = jnp.exp(_phd_lse(lu1m + logcard[None, :], axis=1) - den)  # [M]
    f_det = jnp.where(zv, f_det, 0.0)
    w_miss = f_miss * (1.0 - pd) * w
    w_det = f_det[None, :] * pd * g * area * w[:, None]
    return w_miss, w_det, card_u


# ------------------------------------------------------------------------------------------------ the filter
def _phd_run(z, zvalid, cfg, pd, lam, E, kind, return_info):
    c = _phd_consts(cfg, kind)
    K, M = z.shape[0], z.shape[1]
    C = c["Cmax"] + 1
    pd = jnp.asarray(cfg["pd"] if pd is None else pd, jnp.float64)
    lam = jnp.asarray(cfg["lambda_c"] if lam is None else lam, jnp.float64)
    kappa = lam / c["area"]  # clutter intensity lambda_c * (1/|region|)
    F, Q, bm, bP, ps = c["F"], c["Q"], c["bm"], c["bP"], c["ps"]
    if kind == "cphd":
        Nmax = c["Nmax"]
        tabs = _phd_card_tables(Nmax, ps, c["bw"])
        Bmat, conv = tabs[0], tabs[1]
        card0 = jnp.zeros(Nmax + 1).at[0].set(1.0)
        n_ = np.arange(Nmax + 1)
        if c["lambda0"] is not None:
            l0 = float(c["lambda0"])
            card_l0 = jnp.asarray(np.exp(-l0 + n_ * np.log(l0) - _phd_gammaln_np(n_ + 1)))
    init = (jnp.zeros(C), jnp.broadcast_to(bm, (C, 4)) + 0.0, jnp.broadcast_to(bP, (C, 4, 4)) + 0.0, jnp.asarray(0, jnp.int32),
            card0 if kind == "cphd" else jnp.zeros(1))

    def step(carry, inp):
        w, m, P, n_act, card = carry
        zk, zvk, k = inp
        first = k == 0
        wb = c["bw"] if c["lambda0"] is None else jnp.where(first, float(c["lambda0"]), c["bw"])
        w = w.at[n_act].set(wb)
        m = m.at[n_act].set(bm)
        P = P.at[n_act].set(bP)
        g, mdet, Pu = _phd_update_parts(w, m, P, zk, zvk, c)
        if kind == "phd":
            wdet = pd * w[:, None] * g
            wdet = wdet / (kappa + wdet.sum(0))[None, :]
            w_miss = (1.0 - pd) * w
            card_u = card
        else:
            surv = Bmat @ card
            card_pred = conv @ surv
            card_pred = card_pred / card_pred.sum()
            if c["lambda0"] is not None:
                card_pred = jnp.where(first, card_l0, card_pred)
            w_miss, wdet, card_u = _phd_cphd_weights(w, g, zvk, pd, lam, card_pred, c, tabs)
        wU, mU, PU = _phd_flatten(w_miss, wdet, m, mdet, P, Pu)
        # estimate (after update, before pruning)
        decrement = kind == "phd"  # GMPHD_estimation2: weight -= 1 per pick; GMCPHD_estimation: weight := 0 per pick
        if decrement:
            n_raw = jnp.floor(wU.sum() + 0.5).astype(jnp.int32)  # MATLAB round() on a non-negative number
        else:
            n_raw = jnp.argmax(card_u).astype(jnp.int32)  # MAP cardinality (first max)

        def pick(wc, _):
            j = jnp.argmax(wc)
            return wc.at[j].set(wc[j] - 1.0 if decrement else 0.0), mU[j]

        _, ex = lax.scan(pick, wU, None, length=E)
        if kind == "phd" and cfg.get("phd_estimator", "est2") == "est1":
            # DIAGNOSTIC alternative, official GMPHD_estimation.m (Vo-Ma 2006): component i is reported round(w_i) times, pass-major
            cnt = jnp.floor(wU + 0.5).astype(jnp.int32)
            cand = jnp.arange(3)[:, None] < cnt[None, :]  # [3, N] (copies beyond 3 are ignored)
            flat = cand.reshape(-1)
            order = jnp.argsort(~flat, stable=True)[:E]
            ex = mU[order % wU.shape[0]]
            n_raw = flat.sum().astype(jnp.int32)
        n_hat = jnp.clip(n_raw, 0, E)
        emask = jnp.arange(E) < n_hat
        ex = jnp.where(emask[:, None], ex, 0.0)
        # prune / merge / cap
        Pinv_b = jnp.linalg.inv(P)
        Pinv_u = jnp.linalg.inv(Pu)
        Pinv = jnp.concatenate([Pinv_b, jnp.broadcast_to(Pinv_u[None], (M,) + Pinv_u.shape).reshape(M * C, 4, 4)])
        Wo, mo, Po, n_trunc = _phd_prune_merge(wU, mU, PU, c, Pinv)
        n_new = jnp.sum(Wo > 0).astype(jnp.int32)
        w2 = jnp.concatenate([Wo, jnp.zeros(1)])
        m2 = jnp.concatenate([mo, bm[None]])
        P2 = jnp.concatenate([Po, bP[None]])
        # predict
        w3 = ps * w2
        m3 = m2 @ F.T
        P3 = jnp.einsum("ij,cjk,lk->cil", F, P2, F) + Q
        out = (emask, ex, dict(n_hat=n_raw, sum_w=wU.sum(), n_active=n_new, n_trunc=n_trunc))
        return (w3, m3, P3, n_new, card_u), out

    _, (em, ex, info) = lax.scan(step, init, (z, zvalid, jnp.arange(K)))
    return (em, ex, info) if return_info else (em, ex)


def run_gmphd(z, zvalid, cfg, pd=None, lambda_c=None, E=16, return_info=False):
    """GM-PHD, one run. z f64[K,M,2], zvalid bool[K,M] -> (est_mask bool[K,E], est_x f64[K,E,4]) [+ info dict of [K] arrays]."""
    return _phd_run(z, zvalid, cfg, pd, lambda_c, E, "phd", return_info)


def run_gmcphd(z, zvalid, cfg, pd=None, lambda_c=None, E=16, return_info=False):
    """GM-CPHD (N_max = cfg['cphd']['n_card_max']), one run. E should be >= N_max (10)."""
    return _phd_run(z, zvalid, cfg, pd, lambda_c, E, "cphd", return_info)


def make_phd_batch(cfg, kind="phd", E=16):
    """jit(vmap over runs) of the filter; pd and lambda_c are traced arguments (one compile for all Table II settings)."""
    f = run_gmphd if kind == "phd" else run_gmcphd

    def one(z, zv, pd, lam):
        return f(z, zv, cfg, pd, lam, E, True)

    return jax.jit(jax.vmap(one, in_axes=(0, 0, None, None)))


# ------------------------------------------------------------------- independent numpy reference (tests only)
def _ref_mvn(x, mu, S):
    d = x - mu
    return float(np.exp(-0.5 * d @ np.linalg.solve(S, d)) / (2.0 * np.pi * np.sqrt(np.linalg.det(S))))


def _ref_esf(zs):
    """All elementary symmetric functions of the list zs (Compute_esf.m recursion), normal domain."""
    e = np.zeros(len(zs) + 1)
    e[0] = 1.0
    for i, x in enumerate(zs):
        new = e.copy()
        for j in range(1, i + 2):
            new[j] = e[j] + x * e[j - 1]
        e = new
    return e


def ref_run_phd(z_list, cfg, kind="phd", pd=None, lambda_c=None, trace=None):
    """Dense loop float64 implementation, written from the official Octave semantics with dynamic component lists.
    z_list: list of K arrays [m_k, 2]. Returns list of K arrays [n_k, 4] (estimates) -- unbounded size (no E cap)."""
    s = cfg["phd"] if kind == "phd" else cfg["cphd"]
    pd = cfg["pd"] if pd is None else pd
    lam = cfg["lambda_c"] if lambda_c is None else lambda_c
    F, Q, H, R = (np.asarray(cfg[k], float) for k in ("F", "Q", "H", "R"))
    bw, bm, bP = float(cfg["birth_rate"]), np.asarray(cfg["birth_mean"], float), np.asarray(cfg["birth_cov"], float)
    ps, area, lambda0 = float(cfg["p_s"]), float(cfg["area"]), cfg.get("lambda0")
    prune, merge, Cmax = s["prune"], s["merge"], s["max_comp"]
    comps = []  # list of (w, m, P)
    out = []
    if kind == "cphd":
        Nmax = s["n_card_max"]
        card = np.zeros(Nmax + 1)
        card[0] = 1.0
        fact = np.array([np.prod(np.arange(1, i + 1)) for i in range(Nmax + 1)], float)
        birth_card = np.exp(-bw) * bw ** np.arange(Nmax + 1) / fact
    for k, z in enumerate(z_list):
        z = np.asarray(z, float).reshape(-1, 2)
        w0 = lambda0 if (k == 0 and lambda0 is not None) else bw
        comps = comps + [(w0, bm.copy(), bP.copy())]
        Nc, Mz = len(comps), len(z)
        pre = []
        for (w, m, P) in comps:
            S = H @ P @ H.T + R
            S = (S + S.T) / 2
            Kg = P @ H.T @ np.linalg.inv(S)
            Pu = (np.eye(4) - Kg @ H) @ P
            pre.append((H @ m, S, Kg, (Pu + Pu.T) / 2))
        gz = np.array([[_ref_mvn(z[p], pre[i][0], pre[i][1]) for p in range(Mz)] for i in range(Nc)]).reshape(Nc, Mz)
        upd = []
        if kind == "phd":
            for i, (w, m, P) in enumerate(comps):
                upd.append(((1 - pd) * w, m, P))
            for p in range(Mz):
                tmp = []
                for i, (w, m, P) in enumerate(comps):
                    tmp.append((pd * w * gz[i, p], m + pre[i][2] @ (z[p] - pre[i][0]), pre[i][3]))
                den = lam / area + sum(t[0] for t in tmp)
                upd += [(t[0] / den, t[1], t[2]) for t in tmp]
            card_u = None
        else:
            wp = np.array([c_[0] for c_ in comps])
            Wsum = wp.sum()
            if k == 0 and lambda0 is not None:
                card_pred = np.exp(-lambda0) * lambda0 ** np.arange(Nmax + 1) / fact
            else:
                surv = np.zeros(Nmax + 1)
                for j in range(Nmax + 1):
                    for p_ in range(j, Nmax + 1):
                        surv[j] += (fact[p_] / (fact[j] * fact[p_ - j])) * ps ** j * (1 - ps) ** (p_ - j) * card[p_]
                card_pred = np.array([sum(surv[i] * birth_card[n - i] for i in range(n + 1)) for n in range(Nmax + 1)])
                card_pred = card_pred / card_pred.sum()
            Lam = [pd * float(wp @ gz[:, p]) * area for p in range(Mz)]
            esf_all = _ref_esf(Lam)
            esf_m1 = [_ref_esf(Lam[:i] + Lam[i + 1:]) for i in range(Mz)]
            u0 = np.zeros(Nmax + 1)
            u1 = np.zeros(Nmax + 1)
            u1m = np.zeros((Nmax + 1, Mz))
            for n in range(Nmax + 1):
                for j in range(0, min(Mz, n) + 1):
                    u0[n] += np.exp(-lam + (Mz - j) * np.log(lam) + sum(np.log(np.arange(n - j + 1, n + 1))) + (n - j) * np.log(1 - pd) - j * np.log(Wsum)) * esf_all[j]
                for j in range(0, min(Mz, n - 1) + 1):
                    u1[n] += np.exp(-lam + (Mz - j) * np.log(lam) + sum(np.log(np.arange(n - j, n + 1))) + (n - (j + 1)) * np.log(1 - pd) - (j + 1) * np.log(Wsum)) * esf_all[j]
                for i in range(Mz):
                    for j in range(0, min(Mz - 1, n - 1) + 1):
                        u1m[n, i] += np.exp(-lam + ((Mz - 1) - j) * np.log(lam) + sum(np.log(np.arange(n - j, n + 1))) + (n - (j + 1)) * np.log(1 - pd) - (j + 1) * np.log(Wsum)) * esf_m1[i][j]
            card_u = u0 * card_pred
            card_u = card_u / card_u.sum()
            fm = (u1 @ card_pred) / (u0 @ card_pred)
            for i, (w, m, P) in enumerate(comps):
                upd.append((fm * (1 - pd) * w, m, P))
            for p in range(Mz):
                fd = (u1m[:, p] @ card_pred) / (u0 @ card_pred)
                for i, (w, m, P) in enumerate(comps):
                    upd.append((fd * pd * gz[i, p] * area * w, m + pre[i][2] @ (z[p] - pre[i][0]), pre[i][3]))
        # estimate
        wu = np.array([u[0] for u in upd])
        mu = np.array([u[1] for u in upd])
        if kind == "phd" and cfg.get("phd_estimator", "est2") == "est1":  # GMPHD_estimation.m
            det = np.floor(wu + 0.5).astype(int)
            est = []
            while det.sum() > 0:
                est += [mu[i] for i in np.where(det > 0)[0]]
                det = det - (det > 0)
            n_est = 0
            wc = wu.copy()
        elif kind == "phd":
            n_est = int(np.floor(wu.sum() + 0.5))
            wc = wu.copy()
            est = []
            for _ in range(n_est):
                j = int(np.argmax(wc))
                wc[j] -= 1.0
                est.append(mu[j])
        else:
            n_est = int(np.argmax(card_u))
            wc = wu.copy()
            est = []
            for _ in range(n_est):
                j = int(np.argmax(wc))
                est.append(mu[j])
                wc[j] = 0.0
        out.append(np.array(est).reshape(-1, 4))
        # prune / merge
        cand = [i for i in range(len(upd)) if upd[i][0] > prune]
        new = []
        while cand:
            jmax = max(cand, key=lambda i_: (upd[i_][0], -i_))  # first index among equal weights
            mm = upd[jmax][1]
            Wt, mt, Pt = 0.0, np.zeros(4), np.zeros((4, 4))
            rest = []
            for i in cand:
                dm = upd[i][1] - mm
                if dm @ np.linalg.solve(upd[i][2], dm) < merge:
                    Wt += upd[i][0]
                    mt += upd[i][0] * upd[i][1]
                    Pt += upd[i][0] * (upd[i][2] + np.outer(upd[i][1], upd[i][1]))
                else:
                    rest.append(i)
            mt = mt / Wt
            Pt = Pt / Wt - np.outer(mt, mt)
            new.append((Wt, mt, (Pt + Pt.T) / 2))
            cand = rest
            if len(new) == Cmax:
                if cfg.get("phd_cap_offbyone", True):
                    new = new[:Cmax - 1]  # official off-by-one (see _phd_prune_merge)
                break
        if trace is not None:
            trace.append((np.array([u[0] for u in upd]), new))
        comps = [(ps * w, F @ m, F @ P @ F.T + Q) for (w, m, P) in new]
        if kind == "cphd":
            card = card_u
    return out

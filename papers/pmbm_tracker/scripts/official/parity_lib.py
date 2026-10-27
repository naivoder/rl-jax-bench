"""Differential-test layer: canonical forms of the AUTHORS' PMBM state (from drive_pmbm.m dumps, GNU Octave) and of OUR
state (src/algorithm.py), plus comparison helpers.  Own code.

Canonical state = (poisson, hyps)
    poisson : (w[P], m[P,4], C[P,4,4]) sorted by (w desc, m0)
    hyps    : list of (weight, bern) sorted by weight desc;  bern = list of (r, m[4], C[4,4]) sorted by (m0, m2)
The authors' table indices (track i / single-target hypothesis j) and our slot indices are labels with no meaning, so
only label-free content is compared.
"""
from __future__ import annotations

import numpy as np
from scipy.io import loadmat

# ------------------------------------------------------------------------------------------------ official side


def load_run(out_dir, r):
    d = loadmat(f"{out_dir}/run_{r:03d}.mat", squeeze_me=False)
    return {k: v for k, v in d.items() if not k.startswith("__")}


def _a(d, key):
    return np.asarray(d[key])


def _poisson_canon(w, m, C):
    w = np.asarray(w, float).reshape(-1)
    if w.size == 0:
        return np.zeros(0), np.zeros((0, 4)), np.zeros((0, 4, 4))
    m = np.asarray(m, float).reshape(4, -1).T
    C = np.asarray(C, float).reshape(4, 4, -1)
    C = np.transpose(C, (2, 0, 1))
    o = np.lexsort((m[:, 2], m[:, 0], -w))
    return w[o], m[o], C[o]


def _bern_sort(bern):
    return sorted(bern, key=lambda t: (round(float(t[1][0]), 9), round(float(t[1][2]), 9)))


def official_state(d, pre):
    """pre in {'rk003', 'uk003', 'pk003'} (predicted / updated-before-pruning / pruned)."""
    pw = _a(d, pre + "_poisW")
    pm = _a(d, pre + "_poisM")
    pC = _a(d, pre + "_poisP")
    pois = _poisson_canon(pw, pm, pC)
    htrk = _a(d, pre + "_h_trk").reshape(-1).astype(int)
    hj = _a(d, pre + "_h_j").reshape(-1).astype(int)
    heB = _a(d, pre + "_h_eB").reshape(-1)
    hval = _a(d, pre + "_h_valid").reshape(-1)
    hmean = _a(d, pre + "_h_mean").reshape(4, -1)
    hcov = _a(d, pre + "_h_cov").reshape(4, 4, -1)
    gh = _a(d, pre + "_globHyp")
    gw = _a(d, pre + "_globW").reshape(-1)
    if gh.size == 0:
        return pois, [(1.0, [])]   # the authors' code stores NO hypothesis for the empty (no-track) state; weight 1
    lookup = {(int(t), int(j)): q for q, (t, j) in enumerate(zip(htrk, hj))}
    hyps = []
    for g in range(gh.shape[0]):
        bern = []
        for i in range(gh.shape[1]):
            j = int(gh[g, i])
            if j > 0:
                q = lookup[(i + 1, j)]
                bern.append((float(heB[q]), hmean[:, q].copy(), hcov[:, :, q].copy()))
        hyps.append((float(gw[g]), _bern_sort(bern)))
    hyps.sort(key=lambda t: -t[0])
    return pois, hyps


def official_estimates(d, e, k):
    """estimate e (1..3) at 0-based step k as an array [n,4] sorted by (x, y)."""
    A = _a(d, f"est{e}_x")  # [4, Emax, K]
    n = int(_a(d, f"est{e}_n").reshape(-1)[k])
    X = A[:, :n, k].T if n > 0 else np.zeros((0, 4))
    return X[np.lexsort((X[:, 2], X[:, 0]))] if n else X


# ------------------------------------------------------------------------------------------------ our side


def our_state(st):
    lw = np.asarray(st.pois_logw)
    ok = np.isfinite(lw)
    m = np.asarray(st.pois_d["m"])[ok]
    C = np.asarray(st.pois_d["P"])[ok]
    w = np.exp(lw[ok])
    if w.size:
        o = np.lexsort((m[:, 2], m[:, 0], -w))
        pois = (w[o], m[o], C[o])
    else:
        pois = (np.zeros(0), np.zeros((0, 4)), np.zeros((0, 4, 4)))
    gl = np.asarray(st.glogW)
    gi = np.asarray(st.gidx)
    r = np.asarray(st.trk_r)
    tm = np.asarray(st.trk_d["m"])
    tP = np.asarray(st.trk_d["P"])
    hyps = []
    for g in range(gl.shape[0]):
        if not np.isfinite(gl[g]):
            continue
        bern = [(float(r[t, gi[g, t]]), tm[t, gi[g, t]].copy(), tP[t, gi[g, t]].copy()) for t in range(gi.shape[1]) if gi[g, t] >= 0]
        hyps.append((float(np.exp(gl[g])), _bern_sort(bern)))
    hyps.sort(key=lambda t: -t[0])
    return pois, hyps


def our_estimates(out, e):
    mask, x = out[f"est{e}"]
    mask = np.asarray(mask)
    X = np.asarray(x)[mask]
    return X[np.lexsort((X[:, 2], X[:, 0]))] if len(X) else X


# ------------------------------------------------------------------------------------------------ comparison


def compare_poisson(a, b):
    """-> dict(n_a, n_b, max_abs) ; max_abs over weights, means, covs (inf if counts differ)."""
    wa, ma, Ca = a
    wb, mb, Cb = b
    if len(wa) != len(wb):
        return dict(n_a=len(wa), n_b=len(wb), max_abs=np.inf)
    if len(wa) == 0:
        return dict(n_a=0, n_b=0, max_abs=0.0)
    return dict(n_a=len(wa), n_b=len(wb), max_abs=float(max(np.abs(wa - wb).max(), np.abs(ma - mb).max(), np.abs(Ca - Cb).max())))


def compare_hyps(a, b, wmin=0.0):
    """a, b: hyps lists.  Hypotheses with weight < wmin are ignored on both sides.  -> dict with max abs difference of
    weights / existence probabilities / means / covs, counts, and 'struct_ok' (same Bernoulli counts per hypothesis)."""
    a = [h for h in a if h[0] >= wmin]
    b = [h for h in b if h[0] >= wmin]
    out = dict(n_a=len(a), n_b=len(b), w=np.inf, r=np.inf, m=np.inf, C=np.inf, struct_ok=False)
    if len(a) != len(b):
        return out
    dw = dr = dm = dC = 0.0
    ok = True
    for (wa, ba), (wb, bb) in zip(a, b):
        dw = max(dw, abs(wa - wb))
        if len(ba) != len(bb):
            ok = False
            continue
        for (ra, ma, Ca), (rb, mb, Cb) in zip(ba, bb):
            dr = max(dr, abs(ra - rb))
            dm = max(dm, float(np.abs(ma - mb).max()))
            dC = max(dC, float(np.abs(Ca - Cb).max()))
    if ok:
        out.update(w=dw, r=dr, m=dm, C=dC, struct_ok=True)
    else:
        out.update(w=dw, struct_ok=False)
    return out


def compare_est(a, b):
    if a.shape != b.shape:
        return np.inf
    return 0.0 if a.size == 0 else float(np.abs(a - b).max())


# ------------------------------------------------------------------------------------------------ fixture digests
DIG_NH = 200      # padded hypothesis count (N_h of every case is <= 200)
DIG_TOP = 3       # hypotheses stored in full
DIG_B = 128       # Bernoulli components per stored hypothesis
DIG_P = 8         # Poisson components
DIG_E = 16        # estimate set size


def digest_state(state):
    """Label-free digest of a canonical state (poisson, hyps): fixed-shape float arrays padded with NaN."""
    (w, m, C), hyps = state
    D = {}
    D["pois_w"] = np.full(DIG_P, np.nan); D["pois_m"] = np.full((DIG_P, 4), np.nan); D["pois_C"] = np.full((DIG_P, 4, 4), np.nan)
    n = min(len(w), DIG_P)
    assert len(w) <= DIG_P, "increase DIG_P"
    D["pois_w"][:n], D["pois_m"][:n], D["pois_C"][:n] = w[:n], m[:n], C[:n]
    assert len(hyps) <= DIG_NH
    for key in ("hyp_w", "hyp_nb", "hyp_sum_r", "hyp_sum_m"):
        D[key] = np.full(DIG_NH, np.nan)
    for g, (wt, bern) in enumerate(hyps):
        D["hyp_w"][g] = wt
        D["hyp_nb"][g] = len(bern)
        D["hyp_sum_r"][g] = sum(b[0] for b in bern)
        D["hyp_sum_m"][g] = sum(float(b[1][0] + b[1][2]) for b in bern)
    D["top_r"] = np.full((DIG_TOP, DIG_B), np.nan); D["top_m"] = np.full((DIG_TOP, DIG_B, 4), np.nan)
    D["top_C"] = np.full((DIG_TOP, DIG_B, 4, 4), np.nan)
    for g in range(min(DIG_TOP, len(hyps))):
        bern = hyps[g][1]
        assert len(bern) <= DIG_B, "increase DIG_B"
        for b, (r, mm, CC) in enumerate(bern):
            D["top_r"][g, b], D["top_m"][g, b], D["top_C"][g, b] = r, mm, CC
    return D


def digest_est(estimates):
    """estimates: dict e -> [n,4] arrays."""
    E = np.full((3, DIG_E, 4), np.nan)
    for e in (1, 2, 3):
        X = estimates[e]
        assert len(X) <= DIG_E
        E[e - 1, :len(X)] = X
    return E


def compare_digest(a, b, atol_w=1e-9, atol_m=1e-8, rtol=1e-9):
    """a, b digests (dict of arrays).  Returns dict key -> max |diff| (inf when the NaN pattern, i.e. a count, differs)."""
    res = {}
    for k in a:
        x, y = np.asarray(a[k], float), np.asarray(b[k], float)
        if x.shape != y.shape or not np.array_equal(np.isnan(x), np.isnan(y)):
            res[k] = np.inf
            continue
        ok = ~np.isnan(x)
        res[k] = float(np.abs(x[ok] - y[ok]).max()) if ok.any() else 0.0
    return res

"""Loader for the dumps written by scripts/official/drive_pmbm.m (GNU Octave, official PMBM code as oracle).

    from load_dump import load_run, load_meta, canonical_hyps
    d = load_run('runs/oracle_dumps/paper/run_000.mat')
    d['est1'][k]          -> float64 (n,4)   estimator-1 output at step k (0-based), computed on the UPDATED density
    d['est2'], d['est3']  likewise;  d['n_glob_upd'], d['n_glob_pruned'], d['n_trk_upd'], d['n_trk_pruned'], d['t_step']  (K,)
    if d['full']:   d['steps'][k] = {'z': (n,2), 'pred': state, 'upd': state, 'pruned': state, 'murty': [dict, ...]}
    state = {'pois_w': (N,), 'pois_m': (N,4), 'pois_P': (N,4,4),
             'tracks': [ {'t_ini': int, 'eB': (n,), 'wlog': (n,) or NaN, 'wkind': (n,) 0 none/1 weightBLog/2 weightBLog_k,
                          'mean': (n,4), 'cov': (n,4,4), 'ahis': [tuple,...] (empty tuple for phantom entries),
                          'valid': (n,) bool} , ...],
             'globHyp': (G,T) int (0 = track absent, else 1-based index into the track's hypotheses),
             'globW': (G,)}
All indices are kept 1-based exactly as in MATLAB (globHyp, aHis measurement indices, track numbers); steps/runs are 0-based.
'wlog' is the weight of the single-target hypothesis at step k: for old tracks weightBLog_k = log(1-r+r(1-p_d)) (missed) or
log(r p_d N(z;.)) (detected), for the new tracks weightBLog = log(weightB incl. clutter); NaN in the 'pruned'/'pred' states.
'valid' = False marks entries the official update leaves empty by array auto-extension (gated-out hypotheses; eB = 0).
Murty entries (only the first murty_steps steps of full-dump runs): {'p','kbest','cost_trimmed','indices_stay_row',
'indices_stay_column','fixed_row','fixed_column','cost_fixed','cost_misdetection_sum','globWeightLog_pred',
'assign_meas_to_track','glob_weight_log_children'}  (1-based indices; the matrix fed to the official murty.m is -cost_trimmed.T;
'assign_meas_to_track'[c, m] = row index (track) assigned to measurement m in child c, 0 = not assigned to a trimmed row/fixed).
"""
import numpy as np
from scipy.io import loadmat


def _v(a):
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _state(m, pre):
    g = lambda n: m[f"{pre}_{n}"]
    pw = _v(g("poisW"))
    N = pw.size
    pm = np.asarray(g("poisM"), dtype=np.float64).reshape(4, N, order="F").T
    pP = np.asarray(g("poisP"), dtype=np.float64).reshape(4, 4, N, order="F").transpose(2, 0, 1)
    tini = _v(g("trk_tini")).astype(int)
    nh = _v(g("trk_nhyp")).astype(int)
    Nh = int(nh.sum())
    h_eB, h_w, h_wk, h_valid = _v(g("h_eB")), _v(g("h_wlog")), _v(g("h_wkind")).astype(int), _v(g("h_valid")) > 0
    h_mean = np.asarray(g("h_mean"), dtype=np.float64).reshape(4, Nh, order="F").T
    h_cov = np.asarray(g("h_cov"), dtype=np.float64).reshape(4, 4, Nh, order="F").transpose(2, 0, 1)
    h_len = _v(g("h_ahis_len")).astype(int)
    AH = np.asarray(g("h_ahis"), dtype=np.float64).reshape(-1, Nh, order="F") if Nh else np.zeros((1, 0))
    tracks, o = [], 0
    for i, n in enumerate(nh):
        sl = slice(o, o + n)
        tracks.append(dict(
            t_ini=int(tini[i]), eB=h_eB[sl], wlog=h_w[sl], wkind=h_wk[sl], mean=h_mean[sl], cov=h_cov[sl],
            valid=h_valid[sl], ahis=[tuple(int(x) for x in AH[: h_len[q], q]) for q in range(o, o + n)]))
        o += n
    gw = _v(g("globW"))
    G, T = gw.size, len(tracks)
    gh = np.asarray(g("globHyp"), dtype=np.float64)
    gh = gh.reshape(G, T).astype(int) if G * T else np.zeros((G, T), dtype=int)
    return dict(pois_w=pw, pois_m=pm, pois_P=pP, tracks=tracks, globHyp=gh, globW=gw)


def _murty(m, pre, i):
    out = {}
    for key in ("p", "kbest", "cost_trimmed", "indices_stay_row", "indices_stay_column", "fixed_row", "fixed_column", "cost_fixed",
                "cost_misdetection_sum", "globWeightLog_pred", "assign_meas_to_track", "glob_weight_log_children"):
        a = np.asarray(m[f"{pre}_mur{i:02d}_{key}"], dtype=np.float64)
        if key in ("p", "kbest"):
            out[key] = int(a.reshape(-1)[0])
        elif key in ("cost_fixed", "cost_misdetection_sum", "globWeightLog_pred"):
            out[key] = float(a.reshape(-1)[0])
        elif key in ("cost_trimmed", "assign_meas_to_track"):
            out[key] = a
        else:
            out[key] = a.reshape(-1)
    return out


def load_meta(path):
    m = loadmat(path, squeeze_me=True)
    return {k: m[k] for k in m if not k.startswith("__")}


def load_run(path, with_murty=True):
    m = loadmat(path)
    K = int(m["K"].reshape(-1)[0])
    d = dict(K=K, full=bool(m["full_dump"].reshape(-1)[0]), run_index=int(m["run_index"].reshape(-1)[0]))
    for e in (1, 2, 3):
        A, cnt = np.asarray(m[f"est{e}_x"]).reshape(4, -1, K, order="F"), _v(m[f"est{e}_n"]).astype(int)
        d[f"est{e}"] = [A[:, : cnt[k], k].T.copy() for k in range(K)]
    for n in ("n_glob_upd", "n_glob_pruned", "n_trk_upd", "n_trk_pruned", "t_step"):
        d[n] = _v(m[n])
    if d["full"]:
        d["steps"] = []
        for k in range(1, K + 1):
            s = f"k{k:03d}"
            st = dict(z=np.asarray(m[f"z{s}"], dtype=np.float64).reshape(2, -1, order="F").T,
                      upd=_state(m, f"u{s}"), pruned=_state(m, f"p{s}"))
            if f"r{s}_poisW" in m:
                st["pred"] = _state(m, f"r{s}")
            if with_murty and f"u{s}_mur_n" in m:
                st["murty"] = [_murty(m, f"u{s}", i) for i in range(1, int(m[f"u{s}_mur_n"].reshape(-1)[0]) + 1)]
            d["steps"].append(st)
    return d


def canonical_hyps(state):
    """Implementation-independent global hypotheses: {frozenset of (t_ini, ahis) per present track : weight}.  Two
    hypotheses identical as sets of association histories are the same hypothesis whatever the track/row order is."""
    out = {}
    tr = state["tracks"]
    for row, w in zip(state["globHyp"], state["globW"]):
        key = frozenset((tr[i]["t_ini"], tr[i]["ahis"][j - 1]) for i, j in enumerate(row) if j > 0)
        out[key] = out.get(key, 0.0) + float(w)
    return out


def load_williams_run(path):
    """Dump of drive_williams.m (Williams TOMB/MOMB ancillary code).  Returns
    d['est_thr'][t], d['est_map'][t] (n,4) ; d['n_mb'], d['n_ppp'], d['sum_r'], d['t_step'] (K,) ;
    if full: d['steps'][t] = {'z','pred': {r,x,P,lam,xu,Pu}, 'upd': {wupd,rupd,xupd,Pupd,wnew,rnew,xnew,Pnew,pupd,pnew,lam,xu,Pu},
    'mb': {r (n,), x (n,4), P (n,4,4)}}   (wupd[i,0] = miss likelihood, wupd[i,j+1] = measurement j; same for rupd/pupd/xupd/Pupd)."""
    m = loadmat(path)
    K = int(m["K"].reshape(-1)[0])
    d = dict(K=K, full=bool(m["full_dump"].reshape(-1)[0]), run_index=int(m["run_index"].reshape(-1)[0]))
    for nm in ("est_thr", "est_map"):
        A, cnt = np.asarray(m[f"{nm}_x"]).reshape(4, -1, K, order="F"), _v(m[f"{nm}_n"]).astype(int)
        d[nm] = [A[:, : cnt[k], k].T.copy() for k in range(K)]
    for n in ("n_mb", "n_ppp", "sum_r", "t_step"):
        d[n] = _v(m[n])

    def mb(pre):
        r = _v(m[f"{pre}_r"]); n = r.size
        return dict(r=r, x=np.asarray(m[f"{pre}_x"], float).reshape(4, n, order="F").T,
                    P=np.asarray(m[f"{pre}_P"], float).reshape(4, 4, n, order="F").transpose(2, 0, 1))

    if d["full"]:
        d["steps"] = []
        for t in range(1, K + 1):
            s = f"k{t:03d}"
            pred = mb(f"r{s}")
            lam = _v(m[f"r{s}_lam"]); nu = lam.size
            pred.update(lam=lam, xu=np.asarray(m[f"r{s}_xu"], float).reshape(4, nu, order="F").T,
                        Pu=np.asarray(m[f"r{s}_Pu"], float).reshape(4, 4, nu, order="F").transpose(2, 0, 1))
            n, mm = pred["r"].size, np.asarray(m[f"u{s}_z"]).size // 2
            u = {k: np.asarray(m[f"u{s}_{k}"], float) for k in ("wupd", "rupd", "wnew", "rnew", "pupd", "pnew")}
            for k in ("wupd", "rupd", "pupd"):
                u[k] = u[k].reshape(n, mm + 1, order="F")
            for k in ("wnew", "rnew", "pnew"):
                u[k] = u[k].reshape(-1)
            u["xupd"] = np.asarray(m[f"u{s}_xupd"], float).reshape(4, n, mm + 1, order="F").transpose(1, 2, 0)   # (n, m+1, 4)
            u["Pupd"] = np.asarray(m[f"u{s}_Pupd"], float).reshape(4, 4, n, mm + 1, order="F").transpose(2, 3, 0, 1)  # (n, m+1, 4, 4)
            u["xnew"] = np.asarray(m[f"u{s}_xnew"], float).reshape(4, mm, order="F").T
            u["Pnew"] = np.asarray(m[f"u{s}_Pnew"], float).reshape(4, 4, mm, order="F").transpose(2, 0, 1)
            lam = _v(m[f"u{s}_lam"]); nu = lam.size
            u.update(lam=lam, xu=np.asarray(m[f"u{s}_xu"], float).reshape(4, nu, order="F").T,
                     Pu=np.asarray(m[f"u{s}_Pu"], float).reshape(4, 4, nu, order="F").transpose(2, 0, 1))
            d["steps"].append(dict(z=np.asarray(m[f"u{s}_z"], float).reshape(2, -1, order="F").T, pred=pred, upd=u, mb=mb(f"m{s}")))
    return d

"""Differential test driver: run OUR filter step by step on the exported measurements of one run and compare every
step with the AUTHORS' dump (drive_pmbm.m, GNU Octave).

    python diff_run.py in.mat out_dir run_index [--setting paper|code] [--json report.json]

Compared per step k (all label-free, see parity_lib):  predicted state (before the update), pruned posterior state
(after update + pruning), the three pre-pruning estimates, the number of pre-pruning children / post-pruning global
hypotheses.  Tolerances are decided by the caller (report is the raw max-abs numbers).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from scipy.io import loadmat  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)   # the paper path is float64 (the modules do not set the flag at import)

from src import algorithm as al  # noqa: E402
from src import model as md  # noqa: E402
import parity_lib as pl  # noqa: E402


def _f(x):
    return float(np.asarray(x).reshape(-1)[0])


def our_model_from_mat(P):
    """Build OUR LG model + initial state from the .mat input (the SAME explicit parameters the Octave driver reads)."""
    lg = md.LGParams(F=P["F"], Q=P["Q"], H=P["H"], R=P["Rcov"], ps=_f(P["p_s"]), pd=_f(P["p_d"]),
                     clutter_density=_f(P["intensity_clutter"]), gate=_f(P["gating"]),
                     birth_w=np.asarray(P["w_b"]).reshape(-1), birth_m=np.asarray(P["m_b"]).reshape(-1, 4),
                     birth_P=np.asarray(P["P_b"]).reshape(-1, 4, 4))
    return md.make_lg_model(lg)


def our_cfg_from_mat(P, M, T=96, H=48, Pn=16, **kw):
    d = dict(T=T, H=H, Nh=int(_f(P["N_h"])), P=Pn, M=M, prune_w=_f(P["T_pruning"]), prune_pois=_f(P["T_pruningPois"]),
             prune_r=_f(P["existence_threshold"]), est_gamma=_f(P["est1_thr"]), est_E=16, backend="host")
    d.update(kw)
    return al.PMBMConfig(**d)


def run_ours(P, run, K=None, cfg_over=None, record=True, M=None):
    """Step our filter on run `run` (0-based).  Returns list of per-step dicts: pred state, post-prune state, outs."""
    z = np.asarray(P["z"])[run]
    zv = np.asarray(P["zvalid"])[run] != 0
    K = K or z.shape[0]
    M = M or z.shape[1]
    z, zv = z[:, :M], zv[:, :M]
    assert not zv[:, :].sum(1).max() > M
    model = our_model_from_mat(P)
    cfg = our_cfg_from_mat(P, M, **(cfg_over or {}))
    predict_first = bool(_f(P["predict_first"]))
    st = al.init_state(model, cfg)
    if not predict_first:
        lam = np.asarray(P["lambda0"]).reshape(-1)
        m0 = np.asarray(P["m0"]).reshape(-1, 4)
        P0 = np.asarray(P["P0"]).reshape(-1, 4, 4)
        st = al.init_state(model, cfg, pois=(jnp.log(jnp.asarray(lam)), dict(m=jnp.asarray(m0), P=jnp.asarray(P0))))
    pred = jax.jit(lambda s: al.pmbm_predict(s, model, cfg))
    upd = jax.jit(lambda s, zz, vv: al.pmbm_update(s, zz, vv, model, cfg))
    rec = []
    for k in range(K):
        if predict_first or k > 0:
            st = pred(st)
        pred_state = pl.our_state(st) if record else None
        st, out = upd(st, jnp.asarray(z[k]), jnp.asarray(zv[k]))
        rec.append(dict(pred=pred_state, post=pl.our_state(st) if record else None,
                        est={e: pl.our_estimates(out, e) for e in (1, 2, 3)},
                        n_children=int(out["n_children"]), n_hyp=int(out["n_hyp"]), ovf=np.asarray(out["ovf"]).tolist()))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("in_mat"), ap.add_argument("out_dir"), ap.add_argument("run", type=int)
    ap.add_argument("--json")
    ap.add_argument("--K", type=int)
    ap.add_argument("--M", type=int)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--H", type=int, default=200)
    a = ap.parse_args(argv)
    P = loadmat(a.in_mat)
    d = pl.load_run(a.out_dir, a.run)
    full = int(d["full_dump"].reshape(-1)[0])
    rec = run_ours(P, a.run, a.K, cfg_over=dict(T=a.T, H=a.H), M=a.M)
    rows = []
    for k, r in enumerate(rec):
        pre = f"k{k + 1:03d}"
        row = dict(k=k + 1)
        if full:
            row["pred_pois"] = pl.compare_poisson(pl.official_state(d, "r" + pre)[0], r["pred"][0])["max_abs"]
            ho = pl.official_state(d, "r" + pre)[1]
            row["pred_hyp"] = pl.compare_hyps(ho, r["pred"][1])
            po, pp = pl.official_state(d, "p" + pre), r["post"]
            row["post_pois"] = pl.compare_poisson(po[0], pp[0])["max_abs"]
            row["post_hyp"] = pl.compare_hyps(po[1], pp[1])
            row["n_children_official"] = len(pl.official_state(d, "u" + pre)[1])
        row["n_children_ours"] = r["n_children"]
        row["n_glob_pruned_official"] = int(d["n_glob_pruned"].reshape(-1)[k])
        row["n_hyp_ours"] = r["n_hyp"]
        for e in (1, 2, 3):
            row[f"est{e}"] = pl.compare_est(pl.official_estimates(d, e, k), r["est"][e])
        rows.append(row)
    worst = lambda f: max((f(x) for x in rows), default=0.0)
    print(f"run {a.run}: steps {len(rows)}")
    for key in ("pred_pois", "post_pois"):
        if key in rows[0]:
            print(key, "max abs", worst(lambda x: x[key]))
    for key in ("pred_hyp", "post_hyp"):
        if key in rows[0]:
            print(key, "max w", worst(lambda x: x[key]["w"]), "r", worst(lambda x: x[key]["r"]), "m", worst(lambda x: x[key]["m"]),
                  "C", worst(lambda x: x[key]["C"]), "struct_ok all", all(x[key]["struct_ok"] for x in rows))
    for e in (1, 2, 3):
        print(f"est{e} max abs", worst(lambda x: x[f'est{e}']))
    print("n_hyp mismatch steps", [x["k"] for x in rows if x["n_hyp_ours"] != x["n_glob_pruned_official"]])
    if a.json:
        json.dump(rows, open(a.json, "w"), default=lambda o: float(o) if np.isscalar(o) else list(o))


if __name__ == "__main__":
    main()

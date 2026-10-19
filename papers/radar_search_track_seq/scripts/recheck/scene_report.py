"""Recheck: per-scene fixed-policy signature table from stored scene JSONs (review runs/review and the
fix-stage runs/recheck/fixstage/scenes). For every scene: T2T (n_init/N_tgt) per policy, Coverage/Random and
Static/Random M_cov (live, library, zero, pen; Static with its track-bearing episode counts), M_cov / coast value of
the same scene, Random per-episode M_cov CoV, GOSPA components, assigned / unassigned tracks per step.
Writes runs/recheck/fixstage/scene_signature.json."""
import json, os, sys
from dataclasses import replace
sys.path.insert(0, ".")
import src.common  # noqa
import jax.numpy as jnp, numpy as np  # noqa
from src.configs.config import EnvConfig  # noqa
from src import data  # noqa

R = "runs"
SRC = {**{n: f"{R}/review/scene_{n}.json" for n in ("primary", "dt5", "frob", "dt5_frob", "gnn", "clutter", "clutter5", "n12", "na20")},
       **{n: f"{R}/recheck/fixstage/scenes/scene_{n}.json" for n in ("uniform_speed", "filt_sv10", "dt5_svrule")}}


def coast_value(cfg):
    z = jnp.array([0.1, 0.2, 20e3])
    _, C0 = data.initiate_state(z, cfg)
    F, Q = (np.asarray(a) for a in data.cv_FQ(cfg.dt, cfg.q_filt))
    P = np.asarray(C0); norms = [np.linalg.norm(P)]
    for _ in range(200000):
        P = F @ P @ F.T + Q
        st = np.trace(P) if cfg.del_stat == "trace" else np.linalg.norm(P)
        if st > cfg.del_thr:
            break
        norms.append(np.linalg.norm(P))
    return float(np.mean(norms)), len(norms)


def pe(arm, s):
    return {k: np.asarray(v, float) for k, v in {**arm[s]["per_episode"], **arm[s]["per_episode_derived"]}.items()}


out = {}
for name, path in SRC.items():
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    cfg = replace(EnvConfig(), **{k: v for k, v in d["overrides"].items() if k != "gospa_metrics"})
    cv, life = coast_value(cfg)
    A = d["arms"]
    row = dict(overrides=d["overrides"], coast_value=cv, coast_life_steps=life, n_tgt=cfg.n_tgt)
    rnd = {s: pe(A["random"], s) for s in A["random"]}
    for arm in A:
        seeds = sorted(A[arm])
        t2t, cn, tb, rr = [], [], [], {"live": [], "lib": [], "zero": [], "pen": []}
        g = {k: [] for k in ("gospa_d", "gospa_loc", "gospa_miss", "gospa_false", "assigned_tracks_per_step", "unassigned_tracks_per_step")}
        for s in seeds:
            p = pe(A[arm], s)
            r = rnd[s] if s in rnd else rnd["0"]
            t2t.append(p["n_init"].mean() / cfg.n_tgt)
            fin = np.isfinite(p["m_cov"])
            tb.append(int(fin.sum()))
            if fin.any():
                cn.append(np.nanmean(p["m_cov"]) / cv)
                rr["live"].append(np.nanmean(p["m_cov"]) / np.nanmean(r["m_cov"]))
                rr["lib"].append(np.nanmean(p["m_cov_lib"]) / np.nanmean(r["m_cov_lib"]))
            rr["zero"].append(p["m_cov_zero"].mean() / r["m_cov_zero"].mean())
            rr["pen"].append(p["m_cov_pen"].mean() / r["m_cov_pen"].mean())
            for k in g:
                if k in p:
                    g[k].append(float(np.mean(p[k])))
        row[arm] = dict(t2t=float(np.mean(t2t)), track_bearing_eps=tb,
                        mcov_over_coast=float(np.mean(cn)) if cn else None,
                        ratio_to_random={k: (float(np.mean(v)) if v else None) for k, v in rr.items()},
                        gospa={k: (float(np.mean(v)) if v else None) for k, v in g.items()},
                        mcov_ep_cov=float(np.nanstd(pe(A[arm], seeds[0])["m_cov"]) / np.nanmean(pe(A[arm], seeds[0])["m_cov"]))
                        if np.isfinite(pe(A[arm], seeds[0])["m_cov"]).sum() > 1 else None)
    out[name] = row
    st, rd, cv_ = row["static"], row["random"], row["coverage"]
    print(f"{name:13s} T2T S/R/C {st['t2t']:.3f}/{rd['t2t']:.2f}/{cv_['t2t']:.2f}  C/R live {cv_['ratio_to_random']['live']:.3f} "
          f"zero {cv_['ratio_to_random']['zero']:.3f}  S/R live {st['ratio_to_random']['live']} tb {st['track_bearing_eps']}  "
          f"/coast R {rd['mcov_over_coast']:.3f} C {cv_['mcov_over_coast']:.3f}  R CoV {rd['mcov_ep_cov']:.2f}  coast {cv:.0f}/{life}")
json.dump(out, open(f"{R}/recheck/fixstage/scene_signature.json", "w"), indent=1)

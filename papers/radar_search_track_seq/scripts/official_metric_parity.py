"""Official-code check, P9: compute the evaluation metrics both ways on the SAME rollouts.

Runs our environment (analytic initiator Jacobian, CPU float64) for Random / Coverage / hand rule / ceiling on
small-scale scenarios, records detections and truths per step, our per-episode M_cov and summed GOSPA (src/eval.py), then
hands the detections and truths to the Stone Soup metric pipeline (scripts/official/ss_metric_ref.py, oracle venv) and
writes tests/fixtures/official/ss_metrics.npz + runs/official/metric_parity.json.

    JAX_PLATFORMS=cpu python scripts/official_metric_parity.py [--n-scen 3] [--T 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common as C  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src.configs.config import EnvConfig, K_SMALL, K_BASE_ACT  # noqa: E402
from src import data, baselines as B, eval as E  # noqa: E402

OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official"))
ORACLE_PY = os.path.join(OFFICIAL_ROOT, ".venv", "bin", "python")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scen", type=int, default=3)
    ap.add_argument("--T", type=int, default=1000)
    a = ap.parse_args()
    cfg = replace(EnvConfig(), init_jacobian="analytic", T=a.T, gospa_metrics=True)
    root = jax.random.PRNGKey(15)
    sk = E.scenario_keys(root, K_SMALL, a.n_scen)
    pols = {"random": B.make_random(cfg), "coverage": B.make_coverage(cfg), "hand": B.make_hand_rule(cfg, 12, 4500.0),
            "ceiling": B.make_ceiling(cfg)}

    def record(pinit, policy, scen_key, pol_key):
        s = data.reset(scen_key, cfg)
        obs = data.observe(s, cfg)
        p = pinit(jax.random.fold_in(pol_key, 0xFFFF))

        def body(carry, i):
            s, obs, p = carry
            act, p = policy(p, s, obs, jax.random.fold_in(pol_key, i))
            s, obs, info = data.step(s, act, cfg)
            return (s, obs, p), (info["z"], info["zvalid"], s.truth, info["cov_sum"], info["n_tracks"])
        _, out = jax.lax.scan(body, (s, obs, p), jnp.arange(cfg.T))
        return out

    rows, Z, ZV, TR, meta = [], [], [], [], []
    for name, (pinit, pol) in pols.items():
        pk = jax.vmap(lambda k: jax.random.fold_in(jax.random.fold_in(k, K_BASE_ACT), 0))(sk)
        ev = jax.tree.map(np.asarray, jax.jit(lambda x, y: E.eval_many(pinit, pol, x, y, cfg))(sk, pk))
        z, zv, tr, cov, ntr = (np.asarray(v) for v in jax.jit(jax.vmap(lambda x, y: record(pinit, pol, x, y)))(sk, pk))
        has = ntr > 0
        m_rep = np.where(has.any(1), (cov * has).sum(1) / np.maximum(has.sum(1), 1), np.nan)
        np.testing.assert_allclose(m_rep, ev["m_cov"], rtol=1e-12)        # the recorded rollout IS the eval rollout
        for i in range(a.n_scen):
            rows.append(dict(policy=name, scen=i, m_cov=float(ev["m_cov"][i]),
                             m_cov_lib=float(ev["m_cov_lib"][i]),
                             gospa=[float(ev[k][i]) for k in ("gospa_d", "gospa_loc", "gospa_miss", "gospa_false")]))
        Z.append(z), ZV.append(zv), TR.append(tr)
    R = np.asarray(data.meas_R(cfg))
    d = dict(traj_z=np.concatenate(Z), traj_valid=np.concatenate(ZV), truth=np.concatenate(TR), R=R,
             vel_sigma=cfg.init_vel_sigma, dt=cfg.dt, q=cfg.q_filt, gate=cfg.gate, missed=cfg.missed,
             del_thr=cfg.del_thr, c=cfg.gospa_c, p=cfg.gospa_p, analytic_jac=1)
    tmp = os.path.join(C.REPO, "runs", "official")
    os.makedirs(tmp, exist_ok=True)
    fin, fout = os.path.join(tmp, "ss_metric_in.npz"), os.path.join(tmp, "ss_metric_out.npz")
    np.savez(fin, **d)
    C.info("running Stone Soup metric pipeline", episodes=len(rows))
    subprocess.run([ORACLE_PY, "-I", os.path.join(C.REPO, "scripts", "official", "ss_metric_ref.py"), fin, fout], check=True)
    o = dict(np.load(fout))
    ours_m = np.array([r["m_cov"] for r in rows])
    ours_g = np.array([r["gospa"] for r in rows])
    fix = dict(ours_m_cov=ours_m, ours_m_cov_lib=np.array([r["m_cov_lib"] for r in rows]), ours_gospa=ours_g, ss_live_m_cov=o["live_m_cov"], ss_live_gospa=o["live_gospa"],
               ss_lib_m_cov=o["lib_m_cov"], ss_lib_gospa=o["lib_gospa"], policy=np.array([r["policy"] for r in rows]))
    np.savez_compressed(os.path.join(C.REPO, "tests", "fixtures", "official", "ss_metrics.npz"), **fix)
    rep = {}
    for name in pols:
        idx = [i for i, r in enumerate(rows) if r["policy"] == name]
        rep[name] = dict(
            ours_m_cov=ours_m[idx].tolist(), ss_live_m_cov=o["live_m_cov"][idx].tolist(),
            ss_library_m_cov=o["lib_m_cov"][idx].tolist(),
            ours_m_cov_lib=[rows[i]["m_cov_lib"] for i in idx],
            max_rel_library_m_cov=float(np.nanmax(np.abs(np.array([rows[i]["m_cov_lib"] for i in idx])
                                                         - o["lib_m_cov"][idx]) / np.abs(o["lib_m_cov"][idx]))),
            ours_gospa=ours_g[idx].tolist(), ss_live_gospa=o["live_gospa"][idx].tolist(),
            ss_library_gospa=o["lib_gospa"][idx].tolist(),
            max_rel_live_m_cov=float(np.nanmax(np.abs(ours_m[idx] - o["live_m_cov"][idx]) / np.abs(o["live_m_cov"][idx]))),
            max_rel_live_gospa=float(np.max(np.abs(ours_g[idx] - o["live_gospa"][idx]) / (np.abs(o["live_gospa"][idx]) + 1.0))),
            library_over_live_m_cov=float(np.nanmean(o["lib_m_cov"][idx]) / np.nanmean(o["live_m_cov"][idx])),
            library_over_live_gospa=(np.mean(o["lib_gospa"][idx], 0) / np.maximum(np.mean(o["live_gospa"][idx], 0), 1e-300)).tolist(),
            n_ts_live=o["live_n_ts"][idx].tolist(), n_ts_library=o["lib_n_ts"][idx].tolist(),
            n_tracks_total=o["n_tracks_total"][idx].tolist())
        C.metric("metric_parity", policy=name, **{k: v for k, v in rep[name].items() if k.startswith(("max_", "library_"))})
    C.write_json(os.path.join(tmp, "metric_parity.json"), dict(stonesoup=str(o["stonesoup_version"]), T=cfg.T,
                                                               n_scen=a.n_scen, per_policy=rep))


if __name__ == "__main__":
    main()

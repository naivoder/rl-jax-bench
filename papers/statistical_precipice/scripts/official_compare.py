"""the official-code check (standing rule 2): our pipeline vs the authors' pipeline END TO END on the public Dopamine ALE runs.
The oracle side is scripts/official/reference_ale.py (rliable@3ccd9f4 + the Colab's own normalisation cells, raw JSON
parsed independently with pandas). Writes runs/official/compare.json.

    python scripts/official_compare.py [--oracle data/official/reference_ale.npz]

Criteria (declared here before the first comparison was run):
  data        human / random / record constants and the normalised (runs, games) matrices: exact (0.0)
  points      Fig 9 aggregates, DP-25/DP-50/superhuman, profiles (81 tau), P(X>Y) (20 ordered pairs), Fig 10 IQM
              at 10 frames, per-iteration IQM/mean diagnostics: |ours - oracle| <= 1e-12
  replay      arch conf_int on a seeded stream (percentile / basic / bc / bca, B = 500) vs our ci_from_dist fed the
              SAME replicates (regenerated indices): replicates <= 1e-12, endpoints <= 1e-9 (norm.ppf / ndtri and
              percentile interpolation are the only arithmetic that differs)
  in-dist     our JAX bootstrap (B = oracle's B) vs the oracle's endpoints: |diff| <= half_band(B) + half_band(B) on
              our replicate distribution (E0 AMENDMENT 1 band); reported per endpoint as a z-like ratio
"""
from __future__ import annotations

import argparse
import os
import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO, "external", "data"))  # large inputs, see README
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from rl_jax_bench.jaxlog import check_device, info, metric, write_json  # noqa: E402

from e0_difftest import half_band  # noqa: E402
from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
from src.data import sha256  # noqa: E402


def band_ratio(dist, base, ci_o, ci_r, method, B):
    """max over endpoints of |ours - oracle| / (half_band_o + half_band_r); <= 1 passes."""
    dist, base, ci_o, ci_r = map(np.asarray, (dist, base, ci_o, ci_r))
    worst = 0.0
    for k in range(dist.shape[1]):
        hb = half_band(dist[:, k], base[k], method, B) * 2.0
        d = np.abs(ci_o[:, k] - ci_r[:, k])
        r = np.where(hb > 0, d / np.where(hb > 0, hb, 1.0), np.where(d > 1e-12, np.inf, 0.0))
        worst = max(worst, float(np.max(r)))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default=os.path.join(DATA_ROOT, "official", "reference_ale.npz"))
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "official", "compare.json"))
    args = ap.parse_args()
    check_device()
    O = np.load(args.oracle)
    algs = [str(a) for a in O["algs"]]
    games = [str(g) for g in O["games"]]
    res = {"oracle": os.path.relpath(args.oracle, REPO), "oracle_sha256": sha256(args.oracle)}

    # ---------------------------------------------------------------- data + normalisation (our loader, no fetch)
    src = D.sources()["normalisation"]
    ours_games = D.ale_games()
    res["games_same_set"] = sorted(ours_games) == sorted(games)
    hum = np.array([src["ale_human"]["value"][g] for g in games])
    rnd = np.array([src["ale_random"]["value"][g] for g in games])
    rec = np.array([src["ale_record"]["value"][g] for g in games])
    res["const_maxdiff"] = float(max(np.abs(hum - O["human"]).max(), np.abs(rnd - O["random"]).max(),
                                     np.abs(rec - O["record"]).max()))
    cube, g_ours, _ = D.load_dopamine_ale(final_only=False, normalised=True, fetch=False)
    perm = [g_ours.index(g) for g in games]
    cube = {a: v[:, perm, :] for a, v in cube.items()}
    final = {a: cube[a][:, :, -1] for a in algs}
    raw_cube, _, _ = D.load_dopamine_ale(final_only=True, normalised=False, fetch=False)
    rec_norm = {a: (raw_cube[a][:, perm] - rnd) / (rec - rnd) for a in algs}
    res["norm_maxdiff"] = float(max(np.abs(final[a] - O[f"norm_{j}"]).max() for j, a in enumerate(algs)))
    res["normrec_maxdiff"] = float(max(np.abs(rec_norm[a] - O[f"normrec_{j}"]).max() for j, a in enumerate(algs)))
    metric("data", const=res["const_maxdiff"], norm=res["norm_maxdiff"], normrec=res["normrec_maxdiff"])

    # ---------------------------------------------------------------- points
    pts = {}
    agg = lambda x: A.aggregates4(x)  # noqa: E731  [median, IQM, mean, OG] -- same order as the Colab
    alt = lambda x: jnp.stack([A.difficulty_progress(x, 0.25), A.difficulty_progress(x, 0.5), A.superhuman_prob(x)])  # noqa: E731
    tau = jnp.asarray(O["tau"])
    fr = O["fig10_frames_idx"]
    for j, a in enumerate(algs):
        x = jnp.asarray(final[a])
        c = jnp.asarray(cube[a])
        pts[f"fig9_{a}"] = float(np.abs(np.asarray(agg(x)) - O[f"fig9_s0_{j}_pt"]).max())
        pts[f"a25_{a}"] = float(np.abs(np.asarray(alt(x)) - O[f"a25_{j}_pt"]).max())
        pts[f"prof_{a}"] = float(np.abs(np.asarray(A.score_profile(x, tau)) - O[f"prof_{j}"]).max())
        f10 = np.array([float(A.agg_iqm(c[:, :, f])) for f in fr])
        pts[f"fig10_{a}"] = float(np.abs(f10 - O[f"fig10_{j}_pt"]).max())
        iq = np.array([float(A.agg_iqm(c[:, :, i])) for i in range(c.shape[-1])])
        mn = np.array([float(A.agg_mean(c[:, :, i])) for i in range(c.shape[-1])])
        pts[f"diag_{a}"] = float(max(np.abs(iq - O[f"diag_iqm_by_iter_{j}"]).max(),
                                     np.abs(mn - O[f"diag_mean_by_iter_{j}"]).max()))
    poi = []
    for k, nm in enumerate(O["poi_names"]):
        xa, ya = str(nm).split("|")
        poi.append(abs(float(A.poi(jnp.asarray(final[xa]), jnp.asarray(final[ya]))) - O["poi_pt"][k]))
    pts["poi_all_pairs"] = float(max(poi))
    res["points_maxdiff"] = pts
    res["points_max"] = float(max(pts.values()))
    metric("points", max=res["points_max"])

    # ---------------------------------------------------------------- exact CI-construction replay
    rp = {}
    for j, a in enumerate(algs):
        x = jnp.asarray(final[a])
        dist = A.bootstrap_dist_from_indices((x,), (jnp.asarray(O[f"cir_{j}_idx"], jnp.int32),), A.aggregates4)
        rp[f"replicates_{a}"] = float(np.abs(np.asarray(dist) - O[f"cir_{j}_res"]).max())
        base = A.aggregates4(x)
        jk = A.jackknife((x,), A.aggregates4)
        for m in ("percentile", "basic", "bc", "bca"):
            ci = np.asarray(A.ci_from_dist(dist, base, m, 0.95, jk if m == "bca" else None, pooled_jk_mean=True))
            rp[f"{m}_{a}"] = float(np.nanmax(np.abs(ci - O[f"cir_{j}_{m}"])))
            if np.isnan(ci).any() != np.isnan(O[f"cir_{j}_{m}"]).any():
                rp[f"{m}_{a}"] = float("inf")
    res["replay_maxdiff"] = rp
    res["replay_replicates_max"] = float(max(v for k, v in rp.items() if k.startswith("replicates")))
    res["replay_ci_max"] = float(max(v for k, v in rp.items() if not k.startswith("replicates")))
    metric("replay", replicates=res["replay_replicates_max"], ci=res["replay_ci_max"])

    # ---------------------------------------------------------------- in distribution (different RNGs)
    B = int(O["meta_reps_agg"])
    key = A.branch_key(0, A.EXP_IDS["e3"], 0, 25)
    ind, ours_ci = {}, {}
    for j, a in enumerate(algs):
        kj = jax.random.fold_in(key, j)
        x = jnp.asarray(final[a])
        base, ci, dist = A.interval_estimate(kj, (x,), A.aggregates4, B)
        ind[f"fig9_{a}"] = max(band_ratio(dist, base, O[f"fig9_s{s}_{j}_ci"], ci, "percentile", B) for s in (0, 1, 2))
        ours_ci[a] = {"pt": np.asarray(base).tolist(), "ci": np.asarray(ci).tolist()}
        dist_t = A.bootstrap_dist(jax.random.fold_in(kj, 1), (x,), A.aggregates4, B, task_bootstrap=True)
        ci_t = A.ci_from_dist(dist_t, base, "percentile")
        ind[f"a20_{a}"] = band_ratio(dist_t, base, O[f"a20_{j}_ci"], ci_t, "percentile", B)
        ours_ci[a]["a20_ci"] = np.asarray(ci_t).tolist()
        base2, ci2, dist2 = A.interval_estimate(jax.random.fold_in(kj, 2), (x,), alt, B)
        ind[f"a25_{a}"] = band_ratio(dist2, base2, O[f"a25_{j}_ci"], ci2, "percentile", B)
        prof = lambda y: A.score_profile(y, tau)  # noqa: E731
        base3, ci3, dist3 = A.interval_estimate(jax.random.fold_in(kj, 3), (x,), prof, 2000)
        ind[f"prof_{a}"] = band_ratio(dist3, base3, O[f"prof_{j}_ci"], ci3, "percentile", 2000)
        c = jnp.asarray(cube[a][:, :, fr])
        f10 = lambda y: jnp.stack([A.agg_iqm(y[:, :, f]) for f in range(y.shape[-1])])  # noqa: E731
        # rliable draws an independent run index per (n, m, t) for 3-D inputs (R8); ours resamples per frame slice
        # independently too: bootstrap each frame with its own key
        dists, bases = [], []
        for f in range(c.shape[-1]):
            b_, _, d_ = A.interval_estimate(jax.random.fold_in(jax.random.fold_in(kj, 4), f), (c[:, :, f],),
                                            lambda y: A.agg_iqm(y), 2000)
            dists.append(np.asarray(d_)[:, 0]); bases.append(float(np.asarray(b_)[0]))
        d4 = np.stack(dists, 1)
        ci4 = np.asarray(A.ci_from_dist(jnp.asarray(d4), jnp.asarray(bases), "percentile"))
        ind[f"fig10_{a}"] = band_ratio(d4, bases, O[f"fig10_{j}_ci"], ci4, "percentile", 2000)
        ours_ci[a]["fig10_pt"], ours_ci[a]["fig10_ci"] = bases, ci4.tolist()
    pr = []
    for k, nm in enumerate(O["poi_names"]):
        xa, ya = str(nm).split("|")
        b5, ci5, d5 = A.interval_estimate(jax.random.fold_in(key, 100 + k),
                                          (jnp.asarray(final[xa]), jnp.asarray(final[ya])), A.poi, 1000)
        pr.append(band_ratio(d5, b5, O["poi_ci"][k][:, None], ci5, "percentile", 1000))
    ind["poi_all_pairs"] = float(max(pr))
    res["in_distribution_band_ratio"] = {k: float(v) for k, v in ind.items()}
    res["in_distribution_max_ratio"] = float(max(ind.values()))
    res["ours"] = ours_ci
    metric("in_dist", max_ratio=res["in_distribution_max_ratio"])

    res["pass"] = {
        "data": res["games_same_set"] and res["const_maxdiff"] == 0.0 and res["norm_maxdiff"] == 0.0
                and res["normrec_maxdiff"] <= 1e-15,
        "points": res["points_max"] <= 1e-12,
        "replay": res["replay_replicates_max"] <= 1e-12 and res["replay_ci_max"] <= 1e-9,
        "in_distribution": res["in_distribution_max_ratio"] <= 1.0,
    }
    res["PASS"] = all(res["pass"].values())
    info(f"PASS={res['PASS']} {res['pass']}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_json(args.out, res)


if __name__ == "__main__":
    main()

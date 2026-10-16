"""Official-code parity (standing rule 2, the official-code check) on the authors' pipeline END TO END: the fixture
tests/fixtures/official_ale.npz was recorded from scripts/official/reference_ale.py, which ran rliable@3ccd9f4 plus the
Colab's own normalisation cells (59-61) on the public Dopamine ALE runs (5 agents x 5 runs x 55 games), in the isolated
official venv. Numbers only; none of their code is in this repo.

Tolerances: constants and normalised matrices exact; point estimates 1e-12 (float64); arch's percentile / basic / bc /
bca endpoints recomputed by our ci_from_dist on the replayed replicate set 1e-9 (observed <= 1e-15)."""
import os

import jax.numpy as jnp
import numpy as np

from src import algorithm as A
from src import data as D

HERE = os.path.dirname(os.path.abspath(__file__))
F = np.load(os.path.join(HERE, "fixtures", "official_ale.npz"))
GAMES = [str(g) for g in F["games"]]
ALGS = [str(a) for a in F["algs"]]


def test_constants_match_colab():
    s = D.sources()["normalisation"]
    assert sorted(D.ale_games()) == GAMES
    for key, col in (("ale_human", "human"), ("ale_random", "random"), ("ale_record", "record")):
        np.testing.assert_array_equal([s[key]["value"][g] for g in GAMES], F[col])


def test_normalisation_exact():
    s = D.sources()["normalisation"]
    for j in range(len(ALGS)):
        ours = D.normalise(F[f"raw_{j}"], GAMES, s["ale_random"]["value"], s["ale_human"]["value"])
        np.testing.assert_array_equal(ours, F[f"norm_{j}"])
        rec = D.normalise(F[f"raw_{j}"], GAMES, s["ale_random"]["value"], s["ale_record"]["value"])
        np.testing.assert_allclose(rec, F[f"normrec_{j}"], rtol=0, atol=1e-15)


def test_points_fig9_a25_profiles_fig10():
    s = D.sources()["normalisation"]
    tau = jnp.asarray(F["tau"])
    for j in range(len(ALGS)):
        x = jnp.asarray(F[f"norm_{j}"])
        np.testing.assert_allclose(A.aggregates4(x), F[f"fig9_s0_{j}_pt"], rtol=0, atol=1e-12)
        alt = [A.difficulty_progress(x, 0.25), A.difficulty_progress(x, 0.5), A.superhuman_prob(x)]
        np.testing.assert_allclose(alt, F[f"a25_{j}_pt"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(A.score_profile(x, tau), F[f"prof_{j}"], rtol=0, atol=1e-12)
        fr = F[f"rawframes_{j}"]  # (runs, games, frames)
        cube = np.moveaxis(D.normalise(np.moveaxis(fr, 1, -1), GAMES, s["ale_random"]["value"],
                                       s["ale_human"]["value"]), -1, 1)
        f10 = [A.agg_iqm(jnp.asarray(cube[:, :, f])) for f in range(cube.shape[-1])]
        np.testing.assert_allclose(f10, F[f"fig10_{j}_pt"], rtol=0, atol=1e-12)


def test_poi_all_pairs():
    idx = {a: j for j, a in enumerate(ALGS)}
    for k, nm in enumerate(F["poi_names"]):
        xa, ya = str(nm).split("|")
        p = A.poi(jnp.asarray(F[f"norm_{idx[xa]}"]), jnp.asarray(F[f"norm_{idx[ya]}"]))
        np.testing.assert_allclose(float(p), F["poi_pt"][k], rtol=0, atol=1e-12)


def test_ci_construction_replay():
    for j in (0, 4):
        x = jnp.asarray(F[f"norm_{j}"])
        dist = A.bootstrap_dist_from_indices((x,), (jnp.asarray(F[f"cir_{j}_idx"], jnp.int32),), A.aggregates4)
        np.testing.assert_allclose(dist, F[f"cir_{j}_res"], rtol=0, atol=1e-12)
        # replicate MEDIANS bit-identical (bc / bca use the strict mean(dist < base); a 1-ulp task mean moves an
        # endpoint by one atom -- the CPU exact_div-under-vmap bug found in the official-code check)
        np.testing.assert_array_equal(np.asarray(dist)[:, 0], F[f"cir_{j}_res"][:, 0])
        base = A.aggregates4(x)
        jk = A.jackknife((x,), A.aggregates4)
        for m in ("percentile", "basic", "bc", "bca"):
            ci = A.ci_from_dist(dist, base, m, 0.95, jk if m == "bca" else None, pooled_jk_mean=True)
            np.testing.assert_allclose(ci, F[f"cir_{j}_{m}"], rtol=0, atol=1e-9)

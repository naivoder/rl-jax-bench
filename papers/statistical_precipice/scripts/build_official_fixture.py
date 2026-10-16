"""Extract tests/fixtures/official_ale.npz from the authors'-pipeline reference run data/official/reference_ale.npz
(numbers only, never their code) for tests/test_official_parity.py. Run with python.

Contents: the Colab's human / random / record constants (55 games, sorted names), the raw Dopamine final-iteration
scores (parsed here with plain json -- and asserted to reproduce the oracle's normalised matrices bit for bit with
the oracle's constants, so the raw values are validated by the oracle), the oracle's normalised matrices, its point
estimates (Fig 9, A.25, profiles, Fig 10 frames, P(X>Y) pairs), the per-frame normalised cubes, and the exact
CI-construction replay (indices, replicates, arch percentile / basic / bc / bca endpoints) for two agents."""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO, "external", "data"))  # large inputs, see README
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_ROOT, "official", "reference_ale.npz")
O = np.load(src)
algs = [str(a) for a in O["algs"]]
games = [str(g) for g in O["games"]]
DOP = {"DQN (Nature)": "DQN", "DQN (Adam)": "DQN (Adam + MSE in JAX)", "C51": "C51", "IQN": "IQN", "Rainbow": "Rainbow"}


def runs_of(path, agent):
    rows = [r for r in json.load(open(path)) if r["Agent"] == agent]
    runs, cur = [], []
    for r in rows:
        if int(r["Iteration"]) == 0 and cur:
            runs.append(cur)
            cur = []
        cur.append(float(r["Value"]))
    runs.append(cur)
    return np.array(runs)


out = {k: O[k] for k in ("games", "human", "random", "record", "algs", "tau", "fig10_frames_idx", "poi_names",
                         "poi_pt")}
for j, a in enumerate(algs):
    cube = np.stack([runs_of(os.path.join(DATA_ROOT, "dopamine", f"{g.lower()}.json"), DOP[a]) for g in games], 1)
    raw = cube[:, :, -1]
    norm = (raw - O["random"]) / (O["human"] - O["random"])
    assert np.array_equal(norm, O[f"norm_{j}"]), a
    out[f"raw_{j}"] = raw
    out[f"rawframes_{j}"] = cube[:, :, O["fig10_frames_idx"]]
    for k in ("norm_{j}", "normrec_{j}", "fig9_s0_{j}_pt", "a25_{j}_pt", "prof_{j}", "fig10_{j}_pt"):
        out[k.format(j=j)] = O[k.format(j=j)]
for j in (0, 4):  # DQN (Nature), Rainbow
    for m in ("idx", "res", "percentile", "basic", "bc", "bca"):
        out[f"cir_{j}_{m}"] = O[f"cir_{j}_{m}"]
p = os.path.join(REPO, "tests", "fixtures", "official_ale.npz")
np.savez_compressed(p, **out)
print(p, os.path.getsize(p))

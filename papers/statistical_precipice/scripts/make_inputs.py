"""E0 differential-test inputs the README (ii)): 200 score matrices (N runs, M tasks) with
N in {3, 5, 10, 100} x M in {6, 16, 26, 55} from five generators (heavy-tailed, multimodal, tied, constant-column,
uniform), plus the the README toy. numpy only, fixed seed; writes data/e0/inputs.npz and prints its sha256.
Run with python (never imports rliable)."""
import hashlib
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO, "external", "data"))  # large inputs, see README
OUT = os.path.join(DATA_ROOT, "e0", "inputs.npz")
NS, MS = (3, 5, 10, 100), (6, 16, 26, 55)
GENS = ("heavy", "multimodal", "tied", "constcol", "uniform")


def gen(rng, kind, n, m):
    loc = rng.uniform(0.0, 1.5, m)                       # task-dependent centres
    if kind == "heavy":
        return loc + 0.2 * rng.standard_t(2, (n, m))
    if kind == "multimodal":
        mode = rng.random((n, m)) < 0.3
        return np.where(mode, loc + 1.5, loc) + 0.1 * rng.standard_normal((n, m))
    if kind == "tied":
        return np.round(loc + 0.3 * rng.standard_normal((n, m)), 1)
    if kind == "constcol":
        x = loc + 0.3 * rng.standard_normal((n, m))
        x[:, rng.integers(m)] = 0.5                        # one task with all runs equal
        return x
    return rng.uniform(0.0, 2.0, (n, m))


def main():
    rng = np.random.default_rng(20211013)
    arrs = {}
    i = 0
    while i < 200:
        n, m, kind = NS[i % 4], MS[(i // 4) % 4], GENS[(i // 16) % 5]
        arrs[f"x{i:03d}"] = gen(rng, kind, n, m)
        arrs[f"y{i:03d}"] = gen(rng, kind, n if i % 3 else max(3, n // 2 + 1), m)  # every third pair has unequal N (independent bootstrap)
        arrs[f"kind{i:03d}"] = np.array(kind)
        i += 1
    arrs["toy_x"] = np.array([[0.1, 0.9], [0.3, 0.3], [1.5, 0.6]]).T   # D9, (runs, tasks)
    arrs["toy_y"] = np.array([[0.2, 0.9], [0.0, 0.4], [0.6, 0.8]]).T
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **arrs)
    print(OUT, hashlib.sha256(open(OUT, "rb").read()).hexdigest())


if __name__ == "__main__":
    sys.exit(main())

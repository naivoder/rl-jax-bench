"""GOSPA (eq. 18) JAX DP vs the authors' gospapy-derived calculate_gospa and vs Stone Soup 1.9.1 (when available)."""
import numpy as np
import jax, jax.numpy as jnp
import pytest
from src import gospa_jax as G
from src import model as M
import official


def _random_case(rng):
    nt = rng.integers(0, 6)
    ne = rng.integers(0, 9)
    scale = rng.choice([5.0, 40.0, 120.0])
    T = rng.normal(size=(nt, 4)) * scale
    E = rng.normal(size=(ne, 4)) * scale
    # make some estimates near truths
    for i in range(min(nt, ne)):
        if rng.random() < 0.6:
            E[i] = T[i] + rng.normal(size=4) * rng.choice([1.0, 20.0, 70.0])
    return T, E


def _pad(T, E):
    tx = np.zeros((M.K_TRUTH, 4)); tm = np.zeros(M.K_TRUTH, bool)
    ex = np.zeros((M.E_MAX, 4)); em = np.zeros(M.E_MAX, bool)
    tx[: len(T)] = T; tm[: len(T)] = True
    ex[: len(E)] = E; em[: len(E)] = True
    return tx, tm, ex, em


def test_gospa_matches_official_random():
    off = official.load()
    if off is None:
        pytest.skip(official.SKIP)
    f = jax.jit(lambda tx, tm, ex, em: G.gospa_parts(tx, tm, ex, em, 80.0))
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(300):
        T, E = _random_case(rng)
        g, _, loc, mis, fal = off["gospa"].calculate_gospa(T, E, 80.0, 2)
        ref = (g ** 2, loc, mis, fal)
        out = [float(v) for v in f(*_pad(T, E))]
        for a, b in zip(out, ref):
            worst = max(worst, abs(a - b) / max(1.0, abs(b)))
    print(f"[METRIC] gospa vs gospapy: 300 cases, worst rel err = {worst:.2e}")
    assert worst < 1e-12, worst


def test_gospa_position_subspace():
    f = jax.jit(lambda tx, tm, ex, em: G.gospa_parts(tx, tm, ex, em, 80.0, M.POS_IDX))
    T = np.array([[0, 5, 0, 5.0], [100, 0, 0, 0]]); E = np.array([[3, -9, 4, 9.0]])
    tot = float(f(*_pad(T, E))[0])
    # target 0 <-> estimate: position distance 5 -> 25; target 1 missed: 3200
    assert abs(tot - (25 + 3200)) < 1e-9


def test_gospa_stonesoup_crosscheck():
    import subprocess, os, json, sys
    py = os.path.join(official.OFFICIAL_ROOT, "stonesoup_venv", "bin", "python")
    if not os.path.exists(py):
        pytest.skip("Stone Soup venv absent: run `sh scripts/official/setup.sh` (or set OFFICIAL_ROOT)")
    rng = np.random.default_rng(1)
    cases = [_random_case(rng) for _ in range(30)]
    payload = json.dumps([[T.tolist(), E.tolist()] for T, E in cases])
    code = r'''
import sys, json, numpy as np
from datetime import datetime
from stonesoup.measures import Euclidean
from stonesoup.metricgenerator.ospametric import GOSPAMetric
from stonesoup.types.state import State
from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
from stonesoup.types.track import Track
from stonesoup.types.detection import Detection
data = json.loads(sys.stdin.read())
out = []
for T, E in data:
    # direct GOSPA via the Stone Soup measure/assignment used by its GOSPAMetric (compute_gospa_metric)
    gm = GOSPAMetric(c=80.0, p=2, measure=Euclidean())
    t0 = datetime(2020, 1, 1)
    truths = [GroundTruthState(np.array(t), timestamp=t0) for t in T]
    tracks = [State(np.array(e), timestamp=t0) for e in E]
    res = gm.compute_gospa_metric(tracks, truths)
    out.append(float(res[0].value["distance"]))
print(json.dumps(out))
'''
    r = subprocess.run([py, "-I", "-c", code], input=payload, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        pytest.skip("stonesoup call failed: " + r.stderr[-300:])
    ss = json.loads(r.stdout.strip().splitlines()[-1])
    f = jax.jit(lambda tx, tm, ex, em: G.gospa_parts(tx, tm, ex, em, 80.0))
    worst = 0.0
    for (T, E), s in zip(cases, ss):
        mine = float(f(*_pad(T, E))[0]) ** 0.5
        worst = max(worst, abs(mine - s))
    print(f"[METRIC] gospa vs Stone Soup 1.9.1: 30 cases, worst abs err (d) = {worst:.2e}")
    assert worst < 1e-9, worst

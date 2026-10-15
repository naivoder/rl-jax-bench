"""src/timelimits.py (float64 numpy reference helpers) against src/algorithm.py::gae_peb and hand-checkable cases
(CPU, < 5 s)."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import timelimits as TLS  # noqa: E402


@pytest.mark.parametrize("peb", [0, 1])
@pytest.mark.parametrize("gamma,lam", [(0.99, 0.95), (1.0, 0.7)])
def test_gae_two_flag_equals_gae_peb(peb, gamma, lam):
    rng = np.random.default_rng(3)
    M, B = 64, 5
    r = rng.normal(size=(M, B)).astype(np.float32)
    v = rng.normal(size=(M, B)).astype(np.float32)
    fv = rng.normal(size=(M, B)).astype(np.float32)
    resets = rng.random((M, B)) < 0.08
    terms = resets & (rng.random((M, B)) < 0.5)
    a_j, t_j = A.gae_peb(jnp.asarray(r), jnp.asarray(v), jnp.asarray(fv), jnp.asarray(terms), jnp.asarray(resets),
                         gamma, lam, jnp.float32(peb))
    a_n, t_n = TLS.gae_two_flag(r, v, fv, terms, resets, gamma, lam, peb=bool(peb))
    np.testing.assert_allclose(np.asarray(a_j), a_n, atol=2e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(t_j), t_n, atol=2e-4, rtol=1e-4)


def test_aliasing_cost_and_boundary_windows():
    # return is a function of the fine time only: the fine lookup explains (almost) all of it, the coarse one less
    rng = np.random.default_rng(5)
    N, T = 12, 400
    wtime = np.tile(np.arange(T) % 200, (N, 1)) * 0.6
    ep = np.tile((np.arange(T) >= 200).astype(np.int64), (N, 1))
    ttime = wtime + ep * 120.0
    G = 0.01 * (240.0 - ttime) + rng.normal(0, 0.3, (N, T))
    live = rng.random((N, T)) > 0.05
    a = TLS.aliasing_cost(G, live, ttime, wtime, 6.0)
    assert a["R2_fine"] > a["R2_coarse"] and a["A_alias"] > 0.1
    assert abs(a["A_alias"] - (a["R2_fine"] - a["R2_coarse"])) < 1e-12
    done = np.zeros((2, 10), np.int8)
    done[0, 4] = 1
    done[1, 9] = 1
    W = TLS.boundary_windows(done, 3)
    assert W[0].tolist() == [False, False, True, True, True, False, False, False, False, False]
    assert W[1].tolist() == [False] * 7 + [True] * 3
    adv = np.zeros((2, 10))
    adv[0, 3] = 1.0
    assert 0.0 < TLS.adv_share(adv, W) <= 1.0
    assert abs(TLS.adv_share(adv, np.ones_like(W)) - 1.0) < 1e-12

"""Tests for src/kbest.py (k-best data association).  JAX_PLATFORMS=cpu, whole file < 30 s."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from src import kbest as K  # noqa: E402

INF = np.inf


def rand_problem(rng, G, M, T, kmax, p_inf=0.4, integer=False, p_valid=0.85):
    co = rng.normal(size=(G, M, T)) * 5
    co[rng.random((G, M, T)) < p_inf] = INF
    cn = rng.normal(size=M) * 5
    if integer:
        co, cn = np.round(co), np.round(cn)
    mv = rng.random(M) < p_valid
    kv = rng.integers(0, kmax + 2, size=G)  # may exceed kmax
    return co, cn, mv, kv


def check_valid(co, cn, mv, kv, kmax, assign, cost):
    """Every returned assignment is feasible, its cost matches, rows ascending, +inf padding past k."""
    G, M, T = co.shape
    for g in range(G):
        kg = min(int(kv[g]), kmax)
        prev = -INF
        seen = set()
        for k in range(kmax):
            if not np.isfinite(cost[g, k]):
                assert (assign[g, k] == -1).all()
                continue
            assert k < kg
            a = assign[g, k]
            used = [t for t in a if t >= 0]
            assert len(used) == len(set(used))
            tot = 0.0
            for m in range(M):
                if not mv[m]:
                    assert a[m] == -1
                elif a[m] < 0:
                    tot += cn[m]
                else:
                    assert np.isfinite(co[g, m, a[m]])
                    tot += co[g, m, a[m]]
            assert abs(tot - cost[g, k]) < 1e-9
            assert cost[g, k] >= prev - 1e-9
            prev = cost[g, k]
            assert tuple(a) not in seen
            seen.add(tuple(a))


def compare(ref, got, atol=1e-12, distinct_sets=True):
    a0, c0 = np.asarray(ref[0]), np.asarray(ref[1])
    a1, c1 = np.asarray(got[0]), np.asarray(got[1])
    assert (np.isinf(c0) == np.isinf(c1)).all()
    f = np.isfinite(c0)
    assert np.abs(c0[f] - c1[f]).max(initial=0.0) <= atol
    if distinct_sets:
        # assignments must agree wherever the cost is separated from both neighbours
        G, kmax = c0.shape
        for g in range(G):
            for k in range(kmax):
                if not f[g, k]:
                    continue
                lo = abs(c0[g, k] - c0[g, k - 1]) if k > 0 else INF
                hi = abs(c0[g, k] - c0[g, k + 1]) if k + 1 < kmax and f[g, k + 1] else INF
                if min(lo, hi) > 1e-9:
                    assert (a0[g, k] == a1[g, k]).all(), (g, k)
    else:
        # ties: the multiset of assignments among the best (k-1) costs agrees
        pass


G, M, T, KMAX = 8, 5, 4, 12
_jax = jax.jit(lambda a, b, c, d: K.kbest_jax(a, b, c, d, KMAX))
_jax_vm = jax.jit(lambda a, b, c, d: K.kbest_jax(a, b, c, d, KMAX, mode="vmap"))


@pytest.mark.parametrize("integer", [False, True])
def test_host_and_jax_vs_brute_small(integer):
    rng = np.random.default_rng(1 + integer)
    nprob = 0
    for it in range(110):
        co, cn, mv, kv = rand_problem(rng, G, M, T, KMAX, integer=integer)
        ref = K.kbest_brute(co, cn, mv, kv, KMAX)
        h = K.kbest_host_numpy(co, cn, mv, kv, KMAX)
        j = _jax(co, cn, mv, kv)
        check_valid(co, cn, mv, kv, KMAX, h[0], h[1])
        assert not np.asarray(h[2]).any() and not np.asarray(j[2]).any()
        compare(ref, h, distinct_sets=not integer)
        compare(ref, j, distinct_sets=not integer)
        if integer:  # ties: the sets agree up to the cost level of the last returned element
            for g in range(G):
                for name, r in (("host", h), ("jax", j)):
                    c = np.asarray(r[1])[g]
                    if np.isfinite(c).any():
                        cut = c[np.isfinite(c)].max()
                        s0 = {tuple(x) for x, cc in zip(ref[0][g], ref[1][g]) if cc < cut - 1e-9}
                        s1 = {tuple(x) for x, cc in zip(np.asarray(r[0])[g], c) if cc < cut - 1e-9}
                        assert s0 == s1, name
        nprob += G
    assert nprob >= 800


def test_jax_vmap_mode_and_split_vs_brute():
    rng = np.random.default_rng(5)
    f_split = jax.jit(lambda a, b, c, d: K.kbest_jax(a, b, c, d, KMAX, mode="vmap", split=(2, 3)))
    for it in range(15):
        co, cn, mv, kv = rand_problem(rng, G, M, T, KMAX)
        ref = K.kbest_brute(co, cn, mv, kv, KMAX)
        compare(ref, _jax_vm(co, cn, mv, kv))
        a, c, o = f_split(co, cn, mv, kv)
        o = np.asarray(o)
        # parents outside the 2 biggest with k>3 must be flagged; everything unflagged must be exact
        for g in range(G):
            if not o[g]:
                compare((ref[0][g:g + 1], ref[1][g:g + 1]), (np.asarray(a)[g:g + 1], np.asarray(c)[g:g + 1]))
        order = np.argsort(-np.minimum(kv, KMAX), kind="stable")
        for g in order[2:]:
            if min(kv[g], KMAX) > 3:
                assert o[g]


def test_host_vs_jax_midsize():
    rng = np.random.default_rng(7)
    Gm, Mm, Tm, km = 12, 14, 30, 40
    fj = jax.jit(lambda a, b, c, d: K.kbest_jax(a, b, c, d, km))
    for it in range(6):
        co = np.full((Gm, Mm, Tm), INF)
        for g in range(Gm):
            for m in rng.choice(Mm, 6, replace=False):
                ts = rng.choice(Tm, rng.integers(1, 6), replace=False)
                co[g, m, ts] = rng.uniform(-8, 8, size=len(ts))
        cn = rng.uniform(2, 9, size=Mm)
        mv = rng.random(Mm) < 0.9
        kv = rng.integers(0, km + 1, size=Gm)
        h = K.kbest_host_numpy(co, cn, mv, kv, km)
        j = fj(co, cn, mv, kv)
        assert not np.asarray(j[2]).any()
        check_valid(co, cn, mv, kv, km, h[0], h[1])
        compare(h, j, atol=1e-9)
        # also the scalable "map" amax/dmax options
        a2 = K.kbest_jax(co, cn, mv, kv, km, amax=8, dmax=6)
        assert not np.asarray(a2[2]).any()
        compare(h, a2, atol=1e-9)


def test_edge_cases():
    rng = np.random.default_rng(3)
    kmax = 6
    # kvec = 0 and k larger than the number of feasible assignments
    co = np.full((3, 2, 2), INF)
    co[1, 0, 0] = 1.0
    co[2, 0, 0] = 1.0
    co[2, 1, 0] = 2.0
    cn = np.array([3.0, 4.0])
    mv = np.array([True, True])
    kv = np.array([0, 5, 99])
    for fn in (K.kbest_brute, K.kbest_host_numpy, lambda *a: K.kbest_jax(*a)):
        a, c, o = fn(co, cn, mv, kv, kmax)
        a, c = np.asarray(a), np.asarray(c)
        assert np.isinf(c[0]).all() and (a[0] == -1).all()
        assert np.allclose(c[1][:2], [1.0 + 4.0, 7.0]) and np.isinf(c[1][2:]).all()  # 2 feasible, k=5
        # g=2: (t0,new)=1+4, (new,t0)=3+2, (new,new)=7 ; (t0 for both) impossible
        assert np.allclose(c[2][:3], [5.0, 5.0, 7.0]) and np.isinf(c[2][3:]).all()
        assert not np.asarray(o).any()
    # all measurements invalid
    mv0 = np.zeros(2, bool)
    for fn in (K.kbest_brute, K.kbest_host_numpy, lambda *a: K.kbest_jax(*a)):
        a, c, o = fn(co, cn, mv0, np.array([1, 2, 3]), kmax)
        c, a = np.asarray(c), np.asarray(a)
        assert (c[:, 0] == 0.0).all() and np.isinf(c[:, 1:]).all() and (a == -1).all()
    # negative costs, all-fixed (no finite old) parents
    co = np.full((2, 3, 4), INF)
    cn = np.array([-1.5, 2.0, -7.0])
    for fn in (K.kbest_host_numpy, lambda *a: K.kbest_jax(*a)):
        a, c, o = fn(co, cn, np.ones(3, bool), np.array([4, 1]), kmax)
        c = np.asarray(c)
        assert np.allclose(c[:, 0], -6.5) and np.isinf(c[0, 1:]).all()
    # T = 0 and M = 0 (zero-size dims)
    for shape in [(2, 3, 0), (2, 0, 3), (2, 0, 0)]:
        co = np.zeros(shape)
        cn = np.array([1.0, 2.0, 3.0])[: shape[1]]
        mv = np.ones(shape[1], bool)
        kv = np.array([2, 1])
        ref = K.kbest_brute(co, cn, mv, kv, kmax)
        for backend in ("host", "jax"):
            a, c, o = K.kbest_assign(co, cn, mv, kv, kmax, backend=backend)
            assert np.asarray(a).shape == (2, kmax, shape[1])
            assert np.allclose(np.asarray(c)[0, 0], ref[1][0, 0]) and np.isinf(np.asarray(c)[:, 1:]).all()
    # kmax = 1 and tiny problems
    a, c, o = K.kbest_jax(np.array([[[0.5]]]), np.array([1.0]), np.array([True]), np.array([1]), 1)
    assert int(a[0, 0, 0]) == 0 and abs(float(c[0, 0]) - 0.5) < 1e-15


@pytest.mark.parametrize("backend", ["host", "jax"])
def test_jit_and_vmap_over_worlds(backend):
    rng = np.random.default_rng(11)
    W = 3
    probs = [rand_problem(rng, G, M, T, KMAX) for _ in range(W)]
    co = np.stack([p[0] for p in probs])
    cn = np.stack([p[1] for p in probs])
    mv = np.stack([p[2] for p in probs])
    kv = np.stack([p[3] for p in probs])
    f = lambda a, b, c, d: K.kbest_assign(a, b, c, d, KMAX, backend=backend)
    refs = [K.kbest_brute(*p, KMAX) for p in probs]
    # jit, single world
    for w in range(W):
        compare(refs[w], jax.jit(f)(co[w], cn[w], mv[w], kv[w]))
    # jit(vmap) over the leading world axis, all inputs batched
    a, c, o = jax.jit(jax.vmap(f))(co, cn, mv, kv)
    assert a.shape == (W, G, KMAX, M) and c.shape == (W, G, KMAX) and o.shape == (W, G)
    for w in range(W):
        compare(refs[w], (a[w], c[w]))
    # vmap with only some inputs batched (shared cost_new / mvalid)
    a, c, o = jax.vmap(f, in_axes=(0, None, None, 0))(co, cn[0], mv[0], kv)
    refs0 = [K.kbest_brute(co[w], cn[0], mv[0], kv[w], KMAX) for w in range(W)]
    for w in range(W):
        compare(refs0[w], (a[w], c[w]))
    # nested vmap and eager (no jit) call
    a, c, o = jax.vmap(jax.vmap(f))(co[None], cn[None], mv[None], kv[None])
    compare(refs[1], (a[0, 1], c[0, 1]))
    compare(refs[0], f(co[0], cn[0], mv[0], kv[0]))


def test_overflow_flag():
    rng = np.random.default_rng(13)
    n_flag = n_exact = 0
    for qcap in (2, 4, 8):
        fj = jax.jit(lambda a, b, c, d, q=qcap: K.kbest_jax(a, b, c, d, KMAX, queue_cap=q))
        for it in range(20):
            co, cn, mv, kv = rand_problem(rng, G, M, T, KMAX, p_inf=0.2, p_valid=1.0)
            ref = K.kbest_brute(co, cn, mv, kv, KMAX)
            a, c, o = fj(co, cn, mv, kv)
            o = np.asarray(o)
            for g in range(G):
                if o[g]:
                    n_flag += 1
                else:  # no overflow flag => proven exact
                    n_exact += 1
                    compare((ref[0][g:g + 1], ref[1][g:g + 1]), (np.asarray(a)[g:g + 1], np.asarray(c)[g:g + 1]))
    assert n_flag > 50 and n_exact > 50  # the flag fires, and is not always on
    # a tiny queue on a deep problem must flag; a big one must not
    co = np.array([[[0.0, 1.0, 2.0, 3.0]] * 6])
    cn = np.full(6, 4.0)
    mv = np.ones(6, bool)
    kv = np.array([30])
    assert bool(K.kbest_jax(co, cn, mv, kv, 30, queue_cap=2)[2][0])
    assert not bool(K.kbest_jax(co, cn, mv, kv, 30, queue_cap=256)[2][0])
    # iteration cap
    assert bool(K.kbest_jax(co, cn, mv, kv, 30, max_iter=3)[2][0])
    # amax truncation
    assert bool(K.kbest_jax(co, cn, mv, kv, 30, amax=2)[2][0])
    # kvec = 0 never overflows
    assert not bool(K.kbest_jax(co, cn, mv, np.array([0]), 30, queue_cap=2)[2][0])


def test_bad_args():
    co, cn, mv, kv = np.zeros((1, 1, 1)), np.zeros(1), np.ones(1, bool), np.array([1])
    with pytest.raises(ValueError):
        K.kbest_assign(co, cn, mv, kv, 2, backend="nope")
    with pytest.raises(TypeError):
        K.kbest_assign(co, cn, mv, kv, 2, backend="host", queue_cap=3)

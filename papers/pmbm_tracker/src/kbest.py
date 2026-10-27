"""k-best data association for the PMBM update (paper eq. (47), Section V-C3).

PROBLEM
    Parent global hypothesis g has M measurements (``mvalid`` marks the real ones, the rest is padding) and T old
    tracks.  Every valid measurement m is assigned to exactly one column: its private 'new' column (cost
    ``cost_new[m]``, always available) or an old track t (cost ``cost_old[g,m,t]``, +inf = not gated); each old
    track is used by at most one measurement.  This is the Murty matrix [C_old | diag(cost_new)] of eq.(47).
    Padding measurements are forced to 'new' with cost 0 and ignored.  The k_g = kvec[g] cheapest assignments
    are returned in ascending cost order.  Costs may be negative (float64).

STRUCTURE (shared by the host and the JAX solver)
    * Measurements with no finite ``cost_old`` entry are 'fixed' (always new); their cost_new is a constant
      ``base`` added to every assignment.  Only the 'active' measurements branch; they are the tree LEVELS.
    * Per level the options (finite old tracks + 'new') are sorted by cost (stable: track index, new last).
    * Best-first (A*) search over the search tree: a node = a prefix assignment of levels 0..d-1.
      f(node) = (cost of the prefix) + h[d], h[d] = sum over levels >= d of the level's cheapest option
      (admissible + consistent: it ignores the 'each track once' constraint).  Nodes are generated with the
      Lawler / next-sibling trick: popping a node pushes (i) its first FEASIBLE child (level d, lowest-cost
      option whose track is unused) and (ii) its next feasible sibling (same level, next higher-cost option).
      Hence two pushes per pop, f never decreases along an edge, and the first k popped complete nodes are
      exactly the k best assignments (the tree has no duplicates).  Ties are broken deterministically
      (host: insertion counter; jax: lowest queue slot), tie ORDER may differ between backends.

THREE FORMS behind one interface
    kbest_host_numpy  exact heapq A* in pure Python (numpy for the cost preprocessing).  Never overflows.
    kbest_brute       enumerate everything and sort -- the oracle for the tests only (small M, T).
    kbest_jax         pure-JAX lax.while_loop A* with a fixed-capacity queue, vmapped over the G parents
                      (mode='vmap', lock-step) or mapped sequentially (mode='map'), optional big/small split.
    kbest_assign      dispatcher; backend='host' wraps kbest_host_numpy in jax.pure_callback (works under
                      jit and vmap over a leading world axis), backend='jax' calls kbest_jax.

JAX BACKEND LIMITS / OVERFLOW (document)
    The returned ``overflow[g]`` is True iff the solver cannot PROVE that it returned the true k best costs:
      (1) queue full: a node is dropped (the new child, or the worst queued node it replaces).  Let
          min_drop = smallest f ever dropped.  Every completion below a dropped node costs >= its f, so the
          result is still exact iff k solutions were produced and min_drop >= cost of the k-th (ties at
          equality only permute equal-cost assignments).  Otherwise overflow.
      (2) iteration cap (``max_iter`` pops) hit before k solutions were produced and the queue is not empty.
      (3) the number of active measurements exceeds ``amax`` or a row has more finite options than ``dmax``
          (truncated tree), or (split mode) a parent outside the 'big' group has kvec > ksmall.
    When overflow is False the answer is exact (costs equal to the host/brute ones up to float round-off of
    the summation order, ~1e-13).  The flag is conservative: it can be True although the answer is right.
    Inputs with +inf cost_new on an active level: that option does not exist (fewer solutions are returned).
"""
import heapq
import itertools

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


def pm_fdt():
    """The float dtype of the AMBIENT precision: float64 when ``jax_enable_x64`` is on (the reference path, bit-identical to the paper path), float32 when it is off
    (a reduced-precision option, not used for the paper's results).  Read at trace time, so one source file serves both; every ``jnp.float64`` of the tracker layer is this call."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def pm_tiny():
    """The probability floor that replaces the literal 1e-300 (underflows to 0 in float32): 1e-300 in float64, 1e-30 in float32 (a normal float32 number)."""
    return 1e-300 if jax.config.jax_enable_x64 else 1e-30


PM_TIE_RTOL = 1.0e-5          # option not used by the paper's experiments


def pm_argmax(x, axis=-1):
    """Option not used by the paper's experiments: argmax that breaks near-ties (within PM_TIE_RTOL) toward the lowest index."""
    m = jnp.max(x, axis=axis, keepdims=True)
    near = x >= m - PM_TIE_RTOL * jnp.abs(m)
    return jnp.argmax(near, axis=axis)


def pm_nonzero(mask, size, fill_value=0):
    """``jnp.nonzero(mask, size=size, fill_value=fill_value)[0]`` for a 1-D bool ``mask``, without the scatter.  ``jnp.nonzero`` is built on ``bincount``, a scatter-add whose ids run past the end whenever
    the mask has ``size`` or more True entries; under ``--xla_gpu_deterministic_ops=true`` XLA linearises an out-of-range id onto the next row / the next vmapped batch element, so
    that element silently loses entries.  Here the True positions are the head of a stable argsort: the same indices, in the same order, with the same fill."""
    mask = jnp.asarray(mask, bool)
    n = mask.shape[0]
    order = jnp.argsort(~mask, stable=True)
    idx = order[:size] if size <= n else jnp.concatenate([order, jnp.zeros((size - n,), order.dtype)])
    return jnp.where(jnp.arange(size) < jnp.sum(mask), idx, jnp.asarray(fill_value, idx.dtype))


def pm_scatter(a, idx, v, op="set"):
    """``a.at[idx].<op>(v, mode="drop")``: an index entry outside ``a`` is dropped, but the scatter itself only ever sees IN-BOUNDS indices.  XLA's deterministic scatter
    (``--xla_gpu_deterministic_ops=true``, which ``rl_jax_bench.gpu_bootstrap.setup()`` sets) linearises a multi-dimensional index without a per-dimension bounds check, so an out-of-range coordinate lands on
    a valid cell of another row, or of the NEXT vmapped batch element, and overwrites it.  ``idx`` is an index array or a tuple
    of index arrays over the leading axes of ``a`` (broadcast against each other); the dropped entries are sent to a trash cell appended to every indexed axis, which is sliced off again."""
    idx = idx if isinstance(idx, tuple) else (idx,)
    n = len(idx)
    sizes = a.shape[:n]
    bad = False
    for ix, s in zip(idx, sizes):
        bad = bad | (ix < 0) | (ix >= s)
    safe = tuple(jnp.where(bad, s, ix) for ix, s in zip(idx, sizes))
    ap = jnp.pad(a, [(0, 1)] * n + [(0, 0)] * (a.ndim - n))
    ap = getattr(ap.at[safe], op)(v, mode="clip")
    return ap[tuple(slice(0, s) for s in sizes)]


def pm_map_alive(f, xs, alive, chunk):
    """Option not used by the paper's experiments (PMBMConfig.pred_batch > 0): apply ``f`` only to the elements of ``xs`` where ``alive``."""
    N = alive.shape[0]
    K = max(1, min(int(chunk), N))
    n = jnp.sum(alive.astype(jnp.int32))
    order = jnp.argsort(~alive, stable=True)
    rounds = (n + (K - 1)) // K

    def body(c):
        i, out = c
        pos = i * K + jnp.arange(K)
        idx = order[jnp.minimum(pos, N - 1)]
        part = jax.vmap(f)(jax.tree_util.tree_map(lambda a: a[idx], out))                 # (the elements of this round are not yet written: out == xs there)
        sidx = jnp.where(pos < n, idx, N)                                    # N = out of bounds -> dropped
        out = jax.tree_util.tree_map(lambda o, p: pm_scatter(o, sidx, p.astype(o.dtype)), out, part)
        return i + 1, out

    return lax.while_loop(lambda c: c[0] < rounds, body, (jnp.zeros((), jnp.int32), xs))[1]


_KB_INF = np.inf  # sentinel for 'no node' / 'no solution'
_KB_QUEUE_CAP = 256  # ASSUMED (task spec default; overflow flag makes a too-small value safe, not wrong)


# ----------------------------------------------------------------------------------------------------------
# (a) host: heapq best-first search
# ----------------------------------------------------------------------------------------------------------
def _kb_host_one(co, cn, mv, k):
    """k best assignments of ONE parent.  co[M,T] float64, cn[M], mv[M] bool, k int.
    Returns list of (cost, {m: t}) in ascending cost order, only the old-track entries listed."""
    M, T = co.shape
    fin = np.isfinite(co)
    active = mv & fin.any(axis=1)
    base = 0.0
    for m in np.nonzero(mv & ~active)[0]:
        base += float(cn[m])
    act = np.nonzero(active)[0]
    n = len(act)
    # options per level, sorted by cost (ties: lower track index first, 'new' last) -- same as jax backend
    oc, ot = [], []
    for m in act:
        ts = np.flatnonzero(fin[m])
        opts = sorted(zip(co[m, ts].tolist() + [float(cn[m])], ts.tolist() + [T]))
        opts = [(c, t) for c, t in opts if c < _KB_INF]
        oc.append([c for c, _ in opts])
        ot.append([(-1 if t == T else t) for _, t in opts])
    h = [0.0] * (n + 1)  # h[d] = sum_{l>=d} cheapest option of level l (accumulated from the back)
    for l in range(n - 1, -1, -1):
        h[l] = (oc[l][0] if oc[l] else _KB_INF) + h[l + 1]
    out = []
    if k <= 0 or not np.isfinite(h[0]):
        return out
    cnt = itertools.count()
    # node: (f, counter, gb, d, r, asg, ub)   gb = cost before the last choice, r = rank of the last choice,
    #       asg = tuple of tracks (-1 new) for levels < d, ub = bitmask of tracks used by levels < d-1
    heap = [(base + h[0], next(cnt), base, 0, -1, (), 0)]
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        f, _, gb, d, r, asg, ub = pop(heap)
        g = gb + oc[d - 1][r] if d else gb
        if d == n:
            out.append((g, asg))
            if len(out) >= k:
                break
        else:
            used = ub
            if d and asg[-1] >= 0:
                used |= 1 << asg[-1]
            o_c, o_t = oc[d], ot[d]
            for rc in range(len(o_c)):
                t = o_t[rc]
                if t < 0 or not (used >> t) & 1:
                    push(heap, (g + o_c[rc] + h[d + 1], next(cnt), g, d + 1, rc, asg + (t,), used))
                    break
        if d:
            o_c, o_t = oc[d - 1], ot[d - 1]
            for rs in range(r + 1, len(o_c)):
                t = o_t[rs]
                if t < 0 or not (ub >> t) & 1:
                    push(heap, (gb + o_c[rs] + h[d], next(cnt), gb, d, rs, asg[:-1] + (t,), ub))
                    break
    return [(c, {int(act[l]): t for l, t in enumerate(a) if t >= 0}) for c, a in out]


def kbest_host_numpy(cost_old, cost_new, mvalid, kvec, kmax):
    """Exact numpy/heapq k-best.  cost_old[G,M,T], cost_new[M], mvalid[M], kvec[G].
    Returns (assign int32[G,kmax,M], cost float64[G,kmax], overflow bool[G] (always False))."""
    cost_old = np.asarray(cost_old, np.float64)
    cost_new = np.asarray(cost_new, np.float64)
    mvalid = np.asarray(mvalid, bool)
    kvec = np.asarray(kvec)
    G, M, T = cost_old.shape
    assign = -np.ones((G, kmax, M), np.int32)
    cost = np.full((G, kmax), np.inf, np.float64)
    for g in range(G):
        kg = int(min(max(int(kvec[g]), 0), kmax))
        if kg == 0:
            continue
        for i, (c, asg) in enumerate(_kb_host_one(cost_old[g], cost_new, mvalid, kg)):
            cost[g, i] = c
            for m, t in asg.items():
                assign[g, i, m] = t
    return assign, cost, np.zeros(G, bool)


# ----------------------------------------------------------------------------------------------------------
# (b) brute force oracle (tests only)
# ----------------------------------------------------------------------------------------------------------
def kbest_brute(cost_old, cost_new, mvalid, kvec, kmax):
    """Enumerate ALL assignments and sort (stable, ties in enumeration order).  Tests only: exponential."""
    cost_old = np.asarray(cost_old, np.float64)
    cost_new = np.asarray(cost_new, np.float64)
    mvalid = np.asarray(mvalid, bool)
    kvec = np.asarray(kvec)
    G, M, T = cost_old.shape
    assign = -np.ones((G, kmax, M), np.int32)
    cost = np.full((G, kmax), np.inf, np.float64)
    vm = [m for m in range(M) if mvalid[m]]
    for g in range(G):
        sols = []

        def rec(i, used, cur, c):
            if i == len(vm):
                sols.append((c, tuple(cur)))
                return
            m = vm[i]
            if np.isfinite(cost_new[m]):
                rec(i + 1, used, cur + [-1], c + cost_new[m])
            for t in range(T):
                if t not in used and np.isfinite(cost_old[g, m, t]):
                    rec(i + 1, used | {t}, cur + [t], c + cost_old[g, m, t])

        rec(0, frozenset(), [], 0.0)
        sols.sort(key=lambda s: s[0])
        kg = int(min(max(int(kvec[g]), 0), kmax))
        for i, (c, cur) in enumerate(sols[:kg]):
            cost[g, i] = c
            for j, m in enumerate(vm):
                assign[g, i, m] = cur[j]
    return assign, cost, np.zeros(G, bool)


# ----------------------------------------------------------------------------------------------------------
# (c) JAX: fixed-capacity-queue A*
# ----------------------------------------------------------------------------------------------------------
def _kb_default_max_iter(kmax, A):
    # ASSUMED: pops per returned solution are O(depth+1) in the worst case; measured typically far fewer
    # (benchmark in the task report).  A too-small value only produces overflow=True, never a wrong answer.
    return int(4 * kmax * (A + 1) + 64)


def _kb_used(asg, lim, T):
    """bool[T+1] mask of tracks used by levels < lim (slot T is a dump for 'new')."""
    lv = jnp.arange(asg.shape[0])
    idx = jnp.where((lv < lim) & (asg >= 0), asg, T)
    return jnp.zeros(T + 1, bool).at[idx].set(True, mode="clip")


def _kb_one(co, cn, mv, k, kmax, Q, max_iter, A, D):
    """One parent.  co[M,T], cn[M], mv[M], k scalar int.  Returns (assign[kmax,M], cost[kmax], overflow)."""
    M, T = co.shape
    inf = jnp.inf
    k = jnp.clip(k, 0, kmax).astype(jnp.int32)
    fin = jnp.isfinite(co)
    active = mv & fin.any(axis=1)
    base = jnp.sum(jnp.where(mv & ~active, cn, 0.0))
    n_true = jnp.sum(active).astype(jnp.int32)
    n = jnp.minimum(n_true, A)
    ar = jnp.arange(M)
    order = jnp.argsort(jnp.where(active, ar, M + ar), stable=True)[:A]  # active measurements first, index order
    lvl_ok = jnp.arange(A) < n
    full = jnp.concatenate([co[order], cn[order][:, None]], axis=1)  # [A, T+1], column T = new
    full = jnp.where(lvl_ok[:, None], full, inf)
    sidx = jnp.argsort(full, axis=1, stable=True)[:, :D]
    opt_c = jnp.take_along_axis(full, sidx, axis=1)  # [A, D] ascending, inf = no option
    opt_t = jnp.where(sidx == T, -1, sidx).astype(jnp.int32)
    nfin = jnp.sum(jnp.isfinite(full), axis=1)
    trunc = (n_true > A) | jnp.any(nfin > D)
    m_l = jnp.where(lvl_ok, opt_c[:, 0], 0.0)
    h = lax.cumsum(m_l, reverse=True)  # h[d] = sum_{l>=d} m_l (accumulated from the back)
    h = jnp.concatenate([h, jnp.zeros(1)])  # [A+1]

    qf = jnp.full(Q, inf).at[0].set(base + h[0], mode="clip")
    qgb = jnp.zeros(Q).at[0].set(base, mode="clip")
    qd = jnp.zeros(Q, jnp.int32)
    qr = jnp.zeros(Q, jnp.int32)
    qa = -jnp.ones((Q, A), jnp.int32)
    o_cost = jnp.full(kmax, inf)
    o_asg = -jnp.ones((kmax, A), jnp.int32)
    st0 = (jnp.int32(0), jnp.int32(0), qf, qgb, qd, qr, qa, o_cost, o_asg, jnp.asarray(inf, pm_fdt()))

    def cond(st):
        it, n_out, qf = st[0], st[1], st[2]
        return (n_out < k) & (it < max_iter) & jnp.any(jnp.isfinite(qf))

    def push(qs, ok, f, gb_, d_, r_, a_, min_drop):
        qf, qgb, qd, qr, qa = qs
        free = ~jnp.isfinite(qf)
        has_free = jnp.any(free)
        s_free = jnp.argmax(free)
        worst = jnp.argmax(jnp.where(free, -inf, qf))
        w_f = qf[worst]
        rep = ok & ~has_free & (f < w_f)
        slot = jnp.where(has_free, s_free, worst)
        write = ok & (has_free | rep)
        dropped = jnp.where(ok & ~has_free, jnp.where(rep, w_f, f), inf)
        min_drop = jnp.minimum(min_drop, dropped)
        ix = jnp.where(write, slot, Q)  # Q = out of bounds -> dropped by mode='drop'
        qf = pm_scatter(qf, ix, f)
        qgb = pm_scatter(qgb, ix, gb_)
        qd = pm_scatter(qd, ix, d_)
        qr = pm_scatter(qr, ix, r_)
        qa = pm_scatter(qa, ix, a_)
        return (qf, qgb, qd, qr, qa), min_drop

    def body(st):
        it, n_out, qf, qgb, qd, qr, qa, o_cost, o_asg, min_drop = st
        i = jnp.argmin(qf)
        d, r, gb, asg = qd[i], qr[i], qgb[i], qa[i]
        dm1 = jnp.maximum(d - 1, 0)
        g = jnp.where(d > 0, gb + opt_c[dm1, r], gb)
        complete = d == n
        oi = jnp.where(complete, n_out, kmax)
        o_cost = pm_scatter(o_cost, oi, g)
        o_asg = pm_scatter(o_asg, oi, asg)
        n_out = n_out + complete.astype(jnp.int32)
        qf = qf.at[i].set(inf, mode="clip")
        # child: first feasible option of level d given tracks used by levels < d
        dc = jnp.minimum(d, A - 1)
        used = _kb_used(asg, d, T)
        okc = jnp.isfinite(opt_c[dc]) & ((opt_t[dc] < 0) | ~used[jnp.maximum(opt_t[dc], 0)])
        c_exists = ~complete & jnp.any(okc)
        rc = jnp.argmax(okc).astype(jnp.int32)
        cf = g + opt_c[dc, rc] + h[jnp.minimum(d + 1, A)]
        c_asg = asg.at[dc].set(opt_t[dc, rc], mode="clip")
        # sibling: next feasible option of level d-1 (rank > r) given tracks used by levels < d-1
        used_p = _kb_used(asg, d - 1, T)
        oks = (jnp.isfinite(opt_c[dm1]) & ((opt_t[dm1] < 0) | ~used_p[jnp.maximum(opt_t[dm1], 0)])
               & (jnp.arange(D) > r))
        s_exists = (d > 0) & jnp.any(oks)
        rs = jnp.argmax(oks).astype(jnp.int32)
        sf = gb + opt_c[dm1, rs] + h[d]
        s_asg = asg.at[dm1].set(opt_t[dm1, rs], mode="clip")
        qs = (qf, qgb, qd, qr, qa)
        qs, min_drop = push(qs, c_exists & jnp.isfinite(cf), cf, g, d + 1, rc, c_asg, min_drop)
        qs, min_drop = push(qs, s_exists & jnp.isfinite(sf), sf, gb, d, rs, s_asg, min_drop)
        qf, qgb, qd, qr, qa = qs
        return (it + 1, n_out, qf, qgb, qd, qr, qa, o_cost, o_asg, min_drop)

    it, n_out, qf, _, _, _, _, o_cost, o_asg, min_drop = lax.while_loop(cond, body, st0)
    kth = jnp.where(n_out >= k, o_cost[jnp.maximum(k - 1, 0)], inf)
    cap_hit = (n_out < k) & jnp.any(jnp.isfinite(qf))
    overflow = (k > 0) & (cap_hit | (min_drop < kth) | trunc)
    # scatter level assignments back to measurement positions
    lv_asg = jnp.where(lvl_ok[None, :], o_asg, -1)
    assign = (-jnp.ones((kmax, M), jnp.int32)).at[:, order].set(lv_asg, mode="clip")
    return assign, o_cost, overflow


def _kb_run(cost_old, cost_new, mvalid, kvec, kmax, Q, max_iter, A, D, mode):
    one = lambda co, k: _kb_one(co, cost_new, mvalid, k, kmax, Q, max_iter, A, D)
    if mode == "vmap":
        return jax.vmap(one)(cost_old, kvec)
    if mode == "map":
        return lax.map(lambda a: one(a[0], a[1]), (cost_old, kvec))
    raise ValueError(f"unknown mode {mode!r}")


def kbest_jax(cost_old, cost_new, mvalid, kvec, kmax, queue_cap=_KB_QUEUE_CAP, max_iter=None, amax=None,
              dmax=None, mode="map", split=None):
    """Pure-JAX k-best.  cost_old[G,M,T], cost_new[M], mvalid[M], kvec[G]; kmax static.
    Options (all static): queue_cap Q; max_iter pop cap (default 4*kmax*(A+1)+64); amax = max number of ACTIVE
    measurements handled (default M; smaller = much faster, overflow flagged if exceeded); dmax = max options per
    measurement (default T+1; overflow flagged if exceeded); mode 'map' (default: lax.map over the parents, no
    lock-step waste) or 'vmap' (lock-step over parents: 5-10x slower on Zipf-like kvec); split=(nbig, ksmall):
    parents are sorted by kvec, the nbig largest run with kmax, the rest only with kmax=ksmall (any of them
    with kvec>ksmall is flagged overflow); only useful with mode='vmap'.
    Returns (assign int32[G,kmax,M], cost float64[G,kmax], overflow bool[G])."""
    cost_old = jnp.asarray(cost_old, pm_fdt())
    cost_new = jnp.asarray(cost_new, pm_fdt())
    mvalid = jnp.asarray(mvalid, bool)
    kvec = jnp.asarray(kvec).astype(jnp.int32)
    G, M0, T0 = cost_old.shape
    # pad M and T to >= 1 so that no zero-size array reaches the solver (padding = invalid / +inf)
    M, T = max(M0, 1), max(T0, 1)
    if (M, T) != (M0, T0):
        cost_old = jnp.pad(cost_old, ((0, 0), (0, M - M0), (0, T - T0)), constant_values=jnp.inf)
        cost_new = jnp.pad(cost_new, (0, M - M0))
        mvalid = jnp.pad(mvalid, (0, M - M0))
    A = M if amax is None else max(1, min(int(amax), M))
    D = T + 1 if dmax is None else max(1, min(int(dmax), T + 1))
    Q = int(queue_cap)
    mi = _kb_default_max_iter(kmax, A) if max_iter is None else int(max_iter)
    if split is None or int(split[0]) >= G:
        assign, cost, ovf = _kb_run(cost_old, cost_new, mvalid, kvec, kmax, Q, mi, A, D, mode)
    else:
        nbig, ks = int(split[0]), min(int(split[1]), kmax)
        perm = jnp.argsort(-kvec, stable=True)
        big, small = perm[:nbig], perm[nbig:]
        a1, c1, o1 = _kb_run(cost_old[big], cost_new, mvalid, kvec[big], kmax, Q, mi, A, D, mode)
        mi_s = _kb_default_max_iter(ks, A) if max_iter is None else mi
        a2, c2, o2 = _kb_run(cost_old[small], cost_new, mvalid, kvec[small], ks, Q, mi_s, A, D, mode)
        o2 = o2 | (kvec[small] > ks)
        gs = G - nbig
        a2 = jnp.concatenate([a2, -jnp.ones((gs, kmax - ks, M), jnp.int32)], axis=1)
        c2 = jnp.concatenate([c2, jnp.full((gs, kmax - ks), jnp.inf)], axis=1)
        assign = jnp.zeros((G, kmax, M), jnp.int32).at[big].set(a1, mode="clip").at[small].set(a2, mode="clip")
        cost = jnp.zeros((G, kmax)).at[big].set(c1, mode="clip").at[small].set(c2, mode="clip")
        ovf = jnp.zeros(G, bool).at[big].set(o1, mode="clip").at[small].set(o2, mode="clip")
    return assign[:, :, :M0], cost, ovf


# ----------------------------------------------------------------------------------------------------------
# dispatcher
# ----------------------------------------------------------------------------------------------------------
def _kb_host_callback(cost_old, cost_new, mvalid, kvec, kmax):
    """pure_callback body with arbitrary leading batch dims (vmap_method='broadcast_all')."""
    cost_old, cost_new = np.asarray(cost_old), np.asarray(cost_new)
    mvalid, kvec = np.asarray(mvalid), np.asarray(kvec)
    lead = cost_old.shape[:-3]
    G, M, T = cost_old.shape[-3:]
    nb = int(np.prod(lead)) if lead else 1
    co = cost_old.reshape(nb, G, M, T)
    cn, mv, kv = cost_new.reshape(nb, M), mvalid.reshape(nb, M), kvec.reshape(nb, G)
    assign = np.empty((nb, G, kmax, M), np.int32)
    cost = np.empty((nb, G, kmax), np.float64)
    for b in range(nb):
        a, c, _ = kbest_host_numpy(co[b], cn[b], mv[b], kv[b], kmax)
        assign[b], cost[b] = a, c
    return assign.reshape(lead + (G, kmax, M)), cost.reshape(lead + (G, kmax))


def kbest_assign(cost_old, cost_new, mvalid, kvec, kmax, backend="host", **opts):
    """k-best assignments per parent.  See the module docstring.

    cost_old[G,M,T] float64 (+inf = not gated), cost_new[M], mvalid[M] bool, kvec[G] int, kmax static int.
    Returns (assign int32[G,kmax,M], cost float64[G,kmax], overflow bool[G]).  jit- and vmap-compatible
    (vmap over a leading world axis of all inputs) for both backends.  backend='jax' options: see kbest_jax;
    backend='host' takes none."""
    kmax = int(kmax)
    if backend == "jax":
        return kbest_jax(cost_old, cost_new, mvalid, kvec, kmax, **opts)
    if backend != "host":
        raise ValueError(f"unknown backend {backend!r}")
    if opts:
        raise TypeError(f"host backend takes no options, got {sorted(opts)}")
    cost_old = jnp.asarray(cost_old, pm_fdt())
    cost_new = jnp.asarray(cost_new, pm_fdt())
    mvalid = jnp.asarray(mvalid, bool)
    kvec = jnp.asarray(kvec).astype(jnp.int32)
    G, M, T = cost_old.shape
    shapes = (jax.ShapeDtypeStruct((G, kmax, M), jnp.int32), jax.ShapeDtypeStruct((G, kmax), pm_fdt()))
    assign, cost = jax.pure_callback(
        lambda a, b, c, d: _kb_host_callback(a, b, c, d, kmax), shapes, cost_old, cost_new, mvalid, kvec,
        vmap_method="broadcast_all")
    return assign, cost, jnp.zeros((G,), bool)

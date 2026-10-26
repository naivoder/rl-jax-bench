"""Theory checks (CPU, one process).  The paper claims PCLI1 and PCLI2 'hold' from plots and never checks PCLI3; this script tests, numerically,

  A  PCLI1 (g > 0) and PCLI2 (mp* non-decreasing, continuous, bounded below) on a FINE sweep Q^CT = 2..40, both types, P grid 1e-2 on (0, 15]
  B  the rollout MP index against an independent route: Whittle's index by INDIFFERENCE lambda of the grid value iteration of the single-target
     problem (bound.py tables), plus a direct test of Whittle-indexability (the passive set is nested in lambda)
  C  PCLI3 per jump (src/pcl.py, exact walk over the jump points of z -> (F,G)), at tau = 100 and tau = 400
  E  index policy vs the EXACT optimum (enumeration of all open-loop sequences; deterministic dynamics => open loop is optimal) on N = 2, 3, 4 targets

    python scripts/theory_checks.py [A] [B] [C] [E]      -> runs/full_theory/{A,B,C,E}.json   (default: all)
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import jax, jax.numpy as jnp, numpy as np
from rl_jax_bench import jaxlog
import model as M, algorithm as A, data as D, bound as B_, oracle as O, pcl
from rl_jax_bench.stats import mean_se, paired_compare

OUT = os.path.join(HERE, "..", "runs", "full_theory"); os.makedirs(OUT, exist_ok=True)
TYPES = ("reckless", "cautious")


def dump(name, obj):
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, default=float)


# ------------------------------------------------------------------------------------------------------------------ A
def part_A():
    Ps = np.round(np.arange(1, 1501) * 0.01, 10)                 # (0, 15] step 1e-2 (the paper's grid)
    n = len(Ps); P = jnp.asarray(Ps)[:, None, None]
    rows, tot = [], dict(n_curves=0, pcli1_viol_z_eq_P=0, pcli1_viol_fixed_z=0, pcli2_decreases=0, mp_min=np.inf, worst_rel_drop=0.0)
    for typ in TYPES:
        for q in range(2, 41):
            b = M.scalar_bank([typ] * n, [float(q)] * n, [1.0] * n)
            f, g = A.fg_marginal(b, P, M.stat(P), 100, 0.9)
            f, g = np.asarray(f), np.asarray(g); mp = f / g
            viol_g = int((g <= 1e-12).sum())
            dec = np.diff(mp); n_dec = int((dec < -1e-9).sum())
            rel_drop = float(np.max(np.maximum(-dec, 0) / np.maximum(mp[:-1], 1e-12)))
            jump = float(np.max(np.abs(dec) / np.maximum(np.abs(mp[1:]), 1e-12)))   # continuity proxy: largest neighbour change relative to the value
            vf = 0
            for z in (1.0, 2.0, 4.0, 10.0):                       # g(P, z) at fixed thresholds, as in Figs 2-3
                _, gz = A.fg_marginal(b, P, jnp.full(n, z), 100, 0.9)
                vf += int((np.asarray(gz) <= 1e-12).sum())
            rows.append(dict(type=typ, Q=q, min_g=float(g.min()), n_viol_g=viol_g, n_viol_g_fixed_z=vf, n_decreases=n_dec, worst_rel_drop=rel_drop,
                             max_neighbour_rel_change=jump, mp_min=float(mp.min()), mp_max=float(mp.max())))
            tot["n_curves"] += 1; tot["pcli1_viol_z_eq_P"] += viol_g; tot["pcli1_viol_fixed_z"] += vf; tot["pcli2_decreases"] += n_dec
            tot["mp_min"] = min(tot["mp_min"], float(mp.min())); tot["worst_rel_drop"] = max(tot["worst_rel_drop"], rel_drop)
    tot["n_points_per_curve"] = n
    bad = [r for r in rows if r["n_viol_g"] or r["n_decreases"] or r["n_viol_g_fixed_z"]]
    tot["curves_with_any_violation"] = [(r["type"], r["Q"], r["n_viol_g"], r["n_viol_g_fixed_z"], r["n_decreases"]) for r in bad]
    jaxlog.info("A done", **{k: v for k, v in tot.items() if k != "curves_with_any_violation"}, n_bad=len(bad))
    dump("A", dict(totals=tot, rows=rows)); return tot


# ------------------------------------------------------------------------------------------------------------------ B
def vi_whittle(typ, q, Pq, lam_hi=300.0, n_bis=42):
    """Whittle index by indifference: for each query state P (on the VI grid) bisect lambda where the optimal action switches active -> passive."""
    tb = B_.build_tables([(typ, float(q), 1.0)])
    grid = np.asarray(tb.grid); idx = np.array([int(np.argmin(np.abs(grid - p))) for p in Pq]); assert np.allclose(grid[idx], Pq, atol=1e-9)
    nq = len(Pq); lo = np.zeros(nq); hi = np.full(nq, lam_hi)
    v = jnp.zeros((nq, 1, grid.size))
    for _ in range(n_bis):
        mid = 0.5 * (lo + hi)
        v, k, ch, a = B_.vi(tb, jnp.asarray(mid), v)
        act = np.asarray(a)[np.arange(nq), 0, idx].astype(bool)
        lo = np.where(act, mid, lo); hi = np.where(act, hi, mid)
    return 0.5 * (lo + hi)


def indexability(typ, q, n_lam=160):
    """Passive set nested in lambda?  For each grid state P (uniform part, <= 20) the greedy action must switch active -> passive at most once as lambda grows."""
    tb = B_.build_tables([(typ, float(q), 1.0)])
    lams = np.geomspace(0.01, 400.0, n_lam)
    v, k, ch, a = B_.vi(tb, jnp.asarray(lams), jnp.zeros((n_lam, 1, tb.grid.size)))
    a = np.asarray(a)[:, 0, :2000].astype(int)                  # (n_lam, G_uniform)
    switches_back = (np.diff(a, axis=0) > 0).sum(axis=0)           # passive -> active when lambda INCREASES = non-nested
    return dict(n_states=int(a.shape[1]), n_states_not_nested=int((switches_back > 0).sum()), n_lambda=n_lam,
                frac_active_at_lam_min=float(a[0].mean()), frac_active_at_lam_max=float(a[-1].mean()))


def part_B():
    Pq = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0])
    rows = []
    for typ in TYPES:
        for q in (2, 4, 10, 40):
            t0 = time.time()
            w = vi_whittle(typ, q, Pq)
            b = M.scalar_bank([typ] * len(Pq), [float(q)] * len(Pq), [1.0] * len(Pq))
            mp, nb = A.index_whittle(b, jnp.asarray(Pq)[:, None, None], A.Cfg(guard=False))
            mp = np.asarray(mp); rel = np.abs(w - mp) / mp
            ix = indexability(typ, q)
            rows.append(dict(type=typ, Q=q, P=Pq.tolist(), vi_whittle=w.tolist(), rollout_mp=mp.tolist(), max_rel=float(rel.max()), rel=rel.tolist(), indexability=ix, seconds=time.time() - t0))
            jaxlog.info("B", type=typ, Q=q, max_rel=float(rel.max()), not_nested=ix["n_states_not_nested"], s=round(time.time() - t0, 1))
    dump("B", dict(rows=rows, tol=1e-3, worst=max(r["max_rel"] for r in rows))); return rows


# ------------------------------------------------------------------------------------------------------------------ C
def part_C():
    rows = []
    cases = [(typ, q, P0) for typ in TYPES for q in (4.0, 10.0) for P0 in (0.5, 2.0, 8.0)]
    for tau in (100, 400):
        for typ, q, P0 in (cases if tau == 100 else cases[::2]):
            u = M.TYPE_U[typ]
            t = O.scalar_tgt(u[0], u[1], q)
            t0 = time.time()
            r = pcl.pcli3_residual(t, np.array([[P0]]), 1e-3, 60.0, tau=tau)
            js = r["jumps"]
            scale = max((abs(j["dF"]) for j in js), default=1.0)
            resolved = [j for j in js if abs(j["dF"]) > 1e-6 * max(scale, 1.0) and abs(j["mp_dG"]) > 1e-6 * max(scale, 1.0)]
            rows.append(dict(type=typ, Q=q, P0=P0, tau=tau, n_jumps=len(js), n_resolved=len(resolved), max_rel_jump_resolved=max([j["rel"] for j in resolved], default=0.0),
                             median_rel_jump_resolved=float(np.median([j["rel"] for j in resolved])) if resolved else 0.0, total_rel=r["total_rel"], seconds=time.time() - t0))
            jaxlog.info("C", **{k: v for k, v in rows[-1].items() if k != "seconds"})
    dump("C", dict(rows=rows)); return rows


# ------------------------------------------------------------------------------------------------------------------ E
EXACT_CASES = {   # name: (types, qct, d, K, T)
    "n2_k1_T10": (("reckless", "cautious"), (4.0, 4.0), (5.0, 1.0), 1, 10),
    "n3_k1_T8": (("reckless", "cautious", "reckless"), (2.0, 4.0, 6.0), (5.0, 1.0, 1.0), 1, 8),
    "n4_k2_T6": (("reckless", "reckless", "cautious", "cautious"), (2.0, 5.0, 2.0, 5.0), (5.0, 5.0, 1.0, 1.0), 2, 6),
}


def part_E(nrep=40):
    out = {}
    for name, (types, qct, d, K, T) in EXACT_CASES.items():
        bank = M.scalar_bank(list(types), list(qct), list(d)); N = len(types)
        rk = D.rep_keys(0, "X2", 50 + len(types), K, nrep); p0k, tie, mode = D.split_rep_keys(rk)
        P0 = jnp.maximum(jax.vmap(lambda k: jax.random.uniform(k, (N,), minval=0.0, maxval=2.0))(p0k), 1e-12)[..., None, None]
        cfg = A.Cfg(T=T, guard=False)
        J = {pol: np.asarray(A.simulate(bank, P0, tie, mode, pol, K, cfg)["J"]) for pol in ("whittle", "myopic", "tev", "passive_cont_revisit", "rollout1")}
        Jx = np.array([A.exact_enumerate(bank, P0[r], T, K)[0] for r in range(nrep)])
        gap = {p: 100 * (J[p] - Jx) / Jx for p in J}
        o = dict(T=T, N=N, K=K, nrep=nrep, J_exact=mean_se(Jx), regret_pct={p: mean_se(gap[p]) for p in gap},
                 n_index_optimal=int((np.abs(J["whittle"] - Jx) <= 1e-9 * Jx).sum()),
                 index_vs_myopic=paired_compare(J["whittle"], J["myopic"]), index_vs_tev=paired_compare(J["whittle"], J["tev"]),
                 min_regret_ok=bool(all(float((J[p] - Jx).min()) >= -1e-9 * Jx.max() for p in J)))
        out[name] = o
        jaxlog.info("E", case=name, exact=o["J_exact"][0], regret_index=o["regret_pct"]["whittle"][0], regret_myopic=o["regret_pct"]["myopic"][0],
                    n_opt=o["n_index_optimal"], sane=o["min_regret_ok"])
    dump("E", out); return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["A", "B", "C", "E"]
    jaxlog.check_device()
    for w in which:
        {"A": part_A, "B": part_B, "C": part_C, "E": part_E}[w]()

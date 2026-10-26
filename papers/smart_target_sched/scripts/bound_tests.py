"""Hypothesis tests for the Figs 5-6 gap-LEVEL offset (ours +3.7..4.5 pp above the authors' exact points, a policy-independent multiplicative scale
(1+gap_ours)/(1+gap_authors) = 1.0325 / 1.0341 / 1.0401 / 1.0367 for Fig 5a / 5b / 5c / 6, flat in K).  The authors' implied V is  V_auth = J_ours / (1 + gap_authors/100)
(assuming our absolute J, which Tables I-III and Figs 7-8 verify at <= 1 %).  Two hypotheses for the paper's unstated V^D computation, each declared BEFORE the run:

  T1  per-slot multipliers: the relaxation constrains the expected number of tracked targets at EVERY slot (sum_n E a_{n,t} <= K, t = 0..T-1) instead of the
      discounted time-average of eq (12).  Still a valid lower bound of the original problem and tighter than V^D.  Dual optimum by column generation
      (Dantzig-Wolfe): master LP over open-loop action sequences per target, pricing = finite-horizon DP on the P grid with slot prices.
      PREDICTION if true: V_perslot reproduces V_auth in all four scenarios within 0.5 %.
  T2  coarse state grid: the authors' value iteration runs on a coarser P grid (linear interpolation of the convex value function overestimates it, so
      the 'bound' would lie ABOVE the true relaxed optimum -- consistent with V_auth exceeding the cost of a feasible policy).
      PREDICTION if true: ONE grid step reproduces V_auth in all four scenarios within 0.5 %.

    python scripts/bound_tests.py T2 [T1]   -> runs/full_theory/bound_T{1,2}.json
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import jax, jax.numpy as jnp, numpy as np
from rl_jax_bench import jaxlog
import model as M, algorithm as A, data as D, bound as B
import eval as E

OUT = os.path.join(HERE, "..", "runs", "full_theory"); os.makedirs(OUT, exist_ok=True)
CURVES = np.load(os.path.join(HERE, "..", "tests", "fixtures", "paper_fig_curves.npz"))
SCEN = {"fig5a": ("Bound_gap_rec", "E6"), "fig5b": ("Bound_gap_cau", "E6"), "fig5c": ("Bound_gap_rec_cau", "E6"), "fig6": ("Bound_gap_rec_cau_ratio", "E7")}
ACTS = {}
ACTS_J = {}
NREP = 10          # replications for the P0-dependent Fig 6 (the bound is per replication; ratio of means, as in the report)


def setup(fig, K):
    scn = D.figure_scenario(fig, K)
    key, exp = SCEN[fig]
    rk = D.rep_keys(0, exp, scn.sid, K, NREP); p0k, tie, mode = D.split_rep_keys(rk)
    P0 = D.sample_P0(scn, p0k); bank = D.build_bank(scn)
    outs = {pol: A.simulate(bank, P0, tie, mode, pol, K, A.Cfg(guard=False)) for pol in ("whittle", "myopic", "tev")}
    ACTS_J[fig] = np.asarray(outs['whittle']['J'])
    out = outs["whittle"]; ACTS[fig] = {pol: np.asarray(o["acts"]) for pol, o in outs.items()}      # (T,R,N) feasible action sequences = initial columns
    J = float(np.asarray(out["J"]).mean())
    xs = CURVES[f"{key}__0__x"]; gap_auth = float(CURVES[f"{key}__0__y"][int(np.argmin(np.abs(xs - K)))]); assert abs(xs[int(np.argmin(np.abs(xs - K)))] - K) < 0.05
    return scn, np.asarray(P0)[:, :, 0, 0], J, gap_auth


def t2():
    rows = []
    steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0]
    for fig in SCEN:
        K = 2
        scn, P0, J, gap_auth = setup(fig, K)
        V_auth = J / (1 + gap_auth / 100)
        types, qct, d = list(scn.types), list(scn.qct), list(scn.d)
        const = bool(np.all(P0 == P0[0])); Pq = P0[:1] if const else P0
        for st in steps:
            B.GRID_STEP = st
            a = B.vi_dual(types, qct, d, Pq, K)
            V = float(np.asarray(a["V_D"]).mean())
            rows.append(dict(fig=fig, K=K, step=st, V=V, V_auth=V_auth, J_ours=J, gap_auth=gap_auth, gap_with_this_V=100 * (J - V) / V, ratio_V_over_Vauth=V / V_auth))
            jaxlog.info("T2", fig=fig, step=st, V=round(V, 3), V_auth=round(V_auth, 3), gap_if=round(100 * (J - V) / V, 3), gap_auth=round(gap_auth, 3))
        B.GRID_STEP = 0.01
    json.dump(rows, open(os.path.join(OUT, "bound_T2.json"), "w"), indent=1)
    return rows


# ---------------------------------------------------------------------------------------------------------------- T1
def best_response(tb, bank_u, q, P0, T, beta):
    """Finite-horizon DP v_t(P) = min_a { d P + q_t a + beta v_{t+1}(phi^a(P)) } on the grid (numpy/jax), then the greedy open-loop sequence from the EXACT P0 by exact forward simulation.
    Returns (lagrangian value v_0(P0), exact cost c, action vector a (T,))."""
    c, j, w, out0, jo, wo, co = [np.asarray(x) for x in (tb.c, tb.j, tb.w, tb.out0, tb.jo, tb.wo, tb.co)]
    G = c.shape[1]
    v = np.zeros((1, G)); acts = np.zeros((T, G), np.int8)
    for t in range(T - 1, -1, -1):
        interp = lambda jj, ww: (1 - ww) * np.take_along_axis(v, jj, 1) + ww * np.take_along_axis(v, jj + 1, 1)
        q1 = c + q[t] + beta * interp(j[1], w[1])
        vout = co + q[t] + beta * interp(jo, wo)
        q0 = c + beta * np.where(out0, vout, interp(j[0], w[0]))
        acts[t] = (q1 < q0)[0]; v = np.minimum(q0, q1)
    g = np.asarray(tb.grid)
    k = int(np.clip(np.searchsorted(g, P0, side="right") - 1, 0, G - 2)); wt = (P0 - g[k]) / (g[k + 1] - g[k]); v0 = (1 - wt) * v[0, k] + wt * v[0, k + 1]
    P = jnp.array([[[P0]]]); a_seq = np.zeros(T); cost = 0.0
    for t in range(T):
        kk = int(np.clip(np.searchsorted(g, float(P[0, 0, 0]), side="right") - 1, 0, G - 1)); a = float(acts[t, kk]); a_seq[t] = a
        cost += beta ** t * float(bank_u.d[0]) * float(P[0, 0, 0]); P = M.phi(bank_u, P, jnp.array([a]))
    return float(v0), cost, a_seq


class Tab:
    """Compact single-target tables on a grid (uniform 0.01 to 20, then geometric ratio 1.01 to 2000).  A passive move that leaves the grid is DISALLOWED (the
    policy must track there): states above 2000 cost > 1e3 per slot and no optimal policy goes near them (max visited <= 100), so this is immaterial; it keeps the
    adjoint (forward mass transport) exactly consistent with the backward recursion."""
    def __init__(self, typ, q, d):
        uni = np.arange(0.01, 20.0 + 1e-9, 0.01)
        geo = 20.0 * 1.01 ** np.arange(1, int(np.ceil(np.log(100.0) / np.log(1.01))) + 1)
        self.g = np.concatenate([uni, geo]); G = self.g.size
        bank = M.scalar_bank([typ], [float(q)], [float(d)]); self.bank = bank; self.d = float(d)
        f = lambda a, x: np.asarray(M.phi(M.scalar_bank([typ] * len(x), [float(q)] * len(x), [float(d)] * len(x)), jnp.asarray(x)[:, None, None], jnp.full(len(x), a)))[:, 0, 0]  # noqa: E731
        x0, x1 = f(0.0, self.g), f(1.0, self.g)
        self.out0 = x0 > self.g[-1]
        def idx(x):
            j = np.clip(np.searchsorted(self.g, np.minimum(x, self.g[-1]), side="right") - 1, 0, G - 2)
            return j, (np.minimum(x, self.g[-1]) - self.g[j]) / (self.g[j + 1] - self.g[j])
        self.j0, self.w0 = idx(x0); self.j1, self.w1 = idx(x1)
        self.c = self.d * self.g


def soft_dual(tabs, P0, p, K, T, beta, temp):
    """Entropy-smoothed (temperature `temp`; temp = 0 -> hard min) per-slot dual  D(p) = sum_n v_n(P0_n; p) - K sum_t p_t  and its gradient (expected activations - K).
    soft-min <= min, so D_temp(p) <= D_0(p) <= V* for every temp: a valid lower bound at every p.  Returns (D, grad (T,), acts-profile per target)."""
    q = p / beta ** np.arange(T)
    total = -K * float(np.sum(p)); grad = np.full(T, -float(K))
    for n, tb in enumerate(tabs):
        G = tb.g.size; V = [None] * (T + 1); V[T] = np.zeros(G); Pi = np.zeros((T, G))
        for t in range(T - 1, -1, -1):
            v = V[t + 1]
            q0 = tb.c + beta * ((1 - tb.w0) * v[tb.j0] + tb.w0 * v[tb.j0 + 1]); q0 = np.where(tb.out0, np.inf, q0)
            q1 = tb.c + q[t] + beta * ((1 - tb.w1) * v[tb.j1] + tb.w1 * v[tb.j1 + 1])
            if temp > 0:
                m = np.minimum(q0, q1); a0 = np.exp(-(q0 - m) / temp); a1 = np.exp(-(q1 - m) / temp)
                V[t] = m - temp * np.log(a0 + a1); Pi[t] = a1 / (a0 + a1)
            else:
                V[t] = np.minimum(q0, q1); Pi[t] = (q1 < q0).astype(float)
        k = int(np.clip(np.searchsorted(tb.g, P0[n], side="right") - 1, 0, G - 2)); w = (P0[n] - tb.g[k]) / (tb.g[k + 1] - tb.g[k])
        total += (1 - w) * V[0][k] + w * V[0][k + 1]
        m = np.zeros(G); m[k] += 1 - w; m[k + 1] += w
        for t in range(T):                                     # forward mass transport = adjoint of the backward recursion
            grad[t] += float(np.sum(m * Pi[t]))
            nm = np.bincount(tb.j0, weights=m * (1 - Pi[t]) * (1 - tb.w0), minlength=G + 1) + np.bincount(tb.j0 + 1, weights=m * (1 - Pi[t]) * tb.w0, minlength=G + 1)
            nm += np.bincount(tb.j1, weights=m * Pi[t] * (1 - tb.w1), minlength=G + 1) + np.bincount(tb.j1 + 1, weights=m * Pi[t] * tb.w1, minlength=G + 1)
            m = nm[:G]
    return total, grad


def perslot_dual(types, qct, d, P0, K, T=100, beta=0.9, lam0=None, temps=(1.0, 0.2), maxiter=150):
    from scipy.optimize import minimize
    N = len(types)
    tabs = [Tab(types[n], qct[n], d[n]) for n in range(N)]
    p = np.full(T, 0.0) if lam0 is None else lam0 * beta ** np.arange(T)
    hard0, _ = soft_dual(tabs, P0, p, K, T, beta, 0.0)
    best = (hard0, p.copy()); trace = [("init (time-average lambda*)", hard0)]
    for temp in temps:
        def fg(x):
            v, g = soft_dual(tabs, P0, x, K, T, beta, temp)
            return -v, -g
        res = minimize(fg, p, jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * T, options=dict(maxiter=maxiter, maxfun=2 * maxiter))
        p = res.x; hard, _ = soft_dual(tabs, P0, p, K, T, beta, 0.0)
        trace.append((f"temp {temp}", hard))
        if hard > best[0]:
            best = (hard, p.copy())
        jaxlog.info("T1 stage", temp=temp, soft=round(-res.fun, 4), hard_LB=round(hard, 4), nit=res.nit)
    return dict(V_perslot_lb=best[0], trace=trace, p=best[1].tolist())


def t1():
    rows = []
    for fig in SCEN:
        K = 2
        scn, P0, J, gap_auth = setup(fig, K)
        types, qct, d = list(scn.types), list(scn.qct), list(scn.d)
        const = bool(np.all(P0 == P0[0]))
        reps = [0] if const else [0, 1, 2]
        J = float(ACTS_J[fig][reps].mean()); V_auth = J / (1 + gap_auth / 100)       # implied authors' V on the SAME replications                    # the bound is per replication: 3 replications for the P0-dependent Fig 6
        Vl, Vd, Ji = [], [], []
        t0 = time.time()
        for r in reps:
            a = B.vi_dual(types, qct, d, P0[r:r + 1], K)
            lam = float(np.asarray(a["lam"])[0]); Vd.append(float(np.asarray(a["V_D"])[0]))
            out = perslot_dual(types, qct, d, P0[r], K, lam0=lam)
            Vl.append(out["V_perslot_lb"])
            jaxlog.info("T1", fig=fig, rep=r, V_timeavg=round(Vd[-1], 3), V_perslot_lb=round(Vl[-1], 3), trace=[(t, round(v, 3)) for t, v in out["trace"]], s=round(time.time() - t0, 1))
        rows.append(dict(fig=fig, K=K, reps=reps, V_timeavg_ours=float(np.mean(Vd)), V_perslot_lb=float(np.mean(Vl)), V_auth=V_auth, J_ours=J, gap_auth=gap_auth,
                         ratio_perslot_over_auth=float(np.mean(Vl)) / V_auth, ratio_timeavg_over_auth=float(np.mean(Vd)) / V_auth, gap_if_perslot=100 * (J - float(np.mean(Vl))) / float(np.mean(Vl))))
    json.dump(rows, open(os.path.join(OUT, "bound_T1.json"), "w"), indent=1)
    return rows



def t3():
    """T3 (declared before the run): the relaxation's budget is the UNDISCOUNTED count  sum_t sum_n E a_{n,t} <= K T  (a natural reading of 'at most K targets per slot' summed over the
    horizon) instead of the discounted count of eq (12).  One multiplier lambda; price per action in discounted-cost units is lambda at EVERY slot (p_t = lambda).  PREDICTION if true:
    V_undisc reproduces V_auth in all four scenarios within 0.5 %.  Control: p_t = lambda beta^t is the discounted (eq 12) dual on the same T = 100 horizon grid = V^D up to the tail."""
    from scipy.optimize import minimize_scalar
    rows = []
    for fig in SCEN:
        K = 2
        scn, P0, J, gap_auth = setup(fig, K)
        types, qct, d = list(scn.types), list(scn.qct), list(scn.d)
        const = bool(np.all(P0 == P0[0])); reps = [0] if const else [0, 1, 2]
        J = float(ACTS_J[fig][reps].mean()); V_auth = J / (1 + gap_auth / 100)
        Vu, Vdisc, Vd = [], [], []
        for r in reps:
            tabs = [Tab(types[n], qct[n], d[n]) for n in range(len(types))]
            T, beta = 100, 0.9
            def best1(shape, lam_hi=500.0):
                f = lambda lam: -soft_dual(tabs, P0[r], lam * shape, K, T, beta, 0.0)[0]
                res = minimize_scalar(f, bounds=(0.0, lam_hi), method="bounded", options=dict(xatol=1e-4))
                return -res.fun, res.x
            vu, lu = best1(np.ones(T), 200.0); vdisc, ld = best1(beta ** np.arange(T), 500.0)
            a = B.vi_dual(types, qct, d, P0[r:r + 1], K); Vd.append(float(np.asarray(a["V_D"])[0]))
            Vu.append(vu); Vdisc.append(vdisc)
            jaxlog.info("T3", fig=fig, rep=r, V_undiscounted=round(vu, 3), lam_u=round(lu, 3), V_discounted_T100=round(vdisc, 3), V_D_infinite=round(Vd[-1], 3), V_auth_single=round(J / (1 + gap_auth / 100), 3))
        rows.append(dict(fig=fig, K=K, reps=reps, V_undiscounted=float(np.mean(Vu)), V_discounted_T100=float(np.mean(Vdisc)), V_D_infinite=float(np.mean(Vd)), V_auth=V_auth, J_ours=J, gap_auth=gap_auth,
                         ratio_undisc_over_auth=float(np.mean(Vu)) / V_auth, ratio_disc_over_auth=float(np.mean(Vdisc)) / V_auth, gap_if_undisc=100 * (J - float(np.mean(Vu))) / float(np.mean(Vu))))
    json.dump(rows, open(os.path.join(OUT, "bound_T3.json"), "w"), indent=1)
    return rows



class TabGrid:
    """Uniform grid of step `step` up to `pmax` with a state-MAPPING scheme for next states ('ceil' / 'nearest' / 'floor' map a next state to a grid node, no interpolation).  A passive move
    that leaves the grid is disallowed (forced tracking).  Same attribute interface as Tab so `soft_dual(temp=0)` runs the hard finite-horizon DP."""
    def __init__(self, typ, q, d, step, scheme, pmax=130.0):
        self.g = np.arange(step, pmax + 1e-9, step); G = self.g.size
        self.d = float(d); self.c = self.d * self.g
        f = lambda a, x: np.asarray(M.phi(M.scalar_bank([typ] * len(x), [float(q)] * len(x), [float(d)] * len(x)), jnp.asarray(x)[:, None, None], jnp.full(len(x), a)))[:, 0, 0]  # noqa: E731
        x0, x1 = f(0.0, self.g), f(1.0, self.g)
        self.out0 = x0 > self.g[-1]
        def node(x):
            x = np.minimum(x, self.g[-1]); u = x / step - 1.0          # fractional node index (node i sits at step * (i + 1))
            i = {"ceil": np.ceil(u - 1e-9), "floor": np.floor(u + 1e-9), "nearest": np.round(u)}[scheme]
            return np.clip(i, 0, G - 1).astype(int)
        def jw(x):
            i = node(x); j = np.maximum(i - 1, 0); w = np.where(i == 0, 0.0, 1.0)     # value = (1-w) v[j] + w v[j+1] = v[i]
            j = np.where(i == 0, 0, i - 1); return j, np.where(i == 0, 0.0, 1.0)
        self.j0, self.w0 = jw(x0); self.j1, self.w1 = jw(x1)
        self.node = node


def soft_dual_grid(tabs, P0, p, K, T, beta):
    """hard DP value with P0 mapped to a grid node by the SAME rule (ceil/nearest/floor); returns the dual D(p)."""
    P0m = np.array([tb.g[tb.node(np.array([P0[n]]))[0]] for n, tb in enumerate(tabs)])
    k_nodes = [int(tb.node(np.array([P0[n]]))[0]) for n, tb in enumerate(tabs)]
    # reuse soft_dual by feeding the mapped initial state exactly at a node (w = 0 interpolation)
    return soft_dual(tabs, P0m, p, K, T, beta, 0.0)[0]


def t5():
    """T5 (declared before the run): the authors' value iteration maps next states to a COARSE uniform grid without interpolation (ceil / nearest / floor to a node) and starts from the mapped P0.
    PREDICTION if true: ONE (scheme, step) reproduces V_auth within 0.5 % in all four scenarios at K = 2 AND the K-dependence of the Fig 5a offset (ratio V_auth/V^D: 1.0325 at K = 2,
    1.0185 at K = 10, 1.0152 at K = 26, 1.0059 at K = 82)."""
    from scipy.optimize import minimize_scalar
    T, beta = 100, 0.9
    rows = []
    def Vgrid(types, qct, d, P0r, K, step, scheme):
        tabs = [TabGrid(types[n], qct[n], d[n], step, scheme) for n in range(len(types))]
        f = lambda lam: -soft_dual_grid(tabs, P0r, lam * beta ** np.arange(T), K, T, beta)
        res = minimize_scalar(f, bounds=(0.0, 600.0), method="bounded", options=dict(xatol=1e-3))
        return -res.fun
    for scheme in ("ceil", "nearest", "floor"):
        for step in (0.1, 0.25, 0.5, 1.0):
            for fig in SCEN:
                K = 2
                scn, P0, J, gap_auth = setup(fig, K)
                types, qct, d = list(scn.types), list(scn.qct), list(scn.d)
                const = bool(np.all(P0 == P0[0])); reps = [0] if const else [0, 1, 2]
                J = float(ACTS_J[fig][reps].mean()); V_auth = J / (1 + gap_auth / 100)
                V = float(np.mean([Vgrid(types, qct, d, P0[r], K, step, scheme) for r in reps]))
                rows.append(dict(scheme=scheme, step=step, fig=fig, K=K, V=V, V_auth=V_auth, ratio=V / V_auth))
                jaxlog.info("T5", scheme=scheme, step=step, fig=fig, V=round(V, 2), V_auth=round(V_auth, 2), ratio=round(V / V_auth, 4))
    json.dump(rows, open(os.path.join(OUT, "bound_T5.json"), "w"), indent=1)
    return rows



def t5b():
    """T5b (declared before the run; follow-up of T5 whose 'ceil' rows were uniformly above V_auth by 1.3-1.8 % at step 0.1 and below it at 0.05): does the CEIL mapping reproduce the
    K-dependence of the Fig 5a offset?  Steps {0.05, 0.075, 0.1}; K in {2, 10, 26, 82} for fig5a; K = 2 for the other scenarios.  The offset to explain: V_auth/V^D(ours) = 1.0325 / 1.0185 / 1.0152 / 1.0059 (fig5a, K = 2 / 10 / 26 / 82),
    1.0341 (5b), 1.0401 (5c), 1.0289-1.0367 (6).  PREDICTION if the ceil grid is what the authors used: one step reproduces all within 0.3 %."""
    from scipy.optimize import minimize_scalar
    T, beta = 100, 0.9
    rows = []
    for fig, K in [("fig5a", 2), ("fig5a", 10), ("fig5a", 26), ("fig5a", 82), ("fig5b", 2), ("fig5c", 2), ("fig6", 2)]:
        scn, P0, J, gap_auth = setup(fig, K)
        types, qct, d = list(scn.types), list(scn.qct), list(scn.d)
        const = bool(np.all(P0 == P0[0])); reps = [0] if const else [0, 1, 2]
        J = float(ACTS_J[fig][reps].mean()); V_auth = J / (1 + gap_auth / 100)
        for step in (0.05, 0.075, 0.1):
            V = []
            for r in reps:
                tabs = [TabGrid(types[n], qct[n], d[n], step, "ceil") for n in range(len(types))]
                f = lambda lam: -soft_dual_grid(tabs, P0[r], lam * beta ** np.arange(T), K, T, beta)
                V.append(-minimize_scalar(f, bounds=(0.0, 800.0), method="bounded", options=dict(xatol=1e-3)).fun)
            rows.append(dict(fig=fig, K=K, step=step, V=float(np.mean(V)), V_auth=V_auth, ratio=float(np.mean(V)) / V_auth))
            jaxlog.info("T5b", fig=fig, K=K, step=step, V=round(float(np.mean(V)), 2), V_auth=round(V_auth, 2), ratio=round(float(np.mean(V)) / V_auth, 4))
    json.dump(rows, open(os.path.join(OUT, "bound_T5b.json"), "w"), indent=1)
    return rows



def t5c():
    """T5c (EXPLORATORY, post-hoc: the step 0.075 and the 'ceil' scheme were selected from T5/T5b, which scanned three schemes and steps 0.05-1): the ceil-mapped grid of step 0.075 at the other K of the
    authors' grids, plus the sensitivity to the step (0.07, 0.08) at two cells.  Reported as a consistency check of a hypothesis about the authors' unstated numerics, not as a finding about the paper."""
    from scipy.optimize import minimize_scalar
    T, beta = 100, 0.9
    path = os.path.join(OUT, "bound_T5c.json")
    rows = json.load(open(path)) if os.path.exists(path) else []          # resumable (cells already done are skipped)
    done = {(r["fig"], r["K"], r["step"]) for r in rows}
    cells = [("fig5a", K, (0.075,)) for K in (18, 34, 42, 50, 58, 66, 98)] + [("fig5a", 2, (0.07, 0.08)), ("fig5a", 26, (0.07, 0.08))] \
          + [("fig5b", K, (0.075,)) for K in (10, 66)] + [("fig5c", K, (0.075,)) for K in (10, 66)] + [("fig6", K, (0.075,)) for K in (6, 22, 46)]
    for fig, K, steps in cells:
        steps = tuple(s_ for s_ in steps if (fig, K, s_) not in done)
        if not steps:
            continue
        scn, P0, J, gap_auth = setup(fig, K)
        types, qct, d = list(scn.types), list(scn.qct), list(scn.d)
        const = bool(np.all(P0 == P0[0])); reps = [0] if const else [0, 1, 2]
        J = float(ACTS_J[fig][reps].mean()); V_auth = J / (1 + gap_auth / 100)
        for step in steps:
            V = []
            for r in reps:
                tabs = [TabGrid(types[n], qct[n], d[n], step, "ceil") for n in range(len(types))]
                f = lambda lam: -soft_dual_grid(tabs, P0[r], lam * beta ** np.arange(T), K, T, beta)
                V.append(-minimize_scalar(f, bounds=(0.0, 1200.0), method="bounded", options=dict(xatol=1e-3)).fun)
            rows.append(dict(fig=fig, K=K, step=step, V=float(np.mean(V)), V_auth=V_auth, ratio=float(np.mean(V)) / V_auth))
            jaxlog.info("T5c", fig=fig, K=K, step=step, V=round(float(np.mean(V)), 2), V_auth=round(V_auth, 2), ratio=round(float(np.mean(V)) / V_auth, 4))
            json.dump(rows, open(os.path.join(OUT, "bound_T5c.json"), "w"), indent=1)
    return rows


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["T2", "T1"]):
        {"T1": t1, "T2": t2, "T3": t3, "T5": t5, "T5b": t5b, "T5c": t5c}[w]()

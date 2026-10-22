"""Records the small fixtures of tests/test_official_parity.py FROM THE AUTHORS' CODE (read-only copy, commit b5959f6).

Run it with the OFFICIAL environment (numpy 1.26.4 / scipy 1.11.4 -- the era the code was written for; no JAX needed):
    $OFFICIAL_ROOT/.venv/bin/python scripts/official/record_official_fixtures.py   (OFFICIAL_ROOT defaults to external/official)
It imports the authors' modules (never copies them into this repo), draws random inputs from fixed seeds, and stores inputs + the authors'
outputs in tests/fixtures/official/official_fixtures.npz.  The tests then feed the same inputs to src/ and compare, without needing the
official checkout.  Generators mirror the earlier live differential tests (tests/test_planner_node_parity.py etc.)."""
import os, sys, importlib
import numpy as np

PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(PAPER, "external", "official")), "repo")
OUT = os.path.join(PAPER, "tests", "fixtures", "official", "official_fixtures.npz")
sys.path.insert(0, REPO)
import matplotlib; matplotlib.use("Agg")
names = ["Parameters", "obstacles", "utils", "lbp", "tomb", "predict", "update", "MCTS", "MCTS_joint", "sensor", "gospa"]
off = {n: importlib.import_module(n) for n in names}
P_ = off["Parameters"]
P_.obstacles_list = [off["obstacles"].obstacle_three(P_.coordD3, P_.coordB3)]
F = {}
F["meta/numpy"] = np.array(np.__version__); F["meta/commit"] = np.array("b5959f6a696a0e566e0925ec2376be361982e773")


def pad(a, n, fill=0.0):
    a = np.asarray(a, float)
    o = np.full((n,) + a.shape[1:], fill); o[: len(a)] = a
    return o


# ------------------------------------------------------------------ node (individual + joint, GD + KLD)
def rand_belief(rng, n):
    pos = rng.uniform(-60, 60, (n, 2)); P = np.zeros((n, 2, 2))
    for i in range(n):
        A = rng.normal(size=(2, 2)) * rng.choice([1.0, 3.0, 8.0]); P[i] = A @ A.T + np.diag(rng.uniform(1.0, 20.0, 2))
    return rng.uniform(0.02, 0.99, n), pos, P


def record_nodes(N=40):
    P_.simulation_search_depth = 5
    for S, seed in ((1, 11), (2, 12)):
        for kind, fn in (("GD", "GOSPA"), ("KLD", "KLD")):
            rng = np.random.default_rng(seed)           # same stream as the earlier live test (first N cases)
            rows = []
            for it in range(N):
                n = int(rng.integers(1, 9)); r, pos, P = rand_belief(rng, n)
                if S == 1:
                    s = rng.uniform(-60, 60, (2, 1)); cls = off["MCTS"].MonteCarloTreeSearchNode_individual
                    root = cls(parent=None, means_existing=pos.T.copy(), covariances_existing=P.copy(), sensor_mean=s, prob_existences=r.copy(), cost=0, cost_function=fn)
                else:
                    s = [rng.uniform(-60, 60, (2, 1)) for _ in range(2)]; cls = off["MCTS_joint"].MonteCarloTreeSearchNode_joint
                    root = cls(parent=None, means_existing=pos.T.copy(), covariances_existing=P.copy(), sensor_means=s, prob_existences=r.copy(), cost=0, cost_function=fn)
                acts = list(root.untried_actions); root.untried_actions = [acts[int(rng.integers(0, len(acts)))]]
                ch = root.expand()
                acts = list(ch.untried_actions); ch.untried_actions = [acts[int(rng.integers(0, len(acts)))]]
                g = ch.expand()
                pos_of = (lambda c: np.asarray(c.sensor_mean, float).reshape(1, 2)) if S == 1 else (lambda c: np.stack([np.asarray(q, float).reshape(2) for q in c.sensor_means]))
                rows.append(dict(n=n, r=pad(r, 8), pos=pad(pos, 8), P=pad(P, 8), s=(np.asarray(s, float).reshape(S, 2)),
                                 a1=pos_of(ch), c1=float(ch.cost), r1=pad(ch.prob_existences, 10), x1=pad(ch.means_existing.T, 10), P1=pad(ch.covariances_existing, 10),
                                 a2=pos_of(g), c2=float(g.cost), r2=pad(g.prob_existences, 10), x2=pad(g.means_existing.T, 10), P2=pad(g.covariances_existing, 10)))
            for k in rows[0]:
                F[f"node/S{S}/{kind}/{k}"] = np.stack([np.asarray(r_[k]) for r_ in rows])


# ------------------------------------------------------------------ legal actions
def record_legal(N=150):
    rng = np.random.default_rng(13); P_.simulation_search_depth = 5
    pts, L, cnt = [], [], []
    cls = off["MCTS"].MonteCarloTreeSearchNode_individual
    for it in range(N):
        kind = it % 3
        if kind == 0: p = rng.uniform(-260, 260, 2)
        elif kind == 1: p = np.array([rng.uniform(-50, 50), rng.uniform(45, 105)])
        else: p = np.array([rng.choice([-235.0, 235.0, 240.0, -245.0]) + rng.normal(), rng.uniform(-250, 250)])
        legal = cls.__new__(cls).get_legal_actions(p.reshape(2, 1))
        pts.append(p); cnt.append(len(legal)); L.append(pad(np.stack([np.asarray(a, float).reshape(2) for a in legal]) if legal else np.zeros((0, 2)), 7))
    F["legal/p"] = np.stack(pts); F["legal/count"] = np.array(cnt); F["legal/actions"] = np.stack(L)


# ------------------------------------------------------------------ filter (update + LBP + TOMB + merge + prune)
class _Sens:
    def __init__(self, p): self.current_sensor_position = np.array(p, float).reshape(2, 1)


def rand_case(rng):
    n = int(rng.integers(1, 9)); sp = [rng.uniform(-40, 40, 2), rng.uniform(-40, 40, 2)]
    pos = rng.uniform(-70, 70, (n, 2))
    for i in range(1, n):
        if rng.random() < 0.3: pos[i] = pos[i - 1] + rng.normal(size=2) * 2.0
    x = np.zeros((4, n)); x[0], x[2] = pos[:, 0], pos[:, 1]; x[1], x[3] = rng.normal(size=n) * 0.5, rng.normal(size=n) * 0.5
    P = np.zeros((n, 4, 4))
    for i in range(n):
        A = rng.normal(size=(4, 4)) * rng.choice([0.5, 2.0, 5.0]); P[i] = A @ A.T + np.diag(rng.uniform(0.5, 4.0, 4))
    r = rng.uniform(0.02, 0.99, n); zs = []
    for s in range(2):
        m = int(rng.integers(0, 6)); z = np.zeros((2, m))
        for j in range(m):
            if rng.random() < 0.6: z[:, j] = x[[0, 2], rng.integers(0, n)] + rng.normal(size=2) * 2.0
            else: z[:, j] = sp[s] + rng.uniform(-40, 40, 2)
        zs.append(z)
    return r, x, P, sp, zs


def record_filter(N=60):
    for cap in (False, True):
        rng = np.random.default_rng(7)
        rows = []
        for it in range(N):
            r, x, P, sp, zs = rand_case(rng); lam = float(rng.choice([0.1, 1.0, 2.0]))
            Pm = P_; Pm.lambda_false_alarm = lam / (np.pi * 40.0 ** 2)
            upd = off["update"].Updater(off["predict"].Predictor(), None)
            lambdau = np.zeros(Pm.lambdau.shape); mn = np.zeros(Pm.birth_means.shape); cn = np.zeros(Pm.birth_covariances.shape)
            lbp, tomb = off["lbp"].lbp, off["tomb"].tomb
            r1, x1, P1 = r.copy(), x.copy(), P.copy()
            lu, mnt, cnt_, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = upd.update_pmb_empty_poisson(lambdau, mn, cn, r1, x1, P1, zs[0], [_Sens(sp[0])])
            pupd, pnew = lbp(wupd, wnew); r1, x1, P1 = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
            r1 = r1.reshape(-1, 1)                      # main.py's own reshape (numpy 1.26 accepts it)
            lu, mn2, cn2, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = upd.update_pmb_empty_poisson(lu, mnt, cnt_, r1, x1, P1, zs[1], [_Sens(sp[1])])
            pupd, pnew = lbp(wupd, wnew); r1, x1, P1 = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
            if len(r1) > 0: r1, x1, P1 = off["utils"].merge_gaussians(r1, x1, P1)
            if cap: r1[r1 >= 1] = 1
            keep = r1 > Pm.pruning_threshold; r1, x1, P1 = r1[keep], x1[:, keep], P1[keep]
            xc = (x1[0] >= -250) & (x1[0] <= 250) & (x1[2] >= -250) & (x1[2] <= 250)
            r1, x1, P1 = r1[xc], x1[:, xc], P1[xc]
            rows.append(dict(n=len(r), r=pad(r, 8), x=pad(x.T, 8), P=pad(P, 8), sp=np.stack(sp), m=np.array([zs[0].shape[1], zs[1].shape[1]]),
                             z0=pad(zs[0].T, 5), z1=pad(zs[1].T, 5), lam=lam, no=len(r1), ro=pad(r1, 24), xo=pad(x1.T, 24), Po=pad(P1, 24)))
        for k in rows[0]:
            F[f"filter/cap{int(cap)}/{k}"] = np.stack([np.asarray(r_[k]) for r_ in rows])


# ------------------------------------------------------------------ GOSPA
def record_gospa(N=120):
    rng = np.random.default_rng(0); rows = []
    for _ in range(N):
        nt = int(rng.integers(0, 6)); ne = int(rng.integers(0, 9)); scale = rng.choice([5.0, 40.0, 120.0])
        T = rng.normal(size=(nt, 4)) * scale; E = rng.normal(size=(ne, 4)) * scale
        for i in range(min(nt, ne)):
            if rng.random() < 0.6: E[i] = T[i] + rng.normal(size=4) * rng.choice([1.0, 20.0, 70.0])
        g, _, loc, mis, fal = off["gospa"].calculate_gospa(T, E, 80.0, 2)
        rows.append(dict(nt=nt, ne=ne, T=pad(T, 5), E=pad(E, 8), ref=np.array([g ** 2, loc, mis, fal])))
    for k in rows[0]:
        F[f"gospa/{k}"] = np.stack([np.asarray(r_[k]) for r_ in rows])


# ------------------------------------------------------------------ one full MCTS decision with injected rollout uniforms
class Inj:
    it = -1; j = 0; U = None


def record_mcts():
    cases = [(1, 1, 7), (1, 5, 7), (1, 5, 40), (1, 10, 40), (2, 1, 49), (2, 5, 49), (2, 5, 200), (2, 10, 60)]
    for fn in ("GOSPA", "KLD"):
        for S, T, B in cases:
            cls = off["MCTS"].MonteCarloTreeSearchNode_individual if S == 1 else off["MCTS_joint"].MonteCarloTreeSearchNode_joint
            orig = cls.rollout
            def rollout(self, orig=orig):
                Inj.it += 1; Inj.j = 0
                return orig(self)
            def rollout_policy(self, avail):
                u = Inj.U[Inj.it, Inj.j]; Inj.j += 1
                return avail[int(np.floor(u * len(avail)))]
            cls.rollout = rollout; cls.rollout_policy = rollout_policy
            try:
                rng = np.random.default_rng(100 + 10 * S + T)
                for trial in range(3 if B >= 200 else 4):
                    n = int(rng.integers(2, 8)); pos = rng.uniform(-40, 40, (n, 2)) + np.array([0.0, 60.0 * (trial % 2)]); P = np.zeros((n, 2, 2))
                    for i in range(n):
                        A = rng.normal(size=(2, 2)) * 3.0; P[i] = A @ A.T + np.diag(rng.uniform(2.0, 15.0, 2))
                    r = rng.uniform(0.05, 0.99, n)
                    sp = [np.array([4.0, 94.0]), np.array([-6.0, 96.0])] if trial == 1 else [rng.uniform(-100, 100, 2) for _ in range(2)]
                    U = rng.uniform(size=(B, max(T - 1, 1)))
                    Inj.it, Inj.U = -1, U; P_.simulation_search_depth = T; P_.MCTS_simulation_number = B
                    kw = dict(parent=None, means_existing=pos.T.copy(), covariances_existing=P.copy(), prob_existences=r.copy(), cost=0, cost_function=fn)
                    root = cls(sensor_mean=sp[0].reshape(2, 1).copy(), **kw) if S == 1 else cls(sensor_means=[q.reshape(2, 1).copy() for q in sp], **kw)
                    best = root.best_action()
                    gp = (lambda c: np.asarray(c.sensor_mean, float).reshape(1, 2)) if S == 1 else (lambda c: np.stack([np.asarray(q, float).reshape(2) for q in c.sensor_means]))
                    pre = f"mcts/{fn}/{S}_{T}_{B}/{trial}"
                    F[pre + "/n"] = np.array(n); F[pre + "/r"] = pad(r, 8); F[pre + "/pos"] = pad(pos, 8); F[pre + "/P"] = pad(P, 8)
                    F[pre + "/sp"] = np.stack(sp); F[pre + "/U"] = U
                    F[pre + "/child_pos"] = np.stack([gp(c) for c in root.children]); F[pre + "/child_cost"] = np.array([c.cost for c in root.children])
                    F[pre + "/best"] = gp(best)
            finally:
                cls.rollout = orig; del cls.rollout_policy


# ------------------------------------------------------------------ measurement generator (distributional parity; RNG streams differ)
MEAS_SCEN = {                  # name: (alive truth positions [m,2] (None = scenario k=75 of the authors' truth), sensor position, lambda_C, origin for offsets)
    "single_d0": ([[0.0, 0.0]], [0.0, 0.0], 0.0, "target"),
    "single_d36": ([[36.0, 0.0]], [0.0, 0.0], 0.0, "target"),
    "single_d40": ([[40.0, 0.0]], [0.0, 0.0], 0.0, "target"),
    "single_d44": ([[44.0, 0.0]], [0.0, 0.0], 0.0, "target"),
    "clutter_only_lam2": ([], [3.0, -7.0], 2.0, "sensor"),
    "clutter_only_lam0p1": ([], [3.0, -7.0], 0.1, "sensor"),
    "scenario_k75_lam1": (None, None, 1.0, "sensor"),
    "scenario_k75_lam2": (None, None, 2.0, "sensor"),
}


def record_measurements(n=40000):
    sys.path.insert(0, os.path.join(PAPER, "tests"))
    import meas_stats
    truth = np.load(os.path.join(PAPER, "data", "truth_seed123546.npy"))   # [9, 4, 200]
    tk = truth[:, :, 75].T                                                       # (4, 9): the authors' target_states[:, :, k].T
    alive = ~np.isnan(tk[0]); tpos = tk[[0, 2]][:, alive].T
    SP75 = tpos[0] + np.array([12.0, -9.0])                                      # a sensor 15 m from target 0 so that several targets are in/near the FOV
    F["meas/sp75"] = SP75; F["meas/tpos75"] = tpos
    np.random.seed(2024)
    for name, (tp, sp, lam, org) in MEAS_SCEN.items():
        if tp is None:
            tstate = tk; sp = SP75
        elif len(tp) == 0:
            tstate = np.full((4, 1), np.nan)
        else:
            tstate = np.zeros((4, len(tp))); tstate[0] = [q[0] for q in tp]; tstate[2] = [q[1] for q in tp]
            tstate[1] = tstate[3] = 0.1      # NB: the authors' generate_observation returns a NOISE-FREE (0,0) for an exactly-zero state (state tests `i != 0`); avoid that corner
        P_.clutter_rate = lam
        sets = []
        for i in range(n):
            m = off["sensor"].Measurements(); m.all_measurements = [np.array([[np.nan], [np.nan]])]
            m.generate_measurements_multiple_targets(tstate, np.array(sp, float).reshape(2, 1), 0)
            sets.append(m.all_measurements[0].T.copy())
        origin = np.array(tp[0], float) if org == "target" else np.array(sp, float)
        for k_, (v, se) in meas_stats.stats(sets, origin).items():
            F[f"meas/{name}/{k_}"] = np.array([v, se])
    P_.clutter_rate = 2


if __name__ == "__main__":
    record_measurements()
    record_nodes(); record_legal(); record_filter(); record_gospa(); record_mcts()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **F)
    print("[INFO] wrote", os.path.abspath(OUT), os.path.getsize(OUT) // 1024, "kB,", len(F), "arrays")

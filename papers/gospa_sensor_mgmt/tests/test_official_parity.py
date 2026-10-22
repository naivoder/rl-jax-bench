"""Authors' code as oracle, FIXTURE form: every check passes WITHOUT the authors' checkout.

The fixtures in tests/fixtures/official/ were recorded from the authors' code (github.com/sggjon5/GOSPA-Driven-Non-Myopic-Multi-Sensor-
Management-with-Multi-Bernoulli-Filtering, commit b5959f6a696a0e566e0925ec2376be361982e773) by scripts/official/record_official_fixtures.py and
scripts/official/trace_main.py (oracle trace of the UNMODIFIED main.py; README 'Checked against').  Only the recorded inputs/outputs are committed,
never their code.  Configuration under test: CFG-O (the authors' HEAD behavior).  Tolerances (pre-declared): node/filter/MCTS
relative 1e-9, GOSPA relative 1e-12, trace replay (200 steps) sensor positions 1e-9 and per-step GOSPA^2 relative 1e-8.

  official_fixtures.npz           node cost + merged Bernoullis (S = 1, 2; GD, KLD; depth 1, 2), legal actions, PMB filter step
                                  (update + LBP + TOMB + merge + prune, cap on/off), calculate_gospa, full MCTS decisions with injected rollouts
  main_trace_seed123546_lam2.npz  the authors' main.py run itself (np.random seed 123546, lambda_C = 2): every measurement set, every rollout draw,
                                  the sensor positions it chose and its own per-step GOSPA^2, for the blocks Myopic-KLD, Myopic-GD, MCTS1, MCTS3
"""
import os
import numpy as np
import jax, jax.numpy as jnp
import pytest
from src import planner as PL
from src import filter as FL
from src import gospa_jax as G
from src import algorithm as A
from src import model as M

CFG = M.CFG_O
FD = os.path.join(os.path.dirname(__file__), "fixtures", "official")


@pytest.fixture(scope="module")
def F():
    p = os.path.join(FD, "official_fixtures.npz")
    assert os.path.exists(p), "fixture missing: run scripts/official/record_official_fixtures.py in the official venv"
    z = np.load(p)
    assert str(z["meta/commit"]).startswith("b5959f6")
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ GOSPA
def _pad_g(T, E):
    tx = np.zeros((M.K_TRUTH, 4)); tm = np.zeros(M.K_TRUTH, bool); ex = np.zeros((M.E_MAX, 4)); em = np.zeros(M.E_MAX, bool)
    tx[: len(T)] = T; tm[: len(T)] = True; ex[: len(E)] = E; em[: len(E)] = True
    return tx, tm, ex, em


def test_gospa_matches_official_fixture(F):
    f = jax.jit(lambda tx, tm, ex, em: G.gospa_parts(tx, tm, ex, em, 80.0))
    worst = 0.0
    for i in range(len(F["gospa/nt"])):
        nt, ne = int(F["gospa/nt"][i]), int(F["gospa/ne"][i])
        out = [float(v) for v in f(*_pad_g(F["gospa/T"][i][:nt], F["gospa/E"][i][:ne]))]
        for a, b in zip(out, F["gospa/ref"][i]):
            worst = max(worst, abs(a - b) / max(1.0, abs(b)))
    print(f"[METRIC] official fixture gospa: {len(F['gospa/nt'])} cases, worst rel err = {worst:.2e}")
    assert worst < 1e-12, worst


# ------------------------------------------------------------------ legal actions
def test_legal_actions_match_official_fixture(F):
    offs = jnp.asarray(M.action_offsets())
    for i in range(len(F["legal/count"])):
        p = F["legal/p"][i]
        feas = np.asarray(PL.feas_sensor(jnp.asarray(p), offs)); cand = p[None, :] + np.asarray(offs)
        exp = [cand[j] for j in range(7) if feas[j]]
        assert len(exp) == int(F["legal/count"][i]), (i, p)
        for j, b in enumerate(exp):
            assert np.allclose(F["legal/actions"][i][j], b, atol=1e-12)


# ------------------------------------------------------------------ node (cost + merged Bernoullis)
NP = 24


def _belief(r, pos, P, n):
    return PL.Belief(jnp.zeros(NP).at[:n].set(r[:n]), jnp.zeros((NP, 2)).at[:n].set(pos[:n]),
                     jnp.tile(jnp.eye(2), (NP, 1, 1)).at[:n].set(P[:n]), jnp.asarray(n, jnp.int32))


def _cmp(c_o, r_o, x_o, P_o, c_m, b, n):
    rm, xm, Pm = np.asarray(b.r)[:n], np.asarray(b.x)[:n], np.asarray(b.P)[:n]
    return max(abs(c_o - float(c_m)) / max(1.0, abs(c_o)), np.max(np.abs(r_o[:n] - rm)),
               np.max(np.abs(x_o[:n] - xm) / (1 + np.abs(xm))), np.max(np.abs(P_o[:n] - Pm) / (1 + np.abs(Pm))))


def _node_worst(F, S, kind, cfg, ncase=None):
    pc = PL.PC(cfg)
    f = jax.jit(lambda b, pos, d: PL.expand_belief(b, pos, d, pc, cfg, kind))
    pre = f"node/S{S}/{kind}/"
    N = len(F[pre + "n"]) if ncase is None else ncase
    worst = 0.0
    for i in range(N):
        n = int(F[pre + "n"][i])
        b0 = _belief(F[pre + "r"][i], F[pre + "pos"][i], F[pre + "P"][i], n)
        c1 = f(b0, jnp.asarray(F[pre + "a1"][i]), jnp.asarray(0))
        worst = max(worst, _cmp(F[pre + "c1"][i], F[pre + "r1"][i], F[pre + "x1"][i], F[pre + "P1"][i], c1[0], c1[1], n + 1))
        c2 = f(c1[1], jnp.asarray(F[pre + "a2"][i]), jnp.asarray(1))
        worst = max(worst, _cmp(F[pre + "c2"][i], F[pre + "r2"][i], F[pre + "x2"][i], F[pre + "P2"][i], c2[0], c2[1], n + 2))
    return worst


@pytest.mark.parametrize("S", [1, 2])
@pytest.mark.parametrize("kind", ["GD", "KLD"])
def test_node_matches_official_fixture(F, S, kind):
    w = _node_worst(F, S, kind, CFG)
    print(f"[METRIC] official fixture node S={S} {kind}: {len(F[f'node/S{S}/{kind}/n'])} cases x depth 1,2, worst rel err = {w:.2e}")
    assert w < 1e-9, w


@pytest.mark.parametrize("flip", [dict(root_double_predict=False), dict(ph_mode="product"), dict(kld_mode="official_nosum")])
def test_node_fixture_is_not_vacuous(F, flip):
    """Negative control: flipping one documented official behavior must break the comparison by far more than the tolerance."""
    S, kind = (2, "KLD") if "kld_mode" in flip else (2, "GD")
    w = _node_worst(F, S, kind, CFG.replace(**flip), ncase=20)
    print(f"[METRIC] negative control {flip}: worst rel err = {w:.2e}")
    assert w > 1e-3, w


# ------------------------------------------------------------------ filter
def _mine_filter(cfg, r, x, P, n, sp, zs, ms, lam, cap):
    NB = M.NB_MAX
    fs = FL.FState(jnp.zeros(NB).at[:n].set(r[:n]), jnp.zeros((NB, 4)).at[:n].set(x[:n]), jnp.tile(jnp.eye(4), (NB, 1, 1)).at[:n].set(P[:n]), jnp.asarray(n, jnp.int32))
    ld = lam / (np.pi * 40.0 ** 2)
    for s in range(2):
        z = np.zeros((M.M_MEAS, 2)); zv = np.zeros(M.M_MEAS, bool); m = int(ms[s])
        z[:m] = zs[s][:m]; zv[:m] = True
        fs, _ = FL.update_sensor(fs, jnp.asarray(z), jnp.asarray(zv), jnp.asarray(sp[s]), ld, cfg)
    fs = FL.merge(fs, cap)
    k = int(fs.n)
    return np.asarray(fs.r)[:k], np.asarray(fs.x)[:k], np.asarray(fs.P)[:k]


def _filter_worst(F, cap, cfg, ncase=None):
    pre = f"filter/cap{int(cap)}/"
    N = len(F[pre + "n"]) if ncase is None else ncase
    worst = 0.0
    for i in range(N):
        rm, xm, Pm = _mine_filter(cfg, F[pre + "r"][i], F[pre + "x"][i], F[pre + "P"][i], int(F[pre + "n"][i]), F[pre + "sp"][i],
                                  [F[pre + "z0"][i], F[pre + "z1"][i]], F[pre + "m"][i], float(F[pre + "lam"][i]), cap)
        no = int(F[pre + "no"][i])
        assert len(rm) == no, (i, len(rm), no)
        if no:
            ro, xo, Po = F[pre + "ro"][i][:no], F[pre + "xo"][i][:no], F[pre + "Po"][i][:no]
            worst = max(worst, np.max(np.abs(ro - rm)), np.max(np.abs(xo - xm) / (1 + np.abs(xo))), np.max(np.abs(Po - Pm) / (1 + np.abs(Po))))
    return worst


@pytest.mark.parametrize("cap", [False, True])
def test_filter_step_matches_official_fixture(F, cap):
    w = _filter_worst(F, cap, CFG)
    print(f"[METRIC] official fixture filter cap={cap}: {len(F[f'filter/cap{int(cap)}/n'])} cases, worst rel err = {w:.2e}")
    assert w < 1e-8, w


def test_filter_fixture_is_not_vacuous(F):
    """The authors' LBP runs ONE sweep (alias bug, D30); a converged BP must disagree with the fixture on at least some cases."""
    w = _filter_worst(F, False, CFG.replace(lbp_mode="converged", lbp_tol=1e-9, lbp_max_iter=1000), ncase=60)
    print(f"[METRIC] negative control converged-LBP: worst err = {w:.2e}")
    assert w > 1e-6, w


# ------------------------------------------------------------------ full MCTS decision with injected rollouts
CASES = [(1, 1, 7), (1, 5, 7), (1, 5, 40), (1, 10, 40), (2, 1, 49), (2, 5, 49), (2, 5, 200), (2, 10, 60)]


@pytest.mark.parametrize("fn,kind", [("GOSPA", "GD"), ("KLD", "KLD")])
@pytest.mark.parametrize("S,T,B", CASES)
def test_mcts_decision_matches_official_fixture(F, fn, kind, S, T, B):
    plan = jax.jit(PL.make_mcts(CFG, S, T, B, kind))
    offs = M.action_offsets()
    gaps = []
    trial = 0
    while f"mcts/{fn}/{S}_{T}_{B}/{trial}/n" in F:
        pre = f"mcts/{fn}/{S}_{T}_{B}/{trial}"
        n = int(F[pre + "/n"]); sp = F[pre + "/sp"]
        NPb = CFG.r_cap + T
        b = PL.Belief(jnp.zeros(NPb).at[:n].set(F[pre + "/r"][:n]), jnp.zeros((NPb, 2)).at[:n].set(F[pre + "/pos"][:n]),
                      jnp.tile(jnp.eye(2), (NPb, 1, 1)).at[:n].set(F[pre + "/P"][:n]), jnp.asarray(n, jnp.int32))
        o = plan(b, jnp.asarray(sp[:S]), jnp.asarray(F[pre + "/U"]), jnp.zeros(B))
        mine = np.asarray(o["root_cost"]); got = {}
        for a in range(7 ** S):
            if not np.isnan(mine[a]):
                pp = np.asarray(PL.pos_of(jnp.asarray(sp[:S]), a, jnp.asarray(offs)))
                got[tuple(np.round(pp.reshape(-1), 9))] = mine[a]
        cp, cc = F[pre + "/child_pos"], F[pre + "/child_cost"]
        exp = {tuple(np.round(cp[j].reshape(-1), 9)): cc[j] for j in range(len(cc))}
        assert set(got) == set(exp), (trial, len(got), len(exp))
        gap = max(abs(v - got[k]) / max(1.0, abs(v)) for k, v in exp.items())
        gaps.append(gap)
        assert gap < 1e-9, (trial, gap)
        assert np.allclose(F[pre + "/best"].reshape(S, 2), np.asarray(o["pos"]).reshape(S, 2), atol=1e-9), (trial, F[pre + "/best"], np.asarray(o["pos"]))
        trial += 1
    assert trial >= 3
    print(f"[METRIC] official fixture MCTS {kind} S={S} T={T} B={B}: {trial} decisions, max root-child C-bar rel gap = {max(gaps):.2e}, chosen action identical")


# ------------------------------------------------------------------ replay of the authors' main.py run
TRACE_BLOCKS = [("Myopic-KLD", "Myopic-KLD"), ("Myopic-GD", "Myopic-GD"), ("MCTS1-GD", "MCTS1-GD"), ("MCTS3-GD", "MCTS3-GD")]


@pytest.fixture(scope="module")
def TR():
    p = os.path.join(FD, "main_trace_seed123546_lam2.npz")
    assert os.path.exists(p), "main.py trace fixture missing (oracle/trace_main.py in the official venv)"
    z = np.load(p)
    return {k: z[k] for k in z.files}


def _inject(tr, name, alg, steps):
    p = name + "/"
    K = steps
    z = np.zeros((K, 2, M.M_MEAS, 2)); zv = np.zeros((K, 2, M.M_MEAS), bool)
    m = tr[p + "z"].shape[2]
    z[:, :, :m] = tr[p + "z"][:K]; zv[:, :, :m] = tr[p + "zv"][:K]
    Tm = max(alg.T - 1, 1)
    sel, nav, nits, joint = tr[p + "sel"][:K], tr[p + "nav"][:K], tr[p + "nits"][:K], tr[p + "joint"][:K]
    u = np.where(sel >= 0, (sel + 0.5) / nav, 0.0)                    # randint(0, n) = sel  <=>  floor(u * n) = sel
    UJ = np.zeros((K, alg.b_joint, Tm)); UI = np.zeros((K, 2, alg.b_ind, Tm))
    for k in range(K):
        w = min(Tm, u.shape[2])
        if joint[k]:
            assert nits[k] == alg.b_joint or alg.T == 1, (k, nits[k])
            UJ[k, : nits[k], :w] = u[k, : nits[k], :w]
        elif alg.T > 1:
            assert nits[k] == 2 * alg.b_ind, (k, nits[k])
            for it in range(nits[k]):
                UI[k, it // alg.b_ind, it % alg.b_ind, :w] = u[k, it, :w]
    return z, zv, dict(z=z, zv=zv, u=(UJ, np.zeros((K, alg.b_joint)), UI, np.zeros((K, 2, alg.b_ind))))


def _run_mine(alg, inj, steps, lam=2.0):
    lane = A.make_lane_step(CFG, alg, nan_mode="off", inj=inj)
    carry = jax.tree_util.tree_map(lambda a: a[0], A.init_carry(CFG, 1))
    f = jax.jit(lambda c, k: lane(c, k, jnp.asarray(0), jnp.asarray(lam)))
    outs = []
    for k in range(steps):
        carry, out = f(carry, jnp.asarray(k)); outs.append(out)
    return {kk: np.stack([np.asarray(o[kk]) for o in outs]) for kk in outs[0]}


@pytest.mark.parametrize("name,block", TRACE_BLOCKS)
def test_replay_of_authors_main_py(TR, name, block):
    """Feed main.py's OWN measurements (and, free-running, its own rollout draws) to our lane; compare with ITS sensor tracks and ITS per-step
    GOSPA^2 (the authors' all_gospa list, not a re-run).   GOSPA_PARITY_STEPS shortens the replay."""
    alg = M.ALGS[name]
    steps = int(os.environ.get("GOSPA_PARITY_STEPS", "200"))
    z, zv, inj = _inject(TR, block, alg, steps)
    gs = TR[block + "/gospa_sq"][:steps]
    spos = TR[block + "/spos"][:steps]
    # teacher-forced: positions forced to the authors', so only filter + merge + GOSPA are tested
    out_f = _run_mine(alg, dict(z=z, zv=zv, spos=spos), steps)
    ef = np.max(np.abs(out_f["gospa_sq"] - gs) / (1 + gs))
    print(f"[METRIC] main.py replay {name} {steps} steps teacher-forced: max rel GOSPA^2 err = {ef:.2e}")
    assert ef < 1e-8
    # free-running: our planner with the authors' rollout draws must pick the authors' positions every step
    out = _run_mine(alg, inj, steps)
    same = np.all(np.abs(out["spos"] - spos) < 1e-9, axis=(1, 2))
    assert same.all(), f"first sensor-track divergence from main.py at step {int(np.argmin(same))}"
    assert (out["joint"].astype(bool) == TR[block + "/joint"][:steps]).all()
    e2 = np.max(np.abs(out["gospa_sq"] - gs) / (1 + gs))
    print(f"[METRIC] main.py replay {name} {steps} steps free-running: tracks identical, joint-mode fraction {TR[block + '/joint'][:steps].mean():.2f}, max rel GOSPA^2 err = {e2:.2e}")
    assert e2 < 1e-8


# ------------------------------------------------------------------ measurement generator (distributional; the RNG streams cannot match)
from src import data as D
import meas_stats

MEAS_NAMES = ["single_d0", "single_d36", "single_d40", "single_d44", "clutter_only_lam2", "clutter_only_lam0p1", "scenario_k75_lam1", "scenario_k75_lam2"]
MEAS_LAM = {"single_d0": 0.0, "single_d36": 0.0, "single_d40": 0.0, "single_d44": 0.0, "clutter_only_lam2": 2.0, "clutter_only_lam0p1": 0.1,
            "scenario_k75_lam1": 1.0, "scenario_k75_lam2": 2.0}


def _our_sets(F, name, n=60000):
    tr, al, comp, cm = D.truth_arrays()
    if name.startswith("single"):
        d = float(name.split("_d")[1]); t = np.zeros((9, 4)); t[0, 0] = d; t[0, 1] = t[0, 3] = 0.1; a = np.zeros(9, bool); a[0] = True
        sp = np.zeros(2); org = np.array([d, 0.0])
    elif name.startswith("clutter_only"):
        t = np.zeros((9, 4)); a = np.zeros(9, bool); sp = np.array([3.0, -7.0]); org = sp
    else:
        t, a = tr[75], al[75]; sp = F["meas/sp75"]; org = sp
    f = jax.jit(jax.vmap(lambda key: D.gen_measurements(key, jnp.asarray(t), jnp.asarray(a), jnp.asarray(sp), MEAS_LAM[name], CFG)))
    z, v, over = f(jax.random.split(jax.random.PRNGKey(2024), n))
    z, v = np.asarray(z), np.asarray(v)
    assert not np.asarray(over).any()
    return [z[i][v[i]] for i in range(n)], org


@pytest.mark.parametrize("name", MEAS_NAMES)
def test_measurement_generator_matches_official_distribution(F, name):
    """The authors' Measurements.generate_measurements_multiple_targets (40 000 draws, their RNG) vs src/data.py CFG-O (60 000 draws): every summary statistic
    (detection / count histogram, residual moments incl. the effect of deleting noisy measurements > 40 m, clutter-in-disc moments) within 4.5 combined SE."""
    sets, org = _our_sets(F, name)
    mine = meas_stats.stats(sets, org)
    zs = []
    keys = [k[len(f"meas/{name}/"):] for k in F if k.startswith(f"meas/{name}/")]
    assert len(keys) >= 8
    for k in keys:
        vo, so = F[f"meas/{name}/{k}"]
        if k not in mine:
            continue
        vm, sm = mine[k]
        zs.append((abs(vo - vm) / np.sqrt(so ** 2 + sm ** 2 + 1e-18), k, vo, vm))
    worst = max(zs)
    print(f"[METRIC] measurement generator {name}: {len(zs)} statistics, max |z| = {worst[0]:.2f} ({worst[1]}: official {worst[2]:.4f} vs ours {worst[3]:.4f})")
    assert worst[0] < 4.5, worst

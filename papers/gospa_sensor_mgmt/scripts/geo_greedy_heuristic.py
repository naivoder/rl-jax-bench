"""No-search control: is RANDOM a handicapped 'search removed' arm in the F-A ablation?  A competent NO-SEARCH controller, declared before running:

GEO-GREEDY (no tree, no rollouts, no objective evaluation): goals = means of the predicted Bernoullis with r >= 0.5, plus the birth location (0, 0).
Sensor 0 takes the goal with the smallest obstacle-aware (geodesic, around the obstacle rectangle inflated by 1 m) distance; sensor 1 takes the nearest goal
other than sensor 0's when >= 2 goals exist, else the same goal.  Each sensor then takes the legal action (incl. stay) whose end point minimizes the
geodesic distance to its goal.  CFG-O, 50 runs (keys of E1, CRN measurements), clutter 0.1/1/2; algorithm slot = RANDOM's id (planner key only).
Declared reading: GEO-GREEDY G within 5 units of MCTS3-GD (or below) -> RANDOM understates a no-search controller and the sentence 'the whole 22-27 unit
gain comes from the planned navigation' is wrong; >= 10 units above MCTS3-GD -> the sentence stands; in between -> partial.
"""
import sys, json, os
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import model as M, algorithm as A, planner as PL, train as TR

OUT = os.path.join(os.environ.get("GOSPA_RUNS", "runs"), "geo_greedy")
os.makedirs(OUT, exist_ok=True)
x0, y0, x1, y1 = M.OBSTACLE
INF = 1e9
C = jnp.asarray([[x0 - 1, y0 - 1], [x1 + 1, y0 - 1], [x1 + 1, y1 + 1], [x0 - 1, y1 + 1]])

def seg_blocked(p, q):
    """segment p->q intersects the open obstacle rectangle (Liang-Barsky)."""
    d = q - p
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-d[0], p[0] - x0), (d[0], x1 - p[0]), (-d[1], p[1] - y0), (d[1], y1 - p[1])):
        par = jnp.abs(pk) < 1e-12
        t = qk / jnp.where(par, 1.0, pk)
        t0 = jnp.where(~par & (pk < 0), jnp.maximum(t0, t), t0)
        t1 = jnp.where(~par & (pk > 0), jnp.minimum(t1, t), t1)
        t1 = jnp.where(par & (qk <= 0), -1.0, t1)
    return t0 < t1 - 1e-9

def leg(p, q):
    return jnp.where(seg_blocked(p, q), INF, jnp.linalg.norm(q - p))

def geo(p, g):
    best = leg(p, g)
    for i in range(4):
        best = jnp.minimum(best, leg(p, C[i]) + leg(C[i], g))
        for j in range(4):
            if j != i:
                best = jnp.minimum(best, leg(p, C[i]) + leg(C[i], C[j]) + leg(C[j], g))
    return best

def make_plan_fn(cfg, alg, inj_u=None):
    off = jnp.asarray(M.action_offsets())
    def plan(fs, spos, key, k):
        NB = fs.r.shape[0]
        act = (jnp.arange(NB) < fs.n) & (fs.r >= 0.5)
        gpos = jnp.concatenate([fs.x[:, jnp.asarray(M.POS_IDX)], jnp.zeros((1, 2))], 0)
        gok = jnp.concatenate([act, jnp.ones((1,), bool)])
        d0 = jnp.where(gok, jax.vmap(lambda g: geo(spos[0], g))(gpos), INF)
        g0 = jnp.argmin(d0)
        d1 = jnp.where(gok, jax.vmap(lambda g: geo(spos[1], g))(gpos), INF)
        d1x = jnp.where(jnp.arange(gpos.shape[0]) == g0, INF, d1)
        g1 = jnp.where(jnp.sum(gok) >= 2, jnp.argmin(d1x), g0)
        new = []
        for s, gi in ((0, g0), (1, g1)):
            fe = PL.feas_sensor(spos[s], off)
            cand = spos[s][None, :] + off
            dd = jnp.where(fe, jax.vmap(lambda c: geo(c, gpos[gi]))(cand), 2 * INF)
            new.append(cand[jnp.argmin(dd)])
        z = jnp.asarray(False)
        return jnp.stack(new, 0), dict(joint=z, plan_over=z, root_over=z, plan_nan=z)
    return plan

A.make_plan_fn = make_plan_fn
res = TR.run_cell(M.CFG_O, M.ALGS["RANDOM"], 50, [0.1, 1.0, 2.0], os.path.join(OUT, "unused_ckpt"), ckpt=False, nan_mode="off")
gs = np.asarray(res["outs"]["gospa_sq"])          # [steps, lanes]; lane layout from train.lane_layout
ms, lams = res["ms"], res["lams"]
np.savez(os.path.join(OUT, "geo_greedy.npz"), gospa_sq=gs, ms=ms, lams=lams,
         spos=np.asarray(res["outs"]["spos"]), counters=res["counters"])
print("bad", res["bad"])

"""planner.py -- GOSPA-driven (GD) and KLD-driven (KLD) MCTS sensor management on the reduced tree (paper SS III-IV, Algorithm 1).

Per tree node: the sensors' positions and ONE merged Gaussian Bernoulli list (eqs. 42, 54-56): the 2^S detect/miss branches of the
synthetic update (eqs. 32-37) are moment-matched back into a single Bernoulli per target.  Node cost = sum_i sum_h p(h) C(r^h, P^h)
(eq. 43) with the closed-form GOSPA bound (eqs. 28-31, 39) or the KLD objective of the baseline.  The search is the paper's MCTS
(UCT selection, one expansion per iteration, uniformly random rollout to the horizon, running-mean back-propagation) written with
fixed-size node arrays so that a whole decision is one jit/vmap-able function.

CFG-O replicates the authors' code in the places the text leaves open or contradicts (README differences table): position random-walk planner
state (D8), root predicted twice (D10), sequential (exact-joint) p(h) (D11), KLD = Gaussian part with the joint tree scaled by
the sum of all existence probabilities (D12), UCT = argmax -Cbar + 0.05 sqrt(ln n / n_j) (D13), delta = tree first-costs + the
selected node's running mean + rollout costs weighted 0, 1, lambda, ... (D14), budget = iterations (D15), deterministic pop-last
expansion (D16), legal actions = end point not strictly inside the obstacle and inside the region (D18).  CFG-P: the paper's
stated model (4-D NCV planner state, eq. 38, random expansion, no root double predict, Bernoulli-form KLD).
"""
from __future__ import annotations

import itertools
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from src import model as M
from src.model import Cfg

A1 = 7  # actions per sensor (stay + 6 moves)


class Belief(NamedTuple):
    r: jnp.ndarray    # [NP]
    x: jnp.ndarray    # [NP, nx]
    P: jnp.ndarray    # [NP, nx, nx]
    nb: jnp.ndarray   # int32 number of valid Bernoullis (slots 0..nb-1)


def _tree_where(c, a, b):
    return jax.tree_util.tree_map(lambda u, v: jnp.where(c, u, v), a, b)


class PC:
    """Planner constants (jnp) for a configuration."""

    def __init__(self, cfg: Cfg):
        F, Q, H, bm, bc, pidx = M.planner_model(cfg)
        self.F, self.Q, self.H = jnp.asarray(F), jnp.asarray(Q), jnp.asarray(H)
        self.bm, self.bc = jnp.asarray(bm), jnp.asarray(bc)
        self.pidx = np.asarray(pidx)
        self.nx = cfg.plan_nx
        self.off = jnp.asarray(M.action_offsets())
        self.cost_idx = np.arange(self.nx) if cfg.plan_trace == "full" else self.pidx
        self.R = cfg.R_var * jnp.eye(2)


def predict_belief(b: Belief, pc: PC, cfg: Cfg):
    """Predict every valid Bernoulli (r <- p_S r; x <- F x; P <- F P F' + Q) and append the birth Bernoulli at slot nb."""
    NP = b.r.shape[0]
    act = jnp.arange(NP) < b.nb
    r = jnp.where(act, M.P_SURV * b.r, b.r)
    x = jnp.where(act[:, None], b.x @ pc.F.T, b.x)
    P = jnp.where(act[:, None, None], jnp.einsum("ab,nbc,dc->nad", pc.F, b.P, pc.F) + pc.Q, b.P)
    over = b.nb >= NP
    r = r.at[b.nb].set(cfg.r_b, mode="drop")
    x = x.at[b.nb].set(pc.bm, mode="drop")
    P = P.at[b.nb].set(pc.bc, mode="drop")
    return Belief(r, x, P, jnp.minimum(b.nb + 1, NP)), over


def _inv2(S):
    a, b, c, d = S[..., 0, 0], S[..., 0, 1], S[..., 1, 0], S[..., 1, 1]
    det = a * d - b * c
    inv = jnp.stack([jnp.stack([d, -b], -1), jnp.stack([-c, a], -1)], -2) / det[..., None, None]
    return inv, det


def _syn_update_cov(P, pc: PC):
    """eq. 36: P - P H' (H P H' + R)^-1 H P  (synthetic detection at the predicted measurement; mean unchanged, eq. 35)."""
    H = pc.H
    PHt = jnp.einsum("nab,cb->nac", P, H)
    HP = jnp.einsum("ab,nbc->nac", H, P)
    S = jnp.einsum("ab,nbc,dc->nad", H, P, H) + pc.R
    Sinv, _ = _inv2(S)
    return P - jnp.einsum("nab,nbc,ncd->nad", PHt, Sinv, HP)


def gospa_cost(r, P, pc: PC, c: float):
    """Single-Bernoulli GOSPA bound after the update, eqs. 28-31 and 39 (literal form, as in the official code)."""
    t = jnp.sum(jnp.stack([P[:, i, i] for i in pc.cost_idx], axis=0), axis=0)
    c2 = c * c
    tp = 2.0 * t / c2
    gam = 1.0 / (2.0 - jnp.fmin(tp, 1.0))
    rep = 0.5 * c2 * (1.0 - r) + r * jnp.fmin(t, c2)
    return jnp.where(r > gam, rep, 0.5 * c2 * r)


def _pos_block(P, pc: PC):
    return P[:, pc.pidx][:, :, pc.pidx]


def _cost_block(P, pc: PC):
    """covariance block of the coordinates the objective is measured in (pc.cost_idx: position block or the full state)."""
    return P[:, pc.cost_idx][:, :, pc.cost_idx]


def _inv_det(S):
    """inverse and determinant of a batch of SPD blocks (closed form for 2x2, the same code path as the archived CFG-O/CFG-P runs)."""
    if S.shape[-1] == 2:
        return _inv2(S)
    # 4x4: block (Schur-complement) inversion with closed-form 2x2 blocks -- elementwise, ~3x faster than LAPACK calls under vmap on CPU
    A, B, C, D = S[..., :2, :2], S[..., :2, 2:], S[..., 2:, :2], S[..., 2:, 2:]
    Ai, dA = _inv2(A)
    SC = D - C @ Ai @ B
    SCi, dSC = _inv2(SC)
    AiB = Ai @ B
    CAi = C @ Ai
    top = jnp.concatenate([Ai + AiB @ SCi @ CAi, -AiB @ SCi], axis=-1)
    bot = jnp.concatenate([-SCi @ CAi, SCi], axis=-1)
    return jnp.concatenate([top, bot], axis=-2), dA * dSC


def n_iter(cfg: Cfg, B: int) -> int:
    """number of MCTS iterations (= length of the pre-drawn uniform arrays) of a tree with budget B: B ('iter'); a cap of 2 B for the node
    budget ('nodes': iterations after B nodes have been added are masked out; iterations that end at a depth-T node add nothing, so more than
    B iterations may be needed to add B nodes, and a tree that cannot grow would otherwise never stop)."""
    return B if cfg.budget_mode == "iter" else 2 * B


def eval_belief(b: Belief, pos, pc: PC, cfg: Cfg, kind: str, pd_override=None):
    """Cost of the (already predicted) Bernoulli list b for sensors at pos[S,2], and the merged posterior Bernoulli list.

    Returns (cost scalar, Belief).  Hypotheses h in {0,1}^S are enumerated statically (2 or 4).
    """
    S = pos.shape[0]
    NP = b.r.shape[0]
    act = jnp.arange(NP) < b.nb
    xpos = b.x[:, pc.pidx]
    dist = jnp.sqrt(jnp.sum((xpos[:, None, :] - pos[None, :, :]) ** 2, axis=-1))       # [NP, S]
    pd = M.PD_MAX * jnp.exp(-0.5 * (dist / M.FOV) ** 2)                               # eq. 48 at the predicted mean
    if pd_override is not None:                                                       # exact toy: p_D given directly
        pd = jnp.broadcast_to(jnp.asarray(pd_override), pd.shape)
    sumr = jnp.sum(jnp.where(act, b.r, 0.0))
    P0pos = _pos_block(b.P, pc)
    inv0, det0 = _inv2(P0pos)
    cost_i = jnp.zeros((NP,))
    rbar = jnp.zeros((NP,))
    Pnum = jnp.zeros_like(b.P)
    for h in itertools.product((0, 1), repeat=S):
        rc = b.r
        Pc = b.P
        w = jnp.ones((NP,))
        for s in range(S):
            ps = pd[:, s]
            prob = (rc if cfg.ph_mode == "sequential" else b.r) * ps
            if h[s]:
                w = w * prob
                Pc = _syn_update_cov(Pc, pc)
                rc = jnp.ones_like(rc)
            else:
                w = w * (1.0 - prob)
                rc = ((1.0 - ps) * rc) / ((1.0 - rc) + (1.0 - ps) * rc)
        if kind == "GD":
            ch = gospa_cost(rc, Pc, pc, M.GOSPA_C)
        elif cfg.kld_mode in ("official", "official_nosum"):
            Phpos = _pos_block(Pc, pc)
            invh, deth = _inv2(Phpos)
            tr = jnp.einsum("nab,nba->n", invh, P0pos)
            kl = tr - jnp.log(det0 / deth) - 2.0
            fac = b.r if (S == 1 or cfg.kld_mode == "official_nosum") else sumr   # D12: the joint tree multiplies by sum_j r_j (bug)
            ch = -(fac / 2.0) * kl
        elif cfg.kld_mode == "info_gain":
            # sensitivity: expected information gain E_h[ KL(post_h || pred) ] of the Bernoulli-Gaussian densities (the synthetic update
            # keeps the mean, eq. 35): KL = KL(Ber(r^h) || Ber(r)) + r^h * KL(N(x, P^h) || N(x, P)); finite also when r^h = 1.
            P0k, Phk = _cost_block(b.P, pc), _cost_block(Pc, pc)
            inv0k, det0k = _inv_det(P0k)
            _, dethk = _inv_det(Phk)
            trk = jnp.einsum("nab,nba->n", inv0k, Phk)
            dk = float(len(pc.cost_idx))
            klg = 0.5 * (trk - dk - jnp.log(dethk / det0k))
            r0 = b.r
            t_on = jnp.where(rc > 0, rc * jnp.log(jnp.where(rc > 0, rc, 1.0) / jnp.where(r0 > 0, r0, 1.0)), 0.0)
            t_off = jnp.where(rc < 1, (1.0 - rc) * jnp.log(jnp.where(rc < 1, 1.0 - rc, 1.0) / jnp.where(r0 < 1, 1.0 - r0, 1.0)), 0.0)
            ch = -(t_on + t_off + rc * klg)
        else:
            ch = jnp.zeros((NP,))
        cost_i = cost_i + w * ch
        rbar = rbar + w * rc
        Pnum = Pnum + (w * rc)[:, None, None] * Pc
    Pbar = Pnum / rbar[:, None, None]
    xbar = b.x * (rbar / rbar)[:, None]       # exact x unless rbar == 0 (then NaN, as in the official code's 0/0)
    if kind == "KLD" and cfg.kld_mode == "bernoulli":
        r1 = b.r
        rb = jnp.minimum(rbar, 1.0 - 1e-12)
        t1 = jnp.where(1.0 - r1 > 0, (1.0 - r1) * jnp.log((1.0 - r1) / (1.0 - rb)), 0.0)
        t2 = jnp.where(r1 > 0, r1 * jnp.log(r1 / rb), 0.0)
        Pbk, P0k = _cost_block(Pbar, pc), _cost_block(b.P, pc)     # position block (CFG-P, archived) or the full state (CFG-P2)
        invb, detb = _inv_det(Pbk)
        _, det0k = _inv_det(P0k)
        trb = jnp.einsum("nab,nba->n", invb, P0k)
        gk = 0.5 * r1 * (trb - jnp.log(det0k / detb) - float(len(pc.cost_idx)))
        cost_i = -(t1 + t2 + gk)
    cost = jnp.sum(jnp.where(act, cost_i, 0.0))
    out = Belief(jnp.where(act, rbar, b.r), jnp.where(act[:, None], xbar, b.x),
                 jnp.where(act[:, None, None], Pbar, b.P), b.nb)
    return cost, out


def expand_belief(b: Belief, pos, depth_parent, pc: PC, cfg: Cfg, kind: str):
    """Child node of belief b (the parent's merged posterior) for sensors moved to pos: predict (unless the parent is the
    root and the configuration does not double-predict), evaluate, merge.  Returns (cost, Belief, overflow)."""
    bp, over = predict_belief(b, pc, cfg)
    do_pred = (depth_parent > 0) | cfg.root_double_predict
    bin_ = _tree_where(do_pred, bp, b)
    cost, out = eval_belief(bin_, pos, pc, cfg, kind)
    return cost, out, over & do_pred


# --------------------------------------------------------------------------------------------------------------------
# legal actions
# --------------------------------------------------------------------------------------------------------------------

def feas_sensor(p, off):
    """[7] bool: end point inside the region (inclusive) and not strictly inside the obstacle rectangle (D18)."""
    cand = p[None, :] + off
    x0, y0, x1, y1 = M.OBSTACLE
    inreg = jnp.all(jnp.abs(cand) <= M.REGION, axis=-1)
    inobs = (cand[:, 0] > x0) & (cand[:, 0] < x1) & (cand[:, 1] > y0) & (cand[:, 1] < y1)
    return inreg & ~inobs


def feas_joint(pos, off):
    """[7^S] bool in itertools.product order (sensor 0 is the major index), as the official get_legal_actions."""
    S = pos.shape[0]
    f = feas_sensor(pos[0], off)
    for s in range(1, S):
        f = (f[:, None] & feas_sensor(pos[s], off)[None, :]).reshape(-1)
    return f


def pos_of(pos, a, off):
    S = pos.shape[0]
    out = []
    for s in range(S):
        i = (a // (A1 ** (S - 1 - s))) % A1
        out.append(pos[s] + off[i])
    return jnp.stack(out, axis=0)


def kth_true(mask, k):
    """Index of the k-th (0-based) True entry of mask."""
    cs = jnp.cumsum(mask.astype(jnp.int32))
    return jnp.argmax(cs > k).astype(jnp.int32)


# --------------------------------------------------------------------------------------------------------------------
# MCTS
# --------------------------------------------------------------------------------------------------------------------

def make_mcts(cfg: Cfg, S: int, T: int, B: int, kind: str):
    """Build plan(root: Belief, pos0[S,2], u_roll[B, max(T-1,1)], u_exp[B]) -> dict for one MCTS decision (one lane).

    S sensors planned jointly (A = 7^S children per node), lookahead T, B iterations, objective kind in {'GD','KLD'}.
    """
    pc = PC(cfg)
    A = A1 ** S
    N = B + 1
    lam = cfg.lam
    eps = cfg.uct_eps
    off = pc.off

    def plan(root: Belief, pos0, u_roll, u_exp):
        NP = root.r.shape[0]
        nx = root.x.shape[1]
        st = dict(
            pos=jnp.zeros((N, S, 2)).at[0].set(pos0),
            br=jnp.zeros((N, NP)).at[0].set(root.r),
            bx=jnp.zeros((N, NP, nx)).at[0].set(root.x),
            bP=jnp.tile(jnp.eye(nx), (N, NP, 1, 1)).at[0].set(root.P),
            bnb=jnp.zeros((N,), jnp.int32).at[0].set(root.nb),
            parent=jnp.full((N,), -1, jnp.int32),
            depth=jnp.zeros((N,), jnp.int32),
            nvis=jnp.zeros((N,)),
            cost=jnp.zeros((N,)),
            first=jnp.zeros((N,)),
            prefix=jnp.zeros((N,)),
            cact=jnp.zeros((N,), jnp.int32),
            tried=jnp.zeros((N, A), bool),
            nchild=jnp.zeros((N,), jnp.int32),
            child=jnp.zeros((N, A), jnp.int32),
            nn=jnp.asarray(1, jnp.int32),
            over=jnp.asarray(False),
        )

        def belief_of(st, i):
            return Belief(st["br"][i], st["bx"][i], st["bP"][i], st["bnb"][i])

        def best_child(st, node, e):
            ids = st["child"][node]
            valid = jnp.arange(A) < st["nchild"][node]
            nv = jnp.maximum(st["nvis"][ids], 1.0)
            wts = -st["cost"][ids] + e * jnp.sqrt(jnp.log(jnp.maximum(st["nvis"][node], 1.0)) / nv)
            wts = jnp.where(valid, wts, -jnp.inf)
            k = jnp.argmax(wts)
            return ids[k]

        def iteration(it, st):
            # ---------------- tree policy ----------------
            def tp(i, c):
                node, done, flag, a = c
                d = st["depth"][node]
                term = d >= T
                fe = feas_joint(st["pos"][node], off)
                untried = fe & ~st["tried"][node]
                has = jnp.any(untried)
                stop_exp = ~done & ~term & has
                stop_term = ~done & term
                if cfg.expansion == "det":
                    a_new = (A - 1 - jnp.argmax(untried[::-1])).astype(jnp.int32)
                else:
                    k = jnp.floor(u_exp[it] * jnp.sum(untried)).astype(jnp.int32)
                    a_new = kth_true(untried, k)
                bc = best_child(st, node, eps)
                node2 = jnp.where(done | stop_exp | stop_term, node, bc)
                return node2, done | stop_exp | stop_term, flag | stop_exp, jnp.where(stop_exp, a_new, a)

            node, done, flag, a = lax.fori_loop(0, T + 1, tp, (jnp.asarray(0, jnp.int32), jnp.asarray(False),
                                                              jnp.asarray(False), jnp.asarray(0, jnp.int32)))
            # node budget (paper SS IV-C.2): once B nodes have been added the remaining iterations are masked out (no expansion, no back-propagation)
            live = (st["nn"] - 1 < B) if cfg.budget_mode == "nodes" else jnp.asarray(True)
            flag = flag & live
            # ---------------- expansion (masked) ----------------
            nn = st["nn"]
            pnew = pos_of(st["pos"][node], a, off)
            cst, cb, ov = expand_belief(belief_of(st, node), pnew, st["depth"][node], pc, cfg, kind)
            wi = jnp.where(flag, nn, N)                       # out-of-range index drops the write
            pf = jnp.where(flag, node, N)
            dnew = st["depth"][node] + 1
            st = dict(st)
            st["pos"] = st["pos"].at[wi].set(pnew, mode="drop")
            st["br"] = st["br"].at[wi].set(cb.r, mode="drop")
            st["bx"] = st["bx"].at[wi].set(cb.x, mode="drop")
            st["bP"] = st["bP"].at[wi].set(cb.P, mode="drop")
            st["bnb"] = st["bnb"].at[wi].set(cb.nb, mode="drop")
            st["parent"] = st["parent"].at[wi].set(node, mode="drop")
            st["depth"] = st["depth"].at[wi].set(dnew, mode="drop")
            st["cost"] = st["cost"].at[wi].set(cst, mode="drop")
            st["first"] = st["first"].at[wi].set(cst, mode="drop")
            st["prefix"] = st["prefix"].at[wi].set(st["prefix"][node] + lam ** (dnew - 1.0) * cst, mode="drop")
            st["cact"] = st["cact"].at[wi].set(a, mode="drop")
            st["tried"] = st["tried"].at[pf, a].set(True, mode="drop")
            st["child"] = st["child"].at[pf, st["nchild"][node]].set(nn, mode="drop")
            st["nchild"] = st["nchild"].at[pf].add(1, mode="drop")
            st["nn"] = nn + jnp.where(flag, 1, 0)
            st["over"] = st["over"] | (flag & ov)
            sel = jnp.where(flag, nn, node)
            dsel = st["depth"][sel]
            # ---------------- rollout ----------------
            def ro(j, c):
                b, pos, tot, over = c
                active = j < (T - dsel)
                fe = feas_joint(pos, off)
                k = jnp.floor(u_roll[it, j] * jnp.sum(fe)).astype(jnp.int32)
                aa = kth_true(fe, k)
                p2 = pos_of(pos, aa, off)
                c2, b2, o2 = expand_belief(b, p2, jnp.asarray(1), pc, cfg, kind)
                tot = tot + jnp.where(active, lam ** (dsel + j).astype(jnp.float64) * c2, 0.0)
                return (_tree_where(active, b2, b), jnp.where(active, p2, pos), tot, over | (active & o2))

            if T > 1:
                _, _, tot, ovr = lax.fori_loop(0, T - 1, ro, (belief_of(st, sel), st["pos"][sel], jnp.asarray(0.0), jnp.asarray(False)))
            else:
                tot, ovr = jnp.asarray(0.0), jnp.asarray(False)
            st["over"] = st["over"] | (ovr & live)
            par = st["parent"][sel]
            csel = st["cost"][sel] if cfg.delta_selected == "running" else st["first"][sel]
            delta = st["prefix"][par] + lam ** (dsel - 1.0) * csel + tot

            # ---------------- back-propagation ----------------
            def bp(t, c):
                cost_a, nv_a, nd = c
                ok = nd >= 0
                idx = jnp.where(ok, nd, N)
                n0 = nv_a[jnp.maximum(nd, 0)]
                c0 = cost_a[jnp.maximum(nd, 0)]
                cost_a = cost_a.at[idx].set((c0 * n0 + delta) / (n0 + 1.0), mode="drop")
                nv_a = nv_a.at[idx].add(1.0, mode="drop")
                return cost_a, nv_a, jnp.where(ok, st["parent"][jnp.maximum(nd, 0)], -1)

            cost_a, nv_a, _ = lax.fori_loop(0, T + 1, bp, (st["cost"], st["nvis"], jnp.where(live, sel, -1)))
            st["cost"], st["nvis"] = cost_a, nv_a
            return st

        st = lax.fori_loop(0, n_iter(cfg, B), iteration, st)      # B iterations, or the 2 B-iteration cap of the node budget
        ids = st["child"][0]
        valid = jnp.arange(A) < st["nchild"][0]
        wts = jnp.where(valid, -st["cost"][ids], -jnp.inf)
        k = jnp.argmax(wts)
        best = ids[k]
        rc = jnp.full((A,), jnp.nan).at[jnp.where(valid, st["cact"][ids], A)].set(st["cost"][ids], mode="drop")
        return dict(pos=st["pos"][best], action=st["cact"][best], root_cost=rc, n_nodes=st["nn"], overflow=st["over"],
                    root_nvis=st["nvis"][0], any_nan=jnp.any(valid & jnp.isnan(st["cost"][ids])))

    return plan

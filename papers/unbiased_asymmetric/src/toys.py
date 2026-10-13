"""toy check: exact (enumerated) check of the paper's central claims on toys where V(h), V(h,s), V(s) are computable.

    JAX_PLATFORMS=cpu python src/toys.py [--out runs/toys/toys.json]

Toys:
  * MHH  -- "Mini Heaven-Hell", closed form: states (pos in {F, P}, side X in {L, R}); at F the agent exits L/R
            (+1 if X, else -1; terminal) or visits the priest P (obs = X), then is forced back to F. Policy: root
            logits (0, 0, th_q) -> q = pi(P); after the priest pick the correct exit w.p. c = sigmoid(th_c).
  * MHH-R -- the same POMDP with priest revisits allowed, a generic tabular softmax policy over every history,
            horizon H (time limit = terminal, as in training). Here V_MC(s) is biased too.
  * Good/Bad -- paper App. B.2 (reactive policy, frozen state): value bias 1/6 but zero gradient bias.

For each estimator X in {h (A2C), hs (A2C-asym-hs), s_TD (A2C-asym-s at its TD(0) fixed point), s_MC, s_timed}
the EXACT expected policy gradient sum_tau Pr(tau) sum_t g^t delta^X_t grad log pi(a_t|h_t) is computed by
enumerating every trajectory, and compared with grad J (jax.grad of the enumerated objective). A Monte-Carlo
check then samples episodes and pushes them through the PRODUCTION actor-loss code (algorithm.pg_terms) with the
exact critics, verifying the sample mean converges to the exact expectation at the 1/sqrt(N) rate.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from algorithm import Episode, pg_terms  # noqa: E402

jax.config.update("jax_enable_x64", True)
NEG = -1e30                        # finite "-inf" logit for disallowed actions (keeps gradients finite)
L_, R_, P_ = 0, 1, 2               # actions: exit left, exit right, go to / come back from the priest
OBS_F, OBS_L, OBS_R, OBS_0 = 0, 1, 2, 3   # observations; OBS_0 = reset observation (as in training)


# ----------------------------------------------------------------------------- MHH POMDP
def mhh_step(s, a):
    """s = (pos, X) encoded 2*pos + X (pos 0=F, 1=P; X 0=L, 1=R). Returns list of (prob, s', o, r, done)."""
    pos, X = divmod(s, 2)
    if pos == 1:                                  # at the priest: forced back to the fork
        return [(1.0, X, OBS_F, 0.0, False)]
    if a == P_:
        return [(1.0, 2 + X, OBS_L if X == 0 else OBS_R, 0.0, False)]
    return [(1.0, s, OBS_F, 1.0 if a == X else -1.0, True)]


def enumerate_tree(step, starts, allowed, horizon):
    """All trajectories of a finite-horizon POMDP under ANY full-support policy.
    allowed(hist, s) -> list of actions (the policy acts on hist only; `s` is used only to detect forced nodes,
    which are state-determined here: at P the observation already reveals pos=P).
    Returns node table (history -> id), allowed-action mask, and a list of trajectories, each a list of steps
    (node, s, a, r, done, env_prob)."""
    nodes, masks, trajs = {}, [], []

    def nid(h, acts):
        if h not in nodes:
            nodes[h] = len(nodes)
            m = np.zeros(3, bool)
            m[acts] = True
            masks.append(m)
        return nodes[h]

    def rec(h, s, t, steps, penv):
        acts = allowed(h, s)
        n = nid(h, acts)
        for a in acts:
            for p, s2, o, r, done in step(s, a):
                end = done or t + 1 >= horizon
                st = steps + [(n, s, a, r, done, s2)]
                if end:
                    trajs.append((penv * p, st, end and not done))
                else:
                    rec(h + (a, o), s2, t + 1, st, penv * p)

    for s0, p0 in starts:
        rec((OBS_0,), s0, 0, [], p0)
    return nodes, np.array(masks), trajs


class Enumerated:
    """Exact quantities for a tabular policy given by logits(theta) -> [n_nodes, 3]."""

    def __init__(self, nodes, masks, trajs, gamma, n_states):
        self.nodes, self.masks, self.gamma, self.nS = nodes, masks, gamma, n_states
        T = max(len(st) for _, st, _ in trajs)
        n = len(trajs)
        self.T, self.n = T, n
        self.penv = np.array([p for p, _, _ in trajs])
        self.node = np.zeros((n, T + 1), int)
        self.state = np.zeros((n, T + 1), int)
        self.act = np.zeros((n, T + 1), int)
        self.rew = np.zeros((n, T))
        self.mask = np.zeros((n, T), bool)
        self.trunc = np.array([tr for _, _, tr in trajs])
        self.length = np.array([len(st) for _, st, _ in trajs])
        for i, (_, st, _) in enumerate(trajs):
            for t, (nd, s, a, r, done, s2) in enumerate(st):
                self.node[i, t], self.state[i, t], self.act[i, t], self.rew[i, t], self.mask[i, t] = nd, s, a, r, True
            self.state[i, len(st)] = st[-1][5]
        self.disc = gamma ** np.arange(T)

    def logp(self, logits):
        return jax.nn.log_softmax(jnp.where(self.masks, logits, NEG), -1)

    def traj_prob(self, lp):
        steps = jnp.where(self.mask, jnp.take_along_axis(lp[self.node[:, :-1]], self.act[:, :-1, None], -1)[..., 0],
                          0.0)
        return self.penv * jnp.exp(steps.sum(-1))

    def J(self, logits):
        return jnp.sum(self.traj_prob(self.logp(logits)) * (self.rew * self.disc).sum(-1))

    def critics(self, logits):
        """Exact critic values at every trajectory position [n, T+1] for each estimator (position after the end = 0)."""
        P = np.asarray(self.traj_prob(self.logp(logits)))
        g = self.gamma
        G = np.zeros((self.n, self.T + 1))                       # discounted return-to-go
        for t in range(self.T - 1, -1, -1):
            G[:, t] = np.where(self.mask[:, t], self.rew[:, t] + g * G[:, t + 1], 0.0)
        valid = np.concatenate([self.mask, np.zeros((self.n, 1), bool)], 1)
        w = P[:, None] * valid
        tt = np.broadcast_to(np.arange(self.T + 1), w.shape)

        def grouped(*keys):
            k = np.stack([np.asarray(x) for x in keys], -1).reshape(-1, len(keys))
            uniq, inv = np.unique(k, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            num = np.bincount(inv, (w * G).reshape(-1), len(uniq))
            den = np.bincount(inv, w.reshape(-1), len(uniq))
            return np.where(valid, (num / np.maximum(den, 1e-300))[inv].reshape(w.shape), 0.0)

        out = {"h": grouped(self.node), "hs": grouped(self.node, self.state), "s_MC": grouped(self.state),
               "s_timed": grouped(tt, self.state)}
        # TD(0) fixed point of a tabular state critic trained on on-policy data (time-uniform visitation, the
        # unweighted sum_t delta_t^2 loss): V = r_d + g P_d V on the state-aggregated chain
        nS = self.nS
        wsum, rsum, Pm = np.zeros(nS), np.zeros(nS), np.zeros((nS, nS))
        for i in range(self.n):
            for t in range(self.length[i]):
                s = self.state[i, t]
                wsum[s] += P[i]
                rsum[s] += P[i] * self.rew[i, t]
                if t + 1 < self.length[i]:                     # nonterminal (time limit = terminal too)
                    Pm[s, self.state[i, t + 1]] += P[i]
        seen = wsum > 0
        rd = np.where(seen, rsum / np.maximum(wsum, 1e-300), 0.0)
        Pd = Pm / np.maximum(wsum, 1e-300)[:, None]
        Vtd = np.linalg.solve(np.eye(nS) - g * Pd, rd)
        out["s_TD"] = np.where(valid, Vtd[self.state], 0.0)
        out["_V_TD_table"] = Vtd
        out["_V_MC_table"] = np.array([np.sum(w * G * (self.state == s)) / max(np.sum(w * (self.state == s)), 1e-300)
                                       for s in range(nS)])
        return out

    def expected_grad(self, logits, v):
        """Exact E[sum_t g^t delta_t grad log pi] for critic values v [n, T+1] (stop-gradient), as grad of a
        probability-weighted surrogate with stop-gradient weights."""
        P = jax.lax.stop_gradient(self.traj_prob(self.logp(logits)))
        nxt = np.concatenate([self.mask[:, 1:], np.zeros((self.n, 1), bool)], 1)
        delta = self.rew + self.gamma * np.where(nxt, v[:, 1:], 0.0) - v[:, :-1]

        def sur(lg):
            lp = self.logp(lg)
            la = jnp.take_along_axis(lp[self.node[:, :-1]], self.act[:, :-1, None], -1)[..., 0]
            return jnp.sum(P[:, None] * jnp.where(self.mask, self.disc * delta * la, 0.0))
        return jax.grad(sur)(logits)

    def episodes(self, idx):
        """Production Episode structs for trajectory indices idx (obs/actions as in training; obs unused here)."""
        ep = Episode(states=jnp.asarray(self.state[idx]), obs=jnp.zeros((len(idx), self.T + 1), int),
                     actions=jnp.asarray(self.act[idx]), rewards=jnp.asarray(self.rew[idx]),
                     mask=jnp.asarray(self.mask[idx]), trunc=jnp.asarray(self.trunc[idx]),
                     length=jnp.asarray(self.length[idx]))
        return ep


# ----------------------------------------------------------------------------- MHH closed form
def mhh_closed_form_check(gamma, q, c, mc_n=(1000, 10000, 100000), seed=0):
    def allowed(h, s):
        if s >= 2:
            return [P_]                     # forced back to F (single action)
        return [L_, R_, P_] if len(h) == 1 else [L_, R_]
    nodes, masks, trajs = enumerate_tree(mhh_step, [(0, 0.5), (1, 0.5)], allowed, horizon=3)
    E = Enumerated(nodes, masks, trajs, gamma, 4)
    root = nodes[(OBS_0,)]
    post = {X: nodes[(OBS_0, P_, OBS_L if X == 0 else OBS_R, P_, OBS_F)] for X in (0, 1)}
    th_q, th_c = np.log(2 * q / (1 - q)), np.log(c / (1 - c))      # q = e^th/(2 + e^th); c = sigmoid(th_c)

    def logits_of(th):
        lg = jnp.zeros((len(nodes), 3))
        lg = lg.at[root].set(jnp.array([0.0, 0.0, th[0]]))
        lg = lg.at[post[0]].set(jnp.array([th[1], 0.0, 0.0]))       # X = L: correct exit is L
        lg = lg.at[post[1]].set(jnp.array([0.0, th[1], 0.0]))
        return lg

    th = jnp.array([th_q, th_c])
    lg = logits_of(th)
    J = float(E.J(lg))
    gJ = np.asarray(jax.grad(lambda t: E.J(logits_of(t)))(th))
    crit = E.critics(lg)
    res = dict(gamma=gamma, q=q, c=c, J=J, J_closed=q * gamma ** 2 * (2 * c - 1),
               dJ=gJ.tolist(), dJ_closed=[gamma ** 2 * (2 * c - 1) * q * (1 - q), 2 * q * gamma ** 2 * c * (1 - c)])
    # chain rule to (th_q, th_c) of the per-node logits gradient
    _, vjp = jax.vjp(logits_of, th)
    est = {}
    for X in ("h", "hs", "s_TD", "s_MC", "s_timed"):
        g = np.asarray(vjp(E.expected_grad(lg, crit[X]))[0])
        est[X] = dict(grad=g.tolist(), rel_bias=((g - gJ) / np.abs(gJ)).tolist())
    res["estimators"] = est
    VTD, VMC = crit["_V_TD_table"], crit["_V_MC_table"]
    k = 1 + q - q * gamma ** 2
    res["V_TD_F"], res["V_TD_F_closed"] = float(VTD[0]), q * (2 * c - 1) / k
    res["V_TD_P"], res["V_TD_P_closed"] = float(VTD[2]), gamma * q * (2 * c - 1) / k
    res["V_MC_F"], res["V_MC_F_closed"] = float(VMC[0]), q * (2 * c - 1) * (1 + gamma ** 2) / (1 + q)
    res["rel_bias_sTD_q_closed"] = -(1 - q * gamma ** 2) / k
    res["mc"] = monte_carlo(E, lg, crit, vjp, gamma, mc_n, seed)
    return res


# ----------------------------------------------------------------------------- MHH with revisits
def mhh_revisit_check(gamma, horizon, seed=0, mc_n=(1000, 10000, 100000)):
    def allowed(h, s):
        return [P_] if s >= 2 else [L_, R_, P_]
    nodes, masks, trajs = enumerate_tree(mhh_step, [(0, 0.5), (1, 0.5)], allowed, horizon=horizon)
    E = Enumerated(nodes, masks, trajs, gamma, 4)
    lg = jnp.asarray(np.random.default_rng(seed).normal(0.0, 1.0, (len(nodes), 3)))
    gJ = np.asarray(jax.grad(E.J)(lg))
    crit = E.critics(lg)
    res = dict(gamma=gamma, horizon=horizon, n_nodes=len(nodes), n_traj=E.n, J=float(E.J(lg)),
               grad_norm=float(np.linalg.norm(gJ)), estimators={})
    for X in ("h", "hs", "s_TD", "s_MC", "s_timed"):
        g = np.asarray(E.expected_grad(lg, crit[X]))
        res["estimators"][X] = dict(rel_bias_norm=float(np.linalg.norm(g - gJ) / np.linalg.norm(gJ)),
                                    cos_to_true=float(np.dot(g.ravel(), gJ.ravel())
                                                      / (np.linalg.norm(g) * np.linalg.norm(gJ))))
    res["mc"] = monte_carlo(E, lg, crit, lambda x: (x,), gamma, mc_n, seed)
    return res


# ----------------------------------------------------------------------------- Monte-Carlo through production code
def monte_carlo(E: Enumerated, lg, crit, vjp, gamma, ns, seed):
    """Sample episodes exactly (trajectory index ~ Pr(tau)), push them through algorithm.pg_terms (the production
    actor loss) with the exact critic of each estimator, and compare the sample-mean gradient with the exact
    expectation. Returns per-estimator max |z| (z = (mean - exact) / SE) and the error decay with N."""
    P = np.asarray(E.traj_prob(E.logp(lg)))
    rng = np.random.default_rng(seed + 1)
    out = {}
    for X in ("h", "hs", "s_TD"):
        exact = np.asarray(vjp(E.expected_grad(lg, crit[X]))[0]).ravel()
        v_all = jnp.asarray(crit[X])

        def per_ep_grad(i):
            ep = jax.tree.map(lambda x: x[i], E.episodes(np.arange(E.n)))
            def f(lgx):
                logp = E.logp(lgx)[jnp.asarray(E.node)[i, :-1]]
                return pg_terms(logp, ep, v_all[i], gamma)[0]
            return -jnp.ravel(vjp(jax.grad(f)(lg))[0])     # production `pol` is the NEGATIVE PG surrogate
        G = np.asarray(jax.jit(jax.vmap(per_ep_grad))(jnp.arange(E.n)))   # gradient per distinct trajectory
        rows = []
        for N in ns:
            idx = rng.choice(E.n, size=N, p=P / P.sum())
            g = G[idx]
            m, se = g.mean(0), g.std(0, ddof=1) / np.sqrt(N)
            z = np.where(se > 0, (m - exact) / np.maximum(se, 1e-300), 0.0)
            rows.append(dict(N=N, max_abs_err=float(np.max(np.abs(m - exact))), max_abs_z=float(np.max(np.abs(z))),
                             mean_se=float(np.mean(se))))
        exp_check = float(np.max(np.abs((P / P.sum()) @ G - exact)))   # exact mean of per-episode grads
        out[X] = dict(rows=rows, exact_mean_of_production_grads_err=exp_check,
                      per_episode_grad_var=float(np.sum((P / P.sum()) @ (G - exact) ** 2)))
    return out


# ----------------------------------------------------------------------------- Good/Bad (App. B.2)
def good_bad_check(gamma, horizon=60, n=200000, seed=0):
    """Closed forms (29)-(48) + our value-gap/gradient-bias claims; MC of the asym-s gradient through pg_terms."""
    VG, VB = 1 / (1 - gamma), 1 / (2 * (1 - gamma))
    bG = 2 / 3
    EsV = bG * VG + (1 - bG) * VB
    Vh = 1 + gamma * EsV                                   # V(h=G) = 1[o=G] + g E_{s|h} V(s)
    res = dict(gamma=gamma, V_G=VG, V_B=VB, E_s_given_hG_V=EsV, E_closed=5 / (6 * (1 - gamma)), V_hG=Vh,
               V_hG_closed=(6 - gamma) / (6 * (1 - gamma)), gap=Vh - EsV)
    # gradient: pi(a=G|o) = p_o = sigmoid(th_o); J_H = sum_t g^t (3 p_G / 4 + p_B / 4) (actions do not move s)
    th = jnp.array([0.4, -0.3])                            # (th_G, th_B), arbitrary
    pG, pB = jax.nn.sigmoid(th)
    disc = (1 - gamma ** horizon) / (1 - gamma)
    dJ = np.asarray(jax.grad(lambda t: disc * (0.75 * jax.nn.sigmoid(t[0]) + 0.25 * jax.nn.sigmoid(t[1])))(th))
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, n)                              # 0 = G, 1 = B; frozen
    o = np.where(s[:, None] == 0, 0, rng.integers(0, 2, (n, horizon)))
    p = np.where(o == 0, float(pG), float(pB))
    a = (rng.random((n, horizon)) >= p).astype(int)        # a = 0 (G) w.p. p_o
    r = (a == 0).astype(float)
    vs = np.where(s == 0, VG, VB)[:, None] * np.ones((1, horizon + 1))
    ep = Episode(states=jnp.asarray(np.repeat(s[:, None], horizon + 1, 1)), obs=jnp.asarray(np.pad(o, ((0, 0), (0, 1)))),
                 actions=jnp.asarray(np.pad(a, ((0, 0), (0, 1)))), rewards=jnp.asarray(r),
                 mask=jnp.ones((n, horizon), bool), trunc=jnp.ones((n,), bool), length=jnp.full((n,), horizon))

    def one(e, v):
        def f(t):
            lp1 = jax.nn.log_sigmoid(t)                    # log p_o for a = G
            lp0 = jax.nn.log_sigmoid(-t)
            logp = jnp.stack([lp1[e.obs[:-1]], lp0[e.obs[:-1]]], -1)
            return pg_terms(logp, e, v, gamma)[0]
        return -jax.grad(f)(th)
    G = np.asarray(jax.jit(jax.vmap(one))(ep, jnp.asarray(vs)))
    m, se = G.mean(0), G.std(0, ddof=1) / np.sqrt(n)
    res.update(dJ=dJ.tolist(), asym_s_mc_grad=m.tolist(), asym_s_mc_se=se.tolist(),
               z=((m - dJ) / se).tolist())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "runs", "toys", "toys.json"))
    args = ap.parse_args()
    out = {"mhh_closed_form": [], "mhh_revisit": [], "good_bad": []}
    for gamma, q in itertools.product([0.9, 0.99], [0.05, 0.3, 0.5, 0.9]):    # sources.yaml toys grids
        r = mhh_closed_form_check(gamma, q, 0.8)
        out["mhh_closed_form"].append(r)
        e = r["estimators"]
        print(f"[METRIC] MHH g={gamma} q={q}: J={r['J']:.6f} (closed {r['J_closed']:.6f}) "
              f"relbias_q h={e['h']['rel_bias'][0]:+.1e} hs={e['hs']['rel_bias'][0]:+.1e} "
              f"sTD={e['s_TD']['rel_bias'][0]:+.4f} (closed {r['rel_bias_sTD_q_closed']:+.4f}) "
              f"sMC={e['s_MC']['rel_bias'][0]:+.4f} timed={e['s_timed']['rel_bias'][0]:+.4f}", flush=True)
    for gamma in (0.9, 0.99):
        r = mhh_revisit_check(gamma, horizon=6)
        out["mhh_revisit"].append(r)
        print(f"[METRIC] MHH-R g={gamma} H=6 nodes={r['n_nodes']} traj={r['n_traj']}: " + " ".join(
            f"{k}={v['rel_bias_norm']:.2e}" for k, v in r["estimators"].items()), flush=True)
    for gamma in (0.9, 0.99):
        r = good_bad_check(gamma)
        out["good_bad"].append(r)
        print(f"[METRIC] Good/Bad g={gamma}: gap={r['gap']:.12f} z(asym-s MC vs dJ)={np.round(r['z'], 2).tolist()}",
              flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[INFO] wrote {args.out}")


if __name__ == "__main__":
    main()

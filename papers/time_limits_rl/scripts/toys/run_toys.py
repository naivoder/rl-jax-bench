"""toy stage runner: learners and environments vs the exact ground truth .

Writes runs/toys/toys.json (+ per-seed metrics under runs/toys/<exp>/<agent>/seed_<n>/metrics.json). CPU-only
(tabular / tiny Monte-Carlo; logged as a deliberate CPU placement). The closed-form unit checks are in
tests/test_grads.py; the oracle diffs in tests/test_oracle_tonic.py; this script adds what needs learning or sampling:
  G1-E1  Last Moment: TA tabular Q-learning reaches gamma^(T-1); memoryless REINFORCE reaches max_p J(p) (MC error);
         the time-unaware Q-learner's greedy value.
  G1-E2  Two-Goal: tabular Q-learning (standard / TA / PEB, 10 seeds, 4e5 episodes) converges to the three exact
         tables (max |dQ|; per-entry bias z-test across seeds); greedy arrows vs Figure 1b-d.
  G1-MC  Our JAX environments reproduce the exact values by Monte-Carlo: Queue of Cars (TA optimum, best memoryless)
         through the runner-side time limit, Difficult Gridworld eps-greedy return.
Usage: uv run python scripts/toys/run_toys.py
"""
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from rl_jax_bench.jaxlog import Timer, info, metric, write_json  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
from src import exact as X  # noqa: E402
from src.train import exp_key, git_sha_at_start  # noqa: E402

import yaml  # noqa: E402

OUT = os.environ.get("TOYS_OUT", os.path.join(REPO, "runs", "toys"))  # override for verification reruns
#: every E1/E2 constant comes from a source-tagged config (values identical to the toy-stage run)
C1 = yaml.safe_load(open(os.path.join(REPO, "src", "configs", "e1_last_moment.yaml")))
C2 = yaml.safe_load(open(os.path.join(REPO, "src", "configs", "e2_two_goal.yaml")))
assert C1["seeds"] == C2["seeds"]
SEEDS = C1["seeds"]   # paper §1


def tabarrays(tab):
    return (jnp.asarray(tab.NX, jnp.int32), jnp.asarray(tab.R, jnp.float32),
            jnp.asarray(np.asarray(tab.is_terminal)[np.asarray(tab.NX)], jnp.float32),
            jnp.asarray(tab.start, jnp.float32))


def run_tabq(tab, agents, T, gamma, eps, n_episodes, lr_power, exp_id):
    NX, R, TN, st = tabarrays(tab)
    names = [(a, s) for a in agents for s in SEEDS]
    ag = jnp.asarray([a for a, _ in names], jnp.int32)
    sd = jnp.asarray([s for _, s in names], jnp.int32)
    f = jax.jit(jax.vmap(lambda s, a: A.tabular_q(exp_key(s, exp_id), NX, R, TN, st, agent=a, T=T, gamma=gamma,
                                                 eps=eps, n_episodes=n_episodes, lr_power=lr_power)))
    out = jax.device_get(f(sd, ag))
    return names, out


# ------------------------------------------------------------------ G1-E1 Last Moment
def gate_e1():
    lm = D.last_moment()
    res = {}
    for gamma in C1["gammas"]:
        for T in C1["T_values"]:
            names, out = run_tabq(lm, [A.AGENT_TA, A.AGENT_STANDARD], T, gamma, C1["epsilon"], C1["episodes"],
                                  C1["lr_power"], C1["exp_id"])
            Q = np.asarray(out.Q)
            ta_vals, std_vals = [], []
            for i, (a, s) in enumerate(names):
                if a == A.AGENT_TA:
                    pi = jnp.stack([X.greedy_onehot(jnp.asarray(Q[i, t], jnp.float64)) for t in range(T)])
                    ta_vals.append(float(X.policy_value_finite(lm, gamma, pi)[0]))
                else:
                    pj = float(np.argmax(Q[i, 0, 0]) == 1)  # greedy deterministic jump prob in A
                    std_vals.append(float(X.last_moment_J(pj, T, gamma)))
            th, _ = jax.vmap(lambda s: A.reinforce_jump(exp_key(s, C1["exp_id"]), T=T, gamma=gamma,
                                                         n_episodes=C1["reinforce_episodes"],
                                                         lr=C1["reinforce_lr"]))(jnp.asarray(SEEDS))
            p = np.asarray(jax.nn.sigmoid(th), np.float64)
            J = np.array([X.last_moment_J(pp, T, gamma) for pp in p])
            pb, Jb = X.last_moment_best_memoryless(T, gamma)
            ta_opt = gamma ** (T - 1)
            key = f"gamma{gamma}_T{T}"
            res[key] = dict(ta_optimum=ta_opt, ta_learned=ta_vals, ta_pass=bool(np.allclose(ta_vals, ta_opt, atol=1e-9)),
                            best_memoryless=float(Jb), best_p=float(pb), reinforce_p=p.tolist(), reinforce_J=J.tolist(),
                            reinforce_J_mean=float(J.mean()), reinforce_gap_to_best=float(Jb - J.mean()),
                            reinforce_pass=bool(Jb - J.mean() < 0.01 and np.all(J <= Jb + 1e-12)),
                            std_q_greedy_value=std_vals, ta_minus_best_memoryless=float(ta_opt - Jb))
            metric("G1-E1", case=key, ta=float(np.mean(ta_vals)), ta_opt=ta_opt, reinforce=float(J.mean()),
                   best_memoryless=float(Jb), std_greedy=float(np.mean(std_vals)))
    return res


# ------------------------------------------------------------------ G1-E2 Two-Goal
def gate_e2(N_EP=C2["episodes"], agents_out=True):
    tg = D.two_goal(C2["bump_reward"])
    g, T = C2["gamma"], C2["T"]
    pb, _ = X.bootstrap_fraction(tg, T)
    ref = {A.AGENT_STANDARD: np.asarray(X.infinite_q(tg, g, pb=pb)), A.AGENT_PEB: np.asarray(X.infinite_q(tg, g)),
           A.AGENT_TA: np.asarray(X.finite_horizon_q(tg, g, T))}
    t = Timer()
    names, out = run_tabq(tg, [A.AGENT_STANDARD, A.AGENT_TA, A.AGENT_PEB], T, g, C2["epsilon"], N_EP, C2["lr_power"],
                          C2["exp_id"])
    info("E2 tabular Q-learning", seconds=round(t.lap(), 1), lanes=len(names))
    Q, N = np.asarray(out.Q, np.float64), np.asarray(out.N)
    ng = ~tg.is_terminal
    lab = {A.AGENT_STANDARD: "standard", A.AGENT_TA: "time_aware", A.AGENT_PEB: "peb"}
    res = {}
    m = lambda V: float(np.asarray(V)[ng].mean())  # noqa: E731
    for ag in (A.AGENT_STANDARD, A.AGENT_TA, A.AGENT_PEB):
        idx = [i for i, (a, _) in enumerate(names) if a == ag]
        errs, arrows, vT, vinf = [], [], [], []
        if ag == A.AGENT_TA:
            tabs = [(t_, ref[ag][T - t_]) for t_ in range(T)]
        else:
            tabs = [(0, ref[ag])]
        dev = []
        for i in idx:
            e, ar, dv = 0.0, [], []
            for t_, R_ in tabs:
                vis = (N[i, t_].sum(1) > 0) & ng
                e = max(e, float(np.abs(Q[i, t_] - R_)[vis].max()))
                dv.append(np.where(vis[:, None], Q[i, t_] - R_, np.nan))
                # arrows: on visited cells whose exact best action is unique with a gap > thr, learned argmax agrees.
                # thr = 0.1 for TA / PEB (their targets are noise-free here: deterministic env); 1.0 for the standard
                # agent, whose targets carry Bernoulli(p_b) bootstrap noise (per-entry sd ~0.65 at 4e5 episodes)
                thr = 1.0 if ag == A.AGENT_STANDARD else 0.1
                gap = np.sort(R_, 1)[:, -1] - np.sort(R_, 1)[:, -2]
                ok = vis & (gap > thr)
                ar.append(float((np.argmax(Q[i, t_], 1)[ok] == np.argmax(R_, 1)[ok]).mean()))
            errs.append(e)
            arrows.append(float(np.min(ar)))
            dev.append(np.stack(dv))
            if ag == A.AGENT_TA:
                pi = jnp.stack([X.greedy_onehot(jnp.asarray(Q[i, t_])) for t_ in range(T)])
            else:
                pi = jnp.stack([X.greedy_onehot(jnp.asarray(Q[i, 0]))] * T)
            vT.append(m(X.policy_value_finite(tg, g, pi)))
            vinf.append(m(X.policy_value_infinite(tg, g, pi[0])))
        dev = np.stack(dev)  # (seeds, tables, S, A)
        mean_dev = np.nanmean(dev, 0)
        se = np.nanstd(dev, 0, ddof=1) / np.sqrt(len(idx))
        z = mean_dev / np.where(se > 0, se, np.nan)
        zf = z[np.isfinite(z)]
        res[lab[ag]] = dict(max_abs_dq_per_seed=errs, max_abs_dq_max=float(max(errs)),
                            pass_tol_0_05=bool(max(errs) < 0.05),
                            mean_bias_max_abs=float(np.nanmax(np.abs(mean_dev))),
                            bias_z_frac_gt3=float((np.abs(zf) > 3).mean()), bias_z_max=float(np.abs(zf).max()),
                            n_entries=int(len(zf)), arrow_agreement_min=float(min(arrows)),
                            value_T3=vT, value_inf=vinf)
        metric("G1-E2", agent=lab[ag], max_dq=float(max(errs)), mean_bias=float(np.nanmax(np.abs(mean_dev))),
               z_frac_gt3=float((np.abs(zf) > 3).mean()), arrows=float(min(arrows)),
               v_T3=float(np.mean(vT)), v_inf=float(np.mean(vinf)))
        for k, i in enumerate(idx if agents_out else []):
            write_json(os.path.join(OUT, "e2_two_goal", lab[ag], f"seed_{names[i][1]}", "metrics.json"),
                       dict(exp="e2_two_goal", agent=lab[ag], seed=names[i][1], git=git_sha_at_start(),
                            max_abs_dq=errs[k], value_T3=vT[k], value_inf=vinf[k], Q=Q[i, :T if ag == A.AGENT_TA else 1],
                            visits=N[i, :T if ag == A.AGENT_TA else 1],
                            config=dict(T=T, gamma=g, eps=C2["epsilon"], episodes=N_EP, lr_power=C2["lr_power"],
                                        bump_reward=C2["bump_reward"])))
    return res


# ------------------------------------------------------------------ G1-MC environments by Monte-Carlo
def gate_mc():
    res = {}
    out = D.qoc_outcomes(D.QoCParams())
    vstar, P = X.qoc_optimal(out, 0.99, 15)
    env = D.queue_of_cars()
    Pj = jnp.asarray(np.where(P == 0.5, 0.0, P), jnp.float32)
    n_ep = 400000

    def episode(key, ptab):
        st, o, _ = D.tl_reset_obs(env, jax.random.fold_in(key, 0), 15, 0)

        def body(c, t):
            st, alive, G = c
            x = st.env_state
            a = (jax.random.uniform(jax.random.fold_in(jax.random.fold_in(key, 1), t)) < ptab[t, x]).astype(jnp.int32)
            s = D.tl_step(env, jax.random.fold_in(jax.random.fold_in(key, 2), t), st, a, 15, 0)
            G = G + jnp.where(alive, 0.99 ** t * s.reward, 0.0)
            alive = jnp.logical_and(alive, jnp.logical_not(s.reset))
            return (s.state, alive, G), s.reset

        (_, _, G), resets = jax.lax.scan(body, (st, jnp.bool_(True), 0.0), jnp.arange(15))
        return G, resets.sum()

    keys = jax.random.split(jax.random.PRNGKey(123), n_ep)
    for name, tab, want in (("qoc_ta_optimum", Pj, vstar),
                            ("qoc_memoryless_0.4708", jnp.full((15, 9), 0.4708, jnp.float32), 0.398990)):
        G, nres = jax.jit(jax.vmap(lambda k: episode(k, tab)))(keys)
        G = np.asarray(G, np.float64)
        mc, se = float(G.mean()), float(G.std(ddof=1) / np.sqrt(n_ep))
        res[name] = dict(exact=float(want), mc=mc, se=se, z=(mc - want) / se, n=n_ep, pass_=bool(abs(mc - want) < 4 * se))
        metric("G1-MC", case=name, exact=float(want), mc=mc, se=se)
    # Difficult Gridworld: eps-greedy (first-argmax) w.r.t. Q*, T = 200, via the tabular env
    dg = D.difficult_grid()
    Qs = np.asarray(X.infinite_q(dg, 1.0, iters=2000))
    want = float(X.dgrid_eps_greedy_return(dg, Qs, 0.1, 200, D.DGRID_START))
    NX = jnp.asarray(dg.NX, jnp.int32)
    gq = jnp.asarray(np.argmax(Qs, 1), jnp.int32)

    def dep(key):
        def body(c, t):
            s, alive, Gm = c
            k1, k2 = jax.random.split(jax.random.fold_in(key, t))
            a = jnp.where(jax.random.uniform(k1) < 0.1, jax.random.randint(k2, (), 0, 4), gq[s])
            s2 = NX[s, a]
            Gm = Gm - alive.astype(jnp.float32)
            alive = jnp.logical_and(alive, s2 != D.DGRID_GOAL)
            return (s2, alive, Gm), None
        (_, _, Gm), _ = jax.lax.scan(body, (jnp.int32(D.DGRID_START), jnp.bool_(True), 0.0), jnp.arange(200))
        return Gm

    n_ep = 200000
    G = np.asarray(jax.jit(jax.vmap(dep))(jax.random.split(jax.random.PRNGKey(7), n_ep)), np.float64)
    mc, se = float(G.mean()), float(G.std(ddof=1) / np.sqrt(n_ep))
    res["dgrid_eps_greedy"] = dict(exact=want, mc=mc, se=se, z=(mc - want) / se, n=n_ep, pass_=bool(abs(mc - want) < 4 * se))
    metric("G1-MC", case="dgrid_eps_greedy", exact=want, mc=mc, se=se)
    return res


def main():
    info("toys start", device=str(jax.devices()), note="tabular/MC on CPU by design (JAX_PLATFORMS=cpu)")
    t = Timer()
    res = dict(git=git_sha_at_start(), e1=gate_e1(), mc=gate_mc(), e2=gate_e2())
    # convergence check of the standard agent: 10x the episodes must shrink its residual (noise + max-) bias
    e2l = gate_e2(N_EP=C2["episodes_long"], agents_out=False)
    res["e2_long_4e6"] = e2l
    res["seconds"] = t.lap()
    write_json(os.path.join(OUT, "toys.json"), res)
    info("toys done", seconds=round(res["seconds"], 1), out=os.path.join(OUT, "toys.json"))


if __name__ == "__main__":
    main()

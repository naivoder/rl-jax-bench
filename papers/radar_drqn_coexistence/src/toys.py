"""toy checks (toy correctness) -> runs/toys/toys.json.

  toy check 1 exact solver = Table D1 (tests/test_grads.py::test_exact_mdp, re-run here and its numbers recorded)
  toy check 2 JAX env under SAA on FH = Table II FH-SAA row 60 / 40 / 64.0 / 100 exactly
  toy check 3 Monte-Carlo J of pi*, SAA, random on the JAX env within 3 SE of the exact values (10 seeds x 70k steps,
       p in {0.1, 0.2, 0.4, 0.6, 0.8} and FH, both readings)
  toy check 4/5 Eq. 11 / 14 targets by hand, check_grads (float64): the pytest files, results recorded
  toy check 6 tabular double-Q on the 2-state toy (p = 0.4, R_eq3) vs Q*; the DQN / DDQN / DDRQN cores on the one-hot toy
       recover pi* = (B, B)
  Oracles: tests/test_official_parity.py (CleanRL, pomdp-baselines) results recorded.
Usage: python src/toys.py [--out runs/toys]
"""
from __future__ import annotations

import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import algorithm as alg  # noqa: E402
import baselines as bl  # noqa: E402
import data as D  # noqa: E402
import train as TR  # noqa: E402
from rl_jax_bench.jaxlog import Timer, check_device, info, metric, write_json  # noqa: E402
from model import arm_config  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def mc_checks(cfg) -> dict:
    """toy check 2 + toy check 3: per-seed J (mean reward over 70k eval pulses) of pi*, SAA, random vs exact stationary values."""
    exps = ["mc_p01", "mc_p02", "mc_p04", "mc_p06", "mc_p08", "fh"]
    cfg = dict(cfg, eval_cpis=70)
    lanes, ids = TR.build_lanes(cfg, exps, list(D.READINGS), list(range(10)))
    tab_pi, _, _ = TR.exact_tables(ids, cfg)
    nl = len(ids)
    tab_saa = np.tile(D.ASTAR_ACTION, (nl, D.N_SEG, 1)).astype(np.int32)
    full = jnp.zeros((nl, D.N_POS * D.N_VEL * D.N_THETA), jnp.int32)
    out = {}
    ev = {m: TR.make_policy_eval(70, "random" if m == "random" else "table") for m in ("pistar", "saa", "random")}
    res = {m: np.asarray(ev[m](lanes, jnp.asarray(tab_pi if m == "pistar" else tab_saa), full)[0]) for m in ev}
    ok_all = True
    for e in exps:
        ex = D.experiment(e)
        for r in D.READINGS:
            idx = [i for i, (ee, rr, _) in enumerate(ids) if ee == e and rr == r]
            R = D.reward_table(r)
            sol = bl.exact_solve(ex["train"], R)
            exact = dict(pistar=bl.stationary_J(ex["train"], R, sol["pi"]),
                         saa=bl.stationary_J(ex["train"], R, bl.saa_policy(ex["train"])),
                         random=bl.stationary_J(ex["train"], R, None))
            row = {}
            for m in ("pistar", "saa", "random"):
                J = res[m][idx, :, 0].sum(1) / (70 * D.CPI)
                mean, se = float(J.mean()), float(J.std(ddof=1) / np.sqrt(len(J)))
                z = abs(mean - exact[m]) / se if se > 0 else (0.0 if abs(mean - exact[m]) < 1e-6 else np.inf)
                ok = z <= 3.0
                ok_all &= ok
                row[m] = dict(mc_mean=mean, mc_se=se, exact=exact[m], z=z, ok=bool(ok))
            out[f"{e}/{r}"] = row
            metric("toy check 3", exp=e, reading=r, **{f"{m}_z": round(row[m]["z"], 2) for m in row})
    # toy check 2: FH SAA Table-II row from the JAX env
    idx = [i for i, (ee, rr, _) in enumerate(ids) if ee == "fh" and rr == "eq3"]
    o = res["saa"][idx]                                          # [seeds, 70, 8]
    n = 70 * D.CPI
    row = dict(coll_pct=100 * o[:, :, 1].sum(1) / n, missed_pct=100 * o[:, :, 2].sum(1) / n,
               bw_mhz=20 * o[:, :, 3].sum(1) / n, adapt_pct=100 * o[:, :, 4].sum(1) / (n - 1))
    fh = {k: [float(v.min()), float(v.max())] for k, v in row.items()}
    fh_ok = (np.allclose(row["coll_pct"], 60) and np.allclose(row["missed_pct"], 40) and np.allclose(row["bw_mhz"], 64)
             and np.allclose(row["adapt_pct"], 100))
    metric("toy check 2 FH-SAA (min,max over seeds)", **{k: v for k, v in fh.items()})
    return dict(mc=out, mc_ok=bool(ok_all), fh_saa=fh, fh_saa_ok=bool(fh_ok))


def tabular_double_q(n_steps: int, power: float, avg_from: float, n_seeds: int = 10, p: float = 0.4,
                     reading: str = "eq3") -> dict:
    """Double Q-learning (van Hasselt 2010) on the latent 2-state chain with uniform behaviour; step size
    1 / n(s,a)^power per table; Polyak average of (QA+QB)/2 over steps >= avg_from * n_steps."""
    R = jnp.asarray(D.reward_table(reading), jnp.float32)
    pat = jnp.asarray(D.markov_chain(p).pattern)
    sol = bl.exact_solve(D.markov_chain(p), D.reward_table(reading))
    t_avg = int(avg_from * n_steps)

    def one(key):
        def body(c, t):
            Q, N, z, acc = c
            k = jax.random.fold_in(key, t)
            k1, k2, k3 = jax.random.split(k, 3)
            a = jax.random.randint(k1, (), 0, D.N_ACT)
            z2 = jnp.where(jax.random.uniform(k2) < p, 1 - z, z)
            r = R[a, pat[z2]]
            i = (jax.random.uniform(k3) < 0.5).astype(jnp.int32)
            j = 1 - i
            N = N.at[i, z, a].add(1.0)
            ap = jnp.argmax(Q[i, z2])
            Q = Q.at[i, z, a].add((r + 0.9 * Q[j, z2, ap] - Q[i, z, a]) / N[i, z, a] ** power)
            # running mean (a float32 running SUM of ~1e6 values near 7 loses ~0.5 of resolution)
            kk = jnp.maximum(t - t_avg + 1, 1).astype(jnp.float32)
            acc = jnp.where(t >= t_avg, acc + (0.5 * (Q[0] + Q[1]) - acc) / kk, acc)
            return (Q, N, z2, acc), None

        c0 = (jnp.zeros((2, 2, D.N_ACT)), jnp.zeros((2, 2, D.N_ACT)), jnp.int32(0), jnp.zeros((2, D.N_ACT)))
        (Q, N, z, acc), _ = jax.lax.scan(body, c0, jnp.arange(n_steps))
        final = 0.5 * (Q[0] + Q[1])
        avg = acc if t_avg < n_steps else final
        return final, avg

    keys = jax.random.split(jax.random.PRNGKey(1234), n_seeds)
    final, avg = jax.jit(jax.vmap(one))(keys)
    final, avg = np.asarray(final, np.float64), np.asarray(avg, np.float64)
    est = avg if t_avg < n_steps else final
    err_seed = np.abs(est - sol["Q"][None]).max((1, 2))
    err_mean = float(np.abs(est.mean(0) - sol["Q"]).max())
    pol = [tuple(int(x) for x in q.argmax(1)) for q in est]
    return dict(n_steps=n_steps, power=power, avg_from=avg_from, err_per_seed=err_seed.tolist(), err_seed_mean=err_mean,
                policies=pol, pistar=[int(x) for x in sol["pi"]])


def core_on_toy(arm: str, cfg: dict, reading: str, n_steps: int = 100_000, n_seeds: int = 10, p: float = 0.4,
                **override) -> dict:
    """The arm's core on the theta_obs-only one-hot toy (obs = one_hot(z), 2-d), uniform behaviour, Table-I hypers.
    Greedy policy probed every 1000 steps (zero carry, one step); reports agreement with pi* = (B, B) over the last
    20 % of training, and the final max |Q - Q*|."""
    q0 = arm_config(arm, cfg)
    q = dataclasses.replace(q0, obs_dim=2, **override)
    R = jnp.asarray(D.reward_table(reading), jnp.float32)
    pat = jnp.asarray(D.markov_chain(p).pattern)
    sol = bl.exact_solve(D.markov_chain(p), D.reward_table(reading))
    net = alg.make_network(q)
    L = q.trace_len
    probe_obs = jnp.eye(2, dtype=jnp.float32)
    pistar = jnp.asarray(sol["pi"], jnp.int32)

    def one(key):
        def body(c, t):
            st, buf, wr, z, ac = c
            k = jax.random.fold_in(key, t)
            k1, k2, k3 = jax.random.split(k, 3)
            a = jax.random.randint(k1, (), 0, D.N_ACT)
            z2 = jnp.where(jax.random.uniform(k2) < p, 1 - z, z)
            r = R[a, pat[z2]]
            o, o2 = jax.nn.one_hot(z, 2), jax.nn.one_hot(z2, 2)
            qa, ac = alg.act_q(net, q, st.params, o, ac, t % L == 0)          # act-time carry, reset every L
            act_ok = (alg.greedy(qa) == pistar[z]).astype(jnp.float32)
            wr, buf, _ = alg.writer_step(wr, buf, o, a, r, q.gamma, o2, t % L == 0, (t + 1) % L == 0)

            def learn(s):
                s, _ = alg.update(s, alg.buffer_sample(buf, k3, q.batch_size), q)
                return alg.sync(s, alg.sync_due(s, q.target_period))

            st = jax.lax.cond((t + 1) // L >= q.batch_size, learn, lambda s: s, st)
            qp, _ = alg.act_q(net, q, st.params, probe_obs, alg.initial_carry(q, (2,)))
            return (st, buf, wr, z2, ac), (act_ok, qp)

        c0 = (alg.init(jax.random.fold_in(key, 10 ** 7), q), alg.buffer_init(q), alg.writer_init(q), jnp.int32(0),
              alg.initial_carry(q))
        _, (act_ok, qp) = jax.lax.scan(body, c0, jnp.arange(n_steps))
        return qp[999::1000], act_ok[int(0.8 * n_steps):].mean()

    tm = Timer()
    qps, act_agree = jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(99), n_seeds))
    qps, act_agree = np.asarray(qps), np.asarray(act_agree)                                     # [S, n/1000, 2, 15]
    secs = tm.lap()
    last = qps[:, int(0.8 * qps.shape[1]):]
    agree = (last.argmax(-1) == sol["pi"][None, None]).all(-1).mean(1)          # per seed fraction of probes = pi*
    final_err = np.abs(qps[:, -1] - sol["Q"][None]).max((1, 2))
    gap_err = np.abs((qps[:, -1] - qps[:, -1].max(-1, keepdims=True)) - (sol["Q"] - sol["Q"].max(-1, keepdims=True))[None]).max((1, 2))
    return dict(arm=arm, reading=reading, n_steps=n_steps, seconds=secs, override={k: v for k, v in override.items()},
                act_agree_last20=act_agree.tolist(), act_agree_mean=float(act_agree.mean()), agree_last20=agree.tolist(),
                agree_mean=float(agree.mean()), final_maxabs_err=final_err.tolist(), final_adv_err=gap_err.tolist(),
                final_policy=[tuple(int(x) for x in pq.argmax(-1)) for pq in qps[:, -1]],
                pistar=[int(x) for x in sol["pi"]], action_gap=sol["gap"].tolist())


def pytest_result(path: str) -> dict:
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", path], cwd=ROOT, env=env, capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
    return dict(path=path, returncode=r.returncode, summary=tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "toys"))
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()
    check_device()
    cfg = TR.load_cfg(os.path.join(ROOT, "src", "configs", "default.yaml"))
    tm = Timer()
    res = dict(mc=mc_checks(cfg))
    info("toy check 2/toy check 3 done", seconds=round(tm.lap(), 1), mc_ok=res["mc"]["mc_ok"], fh_saa_ok=res["mc"]["fh_saa_ok"])
    res["tabular_declared_1_over_n_200k"] = tabular_double_q(200_000, 1.0, 1.0)
    res["tabular_poly06_avg_2M"] = tabular_double_q(2_000_000, 0.6, 0.5)
    for k in ("tabular_declared_1_over_n_200k", "tabular_poly06_avg_2M"):
        metric("toy check 6 tabular", variant=k, err_seed_mean=round(res[k]["err_seed_mean"], 4),
               err_max_seed=round(max(res[k]["err_per_seed"]), 4), policies=str(set(res[k]["policies"])))
    info("tabular done", seconds=round(tm.lap(), 1))
    res["core_toy"] = []
    # Table-I hyperparameters (as the learners run), then the convergence control (lr 1e-4, 10x buffer, 200k steps):
    # the control isolates implementation correctness from the optimiser/buffer noise of the Table-I setting.
    variants = [("tableI", {}, 100_000), ("control", dict(lr=1e-4, buffer_traces=None), 200_000)]
    for vname, ov, n in variants:
        for reading in ("eq3", "lin"):
            for arm in ("dqn", "ddqn", "ddrqn"):
                o = dict(ov)
                if "buffer_traces" in o:
                    o["buffer_traces"] = 20_000 // arm_config(arm, cfg).trace_len
                r = core_on_toy(arm, cfg, reading, n_steps=n, **o)
                r["variant"] = vname
                res["core_toy"].append(r)
                metric("toy check 6 core toy", variant=vname, arm=arm, reading=reading, act_agree=round(r["act_agree_mean"], 3),
                       probe_agree=round(r["agree_mean"], 3), final_adv_err_max=round(max(r["final_adv_err"]), 4),
                       final_policy=str(set(r["final_policy"])), seconds=round(r["seconds"], 1))
    if not args.skip_pytest:
        res["pytest"] = [pytest_result(p) for p in ("tests/test_grads.py", "tests/test_official_parity.py", "tests/test_smoke.py")]
        for r in res["pytest"]:
            info("pytest", **r)
    write_json(os.path.join(args.out, "toys.json"), res)
    info("toys written", path=os.path.join(args.out, "toys.json"))


if __name__ == "__main__":
    main()

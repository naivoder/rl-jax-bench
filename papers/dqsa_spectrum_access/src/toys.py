"""Toy checks: closed-form / exactly computable checks against the implementation.
    python src/toys.py [--skip-t1] ; writes runs/toys/toys.json
Every check records (name, ours, truth, tolerance, PASS)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import algorithm as alg  # noqa: E402
import data as D  # noqa: E402
import eval as E  # noqa: E402
import model as M  # noqa: E402
from rl_jax_bench import drqn  # noqa: E402
from rl_jax_bench import jaxlog  # noqa: E402

ROOT = os.path.dirname(SRC)
RES = []


def check(name, ours, truth, tol, note=""):
    ok = bool(abs(ours - truth) <= tol)
    RES.append(dict(name=name, ours=float(ours), truth=float(truth), tol=float(tol), ok=ok, note=note))
    jaxlog.metric("toys", check=name, ours=round(float(ours), 6), truth=round(float(truth), 6), tol=round(float(tol), 6), ok=ok)
    return ok


def t_aloha():
    se_max = 0
    for n in range(3, 12):
        spec = D.EnvSpec(n_min=n, n_max=n)
        r = E.eval_baseline(spec, "aloha", n_exp=1000, T=200, chunk=500)   # 2e5 slots
        u = r["util_t"]
        # slots are independent for Aloha; SE over experiments of the per-experiment mean is conservative-equivalent
        se = u.mean(1).std(ddof=1) / math.sqrt(u.shape[0])
        se_max = max(se_max, se)
        check(f"T-ALOHA n={n}", u.mean(), E.aloha_expected(n), 3 * se, f"2e5 slots, SE {se:.5f}")
    spec = D.EnvSpec()      # n ~ U{3..11}
    r = E.eval_baseline(spec, "aloha", n_exp=4000, T=200, chunk=500)
    u = r["util_t"].mean(1)
    se = u.std(ddof=1) / math.sqrt(len(u))
    check("T-ALOHA mean over U{3..11}", u.mean(), E.aloha_expected_mean(), 3 * se, "4000 experiments x 200 slots")
    # multichannel generalisation (N=20,K=10 and N=100,K=50)
    for N, K in ((20, 10), (100, 50)):
        spec = D.EnvSpec(n_min=N, n_max=N, n_channels=K, k_local=K, subset="all")
        r = E.eval_baseline(spec, "aloha", n_exp=200, T=100, chunk=100)
        u = r["util_t"].mean(1)
        se = u.std(ddof=1) / math.sqrt(len(u))
        check(f"T-ALOHA multichannel N={N} K={K}", u.mean(), E.aloha_multichannel(N, K), 3 * se)


def t_tdma_genie():
    for n, K in ((7, 1), (4, 2), (20, 10)):
        spec = D.EnvSpec(n_min=n, n_max=n, n_channels=K, k_local=K, subset="all")
        r = E.eval_baseline(spec, "tdma_genie", n_exp=20, T=60, chunk=20)
        check(f"T-GENIE utilization n={n} K={K} (Thm 2: collision+idle free)", r["util_t"].mean(), 1.0, 1e-9)
        check(f"T-GENIE collisions n={n} K={K}", r["coll_t"].mean(), 0.0, 1e-9)
        # equal shares: every user succeeds K*T/n times (+-1)
        sh = r["succ_tot"]
        act = r["active"].astype(bool)
        check(f"T-GENIE max share spread n={n} K={K}", float((sh.max(1) - sh.min(1)).max()), 0.0, 1.0)


def t_t2():
    spec = D.EnvSpec(n_min=2, n_max=2, n_channels=2, k_local=2, subset="all")
    n_exp = 100_000
    r = E.eval_baseline(spec, "stay_redraw", n_exp=n_exp, T=10, chunk=10_000)
    u = r["util_t"]
    for t in range(1, 11):
        p = 1 - 2.0 ** (-t)
        se = math.sqrt(max(p * (1 - p), 1e-12) / n_exp)
        check(f"T2 E[u({t})] = 1-2^-{t}", u[:, t - 1].mean(), p, 3 * se + 1e-9)
    # discounted per-user return, T=200
    r = E.eval_baseline(spec, "stay_redraw", n_exp=100_000, T=200, chunk=10_000)
    g = 0.95 ** np.arange(200)
    per = (r["util_t"] * g[None]).sum(1)
    se = per.std(ddof=1) / math.sqrt(len(per))
    exact = sum(0.95 ** (t - 1) * (1 - 2.0 ** (-t)) for t in range(1, 2000))
    check("T2 discounted per-user return (19.047619)", per.mean(), exact, 3 * se, "exact sum over t=1..1999")
    check("T2 exact value", exact, 20 - 0.5 / (1 - 0.95 / 2), 1e-9)


def t_t3():
    spec = D.EnvSpec(n_min=2, n_max=2, n_channels=1, k_local=1)
    for p in (0.2, 0.5, 0.8):
        r = E.eval_baseline(spec, "aloha", n_exp=2000, T=100, chunk=1000, p=p)
        n = r["util_t"].size
        t = 2 * p * (1 - p)
        check(f"T3 N=2 K=1 p={p}: 2p(1-p)", r["util_t"].mean(), t, 3 * math.sqrt(t * (1 - t) / n))


def t_unit():
    # T4 double-Q target arithmetic
    q1 = jnp.array([[1.0, 3.0, 2.0]]); q2 = jnp.array([[0.5, 1.0, 4.0]])
    y = float(drqn.td_target(q1, q2, jnp.array([1.0]), jnp.array([0.95]), True)[0])
    ym = float(drqn.td_target(q1, q2, jnp.array([1.0]), jnp.array([0.95]), False)[0])
    check("T4 double-Q target (1.95)", y, 1.95, 1e-6)
    check("T4 max target (4.8)", ym, 4.8, 1e-6)
    # dueling identity mean_a Q = V
    net = M.make_net(3)
    params = M.init_params(net, jax.random.PRNGKey(0), 6)
    h = jax.random.normal(jax.random.PRNGKey(1), (32, 100))
    q = net.apply(params, h, method=M.DQSANet.heads)
    v = net.apply(params, h, method=M.DQSANet.value)
    check("T4 dueling identity max|mean_a Q - V|", float(jnp.abs(q.mean(-1) - v).max()), 0.0, 1e-5)
    # T5 PF shaped reward identity
    T = 6
    pat = np.zeros((T, 2), bool)
    pat[[0, 1, 3], 0] = True; pat[[2, 5], 1] = True
    succ = np.zeros(2)
    tot = 0.0
    for t in range(T):
        succ = succ + pat[t]
        tot += float(alg.reward_vec("prop_fair", jnp.array(pat[t]), jnp.ones(2, bool), jnp.array(succ))[0])
    H = lambda m: sum(1.0 / i for i in range(1, m + 1))
    check("T5 PF shaped reward sum (3.333333 = H_3 + H_2)", tot, H(3) + H(2), 1e-6)
    approx = math.log(3) + math.log(2) + 2 * 0.5772156649
    gap = tot - approx
    ok = (1 / (2 * 4) + 1 / (2 * 3)) < gap < (1 / 6 + 1 / 4)
    RES.append(dict(name="T5 log-approximation gap inside the [56] bound", ours=gap, truth=float("nan"), tol=0, ok=bool(ok), note="bound (0.2917, 0.4167)"))
    # competitive / sum reward on the same pattern
    s = jnp.array(pat[0]); a = jnp.ones(2, bool)
    check("reward competitive vec[0]", float(alg.reward_vec("competitive", s, a, jnp.ones(2))[0]), 1.0, 0)
    check("reward sum_rate (both succeed -> 2)", float(alg.reward_vec("sum_rate", jnp.array([True, True]), a, jnp.ones(2))[1]), 2.0, 0)
    # Eq. 11
    p = M.eq11_probs(jnp.array([19.0, 20.0]), 0.05, 1.0)
    check("Eq.11 pi(tx) with Q=(19,20), alpha .05, beta 1 (0.719506)", float(p[1]), 0.95 * math.e / (1 + math.e) + 0.025, 1e-6)
    check("Eq.11 sums to 1", float(p.sum()), 1.0, 1e-6)
    # fixed-pi policy evaluation in the env: V^pi = q/(1-gamma)
    q = float(p[1])
    spec = D.EnvSpec(n_min=1, n_max=1, n_channels=1, k_local=1)
    r = E.eval_baseline(spec, "aloha", n_exp=20000, T=300, chunk=5000, p=q)
    g = 0.95 ** np.arange(300)
    ret = (r["util_t"] * g[None]).sum(1)
    se = ret.std(ddof=1) / math.sqrt(len(ret))
    check("T1-PE: discounted return under fixed pi_tx=q equals q/(1-gamma)", ret.mean(), q / 0.05, 3 * se + 1e-3, "env + discount + Eq. 11 prob")


def t_rate():
    spec = D.EnvSpec(n_min=1000, n_max=1000, n_channels=50, k_local=2, subset="random", capacity="rate", fading=True)
    st = jax.vmap(lambda k: D.init_state(spec, k))(jax.random.split(jax.random.PRNGKey(5), 40))
    r = np.asarray(D.rate_mbps(spec, st.h)).ravel()
    se = r.std(ddof=1) / math.sqrt(len(r))
    check("T-RATE Rayleigh mean rate (215.957 Mbps)", r.mean(), 215.957, 3 * se, f"{len(r)} draws, SE {se:.3f}")
    rr = np.asarray(D.rate_mbps(spec, st.h)).reshape(-1, 2)
    best = rr.max(1)
    se2 = best.std(ddof=1) / math.sqrt(len(best))
    check("T-RATE best-of-2 mean rate (235.893 Mbps)", best.mean(), 235.893, 3 * se2)
    # AR(1): lag-1 correlation and stationarity after many steps
    w = D.World(jnp.ones((1000,), bool), jnp.int32(1000), jnp.zeros((1000, 2), jnp.int32))
    s0 = jax.tree.map(lambda x: x[0], st)
    def body(s, k):
        s2, _ = D.step_global(spec, w, s, jnp.zeros((1000,), bool), jnp.zeros((1000,), jnp.int32), k)
        return s2, (s2.h)
    sN, hs = jax.lax.scan(body, s0, jax.random.split(jax.random.PRNGKey(6), 50))
    h = np.concatenate([np.asarray(s0.h)[None], np.asarray(hs)], 0)       # [51,N,K,2]
    z = h[..., 0] + 1j * h[..., 1]
    num = np.real((z[1:] * np.conj(z[:-1])).mean())
    den = (np.abs(z) ** 2).mean()
    se = np.real((z[1:] * np.conj(z[:-1]))).std() / math.sqrt(z[1:].size) / den
    check("T-RATE AR(1) lag-1 correlation J0(2pi*0.1)=0.903713", num / den, spec.rho, 3 * se + 1e-3, f"SE {se:.5f}")
    check("T-RATE stationary E|h|^2 after 50 steps (=1)", float((np.abs(z[-1]) ** 2).mean()), 1.0, 3 * float((np.abs(z[-1]) ** 2).std() / math.sqrt(z[-1].size)))
    check("T-RATE rho = scipy J0", spec.rho, 0.903713, 1e-5)


def t_shk():
    """Secondary (third party): shkrwnd/Deep-Reinforcement-Learning-for-Dynamic-Spectrum-Access env at ATTEMPT_PROB = 1 vs ours."""
    path = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(ROOT, "external", "official")), "shkrwnd", "multi_user_network_env.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run sh scripts/official/setup.sh or set OFFICIAL_ROOT")
    sp = importlib.util.spec_from_file_location("shk_env", path)
    mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    N, K = 5, 3
    env = mod.env_network(N, K, 1.0)
    spec = D.EnvSpec(n_min=N, n_max=N, n_channels=K, k_local=K, subset="all")
    w = D.World(jnp.ones((N,), bool), jnp.int32(N), jnp.broadcast_to(jnp.arange(K, dtype=jnp.int32), (N, K)))
    st = D.init_state(spec, jax.random.PRNGKey(0))
    step = jax.jit(lambda a, k: D.step_local(spec, w, st, a, k)[1]["success"])
    mism = 0
    n = 10_000
    for i in range(n):
        a = rng.integers(0, K + 1, N).astype(np.int32)
        obs = env.step(a.copy())
        theirs = np.array([o[1] for o in obs[:N]]) > 0.5
        ours = np.asarray(step(jnp.asarray(a), jax.random.PRNGKey(i)))
        mism += int((theirs != ours).any())
    check("T-SHK success vectors identical on 1e4 random joint actions (mismatching draws)", mism, 0, 0, "third-party env, ATTEMPT_PROB=1")


def t_patterns():
    """Metric-definition oracles: scripted E6 / E7 patterns must be recognised as 1.0, random Aloha play as ~0."""
    for kind, name, N in (("E6", "scripted_e6", 4), ("E7", "scripted_e7", 3)):
        spec = D.EnvSpec(n_min=N, n_max=N, n_channels=2, k_local=2, subset="all")
        r = E.eval_baseline(spec, name, n_exp=50, T=200, chunk=50)
        s = E.summarize(spec, r, (60, 200), kind)
        check(f"T-PATTERN {kind} scripted pattern frequency (1)", s[f"pattern_{kind}"], 1.0, 0.0)
        check(f"T-PATTERN {kind} scripted utilization (1)", s["plateau_util"], 1.0, 1e-9)
        check(f"T-PATTERN {kind} scripted strict Pareto (1)", s["pareto_strict"], 1.0, 0.0)
        r = E.eval_baseline(spec, "aloha", n_exp=200, T=200, chunk=200)
        s = E.summarize(spec, r, (60, 200), kind)
        check(f"T-PATTERN {kind} random Aloha pattern frequency (0)", s[f"pattern_{kind}"], 0.0, 0.0)
    # Jain index: equal shares -> 1, one winner -> 1/n
    spec = D.EnvSpec(n_min=4, n_max=4, n_channels=2, k_local=2, subset="all")
    r = E.eval_baseline(spec, "tdma_genie", n_exp=10, T=200, chunk=10)
    check("T-PATTERN Jain index of genie TDMA (1)", float(r["jain_all"].mean()), 1.0, 1e-3)
    r = E.eval_baseline(spec, "scripted_e6", n_exp=10, T=200, chunk=10)
    check("T-PATTERN Jain index with 2 of 4 users served (0.5)", float(r["jain_all"].mean()), 0.5, 1e-6)


def t_grads():
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", os.path.join(ROOT, "tests", "test_grads.py")],
                         capture_output=True, text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
    ok = out.returncode == 0
    RES.append(dict(name="T-GRAD check_grads (float64, order 2) + full-vector gradient == scalar TD gradient + by-hand loss", ours=float(ok), truth=1.0, tol=0, ok=ok, note=out.stdout.strip().splitlines()[-1]))
    jaxlog.metric("toys", check="T-GRAD", ok=ok, tail=out.stdout.strip().splitlines()[-1])


def t1_learner(lr=1e-3, iters=6000):
    """T1: single user, one channel, competitive.  Q*(tx)=1/(1-g)=20, Q*(idle)=g/(1-g)=19 for the trained dueling-LSTM DQN."""
    spec = D.EnvSpec(name="T1", n_min=1, n_max=1, n_channels=1, k_local=1)
    tc = alg.TrainCfg(reward="competitive", iterations=iters, lr=lr)
    f = alg.make_train_fns(spec, tc)
    qs = f.init(jax.random.PRNGKey(0))
    tk = jax.random.PRNGKey(1)
    chunk = jax.jit(lambda q, s: f.run_chunk(q, s, 250, tk))
    t0 = time.time()
    for s in range(0, iters, 250):
        qs, met = chunk(qs, s)
    # probe: histories of successful transmissions (x = [onehot(tx), cap, ack=1]) and of silence
    x_tx = jnp.array([[0.0, 1.0, 1.0, 1.0]]); x_0 = jnp.array([[1.0, 0.0, 1.0, 0.0]])
    first = jnp.array([[1.0, 0.0, 1.0, 0.0]])
    unroll = M.make_unroll_fn(f.net)
    seq = jnp.concatenate([first, jnp.repeat(x_tx, 49, 0)], 0)[None]
    q, _ = unroll(qs.params, seq, None)
    q = np.asarray(q[0])
    for t in (1, 25, 49):
        check(f"T1 Q(tx) at t={t + 1} (20)", q[t, 1], 20.0, 0.5, f"lr {lr}, {iters} its")
        check(f"T1 Q(idle) at t={t + 1} (19)", q[t, 0], 19.0, 0.5)
    check("T1 gap Q(tx)-Q(idle) at t=50 (1)", q[49, 1] - q[49, 0], 1.0, 0.25)
    jaxlog.info("T1 done", seconds=round(time.time() - t0, 1), loss=float(met["loss"][-1]))
    return q


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-t1", action="store_true")
    ap.add_argument("--t1-lr", type=float, default=1e-3)
    ap.add_argument("--t1-iters", type=int, default=6000)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    jaxlog.check_device()
    groups = dict(aloha=t_aloha, genie=t_tdma_genie, t2=t_t2, t3=t_t3, unit=t_unit, pattern=t_patterns, rate=t_rate, shk=t_shk, grads=t_grads)
    if not a.skip_t1:
        groups["t1"] = lambda: t1_learner(a.t1_lr, a.t1_iters)
    for k, fn in groups.items():
        if a.only and k != a.only:
            continue
        t0 = time.time()
        fn()
        jaxlog.info("group done", group=k, seconds=round(time.time() - t0, 1))
    out = os.path.join(ROOT, "runs", "toys", "toys.json" if not a.only else f"toys_{a.only}.json")
    jaxlog.write_json(out, dict(checks=RES, n=len(RES), n_pass=sum(r["ok"] for r in RES), all_pass=all(r["ok"] for r in RES)))
    jaxlog.info("TOY CHECKS", n=len(RES), passed=sum(r["ok"] for r in RES), all_pass=all(r["ok"] for r in RES), file=out)

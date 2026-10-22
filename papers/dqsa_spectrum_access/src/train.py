"""DQSA training driver.  One process trains S seeds at once (vmapped over the seed axis; one top-level PRNGKey per seed),
checkpoints with orbax every `--ckpt-every` iterations (resumable), logs [INFO]/[METRIC]/[WARN]/[ERROR], separates first-compile
time from steady-state throughput, guards NaN/Inf inside jit, then evaluates every seed on the frozen test protocol and writes
<out>/<tag>/seed_<n>/metrics.json.

    python src/train.py --exp E7 --seeds 100 101 102 103 104 --lr 3e-4 --out runs/small

PRNG key tree (per seed s):  root = PRNGKey(s);  init_key, train_key, eval_key = split(root, 3)
    init_key  -> network params
    train_key -> fold_in(train_key, iteration i) -> split -> (env_key_i, act_key_i)
        env_key_i -> fold_in(m) per episode m -> split -> (world: clique size / channel subsets, fading init, PU init, per-slot fading+PU noise)
        act_key_i -> fold_in(m) per episode -> fold_in(t) per slot -> Eq. 11 draws for the N users
    eval_key  -> (reserved; eval worlds use the CONSTANT eval root of eval.py so all arms/seeds share worlds; action keys fold in the replicate)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402
import yaml  # noqa: E402

from rl_jax_bench import jaxlog  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import algorithm as alg  # noqa: E402
import data as D  # noqa: E402
import eval as E  # noqa: E402
import model as M  # noqa: E402

ROOT = os.path.dirname(SRC)


def load_exp(name: str) -> dict:
    with open(os.path.join(SRC, "configs", "experiments.yaml")) as f:
        return yaml.safe_load(f)[name]


def sha_files(*names):
    h = hashlib.sha256()
    for n in names:
        h.update(open(os.path.join(SRC, n), "rb").read())
    return h.hexdigest()[:16]


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "none"


def git_dirty():
    """True if tracked source files differ from HEAD (a run launched from a dirty tree is not reproducible from the recorded sha)."""
    try:
        out = subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no", "--", "src", "scripts"], text=True,
                                      stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return None


def launch_provenance(seeds, argv):
    """provenance captured at LAUNCH (not at the end of the run): git sha, dirty flag, source hashes, the vmapped seed group.
    Per-seed outcomes depend on the vmapped group (batched reductions are not bit-identical across group sizes, and the dynamics are
    chaotic), so the group is part of the identity of a result."""
    return dict(git_commit=git_commit(), git_dirty=git_dirty(), env_sha=sha_files("data.py"), algo_sha=sha_files("algorithm.py", "model.py"),
                eval_sha=sha_files("eval.py"), train_sha=sha_files("train.py"), vmapped_group_seeds=list(seeds), n_vmapped_seeds=len(seeds),
                argv=list(argv), jax_version=jax.__version__, backend=jax.default_backend(), launched_unix=time.time())


def build(exp: str, reward: str | None, overrides: dict, tc_over: dict):
    cfg = load_exp(exp)
    spec = D.EnvSpec(**{**cfg["env"], **overrides})
    tc = alg.TrainCfg(reward=reward or cfg["reward"], **tc_over)
    return cfg, spec, tc


def tree_index(tree, i):
    return jax.tree.map(lambda x: x[i], tree)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--reward", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--iters", type=int, default=10_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--monitor-every", type=int, default=1000)
    ap.add_argument("--n-exp", type=int, default=None, help="override final test experiments")
    ap.add_argument("--epochs", type=int, default=1, help="ablation: full-vector regression epochs per iteration (primary 1)")
    ap.add_argument("--sched-iters", type=int, default=None, help="alpha/beta ramp length (continuation: 10000)")
    ap.add_argument("--lstm", type=int, default=100, help="LSTM units (paper 100; tests use a tiny net)")
    ap.add_argument("--terminal", action="store_true", help="ablation no bootstrap at t=T")
    ap.add_argument("--no-dueling", action="store_true")
    ap.add_argument("--reward-scale-by-n", action="store_true", help="ablation divide rewards by n_max")
    ap.add_argument("--argmax-eval", action="store_true", help="ablation greedy test actions")
    ap.add_argument("--fresh-opt", action="store_true", help="re-init the Adam moments every iteration (literal Matlab trainNetwork call); with --epochs > 1")
    ap.add_argument("--l2", type=float, default=0.0, help="L2 factor on weight matrices (Matlab default 1e-4); 0 = off")
    ap.add_argument("--reward-scale", type=float, default=None, help="multiply rewards by this factor (ablation); combine with --rescale-beta")
    ap.add_argument("--rescale-beta", action="store_true", help="divide beta (ramp and test) by --reward-scale so the Eq. 11 policy semantics are unchanged")
    ap.add_argument("--act-rule", default="eq11", choices=["eq11", "argmax_eps"], help="ablation (third-party): training acting rule; paper = eq11")
    ap.add_argument("--act-eps", type=float, default=0.02)
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args(argv)

    jaxlog.check_device()
    cfg, spec, tc0 = build(a.exp, a.reward, {}, {})
    tc_over = dict(lstm=a.lstm, iterations=a.iters, lr=a.lr, grad_steps=a.epochs, sched_iters=a.sched_iters, bootstrap_at_T=not a.terminal, dueling=not a.no_dueling, act_rule=a.act_rule, act_eps=a.act_eps)
    tc_over.update(fresh_opt=a.fresh_opt, l2=a.l2, rescale_beta=a.rescale_beta)
    if a.reward_scale_by_n:
        tc_over["reward_scale"] = 1.0 / spec.n_max
    if a.reward_scale is not None:
        tc_over["reward_scale"] = a.reward_scale
    if (a.reward or "").endswith("_episodic") or load_exp(a.exp)["reward"].endswith("_episodic"):
        tc_over["bootstrap_at_T"] = False                                   # the episodic reward is terminal at T
    cfg, spec, tc = build(a.exp, a.reward, {}, tc_over)
    tag = a.tag or f"{a.exp}_{tc.reward}_lr{a.lr:g}" + (f"_ep{a.epochs}" if a.epochs != 1 else "") + ("_fresh" if a.fresh_opt else "") + (f"_l2{a.l2:g}" if a.l2 else "") + (f"_rs{a.reward_scale:g}" if a.reward_scale else "") + ("_rsb" if a.rescale_beta else "") + ("_argmaxeps" if a.act_rule != "eq11" else "")
    out_dir = os.path.join(a.out, tag)
    os.makedirs(out_dir, exist_ok=True)
    S = len(a.seeds)
    prov = launch_provenance(a.seeds, sys.argv[1:] if argv is None else argv)
    lp = os.path.join(out_dir, "launch.json")
    if os.path.exists(lp):                                  # resume: keep the first launch, append this one
        first = json.load(open(lp))
        first.setdefault("resumes", []).append(prov)
        prov_first = first
    else:
        prov_first = prov
    jaxlog.write_json(lp, prov_first)
    if prov["git_dirty"]:
        jaxlog.warn("launched from a DIRTY tree: results are not reproducible from the recorded git sha", sha=prov["git_commit"])
    jaxlog.info("provenance", git=prov["git_commit"], dirty=prov["git_dirty"], algo_sha=prov["algo_sha"], group=str(a.seeds))
    jaxlog.info("run", exp=a.exp, tag=tag, seeds=str(a.seeds), iters=a.iters, lr=a.lr, reward=tc.reward, spec=str(spec))
    # budget assertions (README): the paper's training budget unless a deviation is declared on the CLI
    if a.iters != 10_000:
        jaxlog.warn("iterations differ from the paper's R=10000", iters=a.iters)
    assert (tc.episodes, tc.horizon, tc.target_every) == (16, 50, 5)

    fns = alg.make_train_fns(spec, tc)
    roots = jax.vmap(jax.random.PRNGKey)(jnp.array(a.seeds))
    ks = jax.vmap(lambda k: jax.random.split(k, 3))(roots)
    init_keys, train_keys = ks[:, 0], ks[:, 1]
    qs = jax.vmap(fns.init)(init_keys)

    chunk_n = a.chunk

    @jax.jit
    def run_chunk(qs, start):
        qs, met = jax.vmap(lambda q, tk: fns.run_chunk(q, start, chunk_n, tk))(qs, train_keys)
        jaxlog.nan_guard((qs.params, met), met, tag=0, dump_dir=os.path.join(out_dir, "nan_dumps"))
        return qs, met

    ck_dir = os.path.abspath(os.path.join(out_dir, "ckpt"))
    mgr = ocp.CheckpointManager(ck_dir, options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True))
    start, history, monitor = 0, [], []
    if mgr.latest_step() is not None:
        step = mgr.latest_step()
        restored = mgr.restore(step, args=ocp.args.StandardRestore(jax.tree.map(np.asarray, dict(qs=qs))))
        qs = jax.tree.map(jnp.asarray, restored["qs"])
        qs = drqn_qstate(qs)
        start = step
        hp = os.path.join(out_dir, "history.json")
        if os.path.exists(hp):
            hist = json.load(open(hp))
            history = [h for h in hist["history"] if h["it_end"] <= step]
            monitor = [m for m in hist["monitor"] if m["it"] <= step]
        jaxlog.info("RESUMED", step=step)

    t_compile, t_steady, n_steady = None, 0.0, 0
    it = start
    while it < a.iters:
        t0 = time.perf_counter()
        qs, met = run_chunk(qs, it)
        met = jax.tree.map(np.asarray, met)
        dt = time.perf_counter() - t0
        if t_compile is None:
            t_compile = dt
            jaxlog.info("first chunk (compile + run)", seconds=round(dt, 2))
        else:
            t_steady += dt
            n_steady += chunk_n
        it += chunk_n
        row = dict(it_end=it, **{k: float(v[:, -1].mean()) for k, v in met.items()},
                   train_util_mean=float(met["train_util"].mean()),
                   per_seed_train_util=met["train_util"].mean(1).tolist())
        history.append(row)
        jaxlog.metric("train", it=it, loss=row["loss"], q_mean=row["q_mean"], train_util=row["train_util_mean"],
                      alpha=row["alpha"], beta=row["beta"], its_per_s=(n_steady / t_steady if t_steady else 0.0))
        if it % a.monitor_every == 0 or it >= a.iters:
            mon, mon_par, mon_pf = [], [], []
            for si in range(S):
                r = E.eval_learned(spec, tc, tree_index(qs.params, si), n_exp=64, T=cfg["test"]["T"], replicate=0, chunk=64,
                                   world_offset=E.MONITOR_WORLD_OFFSET)      # monitor on worlds disjoint from the reporting worlds
                mon.append(float(r["util_t"][:, cfg["test"]["window"][0]:].mean()))
                mon_par.append(float(r["pareto_strict"].mean()))              # budget rule reads this (README)
                mon_pf.append(float(r["pareto_feas_strict"].mean()))
            monitor.append(dict(it=it, test_util=mon, pareto_strict=mon_par, pareto_feas_strict=mon_pf))
            jaxlog.metric("monitor", it=it, test_util_mean=float(np.mean(mon)), per_seed=str([round(m, 3) for m in mon]))
        if it % a.ckpt_every == 0 or it >= a.iters:
            mgr.save(it, args=ocp.args.StandardSave(jax.tree.map(np.asarray, dict(qs=qs))))
            mgr.wait_until_finished()
            json.dump(dict(history=history, monitor=monitor), open(os.path.join(out_dir, "history.json"), "w"))
    throughput = (n_steady / t_steady) if t_steady else None
    jaxlog.info("training done", compile_s=round(t_compile or 0, 2), steady_iters_per_s=throughput)

    if a.no_eval:
        return
    tcfg = cfg["test"]
    n_exp = a.n_exp or tcfg["n_exp"]
    for si, seed in enumerate(a.seeds):
        params = tree_index(qs.params, si)
        r = E.eval_learned(spec, tc, params, n_exp=n_exp, T=tcfg["T"], replicate=seed, greedy=a.argmax_eval)
        summ = E.summarize(spec, r, tuple(tcfg["window"]), tcfg["kind"])
        # Q values on a probe: all-zero input after the first step (diagnostic)
        sd = os.path.join(out_dir, f"seed_{seed}")
        os.makedirs(sd, exist_ok=True)
        flat = {"/".join(str(getattr(k, "key", k)) for k in p): np.asarray(v) for p, v in jax.tree_util.tree_flatten_with_path(params)[0]}
        np.savez(os.path.join(sd, "params.npz"), **flat)
        jaxlog.write_json(os.path.join(sd, "metrics.json"), dict(
            seed=seed, exp=a.exp, tag=tag, reward=tc.reward, lr=a.lr, iterations=a.iters, argmax_eval=a.argmax_eval,
            train_cfg=tc.__dict__, env_spec=spec.__dict__ | dict(rho=spec.rho),
            env_sha=prov_first["env_sha"], algo_sha=prov_first["algo_sha"], eval_sha=prov_first["eval_sha"],     # hashes at LAUNCH 
            git_dirty=prov_first["git_dirty"], vmapped_group_seeds=prov_first["vmapped_group_seeds"], launch_provenance=prov_first,
            config_sha=hashlib.sha256(json.dumps([tc.__dict__, spec.__dict__], sort_keys=True, default=str).encode()).hexdigest()[:16],
            git_commit=prov_first["git_commit"], git_commit_at_end=git_commit(), compile_s=t_compile, steady_iters_per_s=throughput, n_vmapped_seeds=S,
            history_train_util=[h["per_seed_train_util"][si] for h in history],
            history_it=[h["it_end"] for h in history], monitor=[dict(it=m["it"], test_util=m["test_util"][si], pareto_strict=m.get("pareto_strict", [None] * S)[si],
                          pareto_feas_strict=m.get("pareto_feas_strict", [None] * S)[si]) for m in monitor],
            final=summ))
        jaxlog.metric("final", seed=seed, plateau_util=summ["plateau_util"], plateau_coll=summ["plateau_coll"],
                      plateau_idle=summ["plateau_idle"], pareto_strict=summ["pareto_strict"], jain_all=summ["jain_all"],
                      **{k: v for k, v in summ.items() if k.startswith("pattern") or k.startswith("throughput")})


def drqn_qstate(tree):
    """Restore the QState namedtuple after an orbax round trip (orbax returns plain dict/list containers)."""
    from rl_jax_bench import drqn
    if isinstance(tree, drqn.QState):
        return tree
    return drqn.QState(*tree) if isinstance(tree, (tuple, list)) else drqn.QState(**tree)


if __name__ == "__main__":
    main()

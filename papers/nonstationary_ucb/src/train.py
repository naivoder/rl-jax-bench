"""Experiment driver: run every policy of a config over replications for given seeds and write metrics.

    uv run python -m src.train --config src/configs/e1.yaml --seeds 0,1,2,3,4 --reps 2000 --out runs/e1

One process; replications are vmapped; seeds and policies run sequentially. Each seed writes
``<out>/seed_<n>/metrics.json`` (per-replication final regret and arm-1 counts, curve summaries, env hashes,
CRN fingerprint, git SHA, compile time and throughput). Runs longer than one chunk can checkpoint the scan
state with orbax at chunk boundaries (``--chunk``/``--ckpt``) and resume bit-identically.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time

import numpy as np
import yaml

from . import common
from .common import Timer, check_device, info, metric, warn, write_json
from . import algorithm as alg
from . import data

import jax  # noqa: E402  (after the bootstrap in src/__init__)
import jax.numpy as jnp  # noqa: E402


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def policy_params(name: str, pcfg: dict, K: int, T: int) -> tuple[alg.Params, dict, int]:
    """Params for one policy from its config block; returns (Params, plain dict for the record, buffer length)."""
    B = float(pcfg.get("B", 1.0))
    xi = float(pcfg.get("xi", 0.5))
    rec = {"B": B, "xi": xi}
    gamma, tau, g, alpha, buf_len, eta_mult = 1.0, 1, 0.0, 0.0, 1, 1.0
    if name in ("ducb", "ducb_lit"):
        gamma = float(pcfg["gamma"])
        rec["gamma"] = gamma
        rec["effective_memory"] = 1.0 / (1.0 - gamma) if gamma < 1.0 else float("inf")  # gamma = 1: ablation
    if name == "swucb":
        tau = int(pcfg["tau"])
        buf_len = int(pcfg.get("buf_len", tau))
        if not 1 <= tau <= buf_len:
            raise ValueError(f"tau={tau} must be in [1, buf_len={buf_len}]")
        rec["tau"] = tau
    if name == "exp3s":
        alpha = float(pcfg["alpha"])
        g = float(pcfg["g"]) if "g" in pcfg else alg.exp3s_g(K, float(pcfg["S"]), T)
        rec.update({"alpha": alpha, "g": g, "S": pcfg.get("S")})
        if "eta_mult" in pcfg:  # recheck RC3 EXTENSION arm only; absent = Auer's coupling (paper arm)
            eta_mult = float(pcfg["eta_mult"])
            rec["eta_mult"] = eta_mult
    return alg.make_params(B, xi, gamma, tau, g, alpha, eta_mult), rec, buf_len


def uniforms_fingerprint(k_env, K: int, n_rep: int = 3, n_t: int = 100) -> str:
    ts = jnp.arange(1, n_t + 1, dtype=jnp.int32)
    u = jax.vmap(lambda k: jax.vmap(lambda t: alg.env_uniforms(k, t, K))(ts))(k_env[:n_rep])
    return hashlib.sha256(np.asarray(u, dtype=np.float64).tobytes()).hexdigest()


def _summ(curves: np.ndarray) -> dict:
    return {"mean": curves.mean(0), "median": np.median(curves, 0),
            "p05": np.percentile(curves, 5, 0), "p95": np.percentile(curves, 95, 0)}


def run_policy(name, env: data.Env, params, buf_len, k_env, k_pol, *, tie: str, log_every: int,
               chunk_steps: int | None = None, ckpt_dir: str | None = None) -> dict:
    """Run one policy over all replications; returns final accumulators, curves and timings."""
    T, K, R = env.T, env.K, k_env.shape[0]
    chunk_steps = chunk_steps or T
    if T % log_every or chunk_steps % log_every or T % chunk_steps:
        raise ValueError(f"T={T}, chunk={chunk_steps} and log_every={log_every} must nest")
    means, resets = alg.as_device_env(env.means, env.resets)
    runner = alg.make_chunk_runner(name, K, tie, log_every)
    state, acc = alg.batched_init(name, K, R, buf_len)
    regs, arm1s, t_next = [], [], 1

    mgr = None
    if ckpt_dir:
        import orbax.checkpoint as ocp
        mgr = ocp.CheckpointManager(os.path.abspath(ckpt_dir), options=ocp.CheckpointManagerOptions(max_to_keep=2))
        last = mgr.latest_step()
        if last is not None:
            tmpl = {"state": state, "acc": acc, "reg": np.zeros((R, (last - 1) // log_every)),
                    "arm1": np.zeros((R, (last - 1) // log_every), np.int32)}
            rest = mgr.restore(last, args=ocp.args.StandardRestore(tmpl))
            state, acc = rest["state"], rest["acc"]
            regs, arm1s, t_next = [np.asarray(rest["reg"])], [np.asarray(rest["arm1"])], last
            info("resumed from orbax checkpoint", policy=name, t_next=t_next, dir=ckpt_dir)

    compile_s, run_s, n_chunks = None, 0.0, 0
    while t_next <= T:
        sl = slice(t_next - 1, t_next - 1 + chunk_steps)
        m = means[sl].reshape(-1, log_every, K)
        r = resets[sl].reshape(-1, log_every)
        args = (state, acc, k_env, k_pol, params, m, r, jnp.asarray(t_next, jnp.int32))
        if compile_s is None:
            tm = Timer()
            compiled = runner.lower(*args).compile()
            compile_s = tm.lap()
        tm = Timer()
        state, acc, reg, arm1 = compiled(*args)
        reg.block_until_ready()
        run_s += tm.lap()
        n_chunks += 1
        regs.append(np.asarray(reg))
        arm1s.append(np.asarray(arm1))
        t_next += chunk_steps
        if mgr is not None and t_next <= T:
            import orbax.checkpoint as ocp
            mgr.save(t_next, args=ocp.args.StandardSave({"state": state, "acc": acc, "reg": np.concatenate(regs, 1),
                                                         "arm1": np.concatenate(arm1s, 1)}))
            mgr.wait_until_finished()
    if mgr is not None:
        mgr.close()
    reg = np.concatenate(regs, 1)
    arm1 = np.concatenate(arm1s, 1)
    return {"acc": jax.tree_util.tree_map(np.asarray, acc), "reg_curves": reg, "arm1_curves": arm1,
            "compile_s": compile_s, "run_s": run_s, "n_chunks": n_chunks}


def run_experiment(cfg: dict, seed: int, R: int, *, policies=None, tie: str | None = None, out_dir: str | None = None,
                   chunk_steps: int | None = None, ckpt_root: str | None = None, keep_curves: bool = False) -> dict:
    """One seed (thin wrapper over ``run_experiment_seeds``; identical numbers)."""
    return run_experiment_seeds(cfg, [seed], R, policies=policies, tie=tie, out_dir=out_dir, chunk_steps=chunk_steps,
                                ckpt_root=ckpt_root, keep_curves=keep_curves)[0]


def run_experiment_seeds(cfg: dict, seeds, R: int, *, policies=None, tie: str | None = None, out_dir: str | None = None,
                         chunk_steps: int | None = None, ckpt_root: str | None = None, keep_curves: bool = False) -> list:
    """Run every policy of ``cfg`` for all ``seeds`` in ONE vmapped call per policy (batch = len(seeds) * R).

    The replication keys of seed s are exactly those of a single-seed run (``rep_keys(s, ...)``), so batching
    seeds changes no number, only wall-clock time (the scan is latency-bound; tested in tests/test_smoke.py).
    Writes one ``<out_dir>/seed_<s>/metrics.json`` per seed.
    """
    seeds = [int(s) for s in seeds]
    nS = len(seeds)
    env = data.from_config(cfg["env"], cfg["name"])
    T, K = env.T, env.K
    tie = tie or cfg.get("tie", "random")
    log_every = int(cfg.get("log_every", 10))
    policies = list(policies or cfg["policies"].keys())
    exp_id, cfg_idx = int(cfg["exp_id"]), int(cfg.get("cfg_idx", 0))
    st = data.self_test(env)
    msha = data.means_sha256(env)
    recs = []
    for seed in seeds:
        k_env0, _ = alg.rep_keys(seed, exp_id, cfg_idx, 0, R)
        recs.append({"experiment": cfg["name"], "label": cfg.get("label", ""), "seed": seed, "reps": R, "T": T, "K": K,
                     "tie": tie, "log_every": log_every, "config": cfg, "git_sha": common.git_sha(),
                     "git_dirty": common.git_dirty(), "device": str(jax.devices()[0]), "dtype_regret": "float64",
                     "batched_seeds": seeds,
                     "env": {"means_sha256": msha, "uniforms_fp_sha256": uniforms_fingerprint(k_env0, K), "self_test": st},
                     "t_grid": list(range(0, T + 1, log_every)), "policies": {}})
        info("experiment", name=cfg["name"], seed=seed, R=R, T=T, K=K, tie=tie, means_sha=msha[:12],
             u_fp=recs[-1]["env"]["uniforms_fp_sha256"][:12])
    usums = [{} for _ in seeds]
    extra = [{} for _ in seeds]
    zero = lambda a: np.concatenate([np.zeros((a.shape[0], 1), a.dtype), a], 1)  # noqa: E731
    for name in policies:
        algo = cfg["policies"][name].get("algo", name)  # config keys may alias an algorithm (labelled variants)
        params, prec, buf_len = policy_params(algo, cfg["policies"][name], K, T)
        if algo == "oracle_ucb1" and not env.breakpoints:
            warn("oracle_ucb1 needs abrupt breakpoints; skipped", experiment=cfg["name"])
            continue
        ks = [alg.rep_keys(s, exp_id, cfg_idx, alg.POLICY_ID[algo], R) for s in seeds]
        k_env = jnp.concatenate([k[0] for k in ks], 0)
        k_pol = jnp.concatenate([k[1] for k in ks], 0)
        ck = os.path.join(ckpt_root, "seeds_" + "-".join(map(str, seeds)), name) if ckpt_root else None
        out = run_policy(algo, env, params, buf_len, k_env, k_pol, tie=tie, log_every=log_every,
                         chunk_steps=chunk_steps, ckpt_dir=ck)
        thr = nS * R * T / out["run_s"]
        for i, seed in enumerate(seeds):
            sl = slice(i * R, (i + 1) * R)
            acc = jax.tree_util.tree_map(lambda a: a[sl], out["acc"])
            fin = acc.regret
            usums[i][name] = acc.usum
            rc, ac = zero(out["reg_curves"][sl]), zero(out["arm1_curves"][sl])
            at = {str(t): float(rc[:, t // log_every].mean()) for t in cfg.get("report_t", []) if t % log_every == 0}
            recs[i]["policies"][name] = {
                "label": cfg["policies"][name].get("label", alg.LABELS[algo]), "algo": algo, "params": prec,
                "final_regret_mean": float(fin.mean()), "final_regret_se_reps": float(fin.std(ddof=1) / np.sqrt(R)),
                "final_regret_median": float(np.median(fin)),
                "final_regret_p05_p95": [float(np.percentile(fin, 5)), float(np.percentile(fin, 95))],
                "final_arm1_mean": float(acc.arm1.mean()), "ntilde_mean": acc.ntilde.mean(0),
                "regret_at_mean": at, "final_regret": np.round(fin, 6), "final_arm1": acc.arm1,
                "curve": _summ(rc), "arm1_curve": _summ(ac.astype(np.float64)),
                "usum_sha256": hashlib.sha256(np.asarray(acc.usum).tobytes()).hexdigest(),
                "any_bad": bool(acc.bad.any()),
                "compile_s": out["compile_s"], "run_s": out["run_s"], "run_s_note": f"one call for {nS} seed(s)",
                "throughput_steps_reps_per_s": thr, "n_chunks": out["n_chunks"]}
            if keep_curves:
                extra[i][name] = rc
            metric("policy done", exp=cfg["name"], seed=seed, policy=name, regret_mean=float(fin.mean()),
                   regret_se=float(fin.std(ddof=1) / np.sqrt(R)), arm1=float(acc.arm1.mean()),
                   compile_s=out["compile_s"], run_s=out["run_s"], steps_reps_per_s=thr)
    for i, seed in enumerate(seeds):
        names = list(usums[i])
        recs[i]["crn_identical"] = all(np.array_equal(usums[i][names[0]], usums[i][n]) for n in names[1:])
        if not recs[i]["crn_identical"]:
            warn("CRN fingerprints differ across policies", experiment=cfg["name"], seed=seed)
        if out_dir:
            write_json(os.path.join(out_dir, f"seed_{seed}", "metrics.json"), recs[i])
        if keep_curves:
            recs[i]["_curves"] = extra[i]
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policies", default=None)
    ap.add_argument("--tie", default=None, choices=[None, "random", "lowest"])
    ap.add_argument("--chunk", type=int, default=None, help="rounds per chunk (default: T, one chunk)")
    ap.add_argument("--ckpt", default=None, help="orbax checkpoint root (enables resume at chunk boundaries)")
    ap.add_argument("--batch-seeds", action="store_true", help="vmap all seeds in one call per policy (same numbers)")
    a = ap.parse_args(argv)
    check_device()
    cfg = load_config(a.config)
    R = a.reps or int(cfg.get("reps_per_seed", 2000))
    pols = a.policies.split(",") if a.policies else None
    t0 = time.time()
    seeds = [int(x) for x in a.seeds.split(",")]
    if a.batch_seeds:
        run_experiment_seeds(cfg, seeds, R, policies=pols, tie=a.tie, out_dir=a.out, chunk_steps=a.chunk, ckpt_root=a.ckpt)
    else:
        for s in seeds:
            run_experiment(cfg, s, R, policies=pols, tie=a.tie, out_dir=a.out, chunk_steps=a.chunk, ckpt_root=a.ckpt)
    info("all seeds done", config=a.config, wall_s=time.time() - t0)
    os.makedirs(a.out, exist_ok=True)
    shutil.copy(a.config, os.path.join(a.out, "config.yaml"))


if __name__ == "__main__":
    main()

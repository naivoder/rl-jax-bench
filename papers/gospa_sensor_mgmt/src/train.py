"""train.py -- the experiment driver ("train" in the repo template; there is NOTHING TO TRAIN: the paper's method is a model-based
planner, no parameters are learned).  It runs the closed-loop simulation for one configuration and a set of algorithms:

    python -m src.train --cfg CFG-O --algs Myopic-GD MCTS3-GD --runs 10 --clutter 0.1 1 2 --tag small

All (run, clutter) lanes of one (cfg, algorithm) are one vmapped batch advanced in chunks of --chunk steps; after every chunk the
carry and the outputs so far are written with orbax (resumable: rerunning the same command continues from the last chunk).
Per-seed raw metrics land in <out>/<tag>/seed_<m>/metrics.json (per-step GOSPA^2 and parts, sensor tracks, counters).

Logging: structured [INFO]/[METRIC]/[WARN]/[ERROR] lines on stdout; first-compile time is logged separately from steady-state
throughput (AOT lower/compile, then timed chunks).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()                      # BEFORE importing jax (mandatory entry-point rule)

import numpy as np  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
assert jax.config.jax_enable_x64, "float64 must be on"
import jax.numpy as jnp  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402

from rl_jax_bench import jaxlog as JL  # noqa: E402
from src import model as M  # noqa: E402
from src import algorithm as A  # noqa: E402

DEFAULT_OUT = os.environ.get("GOSPA_RUNS", "runs")


def lane_layout(runs: int, clutter, m0: int = 0):
    """lane = c * runs + (m - m0) ; returns (ms[L], lams[L]); m0 = first absolute run index (--run-offset; extension runs)."""
    ms = np.tile(m0 + np.arange(runs), len(clutter))
    lams = np.repeat(np.asarray(clutter, float), runs)
    return ms, lams


def _write_shard(path: str, cell: dict):
    """one shard per (seed, cfg, algorithm): race-free between worker processes; scripts/merge_metrics.py folds the shards of a seed
    into the required runs/<tag>/seed_<m>/metrics.json."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cell, f)
    os.replace(tmp, path)


def shard_path(out_dir: str, tag: str, m: int, cfg_name: str, alg_name: str) -> str:
    safe = cfg_name.replace("+", "_").replace("=", "-").replace(",", "_")
    return os.path.join(out_dir, tag, f"seed_{m}", f"metrics__{safe}__{alg_name}.json")


def layout_key(cfg: M.Cfg, runs: int, clutter, m0: int, seed: int, steps: int, truth_set) -> str:
    """short hash of everything that fixes the meaning of a checkpoint's lanes : the lane layout (runs, clutter, first run
    index), the top-level seed, the horizon, the truth set, the configuration and the carry layout (counters)."""
    ts = "none" if truth_set is None else hashlib.sha256(np.ascontiguousarray(truth_set[0]).tobytes()).hexdigest()[:16]
    blob = json.dumps(dict(runs=int(runs), clutter=[float(c) for c in clutter], m0=int(m0), seed=int(seed), steps=int(steps), truths=ts,
                           cfg=cfg.hash(), counters=list(A.COUNTERS)), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def run_cell(cfg: M.Cfg, alg: M.Alg, runs: int, clutter, out_dir: str, chunk: int = 25, steps: int = M.N_STEPS, seed: int = 0,
             ckpt: bool = True, nan_mode: str = "warn", truth_set=None, m0: int = 0) -> dict:
    """Run one (cfg, algorithm) for all (run, clutter) lanes; returns the dict of numpy outputs [steps, lanes, ...]."""
    ms_np, lams_np = lane_layout(runs, clutter, m0)
    L = len(ms_np)
    ms, lams = jnp.asarray(ms_np), jnp.asarray(lams_np)
    runner = A.make_chunk_runner(cfg, alg, seed, nan_mode, truth_set=truth_set)
    carry = A.init_carry(cfg, L)
    ck_dir = os.path.abspath(os.path.join(out_dir, "ckpt", cfg.name, alg.name + "__" + layout_key(cfg, runs, clutter, m0, seed, steps, truth_set)))
    outs, k0 = None, 0
    cp = ocp.PyTreeCheckpointer()
    if ckpt and os.path.isdir(os.path.join(ck_dir, "state")):
        try:
            st = cp.restore(os.path.join(ck_dir, "state"))
            k0 = int(st["k0"])
            carry = jax.tree_util.tree_map(jnp.asarray, A.Carry(A.FL.FState(*[st["carry"]["fs"][i] for i in range(4)]), st["carry"]["spos"], st["carry"]["cnt"]))
            outs = {k: np.asarray(v) for k, v in st["outs"].items()}
            assert outs["gospa_sq"].shape == (k0, L) and np.asarray(carry.cnt).shape == (L, len(A.COUNTERS)), \
                f"checkpoint shape {outs['gospa_sq'].shape} / {np.asarray(carry.cnt).shape} does not match the lane layout ({k0}, {L})"
            JL.info("resumed", cfg=cfg.name, alg=alg.name, k0=k0)
        except Exception as e:  # noqa: BLE001
            JL.warn("checkpoint restore failed, starting from scratch", err=repr(e)[:200])
            k0, outs = 0, None
    t_c0 = time.time()
    ks0 = jnp.arange(k0, min(k0 + chunk, steps))
    lowered = runner.lower(carry, ks0, ms, lams)
    compiled = lowered.compile()
    t_compile = time.time() - t_c0
    JL.metric("first_compile_seconds", value=round(t_compile, 2), cfg=cfg.name, alg=alg.name, lanes=L)
    compiled_by_len = {len(ks0): compiled}
    t_steady, n_steady = 0.0, 0
    while k0 < steps:
        ks = jnp.arange(k0, min(k0 + chunk, steps))
        fn = compiled_by_len.get(len(ks))
        if fn is None:                                      # last, shorter chunk: compile separately (counted as compile)
            tc = time.time()
            fn = runner.lower(carry, ks, ms, lams).compile()
            compiled_by_len[len(ks)] = fn
            t_compile += time.time() - tc
        t0 = time.time()
        carry, o = fn(carry, ks, ms, lams)
        o = jax.tree_util.tree_map(np.asarray, jax.block_until_ready(o))
        dt = time.time() - t0
        t_steady += dt
        n_steady += len(ks)
        outs = o if outs is None else {k: np.concatenate([outs[k], o[k]], axis=0) for k in o}
        k0 += len(ks)
        if ckpt:
            st = dict(k0=np.asarray(k0), carry=dict(fs=list(carry.fs), spos=carry.spos, cnt=carry.cnt), outs=outs)
            os.makedirs(ck_dir, exist_ok=True)
            cp.save(os.path.join(ck_dir, "state"), st, force=True)
        JL.info("chunk", cfg=cfg.name, alg=alg.name, k=k0, sec_per_step=round(dt / len(ks), 4))
    cnt = np.asarray(carry.cnt)
    bad = {n: int(cnt[:, i].sum()) for i, n in enumerate(A.COUNTERS)}
    JL.metric("steady_state_seconds_per_step", value=round(t_steady / max(n_steady, 1), 5), cfg=cfg.name, alg=alg.name, lanes=L,
              compile_seconds=round(t_compile, 2))
    # invalidating counters must read 0 (plan_nan_steps is the replicated k=0 NaN of CFG-O, r_gt_1 the D21 existence > 1; both informational)
    inval = {n: v for n, v in bad.items() if n not in ("plan_nan_steps", "r_gt_1") and v}
    if inval:
        JL.warn("overflow / health counters non-zero -> affected runs invalid", **inval)
    return dict(outs=outs, counters=cnt, ms=ms_np, lams=lams_np, compile_seconds=t_compile,
                sec_per_step=t_steady / max(n_steady, 1), bad=bad)


def write_cell_metrics(cfg: M.Cfg, alg: M.Alg, res: dict, runs: int, clutter, out_dir: str, tag: str, m0: int = 0):
    outs, cnt = res["outs"], res["counters"]
    Lc = len(clutter)
    for m in range(runs):
        cell = {}
        for c, lam in enumerate(clutter):
            lane = c * runs + m
            gs = np.sqrt(outs["gospa_sq"][:, lane])
            cell[str(lam)] = dict(
                gospa_sq=outs["gospa_sq"][:, lane].tolist(), loc=outs["loc"][:, lane].tolist(), missed=outs["missed"][:, lane].tolist(),
                false=outs["false"][:, lane].tolist(), gospa_pos_sq=outs["gospa_pos_sq"][:, lane].tolist(),
                n_est=outs["n_est"][:, lane].tolist(), n_bern=outs["n_bern"][:, lane].tolist(),
                spos=outs["spos"][:, lane].tolist(), joint=outs["joint"][:, lane].astype(int).tolist(),
                mean_gospa=float(gs.mean()), mean_gospa_sq=float(outs["gospa_sq"][:, lane].mean()),
                counters={n: int(cnt[lane, i]) for i, n in enumerate(A.COUNTERS)})
        cell["_meta"] = dict(cfg_hash=cfg.hash(), alg=alg.name, compile_seconds=res["compile_seconds"], sec_per_step=res["sec_per_step"],
                             steps=int(outs["gospa_sq"].shape[0]), official_commit=M.OFFICIAL_COMMIT, truth_sha256=M.TRUTH_SHA256)
        _write_shard(shard_path(out_dir, tag, m0 + m, cfg.name, alg.name), cell)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="CFG-O", choices=list(M.CONFIGS))
    ap.add_argument("--algs", nargs="+", default=M.ALG_ORDER)
    ap.add_argument("--runs", type=int, default=M.N_RUNS_PAPER)
    ap.add_argument("--clutter", nargs="+", type=float, default=list(M.CLUTTER_RATES))
    ap.add_argument("--steps", type=int, default=M.N_STEPS)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value (sensitivity rows), e.g. R_var=2.0")
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--run-offset", type=int, default=0, help="first absolute run index m0 (runs m0..m0+runs-1); extension runs (P10)")
    ap.add_argument("--truths", default=None, help="S-TRUTH row: npz of independent truths (data/truths_strut.npz); run m uses truth m mod K")
    a = ap.parse_args(argv)
    JL.check_device()
    cfg = M.CONFIGS[a.cfg]
    if a.set:
        kv = {}
        for s in a.set:
            k, v = s.split("=")
            cur = getattr(cfg, k)
            kv[k] = type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true", "yes")
        cfg = cfg.replace(name=a.cfg + "+" + ",".join(a.set), **kv)
    truth_set = None
    if a.truths:
        from src import data as D
        truth_set = D.truth_set_arrays(a.truths)
        JL.info("per-run truth set", file=a.truths, n_truths=int(truth_set[0].shape[0]))
    out_dir = os.path.abspath(a.out)
    os.makedirs(os.path.join(out_dir, a.tag), exist_ok=True)
    with open(os.path.join(out_dir, a.tag, f"config_{cfg.name}.json"), "w") as f:
        json.dump(dict(cfg=cfg.as_dict(), hash=cfg.hash(), algs=a.algs, runs=a.runs, clutter=a.clutter, seed=a.seed,
                       official_commit=M.OFFICIAL_COMMIT, truth_sha256=M.TRUTH_SHA256, devices=[str(d) for d in jax.devices()]), f, indent=1)
    JL.info("start", cfg=cfg.name, hash=cfg.hash(), algs=len(a.algs), runs=a.runs, clutter=a.clutter, tag=a.tag)
    for name in a.algs:
        alg = M.ALGS[name]
        t0 = time.time()
        res = run_cell(cfg, alg, a.runs, a.clutter, out_dir + "/" + a.tag, a.chunk, a.steps, a.seed, not a.no_ckpt, truth_set=truth_set, m0=a.run_offset)
        write_cell_metrics(cfg, alg, res, a.runs, a.clutter, out_dir, a.tag, a.run_offset)
        gs = np.sqrt(res["outs"]["gospa_sq"])            # [steps, lanes]
        L = gs.shape[1]
        R = a.runs
        for c, lam in enumerate(a.clutter):
            g = np.sqrt((res["outs"]["gospa_sq"][:, c * R:(c + 1) * R]).mean(axis=1)).mean()   # Table III statistic
            JL.metric("table3_stat", cfg=cfg.name, alg=alg.name, clutter=lam, value=round(float(g), 3), runs=R)
        JL.info("cell_done", cfg=cfg.name, alg=alg.name, wall_seconds=round(time.time() - t0, 1))
    JL.info("done", cfg=cfg.name)


if __name__ == "__main__":
    main()

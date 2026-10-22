"""E5 (Table IV) wall-clock and FLOPs per method (pre-declared P7).

Wall-clock: ONE lane (one MC run, lambda_C = 2), steady-state seconds per planning+filter+scoring step after compile, mean and median over the chunks of
25 steps of a full 200-step run, one CPU core; first-compile time is logged separately.  FLOPs: XLA `cost_analysis` of (i) one tree-node expansion
(expand_belief, S = 1 and S = 2, depth >= 1) and (ii) one sensor's filter update; loop bodies are counted ONCE by XLA and arrays are padded to their
fixed capacities, so per-step FLOPs are the ANALYTIC product  expansions_per_step x flops_per_expansion + 2 x flops_per_filter_update, with
expansions_per_step = (B_joint x T) if the sensors are planned jointly else 2 x (B_ind x T) (upper bound: every iteration expands one node and rolls out to the horizon).
Usage: python scripts/e5_timing.py --cfg CFG-O --json runs/report/e5.json
"""
import argparse, json, os, sys, time
from rl_jax_bench import gpu_bootstrap
gpu_bootstrap.setup()
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from src import model as M
from src import algorithm as A
from src import planner as PL
from src import filter as FL


def flops_of(compiled):
    ca = compiled.cost_analysis()
    if isinstance(ca, (list, tuple)):
        ca = ca[0]
    return float(ca.get("flops", float("nan"))) if ca else float("nan")


def flops_expand(cfg, S, T, kind):
    pc = PL.PC(cfg)
    NP = cfg.r_cap + T
    nx = cfg.plan_nx
    b = PL.Belief(jnp.zeros(NP).at[:5].set(0.5), jnp.zeros((NP, nx)), jnp.tile(jnp.eye(nx), (NP, 1, 1)), jnp.asarray(5, jnp.int32))
    pos = jnp.zeros((S, 2))
    f = jax.jit(lambda b, pos: PL.expand_belief(b, pos, jnp.asarray(1), pc, cfg, kind))
    return flops_of(f.lower(b, pos).compile())


def flops_filter(cfg):
    fs = FL.empty_state(cfg)
    z = jnp.zeros((M.M_MEAS, 2))
    zv = jnp.zeros((M.M_MEAS,), bool)
    f = jax.jit(lambda fs, z, zv: FL.update_sensor(fs, z, zv, jnp.zeros(2), 2.0 / (jnp.pi * 1600.0), cfg)[0])
    return flops_of(f.lower(fs, z, zv).compile())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="CFG-O")
    ap.add_argument("--json", default=None)
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args()
    cfg = M.CONFIGS[a.cfg]
    out = {"load_at_start": open("/proc/loadavg").read().split()[:3], "cfg": a.cfg, "devices": [str(d) for d in jax.devices()], "algs": {}}
    ff = flops_filter(cfg)
    out["flops_filter_update_per_sensor"] = ff
    for name in M.ALG_ORDER:
        alg = M.ALGS[name]
        runner = A.make_chunk_runner(cfg, alg, 0, "off")
        carry = A.init_carry(cfg, 1)
        ms, lams = jnp.asarray([0]), jnp.asarray([2.0])
        ks = jnp.arange(0, 25)
        t0 = time.time()
        comp = runner.lower(carry, ks, ms, lams).compile()
        t_compile = time.time() - t0
        times, k0 = [], 0
        while k0 < a.steps:
            ks = jnp.arange(k0, min(k0 + 25, a.steps))
            t0 = time.time()
            carry, o = comp(carry, ks, ms, lams)
            jax.block_until_ready(o)
            times.append((time.time() - t0) / len(ks))
            k0 += len(ks)
        row = dict(compile_seconds=t_compile, sec_per_step_mean=float(np.mean(times)), sec_per_step_median=float(np.median(times)), chunk_times=[round(t, 5) for t in times])
        if alg.planner:
            T = alg.T
            fj = flops_expand(cfg, 2, T, alg.cost)
            fi = flops_expand(cfg, 1, T, alg.cost)
            exp_joint = alg.b_joint * T
            exp_ind = 2 * alg.b_ind * T
            row.update(flops_expand_S2=fj, flops_expand_S1=fi, expansions_joint_ub=exp_joint, expansions_indiv_ub=exp_ind,
                       flops_per_step_joint_ub=exp_joint * fj + 2 * ff, flops_per_step_indiv_ub=exp_ind * fi + 2 * ff)
        else:
            row.update(flops_per_step_joint_ub=2 * ff, flops_per_step_indiv_ub=2 * ff)
        out["algs"][name] = row
        print("[METRIC] E5 %-11s compile=%.1fs  sec/step mean=%.4f median=%.4f  flops/step(joint ub)=%.3g" % (name, t_compile, row["sec_per_step_mean"], row["sec_per_step_median"], row["flops_per_step_joint_ub"]), flush=True)
    out["load_at_end"] = open("/proc/loadavg").read().split()[:3]
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

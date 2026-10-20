"""Full-run final evaluation on k_test (protocol fixed before the run).

k_test = fold_in(root, 0x7E57). Replica draw i: sk = fold_in(k_test, i), i = 0..99; pairing index s (= learner seed s)
owns draws 20s..20s+19. Spawn episode j: sk = fold_in(k_test, 100 + j), j = 0..49, table spawn_scenario(fold_in(sk,
0x5A), n0 = 1). Baselines run on all 100 draws (their pairing-s value = mean over the same 20 draws); each learner lane
(variant, seed s) runs on its own 20 draws, all lanes of a run vmapped together. Final checkpoints only.

Parts (--part):
  baselines : EA, DA, OA, OA-myopic (+ OA-blind in Case IV) per case and calibration -> eval/base_<case>_<calib>.json
  learners  : every lane of every run dir on the run's cases -> eval/learn_<run>_<case>.json
  traces    : one draw per pairing (draw 20s) with per-slot dwells / ranges / weights (C8) -> eval/traces_*.npz
  sweep     : Fig. 9 baseline sweep (scan share 0.1..0.9) on the 50 spawn episodes and replica draws 0..19
Usage: python scripts/full_eval.py --part baselines|learners|traces|sweep [--runs runs/full/ddpg_case1,...]
The re-evaluation: --replica v2|v1 (re-digitised replica) and --key_scheme v2|legacy. The full-run
numbers are --replica v1 --key_scheme legacy; the re-evaluation (runs/full_v2/eval) uses the defaults.
Extra cases: case4n / case4p = Case IV with interference also on the scan measurement noise / scan SNR (A-38b).
Case III baselines add oa_blind (OA without the L_bm knowledge, = the full-run 'oa' there).
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402
from src.common import EnvCfg, info, load_calibration, load_yaml, metric, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import algorithm as alg, data, env as E, eval as ev  # noqa: E402

T = 10000
NP, NR, NS = 5, 20, 50                 # pairings, replica draws per pairing, spawn episodes
CALIBS = {"A04": "s_ref", "A04a": "s_ref_table6"}
RUN_CASES = {"case1": ["case1", "case4", "case3f", "case3s", "case4n", "case4p"], "case2": ["case2"],
             "case3": ["case3f", "case3s"]}


def case_cfg(y, case):
    if case in ("case1",):
        return EnvCfg.from_yaml(y)
    if case == "case2":
        return EnvCfg.from_yaml(y, weights="asset")
    if case in ("case3f", "case3s"):
        return EnvCfg.from_yaml(y, lbm=True)
    if case == "case4":
        return EnvCfg.from_yaml(y, interf=True)
    if case == "case4n":
        return EnvCfg.from_yaml(y, interf=True, interf_scan="noise")
    if case == "case4p":
        return EnvCfg.from_yaml(y, interf=True, interf_scan="snr")
    raise ValueError(case)


def case_scen(cfg, case):
    if case in ("case1", "case2"):
        return data.replica_case1(cfg)
    if case in ("case4", "case4n", "case4p"):
        return data.replica_case4(cfg)
    if case == "case3f":
        return data.case3(cfg, 0.5)          # fast Target 4 (primary, Fig. 15)
    if case == "case3s":
        return data.case3(cfg, 0.005)        # slow Target 4 (Fig. 16)
    raise ValueError(case)


def keys(root):
    kt = jax.random.fold_in(root, 0x7E57)
    rk = jnp.stack([jax.random.fold_in(kt, i) for i in range(NP * NR)])
    sk = jnp.stack([jax.random.fold_in(kt, 100 + j) for j in range(NS)])
    return rk, sk


def bcast(sc, n):
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), sc)


def tolist(d):
    return {k: np.asarray(v).tolist() for k, v in d.items() if k != "trace"}


def base_consts(cal, ck, delta=0.9):
    return dict(snr0=jnp.float64(cal["snr0"]), s_ref=jnp.float64(cal[CALIBS[ck]]), delta=jnp.float64(delta))


def part_baselines(a, y, cal, root, outd):
    rk, sk = keys(root)
    for case in a.cases.split(","):
        cfg = case_cfg(y, case)
        lut = ev.make_lut(cfg)
        rscen = bcast(case_scen(cfg, case), NP * NR)
        sscen = jax.vmap(lambda k: data.spawn_scenario(jax.random.fold_in(k, 0x5A), cfg, T, 1))(sk) \
            if case in ("case1", "case2") else None
        # OA-myopic (labelled addition, P6) only where it is graded: Cases I and II
        kinds = ["ea", "da", "oa"] + (["oa_myopic"] if case in ("case1", "case2") else []) + \
            (["oa_blind"] if case.startswith("case4") or case.startswith("case3") else [])
        for ck in a.calibs.split(","):
            res = {}
            for kind in kinds:
                t0 = time.perf_counter()
                f = ev.make_evaluator(cfg, lut, ev.make_policy(kind, cfg, y), T)
                c = base_consts(cal, ck)
                r = jax.device_get(f(None, c, rscen, rk))
                res[kind] = dict(replica=tolist(r))
                if sscen is not None:
                    res[kind]["spawn"] = tolist(jax.device_get(f(None, c, sscen, sk)))
                info("baseline", case=case, calib=ck, kind=kind, wall=round(time.perf_counter() - t0, 1),
                     U=float(np.mean(r["U"])))
            write_json(os.path.join(outd, f"base_{case}_{ck}.json"), res)


def load_run(rd):
    with open(os.path.join(rd, "params.pkl"), "rb") as fh:
        pk = pickle.load(fh)
    cu = json.load(open(os.path.join(rd, "config_used.json")))
    return pk, cu


PROJECT = False


def lane_policy(pk, yy, cfg):
    if pk["algo"] == "ddpg":
        actor, _, _, _ = alg.make_ddpg(yy, cfg.K, cfg.T0)
        pol = ev.make_policy("ddpg", cfg, yy, actor=actor)
    else:
        pol = ev.make_policy("dql", cfg, yy, dql_cfg=alg.make_dql_cfg(yy, cfg.K, 64))
    if not PROJECT:
        return pol

    def proj(p, c, v, es):
        """Verification only (P6, learners vs OA at EQUAL budget): scale the learner's dwells on confirmed tracks so
        that sum tau <= Theta_max T0 (the budget OA is held to); not part of the paper's method."""
        tau = jnp.where(v.act, jnp.clip(pol(p, c, v, es), 0.0, cfg.T0), 0.0)
        cap = cfg.theta_max * cfg.T0
        return tau * jnp.minimum(1.0, cap / jnp.maximum(tau.sum(), 1e-300))
    return proj


def lane_consts(pk):
    return {k: jnp.asarray(v) for k, v in pk["consts"].items()}


def part_learners(a, y, cal, root, outd):
    rk, sk = keys(root)
    for rd in [x for x in a.runs.split(",") if x]:
        pk, cu = load_run(rd)
        yy = cu["yaml"]
        name = a.prefix + os.path.basename(rd.rstrip("/"))
        seeds = np.asarray([s for _, s in pk["lanes"]])
        L = len(seeds)
        idx = jnp.asarray(np.stack([np.arange(20 * s, 20 * s + 20) for s in seeds]))   # [L, 20] pairing draws
        lk = rk[idx]
        cst = lane_consts(pk)
        for case in RUN_CASES[pk["case"]]:
            if a.cases != "all" and case not in a.cases.split(","):
                continue
            t0 = time.perf_counter()
            cfg = case_cfg(yy, case)
            lut = ev.make_lut(cfg)
            pol = lane_policy(pk, yy, cfg)
            one = ev.make_evaluator(cfg, lut, pol, T)          # jit(vmap over draws)
            sc = case_scen(cfg, case)
            rscen = jax.tree.map(lambda x: jnp.broadcast_to(x, (L, NR) + x.shape), sc)
            f = jax.jit(jax.vmap(lambda p, c, s_, k_: one(p, c, s_, k_), in_axes=(0, 0, 0, 0)))
            r = jax.device_get(f(pk["params"], cst, rscen, lk))
            res = dict(lanes=pk["lanes"], variants=pk["variants"], replica=tolist(r))
            if case in ("case1", "case2"):
                sscen = jax.vmap(lambda k: data.spawn_scenario(jax.random.fold_in(k, 0x5A), cfg, T, 1))(sk)
                g = jax.jit(jax.vmap(lambda p, c: one(p, c, sscen, sk), in_axes=(0, 0)))
                res["spawn"] = tolist(jax.device_get(g(pk["params"], cst)))
            write_json(os.path.join(outd, f"learn_{name}_{case}{'_proj' if PROJECT else ''}.json"), res)
            info("learners", run=name, case=case, lanes=L, wall=round(time.perf_counter() - t0, 1),
                 U_mean=float(np.mean(r["U"])))


def trace_rollout(cfg, lut, pol, params, consts, scen, key):
    es0 = E.reset(cfg, consts.get("lam0"), lit="obs_lit" in consts)

    def step(es, _):
        v = E.pre(cfg, scen, es)
        tau = pol(params, consts, v, es)
        es2, out = E.post(cfg, consts, lut, scen, es, v, tau, ev.slot_key(key, es.t))
        rt = jnp.hypot(*v.pos[es.tgt].T)
        return es2, dict(tau=out["tau"], act=es.act, r_hat=v.r_hat, r_true=rt, w=v.w, fac_true=E.interf_factor(
            cfg, v.pos[es.tgt]), tgt=es.tgt, U=out["U"], lam=out["lam"], c=out["dbg_c"])
    _, outs = jax.lax.scan(step, es0, None, length=T)
    return outs


def part_traces(a, y, cal, root, outd):
    """C8 qualitative checks: draw 20s of pairing s for every lane of the main variants and every baseline."""
    rk, _ = keys(root)
    pair_keys = rk[jnp.arange(NP) * NR]
    for rd in [x for x in a.runs.split(",") if x]:
        pk, cu = load_run(rd)
        yy = cu["yaml"]
        name = a.prefix + os.path.basename(rd.rstrip("/"))
        seeds = np.asarray([s for _, s in pk["lanes"]])
        cst = lane_consts(pk)
        for case in RUN_CASES[pk["case"]]:
            cfg = case_cfg(yy, case)
            lut = ev.make_lut(cfg)
            pol = lane_policy(pk, yy, cfg)
            sc = case_scen(cfg, case)
            f = jax.jit(jax.vmap(lambda p, c, k: trace_rollout(cfg, lut, pol, p, c, sc, k)))
            o = jax.device_get(f(pk["params"], cst, pair_keys[jnp.asarray(seeds)]))
            np.savez_compressed(os.path.join(outd, f"traces_{name}_{case}.npz"), lanes=np.asarray(
                [f"{v}|{s}" for v, s in pk["lanes"]]), **{k: np.asarray(v, np.float32) for k, v in o.items()})
            info("traces", run=name, case=case)
    # baselines (A04 and A04a; A04 tracking dwells do not depend on S_ref except through initiation)
    for case in (a.cases.split(",") if a.base_traces else []):
        if all(os.path.exists(os.path.join(outd, f"traces_base_{k}_{case}_{ck}.npz")) for k in ("ea", "da", "oa")
               for ck in a.calibs.split(",")):
            continue
        cfg = case_cfg(y, case)
        lut = ev.make_lut(cfg)
        sc = case_scen(cfg, case)
        for kind in ["ea", "da", "oa"] + (["oa_blind"] if case.startswith("case4") or case.startswith("case3") else []):
            pol = ev.make_policy(kind, cfg, y)
            for ck in a.calibs.split(","):
                c = base_consts(cal, ck)          # passed as an ARGUMENT (closed-over consts made XLA constant-fold OA)
                f = jax.jit(jax.vmap(lambda c_, k: trace_rollout(cfg, lut, pol, None, c_, sc, k), in_axes=(None, 0)))
                o = jax.device_get(f(c, pair_keys))
                np.savez_compressed(os.path.join(outd, f"traces_base_{kind}_{case}_{ck}.npz"),
                                    **{k: np.asarray(v, np.float32) for k, v in o.items()})
        info("traces baselines", case=case)


def part_sweep(a, y, cal, root, outd):
    """Fig. 9: EA/DA/OA at scan share 0.1..0.9 (tracking share delta = 1 - scan share), spawn x 50 and replica x 20."""
    rk, sk = keys(root)
    cfg = EnvCfg.from_yaml(y)
    lut = ev.make_lut(cfg)
    rscen = bcast(data.replica_case1(cfg), NR)
    sscen = jax.vmap(lambda k: data.spawn_scenario(jax.random.fold_in(k, 0x5A), cfg, T, 1))(sk)
    shares = np.round(np.arange(1, 10) * 0.1, 1)
    res = dict(scan_share=shares.tolist())
    for ck in a.calibs.split(","):
        for kind in ["ea", "da", "oa"]:
            one = ev.make_evaluator(cfg, lut, ev.make_policy(kind, cfg, y), T)
            rows = []
            t0 = time.perf_counter()
            c0 = base_consts(cal, ck)
            deltas = jnp.asarray(1.0 - shares)
            fsw = jax.jit(jax.vmap(lambda d, sc_, k_: one(None, dict(c0, delta=d), sc_, k_), in_axes=(0, None, None)))
            R_all = jax.device_get(fsw(deltas, rscen, rk[:NR]))          # one vmapped call over the 9 shares
            S_all = jax.device_get(fsw(deltas, sscen, sk))
            for j, sh in enumerate(shares):
                r = {k: v[j] for k, v in R_all.items()}
                s = {k: v[j] for k, v in S_all.items()}
                rows.append(dict(replica={k: float(np.mean(v)) for k, v in r.items()},
                                 spawn={k: float(np.mean(v)) for k, v in s.items()},
                                 spawn_tinit=float(np.sum(s["tinit_sum"]) / max(np.sum(s["tinit_n"]), 1)),
                                 replica_tinit=float(np.sum(r["tinit_sum"]) / max(np.sum(r["tinit_n"]), 1)),
                                 spawn_U_per_ep=np.asarray(s["U"]).tolist()))
            res[f"{kind}_{ck}"] = rows
            info("sweep", kind=kind, calib=ck, wall=round(time.perf_counter() - t0, 1))
    write_json(os.path.join(outd, "sweep_fig9.json"), res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["baselines", "learners", "traces", "sweep"])
    ap.add_argument("--runs", default="")
    ap.add_argument("--cases", default="case1,case2,case3f,case3s,case4")
    ap.add_argument("--calibs", default="A04,A04a")
    ap.add_argument("--base_traces", action="store_true", help="traces part: also trace the baselines")
    ap.add_argument("--project", action="store_true", help="learners part: verification eval with dwells projected "
                    "onto the Theta_max budget (writes *_proj.json)")
    ap.add_argument("--root_seed", type=int, default=20261019)
    ap.add_argument("--out", default="runs/full_v2/eval")
    ap.add_argument("--prefix", default="", help="output-name prefix for run dirs (e.g. recheck_ for runs/recheck/*)")
    ap.add_argument("--replica", default="v2", choices=["v2", "v1", "v2o"])
    ap.add_argument("--key_scheme", default="v2", choices=["v2", "legacy"])
    a = ap.parse_args()
    data.REPLICA = a.replica
    ev.SLOT_KEY = a.key_scheme
    common.check_device()
    y = load_yaml()
    cal = load_calibration()
    root = jax.random.PRNGKey(a.root_seed)
    os.makedirs(a.out, exist_ok=True)
    global PROJECT
    PROJECT = a.project
    info("k_test evaluation", part=a.part, note="FINAL evaluation; k_test = fold_in(root, 0x7E57)", replica=a.replica,
         key_scheme=a.key_scheme)
    dict(baselines=part_baselines, learners=part_learners, traces=part_traces, sweep=part_sweep)[a.part](
        a, y, cal, root, a.out)
    metric("done", part=a.part)


if __name__ == "__main__":
    main()

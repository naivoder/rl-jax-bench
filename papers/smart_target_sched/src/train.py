"""Experiment driver (there is nothing to TRAIN in this paper: it is an index-policy simulation study).

    JAX_PLATFORMS=cpu python src/train.py --tables t1 t2 t3 t4 --K 1 2 3 --nmc 100 --tag small
    JAX_PLATFORMS=cpu python src/train.py --fig fig6 --Ks 2 6 --nmc 20 --tag small

One call = a set of cells x a set of policies; every policy of a cell receives IDENTICAL P0 and tie keys (paired design, data.py key
tree).  Replications are vmapped inside one process.  Output: <runs>/<tag>/seed_<r>/metrics.json (r = replication = the seed unit; read-
modify-write so several invocations accumulate), <runs>/<tag>/run_log_*.json (timings; compile vs steady state).  `--resume` skips the
cells that are already complete (cell = the checkpoint unit; every cell is minutes at most, so orbax checkpointing is not needed).
"""
from __future__ import annotations

import os
import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()

import argparse  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rl_jax_bench import jaxlog  # noqa: E402
import model as M  # noqa: E402
import algorithm as A  # noqa: E402
import data as D  # noqa: E402

DEFAULT_POLICIES = ["whittle", "myopic", "tev", "passive_cont", "passive_cont_h10", "passive_cont_revisit", "random", "rollout1"]
RUNS_ROOT = os.path.join(HERE, "..", "runs")


def sources_sha() -> str:
    return hashlib.sha256(open(os.path.join(HERE, "configs", "sources.yaml"), "rb").read()).hexdigest()


def git_commit() -> str:
    """HEAD commit, suffixed '+dirty' when tracked source files differ from it (so a run's provenance is never overstated)."""
    try:
        h = subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", HERE, "status", "--porcelain", "--untracked-files=no", "--", "."], text=True,
                                        stderr=subprocess.DEVNULL).strip()
        return h + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def cell_id(scn: D.Scenario, K: int, variant: str) -> str:
    return f"{scn.name}|K={K}|{variant}"


def variant_name(cfg: A.Cfg, ft: str) -> str:
    v = []
    if cfg.tau != M.TAU:
        v.append(f"tau{cfg.tau}")
    if cfg.r1_base != A.ROLLOUT1_BASE_SLOTS:
        v.append(f"r1b{cfg.r1_base}")
    if cfg.skip_neg:
        v.append("skipneg")
    if cfg.sampled:
        v.append("sampled")
    if cfg.pcli1 != "dominant":
        v.append("pcli1_" + cfg.pcli1)
    if ft != "printed":
        v.append("ft_" + ft)
    return "+".join(v) or "primary"


def read_seed(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def write_cell(tag_dir, cid, res, meta):
    """res: {'J': {pol: (R,)}, 'pcli1': {...}, 'pmax': {...}, ...} -> per-replication metrics.json."""
    R = len(next(iter(res["J"].values())))
    for r in range(R):
        p = os.path.join(tag_dir, f"seed_{r}", "metrics.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + ".lock", "w") as lk:                       # several invocations may append to the same seed file
            fcntl.flock(lk, fcntl.LOCK_EX)
            cur = read_seed(p)
            cur.setdefault("meta", meta)
            cell = cur.setdefault("cells", {}).setdefault(cid, {})     # merge: later invocations may add policies to a cell
            for k, d in res.items():
                if isinstance(d, dict):
                    cell.setdefault(k, {}).update({pol: float(v[r]) for pol, v in d.items()})
            jaxlog.write_json(p, cur)


def cell_done(tag_dir, cid, policies, R):
    last = read_seed(os.path.join(tag_dir, f"seed_{R - 1}", "metrics.json")).get("cells", {}).get(cid)
    return bool(last) and all(p in last.get("J", {}) for p in policies)


_seen: dict = {}


def run_cell(scn: D.Scenario, K: int, policies, nmc: int, seed: int, cfg: A.Cfg, ft: str, exp: str, timings: list):
    bank = D.build_bank(scn, ft)
    rk = D.rep_keys(seed, exp, scn.sid, K, nmc)
    p0k, tie, mode = D.split_rep_keys(rk)
    P0 = D.sample_P0(scn, p0k)
    if scn.kind == "matrix":
        me = D.min_eig(P0)
        if me <= 1e-9:
            jaxlog.warn("P0 not positive definite enough (A-P0M would redraw)", min_eig=me)
    res = {"J": {}, "n_pcli1_viol": {}, "pmax": {}}
    for pol in policies:
        if pol == "rollout1" and scn.N > 10:
            continue
        sig = (pol, scn.N, scn.kind, K, nmc, cfg)
        t0 = time.perf_counter()
        out = A.simulate(bank, P0, tie, mode, pol, K, cfg)
        out["J"].block_until_ready()
        t_first = time.perf_counter() - t0
        t_steady = None
        if sig not in _seen:           # first call of this signature includes jit compile: time a second call separately
            t0 = time.perf_counter()
            out2 = A.simulate(bank, P0, tie, mode, pol, K, cfg)
            out2["J"].block_until_ready()
            t_steady = time.perf_counter() - t0
            _seen[sig] = True
        timings.append({"cell": cell_id(scn, K, variant_name(cfg, ft)), "policy": pol, "first_call_s": t_first, "steady_s": t_steady})
        jaxlog.metric("cell", cell=cell_id(scn, K, variant_name(cfg, ft)), policy=pol, J_mean=float(out["J"].mean()),
                      J_se=float(out["J"].std(ddof=1) / np.sqrt(nmc)) if nmc > 1 else 0.0, first_call_s=round(t_first, 2),
                      steady_s=None if t_steady is None else round(t_steady, 2))
        res["J"][pol] = np.asarray(out["J"]); res["n_pcli1_viol"][pol] = np.asarray(out["n_pcli1_viol"])
        res["pmax"][pol] = np.asarray(out["pmax"])
    return res, P0


def run_lower_bound(scn: D.Scenario, K: int, P0, tag_dir, cid, meta, timings):
    import bound as B
    t0 = time.perf_counter()
    types, qct, d = list(scn.types), list(scn.qct), list(scn.d)
    P = np.asarray(P0)[:, :, 0, 0]
    # constant P0 (Fig 5): all replications share one bound -> compute once
    const = bool(np.all(P == P[0]))
    a = B.vi_dual(types, qct, d, P[:1] if const else P, K)
    VD = np.asarray(a["V_D"]); lam = np.asarray(a["lam"]); pv = np.asarray(a["pvis"])
    if const:
        VD, lam, pv = np.repeat(VD, P.shape[0]), np.repeat(lam, P.shape[0]), np.repeat(pv, P.shape[0])
    dt = time.perf_counter() - t0
    jaxlog.metric("lower_bound", cell=cid, V_D_mean=float(VD.mean()), lam_mean=float(lam.mean()), pvis_max=float(pv.max()),
                  P_cap=B.P_CAP, n_problems=a["n_problems"], seconds=round(dt, 1))
    if pv.max() > B.P_CAP / 100:
        jaxlog.warn("lower-bound grid cap check FAILED: visited state within 100x of P_cap", pvis=float(pv.max()))
    timings.append({"cell": cid, "policy": "V_D(VI)", "first_call_s": dt, "steady_s": None})
    return {"V_D": VD, "lam_star": lam, "pvis": pv}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs="*", default=[], choices=["t1", "t2", "t3", "t4"])
    ap.add_argument("--qsets", nargs="*", default=None, help="restrict table scenarios by suffix (e.g. q2 q29)")
    ap.add_argument("--fig", default=None, choices=["fig5a", "fig5b", "fig5c", "fig6", "fig7", "fig8"])
    ap.add_argument("--K", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--Ks", nargs="*", type=int, default=[])
    ap.add_argument("--nmc", type=int, default=M.N_MC)
    ap.add_argument("--policies", nargs="*", default=DEFAULT_POLICIES)
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=int, default=M.TAU)
    ap.add_argument("--T", type=int, default=M.T_HORIZON)
    ap.add_argument("--r1-base", type=int, default=A.ROLLOUT1_BASE_SLOTS, help="rollout1 base-policy length (tuning-parity sweep)")
    ap.add_argument("--skip-neg", action="store_true")
    ap.add_argument("--sampled", action="store_true", help="extension X3: sampled modes")
    ap.add_argument("--pcli1", default="dominant", choices=["dominant", "literal"], help="rule when g <= 0")
    ap.add_argument("--ft", default="printed", choices=["printed", "standard"])
    ap.add_argument("--lb", action="store_true", help="also compute the Lagrangian lower bound (scalar figs)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--runs-root", default=RUNS_ROOT)
    a = ap.parse_args(argv)

    jaxlog.info("start", jax=jax.__version__, x64=bool(jax.config.jax_enable_x64), argv=" ".join(sys.argv[1:]))
    jaxlog.check_device()
    assert jax.config.jax_enable_x64
    jaxlog.info("environment checksum", max_dev=M.assert_env_checksum(), tol=M.REF_TOL)
    cfg = A.Cfg(T=a.T, tau=a.tau, skip_neg=a.skip_neg, sampled=a.sampled, pcli1=a.pcli1, r1_base=a.r1_base)
    exp0 = "X3" if a.sampled else None
    tag_dir = os.path.join(a.runs_root, a.tag)
    os.makedirs(tag_dir, exist_ok=True)
    meta = {"commit": git_commit(), "sources_sha256": sources_sha(), "jax": jax.__version__, "devices": str(jax.devices()),
            "tag": a.tag, "seed": a.seed, "top_seed": M.TOP_SEED, "cfg": cfg._asdict(), "ft": a.ft, "nmc": a.nmc,
            "beta": M.BETA, "tau": a.tau, "T": a.T, "dtype": "float64"}
    jobs = []
    for t in a.tables:
        for name, scn in D.TABLE_SCENARIOS.items():
            if name.startswith(t + "_") and (a.qsets is None or name.split("_", 1)[1] in a.qsets):
                for K in a.K:
                    jobs.append((exp0 or D.TABLE_EXP[t], scn, K))
    if a.fig:
        for K in a.Ks:
            scn = D.figure_scenario(a.fig, K)
            jobs.append((exp0 or {"fig5a": "E6", "fig5b": "E6", "fig5c": "E6", "fig6": "E7", "fig7": "E9", "fig8": "E10"}[a.fig], scn, K))
    timings: list = []
    for exp, scn, K in jobs:
        cid = cell_id(scn, K, variant_name(cfg, a.ft))
        pols = [p for p in a.policies if not (p == "rollout1" and scn.N > 10)]
        if a.resume and cell_done(tag_dir, cid, pols, a.nmc) and (not a.lb or "V_D" in read_seed(os.path.join(tag_dir, f"seed_{a.nmc - 1}", "metrics.json")).get("cells", {}).get(cid, {})):
            jaxlog.info("skip (done)", cell=cid)
            continue
        t0 = time.perf_counter()
        res, P0 = run_cell(scn, K, pols, a.nmc, a.seed, cfg, a.ft, exp, timings)
        if a.lb:
            assert scn.kind == "scalar"
            lb = run_lower_bound(scn, K, P0, tag_dir, cid, meta, timings)
            for k, v in lb.items():
                res[k] = {"lb": v}
        write_cell(tag_dir, cid, res, meta)
        jaxlog.info("cell done", cell=cid, seconds=round(time.perf_counter() - t0, 1))
    jaxlog.write_json(os.path.join(tag_dir, f"run_log_{int(time.time())}.json"), {"meta": meta, "timings": timings})
    jaxlog.info("done", tag=a.tag, cells=len(jobs))


if __name__ == "__main__":
    main()

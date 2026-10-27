"""Experiment driver (there is nothing to train: the PMBM filter has no learnable parameters; this module is the
Monte-Carlo runner).  Resumability = one runs/<tag>/seed_<r>/metrics.json per MC run (finished runs are skipped).

  python -m src.train --tag E1 --pd 0.9 --lam 10 --runs 100 --workers 3 --algos pmbm,phd,cphd

Per MC run it writes metrics.json: per-step squared OSPA / GOSPA (+decomposition) for every algorithm/estimator,
hypothesis counts, overflow counters, timing (compile vs steady state is logged separately by each worker).
Key tree: see data.make_measurements_runs.  Logging: [INFO]/[METRIC]/[WARN]/[ERROR].
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))     # the paper folder, so that `src` imports when run as a file
os.environ.setdefault("JAX_PLATFORMS", "cpu")           # CPU by default
from rl_jax_bench import gpu_bootstrap  # noqa: E402
gpu_bootstrap.setup()

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from rl_jax_bench.jaxlog import info, metric, warn, error, write_json  # noqa: E402

from src import data as D  # noqa: E402
from src import eval as EV  # noqa: E402
from src.algorithm import PMBMConfig, run_filter, OVF_NAMES  # noqa: E402
from src.model import make_lg_model, paper_lg_params, paper_pmbm_config, PAPER_T, PAPER_H, PAPER_M  # noqa: E402


def build_pmbm(cfgs, pd, lam, T=PAPER_T, H=PAPER_H, Nh=200, M=PAPER_M, backend="host", **kw):
    """PMBM exactly as paper §VII (N_h=200, gate 20, prune 1e-5/1e-4, Gamma=0.4); sensitivity rows via cfgs
    (q, lambda0).  T/H/M are array capacities (overflow counters must read 0)."""
    p = paper_lg_params(pd=pd, lambda_c=lam, q=cfgs["q"], area=cfgs["area"])
    model = make_lg_model(p)
    pcfg = paper_pmbm_config(T=T, H=H, Nh=Nh, M=M, backend=backend, **kw)    # paper-path caps live in model.py
    lam0 = cfgs.get("lambda0")
    from src.algorithm import init_state

    if lam0 is None:
        state0, predict_first = None, True                      # paper: empty prior, predict then update
    else:                                                       # official-code row: Poisson weight lambda0 at k=1
        bw, bd = model.birth(None)
        state0 = init_state(model, pcfg, pois=(jnp.log(lam0) + 0 * bw, bd))
        predict_first = False
    f = jax.jit(lambda z, zv: run_filter(model, pcfg, z, zv, None, state0, predict_first))
    return f


def run_pmbm_one(f, z, zv, truth):
    t = time.time()
    st, outs = f(z, zv)
    jax.block_until_ready(outs)
    dt = time.time() - t
    res = {}
    for name, key in (("pmbm_est1", "est1"), ("pmbm_est2", "est2"), ("pmbm_est3", "est3")):
        m, x = outs[key]
        r = EV.evaluate_run(np.asarray(m), np.asarray(x), truth)
        res[name] = {k: v.tolist() for k, v in r.items()}
    diag = dict(ovf=np.asarray(outs["ovf"])[-1].tolist(), n_hyp_max=int(np.asarray(outs["n_hyp"]).max()),
                n_tracks_max=int(np.asarray(outs["n_tracks"]).max()), max_h=int(np.asarray(outs["max_h"]).max()),
                n_children_max=int(np.asarray(outs["n_children"]).max()), est_over=int(np.asarray(outs["est_over"]).sum()),
                seconds=dt)
    return res, diag


def zero_counter_violations(rec):
    """The zero-counter rule: every capacity/truncation counter of a run must read 0.
    ovf = OVF_NAMES (pois_evict, slot_full, cap_rows_dropped, kbest_ovf, meas_ignored, child_trunc); est_over = selected tracks
    beyond the E-slot estimate set (PMBM) / estimates beyond E (PHD, CPHD); n_meas_dropped = clutter cut at M."""
    bad = {}
    if rec.get("n_meas_dropped", 0):
        bad["n_meas_dropped"] = rec["n_meas_dropped"]
    for a, dg in rec["diag"].items():
        for i, v in enumerate(dg.get("ovf", [])):
            if v:
                bad[f"{a}.ovf.{OVF_NAMES[i]}"] = v
        if dg.get("est_over"):
            bad[f"{a}.est_over"] = dg["est_over"]
    return bad


def worker(args):
    (runs, tag, pd, lam, cfg_over, algos, pm_kw, config_id, outdir, truth_seed, filter_q, allow_overflow, truth_kind) = args
    cfg_over = dict(cfg_over)
    base_over = {k: v for k, v in cfg_over.items() if k not in ("phd_merge", "cphd_merge", "phd_max", "cphd_max")}
    cfgs = D.load_scenario_cfg(**base_over)                     # scenario/truth/measurement config
    fcfgs = cfgs if filter_q is None else D.load_scenario_cfg(**dict(base_over, q=filter_q))   # filter-only q (one-factor row)
    fcfgs = dict(fcfgs)
    for key_, f_ in (("phd", "merge"), ("cphd", "merge"), ("phd", "max_comp"), ("cphd", "max_comp")):   # sensitivity dials (labelled)
        v_ = cfg_over.get(f"{key_}_{'merge' if f_ == 'merge' else 'max'}")
        if v_ is not None:
            fcfgs[key_] = dict(fcfgs[key_], **{f_: v_})
    root = jax.random.PRNGKey(cfgs["seed_top"])                 # measurement root
    troot = root if truth_seed is None else jax.random.PRNGKey(truth_seed)   # truth root separable from the measurement root
    truth = D.load_fig3_truth() if truth_kind == "fig3" else D.make_truth(jax.random.fold_in(troot, 0), cfgs)   # graded truth = the paper's Fig. 3
    truth_np = {k: np.asarray(v) for k, v in truth.items()}
    # a truncated measurement set is an input edit (drops clutter only), so it is an error, never a warning
    z, zv, nd = D.make_measurements_runs(root, truth, config_id, pd, lam, max(runs) + 1, cfgs, pm_kw.get("M", PAPER_M),
                                         on_drop="ignore" if allow_overflow else "raise")
    fns = {}
    if "pmbm" in algos:
        fns["pmbm"] = build_pmbm(fcfgs, pd, lam, **pm_kw)
    if any(a in algos for a in ("phd", "phd_vm", "cphd")):
        from src import phd as PH
    for r in runs:
        d = os.path.join(outdir, f"seed_{r}")
        mp_ = os.path.join(d, "metrics.json")
        if os.path.exists(mp_):
            continue
        os.makedirs(d, exist_ok=True)
        rec = dict(run=r, config_id=config_id, pd=pd, lambda_c=lam, n_meas_dropped=int(nd[r]), algos={}, diag={})
        if "pmbm" in fns:
            res, diag = run_pmbm_one(fns["pmbm"], z[r], zv[r], truth_np)
            rec["algos"].update(res)
            rec["diag"]["pmbm"] = diag
        # phd = official default estimator GMPHD_estimation2; phd_vm = original Vo-Ma extraction (est1)
        for name, fn, over_ in (("phd", "run_gmphd", {}), ("phd_vm", "run_gmphd", {"phd_estimator": "est1"}),
                                ("cphd", "run_gmcphd", {})):
            if name in algos:
                t = time.time()
                m, x, info = getattr(PH, fn)(z[r], zv[r], dict(fcfgs, **over_), pd=pd, lambda_c=lam, return_info=True)
                rr = EV.evaluate_run(np.asarray(m), np.asarray(x), truth_np)
                rec["algos"][name] = {k: v.tolist() for k, v in rr.items()}
                rec["diag"][name] = dict(seconds=time.time() - t, est_over=int(np.maximum(np.asarray(info["n_hat"]) - 16, 0).sum()))
        bad = zero_counter_violations(rec)
        rec["valid"] = not bad
        rec["counter_violations"] = bad
        if bad and not allow_overflow:      # a run with a nonzero counter is not the paper's filter -> stop, do not publish it
            write_json(os.path.join(d, "metrics.INVALID.json"), rec)
            error("zero-counter rule violated; run NOT counted", tag=tag, run=r, violations=bad)
            raise RuntimeError(f"zero-counter rule violated in run {r}: {bad}")
        write_json(mp_, rec)
        e1 = np.sqrt(np.mean(rec["algos"].get("pmbm_est1", {"ospa2": [np.nan]})["ospa2"]))
        metric("run done", tag=tag, run=r, pmbm_est1_rms=float(e1), **{k: v for k, v in rec["diag"].get("pmbm", {}).items() if k in ("seconds", "n_hyp_max", "max_h")})
        if bad:
            warn("zero-counter violation (allowed by --allow-overflow; run marked valid=false)", tag=tag, run=r, violations=bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pd", type=float, default=0.9)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--config-id", type=int, default=None, help="Table II row (default: derived from pd, lam)")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--first-run", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--algos", default="pmbm")
    ap.add_argument("--q", type=float, default=None, help="sensitivity row: 0.02")
    ap.add_argument("--lambda0", type=float, default=None, help="sensitivity row: 3.0 (no predict at k=1)")
    ap.add_argument("--filter-q", type=float, default=None, help="q of the FILTER only (truth/measurements stay at the paper q)")
    ap.add_argument("--phd-merge", type=float, default=None, help="sensitivity: GM-PHD merge threshold (official 0.1)")
    ap.add_argument("--cphd-merge", type=float, default=None, help="sensitivity: GM-CPHD merge threshold (official 0.1)")
    ap.add_argument("--phd-max", type=int, default=None, help="sensitivity: GM-PHD component cap")
    ap.add_argument("--cphd-max", type=int, default=None, help="sensitivity: GM-CPHD component cap")
    ap.add_argument("--truth-seed", type=int, default=None, help="seed of the truth root (default: the measurement root seed_top)")
    ap.add_argument("--truth", choices=("synthetic", "fig3"), default="synthetic", help="'fig3' = the paper's own Fig. 3 trajectories (data/fig3_truth.npz); default = a draw from the Sec. VII law")
    ap.add_argument("--meas-seed", type=int, default=None, help="measurement-root seed (default sources.yaml top_seed 18)")
    ap.add_argument("--prune-w", type=float, default=None, help="sensitivity: global-hypothesis weight floor (paper path default 0 = Alg. 1; authors' HEAD code 1e-4)")
    ap.add_argument("--pm-extra", default=None, help="JSON of extra PMBMConfig fields (e.g. '{\"prune_r\": 1e-4}') for probes; labelled non-paper")
    ap.add_argument("--allow-overflow", action="store_true", help="diagnostic only: keep runs that violate the zero-counter rule (marked valid=false)")
    ap.add_argument("--T", type=int, default=PAPER_T)
    ap.add_argument("--H", type=int, default=PAPER_H)
    ap.add_argument("--Nh", type=int, default=200)
    ap.add_argument("--M", type=int, default=PAPER_M)
    ap.add_argument("--backend", default="host")
    ap.add_argument("--outroot", default=os.path.join(HERE, "..", "runs"))
    a = ap.parse_args()
    cid = a.config_id
    if cid is None:
        cid = D.DATA_TABLE2_PD.index(a.pd) * 3 + D.DATA_TABLE2_LAMBDA.index(int(a.lam)) if (a.pd in D.DATA_TABLE2_PD and int(a.lam) in D.DATA_TABLE2_LAMBDA) else 99
    over = {}
    if a.q is not None:
        over["q"] = a.q
    if a.lambda0 is not None:
        over["lambda0"] = a.lambda0
    for k_, v_ in (("phd_merge", a.phd_merge), ("cphd_merge", a.cphd_merge), ("phd_max", a.phd_max), ("cphd_max", a.cphd_max)):
        if v_ is not None:
            over[k_] = v_
    if a.meas_seed is not None:
        over["seed_top"] = a.meas_seed
    outdir = os.path.abspath(os.path.join(a.outroot, a.tag))
    os.makedirs(outdir, exist_ok=True)
    pm_kw = dict(T=a.T, H=a.H, Nh=a.Nh, M=a.M, backend=a.backend)
    if a.prune_w is not None:
        pm_kw["prune_w"] = a.prune_w
    if a.pm_extra:
        pm_kw.update(json.loads(a.pm_extra))
    algos = a.algos.split(",")
    runs = list(range(a.first_run, a.first_run + a.runs))
    chunks = [runs[i::a.workers] for i in range(a.workers)]
    info("launch", tag=a.tag, pd=a.pd, lam=a.lam, config_id=cid, runs=len(runs), workers=a.workers, algos=a.algos,
         devices=str(jax.devices()))
    write_json(os.path.join(outdir, "config.json"), dict(vars(a), config_id=cid, overrides=over, pm_kw=pm_kw))
    t0 = time.time()
    jobs = [(c, a.tag, a.pd, a.lam, over, algos, pm_kw, cid, outdir, a.truth_seed, a.filter_q, a.allow_overflow, a.truth) for c in chunks if c]
    if a.workers == 1:
        worker(jobs[0])
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            pool.map(worker, jobs)
    info("done", tag=a.tag, wall_s=time.time() - t0)


if __name__ == "__main__":
    main()

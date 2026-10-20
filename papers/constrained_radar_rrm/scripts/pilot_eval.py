"""pilot evaluation (validation keys ONLY; k_test is never touched here).

Case I, all arms: EA, DA, OA, OA-myopic on R replica draws + S spawn-model episodes (10000 slots, n0 = 1), and every
learner run directory given (params.pkl + config_used.json), each training seed evaluated on the same draws.
Keys: k_valid = fold_in(root, 0xA11D); replica draw i: fold_in(k_valid, 1000 + i); spawn episode i: fold_in(k_valid,
2000 + i) (scenario fold_in(., 0x5A)). Writes <out>/summary.json and prints a table.
Usage: python scripts/pilot_eval.py --runs runs/pilot_debug/ddpg_A,... --out runs/pilot [--R 8 --S 8]
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

from src import algorithm as alg, data, eval as ev  # noqa: E402
data.REPLICA, ev.SLOT_KEY = "v1", "legacy"   # Re-evaluation: reproduce the recorded (pre-re-digitisation, legacy-key) numbers

PAPER_I = dict(ddpg=-1.90e4, dql=-1.96e4, ea=-2.42e4, da=-1.94e4, oa=-1.82e4)
PAPER_TINIT = dict(ddpg=6.0, dql=7.83, ea=10.5, da=14.67, oa=20.33)


def summarise(s):
    ti, nv = ev.pooled_tinit(s)
    return dict(U=float(np.mean(s["U"])), c=float(np.mean(s["c"])), scan_term=float(np.mean(s["scan_term"])),
                tinit=float(ti), never=int(nv), sat_rate=float(np.mean(s["sat_rate"])),
                g_mean=float(np.mean(s["g_mean"])), track_share=float(np.mean(s["track_share"])),
                lam_multi=float(np.mean(s["lam_multi"])), starve=float(np.mean(s["starve"])),
                viol_near_frac=float(np.mean(s["viol_near_frac"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="")
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--out", default="runs/pilot")
    ap.add_argument("--root_seed", type=int, default=20261019)
    ap.add_argument("--baselines", default="ea,da,oa,oa_myopic")
    ap.add_argument("--s_ref_key", default="s_ref", help="calibration.json key: s_ref (A-04) or s_ref_table6 (A-04a)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    common.check_device()
    y = load_yaml()
    cal = load_calibration()
    cfg = EnvCfg.from_yaml(y)
    lut = ev.make_lut(cfg)
    consts = dict(snr0=jnp.float64(cal["snr0"]), s_ref=jnp.float64(cal[a.s_ref_key]), delta=jnp.float64(0.9))
    root = jax.random.PRNGKey(a.root_seed)
    kv = jax.random.fold_in(root, 0xA11D)
    rkeys = jnp.stack([jax.random.fold_in(kv, 1000 + i) for i in range(a.R)])
    skeys = jnp.stack([jax.random.fold_in(kv, 2000 + i) for i in range(a.S)])
    rep = data.replica_case1(cfg)
    rscen = jax.tree.map(lambda x: jnp.broadcast_to(x, (a.R,) + x.shape), rep)
    sscen = jax.vmap(lambda k: data.spawn_scenario(jax.random.fold_in(k, 0x5A), cfg, 10000, 1))(skeys)
    res = {}
    for kind in [k for k in a.baselines.split(",") if k]:
        t0 = time.perf_counter()
        f = ev.make_evaluator(cfg, lut, ev.make_policy(kind, cfg, y), 10000)
        r = jax.device_get(f(None, consts, rscen, rkeys))
        s = jax.device_get(f(None, consts, sscen, skeys))
        res[kind + a.tag] = dict(replica=summarise(r), spawn=summarise(s), replica_U_draws=[float(x) for x in r["U"]])
        info("baseline", kind=kind, wall=time.perf_counter() - t0, U_rep=res[kind + a.tag]["replica"]["U"],
             U_spawn=res[kind + a.tag]["spawn"]["U"])
    for rd in [x for x in a.runs.split(",") if x]:
        with open(os.path.join(rd, "params.pkl"), "rb") as fh:
            pk = pickle.load(fh)
        cu = json.load(open(os.path.join(rd, "config_used.json")))
        yy = cu["yaml"]
        algo = pk["algo"]
        cfg_l = EnvCfg.from_yaml(yy)          # the run's own env config (e.g. g_on), else Case I defaults
        if algo == "ddpg":
            actor, _, _, _ = alg.make_ddpg(yy, cfg.K, cfg.T0)
            pol = ev.make_policy("ddpg", cfg_l, yy, actor=actor)
        else:
            pol = ev.make_policy("dql", cfg_l, yy, dql_cfg=alg.make_dql_cfg(yy, cfg.K, 64))
        f = ev.make_evaluator(cfg_l, lut, pol, 10000)
        per_seed = []
        for i, sd in enumerate(pk["seeds"]):
            p = jax.tree.map(lambda x: x[i], pk["params"])
            r = jax.device_get(f(p, consts, rscen, rkeys))
            s = jax.device_get(f(p, consts, sscen, skeys))
            per_seed.append(dict(seed=sd, replica=summarise(r), spawn=summarise(s)))
        name = os.path.basename(rd.rstrip("/")) + a.tag
        res[name] = dict(algo=algo, per_seed=per_seed,
                         replica_U_seed_mean=float(np.mean([p_["replica"]["U"] for p_ in per_seed])),
                         replica_U_seed_se=float(np.std([p_["replica"]["U"] for p_ in per_seed], ddof=1)
                                                 / np.sqrt(len(per_seed))) if len(per_seed) > 1 else None)
        info("learner", run=name, U_rep=[round(p_["replica"]["U"]) for p_ in per_seed],
             U_spawn=[round(p_["spawn"]["U"]) for p_ in per_seed])
    os.makedirs(a.out, exist_ok=True)
    out_p = os.path.join(a.out, "summary.json")
    old = json.load(open(out_p)) if os.path.exists(out_p) else {}
    old.update(res)
    write_json(out_p, old)
    for k, v in res.items():
        if "replica" in v:
            metric("arm", arm=k, U_rep=v["replica"]["U"], c=v["replica"]["c"], scan=v["replica"]["scan_term"],
                   tinit=v["replica"]["tinit"], never=v["replica"]["never"], U_spawn=v["spawn"]["U"],
                   tinit_spawn=v["spawn"]["tinit"], sat=v["replica"]["sat_rate"], paper_U=PAPER_I.get(k.replace(a.tag, "")),
                   paper_tinit=PAPER_TINIT.get(k.replace(a.tag, "")))
        else:
            for p_ in v["per_seed"]:
                metric("arm", arm=k, seed=p_["seed"], U_rep=p_["replica"]["U"], c=p_["replica"]["c"],
                       scan=p_["replica"]["scan_term"], tinit=p_["replica"]["tinit"], never=p_["replica"]["never"],
                       sat=p_["replica"]["sat_rate"], share=p_["replica"]["track_share"],
                       lam_multi=p_["replica"]["lam_multi"], U_spawn=p_["spawn"]["U"])


if __name__ == "__main__":
    main()

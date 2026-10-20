"""The re-evaluation regression check: with --replica v1 --key_scheme legacy the current code must reproduce the stored
full-run k_test values (runs/full/eval) -- baselines EA/OA on Case I draws 0..19 and every DQL Case I lane (same lane
set = same vmapped program). Writes runs/full_v2/legacy_repro.json."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import common  # noqa: E402
from src.common import info, load_calibration, load_yaml, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import data, eval as ev  # noqa: E402
import full_eval as G  # noqa: E402

data.REPLICA, ev.SLOT_KEY = "v1", "legacy"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    common.check_device()
    y, cal = load_yaml(), load_calibration()
    root = jax.random.PRNGKey(20261019)
    rk, sk = G.keys(root)
    out = {}
    cfg = G.case_cfg(y, "case1")
    lut = ev.make_lut(cfg)
    stored = json.load(open(os.path.join(REPO, "runs/full/eval/base_case1_A04.json")))
    for kind in ("ea", "oa"):
        f = ev.make_evaluator(cfg, lut, ev.make_policy(kind, cfg, y), G.T)
        r = jax.device_get(f(None, G.base_consts(cal, "A04"), G.bcast(G.case_scen(cfg, "case1"), 20), rk[:20]))
        a, b = np.asarray(r["U"]), np.asarray(stored[kind]["replica"]["U"][:20])
        out[f"base_{kind}"] = dict(max_rel=float(np.max(np.abs(a - b) / np.abs(b))))
        info("legacy repro", what=kind, **out[f"base_{kind}"])
    # DQL Case I, all 30 lanes, replica only (same program shape as full-run apart from the added summary outputs)
    rd = os.path.join(REPO, "runs/full/dql_case1")
    pk, cu = G.load_run(rd)
    yy = cu["yaml"]
    seeds = np.asarray([s for _, s in pk["lanes"]])
    idx = jnp.asarray(np.stack([np.arange(20 * s, 20 * s + 20) for s in seeds]))
    one = ev.make_evaluator(cfg, lut, G.lane_policy(pk, yy, cfg), G.T)
    rscen = jax.tree.map(lambda x: jnp.broadcast_to(x, (len(seeds), 20) + x.shape), G.case_scen(cfg, "case1"))
    f = jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0)))
    r = jax.device_get(f(pk["params"], G.lane_consts(pk), rscen, rk[idx]))
    st = json.load(open(os.path.join(REPO, "runs/full/eval/learn_dql_case1_case1.json")))
    a, b = np.asarray(r["U"]), np.asarray(st["replica"]["U"])
    rel = np.abs(a - b) / np.abs(b)
    out["dql_case1_lanes"] = dict(max_rel=float(rel.max()), n_bitwise=int((a == b).sum()), n=int(a.size),
                                  max_rel_seedmean=float(np.max(np.abs(a.mean(1) - b.mean(1)) / np.abs(b.mean(1)))))
    info("legacy repro", what="dql_case1", **out["dql_case1_lanes"])
    write_json(os.path.join(REPO, "runs/full_v2/legacy_repro.json"), out)


if __name__ == "__main__":
    main()

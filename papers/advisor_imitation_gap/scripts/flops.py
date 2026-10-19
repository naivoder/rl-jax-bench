"""Compute parity (full-scale): XLA cost analysis of ONE training update (rollout of 20 envs x 100 steps + 4 epochs x 2
minibatches) for one run of each arm, per environment, compiled on CPU. Every arm is the same program with a different
stage table (src/train.py ARMS), so FLOPs per update are identical across arms by construction; this script measures
them and checks that claim. Writes runs/report/flops.json.

    JAX_PLATFORMS=cpu python scripts/flops.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import bootstrap, info  # noqa: E402

bootstrap()
import jax  # noqa: E402

from src import model as M  # noqa: E402
from src import train as TR  # noqa: E402
from src.data import get_spec  # noqa: E402
from src.envs import lighthouse as LH  # noqa: E402

ARMS = ["ppo", "bc", "bc_tf1", "bc_ppo", "bc_then_ppo", "advisor"]


def flops_for(env):
    spec = get_spec(env)
    if env == "pd":
        net, cfg, ep, opt = M.PDNet(), TR.TrainCfg(env="pd", total_steps=300000), {"public": False}, 0.0
    elif env == "lh":
        net, cfg, ep, opt = M.LHNet(), TR.TrainCfg(env="lh", total_steps=300000, early_stop=True), {"i": 1, "j": 15}, LH.optimal_len(1)
    else:
        from src.envs import crossing as MG
        net, cfg, ep, opt = M.MGNet(n_actions=MG.N_ACTIONS), TR.TrainCfg(env="mg", total_steps=1000000), {}, 0.0
    upd = TR.make_update(cfg, spec, net)
    out = {}
    for arm in ARMS:
        runs = TR.build_runs(cfg, [dict(arm=arm, seed=0, lr=1e-3, alpha=8.0, beta=0.0, split=0.5, ep=ep, opt_len=opt)])
        run = jax.tree_util.tree_map(lambda x: x[0], runs)
        st = TR.init_run(cfg, spec, net, run)
        c = jax.jit(upd).lower(st, run).compile().cost_analysis()
        c = c[0] if isinstance(c, (list, tuple)) else c
        out[arm] = dict(flops_per_update=float(c.get("flops", float("nan"))),
                        bytes_per_update=float(c.get("bytes accessed", float("nan"))),
                        env_steps_per_update=cfg.steps_per_update, updates=cfg.n_updates)
        out[arm]["flops_per_env_step"] = out[arm]["flops_per_update"] / cfg.steps_per_update
        out[arm]["flops_per_run"] = out[arm]["flops_per_update"] * cfg.n_updates
        info("flops", env=env, arm=arm, gflop_per_update=round(out[arm]["flops_per_update"] / 1e9, 4),
             tflop_per_run=round(out[arm]["flops_per_run"] / 1e12, 3))
    return out


if __name__ == "__main__":
    res = {env: flops_for(env) for env in ("pd", "lh", "mg")}
    os.makedirs("runs/report", exist_ok=True)
    json.dump(res, open("runs/report/flops.json", "w"), indent=1)

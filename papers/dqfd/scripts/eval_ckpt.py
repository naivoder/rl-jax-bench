"""Greedy / eps-greedy evaluation of the latest interaction checkpoint of a finished (or interrupted) run.

usage: eval_ckpt.py OUT GAME DEMONSTRATOR ARMS(comma) SEEDS STEPS [EPS ...]   (arms/seeds/steps exactly as the run was launched)
Restores runs/<OUT>/_ckpt/<GAME>_<DEMO>_int (orbax) into the same Group structure and rolls out n fresh episodes per run.
"""
import json, os, sys
from _common import *
import jax.numpy as jnp, numpy as np
from src import train as T, data as D, algorithm as A
from src.envs import MinAtar
import orbax.checkpoint as ocp

out, game, demo_name, arms, S, steps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","), int(sys.argv[5]), int(float(sys.argv[6]))
epss = [float(x) for x in sys.argv[7:]] or [0.0, 0.01]
cfg = load_cfg(); cfg["run"]["bin_width"] = int(os.environ.get("BIN_W", 5000))
info = json.load(open(f"{REPO}/data/demos/{game}_demos.json"))
demo = D.load_demos(f"{REPO}/data/demos/{game}_{demo_name}.npz", info[demo_name]["sha256"])
env = MinAtar(game, max_steps=info.get("max_steps"), sticky_action_prob=info.get("sticky_action_prob", 0.0),
                 asterix_spawn_repair=info.get("asterix_spawn_repair", False))
int_arms = [x for x in arms if A.base_arm(x) not in A.NO_INTERACTION_ARMS]
g = T.Group(env, cfg, demo, int_arms, list(range(S)), 0, steps, 0, os.path.join(out, "_ckpt"), f"{game}_{demo_name}_int")
c = g.init_int(g.init(g.hp, g.rk), jax.vmap(lambda k: k[2])(g.rk))
m = g._mgr("int"); step = m.latest_step()
c = m.restore(step, args=ocp.args.StandardRestore(c))
jaxlog.info("restored", step=step)
res = {}
for eps in epss:
    f = jax.jit(jax.vmap(lambda p, k: g.fns.greedy_returns(p, k, 50, eps)))
    ret = np.asarray(f(c["params"], jax.vmap(lambda k: jax.random.fold_in(k[4], 7))(g.rk))).mean(1).reshape(len(int_arms), S)
    res[str(eps)] = {a: dict(mean=float(r.mean()), se=float(r.std(ddof=1) / np.sqrt(S)), per_seed=r.tolist()) for a, r in zip(int_arms, ret)}
    for a, r in zip(int_arms, ret):
        jaxlog.metric("ckpt-eval", step=step, eps=eps, arm=a, mean=float(r.mean()), se=float(r.std(ddof=1) / np.sqrt(S)))
json.dump(dict(step=step, results=res), open(os.path.join(out, f"ckpt_eval_step{step}.json"), "w"), indent=1)

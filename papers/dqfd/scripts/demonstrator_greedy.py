"""The demonstrators' own GREEDY (eps = 0) returns, the fair bar for the end-of-run greedy descriptor.
The demonstrator's true mean (data/demos/<game>_demos.json) is at eps = 0.01; a deterministic learner policy is compared with a
deterministic bar here.  1,000 fresh rollouts per (game, demonstrator, eps), keys fold_in(root,2),game -> fold_in(., 40 + j)."""
import argparse, json, os
from _common import *
import jax.numpy as jnp, numpy as np
from flax import serialization
from src.envs import MinAtar
from src.demonstrator import make_act_fn
from src.model import ActorCritic
from src import data as D

GAME_IDX = {"Breakout": 0, "SpaceInvaders": 1, "Asterix": 2, "Seaquest": 3}
ap = argparse.ArgumentParser()   # sticky env: --demo-dir data/demos_sticky --games Breakout --out runs/full/demonstrator_greedy_sticky.json
ap.add_argument("--demo-dir", default=f"{REPO}/data/demos"); ap.add_argument("--games", default=",".join(GAME_IDX))
ap.add_argument("--out", default=f"{REPO}/runs/full/demonstrator_greedy.json")
args = ap.parse_args()
cfg = load_cfg(); jaxlog.check_device()
root = jax.random.PRNGKey(0)
out = {}
for game in args.games.split(","):
    g = GAME_IDX[game]
    info = json.load(open(f"{args.demo_dir}/{game}_demos.json"))
    env = MinAtar(game, max_steps=info.get("max_steps"), sticky_action_prob=info.get("sticky_action_prob", 0.0),
                 asterix_spawn_repair=info.get("asterix_spawn_repair", False))
    net = ActorCritic(env.n_actions, channels=cfg["network"]["conv_channels"], hidden=cfg["ppo"]["hidden"])
    tmpl = net.init(jax.random.PRNGKey(0), jnp.zeros((1,) + env.obs_shape, jnp.float32))
    k_eval = jax.random.fold_in(jax.random.fold_in(root, 2), g)
    for j, (name, fn) in enumerate((("expert", f"ppo_{game}.msgpack"), ("subopt", f"ppo_{game}_subopt.msgpack"))):
        params = serialization.from_bytes(tmpl, open(f"{args.demo_dir}/{fn}", "rb").read())
        for e in (0.0, 0.01):
            r = D.policy_returns(env, make_act_fn(net, params, e, env), jax.random.fold_in(k_eval, 40 + 2 * j + int(e > 0)), 1000)
            out[f"{game}/{name}/eps{e}"] = dict(mean=float(r.mean()), se=float(r.std(ddof=1) / np.sqrt(len(r))), n=len(r), recorded_true_mean_eps001=info[name]["true_mean"])
            jaxlog.metric("demonstrator", game=game, who=name, eps=e, mean=float(r.mean()), se=float(r.std(ddof=1) / np.sqrt(len(r))))
json.dump(out, open(args.out, "w"), indent=1)

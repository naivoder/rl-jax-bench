"""Train the PPO demonstrator for one game, calibrate the sub-optimal demonstrator, collect + hash demo sets.

usage: make_demos.py GAME [--ppo-steps N]   (key tree: see src/train.py)
Outputs (DEMO_DIR): ppo_<game>.msgpack, <game>_expert.npz, <game>_subopt.npz (+.sha256), <game>_demos.json.
"""
import argparse, json, os, time
from _common import *
import jax.numpy as jnp
import numpy as np
from flax import serialization
from src.envs import MinAtar
from src.demonstrator import train_ppo, make_act_fn
from src import data as D

GAME_IDX = {"Breakout": 0, "SpaceInvaders": 1, "Asterix": 2, "Seaquest": 3, "Freeway": 4}
ap = argparse.ArgumentParser()
ap.add_argument("game"); ap.add_argument("--ppo-steps", type=float, default=None)
ap.add_argument("--demo-dir", default=os.path.join(REPO, "runs", "demos")); ap.add_argument("--top-seed", type=int, default=0)
ap.add_argument("--max-steps", type=int, default=None)
ap.add_argument("--sticky", type=float, default=0.0, help="MinAtar sticky_action_prob; 0 = the env of the main runs")
ap.add_argument("--asterix-repair", action="store_true", help="original-MinAtar Asterix spawn; off = the env of the main runs")
a = ap.parse_args()
cfg = load_cfg(); jaxlog.check_device()
env = MinAtar(a.game, max_steps=a.max_steps, sticky_action_prob=a.sticky, asterix_spawn_repair=a.asterix_repair)
g = GAME_IDX[a.game]
root = jax.random.PRNGKey(a.top_seed)
k_train = jax.random.fold_in(jax.random.fold_in(root, 0), g)    # demonstrator training key
k_demo = jax.random.fold_in(jax.random.fold_in(root, 1), g)     # demo collection key
k_eval = jax.random.fold_in(jax.random.fold_in(root, 2), g)     # demonstrator true-score key (disjoint from k_demo)
steps = int(a.ppo_steps or cfg["ppo"]["total_steps"])
os.makedirs(a.demo_dir, exist_ok=True)
t0 = time.time()
params, curve, net, snaps = train_ppo(env, k_train, cfg, steps, log=lambda m: print(m, flush=True), snap_fracs=cfg["ppo"]["snapshot_fractions"])
jaxlog.info("ppo trained", game=a.game, steps=steps, secs=time.time() - t0)
with open(os.path.join(a.demo_dir, f"ppo_{a.game}.msgpack"), "wb") as f:
    f.write(serialization.to_bytes(params))
dc = cfg["demos"]
CAL = dc["calibration_rollouts"]
exp_cal = float(D.policy_returns(env, make_act_fn(net, params, dc["expert_eps"], env), jax.random.fold_in(k_eval, 10), CAL).mean())
# --- PRIMARY weak demonstrator: the strongest EARLY PPO snapshot whose mean is <= fraction * expert
#     (systematically weaker, deterministic policy; eps-noise is denoised by supervised imitation)
snap_rows, chosen_snap = [], None
for fr in sorted(snaps):
    st, sp = snaps[fr]
    m = float(D.policy_returns(env, make_act_fn(net, sp, dc["expert_eps"], env), jax.random.fold_in(k_eval, 12), CAL).mean())
    snap_rows.append((fr, st, m)); jaxlog.metric("snapshot", game=a.game, frac=fr, steps=st, mean=m, expert=exp_cal)
    if m <= dc["suboptimal_max_fraction"] * exp_cal:
        chosen_snap = (fr, sp)
if chosen_snap is None:
    fr = min(snaps); chosen_snap = (fr, snaps[fr][1]); jaxlog.warn("no snapshot <= fraction of expert; using the earliest", frac=fr)
# --- eps-noise variant (first definition; kept as 'subopt_eps')
rows, chosen = [], None
for e in dc["suboptimal_eps_grid"]:
    m = float(D.policy_returns(env, make_act_fn(net, params, e, env), jax.random.fold_in(k_eval, 11), CAL).mean())
    rows.append((e, m)); jaxlog.metric("calibration", game=a.game, eps=e, mean=m, expert=exp_cal)
    if chosen is None and m <= dc["suboptimal_max_fraction"] * exp_cal:
        chosen = e
if chosen is None:
    chosen = dc["suboptimal_eps_grid"][-1]; jaxlog.warn("no eps reached the fraction; using the largest", eps=chosen)
out = dict(game=a.game, ppo_steps=steps, ppo_curve_tail=curve[-3:], chosen_eps=chosen, calibration=rows,
           snapshot_rows=snap_rows, chosen_snapshot_frac=chosen_snap[0], max_steps=env.max_steps, sticky_action_prob=a.sticky, asterix_spawn_repair=a.asterix_repair, expert_calibration_mean=exp_cal)
with open(os.path.join(a.demo_dir, f"ppo_{a.game}_subopt.msgpack"), "wb") as f:
    f.write(serialization.to_bytes(chosen_snap[1]))
for j, (name, pr, eps) in enumerate((("expert", params, dc["expert_eps"]), ("subopt", chosen_snap[1], dc["expert_eps"]), ("subopt_eps", params, chosen))):
    act = make_act_fn(net, pr, eps, env)
    d = D.collect_demos(env, act, jax.random.fold_in(k_demo, j), dc["n_transitions_min"])
    h = D.save_demos(os.path.join(a.demo_dir, f"{a.game}_{name}.npz"), d)
    true = D.policy_returns(env, act, jax.random.fold_in(k_eval, 20 + j), dc["demonstrator_eval_rollouts"])
    out[name] = dict(eps=eps, sha256=h, true_mean=float(true.mean()), true_se=float(true.std(ddof=1) / np.sqrt(len(true))),
                     n_rollouts=len(true), **d["meta"])
    jaxlog.metric("demo set", game=a.game, demonstrator=name, **{k: v for k, v in out[name].items() if k != "sha256"})
    if d["meta"]["cap_fraction"] > 0.2:
        jaxlog.warn("more than 20% of demo episodes hit the episode cap", frac=d["meta"]["cap_fraction"], who=name)
with open(os.path.join(a.demo_dir, f"{a.game}_demos.json"), "w") as f:
    json.dump(out, f, indent=1)
jaxlog.info("done", secs=time.time() - t0)

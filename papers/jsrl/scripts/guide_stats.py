"""Guide statistics: for each game, >= 1000 rollouts of the guide at eps_g = 0.01 on the
env the learners will use -> true mean +- SE, mean episode length (= H1), the stage list h_0..h_n, sha256 of guide and demo files.
Key: fold_in(fold_in(root, 3), game_idx) (disjoint from the earlier DQfD reproduction's 0..2).  Asterix (spawn-repaired env): the demo set used by
DQfD / IMIT / BCinit-FT is RE-COLLECTED from the same snapshot on the repaired env (key fold_in(fold_in(fold_in(root,1), g), 3)).
usage: guide_stats.py [--sticky P] GAME [GAME ...]  ->  data/demos/<GAME>_guide.json (+ data/demos/Asterix_subopt_repaired.npz)
--sticky P: measure on the env with sticky_action_prob P and write data/demos/<GAME>_guide_sticky<P>.json (demo fields copied from
<GAME>_guide.json; no demo re-collection)"""
import json, os, time
from _common import *
import numpy as np
import jax.numpy as jnp
from src.envs import MinAtar
from src.guide import make_guide_fn, GUIDE_DIR
from src import data as D
from src.curriculum import stage_list

GAME_IDX = {"Breakout": 0, "SpaceInvaders": 1, "Asterix": 2, "Seaquest": 3}
import sys
STICKY = 0.0
if len(sys.argv) > 2 and sys.argv[1] == "--sticky":
    STICKY = float(sys.argv[2]); del sys.argv[1:3]
cfg = load_cfg(); jaxlog.check_device()
root = jax.random.PRNGKey(0)
for game in sys.argv[1:]:
    g = GAME_IDX[game]
    info = json.load(open(os.path.join(REPO, "data", "demos", f"{game}_demos.json")))
    env = MinAtar(game, max_steps=info["max_steps"], sticky_action_prob=STICKY)
    guide = make_guide_fn(env, cfg, cfg["jsrl"]["guide_eps"])
    act = lambda obs, key: jax.vmap(guide)(obs, jax.random.split(key, obs.shape[0]))
    key = jax.random.fold_in(jax.random.fold_in(root, 3), g)
    n, chunk = cfg["jsrl"]["guide_rollouts"], 250
    run = jax.jit(lambda k: D.rollout_episodes(env, act, k, chunk, keep_transitions=False))
    rets, lens = [], []
    t0 = time.time()
    for i in range(-(-n // chunk)):
        tr = run(jax.random.fold_in(key, i))
        rets.append(np.asarray(tr["r"].sum(0))); lens.append(np.asarray(tr["valid"].sum(0)))
    rets, lens = np.concatenate(rets)[:n], np.concatenate(lens)[:n]
    out = dict(game=game, max_steps=env.max_steps, asterix_spawn_repair=env.asterix_spawn_repair, n_rollouts=int(n),
               true_mean=float(rets.mean()), true_se=float(rets.std(ddof=1) / np.sqrt(n)), true_sd=float(rets.std(ddof=1)),
               mean_len=float(lens.mean()), len_sd=float(lens.std(ddof=1)), cap_fraction=float(np.mean(lens >= env.max_steps)),
               stage_list=stage_list(lens.mean(), cfg["jsrl"]["n_stages"]), key="fold_in(fold_in(root,3),game_idx)",
               guide_sha256=D.sha256_file(os.path.join(GUIDE_DIR, f"ppo_{game}_subopt.msgpack")),
               iter22_true_mean=info["subopt"]["true_mean"], iter22_true_se=info["subopt"]["true_se"], secs=time.time() - t0)
    if STICKY > 0:          # sticky variant: keep the non-sticky guide JSON's demo fields, add the sticky marker
        base = json.load(open(os.path.join(REPO, "data", "demos", f"{game}_guide.json")))
        out.update(demo_file=base["demo_file"], demo_sha256=base["demo_sha256"], demo_meta=None, sticky_action_prob=STICKY)
        json.dump(out, open(os.path.join(REPO, "data", "demos", f"{game}_guide_sticky{STICKY:g}.json"), "w"), indent=1)
        jaxlog.metric("guide_sticky", **{k: v for k, v in out.items() if k not in ("demo_meta", "stage_list", "guide_sha256", "demo_sha256", "old_demo_sha256")})
        jaxlog.info("stage list", game=game, stages=out["stage_list"])
        continue
    if game == "Asterix":   # spawn repair: re-collect the demo set from the same snapshot on the repaired env
        k_demo = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(root, 1), g), 3)
        d = D.collect_demos(env, act, k_demo, 17500)
        path = os.path.join(REPO, "data", "demos", "Asterix_subopt_repaired.npz")
        h = D.save_demos(path, d)
        out.update(demo_file="Asterix_subopt_repaired.npz", demo_sha256=h, demo_meta=d["meta"],
                   old_demo_sha256=info["subopt"]["sha256"])
    else:
        out.update(demo_file=f"{game}_subopt.npz", demo_sha256=info["subopt"]["sha256"], demo_meta=None)
    json.dump(out, open(os.path.join(REPO, "data", "demos", f"{game}_guide.json"), "w"), indent=1)
    jaxlog.metric("guide", **{k: v for k, v in out.items() if k not in ("demo_meta", "stage_list", "guide_sha256", "demo_sha256", "old_demo_sha256")})
    jaxlog.info("stage list", game=game, stages=out["stage_list"])

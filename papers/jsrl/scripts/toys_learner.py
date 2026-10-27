"""Lock-g (DESCRIPTIVE, no closed form, reported not graded): the REAL training code (src/algorithm.py: PDD DQN + JSRL roll-in + curriculum)
on the combination lock, JSRL-curriculum vs the same learner from scratch (PDD), H in {8, 12, 16, 20}, A = 2.
Prediction (Theorem 4.1 vs A.3): scratch's first-reward time grows like A^H, JSRL's polynomially.  Metric per run: the first evaluation
point (env steps) at which R_alone (pi_e alone, eps 0.01, 32 episodes) >= 0.9, censored at the budget.
Toy-specific learner settings (same for both arms): lr 1e-3, target period 500, learning starts 500, eps 1 -> 0.01 over 20k steps,
n_stages 5, MA horizon 3, chunk 250 steps.   usage: toys_learner.py [STEPS]   (GPU, tiny)"""
import json, os, sys
from _common import *
import numpy as np
import jax.numpy as jnp
import copy
from src import train as T
from src.lock import LockEnv

steps = int(float(sys.argv[1])) if len(sys.argv) > 1 else 100000
cfg = T.load_cfg()
cfg["optim"].update(lr=1e-3, target_update_period=500)
cfg["pdd_dqn"].update(learning_starts=500, eps_decay_steps=20000)
cfg["jsrl"].update(n_stages=5, ma_horizon=3)
cfg["run"]["n_eval_points"] = steps // 250
out = {}
for H in (8, 12, 16, 20):
    env = LockEnv(H, 2)
    fake = dict(obs=np.zeros((0, 25), np.uint8), nobs=np.zeros((0, 25), np.uint8), act=np.zeros(0, np.int32), rew=np.zeros(0, np.float32),
                term=np.zeros(0, bool), end=np.zeros(0, bool), sha256="none")
    g = T.Group(env, cfg, fake, env.guide_fn, ["JSRL", "PDD"], list(range(5)), 0, steps, 0, "/tmp/g1g_ckpt", f"g1g{H}", float(H))
    c, _ = g.run_interact(g.init(g.hp, g.rk), False, 0)
    ev = np.asarray(c["ev_alone"]).reshape(2, 5, -1)
    stg = np.asarray(c["ev_stage"]).reshape(2, 5, -1)
    first = [[(int(np.argmax(ev[a, s, 1:] >= 0.9) + 1) * g.chunk if (ev[a, s, 1:] >= 0.9).any() else None) for s in range(5)] for a in range(2)]
    out[H] = dict(jsrl_first_solved_steps=first[0], scratch_first_solved_steps=first[1], jsrl_final=ev[0, :, -3:].mean(1).tolist(),
                  scratch_final=ev[1, :, -3:].mean(1).tolist(), jsrl_stage_final=stg[0, :, -1].tolist())
    print(f"[Lock-g] H={H}: JSRL first >=0.9 at steps {first[0]}  | scratch {first[1]} | final R JSRL {np.round(ev[0, :, -3:].mean(1), 2).tolist()} scratch {np.round(ev[1, :, -3:].mean(1), 2).tolist()}", flush=True)
json.dump(out, open(os.path.join(T.REPO, "runs", "toys", "lock_learner_large.json"), "w"), indent=1)

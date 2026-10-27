"""Two differential tests of our learner core ('a dry run with JSRL forced to h = 0 is bit-identical to PDD'):
  (a) same program, arms PDD vs JSRL-h0 (curriculum machinery with every stage forced to h = 0), 1 seed: params and every evaluation
      point must be BIT-identical (nothing of the curriculum / guide leaks into the learner when h = 0);
  (b) our PDD (one 10000-step chunk, same key layout) vs the earlier DQfD reproduction's PDD saved to REF.npz by that project's code
      (not shipped here; (b) is skipped without REF.npz): parameter agreement.
usage: identity_checks.py GAME STEPS [REF.npz]"""
import os, sys
from _common import *
import numpy as np
import jax.numpy as jnp
from src import train as T, algorithm as A

game, steps = sys.argv[1], int(sys.argv[2])
ref = sys.argv[3] if len(sys.argv) > 3 else None
cfg = T.load_cfg()
cfg["run"]["n_eval_points"] = 1 if ref else 50       # one chunk = the whole run when comparing with the earlier DQfD reproduction (same key layout)
env, gi, demo = T.load_game(game, os.path.join(T.REPO, "data", "demos"))
from src.guide import make_guide_fn
guide = make_guide_fn(env, cfg, cfg["jsrl"]["guide_eps"])
g = T.Group(env, cfg, demo, guide, ["PDD", "JSRL-h0"], [0], 0, steps, 0, "/tmp/jsrl_ident_ckpt", "ident", gi["mean_len"])
state = g.init(g.hp, g.rk)
c, _ = g.run_interact(state, False, 0)
pa = jax.tree_util.tree_map(lambda x: np.asarray(x[0]), c["params"]); pb = jax.tree_util.tree_map(lambda x: np.asarray(x[1]), c["params"])
la, lb = jax.tree_util.tree_leaves(pa), jax.tree_util.tree_leaves(pb)
bit = all(np.array_equal(x, y) for x, y in zip(la, lb))
ev_same = all(np.array_equal(np.asarray(c[k][0]), np.asarray(c[k][1])) for k in ("ev_alone", "ev_thr", "ev_greedy"))
maxd = max(float(np.abs(x - y).max()) for x, y in zip(la, lb))
print(f"[IDENT-a] {game} {steps} steps: PDD vs JSRL-h0 params bit-identical={bit} (max |diff| {maxd:.3g}); eval points identical={ev_same}; guide_steps={np.asarray(c['guide_steps']).tolist()}; stages={np.asarray(c['cur']['stage']).tolist()}")
res = dict(game=game, steps=steps, ident_a_bit_identical=bool(bit), ident_a_eval_identical=bool(ev_same), ident_a_maxdiff=maxd)
if ref:
    z = np.load(ref); lr = [z[f"arr_{i}"] for i in range(len(la))]
    bit_ref = all(np.array_equal(x, y) for x, y in zip(la, lr))
    maxd_ref = max(float(np.abs(x - y).max()) for x, y in zip(la, lr))
    scale = max(float(np.abs(y).max()) for y in lr)
    print(f"[IDENT-b] {game} {steps} steps: our PDD vs the earlier DQfD reproduction PDD: bit-identical={bit_ref}, max |diff|={maxd_ref:.3g} (max |param| {scale:.3g})")
    res.update(ident_b_bit_identical=bool(bit_ref), ident_b_maxdiff=maxd_ref, ident_b_param_scale=scale)
import json
os.makedirs(os.path.join(T.REPO, "runs", "ident"), exist_ok=True)
json.dump(res, open(os.path.join(T.REPO, "runs", "ident", f"{game}_{steps}.json"), "w"), indent=1)

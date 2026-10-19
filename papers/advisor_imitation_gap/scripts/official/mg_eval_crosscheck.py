"""MiniGrid LC Once: evaluate the AUTHORS' trained policies (ref_runs/out/mg*_weights.npz, run_reference.py
--save_weights) inside OUR evaluation harness (our JAX env + layout generator + make_eval, sampled actions) and compare
with the reward/success the authors' own test harness reported for the same weights. A policy-level differential test
of our env, observation, model forward and metric. Writes runs/official_ref/mg_crosscheck.json.

    uv run python scripts/official/mg_eval_crosscheck.py [n_eval=1000]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common import OFFICIAL_ROOT, bootstrap, check_device  # noqa: E402

bootstrap()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import model as M  # noqa: E402
from src import train as TR  # noqa: E402
from src.data import get_spec  # noqa: E402
from src.envs import crossing as MG  # noqa: E402

OUTD = os.path.join(OFFICIAL_ROOT, "ref_runs", "out")
n_eval = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
plat = check_device()
spec, net = get_spec("mg"), M.MGNet(n_actions=MG.N_ACTIONS)
cfg = TR.TrainCfg(env="mg", total_steps=300000)
ev = jax.jit(TR.make_eval(cfg, spec, net, n_eval))
nA = MG.N_ACTIONS


def to_flax(sd):
    W, b = sd["actor_critic.ac_nonrecurrent_head.linear.weight"], sd["actor_critic.ac_nonrecurrent_head.linear.bias"]
    K = np.zeros((W.shape[1], 2 * nA + 1), np.float32)
    B = np.zeros(2 * nA + 1, np.float32)
    K[:, :nA], B[:nA] = W[:nA].T, b[:nA]
    K[:, -1], B[-1] = W[-1], b[-1]
    if W.shape[0] == 2 * nA + 1:
        K[:, nA:2 * nA], B[nA:2 * nA] = W[nA:2 * nA].T, b[nA:2 * nA]
    se = "actor_critic.state_encoder."
    p = {"core": {"w_ih": sd[se + "rnn.weight_ih_l0"].T, "w_hh": sd[se + "rnn.weight_hh_l0"].T,
                  "b_ih": sd[se + "rnn.bias_ih_l0"], "b_hh": sd[se + "rnn.bias_hh_l0"]},
         "init_state": sd[se + "init_hidden_state"].reshape(-1), "head": {"head": {"kernel": K, "bias": B}},
         "eo": {"embedding": sd["object_embedding.weight"]}, "ec": {"embedding": sd["color_embedding.weight"]},
         "es": {"embedding": sd["state_embedding.weight"]}}
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float32), {"params": p})


rows = []
for f in sorted(glob.glob(f"{OUTD}/mg*_weights.npz")):
    meta = json.load(open(f.replace("_weights.npz", ".json")))
    params = to_flax(dict(np.load(f)))
    reps = []
    for rep in range(3):
        info = ev(params, {"ep": {}, "key": jax.random.PRNGKey(100 + rep)}, 0)
        reps.append((float(np.mean(info["ret"])), float(np.mean(info["success"]))))
    rows.append(dict(name=os.path.basename(f)[:-12], exp=meta["exp"], seed=meta["seed"],
                     official_reward=meta["reward"][-1], official_success=meta.get("success", [None])[-1],
                     ours_harness_reward=[r[0] for r in reps], ours_harness_success=[r[1] for r in reps]))
    print("[METRIC]", json.dumps(rows[-1]))
os.makedirs("runs/official_ref", exist_ok=True)
json.dump({"n_eval": n_eval, "platform": plat, "rows": rows}, open("runs/official_ref/mg_crosscheck.json", "w"), indent=1)

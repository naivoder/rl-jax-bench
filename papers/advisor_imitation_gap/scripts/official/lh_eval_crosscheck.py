"""2D-LH no-gap cell (i = j = 1): are OUR trained policies worse than the AUTHORS' code's, or is the +4-step preliminary
gap the 200-episode evaluation's heavy tail (one 1000-step failure adds ~4.6 steps)? Trains ours (BC, ADVISOR) on
seeds 10..19 with the preliminary settings, loads the authors' trained weights (ref_runs/diag/*_weights.npz from
run_reference.py --save_weights) and evaluates BOTH with the same harness (our make_eval, sampled actions,
balanced goals in an earlier version; the official test-goal list for the first 200 episodes since then) on 200 and 4000
episodes. Writes runs/official_ref/lh_crosscheck.json, or the path given as argv[1] (re-run on the official
goals: runs/official_ref/lh_crosscheck_og.json).

    JAX_PLATFORMS=cpu uv run python scripts/official/lh_eval_crosscheck.py [out.json]
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
from scipy import stats  # noqa: E402

from src import model as M  # noqa: E402
from src import train as TR  # noqa: E402
from src.data import get_spec  # noqa: E402
from src.envs import lighthouse as LH  # noqa: E402

DIAG = os.path.join(OFFICIAL_ROOT, "ref_runs", "diag")
plat = check_device()
spec, net = get_spec("lh"), M.LHNet()
cfg = TR.TrainCfg(env="lh", total_steps=300000, early_stop=True)
ep = {"i": 1, "j": 1}
seeds = list(range(10, 20))
specs = [dict(arm=a, seed=s, lr=0.0242, alpha=20.0, beta=0.0, split=0.5, ep=ep, opt_len=LH.optimal_len(1), cell="i1_j1")
         for a in ("bc", "advisor") for s in seeds]
_, evals, st = TR.run_experiment(cfg, spec, net, specs, eval_at=[0, cfg.n_updates], n_eval=200)
finish = np.asarray(st["finish_steps"])
evs = {n: jax.jit(TR.make_eval(cfg, spec, net, n)) for n in (200, 4000)}


def evaluate(params, key):
    run = {"ep": {"i": jnp.int32(1), "j": jnp.int32(1)}, "key": jax.random.PRNGKey(key)}
    out = {}
    for n, ev in evs.items():
        info = ev(params, run, 7)
        ln = np.asarray(info["len"])
        out[f"len{n}"] = float(ln.mean())
        out[f"fail{n}"] = float(1.0 - np.asarray(info["found"]).mean())
        out[f"len{n}_no_fail"] = float(ln[np.asarray(info["found"]) > 0].mean())
    return out


res = {"ours": {"bc": [], "advisor": []}, "authors": {"bc": [], "advisor_ppo": []}}
for k, r in enumerate(specs):
    p = jax.tree_util.tree_map(lambda x: x[k], st["params"])
    res["ours"][r["arm"]].append(dict(seed=r["seed"], train_steps=int(finish[k]),
                                      train_eval200=float(np.asarray(evals[-1][1]["len"])[k].mean()), **evaluate(p, 1000 + k)))
for f in sorted(glob.glob(f"{DIAG}/lh_*_i1_j1_s*_weights.npz")):
    meta = json.load(open(f.replace("_weights.npz", ".json")))
    w = np.load(f)
    W, b = w["linear.weight"], w["linear.bias"]
    K = np.zeros((6400, 9), np.float32)
    B = np.zeros(9, np.float32)
    K[:, :4], B[:4] = W[:4].T, b[:4]           # main logits
    K[:, -1], B[-1] = W[-1], b[-1]             # value (aux rows: zeros for BC's LinearActorCritic; ADV: rows 4..7)
    if W.shape[0] == 9:
        K[:, 4:8], B[4:8] = W[4:8].T, b[4:8]
    params = {"params": {"kernel": jnp.asarray(K), "bias": jnp.asarray(B)}}
    res["authors"][meta["exp"]].append(dict(seed=meta["seed"], train_steps=int(meta["train_steps"][-1]),
                                            official_test200=float(meta["ep_length"][-1]),
                                            **evaluate(params, 5000 + meta["seed"])))
summary = {}
for who, d in res.items():
    for arm, rows in d.items():
        if rows:
            summary[f"{who}/{arm}"] = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1) / np.sqrt(len(rows)))]
                                       for k in rows[0] if k != "seed"} | {"n": len(rows)}
for a_o, a_a in (("bc", "bc"), ("advisor", "advisor_ppo")):
    x = [r["len4000"] for r in res["ours"][a_o]]
    y = [r["len4000"] for r in res["authors"][a_a]]
    if len(y) >= 3:
        summary[f"mannwhitney_len4000/{a_o}"] = float(stats.mannwhitneyu(x, y).pvalue)
        x2 = [r["len4000_no_fail"] for r in res["ours"][a_o]]
        y2 = [r["len4000_no_fail"] for r in res["authors"][a_a]]
        summary[f"mannwhitney_len4000_no_fail/{a_o}"] = float(stats.mannwhitneyu(x2, y2).pvalue)
OUT = sys.argv[1] if len(sys.argv) > 1 else "runs/official_ref/lh_crosscheck.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"runs": res, "summary": summary, "platform": plat}, open(OUT, "w"), indent=1)
for k, v in summary.items():
    print("[METRIC]", k, v)

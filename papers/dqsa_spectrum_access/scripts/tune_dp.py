"""Give the RECONSTRUCTED DP baseline its best shot on TUNING worlds (world offset 100000, disjoint from the reporting worlds 0..999):
grid over p_update x tau (grid after the review, p_update x tau = 8 x 9 = 72 configs, tau 0.01..5; the first extension, tau 0.02..1.0, still had its optimum on the edge for the proxy), objective = plateau average rate (the Fig. 4
metric), averaged over `--reps` action-key replicates on `n_exp` tuning worlds.  The winner is frozen and used on the reporting worlds; selecting
on the graded metric can only make the baseline stronger, i.e. it is conservative for 'DQSA > DP'.
    python scripts/tune_dp.py E2 --n-exp 200 --reps 2 --rule gibbs      (rule: gibbs = version, count = pre-review same-slot version)"""
import argparse, json, os, sys
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import data as D, eval as E, train as T
from rl_jax_bench import jaxlog
ap = argparse.ArgumentParser()
ap.add_argument("exp")
ap.add_argument("--n-exp", type=int, default=200)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--rule", default="gibbs", choices=["gibbs", "count"])
a = ap.parse_args()
cfg = T.load_exp(a.exp)
spec = D.EnvSpec(**cfg["env"])
tc = cfg["test"]
res = []
for pu in (1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02):
    for tau in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
        rr, ru = [], []
        for rep in range(a.reps):
            r = E.eval_baseline(spec, "dp", n_exp=a.n_exp, T=tc["T"], replicate=rep, chunk=50 if spec.n_max > 50 else 100,
                                world_offset=E.MONITOR_WORLD_OFFSET, p_update=pu, tau=tau, tx_rule=a.rule)
            lo, hi = tc["window"]
            rr.append(float(r["rate_t"][:, lo:hi].mean())); ru.append(float(r["util_t"][:, lo:hi].mean()))
        res.append(dict(p_update=pu, tau=tau, plateau_rate=float(np.mean(rr)), plateau_util=float(np.mean(ru))))
        jaxlog.metric("dp_tune", exp=a.exp, rule=a.rule, p_update=pu, tau=tau, rate=res[-1]["plateau_rate"])
best = max(res, key=lambda x: x["plateau_rate"])
edge = best["tau"] in (0.01, 5.0) or best["p_update"] in (1.0, 0.02)
jaxlog.info("best", edge_of_grid=edge, **best)
os.makedirs(os.path.join(ROOT, "runs", "baselines_v2"), exist_ok=True)
jaxlog.write_json(os.path.join(ROOT, "runs", "baselines_v2", f"dp_tuning_{a.exp}_{a.rule}.json"),
                  dict(exp=a.exp, rule=a.rule, n_exp=a.n_exp, reps=a.reps, world_offset=E.MONITOR_WORLD_OFFSET, grid=res, best=best, best_on_grid_edge=edge))

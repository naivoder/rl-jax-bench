"""(recheck): slot-duration T_s sweep of every CSI-reading baseline, with the DP baseline RE-TUNED at each T_s on tuning worlds
(offset 100000), evaluated on the reporting worlds.  T_s in {0.1, 1, 3, 10} ms -> rho = J0(2 pi f_D T_s) = 0.999 / 0.904 / 0.291 / 0.220.
DQSA with the static capacity input (primary) does not read h, so it is T_s-invariant in expectation; the H2 bar is the strongest comparator over the range.
    python scripts/ts_sweep.py E2proxy --n-exp 200 --n-tune 100     (E2/E3: --n-exp 50 --n-tune 12 --grid small)
Also records the log-rate variants  of every arm at t = 50 / 100, so the unit-invariance rule for H3 can be read from the same worlds."""
import argparse, json, os, sys
from dataclasses import replace
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import data as D, eval as E, train as T
from rl_jax_bench import jaxlog

ap = argparse.ArgumentParser()
ap.add_argument("exp")
ap.add_argument("--n-exp", type=int, default=200)
ap.add_argument("--n-tune", type=int, default=100)
ap.add_argument("--grid", default="small", choices=["small"])
ap.add_argument("--slots", type=float, nargs="+", default=[1e-4, 1e-3, 3e-3, 1e-2])
ap.add_argument("--out", default=os.path.join(ROOT, "runs", "recheck", "rerun"))
a = ap.parse_args()
cfg = T.load_exp(a.exp)
base = D.EnvSpec(**cfg["env"])
PU, TAU = (0.05, 0.1, 0.2, 0.3, 0.5), (0.05, 0.1, 0.2, 0.5)
chunk = 25 if base.n_max > 50 else 100
res = {}
for slot in a.slots:
    spec = replace(base, slot_s=slot)
    key = f"Ts={slot:g}"
    grid = []
    for pu in PU:
        for tau in TAU:
            r = E.eval_baseline(spec, "dp", n_exp=a.n_tune, T=100, replicate=0, chunk=chunk, world_offset=E.MONITOR_WORLD_OFFSET, p_update=pu, tau=tau, tx_rule="gibbs")
            grid.append(dict(p_update=pu, tau=tau, rate=float(r["rate_t"][:, 60:100].mean())))
    best = max(grid, key=lambda g: g["rate"])
    arms = {"aloha": ("aloha", {}), "oca": ("oca", {}), "oca_sameslot": ("oca_sameslot", {}),
            "dp_tuned_at_Ts": ("dp", dict(tx_rule="gibbs", p_update=best["p_update"], tau=best["tau"])),
            "dp_literal_tau0.5": ("dp", dict(tx_rule="gibbs", p_update=1.0, tau=0.5))}
    row = dict(rho=float(spec.rho), dp_best=best, dp_grid=grid)
    for nm, (pol, kw) in arms.items():
        r = E.eval_baseline(spec, pol, n_exp=a.n_exp, T=100, replicate=0, chunk=chunk, **kw)
        s = E.summarize(spec, r, (60, 100))
        lrv = {k[4:]: [float(r[k][:, 49].mean()), float(r[k][:, 99].mean())] for k in r if k.startswith("lrv_")}
        row[nm] = dict(rate=s["plateau_rate"], util=s["plateau_util"], rate_t0=s["rate_curve"][0], lograte_t50_t100=lrv)
        jaxlog.metric("ts", exp=a.exp, Ts=slot, arm=nm, rate=round(s["plateau_rate"], 2), best=str(best))
    res[key] = row
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(exp=a.exp, n_exp=a.n_exp, n_tune=a.n_tune, results=res), open(os.path.join(a.out, f"ts_sweep_{a.exp}.json"), "w"), indent=1)

"""Evaluate the non-learning arms (Aloha, TDMA genie, OCA, DP) on the shared eval worlds, `--reps` replicates (action-key seeds).
    python scripts/baselines_eval.py --exp E2proxy --arms aloha oca dp tdma_genie --reps 10 --out runs/baselines
Writes <out>/<exp>/<arm>.json: per-replicate plateau scalars + the replicate-mean curves."""
import argparse
import json
import os
import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np  # noqa: E402

import data as D  # noqa: E402
import eval as E  # noqa: E402
from rl_jax_bench import jaxlog  # noqa: E402
import train as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--arms", nargs="+", default=["aloha", "tdma_genie"])
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--rep-ids", type=int, nargs="+", default=None, help="explicit replicate ids (e.g. the training seeds, for paired comparison)")
ap.add_argument("--n-exp", type=int, default=None)
ap.add_argument("--out", default=os.path.join(ROOT, "runs", "baselines_v2"))
a = ap.parse_args()
cfg = T.load_exp(a.exp)
spec = D.EnvSpec(**cfg["env"])
tc = cfg["test"]
n_exp = a.n_exp or tc["n_exp"]
os.makedirs(os.path.join(a.out, a.exp), exist_ok=True)
def _tuned(rule):
    base = a.exp.replace("E3", "E2")           # E3 / E3proxy reuse the PU-free tuning of E2 / E2proxy (declared; tuning worlds have no PUs)
    for suf in ("csi", "fixed"):
        base = base.replace(suf, "")
    p = os.path.join(ROOT, "runs", "baselines_v2", f"dp_tuning_{base}_{rule}.json")
    return json.load(open(p))["best"]


def arm_kw(arm):
    """Arm name -> (policy name in algorithm.BASELINES, kwargs).
    dp_lit_tau<t>            : DP, tx_rule gibbs, p_update = 1 ('literal', tau ASSUMED)
    dpv1_lit_tau<t>          : pre-review DP (same-slot oracle tx probability), p_update = 1
    dp_tuned / dpv1_tuned    : best (p_update, tau) of scripts/tune_dp.py on the TUNING worlds (runs/baselines_v2/dp_tuning_<exp>_<rule>.json)
    oca / oca_sameslot       : version (previous-slot counts, PU-aware) / pre-review version"""
    if arm == "dp_tuned" or arm == "dpv1_tuned":
        rule = "gibbs" if arm == "dp_tuned" else "count"
        j = _tuned(rule)
        return "dp", dict(p_update=j["p_update"], tau=j["tau"], tx_rule=rule)
    if arm.startswith("dp_lit_tau") or arm.startswith("dpv1_lit_tau"):
        v1 = arm.startswith("dpv1")
        return "dp", dict(p_update=1.0, tau=float(arm.split("tau")[1]), tx_rule="count" if v1 else "gibbs")
    return arm, {}


for arm in a.arms:
    reps, curves = [], []
    fn_arm, kw = arm_kw(arm)
    for rep in (a.rep_ids if a.rep_ids else range(a.reps)):
        r = E.eval_baseline(spec, fn_arm, n_exp=n_exp, T=tc["T"], replicate=rep, chunk=50 if spec.n_max > 50 else 100, **kw)
        s = E.summarize(spec, r, tuple(tc["window"]), tc["kind"])
        curves.append({k: s[k] for k in ("util_curve", "rate_curve", "lr_strict_curve", "lr_user_curve")})
        reps.append(dict(rep=rep, **{k: v for k, v in s.items() if not k.endswith("_curve")}))
        jaxlog.metric("baseline", exp=a.exp, arm=arm, rep=rep, plateau_util=s["plateau_util"], plateau_rate=s["plateau_rate"],
                      lograte_final=s["lograte_final"], jain=s["jain_all"])
    mean_curve = {k: np.mean([c[k] for c in curves], 0).tolist() for k in curves[0]}
    jaxlog.write_json(os.path.join(a.out, a.exp, f"{arm}.json"), dict(
        exp=a.exp, arm=arm, reconstructed=fn_arm in ("oca", "dp", "oca_sameslot", "dp_sameslot"), kw=kw, n_exp=n_exp, T=tc["T"], window=tc["window"],
        per_rep=reps, mean_curves=mean_curve,
        plateau_util_mean=float(np.mean([r["plateau_util"] for r in reps])), plateau_rate_mean=float(np.mean([r["plateau_rate"] for r in reps]))))

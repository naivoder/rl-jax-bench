"""Small-scale report: per-arm end-point (last 50k-step bin, official plot.py binning) mean +- s.e., paired comparisons in the
paper's direction, the official-code reference curve at the same budget, PPO on the same proxy, ETTm2 MSEs.
Usage: python scripts/small_report.py [tag]   -> runs/<tag>/report.json"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src import eval as ev  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "small"
root = os.path.join(REPO, "runs", tag)
rep = {}


def load(exp):
    out = {}
    for p in glob.glob(os.path.join(root, exp, "*", "seed_*", "metrics.json")):
        r = json.load(open(p))
        out.setdefault(r["arm"], {})[r["seed"]] = r
    return out


def endpoint(runs, key="final_bin"):
    return {s: r["summary"][key] for s, r in sorted(runs.items())}


def arm_table(exp):
    R = load(exp)
    tab = {}
    for a, runs in R.items():
        fb = endpoint(runs)
        m, se = ev.mean_se(list(fb.values()))
        d = [np.mean([x[5] / max(x[1], 1) for x in r["diag"]]) if r["diag"] else np.nan for r in runs.values()]
        tab[a] = dict(final_bin_mean=m, final_bin_se=se, per_seed=fb,
                      auc_bins_mean=float(np.mean([r["summary"]["auc_bins"] for r in runs.values()])),
                      n_episodes_mean=float(np.mean([r["summary"]["n_episodes"] for r in runs.values()])),
                      overshoot_rate_mean=float(np.nanmean(d)) if "e3" in exp or "e6" in exp else None,
                      curve=ev.curve([np.asarray(r["episodes"]) for r in runs.values()],
                                     runs[next(iter(runs))]["config"]["total_steps"],
                                     runs[next(iter(runs))]["config"]["bin_stride"])[1].round(3).tolist())
    return R, tab


def paired(R, a, b, one_sided_greater=True):
    s = sorted(set(R[a]) & set(R[b]))
    x, y = [R[a][i]["summary"]["final_bin"] for i in s], [R[b][i]["summary"]["final_bin"] for i in s]
    if len(s) < 2:
        return None
    pc = ev.paired_compare(x, y)
    if one_sided_greater:
        pc["t_p_one_sided"] = pc["t_p"] / 2 if pc["t"] > 0 else 1 - pc["t_p"] / 2
    return pc


for exp in sorted(os.listdir(root)) if os.path.isdir(root) else []:
    if not os.path.isdir(os.path.join(root, exp)) or exp in ("ppo",):
        continue
    if exp.startswith("e3") or exp.startswith("e6"):
        R, tab = arm_table(exp)
        rep[exp] = dict(arms=tab)
        ref = "stream_q" if exp.startswith("e3") else "stream_ac"
        rep[exp]["paired_vs_" + ref] = {a: paired(R, ref, a) for a in R if a != ref}
        if exp.startswith("e6") and "stream_ac0" in R:
            rep[exp]["paired_stream_ac0_vs_classic_ac0"] = paired(R, "stream_ac0", "classic_ac0")
    elif exp.startswith("e8"):
        R = load(exp)
        rep[exp] = {a: {k: ev.mean_se([r["summary"][k] for r in runs.values()])
                        for k in ("mse_first", "mse_last", "mse_last_excl_tail", "corr_last", "corr_last_excl_tail", "n_active_frac", "overshoot_frac")}
                    for a, runs in R.items()}
        s = sorted(set(R.get("stream_td", {})) & set(R.get("classic_td", {})))
        if len(s) > 1:
            rep[exp]["paired_mse_last_classic_minus_stream"] = ev.paired_compare(
                [R["classic_td"][i]["summary"]["mse_last"] for i in s], [R["stream_td"][i]["summary"]["mse_last"] for i in s])

# official reference curve (stream_q_minatar.py on MinAtar 1.0.15) at the same budget
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official"))  # authors' clone + venv (scripts/official/setup.sh)
refdir = os.path.join(OFFICIAL_ROOT, "runs/q_breakout_500k")
pk = sorted(glob.glob(os.path.join(refdir, "data_stream_q_*", "**", "seed_*.pkl"), recursive=True))
if pk:
    import pickle
    runs = []
    for p in pk:
        rets, steps, _ = pickle.load(open(p, "rb"))
        runs.append(np.stack([steps, rets, np.zeros(len(rets))], 1))
    T = 500_000
    fbs = [ev.run_bins(e, T, 50_000)[-1] for e in runs]
    rep["official_reference_breakout_500k"] = dict(n_seeds=len(runs), final_bin_mean=float(np.mean(fbs)),
                                                    final_bin_se=float(np.std(fbs, ddof=1) / np.sqrt(len(fbs))) if len(fbs) > 1 else None,
                                                    per_seed=[float(x) for x in fbs],
                                                    curve=ev.curve(runs, T, 50_000)[1].round(3).tolist())
# official ETT reference
tdref = sorted(glob.glob(os.path.join(OFFICIAL_ROOT, "runs/td_ett/ref_seed*.npz")))
if tdref:
    rs = []
    for p in tdref:
        z = np.load(p)
        G = ev.discounted_returns(z["cum"], 0.99)
        rs.append(ev.ett_summary(z["pred"], G, z["cum"]))
    rep["official_reference_ett"] = {k: ev.mean_se([r[k] for r in rs]) for k in ("mse_first", "mse_last", "mse_last_excl_tail", "corr_last", "corr_last_excl_tail")}
    rep["official_reference_ett"]["n_seeds"] = len(rs)
# PPO (papers/ppo_implementation_details code) on the same proxy, outputs under runs/ppo/
pp = sorted(glob.glob(os.path.join(REPO, "runs", "ppo", tag, "brax-hopper", "default", "seed_*", "metrics.json")))
if pp:
    fbs = {}
    for p in pp:
        r = json.load(open(p))
        e = np.asarray(r["episodes"])                    # global_step, return_sum, count, length_sum
        e3 = np.stack([e[:, 0], e[:, 1] / np.maximum(e[:, 2], 1), e[:, 3]], 1)
        T = r["config"]["total_timesteps"]
        fbs[r["seed"]] = float(ev.run_bins(e3, T, 50_000)[-1])
    m, se = ev.mean_se(list(fbs.values()))
    rep["ppo_hopper"] = dict(final_bin_mean=m, final_bin_se=se, per_seed=fbs)
    e6 = [k for k in rep if k.startswith("e6")]
    if e6:
        Rr = load(e6[0])
        s = sorted(set(Rr.get("stream_ac", {})) & set(fbs))
        if len(s) > 1:
            rep["stream_ac_vs_ppo"] = ev.paired_compare([Rr["stream_ac"][i]["summary"]["final_bin"] for i in s],
                                                        [fbs[i] for i in s])
with open(os.path.join(root, "report.json"), "w") as f:
    json.dump(rep, f, indent=1, default=float)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: (vv if not isinstance(vv, dict) or "curve" not in vv else
                  {x: y for x, y in vv.items() if x != "curve"}) for kk, vv in v.items()}) for k, v in rep.items()},
                 indent=1, default=float)[:6000])

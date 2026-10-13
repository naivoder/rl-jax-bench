"""Recheck: the paper's own E1 runs, reduced with OUR metric.

Input: the 36 classic-control histories of the paper's public W&B project vwxyzjn/ppo-details (downloaded
anonymously by runs/recheck/measurement/wb_fetch.py into runs/recheck/measurement/wb_hist/, one row per logged
training episode: global_step, charts/episodic_return) and the run listing wb_runs.json (configs, run ids).
Output: data/paper_raw.json with per-seed final window / AUC / steps-to-threshold from src/eval.summarize
(total 500k, the same call as for ours and the authors' code), for the blue curves (ppo = separate, ppo_shared = shared)
and the red ones (baselines-ppo2-mlp-seperate-networks = separate, baselines-ppo2-mlp = shared; N noted)."""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src.eval import summarize  # noqa: E402

M = os.path.join(REPO, "runs", "recheck", "measurement")
EXP = {"ppo": "separate", "ppo_shared": "shared", "baselines-ppo2-mlp-seperate-networks": "baselines_separate",
       "baselines-ppo2-mlp": "baselines_shared"}
runs = {r["displayName"]: r for r in json.load(open(os.path.join(M, "wb_runs.json")))}
out = {}
for f in sorted(glob.glob(os.path.join(M, "wb_hist", "*.json"))):
    dn = os.path.basename(f)[:-5]
    env, exp, seed, _ = dn.split("__")
    cfg = json.loads(runs[dn]["config"])
    get = lambda k: (cfg.get(k) or {}).get("value") if isinstance(cfg.get(k), dict) else cfg.get(k)  # noqa: E731
    rows = json.load(open(f))
    e = np.array([[r["global_step"], r["charts/episodic_return"], 1, r.get("charts/episodic_length", 0) or 0]
                  for r in rows if r.get("charts/episodic_return") is not None], float)
    s = summarize(e, 500000, env)
    d = out.setdefault(f"{env}|{EXP[exp]}", dict(env=env, arm=EXP[exp], wandb_exp=exp, seeds={}))
    d["seeds"][int(seed)] = dict(run_id=runs[dn]["name"], display=dn, n_episodes=len(e),
                                 num_env=get("num_envs") or get("num_env"), final_window=s["final_window"],
                                 auc=s["auc"], steps_to_threshold=s["steps_to_threshold"])
for d in out.values():
    fw = np.array([v["final_window"] for v in d["seeds"].values()])
    d["final_window_mean"], d["final_window_se"] = float(fw.mean()), float(fw.std(ddof=1) / np.sqrt(len(fw)))
    d["final_window_per_seed"] = [float(x) for x in fw]
json.dump(dict(source="wandb.ai/vwxyzjn/ppo-details (public; anonymous GraphQL), reduced with src/eval.summarize",
               rows=out), open(os.path.join(REPO, "data", "paper_raw.json"), "w"), indent=1, default=float)
for k, d in sorted(out.items()):
    print(f"{k:40s} {d['final_window_mean']:8.1f} ± {d['final_window_se']:5.1f}  per seed "
          f"{[round(x, 1) for x in d['final_window_per_seed']]}  ids {[v['run_id'] for v in d['seeds'].values()]} "
          f"N {[v['num_env'] for v in d['seeds'].values()]}")

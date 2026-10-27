"""Small-scale pass criteria Small-1..Small-5 (declared before the first interaction number was seen).
usage: small_check.py runs/small/Breakout Breakout"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval import load, windows

out, game = sys.argv[1], sys.argv[2]
runs, meta = load(out, game)
n = meta["n_eval_points"]; mu = meta["guide"]["true_mean"]
arms = ["JSRL", "JSRL-Random", "PDD", "DQfD", "BCinit-FT", "IMIT"]
chk = {}
seeds = sorted(runs["PDD"])
# Small-1
ok1 = all(a in runs and len(runs[a]) == len(meta["seeds"]) and all(len(runs[a][s]["ev_alone"]) == n + 1 for s in runs[a]) for a in arms)
allfin = all(np.isfinite(np.asarray(runs[a][s]["ev_alone"])).all() for a in arms for s in runs[a])
chk["Small-1"] = dict(ok=bool(ok1 and allfin), arms_complete=ok1, finite=allfin, devices=meta["devices"], n_seeds=len(seeds))
# Small-2
r0 = {a: np.array([runs[a][s]["ev_alone"][0] for s in seeds]) for a in ["IMIT", "BCinit-FT", "DQfD"]}
eq = bool(np.array_equal(r0["IMIT"], r0["BCinit-FT"]))
chk["Small-2"] = dict(ok=bool(eq and abs(r0["IMIT"].mean() - mu) <= 0.2 * mu and abs(r0["DQfD"].mean() - mu) <= 0.3 * mu), imit_eq_bcinit=eq,
                   imit_R0=float(r0["IMIT"].mean()), bcinit_R0=float(r0["BCinit-FT"].mean()), dqfd_R0=float(r0["DQfD"].mean()), mu_g=mu)
# Small-3
thr0 = np.array([runs["JSRL"][s]["ev_thr"][1] for s in seeds])         # combined policy at stage 0 after the first chunk
chunk = meta["steps"] // n
gs1 = np.array([runs["JSRL"][s]["ev_gsteps"][1] / chunk for s in seeds])
share = {a: float(np.mean([runs[a][s]["guide_steps"] / meta["steps"] for s in seeds])) for a in ["JSRL", "JSRL-Random", "PDD"] if "guide_steps" in runs[a][seeds[0]]}
adv = int(sum(max(runs["JSRL"][s]["ev_stage"]) > 0 for s in seeds))
chk["Small-3"] = dict(ok=bool(abs(thr0.mean() - mu) <= 0.2 * mu and gs1.mean() >= 0.9 and share["PDD"] == 0 and 0 < share["JSRL-Random"] < gs1.mean() and adv >= 0.8 * len(seeds)),
                   combined_stage0_thr_mean=float(thr0.mean()), first_chunk_guide_share=float(gs1.mean()), guide_share_run=share, seeds_advanced=adv)
# Small-4
up = int(sum(np.mean(runs["PDD"][s]["ev_alone"][-3:]) > runs["PDD"][s]["ev_alone"][0] for s in seeds))
chk["Small-4"] = dict(ok=up >= 0.8 * len(seeds), pdd_final_gt_R0_seeds=up)


def first_reach(curve, level):
    idx = np.nonzero(np.asarray(curve) >= level)[0]
    return int(idx[0]) * chunk if len(idx) else None


cj = np.mean([runs["JSRL"][s]["ev_alone"] for s in seeds], 0); cp = np.mean([runs["PDD"][s]["ev_alone"] for s in seeds], 0)
tj, tp = first_reach(cj, 0.5 * mu), first_reach(cp, 0.5 * mu)
ratio = (tj / tp) if (tj is not None and tp) else None
ok5 = (ratio is not None and 0.04 <= ratio <= 4) or (tj is not None and tp is None)
chk["Small-5"] = dict(ok=bool(ok5), steps_jsrl_to_half_mu=tj, steps_pdd_to_half_mu=tp, ratio=ratio, paper_ratio=0.39)
chk["ALL"] = bool(all(v["ok"] for v in chk.values() if isinstance(v, dict)))
print(json.dumps(chk, indent=1, default=float))
json.dump(chk, open(os.path.join(out, "small_check.json"), "w"), indent=1, default=float)

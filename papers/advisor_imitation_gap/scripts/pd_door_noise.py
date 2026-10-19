"""Check: on the PD draws where BC->PPO (or BC) stays at the imitation plateau in BOTH implementations,
P(chose the good door) must be 1/3 in expectation (the student never observes the good door). Compares ours (a
re-evaluated tag) with the authors' TSV at the same draws. Usage: pd_door_noise.py <tag> [arm=bc_then_ppo]"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval import official_tsv  # noqa: E402

tag = sys.argv[1]
arms = sys.argv[2:] or ["bc_then_ppo", "bc", "dagger"]
TSVN = {"bc_then_ppo": "bc_then_ppo", "bc": "bc", "dagger": "dagger"}
th = {(r["exp_type"], int(r["seed"])): r for r in official_tsv("PoisonedDoors")}
out = {}
for arm in arms:
    S = sorted([r for r in json.load(open(f"runs/{tag}/summary.json"))["runs"] if r["arm"] == arm and r.get("draw") is not None],
               key=lambda r: r["draw"])
    rows = []
    for r in S:
        t = th[(TSVN[arm], r["draw"])]
        last = lambda v: float(v[-1]) if isinstance(v, (list, tuple)) else float(v)
        plateau = r["final"]["chose_locked"] < 0.01 and last(t["chose_door_0"]) < 0.01   # never commits to d_locked
        rows.append(dict(draw=r["draw"], plateau=bool(plateau), ours_good=r["final"]["chose_good"],
                         theirs_good=last(t["chose_good_door"]), ours_locked=r["final"]["chose_locked"],
                         theirs_locked=last(t["chose_door_0"]), ours_ret=r["final"]["ret"], theirs_ret=last(t["reward"]),
                         ours_none=r["final"].get("chose_none")))
    p = [x for x in rows if x["plateau"]]
    og, tg = np.array([x["ours_good"] for x in p]), np.array([x["theirs_good"] for x in p])
    n_ep = 1000 * len(p)
    se = np.sqrt((1 / 3) * (2 / 3) / n_ep)
    out[arm] = dict(n_plateau_draws=len(p), ours_p_good=float(og.mean()), theirs_p_good=float(tg.mean()),
                    z_ours=float((og.mean() - 1 / 3) / se), z_theirs_if_indep=float((tg.mean() - 1 / 3) / se),
                    theirs_above_third=int((tg > 1 / 3).sum()), ours_above_third=int((og > 1 / 3).sum()),
                    escaping=[x for x in rows if not x["plateau"]], rows=rows)
    print(f"{arm:12s} plateau draws {len(p)}: P(good) ours {og.mean():.4f} (z {out[arm]['z_ours']:+.1f}, above 1/3 on "
          f"{(og > 1/3).sum()}) | theirs {tg.mean():.4f} (z {out[arm]['z_theirs_if_indep']:+.1f} if independent, above 1/3 on "
          f"{(tg > 1/3).sum()}); escaping draws: " +
          ", ".join(f"d{x['draw']} ours {x['ours_ret']:+.3f} theirs {x['theirs_ret']:+.3f}" for x in rows if not x["plateau"]))
json.dump(out, open(f"runs/{tag}/door_noise.json", "w"), indent=1)

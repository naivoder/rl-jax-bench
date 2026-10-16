# Paper-side reference for the Table-2 BASE configuration: rows of the authors' study data (final_time, where every
# non-sampled choice is at the Table-2 base incl. value clipping 0.2 and the clip transform) whose sampled choices are
# also at base (discount 0.99, frame_skip 1, lr 3e-4; handle_abandoned either value -- a no-op in the released env,
#). Score = last entry of _eval_ (per-row, 3-seed config score). Prints n, median, mean, min, max per env.
import pickle
import sys

import numpy as np

import os  # noqa: E402
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' checkout, see README
D = pickle.load(open(os.path.join(OFFICIAL_ROOT, "data", "mujoco123.pkl"), "rb"))
name = {m["title"].split(": ")[1].split(" ")[0]: k for k, (df, m) in D.items()}
df, _ = D[name["final_time"]]
P = "_gin.study_design.choice_value_"
sel = df[(df[P + "discount_factor"].astype(str) == "'0.99'") & (df[P + "frame_skip"].astype(str) == "'1'")
         & (df[P + "learning_rate"].astype(str) == "'0.0003'")]
out = {}
for env in ["Hopper-v1", "Walker2d-v1", "HalfCheetah-v1"]:
    e = sel[sel["env_name"] == env]
    s = np.array([r[max(r)] for r in e["_eval_"]])
    curve = np.array([[r[k] for k in sorted(r)] for r in e["_eval_original_avg"]])
    print(f"[INFO] {env} n_rows={len(s)} median={np.median(s):.0f} mean={s.mean():.0f} se={s.std(ddof=1)/np.sqrt(len(s)):.0f} "
          f"min={s.min():.0f} max={s.max():.0f}  mean curve(0..1M)={np.round(curve.mean(0)).astype(int).tolist()}")
    for h in ["'False'", "'True'"]:
        sh = np.array([r[max(r)] for r in e[e[P + "handle_abandoned_episodes_properly"].astype(str) == h]["_eval_"]])
        print(f"        handle_abandoned={h}: n={len(sh)} mean={sh.mean():.0f}")

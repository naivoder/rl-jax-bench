"""Recheck: pack the branch-sampled env states (runs/recheck/environment/branch_states_ours.py: 2000 Acrobot
states near the terminal surface, 2000 MountainCar states near the goal, 2000 Pendulum states at |thdot| 6.5-8 so the
speed clip binds) with the authors'-stack outputs on them (branch_states_oracle.py: gymnasium 0.29.1 and gym 0.15.7,
unwrapped env, state set, one step) into tests/fixtures/official/envs_branch.npz."""
import os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = os.path.join(REPO, "runs", "recheck", "environment")
o = np.load(os.path.join(E, "ours_branch.npz"))
out = {f"in/{k}_{x}": o[f"{k}_{x}"] for k in ("acro", "mc", "pend") for x in ("S", "A")}
for tag in ("gymnasium", "gym0157"):
    z = np.load(os.path.join(E, f"oracle_branch_{tag}.npz"))
    out.update({f"oracle_{tag}/{k}": z[k] for k in z.files})
np.savez_compressed(os.path.join(REPO, "tests", "fixtures", "official", "envs_branch.npz"), **out)
print(sorted(out))

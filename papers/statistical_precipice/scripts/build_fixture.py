"""Extract a small committed fixture from data/e0/oracle.npz (rliable outputs; numbers only, never its code) for
tests/test_oracle_parity.py: inputs + points for 40 inputs, exact-replay indices/results for 8 small inputs,
task/independent replays for 2, the Colab studentized rows. Run with python."""
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO, "external", "data"))  # large inputs, see README
z = np.load(os.path.join(DATA_ROOT, "e0", "inputs.npz"))
o = np.load(os.path.join(DATA_ROOT, "e0", "oracle.npz"))
out = {}
pts = list(range(0, 200, 5))
for i in pts:
    for k in ("x", "y"):
        out[f"{k}{i:03d}"] = z[f"{k}{i:03d}"]
    for k in ("agg", "trim", "taus", "prof", "avgprof", "poi"):
        out[f"pt_{i:03d}_{k}"] = o[f"pt_{i:03d}_{k}"]
small = [i for i in pts if z[f"x{i:03d}"].shape[0] <= 5][:8]
for i in small:
    out[f"rp_{i:03d}_idx"], out[f"rp_{i:03d}_res"] = o[f"rp_{i:03d}_idx"], o[f"rp_{i:03d}_res"]
for i in (0, 10):
    for k in ("x", "y"):
        out[f"{k}{i:03d}"] = z[f"{k}{i:03d}"]
    for k in ("idx", "tidx", "res"):
        out[f"rptask_{i:03d}_{k}"] = o[f"rptask_{i:03d}_{k}"]
    for k in ("ix", "iy", "res"):
        out[f"rppoi_{i:03d}_{k}"] = o[f"rppoi_{i:03d}_{k}"]
for k in o.files:
    if k.startswith("stud_"):
        out[k] = o[k]
out["meta_points"], out["meta_small"] = np.array(pts), np.array(small)
p = os.path.join(REPO, "tests", "fixtures", "oracle_small.npz")
np.savez_compressed(p, **out)
print(p, os.path.getsize(p))

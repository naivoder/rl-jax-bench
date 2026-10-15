import glob, json, os, sys
import numpy as np
R = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs")
exp = sys.argv[1]
keys = sys.argv[2].split(",")
for p in sorted(glob.glob(f"{R}/review_checks/{exp}/**/metrics.json", recursive=True)):
    q = p.replace("/review_checks/", "/full/")
    a, b = json.load(open(p)), json.load(open(q))
    out = []
    for k in keys:
        x, y = a, b
        for kk in k.split("."):
            x, y = x[kk], y[kk]
        x, y = np.asarray(x, float), np.asarray(y, float)
        out.append(f"{k}: {'BIT-EQUAL' if x.shape == y.shape and np.array_equal(x, y) else 'max|d|=%.3g' % (np.abs(x - y).max() if x.shape == y.shape else -1)}")
    print(os.path.relpath(p, f"{R}/review_checks"), " | ".join(out), "| cfg", a["config_hash"] == b["config_hash"])

import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import eval as E
for exp in ("review_checks/e3_qoc_lambda1", "full/e3_queue_of_cars"):
    for a in ("ppo", "ta_ppo"):
        v = []
        for p in sorted(glob.glob(f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/runs/{exp}/{a}/seed_*/metrics.json")):
            d = json.load(open(p)); pd = np.asarray(d["per_update"]["p_danger"])
            v.append([E.qoc_exact_value(pd[u][:15], 0.99) for u in (0, 20, 50, 100, len(pd) - 1)])
        v = np.array(v)
        print(exp, a, "exact value at updates 1/21/51/101/final (mean over seeds):", v.mean(0).round(4), "final SE", (v[:, -1].std(ddof=1) / np.sqrt(len(v))).round(4), "n", len(v))

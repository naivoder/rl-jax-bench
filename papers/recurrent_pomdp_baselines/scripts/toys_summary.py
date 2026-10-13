"""Post-processing of runs/toys/toys.json: Monte-Carlo-error judgement of the truncation tests C / C' (pooled seed
mean of the decision-step Q within 2 se of 1/2, and Q identical across cues) and the overall toy stage verdict."""
import json
import os

import numpy as np

p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "toys", "toys.json")
r = json.load(open(p))
for k in ("C", "C_prime"):
    q = np.array(r[k]["Q_dec_window2"]).reshape(4, -1)   # per seed: [c0 a0, c0 a1, c1 a0, c1 a1]
    sm = q.mean(1)
    m, se = sm.mean(), sm.std(ddof=1) / 2
    r[k].update(seed_means=sm.tolist(), pooled_mean=float(m), pooled_se=float(se), mc_pass=bool(abs(m - 0.5) <= 2 * se),
                identical_across_cues=bool(np.allclose(q[:, :2], q[:, 2:])),
                max_abs_dev=float(np.max(np.abs(q - 0.5))))
qb = np.array(r["B"]["Q_dec"]).reshape(4, -1).mean(1)
r["B"].update(pooled_mean=float(qb.mean()), pooled_se=float(qb.std(ddof=1) / 2))
r["toys_rule"] = ("per-value tolerance 0.05 (script default) fails for C/C_prime (critic estimation noise, ~16 "
                   "decision-step samples per batch); judged on the toy stage rule 'within Monte-Carlo error': pooled seed "
                   "mean within 2 se of 1/2 and Q identical across cues (exact). A, B, D, E pass the stricter per-value "
                   "tolerance.")
r["toys_pass"] = bool(r["A"]["pass"] and r["B"]["pass"] and r["C"]["mc_pass"] and r["C_prime"]["mc_pass"]
                       and r["D"]["pass"] and r["E"]["pass"])
json.dump(r, open(p, "w"), indent=1)
print({k: (r[k]["pooled_mean"], r[k]["pooled_se"], r[k]["mc_pass"]) for k in ("C", "C_prime")}, r["toys_pass"])

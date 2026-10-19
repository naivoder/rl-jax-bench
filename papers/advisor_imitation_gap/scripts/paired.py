"""Paired (by seed) one-sided comparisons of a focal arm against baselines from runs/<tag>/report.json, Holm across
the family. Usage: paired.py <tag> <cell> <focal_arm> <metric> [lower] ; prints and writes runs/<tag>/paired_<focal>.json"""
import json
import os
import sys

from rl_jax_bench.stats import holm, paired_compare  # noqa: E402

tag, cell, focal, metric = sys.argv[1:5]
lower = len(sys.argv) > 5 and sys.argv[5] == "lower"
rep = json.load(open(os.path.join("runs", tag, "report.json")))
f = rep[f"{cell}/{focal}"]
res, pv = {}, {}
for k, row in rep.items():
    c, arm = k.split("/")
    if c != cell or arm == focal:
        continue
    assert row["seeds"] == f["seeds"], (row["seeds"], f["seeds"])
    a, b = f[metric]["per_seed"], row[metric]["per_seed"]
    if lower:
        a, b = [-x for x in a], [-x for x in b]
    pc = paired_compare(a, b)
    p1 = pc["t_p"] / 2 if pc["diff_mean"] > 0 else 1 - pc["t_p"] / 2
    if pc["diff_se"] == 0:
        p1 = 0.0 if pc["diff_mean"] > 0 else 1.0
    res[arm] = dict(diff_mean=pc["diff_mean"] * (-1 if lower else 1), diff_se=pc["diff_se"], boot95=pc["boot95"],
                    t_p_one_sided=p1, wins=pc["wins"], n=pc["n"])
    pv[arm] = p1
H = holm(pv)
for arm in res:
    res[arm]["p_holm"] = H[arm]["p_holm"]
    print(f"{focal} vs {arm:16s} diff={res[arm]['diff_mean']:+.3f}+-{res[arm]['diff_se']:.3f} "
          f"boot95(oriented)={[round(x, 3) for x in res[arm]['boot95']]} p1={res[arm]['t_p_one_sided']:.2e} "
          f"holm={res[arm]['p_holm']:.2e} wins={res[arm]['wins']}/{res[arm]['n']}")
json.dump(res, open(os.path.join("runs", tag, f"paired_{cell}_{focal}.json"), "w"), indent=1)

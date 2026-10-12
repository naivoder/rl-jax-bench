"""Compare the small-scale JAX runs with the official-code reference runs (official_ref_run.py, gymnasium MuJoCo v4
proxies) at the same iterations: training reward and the held-out / train value MREs. Writes
runs/small/official_compare.json and prints a table."""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official")), "ref")
from rl_jax_bench.stats import mean_se  # noqa: E402

out = {}
for tag in ("hopper_ppo", "hopper_trpo", "walker2d_ppo", "walker2d_trpo"):
    O = [json.load(open(f))["history"] for f in sorted(glob.glob(f"{REF}/{tag}_s*.json"))]
    J = [json.load(open(f))["history"] for f in sorted(glob.glob(f"{REPO}/runs/small/{tag}/seed_*/metrics.json"))]
    r = {"n_official": len(O), "n_jax": len(J)}
    ro = np.array([[h["mean_reward"] for h in hh] for hh in O])
    rj = np.array([[h["mean_reward"] for h in hh] for hh in J])
    r["reward_it0"] = {"official": mean_se(ro[:, 0]), "jax": mean_se(rj[:, 0])}
    r["reward_last9"] = {"official": mean_se(ro[:, 40:49].mean(1)), "jax": mean_se(rj[:, 40:49].mean(1))}
    for split in ("heldout", "train"):
        for q in ("returns_mre", "gae_mre_code", "gae_mre_true"):
            row = {}
            for it in (0, 10, 20, 30, 40):
                o = [hh[it][split][q] for hh in O]
                j = [hh[it][f"{split}_{q}"] for hh in J]
                row[it] = {"official_median": float(np.median(o)), "jax_median": float(np.median(j)),
                           "official": o, "jax": j}
            r[f"{split}_{q}"] = row
    out[tag] = r
    print(f"\n== {tag}  reward it0 off {r['reward_it0']['official'][0]:.1f} jax {r['reward_it0']['jax'][0]:.1f} | "
          f"its40-48 off {r['reward_last9']['official'][0]:.0f}+-{r['reward_last9']['official'][1]:.0f} "
          f"jax {r['reward_last9']['jax'][0]:.0f}+-{r['reward_last9']['jax'][1]:.0f}")
    for k in [k for k in r if k.startswith(("heldout", "train"))]:
        print(f"  {k:24s}", "  ".join(f"it{it}: {v['official_median']:.4g}/{v['jax_median']:.4g}" for it, v in r[k].items()))
json.dump(out, open(os.path.join(REPO, "runs", "small", "official_compare.json"), "w"), indent=1)

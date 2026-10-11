"""Merge runs of one config split over stream blocks (separate processes / tags) into one run directory.

    uv run python -m scripts.merge_parts full_scr_d12 full_scr_d12_p0 full_scr_d12_p1 full_scr_d12_p2

Every part must hold the same config apart from its explicit `seeds` list (checked); the per-seed metrics are copied
with `experiment` set to the merged tag, and summary.json is rebuilt (seeds, arms, whole-run mean over all seeds).
"""
import json
import os
import sys

import numpy as np
import yaml

from src.eval import RUNS, load


def main(out_tag, parts):
    cfgs = [yaml.safe_load(open(os.path.join(RUNS, p, "config.yaml"))) for p in parts]
    strip = lambda c: {k: v for k, v in c.items() if k not in ("seeds", "tag")}
    if any(strip(c) != strip(cfgs[0]) for c in cfgs[1:]):
        raise ValueError("parts differ in more than seeds/tag")
    out = os.path.join(RUNS, out_tag)
    merged = {}
    for p in parts:
        for s, d in load(p).items():
            if s in merged:
                raise ValueError(f"seed {s} in two parts")
            d = dict(d, experiment=out_tag, merged_from=p)
            merged[s] = d
            os.makedirs(os.path.join(out, f"seed_{s}"), exist_ok=True)
            json.dump(d, open(os.path.join(out, f"seed_{s}", "metrics.json"), "w"))
    seeds = sorted(merged)
    arms = list(merged[seeds[0]]["arms"])
    cfg = dict(cfgs[0], seeds=seeds, tag=out_tag, merged_from=parts)
    yaml.safe_dump(cfg, open(os.path.join(out, "config.yaml"), "w"), sort_keys=False)
    summ = {"tag": out_tag, "seeds": seeds, "arms": arms, "merged_from": parts,
            "mse_whole_run_mean": {a: float(np.mean([merged[s]["arms"][a]["mse_whole_run"] for s in seeds])) for a in arms}}
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(f"merged {len(seeds)} seeds x {len(arms)} arms -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

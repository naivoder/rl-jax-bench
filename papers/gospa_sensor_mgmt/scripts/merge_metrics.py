"""Fold the per-(cfg, algorithm) shards of every seed into the required runs/<tag>/seed_<m>/metrics.json
({cfg: {alg: {clutter: {...}}}}); the shards are kept.  Usage: python scripts/merge_metrics.py <runs_root>/<tag>"""
import glob, json, os, sys

tag_dir = sys.argv[1]
for sd in sorted(glob.glob(os.path.join(tag_dir, "seed_*"))):
    merged = {}
    for sh in sorted(glob.glob(os.path.join(sd, "metrics__*.json"))):
        base = os.path.basename(sh)[len("metrics__"):-len(".json")]
        cfg, alg = base.split("__", 1)
        with open(sh) as f:
            merged.setdefault(cfg, {})[alg] = json.load(f)
    tmp = os.path.join(sd, "metrics.json.tmp")
    with open(tmp, "w") as f:
        json.dump(merged, f)
    os.replace(tmp, os.path.join(sd, "metrics.json"))
    print("merged", sd, {c: len(a) for c, a in merged.items()})

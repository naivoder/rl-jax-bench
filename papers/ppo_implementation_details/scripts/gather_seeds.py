"""Gather per-seed summaries of every run under runs/<tag>/<env>/<arm>/seed_<n>/metrics.json into
runs/seed_<n>/metrics.json, keyed "<tag>/<env>/<arm>" (summary + config + timing; the full per-episode records stay
in the per-run files). Usage: python -m scripts.gather_seeds small small_lstm small_lstm_v2"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out: dict[int, dict] = {}
for tag in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(REPO, "runs", tag, "*", "*", "seed_*", "metrics.json"))):
        env, arm, sd = f.split(os.sep)[-4:-1]
        d = json.load(open(f))
        s = int(sd[5:])
        out.setdefault(s, {})[f"{tag}/{env}/{arm}"] = dict(summary=d["summary"], git=d["git"], timing=d["timing"],
                                                           config=d["config"], source=os.path.relpath(f, REPO))
for s, d in out.items():
    p = os.path.join(REPO, "runs", f"seed_{s}", "metrics.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(d)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(old, open(p, "w"), indent=1)
    print(p, len(old))

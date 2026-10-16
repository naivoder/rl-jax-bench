"""Rebuild per-lane metrics.json at the last orbax checkpoint step from its host-side record (episodes + diagnostics).
Used for the full-scale E3 run, which was killed at its wall-clock cap WHILE writing the final metrics files (all 60 primary
lanes were written at 1M; dqn1 seeds 7-10 and every stream_q_eps05 lane were not): the 8e5 checkpoint holds every
lane's episodes up to 8e5, so the secondary comparisons are made on this common 8e5 view (all four arms).
Usage: python scripts/ckpt_view.py <run_dir> <out_exp_dir>"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src import eval as ev  # noqa: E402

run_dir, out_dir = sys.argv[1], sys.argv[2]
ck = os.path.join(run_dir, "ckpt")
host = sorted(f for f in os.listdir(ck) if f.endswith("_host.npz"))[-1]
T = int(host[1:11])
z = np.load(os.path.join(ck, host))
ep, diag = z["ep"], z["diag"]
ref = json.load(open(os.path.join(run_dir, "stream_q", "seed_1", "metrics.json")))
cfg = ref["config"]
arms, seeds = cfg["arms"], cfg["seeds"]
arm_seeds = {a: (cfg.get("arm_seeds") or {}).get(a, seeds) for a in arms}
lanes = [(a, s) for a in arms for s in arm_seeds[a]]
for i, (a, s) in enumerate(lanes):
    e = ep[ep[:, 0] == i][:, 1:]
    d = diag[diag[:, 0] == i][:, 1:]
    c2 = dict(cfg, total_steps=T)
    rec = dict(exp=ref["exp"] + f"@{T}", env=ref["env"], arm=a, seed=s, git=ref["git"], config=c2,
               note=f"view at the {T}-step checkpoint (host record) of runs/full/{os.path.basename(run_dir)}",
               diverged_by_step=None, summary=ev.summarize_run(e, T, 50_000),
               episodes_cols=ref["episodes_cols"], episodes=e.tolist(), diag_cols=ref["diag_cols"], diag=d.tolist())
    p = os.path.join(out_dir, a, f"seed_{s}", "metrics.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(rec, open(p, "w"))
print("[INFO] wrote", len(lanes), "lanes at step", T)

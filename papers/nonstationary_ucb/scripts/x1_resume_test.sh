#!/bin/sh
# README: kill a T = 3e5 X1 run after its 2nd orbax checkpoint of the SW-UCB arm, restart it, and compare
# the final regrets with the uninterrupted run in runs/x1/a/T300000_U4 (written by scripts/full.sh x1 a).
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
rm -rf runs/x1_resume runs/x1_resume_ckpt
$PY -m src.sweep x1 --grid a --points 5 --seeds 0,1,2,3,4 --reps 1000 --out runs/x1_resume --ckpt runs/x1_resume_ckpt > runs/logs/x1_resume_1.log 2>&1 &
PID=$!
until [ -d runs/x1_resume_ckpt/a/T300000_U4/seeds_0-1-2-3-4/swucb/200001 ]; do sleep 1; done
sleep 2; kill -9 $PID; echo "killed $PID after swucb checkpoint 200001"
ls runs/x1_resume_ckpt/a/T300000_U4/seeds_0-1-2-3-4/*/
$PY -m src.sweep x1 --grid a --points 5 --seeds 0,1,2,3,4 --reps 1000 --out runs/x1_resume --ckpt runs/x1_resume_ckpt > runs/logs/x1_resume_2.log 2>&1
grep "resumed" runs/logs/x1_resume_2.log
$PY - <<'PY'
import json, numpy as np
ok = True
for s in range(5):
    a = json.load(open(f"runs/x1/a/T300000_U4/seed_{s}/metrics.json"))["policies"]
    b = json.load(open(f"runs/x1_resume/a/T300000_U4/seed_{s}/metrics.json"))["policies"]
    for p in a:
        same = np.array_equal(a[p]["final_regret"], b[p]["final_regret"]) and np.array_equal(a[p]["final_arm1"], b[p]["final_arm1"])
        ok &= same
print("RESUME_BIT_IDENTICAL", ok)
PY

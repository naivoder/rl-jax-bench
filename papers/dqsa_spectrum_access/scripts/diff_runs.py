"""full-scale reproducibility check : bit-for-bit diff of two runs of the SAME vmapped group.
    python scripts/diff_runs.py <run_dir_A> <run_dir_B>      (each = runs/<out>/<tag>, containing seed_<n>/params.npz + launch.json)
Exit code 0 iff every seed's params.npz is bit-identical AND the vmapped groups, git sha and source hashes recorded at launch agree.
Per-seed outcomes are NOT reproducible across different vmapped groups (batched reductions differ in the last bits and the dynamics are
chaotic), so a report seed must be re-flown inside its own group; use this script on the report run and its re-flight."""
import glob
import json
import os
import sys

import numpy as np

A, B = sys.argv[1], sys.argv[2]
la, lb = json.load(open(os.path.join(A, "launch.json"))), json.load(open(os.path.join(B, "launch.json")))
ok = True
for k in ("vmapped_group_seeds", "algo_sha", "env_sha", "eval_sha", "git_commit"):
    if la[k] != lb[k]:
        print(f"[DIFF] launch.{k}: {la[k]} vs {lb[k]}")
        ok = False
for pa in sorted(glob.glob(os.path.join(A, "seed_*", "params.npz"))):
    pb = pa.replace(A.rstrip("/"), B.rstrip("/"), 1)
    a, b = np.load(pa), np.load(pb)
    worst = max((float(np.abs(a[k] - b[k]).max()) for k in a.files), default=0.0)
    same = all(np.array_equal(a[k], b[k]) for k in a.files)
    print(f"[{'SAME' if same else 'DIFF'}] {os.path.basename(os.path.dirname(pa))} max|d|={worst:.3e}")
    ok &= same
print("BIT-IDENTICAL" if ok else "NOT IDENTICAL")
sys.exit(0 if ok else 1)

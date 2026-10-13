"""Recheck: the post's own ppo.py / ppo_shared.py @fbef824 (.venv_post, gym 0.21,
unmodified, scripts/official/run_post.py), CartPole-v1 + Acrobot-v1, seeds 6-10, 500k, CPU, 5 workers. With authors'-code check seeds
1-5 (oracle_runs/reference_500k) this gives n = 10 per row. Hard line: jobs not started 40 min after launch are skipped."""
import concurrent.futures as cf
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official"))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "runs", "recheck", "fix", "post_s6_10")
os.makedirs(OUT, exist_ok=True)
PY = f"{H}/.venv_post/bin/python"
jobs = [[PY, "-I", f"{REPO}/scripts/official/run_post.py", f"{H}/ppo-implementation-details/{sc}", env, str(s), "500000", OUT]
        for env in ("CartPole-v1", "Acrobot-v1") for sc in ("ppo_shared.py", "ppo.py") for s in range(6, 11)]
t0, LINE = time.time(), 40 * 60


def run(cmd):
    if time.time() - t0 > LINE:
        return "SKIPPED (budget line) " + " ".join(cmd[-5:])
    p = subprocess.run(cmd, cwd=H, capture_output=True, text=True)
    return (p.stdout.strip() or p.stderr[-500:]) + f"  [t={time.time() - t0:.0f}s]"


print(f"[INFO] {len(jobs)} jobs, 5 workers, line {LINE}s", flush=True)
with cf.ThreadPoolExecutor(5) as ex:
    for r in ex.map(run, jobs):
        print("[METRIC]", r, flush=True)
print(f"[INFO] done in {time.time() - t0:.0f}s", flush=True)

"""authors'-code check reference runs of the authors' code at the paper's E1 scale (500k steps), CPU, <= 6 workers.
Hard line: wall clock 75 min from launch; jobs not started by then are skipped and reported.
Usage: python3 scripts/official/launch_reference.py <out_dir>"""
import concurrent.futures as cf
import os
import subprocess
import sys
import time

H = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
SD = os.path.dirname(os.path.abspath(__file__))  # this folder (our harness)
out = os.path.abspath(sys.argv[1])
os.makedirs(out, exist_ok=True)
POST, CL, BL = f"{H}/.venv_post/bin/python", f"{H}/.venv/bin/python", f"{H}/.conda_py37/bin/python"
T = "500000"
jobs = []
for env, seeds in (("Acrobot-v1", range(1, 6)), ("CartPole-v1", range(1, 6)), ("MountainCar-v0", range(1, 4))):
    for s in seeds:
        for sc in ("ppo.py", "ppo_shared.py"):
            jobs.append([POST, "-I", f"{SD}/run_post.py", f"{H}/ppo-implementation-details/{sc}", env, str(s), T, out])
for env in ("Acrobot-v1", "CartPole-v1"):
    for s in range(1, 4):
        for mod in ("run_separate_networks", "run"):
            jobs.append([BL, f"{SD}/run_baselines.py", mod, env, str(s), T, out, "4"])
for env in ("Acrobot-v1", "CartPole-v1"):
    for s in range(1, 6):
        jobs.append([CL, "-I", f"{SD}/run_cleanrl.py", f"{H}/cleanrl/cleanrl/ppo.py", env, str(s), T, out])
        jobs.append([CL, "-I", f"{SD}/run_cleanrl.py", f"{H}/harness/cleanrl_ppo_shared.py", env, str(s), T, out])
for s in range(1, 4):
    jobs.append([POST, "-I", f"{SD}/run_post.py", f"{H}/ppo-implementation-details/ppo_continuous_action.py",
                 "Pendulum-v1", str(s), "200000", out])
t0 = time.time()
LINE = 75 * 60


def run(cmd):
    if time.time() - t0 > LINE:
        return "SKIPPED (budget line) " + " ".join(cmd[-5:])
    p = subprocess.run(cmd, cwd=H, capture_output=True, text=True)
    return (p.stdout.strip() or p.stderr[-500:]) + f"  [t={time.time() - t0:.0f}s]"


print(f"[INFO] {len(jobs)} jobs, 6 workers, line {LINE}s", flush=True)
with cf.ThreadPoolExecutor(6) as ex:
    for r in ex.map(run, jobs):
        print("[METRIC]", r, flush=True)
print(f"[INFO] done in {time.time() - t0:.0f}s", flush=True)

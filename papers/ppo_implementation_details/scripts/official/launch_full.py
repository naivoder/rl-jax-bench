"""full-scale oracle runs of the authors' code, CPU, 5 workers.
Hard line: 75 min wall from launch; jobs not started by then are skipped and reported. Longest jobs first.
Usage: python3 scripts/official/launch_full.py <out_root>"""
import concurrent.futures as cf
import os
import subprocess
import sys
import time

H = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
SD = os.path.dirname(os.path.abspath(__file__))  # this folder (our harness)
root = os.path.abspath(sys.argv[1])
POST, CL = f"{H}/.venv_post/bin/python", f"{H}/.venv/bin/python"
jobs = []  # (out_subdir, cmd, extra_env)
for s in range(1, 6):
    jobs.append(("pendulum_1m", [POST, "-I", f"{SD}/run_post.py",
                 f"{H}/ppo-implementation-details/ppo_continuous_action.py", "Pendulum-v1", str(s), "1000000"], {}))
    jobs.append(("pendulum_1m", [CL, "-I", f"{SD}/run_cleanrl.py", f"{H}/cleanrl/cleanrl/ppo_continuous_action.py",
                 "Pendulum-v1", str(s), "1000000"], {}))
for s in range(1, 6):
    jobs.append(("lstm_500k", [CL, "-I", f"{SD}/run_cleanrl.py", f"{H}/harness/cleanrl_ppo_lstm_maskedcartpole.py",
                 "CartPole-v1", str(s), "500000", "--num-envs", "8", "--num-steps", "128", "--num-minibatches", "4",
                 "--update-epochs", "4", "--clip-coef", "0.2", "--ent-coef", "0.01", "--learning-rate", "2.5e-4"],
                 {"MASKED_RSCALE": "1"}))
for env in ("Acrobot-v1", "CartPole-v1"):
    for s in range(6, 11):
        for sc in (f"{H}/cleanrl/cleanrl/ppo.py", f"{H}/harness/cleanrl_ppo_shared.py"):
            jobs.append(("e1_500k_s6_10", [CL, "-I", f"{SD}/run_cleanrl.py", sc, env, str(s), "500000"], {}))
for env in ("Acrobot-v1", "CartPole-v1"):
    for s in range(1, 6):
        jobs.append(("e1_500k_novclip", [CL, "-I", f"{SD}/run_cleanrl.py", f"{H}/cleanrl/cleanrl/ppo.py", env,
                     str(s), "500000", "--no-clip-vloss"], {}))
t0 = time.time()
LINE = 75 * 60


def run(job):
    sub, cmd, extra = job
    out = os.path.join(root, sub)
    os.makedirs(out, exist_ok=True)
    # run_*.py take <script> <env> <seed> <total> <out_dir> [extra args]: insert out_dir after the 4 positional args
    full = cmd[:7] + [out] + cmd[7:]
    if time.time() - t0 > LINE:
        return "SKIPPED (budget line) " + " ".join(full[3:7])
    p = subprocess.run(full, cwd=H, capture_output=True, text=True, env=dict(os.environ, **extra))
    return (p.stdout.strip() or p.stderr[-500:]) + f"  [{sub}] [t={time.time() - t0:.0f}s]"


print(f"[INFO] {len(jobs)} jobs, 5 workers, line {LINE}s", flush=True)
with cf.ThreadPoolExecutor(5) as ex:
    for r in ex.map(run, jobs):
        print("[METRIC]", r, flush=True)
print(f"[INFO] done in {time.time() - t0:.0f}s", flush=True)

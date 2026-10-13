"""follow-up resolution runs of the authors' code (declaration), CPU,
5 workers, hard line 75 min wall (jobs not started by then are skipped and reported).
lstm recipe: CleanRL LSTM logic on masked CartPole under the post's LSTM recipe (Atari dict: clip 0.1, no reward
      normalization; ClipReward is the identity on CartPole's +1), seeds 1-5, 500k.
lstm L3: CleanRL LSTM logic, dict, L3 on and L3 off, seeds 6-10 (with full-scale seeds 1-5 -> n=10 each), 500k.
baselines Pendulum: baselines-fork ppo2 with the MuJoCo treatment on Pendulum-v0, seeds 1-5, 1M.
Usage: python3 scripts/official/launch_resolution.py <out_root>"""
import concurrent.futures as cf
import os
import subprocess
import sys
import time

H = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
SD = os.path.dirname(os.path.abspath(__file__))  # this folder (our harness)
root = os.path.abspath(sys.argv[1])
CL, BL = f"{H}/.venv/bin/python", f"{H}/.conda_py37/bin/python"
BASE = ["--num-envs", "8", "--num-steps", "128", "--num-minibatches", "4", "--update-epochs", "4",
        "--ent-coef", "0.01", "--learning-rate", "2.5e-4"]
LSTM = f"{H}/harness/cleanrl_ppo_lstm_maskedcartpole.py"
NORESET = f"{H}/harness/cleanrl_ppo_lstm_maskedcartpole_noreset.py"
jobs = []  # (sub, cmd, env, n_positional_before_out)
for s in range(1, 6):  # longest first
    jobs.append(("baselines_pendulum_1m", [BL, f"{SD}/run_baselines_mujoco.py", "Pendulum-v0", str(s), "1000000"],
                 {}, 5))
for s in range(1, 6):
    jobs.append(("lstm_atari_500k", [CL, "-I", f"{SD}/run_cleanrl.py", LSTM, "CartPole-v1", str(s), "500000"]
                 + BASE + ["--clip-coef", "0.1"], {"MASKED_RSCALE": "0"}, 7))
for s in range(6, 11):
    for sub, sc in (("lstm_500k_s6_10", LSTM), ("lstm_noreset_500k_s6_10", NORESET)):
        jobs.append((sub, [CL, "-I", f"{SD}/run_cleanrl.py", sc, "CartPole-v1", str(s), "500000"]
                     + BASE + ["--clip-coef", "0.2"], {"MASKED_RSCALE": "1"}, 7))
t0 = time.time()
LINE = 75 * 60


def run(job):
    sub, cmd, extra, k = job
    out = os.path.join(root, sub)
    os.makedirs(out, exist_ok=True)
    full = cmd[:k] + [out] + cmd[k:]
    if time.time() - t0 > LINE:
        return "SKIPPED (budget line) " + " ".join(full[2:k])
    p = subprocess.run(full, cwd=H, capture_output=True, text=True, env=dict(os.environ, **extra))
    return (p.stdout.strip() or p.stderr[-500:]) + f"  [{sub}] [t={time.time() - t0:.0f}s]"


print(f"[INFO] {len(jobs)} jobs, 5 workers, line {LINE}s", flush=True)
with cf.ThreadPoolExecutor(5) as ex:
    for r in ex.map(run, jobs):
        print("[METRIC]", r, flush=True)
print(f"[INFO] done in {time.time() - t0:.0f}s", flush=True)

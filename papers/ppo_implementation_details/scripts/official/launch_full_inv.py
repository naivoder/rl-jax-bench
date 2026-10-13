"""full-scale investigations on the authors' code, CPU, 5 workers,
started after the batch. I-1: CleanRL ppo.py Acrobot-v1 seeds 11-30 (500k). I-4: CleanRL LSTM logic with L3 off
on masked CartPole, seeds 1-5 (500k). Hard line: 30 min wall; jobs not started by then are skipped.
Usage: python3 scripts/official/launch_full_inv.py <out_root>"""
import concurrent.futures as cf
import os
import subprocess
import sys
import time

H = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
SD = os.path.dirname(os.path.abspath(__file__))  # this folder (our harness)
root = os.path.abspath(sys.argv[1])
CL = f"{H}/.venv/bin/python"
LSTM_ARGS = ["--num-envs", "8", "--num-steps", "128", "--num-minibatches", "4", "--update-epochs", "4",
             "--clip-coef", "0.2", "--ent-coef", "0.01", "--learning-rate", "2.5e-4"]
jobs = [("lstm_noreset_500k", [CL, "-I", f"{SD}/run_cleanrl.py",
         f"{H}/harness/cleanrl_ppo_lstm_maskedcartpole_noreset.py", "CartPole-v1", str(s), "500000"] + LSTM_ARGS,
         {"MASKED_RSCALE": "1"}) for s in range(1, 6)]
jobs += [("acrobot_s11_30", [CL, "-I", f"{SD}/run_cleanrl.py", f"{H}/cleanrl/cleanrl/ppo.py", "Acrobot-v1",
          str(s), "500000"], {}) for s in range(11, 31)]
t0 = time.time()
LINE = 30 * 60


def run(job):
    sub, cmd, extra = job
    out = os.path.join(root, sub)
    os.makedirs(out, exist_ok=True)
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

"""Run openai/baselines ppo2 (the authors' fork @e2cb1c9 of ea25b9e, as used for the post's red curves) for one
(env, seed) on CPU, exactly as scripts/baselines/classic_control*.sh call it but with --num_env pinned and no
wandb. Episodes are read from the bench Monitor files (every episode, r and l); env i's episode that ends after its
k-th step is logged at global step k*N (vector envs step in lockstep).
Usage: .conda_py37/bin/python scripts/official/run_baselines.py <run|run_separate_networks> <env_id> <seed> <total> <out> <N>"""
import csv
import glob
import json
import os
import subprocess
import sys
import time

mod, env_id, seed, total, out, N = sys.argv[1:7]
H = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
tag = f"baselines_{mod}__{env_id}__{seed}"
work = os.path.abspath(os.path.join(out, tag))
os.makedirs(work, exist_ok=True)
cmd = [sys.executable, "-m", f"baselines.{mod}", "--alg=ppo2", f"--num_timesteps={total}", "--num_env", N,
       f"--env={env_id}", "--network", "mlp", "--seed", seed]
if mod == "run_separate_networks":
    cmd += ["--value_network=copy"]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OPENAI_LOGDIR=work, OPENAI_LOG_FORMAT="csv", OMP_NUM_THREADS="1",
           TF_CPP_MIN_LOG_LEVEL="3", WANDB_MODE="disabled")
t0 = time.time()
p = subprocess.run(cmd, cwd=os.path.join(H, "baselines_vwxyzjn_fork"), env=env, capture_output=True, text=True)
dt = time.time() - t0
open(os.path.join(work, "stdout.log"), "w").write(p.stdout[-20000:] + "\n--- stderr ---\n" + p.stderr[-20000:])
eps = []
mons = sorted(glob.glob(os.path.join(work, "*.monitor.csv")))
for m in mons:
    with open(m) as f:
        f.readline()  # json header
        k = 0
        for row in csv.DictReader(f):
            k += int(row["l"])
            eps.append([k * int(N), float(row["r"])])
eps.sort()
json.dump(dict(module=mod, env_id=env_id, seed=int(seed), total_timesteps=int(total), N=int(N), returncode=p.returncode,
               wall_seconds=dt, cmd=cmd, n_monitor_files=len(mons), episodes=eps),
          open(os.path.join(out, f"{tag}.json"), "w"))
print(tag, "rc", p.returncode, "monitors", len(mons), "episodes", len(eps), "secs", round(dt, 1),
      "last20", round(sum(e[1] for e in eps[-20:]) / max(1, len(eps[-20:])), 1))

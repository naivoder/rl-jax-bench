"""Run openai/baselines ppo2 (the authors' fork @e2cb1c9 of ea25b9e) with the MuJoCo treatment on a classic-control
continuous env: --env_type mujoco selects defaults.mujoco() (nsteps 2048, 32 mb, 10 epochs, lr 3e-4*f, clip 0.2,
ent 0, value_network='copy') and run.py:115 wraps the vec env in VecNormalize(use_tf=True) (obs norm + reward scaling,
the post's K6-K9). --num_env 1 (the post's mujoco N=1). Episodes come from the bench Monitor files (raw rewards,
before VecNormalize). gym 0.15.7 has only Pendulum-v0 (same dynamics as v1 except that v0 clips thdot AFTER
integrating theta; recheck: the clip binds on 8-11 % of steps under trained policies, so v0 and v1
differ on those steps by up to 5e-2 rad in theta; the end-to-end effect measured over 30 paired seeds is +44 +- 36, n.s.).
Usage: .conda_py37/bin/python scripts/official/run_baselines_mujoco.py <env_id> <seed> <total> <out>"""
import csv
import glob
import json
import os
import subprocess
import sys
import time

env_id, seed, total, out = sys.argv[1:5]
H = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
tag = f"baselines_mujoco__{env_id}__{seed}"
work = os.path.abspath(os.path.join(out, tag))
os.makedirs(work, exist_ok=True)
cmd = [sys.executable, "-m", "baselines.run", "--alg=ppo2", f"--num_timesteps={total}", "--num_env", "1",
       f"--env={env_id}", "--env_type", "mujoco", "--network", "mlp", "--seed", seed]
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
        f.readline()
        k = 0
        for row in csv.DictReader(f):
            k += int(row["l"])
            eps.append([k, float(row["r"])])
eps.sort()
json.dump(dict(module="run --env_type mujoco", env_id=env_id, seed=int(seed), total_timesteps=int(total), N=1,
               returncode=p.returncode, wall_seconds=dt, cmd=cmd, n_monitor_files=len(mons), episodes=eps),
          open(os.path.join(out, f"{tag}.json"), "w"))
print(tag, "rc", p.returncode, "monitors", len(mons), "episodes", len(eps), "secs", round(dt, 1),
      "last20", round(sum(e[1] for e in eps[-20:]) / max(1, len(eps[-20:])), 1))

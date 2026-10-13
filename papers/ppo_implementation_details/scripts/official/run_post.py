"""Run one script of the POST's own repo (vwxyzjn/ppo-implementation-details @fbef824, unmodified file, gym 0.21.0, torch 1.13.1 CPU) for one (env, seed), exactly as scripts/ours/classic_control*.sh call it minus --track. Derived from run_cleanrl.py.
Usage: .venv/bin/python -I scripts/official/run_cleanrl.py <script.py> <env_id> <seed> <total_timesteps> <out_dir> [extra args]
Episodic returns are parsed from the script's own stdout ("global_step=..., episodic_return=[...]"); the debug
scalars (losses, approx_kl, clipfrac, explained_variance) from its tensorboard event file."""
import glob
import json
import os
import re
import subprocess
import sys
import time

script, env_id, seed, total, out = sys.argv[1:6]
extra = sys.argv[6:]
H = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
os.makedirs(out, exist_ok=True)
tag = f"post_{os.path.basename(script)[:-3]}__{env_id}__{seed}"  # post_ prefix: never collide with CleanRL tags
work = os.path.join(out, tag)
os.makedirs(work, exist_ok=True)
cmd = [sys.executable, "-I", os.path.abspath(script), "--gym-id", env_id, "--seed", seed, "--total-timesteps", total,
       "--cuda", "False"] + extra
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=os.path.join(H, "ppo-implementation-details"))
t0 = time.time()
p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True)
dt = time.time() - t0
open(os.path.join(work, "stdout.log"), "w").write(p.stdout + "\n--- stderr ---\n" + p.stderr[-5000:])
eps = [[int(m.group(1)), float(m.group(2))] for m in
       re.finditer(r"global_step=(\d+), episodic_return=\[?([-\d.eE+]+)", p.stdout)]
scalars = {}
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ev = glob.glob(os.path.join(work, "runs", "*", "events.out.tfevents.*"))
    if ev:
        acc = EventAccumulator(ev[0], size_guidance={"scalars": 0})
        acc.Reload()
        for k in acc.Tags()["scalars"]:
            if k.startswith("losses/") or k in ("charts/learning_rate",):
                scalars[k] = [[e.step, e.value] for e in acc.Scalars(k)]
except Exception as e:  # noqa: BLE001
    scalars["error"] = repr(e)
json.dump(dict(script=script, env_id=env_id, seed=int(seed), total_timesteps=int(total), returncode=p.returncode,
               wall_seconds=dt, cmd=cmd, episodes=eps, scalars=scalars),
          open(os.path.join(out, f"{tag}.json"), "w"))
print(tag, "rc", p.returncode, "episodes", len(eps), "secs", round(dt, 1),
      "last20", round(sum(e[1] for e in eps[-20:]) / max(1, len(eps[-20:])), 1))

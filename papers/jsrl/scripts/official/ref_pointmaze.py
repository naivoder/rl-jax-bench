"""Reference run of the third-party SB3 JSRL (steventango/jumpstart-rl e5f690fe) on its OWN example task (PointMaze_UMaze-v3, TD3).
There is no official JSRL code; this is the second-best oracle's headline experiment at small scale.  Run in the reference venv:

    PYTHONPATH=$OFFICIAL_ROOT/jumpstart-rl/src $OFFICIAL_ROOT/.venv/bin/python scripts/official/ref_pointmaze.py guide  SEED STEPS OUTDIR
    PYTHONPATH=...                                                   scripts/official/ref_pointmaze.py arm    ARM SEED STEPS GUIDE_ZIP OUTDIR
ARM in {scratch, curr_default, curr_paper, random}.   Single-threaded torch/numpy.  Writes OUTDIR/<arm>_s<seed>.json
Every value below is tagged: 'their example' = steventango examples/*.py; 'paper' = our reading of Table 11 / App. A; 'ours' = this harness.
Evaluation is by THIS harness (not their EvalCallback, which is suspected of evaluating the combined policy, issue 10): pi_e alone via the BASE policy's predict,
and the combined policy, each on fresh seeded episodes, at every EVAL_EVERY steps.
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
torch.set_num_threads(1)
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.td3.policies import MultiInputPolicy
from jsrl import get_jsrl_algorithm

gym.register_envs(gymnasium_robotics)
ENV_ID, MAX_STEPS = "PointMaze_UMaze-v3", 150          # their example
EVAL_EVERY, EVAL_EPS = 5000, 20                          # ours (cheap learning curve)


def make_env():
    return gym.make(ENV_ID, continuing_task=False, max_episode_steps=MAX_STEPS)


def run_episodes(act_fn, n, seed0):
    """act_fn(obs, t) -> action.  Returns mean success/return over n seeded episodes."""
    env = make_env(); rets = []
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i); t = 0; ret = 0.0; done = False
        while not done:
            obs, r, term, trunc, _ = env.step(act_fn(obs, t)); ret += r; t += 1; done = term or trunc
        rets.append(ret)
    env.close()
    return float(np.mean(rets))


def pi_alone(policy):
    """Evaluate the exploration policy ALONE through the base TD3 policy's predict (no guide, no horizon logic)."""
    base = TD3.policy_aliases["MultiInputPolicy"]
    def act(obs, t):
        a, _ = base.predict(policy, {k: v[None] for k, v in obs.items()}, deterministic=True)
        return a[0]
    return act


def combined(policy, guide, h):
    base = TD3.policy_aliases["MultiInputPolicy"]
    def act(obs, t):
        o = {k: v[None] for k, v in obs.items()}
        if t <= h:        # THEIR rule (timesteps <= horizon): the guide acts at t = 0..h
            a, _ = guide.predict(o, deterministic=True)
        else:
            a, _ = base.predict(policy, o, deterministic=True)
        return a[0]
    return act


class Probe(BaseCallback):
    """Every EVAL_EVERY env steps: pi_e alone and the combined policy at the current horizon, on fixed seeded episodes; logs the horizon and, once,
    the issue 10 instrumentation (which policy THEIR predict() uses when called outside training: horizon actually applied)."""
    def __init__(self, guide, jsrl):
        super().__init__(); self.guide, self.jsrl, self.rows = guide, jsrl, []; self.t0 = time.time()
    def _on_step(self):
        if self.num_timesteps % EVAL_EVERY == 0:
            pol = self.model.policy
            h = pol.horizon if self.jsrl else -1
            alone = run_episodes(pi_alone(pol), EVAL_EPS, 10_000)
            comb = run_episodes(combined(pol, self.guide, h), EVAL_EPS, 10_000) if self.jsrl else alone
            self.rows.append(dict(step=self.num_timesteps, horizon=int(h), alone=alone, combined=comb, wall=time.time() - self.t0))
            print(f"[METRIC] step={self.num_timesteps} horizon={h} alone={alone:.3f} combined={comb:.3f} wall={time.time()-self.t0:.0f}s", flush=True)
        return True


def main():
    mode = sys.argv[1]
    if mode == "guide":
        seed, steps, out = int(sys.argv[2]), int(float(sys.argv[3])), sys.argv[4]
        os.makedirs(out, exist_ok=True)
        env = make_env()
        model = TD3("MultiInputPolicy", env, seed=seed, verbose=0)                      # their example train_guide_td3.py: defaults
        cb = Probe(None, False)
        t0 = time.time(); model.learn(total_timesteps=steps, callback=cb)
        model.save(os.path.join(out, f"guide_s{seed}.zip"))
        final = run_episodes(pi_alone(model.policy), 100, 20_000)
        json.dump(dict(seed=seed, steps=steps, final100=final, curve=cb.rows, wall=time.time() - t0), open(os.path.join(out, f"guide_s{seed}.json"), "w"))
        print(f"[INFO] guide seed {seed}: 100-episode success {final:.3f} wall {time.time()-t0:.0f}s")
        return
    arm, seed, steps, guide_zip, out = sys.argv[2], int(sys.argv[3]), int(float(sys.argv[4])), sys.argv[5], sys.argv[6]
    os.makedirs(out, exist_ok=True)
    env = make_env()
    guide = TD3.load(guide_zip).policy if arm != "scratch" else None      # scratch never touches the guide
    t0 = time.time()
    if arm == "scratch":
        model = TD3("MultiInputPolicy", env, seed=seed, verbose=0)
        cb = Probe(guide, False)
    else:
        H1, n = 60, 10                                              # their example train_jsrl_curriculum.py: max_horizon 60, n 10, horizons = arange(60, -1, -6)
        kw = dict(guide_policy=guide, max_horizon=H1)
        if arm == "curr_default":                                    # THEIR defaults: tolerance 0.0, window_size 1, eval_freq 1000, n_eval_episodes 20
            kw.update(strategy="curriculum", horizons=np.arange(H1, -1, -H1 // n))
        elif arm == "curr_paper":                                    # paper-style: tolerance 0.05 (our assumption), MA window 5 (App. A.2.1/A.4)
            kw.update(strategy="curriculum", horizons=np.arange(H1, -1, -H1 // n), tolerance=0.05, window_size=5)
        elif arm == "random":
            kw.update(strategy="random", horizons=[H1])
        else:
            raise SystemExit(arm)
        model = get_jsrl_algorithm(TD3)("MultiInputPolicy", env, seed=seed, verbose=0, policy_kwargs=kw)
        cb = Probe(guide, True)
    model.learn(total_timesteps=steps, callback=cb)
    final_alone = run_episodes(pi_alone(model.policy), 100, 30_000)
    res = dict(arm=arm, seed=seed, steps=steps, final_alone100=final_alone, curve=cb.rows, wall=time.time() - t0, numpy=np.__version__)
    json.dump(res, open(os.path.join(out, f"{arm}_s{seed}.json"), "w"))
    print(f"[INFO] {arm} seed {seed}: final pi_e-alone success (100 eps) {final_alone:.3f} wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

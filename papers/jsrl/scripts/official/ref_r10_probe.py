"""Instrumented short run of the third-party JSRL (e5f690fe) to CONFIRM or REFUTE the suspected defects issue 10a/issue 10b.
  issue 10a: does a user EvalCallback (listed before their JSRLEvalCallback, as in their examples) evaluate the COMBINED policy (guide for the first h steps)?
  issue 10b: does JSRLAlgorithm.predict feed the evaluation rollout the TRAINING env's step counters?
Run in the reference venv.  Output JSON to argv[1].
"""
import json
import os
import sys
import inspect
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
torch.set_num_threads(1)
import gymnasium as gym, gymnasium_robotics
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from jsrl import get_jsrl_algorithm

gym.register_envs(gymnasium_robotics)
mk = lambda: gym.make("PointMaze_UMaze-v3", continuing_task=False, max_episode_steps=150)
env = mk()
guide = TD3("MultiInputPolicy", mk(), seed=1).policy                       # untrained guide: only the call pattern matters
H = [60, 54, 48, 42, 36, 30, 24, 18, 12, 6, 0]
model = get_jsrl_algorithm(TD3)("MultiInputPolicy", env, seed=0, verbose=0,
                               policy_kwargs=dict(guide_policy=guide, max_horizon=60, strategy="curriculum", horizons=np.array(H), eval_freq=100000))  # no JSRL eval in this probe
pol = model.policy
log = dict(train=[], user_eval=[])
orig = type(pol).predict
calls_guide = []
g_orig = guide.predict
def guide_pred(obs, state=None, episode_start=None, deterministic=False):
    calls_guide.append(len(next(iter(obs.values()))))
    return g_orig(obs, state, episode_start, deterministic)
guide.predict = guide_pred
def spy(self, observation, timesteps, state=None, episode_start=None, deterministic=False):
    in_eval = any(f.function == "evaluate_policy" for f in inspect.stack())
    n0 = len(calls_guide)
    out = orig(self, observation, timesteps, state, episode_start, deterministic)
    rec = dict(training=bool(self.training), jsrl_evaluation=bool(self.jsrl_evaluation), horizon=int(self.horizon), timesteps=int(np.asarray(timesteps)[0]),
               guide_used=len(calls_guide) > n0)
    (log["user_eval"] if in_eval else log["train"]).append(rec)
    return out
type(pol).predict = spy
user_eval = EvalCallback(mk(), n_eval_episodes=3, eval_freq=500, deterministic=True)       # exactly how their examples attach a user EvalCallback
model.learn(total_timesteps=1500, callback=user_eval)
ue, tr = log["user_eval"], log["train"]
res = dict(
    n_train_calls=len(tr), n_user_eval_calls=len(ue),
    train_training_flag_values=sorted({r["training"] for r in tr}),
    user_eval_training_flag_values=sorted({r["training"] for r in ue}),
    user_eval_jsrl_evaluation_values=sorted({r["jsrl_evaluation"] for r in ue}),
    user_eval_guide_used_fraction=float(np.mean([r["guide_used"] for r in ue])) if ue else None,
    user_eval_first_30_timesteps=[r["timesteps"] for r in ue[:30]],
    user_eval_first_30_guide_used=[bool(r["guide_used"]) for r in ue[:30]],
    horizon_during_eval=sorted({r["horizon"] for r in ue}),
)
json.dump(res, open(sys.argv[1], "w"), indent=1)
print(json.dumps(res, indent=1))

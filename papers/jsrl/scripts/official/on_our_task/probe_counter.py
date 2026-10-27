"""Instrumented run of THEIR real JSRLAlgorithm(DQN) (steventango e5f690fe, imported unchanged) on the original-MinAtar Breakout wrapper.
Question issue 16: is the per-episode step counter `_timesteps` that their predict() compares with the horizon equal to the TRUE
within-episode step?  (their code zeroes it with `buf_dones`, which holds the PREVIOUS env step's done flag.)
Question issue 17: their JSRLEvalCallback evaluates on the TRAINING VecEnv (`JSRLEvalCallback(self.env, ...)`): after an evaluation,
is the trainer's `_last_obs` still the env's current observation (else the next stored transition joins two episodes)?
Question issue 18: with SB3 learning_starts > 0, does the guide act at all before learning_starts?
Instrumentation only reads state (a subclass that logs before calling their predict unchanged)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from oracle_common import Breakout, NumpyGuide, MinAtarCNN, EpsQPolicy, JSRL_SRC
sys.path.insert(0, JSRL_SRC)
from jsrl.jsrl import get_jsrl_algorithm
from stable_baselines3 import DQN
import stable_baselines3.common.callbacks as _cb
IN_EVAL = [False]
_ep = _cb.evaluate_policy
def _wrapped(*a, **k):          # instrumentation only: marks predict calls made inside their evaluation
    IN_EVAL[0] = True
    try:
        return _ep(*a, **k)
    finally:
        IN_EVAL[0] = False
_cb.evaluate_policy = _wrapped

STAGES = [81, 73, 65, 57, 49, 41, 32, 24, 16, 8, 0]
Base = get_jsrl_algorithm(DQN)


def run(learning_starts, steps, seed=0):
    log = []

    class Probe(Base):
        def predict(self, observation, state=None, episode_start=None, deterministic=False):
            e = self.env.envs[0].unwrapped
            log.append(dict(counter=int(self._timesteps[0]), true_t=int(e.t), ep=int(e.ep_id), n=int(self.num_timesteps),
                            h=int(self.policy.horizon), jsrl_eval=bool(self.policy.jsrl_evaluation),
                            in_eval=IN_EVAL[0], last_obs_ok=bool(np.array_equal(np.asarray(observation)[0], e.last_obs))))
            return super().predict(observation, state, episode_start, deterministic)

    env = Breakout(seed=seed, reward_mode="rho")
    m = Probe(EpsQPolicy, env, learning_starts=learning_starts, buffer_size=20000, batch_size=32, train_freq=1, seed=seed, verbose=0,
              policy_kwargs=dict(guide_policy=NumpyGuide(seed=seed), max_horizon=81, horizons=STAGES, tolerance=0.05, window_size=5,
                                 strategy="curriculum", eval_freq=1000, n_eval_episodes=4, features_extractor_class=MinAtarCNN, net_arch=[128]))
    m.learn(steps)
    tr = [r for r in log if not r["jsrl_eval"]]      # training-time predict calls (jsrl_evaluation False only in the 1st episode, issue 15)
    ev = [r for r in log if r["jsrl_eval"]]
    return log, tr, ev


out = {}
log, tr, ev = run(learning_starts=0, steps=6000)
allp = log
# predictions made while TRAINING: identify them as calls where num_timesteps advanced since the previous call or eval flag semantics
# Simplest: training calls are those between evaluations; the eval calls come in bursts at n % 1000 == 0 with n frozen.
train_calls = [r for r in allp if not r["in_eval"]]
eval_calls = [r for r in allp if r["in_eval"]]
def summ(calls):
    c = np.array([r["counter"] for r in calls]); t = np.array([r["true_t"] for r in calls])
    first = [r for r in calls if r["true_t"] == 0]
    return dict(n_calls=len(calls), frac_counter_ne_true=float(np.mean(c != t)),
                at_true_t0_counter_values=[r["counter"] for r in first][:15], n_episode_starts=len(first),
                frac_episode_starts_counter_ne_0=float(np.mean([r["counter"] != 0 for r in first])) if first else None,
                at_true_t1_counter_values=[r["counter"] for r in calls if r["true_t"] == 1][:15],
                guide_acts_by_true_t=None)
out["R16_training"] = summ(train_calls)
out["R16_eval"] = summ(eval_calls)
# guide acts iff counter <= h; which true steps did the guide act on in training episodes 2.. (h = 81 at stage 0)
g = {}
for r in train_calls:
    if r["ep"] >= 1:
        g.setdefault(r["true_t"], []).append(r["counter"] <= r["h"])
out["R16_guide_share_by_true_t_ep>=1"] = {str(k): float(np.mean(v)) for k, v in sorted(g.items()) if k in (0, 1, 2, 3, 79, 80, 81, 82, 83)}
# issue 17: was the observation passed to predict equal to the env's current observation?
bad = [i for i, r in enumerate(train_calls) if not r["last_obs_ok"]]
out["R17_training_calls_with_stale_obs"] = len(bad)
out["R17_their_num_timesteps_at_stale"] = [train_calls[i]["n"] for i in bad][:20]
out["R17_n_evaluations"] = len({r["n"] for r in eval_calls})
out["R17_eval_calls_on_training_env"] = len(eval_calls)
# issue 18
log2, _, _ = run(learning_starts=2000, steps=3000, seed=1)
out["R18_first_predict_call_at_num_timesteps"] = log2[0]["n"] if log2 else None
out["R18_counter_at_first_predict_vs_true_t"] = (log2[0]["counter"], log2[0]["true_t"]) if log2 else None
print(json.dumps(out, indent=1)); json.dump(out, open("probe_counter_r16_r17.json", "w"), indent=1)

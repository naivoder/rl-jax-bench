"""REFERENCE RUN: THEIR JSRL wrapper (steventango e5f690fe src/jsrl/jsrl.py, imported unchanged) around stock SB3 2.4.0 DQN on OUR task
(original-MinAtar Breakout, OUR guide via a numpy port tested to 2.6e-6 logits / guide mean 6.796 vs 6.784), vs the same DQN from scratch.
pi_e ALONE is graded by eval_alone() (no guide, own counter, eps 0.01, 32 episodes, raw return) every 2,000 env steps + point 0:
the same metric, cadence and windows as ours (EARLY = points 1..10, FINAL = points 98..100 of a 200 k run).
usage: ref_run.py ARM SEED STEPS OUTDIR NSTEP     ARM in scratch | jsrl_paper | jsrl_default | jsrl_random
NSTEP 1 = their pinned stack (SB3 2.4.0, vanilla 1-step DQN); NSTEP 10 = SB3 2.7.0 (from $OFFICIAL_ROOT/sb3_nstep) DQN with 10-step returns
(our PDD's n; SB3's NStepReplayBuffer equals our replay's n-step target exactly: nstep_parity.json)."""
import sys, os, json, time
NSTEP = int(sys.argv[5]) if len(sys.argv) > 5 else 1
if NSTEP > 1:
    sys.path.insert(0, os.path.join(os.environ.get('OFFICIAL_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'external', 'official')), 'sb3_nstep'))   # stable-baselines3 2.7.0 (setup.sh)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch as th
th.set_num_threads(1)
from oracle_common import Breakout, NumpyGuide, MinAtarCNN, EpsQPolicy, JSRL_SRC, eval_alone
sys.path.insert(0, JSRL_SRC)
from jsrl.jsrl import get_jsrl_algorithm
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

arm, seed, steps, outdir = sys.argv[1], int(sys.argv[2]), int(float(sys.argv[3])), sys.argv[4]
EVAL_EVERY, N_ALONE = int(os.environ.get("EVAL_EVERY", steps // 100)), 32
STAGES = [81, 73, 65, 57, 49, 41, 32, 24, 16, 8, 0]             # ours = paper's n+1 values for H1 = 80.55, n = 10


class Cb(BaseCallback):
    def __init__(self):
        super().__init__(); self.rec = dict(step=[], alone=[], alone_sd=[], horizon=[], eps=[], guide_steps=[]); self.t0 = time.time()

    def _point(self):
        a, sd = eval_alone(self.model.q_net, N_ALONE, seed=1_000_003 * (seed + 1) + self.model.num_timesteps)
        pol = self.model.policy
        self.rec["step"].append(int(self.model.num_timesteps)); self.rec["alone"].append(a); self.rec["alone_sd"].append(sd)
        self.rec["horizon"].append(int(pol.horizon) if hasattr(pol, "horizon") else 0); self.rec["eps"].append(float(self.model.exploration_rate))

    def _on_training_start(self):
        self._point()

    def _on_step(self):
        if hasattr(self.model.policy, "horizon"):
            EpsQPolicy.cur_eps = float(self.model.exploration_rate)       # eps shim sync (pi_e's branch only)
        if self.model.num_timesteps % EVAL_EVERY == 0:
            self._point()
            if len(self.rec["step"]) % 10 == 1:
                print(f"[{arm} s{seed}] step {self.model.num_timesteps} alone {self.rec['alone'][-1]:.2f} h {self.rec['horizon'][-1]} "
                      f"eps {self.rec['eps'][-1]:.3f} {time.time() - self.t0:.0f}s", flush=True)
        return True


dqn_kw = dict(learning_rate=1e-4, buffer_size=100_000, learning_starts=5000, batch_size=32, gamma=0.99, train_freq=1, gradient_steps=1,
              target_update_interval=10_000, exploration_fraction=100_000 / steps, exploration_initial_eps=1.0, exploration_final_eps=0.01,
              max_grad_norm=10, seed=seed, verbose=0)
if NSTEP > 1:
    dqn_kw["n_steps"] = NSTEP
pk = dict(features_extractor_class=MinAtarCNN, net_arch=[128], optimizer_kwargs=dict(eps=1.5e-4))
env = Breakout(seed=seed, reward_mode="rho")
th.manual_seed(seed)
EpsQPolicy._rng = np.random.default_rng(seed + 7)
if arm == "scratch":
    from stable_baselines3.dqn.policies import DQNPolicy
    m = DQN(DQNPolicy, env, policy_kwargs=pk, **dqn_kw)
else:
    J = get_jsrl_algorithm(DQN)
    if arm == "jsrl_paper":      # paper-style settings (tol 0.05, M 5, our n+1 stage list, eval every 1 % x 16 episodes = ours)
        jk = dict(max_horizon=81, horizons=STAGES, tolerance=0.05, window_size=5, strategy="curriculum", eval_freq=EVAL_EVERY, n_eval_episodes=16)
    elif arm == "jsrl_paper_g0":  # PROBE P-A: jsrl_paper with ONE change, the guide rolls in from env step 0 (paper Algorithm 1) instead of
        #                          SB3's uniform-random actions before learning_starts.  Training still starts at 5 k.
        jk = dict(max_horizon=81, horizons=STAGES, tolerance=0.05, window_size=5, strategy="curriculum", eval_freq=EVAL_EVERY, n_eval_episodes=16)
        _J = J

        class J(_J):
            def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
                return super()._sample_action(0, action_noise, n_envs)
    elif arm == "jsrl_default":  # THEIR defaults + their example's stage list (tol 0, window 1, eval_freq 1000, 20 episodes)
        jk = dict(max_horizon=81, horizons=list(np.arange(81, -1, -81 // 10)), tolerance=0.0, window_size=1, strategy="curriculum",
                  eval_freq=1000, n_eval_episodes=20)
    elif arm == "jsrl_random":   # their random strategy (h ~ U{0..80})
        jk = dict(max_horizon=81, strategy="random", eval_freq=EVAL_EVERY)
    else:
        raise SystemExit(arm)
    m = J(EpsQPolicy, env, policy_kwargs=dict(pk, guide_policy=NumpyGuide(seed=seed + 11), **jk), **dqn_kw)
cb = Cb()
t0 = time.time()
m.learn(steps, callback=cb)
import stable_baselines3
rec = dict(cb.rec, arm=arm, seed=seed, nstep=NSTEP, sb3=stable_baselines3.__version__, steps=steps, secs=time.time() - t0, horizons=[int(h) for h in getattr(m.policy, "horizons", [0])])
os.makedirs(outdir, exist_ok=True)
json.dump(rec, open(os.path.join(outdir, f"{arm}_n{NSTEP}_s{seed}.json"), "w"))
print(f"[{arm} s{seed}] done {rec['secs']:.0f}s points {len(rec['step'])}", flush=True)

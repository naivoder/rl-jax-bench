"""Shared pieces for the official-code oracle on MinAtar Breakout (runs in the THIRD-PARTY venv built by scripts/official/setup.sh).
Run every script of this folder with a work directory as cwd (e.g. runs/on_our_task); inputs and outputs are read and written there.
LABELLED PORTS (not their code, not ours): (1) the ORIGINAL numpy MinAtar Breakout (kenjyoung 4241f246, the same file our
policy_parity.py drives) wrapped as a gymnasium env with gymnax's minimal action set [0,1,3] and a 1,000-step cap (our
MinAtar('Breakout').max_steps); (2) our guide's PPO actor (flax ActorCritic) forward pass in numpy, weights exported by
export_guide.py; (3) a 10-line eps-greedy shim for pi_e's branch of their JSRLPolicy (their JSRLAlgorithm.predict bypasses
DQN.predict, so stock SB3 epsilon is dropped: issue 14 in the README).  Their src/jsrl/jsrl.py is imported unchanged."""
from __future__ import annotations
import importlib.util, os, sys
import numpy as np
import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy

PAPER = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
OFFICIAL_ROOT = os.environ.get('OFFICIAL_ROOT', os.path.join(PAPER, 'external', 'official'))
MINATAR_SRC = os.path.join(OFFICIAL_ROOT, "MinAtar", "minatar", "environments", "breakout.py")
JSRL_SRC = os.path.join(OFFICIAL_ROOT, "jumpstart-rl", "src")
SB3_NSTEP = os.path.join(OFFICIAL_ROOT, "sb3_nstep")   # stable-baselines3 2.7.0 installed with --target (n-step replay)
ACTION_SET = [0, 1, 3]          # gymnax Breakout action_set == original minimal_action_set()


def load_breakout():
    spec = importlib.util.spec_from_file_location("official_breakout", MINATAR_SRC)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_BK = load_breakout()


class Breakout(gym.Env):
    """Original MinAtar Breakout, no sticky actions, 1,000-step truncation; obs float32 (10,10,4) in {0,1}.
    reward_mode 'raw' (evaluation) or 'rho' (training: sign(r) log(1+|r|), our reward_transform)."""
    metadata = {}

    def __init__(self, seed=0, max_steps=1000, reward_mode="raw"):
        self.env = _BK.Env()
        self.env.random = np.random.RandomState(seed)
        self.max_steps, self.reward_mode = max_steps, reward_mode
        self.observation_space = gym.spaces.Box(0.0, 1.0, (10, 10, 4), np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.t = 0
        self.ep_id = -1
        self.last_obs = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.env.random = np.random.RandomState(seed)
        self.env.reset(); self.t = 0; self.ep_id += 1
        o = self.env.state().astype(np.float32); self.last_obs = o
        return o, {}

    def step(self, a):
        r, term = self.env.act(ACTION_SET[int(a)])
        self.t += 1
        trunc = (not term) and self.t >= self.max_steps
        o = self.env.state().astype(np.float32); self.last_obs = o
        rr = float(r) if self.reward_mode == "raw" else float(np.sign(r) * np.log1p(abs(r)))
        return o, rr, bool(term), bool(trunc), {}


class NumpyGuide:
    """Our guide: argmax of the PPO logits, uniform random w.p. eps_g = 0.01.  Weights from export_guide.py.
    `predict(obs, state, episode_start, deterministic)` = the SB3 policy API their JSRLPolicy calls."""

    def __init__(self, path="guide_breakout_np.npz", eps_g=0.01, seed=0):
        w = np.load(path)
        self.ck, self.cb = w["params/Conv_0/kernel"].astype(np.float64), w["params/Conv_0/bias"].astype(np.float64)
        self.d0k, self.d0b = w["params/Dense_0/kernel"].astype(np.float64), w["params/Dense_0/bias"].astype(np.float64)
        self.d1k, self.d1b = w["params/Dense_1/kernel"].astype(np.float64), w["params/Dense_1/bias"].astype(np.float64)
        self.eps_g, self.rng = eps_g, np.random.default_rng(seed)

    def logits(self, x):  # x (B,10,10,4)
        x = np.asarray(x, np.float64)
        B, H, W, C = x.shape
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        out = np.zeros((B, H, W, self.ck.shape[-1]))
        for i in range(3):
            for j in range(3):
                out += xp[:, i:i + H, j:j + W, :] @ self.ck[i, j]
        h = np.maximum(out + self.cb, 0.0).reshape(B, -1)
        h = np.maximum(h @ self.d0k + self.d0b, 0.0)
        return h @ self.d1k + self.d1b

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        obs = np.asarray(obs)
        if obs.ndim == 3:
            obs = obs[None]
        a = np.argmax(self.logits(obs), -1)
        rnd = self.rng.random(len(a)) < self.eps_g
        a = np.where(rnd, self.rng.integers(0, 3, len(a)), a)
        return a.astype(np.int64), state


class MinAtarCNN(BaseFeaturesExtractor):
    """Conv(16, 3x3, s1, SAME) + ReLU + flatten (1,600): MinAtar's standard torso (Young & Tian 2019); same as ours."""

    def __init__(self, observation_space, features_dim=1600):
        super().__init__(observation_space, features_dim)
        self.conv = nn.Conv2d(observation_space.shape[-1], 16, 3, 1, 1)

    def forward(self, x):
        return th.relu(self.conv(x.permute(0, 3, 1, 2))).flatten(1)


class EpsQPolicy(DQNPolicy):
    """pi_e = eps-greedy on the online Q net when not deterministic.  `cur_eps` is set every env step from the algorithm's
    exploration_rate by EpsSync.  Used ONLY inside their JSRLPolicy (whose explorer branch calls super().predict)."""
    cur_eps = 1.0
    _rng = np.random.default_rng(0)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        a, s = super().predict(observation, state, episode_start, deterministic)
        if not deterministic:
            a = np.array(a, copy=True)
            m = self._rng.random(a.shape[0]) < self.cur_eps
            a[m] = self._rng.integers(0, 3, int(m.sum()))
        return a, s


def eval_alone(q_net, n_episodes, seed, eps=0.01, max_steps=1000):
    """GRADED metric, same definition as ours: pi_e ALONE (no guide, own per-episode counter), eps 0.01, raw return, mean of
    n episodes on a separate env.  Episodes are stepped in lock-step for speed."""
    rng = np.random.default_rng(seed)
    envs = [Breakout(seed=int(rng.integers(2 ** 31)), max_steps=max_steps) for _ in range(n_episodes)]
    obs = np.stack([e.reset()[0] for e in envs]); done = np.zeros(n_episodes, bool); ret = np.zeros(n_episodes)
    dev = next(q_net.parameters()).device
    while not done.all():
        with th.no_grad():
            q = q_net(th.as_tensor(obs, device=dev)).cpu().numpy()
        a = np.where(rng.random(n_episodes) < eps, rng.integers(0, 3, n_episodes), q.argmax(-1))
        for i, e in enumerate(envs):
            if done[i]:
                continue
            o, r, te, tr, _ = e.step(a[i]); ret[i] += r; obs[i] = o; done[i] = te or tr
    return float(ret.mean()), float(ret.std())

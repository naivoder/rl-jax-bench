"""Official-code parity fixtures from the UNMODIFIED official code (repo @ e7c19c3, torch 1.13.1 CPU, gym 0.18).

parity.npz:
  ev/<case>/...   official ``Learner.evaluate`` (the unbound method, called on a minimal namespace that holds only the
                  attributes it reads) with an official agent at the PAPER sizes, on the official Pendulum-V-v0
                  (POMDPWrapper + TimeLimit) and PointRobotSparse-v0 (VariBadWrapper, 2 episodes/task) eval envs.
                  A recording gym.Wrapper outside the env logs every reset state, action, obs, reward, done and the
                  post-step internal state; we also store the evaluate() returns and the official
                  ``metrics/return_eval_total`` formula np.mean(np.sum(returns, -1)).
                  Labelled perturbation: the policy's output layer is re-drawn U(-0.5, 0.5) (official init U(+-1e-3)
                  makes every deterministic action ~0 and would test nothing).
  buf/...         official SeqReplayBuffer (vanilla): valid-start weights after adding episodes of lengths 3/9/7/12,
                  and random_episodes() at a FORCED list of start indices (``_sample_indices`` replaced by a queue;
                  nothing else patched): obs, act, rew, term, obs2, mask.
Shims (labelled, as in run_official.py): Pendulum-v1 -> gym 0.18 PendulumEnv (= the paper's v0 dynamics); a stub
pybullet_envs module; gym.make returning None for LunarLander/MuJoCo ids not installed here.
"""
import os, sys, types
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
REPO = os.path.join(ROOT, "repo")
sys.path.insert(0, REPO)
import gym
from gym.envs.registration import register, registry
if "Pendulum-v1" not in [s.id for s in registry.all()]:
    register(id="Pendulum-v1", entry_point="gym.envs.classic_control:PendulumEnv", max_episode_steps=200)
sys.modules.setdefault("pybullet_envs", types.ModuleType("pybullet_envs"))
_orig_make = gym.make
def _make(id, **kw):
    if id.startswith(("LunarLander", "Hopper", "Ant", "Walker", "HalfCheetah", "Humanoid")) and "BLT" not in id:
        try:
            return _orig_make(id, **kw)
        except Exception:
            return None
    return _orig_make(id, **kw)
gym.make = _make
torch.set_num_threads(1)

import envs.pomdp  # noqa: F401  (registers Pendulum-V-v0)
from envs.meta.make_env import make_env
from policies.learner import Learner
from policies.models import AGENT_ARCHS
from policies.models.policy_rnn import ModelFreeOffPolicy_Separate_RNN
from policies.models.policy_rnn_shared import ModelFreeOffPolicy_Shared_RNN
from buffers.seq_replay_buffer_vanilla import SeqReplayBuffer

OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "tests", "fixtures")
ALGO_KW = {"sac": dict(entropy_alpha=0.2, automatic_entropy_tuning=True, alpha_lr=3e-4),
           "td3": dict(exploration_noise=0.1, target_noise=0.2, target_noise_clip=0.5)}
# paper / official config sizes (configs/pomdp/pendulum/v/rnn.yml, configs/meta/point_robot/rnn.yml)
SIZES = dict(observ_embedding_size=32, reward_embedding_size=8, rnn_hidden_size=128,
             dqn_layers=[128, 128], policy_layers=[128, 128], lr=3e-4, gamma=0.9, tau=0.005)


class Rec(gym.Wrapper):
    def __init__(self, env, max_steps):
        super().__init__(env)
        self._max_episode_steps = max_steps
        self.eps = []

    def _state(self):
        u = self.env.unwrapped
        return np.array(u.state if hasattr(u, "state") and not hasattr(u, "_state") else u._state, np.float64).copy()

    def reset(self, **kw):
        o = self.env.reset(**kw)
        self.eps.append(dict(s0=self._state(), obs=[np.array(o)], act=[], rew=[], done=[], state=[],
                             goal=np.array(getattr(self.env.unwrapped, "_goal", np.zeros(2))).copy()))
        return o

    def step(self, a):
        o, r, d, i = self.env.step(a)
        E = self.eps[-1]
        E["act"].append(np.array(a, np.float64)); E["obs"].append(np.array(o)); E["rew"].append(r)
        E["done"].append(float(d)); E["state"].append(self._state())
        return o, r, d, i


def make_agent(algo, cell, arch, act_emb, obs_dim, act_dim, seed):
    torch.manual_seed(seed)
    kw = dict(obs_dim=obs_dim, action_dim=act_dim, encoder=cell, algo_name=algo, action_embedding_size=act_emb,
              **SIZES, **{algo: ALGO_KW[algo]})
    agent = (ModelFreeOffPolicy_Separate_RNN if arch == "separate" else ModelFreeOffPolicy_Shared_RNN)(**kw)
    pol = agent.actor.policy if arch == "separate" else agent.policy
    with torch.no_grad():  # labelled perturbation (see module docstring)
        pol.last_fc.weight.uniform_(-0.5, 0.5)
        pol.last_fc.bias.uniform_(-0.5, 0.5)
    return agent


def eval_case(out, name, algo, cell, arch, act_emb, env_kind, n_tasks, seed):
    if env_kind == "pendulum":
        env = gym.make("Pendulum-V-v0")
        env.seed(seed)
        rec = Rec(env, env._max_episode_steps)
        ns = types.SimpleNamespace(max_rollouts_per_task=1, env_type="pomdp", eval_env=rec,
                                   agent_arch=AGENT_ARCHS.Memory, reward_clip=False)
        tasks = n_tasks * [None]
        obs_dim, act_dim = 1, 1
    else:
        np.random.seed(seed)
        env = make_env("PointRobotSparse-v0", 2, seed=seed, n_tasks=100)
        env.seed(seed + 1)
        rec = Rec(env, None)
        ns = types.SimpleNamespace(max_rollouts_per_task=2, env_type="meta", eval_env=rec,
                                   agent_arch=AGENT_ARCHS.Memory, reward_clip=False,
                                   max_trajectory_len=env.horizon_bamdp)
        tasks = list(range(n_tasks))
        obs_dim, act_dim = 3, 2
    ns.agent = make_agent(algo, cell, arch, act_emb, obs_dim, act_dim, seed)
    returns, succ, _, steps = Learner.evaluate(ns, tasks, deterministic=True)
    p = f"ev/{name}/"
    for k, v in ns.agent.state_dict().items():   # actor-side parameters only (all the eval path reads)
        keep = k.startswith("actor.") if arch == "separate" else not (
            k.startswith(("qf", "policy_target", "current_observ_action_embedder")))
        if keep:
            out[p + "sd/" + k] = v.detach().numpy().copy()
    out[p + "returns"] = returns
    out[p + "metric"] = np.float64(np.mean(np.sum(returns, axis=-1)))   # official metrics/return_eval_total
    out[p + "steps"] = steps
    out[p + "n_eps"] = np.int64(len(rec.eps))
    for e, E in enumerate(rec.eps):
        for k in ("s0", "goal"):
            out[f"{p}e{e}/{k}"] = E[k]
        for k in ("obs", "act", "rew", "done", "state"):
            out[f"{p}e{e}/{k}"] = np.array(E[k], np.float64)
    print("eval", name, "metric", out[p + "metric"], "acts std",
          np.std(np.concatenate([np.array(E["act"]).ravel() for E in rec.eps])))


def dump_buffer(out):
    L = 7
    buf = SeqReplayBuffer(200, 2, 1, sampled_seq_len=L, sample_weight_baseline=0.0)
    rng = np.random.RandomState(3)
    lens = [3, 9, 7, 12]
    for e, n in enumerate(lens):
        obs = rng.normal(size=(n + 1, 2)).astype(np.float32)
        act = rng.uniform(-2, 2, (n, 1)).astype(np.float32)
        rew = rng.normal(size=(n, 1)).astype(np.float32)
        term = np.zeros((n, 1)); term[-1] = float(e % 2)
        buf.add_episode(obs[:-1], act, rew, term, obs[1:])
        out[f"buf/ep{e}/obs"], out[f"buf/ep{e}/act"], out[f"buf/ep{e}/rew"], out[f"buf/ep{e}/term"] = obs, act, rew, term
    out["buf/lens"] = np.array(lens)
    out["buf/L"] = np.int64(L)
    out["buf/valid_starts"] = buf._valid_starts[: sum(lens)].copy()
    # forced starts: every valid start once (global flat index), sampled through the official random_episodes()
    starts = np.where(buf._valid_starts > 0)[0]
    buf._sample_indices = lambda batch_size: starts[:batch_size]
    b = buf.random_episodes(len(starts))
    out["buf/starts"] = starts
    for k, v in b.items():
        out[f"buf/batch/{k}"] = np.asarray(v, np.float64)
    print("buffer starts", starts, "mask sums", b["mask"].sum(0).ravel())


if __name__ == "__main__":
    out = {}
    eval_case(out, "pend_sac_lstm_separate", "sac", "lstm", "separate", 8, "pendulum", 4, 11)
    eval_case(out, "pend_sac_lstm_shared", "sac", "lstm", "shared", 8, "pendulum", 4, 12)
    eval_case(out, "pend_td3_lstm_separate", "td3", "lstm", "separate", 8, "pendulum", 4, 13)
    eval_case(out, "pend_td3_gru_shared", "td3", "gru", "shared", 8, "pendulum", 4, 14)
    eval_case(out, "pend_sac_gru_separate", "sac", "gru", "separate", 8, "pendulum", 4, 15)
    # Semi-Circle: the small stage cell is td3-lstm-64-or (no action input: action_embedding_size 0)
    eval_case(out, "semi_td3_lstm_separate_or", "td3", "lstm", "separate", 0, "semi", 3, 21)
    eval_case(out, "semi_td3_lstm_shared_or", "td3", "lstm", "shared", 0, "semi", 3, 22)
    eval_case(out, "semi_sac_gru_separate_oar", "sac", "gru", "separate", 8, "semi", 3, 23)
    dump_buffer(out)
    np.savez_compressed(os.path.join(OUT, "parity.npz"), **out)
    print("parity ok", len(out))

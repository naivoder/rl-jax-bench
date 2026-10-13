"""Parity fixtures (warm-up, replay buffer) from the UNMODIFIED official code (repo @ e7c19c3, torch 1.13.1 CPU, gym 0.18).

parity2.npz:
  warm/<env>/...   official ``Learner.collect_rollouts(n, random_actions=True)`` (the unbound method, called on a
                   minimal namespace holding only the attributes it reads; ``policy_storage`` is a recorder of the
                   ``add_episode`` arguments, the agent is a stub for ``get_initial_info``).  A gym.Wrapper outside the
                   train env records the internal state before/after every step.  Envs: Pendulum-V-v0 (seed 5) and
                   PointRobotSparse-v0 with the VariBAD wrapper (2 episodes/task, seed 7).
  rbuf/...         official RAMEfficient_SeqReplayBuffer (the App. A.1 buffer): valid-start weights after adding
                   episodes of lengths 3/9/7/12 (L = 7) and random_episodes() at every valid start (``_sample_indices``
                   forced; nothing else patched) -- the same protocol as parity.npz's vanilla-buffer fixture.
Shims as in dump_parity.py (Pendulum-v1 id -> gym 0.18 PendulumEnv; pybullet stub).
"""
import os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dump_parity as P   # noqa: E402  (applies the labelled shims; defines Learner, make_env, gym)
import numpy as np
from buffers.seq_replay_buffer_efficient import RAMEfficient_SeqReplayBuffer
from policies.models import AGENT_ARCHS


class StateRec(P.gym.Wrapper):
    def __init__(self, env, max_steps):
        super().__init__(env)
        self._max_episode_steps = max_steps
        self.pre, self.post, self.goal = [], [], []

    def _state(self):
        u = self.env.unwrapped
        return np.array(u.state if hasattr(u, "state") and not hasattr(u, "_state") else u._state, np.float64).copy()

    def step(self, a):
        self.pre.append(self._state())
        self.goal.append(np.array(getattr(self.env.unwrapped, "_goal", np.zeros(2)), np.float64).copy())
        o, r, d, i = self.env.step(a)
        self.post.append(self._state())
        return o, r, d, i


class Storage:
    def __init__(self):
        self.eps = []

    def add_episode(self, observations, actions, rewards, terminals, next_observations):
        self.eps.append(dict(obs=observations, act=actions, rew=rewards, term=terminals, obs2=next_observations))


class AgentStub:
    def get_initial_info(self):
        return None, None, None


def warmup(out, kind, n, seed):
    if kind == "pendulum":
        env = P.gym.make("Pendulum-V-v0")
        env.seed(seed)
        env.action_space.np_random.seed(seed)    # learner.py: 'crucial'
        rec = StateRec(env, env._max_episode_steps)
        ns = types.SimpleNamespace(env_type="pomdp", max_trajectory_len=env._max_episode_steps)
    else:
        np.random.seed(seed)
        env = P.make_env("PointRobotSparse-v0", 2, seed=seed, n_tasks=100)
        env.seed(seed + 1)
        rec = StateRec(env, None)
        rec.n_tasks = env.n_tasks
        ns = types.SimpleNamespace(env_type="meta", max_trajectory_len=env.horizon_bamdp,
                                   train_tasks=list(range(80)))
    ns.__dict__.update(train_env=rec, agent_arch=AGENT_ARCHS.Memory, act_continuous=True, reward_clip=False,
                       policy_storage=Storage(), agent=AgentStub(), _n_env_steps_total=0, _n_rollouts_total=0,
                       _successes_in_buffer=0)
    P.Learner.collect_rollouts(ns, n, random_actions=True)
    p = f"warm/{kind}/"
    for k in ("obs", "act", "rew", "term", "obs2"):
        out[p + k] = np.concatenate([np.asarray(e[k], np.float64) for e in ns.policy_storage.eps])
    out[p + "pre"], out[p + "post"], out[p + "goal"] = (np.array(rec.pre), np.array(rec.post), np.array(rec.goal))
    out[p + "lens"] = np.array([len(e["act"]) for e in ns.policy_storage.eps])
    a = out[p + "act"]
    print("warmup", kind, "stored action range", a.min(), a.max(), "n", len(a))


def dump_rbuf(out):
    L = 7
    buf = RAMEfficient_SeqReplayBuffer(200, 2, 1, sampled_seq_len=L, sample_weight_baseline=0.0,
                                       observation_type=np.float32)
    rng = np.random.RandomState(3)
    lens = [3, 9, 7, 12]
    for e, n in enumerate(lens):
        obs = rng.normal(size=(n + 1, 2)).astype(np.float32)
        act = rng.uniform(-2, 2, (n, 1)).astype(np.float32)
        rew = rng.normal(size=(n, 1)).astype(np.float32)
        term = np.zeros((n, 1)); term[-1] = float(e % 2)
        buf.add_episode(obs[:-1], act, rew, term, obs[1:])
    out["rbuf/lens"] = np.array(lens)
    out["rbuf/L"] = np.int64(L)
    out["rbuf/valid_starts"] = buf._valid_starts[: sum(lens) + len(lens)].copy()
    starts = np.where(buf._valid_starts > 0)[0]
    buf._sample_indices = lambda batch_size: starts[:batch_size]
    b = buf.random_episodes(len(starts))
    out["rbuf/starts"] = starts
    for k, v in b.items():
        out[f"rbuf/batch/{k}"] = np.asarray(v, np.float64)
    print("ram-efficient buffer starts", starts, "mask sums", b["mask"].sum(0).ravel())


if __name__ == "__main__":
    out = {}
    warmup(out, "pendulum", 3, 5)
    warmup(out, "semi", 3, 7)
    dump_rbuf(out)
    np.savez_compressed(os.path.join(P.OUT, "parity2.npz"), **out)
    print("parity2 ok", len(out))

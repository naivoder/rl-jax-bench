"""Oracle fixtures: run the AUTHORS' code (branch 2024 @ 407dca7, isolated venv) on fixed inputs and
weights, save inputs + outputs to tests/fixtures/oracle_fixtures.npz. Our JAX side is compared in
tests/test_official_parity.py. Nothing from the official code is copied into src/.

Run:  $OFFICIAL_ROOT/.venv/bin/python -I scripts/official/make_fixtures.py
"""
from __future__ import annotations

import os
import subprocess
import sys

OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' clone + venv (scripts/official/setup.sh)
OFFICIAL = os.path.join(OFFICIAL_ROOT, "repo")
sys.path.insert(0, OFFICIAL)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests", "fixtures",
                   "oracle_fixtures.npz")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import optim  # noqa: E402
import sparse_init as si  # noqa: E402
import stream_ac_continuous as sac  # noqa: E402
import stream_q_minatar as sqm  # noqa: E402
from normalization_wrappers import NormalizeObservation, ScaleReward  # noqa: E402
from time_wrapper import AddTimeInfo  # noqa: E402

torch.set_num_threads(1)
rng = np.random.default_rng(1234)
F: dict[str, np.ndarray] = {}
commit = subprocess.check_output(["git", "-C", OFFICIAL, "rev-parse", "HEAD"], text=True).strip()
F["commit"] = np.array(commit)


def put(prefix, module):
    for n, p in module.named_parameters():
        F[f"{prefix}/{n}"] = p.detach().numpy().copy()


def set_random(module, scale=0.3):
    with torch.no_grad():
        for n, p in module.named_parameters():
            p.copy_(torch.from_numpy((rng.standard_normal(p.shape) * scale).astype(np.float32)))


def traces(prefix, opt):
    for gi, g in enumerate(opt.param_groups):
        for pi, p in enumerate(g["params"]):
            F[f"{prefix}/{pi}"] = opt.state[p]["eligibility_trace"].numpy().copy()


# ---------------------------------------------------------------- A. ObGD on a tiny MLP (3 steps, last resets)
net = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Linear(4, 1))
set_random(net, 0.5)
put("obgd/w0", net)
opt = optim.ObGD(net.parameters(), lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0)
for k, (scale, delta, reset) in enumerate([(1.0, 2.5, False), (0.3, -0.4, False), (1e-3, 0.7, True)]):
    for n, p in net.named_parameters():
        g = (rng.standard_normal(p.shape) * scale).astype(np.float32)
        F[f"obgd/g{k}/{n}"] = g
        p.grad = torch.from_numpy(g.copy())
    F[f"obgd/delta{k}"] = np.float32(delta)
    F[f"obgd/reset{k}"] = np.array(reset)
    # torch grads are of -output in the official usage; the optimizer itself is sign-agnostic: record as given
    opt.step(delta, reset=reset)
    put(f"obgd/w{k + 1}", net)
    traces(f"obgd/z{k + 1}", opt)

# ---------------------------------------------------------------- B. SparseInit statistics
torch.manual_seed(7)
for name, shape in [("fc", (128, 1024)), ("conv", (16, 4, 3, 3)), ("small", (128, 12))]:
    w = torch.empty(shape)
    si.sparse_init(w, sparsity=0.9)
    w = w.numpy().reshape(shape[0], -1)
    F[f"sparse/{name}/zeros_per_unit"] = (w == 0).sum(1)
    F[f"sparse/{name}/nonzeros"] = w[w != 0]

# ---------------------------------------------------------------- C/E. stream Q network forward + 3 updates
torch.manual_seed(0)
np.random.seed(0)
agent = sqm.StreamQ(n_channels=4, n_actions=3, hidden_size=128, lr=1.0, gamma=0.99, lamda=0.8, kappa_value=2.0,
                    total_steps=1000, exploration_fraction=0.2)
set_random(agent.network, 0.1)
put("q/w0", agent.network)
xs = (rng.random((4, 10, 10, 4)) < 0.2).astype(np.float32) * rng.standard_normal((4, 10, 10, 4)).astype(np.float32)
F["q/x"] = xs
with torch.no_grad():
    F["q/qvals"] = np.stack([agent.q(torch.from_numpy(x)).numpy() for x in xs])
steps = [(0, 1.0, False, False), (2, 0.0, False, True), (1, -1.0, True, False)]  # (a, r, done, nongreedy)
for k, (a, r, done, ng) in enumerate(steps):
    with torch.no_grad():
        q_sa = agent.q(torch.from_numpy(xs[k]))[a].item()
        qn = agent.q(torch.from_numpy(xs[k + 1])).max().item()
    F[f"q/delta{k}"] = np.float32(r + 0.99 * qn * (0 if done else 1) - q_sa)
    agent.update_params(xs[k], a, np.float32(r), xs[k + 1], done, ng)
    put(f"q/w{k + 1}", agent.network)
    traces(f"q/z{k + 1}", agent.optimizer)
F["q/steps"] = np.array(steps, dtype=np.float32)

# ---------------------------------------------------------------- C/F. stream AC forward + 3 updates
torch.manual_seed(0)
acn = sac.StreamAC(n_obs=12, n_actions=3, hidden_size=128, lr=1.0, gamma=0.99, lamda=0.8, kappa_policy=3.0,
                   kappa_value=2.0)
set_random(acn.policy_net, 0.1)
set_random(acn.value_net, 0.1)
put("ac/pi0", acn.policy_net)
put("ac/v0", acn.value_net)
S = rng.standard_normal((4, 12)).astype(np.float32)
A = rng.standard_normal((3, 3)).astype(np.float32)
F["ac/s"], F["ac/a"] = S, A
with torch.no_grad():
    # one sample at a time: the official LayerNorm normalizes over x.size, i.e. over a whole batch if batched
    outs = [acn.pi(torch.from_numpy(s_)) for s_ in S]
    F["ac/mu"] = np.stack([m.numpy() for m, _ in outs])
    F["ac/std"] = np.stack([sd.numpy() for _, sd in outs])
    F["ac/v"] = np.stack([acn.v(torch.from_numpy(s_)).numpy() for s_ in S])
ac_steps = [(0.5, False), (-2.0, False), (1.0, True)]
for k, (r, done) in enumerate(ac_steps):
    with torch.no_grad():
        d = r + 0.99 * acn.v(torch.from_numpy(S[k + 1])).item() * (0 if done else 1) - acn.v(torch.from_numpy(S[k])).item()
    F[f"ac/delta{k}"] = np.float32(d)
    acn.update_params(S[k], A[k], np.float32(r), S[k + 1], done, entropy_coeff=0.01)
    put(f"ac/pi{k + 1}", acn.policy_net)
    put(f"ac/v{k + 1}", acn.value_net)
    traces(f"ac/zpi{k + 1}", acn.optimizer_policy)
    traces(f"ac/zv{k + 1}", acn.optimizer_value)
F["ac/steps"] = np.array(ac_steps, dtype=np.float32)


# ---------------------------------------------------------------- G. kink case: first obs normalizes to EXACTLY 0
# (Welford's first sample), so every pre-activation is 0 and LeakyReLU's derivative at 0 decides the update.
import stream_td as std_  # noqa: E402
torch.manual_seed(0)
td = std_.StreamTD(n_obs=7, lr=1.0, gamma=0.99, lamda=0.8, kappa_value=2.0)
set_random(td, 0.3)
with torch.no_grad():
    for n_, p_ in td.named_parameters():
        if n_.endswith("bias"):
            p_.zero_()      # biases 0 as after SparseInit: zero input => every pre-activation exactly 0
put("td/w0", td)
s0 = np.zeros(7, np.float32)
s1 = rng.standard_normal(7).astype(np.float32)
s2 = rng.standard_normal(7).astype(np.float32)
F["td/s"] = np.stack([s0, s1, s2])
for k, r in enumerate([0.65, 1.4]):
    td.update_params(F["td/s"][k], np.float32(r), F["td/s"][k + 1], False)
    put(f"td/w{k + 1}", td)
F["td/r"] = np.array([0.65, 1.4], np.float32)
# the same for the MinAtar Q-net with an all-zero (normalized) first observation
torch.manual_seed(0)
q0 = sqm.StreamQ(n_channels=4, n_actions=3, total_steps=1000)
set_random(q0.network, 0.1)
with torch.no_grad():
    for n_, p_ in q0.network.named_parameters():
        if n_.endswith("bias"):
            p_.zero_()
put("qz/w0", q0.network)
q0.update_params(np.zeros((10, 10, 4), np.float32), 1, np.float32(0.5), xs[0], False, False)
put("qz/w1", q0.network)


# ---------------------------------------------------------------- D. wrappers on a scripted env
class Scripted(gym.Env):
    def __init__(self, obs, rew, term, T=5):
        self.obs, self.rew, self.term = obs, rew, term
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs.shape[1],), np.float64)
        self.action_space = gym.spaces.Box(-1, 1, (1,), np.float32)
        self.i = 0   # obs row pointer (resets and steps both consume a row)
        self.k = 0   # step counter (rewards / terminations)
        self.spec = gym.envs.registration.EnvSpec("Scripted-v0", max_episode_steps=T)

    def reset(self, seed=None, options=None):
        self.i += 1  # the reset obs is the next scripted row
        return self.obs[self.i - 1], {}

    def step(self, a):
        o, r, t = self.obs[self.i], self.rew[self.k], bool(self.term[self.k])
        self.i += 1
        self.k += 1
        return o, float(r), t, False, {}


n = 40
obs = rng.standard_normal((n + 10, 3)) * np.array([1.0, 10.0, 0.01]) + np.array([0.0, 5.0, -3.0])
rew = rng.standard_normal(n) * 2.0
term = np.zeros(n, bool)
term[[6, 17, 18, 30]] = True
env = AddTimeInfo(NormalizeObservation(ScaleReward(gym.wrappers.RecordEpisodeStatistics(Scripted(obs, rew, term)),
                                                   gamma=0.99)))
o, _ = env.reset()
seq_obs, seq_rew, seq_done = [o], [], []
for k in range(n):
    o, r, t, tr, _ = env.step(np.zeros(1, np.float32))
    seq_obs.append(o)
    seq_rew.append(r)
    seq_done.append(t)
    if t:
        o, _ = env.reset()
        seq_obs.append(o)
F["wr/obs_in"], F["wr/rew_in"], F["wr/term_in"] = obs, rew, term
F["wr/obs_out"] = np.array(seq_obs)
F["wr/rew_out"] = np.array(seq_rew)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, **F)
print(f"[INFO] wrote {OUT} keys={len(F)} official_commit={commit} torch={torch.__version__}")

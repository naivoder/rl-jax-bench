"""Closed-loop oracle fixtures: run the AUTHORS' main loops (branch 2024 @ 407dca7) exactly as their
``main()`` builds them -- same seeding order, same wrapper stack, same agent -- on the REAL envs (MinAtar 1.0.15
Breakout-v1 for stream Q(lambda); gymnasium MuJoCo Hopper-v4 for stream AC(lambda)), and record every RAW env output
(a pass-through recorder sits directly under their wrapper stack), the agent inputs their wrappers produced, the
actions, epsilon, the non-greedy flags, every TD error, the episode returns RecordEpisodeStatistics reported, and
parameter snapshots. tests/test_official_parity.py drives OUR wrappers + update (agents.QAgent.observe /
ACAgent.observe) with the same raw env outputs and actions and diffs everything step by step.

Tolerance calibration: the same recorded inputs are replayed through the official agent twice more -- unperturbed
(must reproduce the closed-loop run exactly: validates the replay harness) and from initial weights perturbed by
1e-7 relative (float32 rounding scale) -- giving the official code's OWN sensitivity to float32 rounding over the
horizon. Nothing from the official code is copied into src/.

Run:  $OFFICIAL_ROOT/.venv/bin/python -I scripts/official/make_closed_loop_fixtures.py
"""
from __future__ import annotations

import copy
import os
import subprocess
import sys

OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' clone + venv (scripts/official/setup.sh)
OFFICIAL = os.path.join(OFFICIAL_ROOT, "repo")
sys.path.insert(0, OFFICIAL)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests", "fixtures",
                   "closed_loop_fixtures.npz")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import stream_ac_continuous as sac  # noqa: E402
import stream_q_minatar as sqm  # noqa: E402
from normalization_wrappers import NormalizeObservation, ScaleReward  # noqa: E402
from time_wrapper import AddTimeInfo  # noqa: E402

torch.set_num_threads(1)
F: dict[str, np.ndarray] = {}
F["commit"] = np.array(subprocess.check_output(["git", "-C", OFFICIAL, "rev-parse", "HEAD"], text=True).strip())
N_Q, N_AC, SEED = 600, 600, 1
SNAPS = (1, 10, 100)


class Recorder(gym.Wrapper):
    """Pass-through: remembers the raw env output of the last reset / step (no numerical effect)."""

    def reset(self, **kw):
        o, i = self.env.reset(**kw)
        self.reset_obs = np.array(o)
        return o, i

    def step(self, a):
        o, r, te, tr, i = self.env.step(a)
        self.last = (np.array(o), float(r), bool(te), bool(tr))
        return o, r, te, tr, i


def put(prefix, module):
    for n, p in module.named_parameters():
        F[f"{prefix}/{n}"] = p.detach().numpy().copy()


def traces(prefix, opt):
    for pi, p in enumerate(opt.param_groups[0]["params"]):
        F[f"{prefix}/{pi}"] = opt.state[p]["eligibility_trace"].numpy().copy()


def hook_delta(opt, sink):
    orig = opt.step

    def step(delta, reset=False):
        sink.append(delta)
        return orig(delta, reset=reset)
    opt.step = step


def perturb(module, rel, gen):
    with torch.no_grad():
        for p in module.parameters():
            p.mul_(1.0 + rel * torch.from_numpy(gen.standard_normal(p.shape).astype(np.float32)))


# ======================================================================= stream Q(lambda), MinAtar Breakout-v1
def run_q():
    T, EF = 5000, 0.2   # eps 1 -> 0.01 over 1000 steps: the 600 recorded steps mix greedy / non-greedy / cut steps
    torch.manual_seed(SEED); np.random.seed(SEED)
    rec = Recorder(gym.make("MinAtar/Breakout-v1"))
    env = ScaleReward(NormalizeObservation(gym.wrappers.RecordEpisodeStatistics(rec)), gamma=0.99)  # = main
    agent = sqm.StreamQ(n_channels=env.observation_space.shape[-1], n_actions=env.action_space.n, lr=1.0, gamma=0.99,
                        lamda=0.8, epsilon_target=0.01, epsilon_start=1.0, exploration_fraction=EF, total_steps=T,
                        kappa_value=2.0)
    w0 = copy.deepcopy(agent.network.state_dict())
    put("q/w0", agent.network)
    deltas = []
    hook_delta(agent.optimizer, deltas)
    s, _ = env.reset(seed=SEED)
    R = dict(s_in=[], s2_in=[], a=[], ng=[], eps=[], obs2_raw=[], r_raw=[], term=[], trunc=[], r_in=[], reset_raw=[],
             ep_ret=[])
    F["q/reset0_raw"] = rec.reset_obs
    for t in range(1, N_Q + 1):
        a, ng = agent.sample_action(s)
        a = int(a)
        s2, r, te, tr, info = env.step(a)
        agent.update_params(s, a, r, s2, te or tr, ng)
        R["s_in"].append(np.asarray(s, np.float64)); R["s2_in"].append(np.asarray(s2, np.float64))
        R["a"].append(a); R["ng"].append(bool(ng)); R["eps"].append(agent.epsilon)
        o_raw, r_raw, te_raw, tr_raw = rec.last
        R["obs2_raw"].append(o_raw); R["r_raw"].append(r_raw); R["term"].append(te_raw); R["trunc"].append(tr_raw)
        R["r_in"].append(float(r))
        s = s2
        if te or tr:
            R["ep_ret"].append(float(info["episode"]["r"][0]))
            s, _ = env.reset()
            R["reset_raw"].append(rec.reset_obs)
        else:
            R["reset_raw"].append(np.zeros_like(o_raw))
        if t in SNAPS:
            put(f"q/w{t}", agent.network)
    put(f"q/w{N_Q}", agent.network)
    traces(f"q/z{N_Q}", agent.optimizer)
    for k, v in R.items():
        F[f"q/{k}"] = np.array(v)
    F["q/delta"] = np.array(deltas, np.float64)
    F["q/T"], F["q/eps_frac"] = np.int64(T), np.float64(EF)

    # replay (unperturbed = harness check; perturbed = float32-rounding sensitivity of the official code itself)
    for tag, rel in (("replay", 0.0), ("pert", 1e-7)):
        ag2 = sqm.StreamQ(n_channels=4, n_actions=3, total_steps=T, exploration_fraction=EF)
        ag2.network.load_state_dict(w0)
        perturb(ag2.network, rel, np.random.default_rng(99))
        d2 = []
        hook_delta(ag2.optimizer, d2)
        for k in range(N_Q):
            ag2.update_params(R["s_in"][k], R["a"][k], np.float64(R["r_in"][k]), R["s2_in"][k],
                              R["term"][k] or R["trunc"][k], R["ng"][k])
        F[f"q/{tag}_delta"] = np.array(d2, np.float64)
        put(f"q/{tag}_w{N_Q}", ag2.network)
    print(f"[INFO] Q: {N_Q} steps, {len(R['ep_ret'])} episodes, {int(np.sum(R['ng']))} non-greedy, "
          f"replay max|d delta| {np.max(np.abs(F['q/replay_delta'] - F['q/delta'])):.3e}, "
          f"pert max|d delta| {np.max(np.abs(F['q/pert_delta'] - F['q/delta'])):.3e}")


# ======================================================================= stream AC(lambda), MuJoCo Hopper-v4
def run_ac():
    torch.manual_seed(SEED); np.random.seed(SEED)
    rec = Recorder(gym.wrappers.FlattenObservation(gym.make("Hopper-v4")))
    env = AddTimeInfo(NormalizeObservation(ScaleReward(gym.wrappers.ClipAction(
        gym.wrappers.RecordEpisodeStatistics(rec)), gamma=0.99)))                       # = main order
    agent = sac.StreamAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], lr=1.0,
                         gamma=0.99, lamda=0.8, kappa_policy=3.0, kappa_value=2.0)
    w0p, w0v = copy.deepcopy(agent.policy_net.state_dict()), copy.deepcopy(agent.value_net.state_dict())
    put("ac/pi0", agent.policy_net); put("ac/v0", agent.value_net)
    deltas = []
    hook_delta(agent.optimizer_value, deltas)
    s, _ = env.reset(seed=SEED)
    F["ac/reset0_raw"] = rec.reset_obs
    R = dict(s_in=[], s2_in=[], a=[], obs2_raw=[], r_raw=[], term=[], trunc=[], r_in=[], reset_raw=[], ep_ret=[])
    for t in range(1, N_AC + 1):
        a = agent.sample_action(s)
        s2, r, te, tr, info = env.step(a)
        agent.update_params(s, a, r, s2, te or tr, 0.01)
        R["s_in"].append(np.asarray(s, np.float64)); R["s2_in"].append(np.asarray(s2, np.float64))
        R["a"].append(np.asarray(a, np.float32))
        o_raw, r_raw, te_raw, tr_raw = rec.last
        R["obs2_raw"].append(o_raw); R["r_raw"].append(r_raw); R["term"].append(te_raw); R["trunc"].append(tr_raw)
        R["r_in"].append(float(r))
        s = s2
        if te or tr:
            R["ep_ret"].append(float(info["episode"]["r"][0]))
            s, _ = env.reset()
            R["reset_raw"].append(rec.reset_obs)
        else:
            R["reset_raw"].append(np.zeros_like(o_raw))
        if t in SNAPS:
            put(f"ac/pi{t}", agent.policy_net); put(f"ac/v{t}", agent.value_net)
    put(f"ac/pi{N_AC}", agent.policy_net); put(f"ac/v{N_AC}", agent.value_net)
    traces(f"ac/zpi{N_AC}", agent.optimizer_policy); traces(f"ac/zv{N_AC}", agent.optimizer_value)
    for k, v in R.items():
        F[f"ac/{k}"] = np.array(v)
    F["ac/delta"] = np.array(deltas, np.float64)
    for tag, rel in (("replay", 0.0), ("pert", 1e-7)):
        ag2 = sac.StreamAC(n_obs=12, n_actions=3)
        ag2.policy_net.load_state_dict(w0p); ag2.value_net.load_state_dict(w0v)
        g = np.random.default_rng(99)
        perturb(ag2.policy_net, rel, g); perturb(ag2.value_net, rel, g)
        d2 = []
        hook_delta(ag2.optimizer_value, d2)
        for k in range(N_AC):
            ag2.update_params(R["s_in"][k], R["a"][k], np.float64(R["r_in"][k]), R["s2_in"][k],
                              R["term"][k] or R["trunc"][k], 0.01)
        F[f"ac/{tag}_delta"] = np.array(d2, np.float64)
        put(f"ac/{tag}_pi{N_AC}", ag2.policy_net); put(f"ac/{tag}_v{N_AC}", ag2.value_net)
    # official wrappers + agent on the SAME stream with the raw MuJoCo obs rounded to float32 (our Brax proxy and our
    # replay feed float32 obs): isolates the input-precision effect from implementation differences
    class ReplayEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (11,), np.float64)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
            self.spec = gym.envs.registration.EnvSpec("Replay-v0", max_episode_steps=1000)
            self.k, self.first = 0, True

        def reset(self, seed=None, options=None):
            o = F["ac/reset0_raw"] if self.first else R["reset_raw"][self.k - 1]
            self.first = False
            return np.asarray(o, np.float32).astype(np.float64), {}

        def step(self, a):
            k = self.k
            self.k += 1
            return (np.asarray(R["obs2_raw"][k], np.float32).astype(np.float64), R["r_raw"][k], R["term"][k],
                    R["trunc"][k], {})

    env3 = AddTimeInfo(NormalizeObservation(ScaleReward(gym.wrappers.ClipAction(
        gym.wrappers.RecordEpisodeStatistics(ReplayEnv())), gamma=0.99)))
    ag3 = sac.StreamAC(n_obs=12, n_actions=3)
    ag3.policy_net.load_state_dict(w0p); ag3.value_net.load_state_dict(w0v)
    d3, s3 = [], []
    hook_delta(ag3.optimizer_value, d3)
    s, _ = env3.reset()
    for k in range(N_AC):
        s2, r, te, tr, _ = env3.step(R["a"][k])
        ag3.update_params(s, R["a"][k], r, s2, te or tr, 0.01)
        s3.append(np.asarray(s, np.float64))
        s = s2
        if te or tr:
            s, _ = env3.reset()
    F["ac/f32in_delta"], F["ac/f32in_s_in"] = np.array(d3, np.float64), np.array(s3)
    put(f"ac/f32in_pi{N_AC}", ag3.policy_net); put(f"ac/f32in_v{N_AC}", ag3.value_net)
    print(f"[INFO] AC: float32-rounded raw obs => official max|d s_in| {np.max(np.abs(F['ac/f32in_s_in'] - F['ac/s_in'])):.3e}"
          f", max|d delta| {np.max(np.abs(F['ac/f32in_delta'] - F['ac/delta'])):.3e}")
    print(f"[INFO] AC: {N_AC} steps, {len(R['ep_ret'])} episodes, "
          f"replay max|d delta| {np.max(np.abs(F['ac/replay_delta'] - F['ac/delta'])):.3e}, "
          f"pert max|d delta| {np.max(np.abs(F['ac/pert_delta'] - F['ac/delta'])):.3e}")


# ======================================================================= official metric (plot.py) on fixed episodes
def run_metric():
    import plot   # their avg_return_curve (imports matplotlib only)
    rng = np.random.default_rng(5)
    xs, ys = [], []
    for _ in range(4):      # 4 runs, ragged episode lists, some empty-free bins
        n = int(rng.integers(40, 80))
        x = np.sort(rng.choice(np.arange(1, 400_001), n, replace=False))
        xs.append(x.tolist()); ys.append((rng.standard_normal(n) * 10 + x / 4e4).tolist())
    steps, avg, se = plot.avg_return_curve(xs, ys, 50_000, 400_000)
    F["metric/n"] = np.array([len(x) for x in xs])
    F["metric/x"] = np.concatenate(xs); F["metric/y"] = np.concatenate(ys)
    F["metric/steps"], F["metric/avg"], F["metric/se"] = steps, avg, se


run_q()
run_ac()
run_metric()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, **F)
print(f"[INFO] wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB) keys={len(F)} official_commit={F['commit']} "
      f"torch={torch.__version__}")

"""Dump differential-test fixtures from the UNMODIFIED official code (repo @ e7c19c3, torch 1.13.1 CPU, gym 0.18).

nets_<algo>_<cell>_<arch>.npz : initial weights (torch state_dict names), K=3 update batches in the official
    (T,B,dim) layout, the injected standard-normal noise per update, per-update loss outputs, the critic/actor loss
    values and gradients of update 0 (official algo.critic_loss / actor_loss + autograd.grad), the official
    report_grad_norm() after update 0, and all parameters after each update.
envs.npz : gym 0.18 Pendulum-v0 and gym 0.21 Pendulum-v1 one-step maps and 200-step trajectories; official
    SparsePointEnv+VariBadWrapper and WindEnv rollouts; the official wind list and train/eval split.
Noise injection (labelled): TD3._inject_noise's torch.randn_like and TanhNormal.rsample's N(0,1) draw are replaced
by a queue of pre-drawn tensors so both implementations consume identical noise.  Nothing else is patched.
"""
import os, sys, importlib.util
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
REPO = os.path.join(ROOT, "repo")
sys.path.insert(0, REPO)
import gym
from gym.envs.registration import register, registry
torch.set_num_threads(1)

from policies.models.policy_rnn import ModelFreeOffPolicy_Separate_RNN
from policies.models.policy_rnn_shared import ModelFreeOffPolicy_Shared_RNN
import torchkit.distributions as tdist
import policies.rl.td3 as td3mod

NOISE = []
def _rsample(self, return_pretanh_value=False):
    eps = NOISE.pop(0)
    z = self.normal_mean + self.normal_std * eps
    return (torch.tanh(z), z) if return_pretanh_value else torch.tanh(z)
tdist.TanhNormal.rsample = _rsample
_orig_randn_like = torch.randn_like
def _inject_noise(self, actions):
    eps = NOISE.pop(0)
    action_noise = (eps * self.target_noise).clamp(-self.target_noise_clip, self.target_noise_clip)
    return (actions + action_noise).clamp(-1, 1)
td3mod.TD3._inject_noise = _inject_noise

OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "tests", "fixtures")
T, B, O, A = 6, 3, 1, 1
SIZES = dict(action_embedding_size=4, observ_embedding_size=8, reward_embedding_size=4, rnn_hidden_size=16,
             dqn_layers=[16, 16], policy_layers=[16, 16], lr=3e-4, gamma=0.9, tau=0.005)
ALGO_KW = {"sac": dict(entropy_alpha=0.2, automatic_entropy_tuning=True, alpha_lr=3e-4),
           "td3": dict(exploration_noise=0.1, target_noise=0.2, target_noise_clip=0.5)}


def make_batch(rng):
    obs = rng.normal(size=(T, B, O)).astype(np.float32)
    obs2 = np.concatenate([obs[1:], rng.normal(size=(1, B, O))], 0).astype(np.float32)
    act = rng.uniform(-1, 1, size=(T, B, A)).astype(np.float32)
    rew = rng.normal(size=(T, B, 1)).astype(np.float32)
    term = (rng.uniform(size=(T, B, 1)) < 0.2).astype(np.float32)
    mask = np.ones((T, B, 1), np.float32)
    mask[4:, 1] = 0.0   # sequence 1 ends after 4 transitions
    mask[2:, 2] = 0.0   # sequence 2 ends after 2 transitions
    return dict(obs=obs, obs2=obs2, act=act, rew=rew, term=term, mask=mask)


def sd(m):
    return {k: v.detach().cpu().numpy().copy() for k, v in m.state_dict().items()}


def dump_nets(algo, cell, arch, seed=0):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    kw = dict(obs_dim=O, action_dim=A, encoder=cell, algo_name=algo, **SIZES, **{algo: ALGO_KW[algo]})
    agent = (ModelFreeOffPolicy_Separate_RNN if arch == "separate" else ModelFreeOffPolicy_Shared_RNN)(**kw)
    out = {f"init/{k}": v for k, v in sd(agent).items()}
    for u in range(3):
        b = make_batch(rng)
        noise = [rng.normal(size=(T + 1, B, A)).astype(np.float32) for _ in range(2)]
        for k, v in b.items():
            out[f"b{u}/{k}"] = v
        out[f"b{u}/noise_tgt"], out[f"b{u}/noise_act"] = noise
        tb = {k: torch.from_numpy(v) for k, v in b.items()}
        if u == 0:
            # losses + grads of update 0 via the official loss functions
            observs = torch.cat((tb["obs"][[0]], tb["obs2"]), 0)
            actions = torch.cat((torch.zeros(1, B, A), tb["act"]), 0)
            rewards = torch.cat((torch.zeros(1, B, 1), tb["rew"]), 0)
            dones = torch.cat((torch.zeros(1, B, 1), tb["term"]), 0)
            masks = tb["mask"]
            nv = torch.clamp(masks.sum(), min=1.0)
            if arch == "separate":
                NOISE[:] = [torch.from_numpy(noise[0])]
                (q1, q2), qt = agent.algo.critic_loss(False, False, agent.actor, agent.actor_target, agent.critic,
                                                       agent.critic_target, observs, actions, rewards, dones,
                                                       agent.gamma)
                l1 = ((q1 * masks - qt * masks) ** 2).sum() / nv
                l2 = ((q2 * masks - qt * masks) ** 2).sum() / nv
                names = [n for n, _ in agent.critic.named_parameters()]
                g = torch.autograd.grad(l1 + l2, list(agent.critic.parameters()))
                out["g0/critic_loss"] = (l1 + l2).item()
                for n, gg in zip(names, g):
                    out[f"g0/critic/{n}"] = gg.numpy().copy()
                NOISE[:] = [torch.from_numpy(noise[1])]
                pl, lp = agent.algo.actor_loss(False, False, agent.actor, agent.actor_target, agent.critic,
                                               agent.critic_target, observs, actions, rewards)
                pl = (pl * masks).sum() / nv
                names = [n for n, _ in agent.actor.named_parameters()]
                g = torch.autograd.grad(pl, list(agent.actor.parameters()))
                out["g0/actor_loss"] = pl.item()
                for n, gg in zip(names, g):
                    out[f"g0/actor/{n}"] = gg.numpy().copy()
        NOISE[:] = [torch.from_numpy(noise[0]), torch.from_numpy(noise[1])]
        info = agent.update(tb)
        assert not NOISE or algo == "td3" and arch == "separate" or algo == "td3", f"unused noise {len(NOISE)}"
        NOISE[:] = []
        for k, v in info.items():
            out[f"u{u}/info/{k}"] = np.float64(v)
        if u == 0:
            for k, v in agent.report_grad_norm().items():
                out[f"u0/gradnorm/{k}"] = np.float64(v)
        for k, v in sd(agent).items():
            out[f"u{u}/params/{k}"] = v
    np.savez_compressed(os.path.join(OUT, f"nets_{algo}_{cell}_{arch}.npz"), **out)
    print("nets", algo, cell, arch, {k: out[k] for k in out if "/info/" in k and k.startswith("u0")})


def load_v1():
    p = os.path.join(ROOT, "gym021", "gym-0.21.0", "gym", "envs", "classic_control", "pendulum.py")
    spec = importlib.util.spec_from_file_location("pendulum_v1", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.PendulumEnv


def dump_envs():
    from gym.envs.classic_control.pendulum import PendulumEnv as P0
    P1 = load_v1()
    rng = np.random.RandomState(0)
    out = {}
    for name, cls in (("v0", P0), ("v1", P1)):
        env = cls()
        # one-step map on random states (incl. near-clip speeds) and torques
        n = 2000
        th = rng.uniform(-3 * np.pi, 3 * np.pi, n); thd = rng.uniform(-8, 8, n); u = rng.uniform(-2.5, 2.5, n)
        nth, nthd, r = np.zeros(n), np.zeros(n), np.zeros(n)
        for i in range(n):
            env.state = np.array([th[i], thd[i]])
            obs, rew, done, _ = env.step(np.array([u[i]]))
            nth[i], nthd[i], r[i] = env.state[0], env.state[1], rew
        out.update({f"{name}/th": th, f"{name}/thd": thd, f"{name}/u": u, f"{name}/nth": nth,
                    f"{name}/nthd": nthd, f"{name}/r": r})
        # trajectories: 10 x 200 steps from official resets, random torques
        env.seed(123)
        trs = []
        for e in range(10):
            o = env.reset(); s0 = env.state.copy()
            acts = rng.uniform(-2, 2, (200, 1)); obs_l = [o]; rew_l = []
            for t in range(200):
                o, rr, _, _ = env.step(acts[t]); obs_l.append(o); rew_l.append(rr)
            out[f"{name}/traj{e}/s0"] = s0; out[f"{name}/traj{e}/act"] = acts
            out[f"{name}/traj{e}/obs"] = np.array(obs_l); out[f"{name}/traj{e}/rew"] = np.array(rew_l)
    # official point robot + VariBAD wrapper
    sys.path.insert(0, REPO)
    from envs.meta.toy_navigation.point_robot import SparsePointEnv
    from envs.meta.wrappers import VariBadWrapper
    np.random.seed(5)
    base = SparsePointEnv(max_episode_steps=60, n_tasks=4, goal_radius=0.2)
    env = VariBadWrapper(base, episodes_per_task=2)
    for e in range(4):
        o = env.reset(task=e)
        goal = base._goal.copy(); s0 = base._state.copy()
        acts = rng.uniform(-1.3, 1.3, (120, 2)) * (1.0 if e % 2 == 0 else 0.15)
        # steer some episodes toward the goal so the goal/terminal path is exercised
        obs_l, rew_l, term_l, state_l, done_l = [o], [], [], [], []
        for t in range(120):
            a = acts[t]
            if e >= 2:
                d = goal - base._state
                a = np.clip(d / 0.1, -1, 1) * 0.9
                acts[t] = a
            o, r, done, info = env.step(a)
            obs_l.append(o); rew_l.append(r); term_l.append(float(base.is_goal_state()))
            state_l.append(base._state.copy()); done_l.append(done)
        out.update({f"pr{e}/goal": goal, f"pr{e}/s0": s0, f"pr{e}/act": acts, f"pr{e}/obs": np.array(obs_l),
                    f"pr{e}/rew": np.array(rew_l), f"pr{e}/term": np.array(term_l),
                    f"pr{e}/state_after": np.array(state_l), f"pr{e}/done": np.array(done_l)})
    # official wind env: wind list and the learner's train/eval split (np.random.permutation after construction)
    from envs.meta.make_env import make_env
    np.random.seed(0)
    wenv = make_env("Wind-v0", 1, seed=0, n_tasks=60)
    out["wind/winds"] = np.array(wenv.unwrapped.winds)
    out["wind/perm"] = np.random.permutation(wenv.unwrapped.get_all_task_idx())
    wb = wenv.unwrapped
    wo = wenv.reset(task=3)
    acts = rng.uniform(-1, 1, (75, 2)); wobs, wrew = [wo], []
    for t in range(75):
        d = np.array([0.0, 1.0]) - wb._state
        a = np.clip(d / 0.1, -1, 1) if t > 5 else acts[t]
        acts[t] = a
        o, r, done, info = wenv.step(a); wobs.append(o); wrew.append(r)
    out.update({"wind/task3_act": acts, "wind/task3_obs": np.array(wobs), "wind/task3_rew": np.array(wrew)})
    np.savez_compressed(os.path.join(OUT, "envs.npz"), **out)
    print("envs ok", len(out))


if __name__ == "__main__":
    dump_envs()
    for algo in ("sac", "td3"):
        for cell in ("lstm", "gru"):
            for arch in ("separate", "shared"):
                dump_nets(algo, cell, arch)

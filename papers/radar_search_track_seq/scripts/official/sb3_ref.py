"""Learner-side reference fixtures from the AUTHORS' TOOLCHAIN (ORACLE; NOT our implementation).

The authors released no code (README). Their GitHub account carries forks of stable-baselines3 (2.4.0a8,
commit e46a6d5, 15 commits ahead of upstream: a TensorBoard-histogram fix only) and imitation (commit f5cb8a4, one
commit ahead: a HuggingFace path fix). Those forks, installed from the author's repositories, are the closest thing
to their learner code. This harness drives them (plus torch's GRU / MultiheadAttention / LayerNorm, which an SB3
features extractor would be built from) and records inputs and outputs as .npz fixtures that
tests/test_official_parity.py replays through our JAX code.

Run ONLY with the isolated oracle venv (torch 2.3.1, numpy 1.26.4, gymnasium 0.29.1):
    $OFFICIAL_ROOT/.venv-sb3/bin/python -I scripts/official/sb3_ref.py <out_dir>
"""
import json
import os
import sys

import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
import imitation
from imitation.algorithms import bc as imitation_bc
from imitation.data import types as imitation_types

th.set_num_threads(1)
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
N_A = 19


def sd_np(module):
    return {k: v.detach().cpu().numpy().astype(np.float32).copy() for k, v in module.state_dict().items()}


def flat_named(prefix, d):
    return {f"{prefix}{k}": v for k, v in d.items()}


# ----------------------------------------------------------------------------- 1. PPO.train + collect_rollouts
class ToyEnv(gym.Env):
    """Random 12-D observations, MultiDiscrete([19, 19]) action, a deterministic action-dependent reward plus noise.
    A TimeLimit of 7 steps makes truncations (and so SB3's timeout bootstrap) occur inside the 24-step rollout."""
    observation_space = spaces.Box(-5.0, 5.0, (12,), np.float32)
    action_space = spaces.MultiDiscrete([N_A, N_A])

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return self.rng.normal(size=12).astype(np.float32), {}

    def step(self, a):
        r = -(((a[0] - 9) ** 2 + (a[1] - 5) ** 2) / 50.0) + 0.3 * self.rng.normal()
        return self.rng.normal(size=12).astype(np.float32), float(r), False, False, {}


def ppo_fixture():
    venv = DummyVecEnv([lambda: gym.wrappers.TimeLimit(ToyEnv(), 7) for _ in range(2)])
    model = PPO("MlpPolicy", venv, n_steps=24, batch_size=48, n_epochs=3, learning_rate=0.03, ent_coef=0.01,
                vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, gae_lambda=0.95, clip_range=0.2, seed=3, device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])))
    pol = model.policy
    rec = dict(params0=sd_np(pol))
    steps = []          # one entry per optimizer step: params before, grads pre-clip, grads post-clip, params after

    orig_clip = th.nn.utils.clip_grad_norm_

    def clip_spy(parameters, max_norm, *a, **k):
        params = list(parameters)
        st = dict(pre={n: p.grad.detach().numpy().copy() for (n, _), p in zip(pol.named_parameters(), params)},
                  before={n: p.detach().numpy().copy() for (n, _), p in zip(pol.named_parameters(), params)})
        norm = orig_clip(params, max_norm, *a, **k)
        st["post"] = {n: p.grad.detach().numpy().copy() for (n, _), p in zip(pol.named_parameters(), params)}
        st["norm"] = float(norm)
        steps.append(st)
        return norm
    th.nn.utils.clip_grad_norm_ = clip_spy

    # raw rewards / truncation flags / terminal observations, as the VecEnv returns them
    raw = dict(r=[], done=[], trunc=[], term_obs=[])
    orig_step_wait = venv.step_wait

    def step_wait():
        obs, r, d, infos = orig_step_wait()
        raw["r"].append(r.copy())
        raw["done"].append(d.copy())
        raw["trunc"].append(np.array([i.get("TimeLimit.truncated", False) for i in infos]))
        raw["term_obs"].append(np.stack([i.get("terminal_observation", np.zeros(12, np.float32)) for i in infos]))
        return obs, r, d, infos
    venv.step_wait = step_wait

    gae_args = {}
    orig_cr = RolloutBuffer.compute_returns_and_advantage

    def cr_spy(self, last_values, dones):
        gae_args["last_values"] = last_values.detach().numpy().ravel().copy()
        gae_args["dones"] = np.asarray(dones, np.float32).copy()
        return orig_cr(self, last_values, dones)
    RolloutBuffer.compute_returns_and_advantage = cr_spy

    logs = {}

    model.learn(total_timesteps=48)
    th.nn.utils.clip_grad_norm_ = orig_clip
    RolloutBuffer.compute_returns_and_advantage = orig_cr
    for k in ("train/entropy_loss", "train/policy_gradient_loss", "train/value_loss", "train/approx_kl",
              "train/clip_fraction", "train/loss"):
        logs[k] = float(model.logger.name_to_value.get(k, np.nan))

    buf = model.rollout_buffer
    with th.no_grad():                                       # V(terminal obs) under the (unchanged) rollout policy
        pol0 = ActorCriticPolicy(venv.observation_space, venv.action_space, lambda _: 0.0,
                                 net_arch=dict(pi=[64, 64], vf=[64, 64]))
        pol0.load_state_dict({k: th.as_tensor(v) for k, v in rec["params0"].items()})
        to = np.stack(raw["term_obs"])                       # [T, E, 12]
        v_term = pol0.predict_values(th.as_tensor(to.reshape(-1, 12))).numpy().reshape(to.shape[:2])
    out = dict(obs=buf.observations, actions=buf.actions, rewards_buf=buf.rewards, episode_starts=buf.episode_starts,
               values=buf.values, log_probs=buf.log_probs, advantages=buf.advantages, returns=buf.returns,
               raw_r=np.stack(raw["r"]), raw_done=np.stack(raw["done"]).astype(np.float32),
               raw_trunc=np.stack(raw["trunc"]).astype(np.float32), v_term=v_term,
               last_values=gae_args["last_values"], last_dones=gae_args["dones"],
               hp=np.array([0.03, 0.01, 0.5, 0.5, 0.99, 0.95, 0.2, 1e-5]),   # lr ent vf maxgn gamma lam clip eps
               n_opt_steps=np.array(len(steps)))
    out.update(flat_named("p0/", rec["params0"]))
    for i, st in enumerate(steps):
        out.update(flat_named(f"s{i}/before/", st["before"]))
        out.update(flat_named(f"s{i}/pre/", st["pre"]))
        out.update(flat_named(f"s{i}/post/", st["post"]))
        out[f"s{i}/norm"] = np.array(st["norm"])
    out.update(flat_named("pend/", sd_np(pol)))
    for k, v in logs.items():
        out["log/" + k] = np.array(v)
    np.savez_compressed(os.path.join(OUT, "sb3_ppo.npz"), **out)
    return dict(n_opt_steps=len(steps), norms=[st["norm"] for st in steps], logs=logs,
                n_trunc=int(np.stack(raw["trunc"]).sum()))


# ----------------------------------------------------------------------------- 2. imitation BC loss + trainer steps
def bc_fixture():
    obs_space = spaces.Box(-5.0, 5.0, (12,), np.float32)
    act_space = spaces.MultiDiscrete([N_A, N_A])
    th.manual_seed(11)
    pol = ActorCriticPolicy(obs_space, act_space, lambda _: 3e-4, net_arch=dict(pi=[64, 64], vf=[64, 64]))
    rng = np.random.default_rng(5)
    n = 96
    obs = rng.normal(size=(n, 12)).astype(np.float32)
    acts = rng.integers(0, N_A, size=(n, 2))
    p0 = sd_np(pol)
    calc = imitation_bc.BehaviorCloningLossCalculator(ent_weight=1e-3, l2_weight=0.0)
    pol.zero_grad()
    m = calc(pol, obs[:32], acts[:32])
    m.loss.backward()
    g0 = {n_: p.grad.detach().numpy().copy() for n_, p in pol.named_parameters()}
    single = dict(loss=float(m.loss), neglogp=float(m.neglogp), entropy=float(m.entropy),
                  prob_true_act=float(m.prob_true_act))

    # trainer: 3 batches of 32 (shuffled by imitation's DataLoader); record each batch it actually used
    pol.load_state_dict({k: th.as_tensor(v) for k, v in p0.items()})
    trans = imitation_types.Transitions(obs=obs, acts=acts, infos=np.array([{}] * n), next_obs=obs,
                                        dones=np.zeros(n, bool))
    trainer = imitation_bc.BC(observation_space=obs_space, action_space=act_space, rng=np.random.default_rng(7),
                              policy=pol, demonstrations=trans, batch_size=32, device="cpu")
    used = []
    real = trainer.loss_calculator

    def spy(policy, o, a):
        used.append((np.asarray(o, np.float32).copy(), np.asarray(a).copy()))
        return real(policy, o, a)
    trainer.loss_calculator = spy
    trainer.train(n_batches=3, log_interval=10 ** 9, progress_bar=False)
    out = dict(obs=obs, acts=acts, n_batches=np.array(len(used)),
               adam=np.array([trainer.optimizer.defaults["lr"], trainer.optimizer.defaults["eps"]]))
    out.update(flat_named("p0/", p0))
    out.update(flat_named("g0/", g0))
    for k, v in single.items():
        out["m/" + k] = np.array(v)
    for i, (o, a) in enumerate(used):
        out[f"b{i}/obs"], out[f"b{i}/acts"] = o, a
    out.update(flat_named("pend/", sd_np(pol)))
    np.savez_compressed(os.path.join(OUT, "imitation_bc.npz"), **out)
    return dict(single=single, n_batches=len(used), adam=out["adam"].tolist())


# ----------------------------------------------------------------------------- 3. torch MHA + residual + LayerNorm
def mha_fixture():
    th.manual_seed(21)
    d, h, K, B = 64, 4, 17, 6
    mha = th.nn.MultiheadAttention(d, h, batch_first=True)
    ln = th.nn.LayerNorm(d)                      # torch default eps 1e-5
    with th.no_grad():                           # non-trivial biases / LN affine
        mha.in_proj_bias.normal_(0, 0.1)
        mha.out_proj.bias.normal_(0, 0.1)
        ln.weight.normal_(1, 0.1)
        ln.bias.normal_(0, 0.1)
    rng = np.random.default_rng(22)
    e = rng.normal(size=(B, K, d)).astype(np.float32)
    nv = np.array([1, 3, 7, 12, 16, 17])
    valid = np.arange(K)[None] < nv[:, None]
    with th.no_grad():
        a, _ = mha(th.as_tensor(e), th.as_tensor(e), th.as_tensor(e), key_padding_mask=th.as_tensor(~valid),
                   need_weights=False)
        y = ln(th.as_tensor(e) + a).numpy()
    out = dict(e=e, valid=valid, y=y, ln_eps=np.array(ln.eps))
    out.update(flat_named("mha/", sd_np(mha)))
    out.update(flat_named("ln/", sd_np(ln)))
    np.savez_compressed(os.path.join(OUT, "torch_mha.npz"), **out)
    return dict(ln_eps=ln.eps)


# ----------------------------------------------------------------------------- 4. torch bidirectional GRU
def gru_fixture():
    th.manual_seed(31)
    D, H, K, B = 7, 64, 17, 6
    gru = th.nn.GRU(D, H, batch_first=True, bidirectional=True)
    rng = np.random.default_rng(32)
    nv = np.array([1, 2, 5, 9, 16, 17])
    valid = np.arange(K)[None] < nv[:, None]
    x = (rng.uniform(-1, 1, size=(B, K, D)) * valid[..., None]).astype(np.float32)   # zero-padded rows
    with th.no_grad():
        packed = th.nn.utils.rnn.pack_padded_sequence(th.as_tensor(x), th.as_tensor(nv), batch_first=True,
                                                      enforce_sorted=False)
        _, hn = gru(packed)
        h_packed = th.cat([hn[0], hn[1]], -1).numpy()
        _, hn2 = gru(th.as_tensor(x))                                  # padding fed through ("zero rows" reading)
        h_unpacked = th.cat([hn2[0], hn2[1]], -1).numpy()
    out = dict(x=x, valid=valid, h_packed=h_packed, h_unpacked=h_unpacked)
    out.update(flat_named("gru/", sd_np(gru)))
    np.savez_compressed(os.path.join(OUT, "torch_gru.npz"), **out)
    return {}


# ----------------------------------------------------------------------------- 5. SB3 NatureCNN on a 1x48x48 raster
def cnn_fixture():
    th.manual_seed(41)
    sp = spaces.Box(0.0, 1.0, (1, 48, 48), np.float32)
    cnn = NatureCNN(sp, features_dim=128, normalized_image=True)
    rng = np.random.default_rng(42)
    img = rng.uniform(0, 1, size=(5, 1, 48, 48)).astype(np.float32)
    with th.no_grad():
        y = cnn(th.as_tensor(img)).numpy()
    out = dict(img=img, y=y)
    out.update(flat_named("cnn/", sd_np(cnn)))
    np.savez_compressed(os.path.join(OUT, "sb3_naturecnn.npz"), **out)
    return {}


# ----------------------------------------------------------------------------- 6. which modules SB3 re-initialises
class TLExtractor(BaseFeaturesExtractor):
    """A track-list extractor built the way an SB3 user must build it (torch modules): Linear embed -> MHA ->
    LayerNorm -> GRU, plus a Linear output. Used only to observe SB3's init semantics."""

    def __init__(self, space):
        super().__init__(space, 64)
        self.embed = th.nn.Linear(12, 64)
        self.mha = th.nn.MultiheadAttention(64, 4, batch_first=True)
        self.ln = th.nn.LayerNorm(64)
        self.gru = th.nn.GRU(64, 32, batch_first=True, bidirectional=True)
        self.out = th.nn.Linear(64, 64)

    def forward(self, x):
        return self.out(x)


def init_semantics():
    sp = spaces.Box(-5.0, 5.0, (12,), np.float32)
    th.manual_seed(51)
    pol = ActorCriticPolicy(sp, spaces.MultiDiscrete([N_A, N_A]), lambda _: 3e-4,
                            features_extractor_class=TLExtractor, ortho_init=True)
    fe = pol.features_extractor
    res = {}
    for name, p in fe.named_parameters():
        w = p.detach().numpy()
        if w.ndim == 2:
            g = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
            gain2 = float(np.mean(np.diag(g)))
            ortho = bool(np.allclose(g, np.eye(len(g)) * gain2, atol=1e-4))
            res[name] = dict(shape=list(w.shape), orthogonal=ortho, gain=float(np.sqrt(gain2)))
        else:
            res[name] = dict(shape=list(w.shape), all_zero=bool(np.all(w == 0)),
                             all_one=bool(np.all(w == 1)))
    res["out_proj_is_linear_subclass"] = isinstance(fe.mha.out_proj, th.nn.Linear)
    return res


def main():
    summary = dict(versions=dict(torch=th.__version__, sb3=stable_baselines3.__version__, gymnasium=gym.__version__,
                                 numpy=np.__version__, imitation=getattr(imitation, "__version__", "fork f5cb8a4")),
                   ppo=ppo_fixture(), bc=bc_fixture(), mha=mha_fixture(), gru=gru_fixture(), cnn=cnn_fixture(),
                   init_semantics=init_semantics())
    with open(os.path.join(OUT, "sb3_ref_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()

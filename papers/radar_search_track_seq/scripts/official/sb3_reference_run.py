"""REFERENCE RUN with the authors' own learner toolchain on our environment (ORACLE side; not our implementation).

The authors released no code. Their GitHub forks of stable-baselines3 (2.4.0a8 @ e46a6d5) and imitation (@ f5cb8a4)
are the closest thing to their learner. This script drives OUR JAX environment (src/data.py, imported read-only,
run on CPU under the oracle venv's jax 0.7.1; trajectory equality with jax 0.11.1 is checked by --dump-env) from an
SB3 VecEnv and trains the paper's 'flatten' arm exactly as an SB3 user would build it:

    MultiInputPolicy + CombinedExtractor: track list (zero-padded K x 7 rows) -> Flatten; scan-history raster
    (1 x 48 x 48) -> NatureCNN(cnn_output_dim); tanh MLPs pi / vf; MultiDiscrete([19, 19]) action.
    imitation BC (ent_weight 1e-3, batch 32, Adam 1e-3, 5 epochs) of the uniform-random teacher on 50 episodes,
    then SB3 PPO defaults (n_steps 2048 x 8 envs, batch 64, 10 epochs, lr 3e-4, clip 0.2, ...), TimeLimit
    truncation at T = 1000 with SB3's own timeout bootstrap.

Two settings: --setting paper   net_arch 128-128 (paper §3 '2 x 128'), NatureCNN 128 (paper §3)
              --setting sb3     the library defaults: net_arch 64-64, NatureCNN 256
Scenarios / teacher actions use OUR key tree for arm 'flat' (lane key fold_in(fold_in(PRNGKey(15), 1), seed)), so the
training scenarios equal those of our PPO-Flat lane with the same seed. Evaluation: the first 32 small-scale scenarios
(fold_in(root, 0x6A72)), stochastic actions, same per-episode reductions as src/eval.py.

    $OFFICIAL_ROOT/.venv-sb3/bin/python -I scripts/official/sb3_reference_run.py \
        --setting paper --seed 0 --steps 100000 --out runs/official/sb3_ref
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import torch as th  # noqa: E402
from gymnasium import spaces  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import VecEnv  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from imitation.algorithms import bc as imitation_bc  # noqa: E402
from imitation.data import types as itypes  # noqa: E402

from src.configs.config import EnvConfig  # noqa: E402
from src import data  # noqa: E402

K_SMALL = 0x6A72
ARM_FLAT = 1


def lane_key(seed):
    return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(15), ARM_FLAT), seed)


def train_scen_key(k_lane, e, ep):
    return jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(k_lane, 3), e), ep)


def make_fns(cfg):
    def obs_dict(s):
        o = data.observe(s, cfg)
        tl = (o.tl * o.tl_mask[:, None]).astype(jnp.float32)
        img = data.render_raster(o.sv_az, o.sv_el, o.sv_age, o.sv_valid, cfg)[None].astype(jnp.float32)
        return tl, img

    def step(s, a):
        s2, _, info = data.step(s, a, cfg)
        tl, img = obs_dict(s2)
        out = dict(r_sv=info["r_sv"], rtl=info["rtl_flip"], cov=info["cov_sum"], ntr=info["n_tracks"])
        if cfg.gospa_metrics:
            out.update(gd=info["gospa_d"], gm=info["gospa_miss"])
        return s2, tl, img, out

    def reset(k):
        s = data.reset(k, cfg)
        tl, img = obs_dict(s)
        return s, tl, img
    return jax.jit(jax.vmap(step)), jax.jit(jax.vmap(reset))


class JaxVecEnv(VecEnv):
    """n_envs copies of our environment stepped in one vmapped jit call; episodes truncate at T (SB3 TimeLimit
    semantics: done=True, info['TimeLimit.truncated']=True, info['terminal_observation']), then reset to the next
    training scenario of that env (our key tree)."""

    def __init__(self, cfg, n_envs, k_lane):
        self.cfg, self.k_lane = cfg, k_lane
        obs_space = spaces.Dict({"raster": spaces.Box(0.0, np.inf, (1, cfg.raster, cfg.raster), np.float32),
                                 "tl": spaces.Box(-np.inf, np.inf, (cfg.k_max, 7), np.float32)})
        super().__init__(n_envs, obs_space, spaces.MultiDiscrete([cfg.n_a, cfg.n_a]))
        self._step, self._reset = make_fns(cfg)
        self.ep = 0
        self.t = 0
        self.actions = None

    def _keys(self):
        return jnp.stack([train_scen_key(self.k_lane, e, self.ep) for e in range(self.num_envs)])

    def reset(self):
        self.s, tl, img = self._reset(self._keys())
        self.t = 0
        return {"raster": np.asarray(img), "tl": np.asarray(tl)}

    def step_async(self, actions):
        self.actions = np.asarray(actions, np.int32)

    def step_wait(self):
        self.s, tl, img, out = self._step(self.s, jnp.asarray(self.actions))
        self.t += 1
        r = np.asarray(out["r_sv"] + out["rtl"], np.float32)
        obs = {"raster": np.asarray(img), "tl": np.asarray(tl)}
        if self.t == self.cfg.T:
            infos = [{"TimeLimit.truncated": True, "terminal_observation": {k: v[i] for k, v in obs.items()}}
                     for i in range(self.num_envs)]
            self.ep += 1
            obs = self.reset()
            return obs, r, np.ones(self.num_envs, bool), infos
        return obs, r, np.zeros(self.num_envs, bool), [{} for _ in range(self.num_envs)]

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kw):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        return [None] * self.num_envs


def bc_transitions(cfg, k_lane, n_ep):
    """Our bc_dataset (src/train.py): n_ep episodes of the uniform-random teacher on fold_in(k_bc, 0) scenarios with
    actions randint(fold_in(fold_in(fold_in(k_bc, 1), e), t))."""
    k_bc = jax.random.fold_in(k_lane, 2)
    step, reset = make_fns(cfg)
    s, tl, img = reset(jnp.stack([jax.random.fold_in(jax.random.fold_in(k_bc, 0), e) for e in range(n_ep)]))
    kas = [jax.random.fold_in(jax.random.fold_in(k_bc, 1), e) for e in range(n_ep)]
    act_fn = jax.jit(jax.vmap(lambda k, t: jax.random.randint(jax.random.fold_in(k, t), (2,), 0, cfg.n_a)
                              .astype(jnp.int32), in_axes=(0, None)))
    kas = jnp.stack(kas)
    O_tl, O_img, A = [], [], []
    for t in range(cfg.T):
        a = act_fn(kas, t)
        O_tl.append(np.asarray(tl)), O_img.append(np.asarray(img).astype(np.float32)), A.append(np.asarray(a))
        s, tl, img, _ = step(s, a)
    tl = np.stack(O_tl, 1).reshape(-1, cfg.k_max, 7)
    img = np.stack(O_img, 1).reshape(-1, 1, cfg.raster, cfg.raster)
    acts = np.stack(A, 1).reshape(-1, 2)
    obs = itypes.DictObs({"raster": img, "tl": tl})
    n = len(acts)
    return itypes.Transitions(obs=obs, acts=acts, infos=np.array([{}] * n), next_obs=obs, dones=np.zeros(n, bool))


def evaluate(model, cfg, n_scen, seed):
    """Stochastic-action episodes on the first n_scen small-scale scenarios; reductions as src/eval.py."""
    ecfg = cfg.__class__(**{**cfg.__dict__, "gospa_metrics": True})
    step, reset = make_fns(ecfg)
    root = jax.random.PRNGKey(15)
    base = jax.random.fold_in(root, K_SMALL)
    s, tl, img = reset(jnp.stack([jax.random.fold_in(base, i) for i in range(n_scen)]))
    th.manual_seed(10_000 + seed)
    acc = {k: [] for k in ("r_sv", "rtl", "cov", "ntr", "gd", "gm")}
    ent = []
    for t in range(cfg.T):
        obs = {"raster": th.as_tensor(np.asarray(img)), "tl": th.as_tensor(np.asarray(tl))}
        with th.no_grad():
            dist = model.policy.get_distribution(obs)
            a = dist.sample().numpy().astype(np.int32)
            if t % 100 == 0:
                ent.append(float(dist.entropy().mean()))
        s, tl, img, out = step(s, jnp.asarray(a))
        for k in acc:
            acc[k].append(np.asarray(out[k]))
    A = {k: np.stack(v, 1) for k, v in acc.items()}                      # [n_scen, T]
    has = A["ntr"] > 0
    m_cov = np.where(has.any(1), (A["cov"] * has).sum(1) / np.maximum(has.sum(1), 1), np.nan)
    ep = dict(sr=A["r_sv"].mean(1), m_cov=m_cov, gospa_d=A["gd"].sum(1), gospa_miss=A["gm"].sum(1),
              mean_tracks=A["ntr"].mean(1), ret_flip=(A["r_sv"] + A["rtl"]).sum(1))
    res = {k: float(np.nanmean(v)) for k, v in ep.items()}
    res["n_no_track"] = int(np.isnan(m_cov).sum())
    res["policy_entropy"] = float(np.mean(ent))
    return res


def dump_env(path, n_steps=300):
    """Trajectory of our env under fixed actions (for the jax 0.7.1 vs 0.11.1 equality check)."""
    cfg = EnvConfig()
    step, reset = make_fns(cfg)
    keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(99), i) for i in range(4)])
    s, tl, img = reset(keys)
    rng = np.random.default_rng(0)
    rec = {k: [] for k in ("tl", "img", "r_sv", "rtl", "cov", "ntr", "gd")}
    for t in range(n_steps):
        a = jnp.asarray(rng.integers(0, cfg.n_a, (4, 2)), jnp.int32)
        s, tl, img, out = step(s, a)
        rec["tl"].append(np.asarray(tl)), rec["img"].append(np.asarray(img))
        for k in ("r_sv", "rtl", "cov", "ntr", "gd"):
            rec[k].append(np.asarray(out[k]))
    np.savez_compressed(path, **{k: np.stack(v) for k, v in rec.items()}, jax_version=np.array(jax.__version__))


def record_update(path, n_steps=128, n_envs=2, batch=64, epochs=2, seed=0):
    """Full-network differential fixture (P10): one SB3 PPO update of the paper-setting Flat policy on a real rollout
    of our env. Records the compact observations (so our ActorCritic renders the raster itself), the buffer, every
    minibatch's index set, and per optimizer step the parameters before and the pre-clip gradients."""
    cfg = EnvConfig(gospa_metrics=False)
    k_lane = lane_key(seed)
    venv = JaxVecEnv(cfg, n_envs, k_lane)
    compact = []                                     # compact Obs of every observation handed to SB3, in step order
    orig_reset, orig_wait = venv.reset, venv.step_wait

    def grab():
        o = jax.vmap(lambda st: data.observe(st, cfg))(venv.s)
        compact.append({k: np.asarray(getattr(o, k)) for k in o._fields})

    def reset():
        r = orig_reset()
        grab()
        return r

    def wait():
        r = orig_wait()
        grab()
        return r
    venv.reset, venv.step_wait = reset, wait
    pk = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]),
              features_extractor_kwargs=dict(cnn_output_dim=128, normalized_image=True))
    model = PPO("MultiInputPolicy", venv, n_steps=n_steps, batch_size=batch, n_epochs=epochs, learning_rate=3e-4,
                seed=seed, device="cpu", policy_kwargs=pk, verbose=0)
    pol = model.policy
    p0 = {k: v.detach().numpy().copy() for k, v in pol.state_dict().items()}
    perms, steps = [], []
    orig_perm = np.random.permutation

    def perm_spy(n):
        idx = orig_perm(n)
        perms.append(np.asarray(idx).copy())
        return idx
    np.random.permutation = perm_spy
    orig_clip = th.nn.utils.clip_grad_norm_

    def clip_spy(parameters, max_norm, *a, **k):
        params = list(parameters)
        names = [n for n, _ in pol.named_parameters()]
        steps.append(dict(before={n: p.detach().numpy().copy() for n, p in zip(names, params)},
                          pre={n: p.grad.detach().numpy().copy() for n, p in zip(names, params)}))
        return orig_clip(params, max_norm, *a, **k)
    th.nn.utils.clip_grad_norm_ = clip_spy
    model.learn(total_timesteps=n_steps * n_envs)
    np.random.permutation, th.nn.utils.clip_grad_norm_ = orig_perm, orig_clip
    buf = model.rollout_buffer
    # compact obs aligned with the buffer: rollout step t uses the observation BEFORE action t (compact[t]); the
    # buffer is env-major after swap_and_flatten -> index e * n_steps + t
    comp = {k: np.stack([c[k] for c in compact[:n_steps]], 0) for k in compact[0]}          # [T, E, ...]
    comp = {k: v.swapaxes(0, 1).reshape((n_steps * n_envs,) + v.shape[2:]) for k, v in comp.items()}
    raster_sb3 = buf.observations["raster"]
    out = dict(actions=buf.actions, log_probs=buf.log_probs, values=buf.values, advantages=buf.advantages,
               returns=buf.returns, raster=raster_sb3, tl_sb3=buf.observations["tl"], batch=np.array(batch),
               n_steps_opt=np.array(len(steps)), perms=np.stack(perms))
    out.update({"obs/" + k: v for k, v in comp.items()})
    out.update({"p0/" + k: v for k, v in p0.items()})
    for i, st in enumerate(steps):
        out.update({f"s{i}/before/{k}": v for k, v in st["before"].items()})
        out.update({f"s{i}/pre/{k}": v for k, v in st["pre"].items()})
    np.savez_compressed(path, **out)
    print("[INFO] recorded", len(steps), "optimizer steps,", len(perms), "permutations", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-update", default=None)
    ap.add_argument("--setting", choices=["paper", "sb3"], default="paper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--bc-episodes", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "official", "sb3_ref"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--dump-env", default=None)
    a = ap.parse_args()
    if a.dump_env:
        dump_env(a.dump_env)
        return
    if a.record_update:
        th.set_num_threads(a.threads)
        record_update(a.record_update)
        return
    th.set_num_threads(a.threads)
    th.manual_seed(a.seed)
    out_dir = os.path.join(a.out, f"{a.setting}_seed{a.seed}")
    os.makedirs(out_dir, exist_ok=True)
    cfg = EnvConfig(gospa_metrics=False)
    k_lane = lane_key(a.seed)
    venv = JaxVecEnv(cfg, 8, k_lane)
    pk = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]),
              features_extractor_kwargs=dict(cnn_output_dim=128, normalized_image=True)) if a.setting == "paper" \
        else dict(features_extractor_kwargs=dict(normalized_image=True))
    model = PPO("MultiInputPolicy", venv, n_steps=2048, batch_size=64, n_epochs=10, learning_rate=3e-4, gamma=0.99,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, seed=a.seed,
                device="cpu", policy_kwargs=pk, verbose=0)
    model.set_logger(configure(out_dir, ["csv"]))
    log = dict(setting=a.setting, seed=a.seed, policy=str(model.policy.__class__.__name__),
               n_params=int(sum(p.numel() for p in model.policy.parameters())), torch_threads=a.threads)
    t0 = time.time()
    eval0 = evaluate(model, cfg, a.n_eval, a.seed)
    print("[METRIC] eval_init", json.dumps(eval0), flush=True)
    trans = bc_transitions(cfg, k_lane, a.bc_episodes)
    t_bc0 = time.time()
    trainer = imitation_bc.BC(observation_space=venv.observation_space, action_space=venv.action_space,
                              rng=np.random.default_rng(a.seed), policy=model.policy, demonstrations=trans,
                              batch_size=32, device="cpu")
    trainer.train(n_epochs=5, log_interval=10 ** 9, progress_bar=False)
    log["bc_seconds"] = time.time() - t_bc0
    del trans
    eval_bc = evaluate(model, cfg, a.n_eval, a.seed)
    print("[METRIC] eval_after_bc", json.dumps(eval_bc), flush=True)
    t_ppo0 = time.time()
    model.learn(total_timesteps=a.steps)
    log["ppo_seconds"] = time.time() - t_ppo0
    log["ppo_env_steps"] = int(model.num_timesteps)
    eval_end = evaluate(model, cfg, a.n_eval, a.seed)
    print("[METRIC] eval_end", json.dumps(eval_end), flush=True)
    log.update(eval_init=eval0, eval_after_bc=eval_bc, eval_end=eval_end, wall_seconds=time.time() - t0)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(log, f, indent=1)
    print("[INFO] done", json.dumps({k: v for k, v in log.items() if not k.startswith("eval")}), flush=True)


if __name__ == "__main__":
    main()

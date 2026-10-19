"""RECHECK (official-code oracle lens; ORACLE side, throwaway): the paper's headline arm, PPO-BiGRU+MHSA, built the
way an SB3 user must build it -- a torch BaseFeaturesExtractor inside the AUTHORS' OWN SB3 fork (2.4.0a8 @ e46a6d5),
pre-trained with torch AE + the authors' imitation fork (@ f5cb8a4) -- driven on OUR JAX env.

Extractor (paper §2.6 / §3 + companion §III-F/ wiring): track list [K, 7] -> Linear(7, 64) -> torch
MultiheadAttention(64, 4, key_padding_mask from all-zero rows) -> residual + LayerNorm -> packed bidirectional GRU(32)
-> [h_fwd; h_bwd] = z_TL (64); raster [1, 48, 48] -> SB3 NatureCNN(128) = z_SH; features = [z_TL; z_SH]. An empty
track list gives z_TL = 0 (torch MHA would return NaN with every key masked and pack_padded_sequence refuses length 0).

Modes
  --record-update PATH   one SB3 PPO update (128 x 2 rollout, batch 64, 2 epochs) at fresh init; records compact
                         obs, buffer, permutations, parameters and pre-clip gradients per optimizer step (P10-style)
  (default)              reference run: AE (2e4 Adam steps, lr 1e-3, batch 256, x ~ U(-1,1)^{17x7}, no mask,
                         decoder 64-256-119) -> imitation BC (uniform-random teacher, 50 episodes, 5 epochs, batch 32)
                         with the track extractor frozen (--bc-train-extractor = imitation's default: everything trains)
                         -> SB3 PPO defaults for --steps -> evaluation on the first 32 small-scale scenarios (stochastic).
Lane/scenario keys: OUR key tree for arm 'mhsa' (id 3), so seed s trains on the scenarios of our PPO-MHSA lane s.

    $OFFICIAL_ROOT/.venv-sb3/bin/python -I mhsa_sb3.py --seed 0 --out DIR
"""
import argparse, json, os, sys, time
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "official"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa
import jax  # noqa
import jax.numpy as jnp  # noqa
import torch as th  # noqa
from torch import nn  # noqa
from torch.nn.utils.rnn import pack_padded_sequence  # noqa
from stable_baselines3 import PPO  # noqa
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN  # noqa
from stable_baselines3.common.logger import configure  # noqa
from imitation.algorithms import bc as imitation_bc  # noqa
import sb3_reference_run as R  # noqa  (JaxVecEnv, bc_transitions, evaluate, make_fns; jax x64 enabled there)
from src.configs.config import EnvConfig  # noqa
from src import data  # noqa

ARM_MHSA = 3
K_VALID = 0x7A11


def evaluate_tag(model, cfg, n_scen, seed, tag):
    """R.evaluate with the scenario tag as a parameter (0x6A72 small-scale, 0x7A11 our 16 validation scenarios)."""
    old = R.K_SMALL
    R.K_SMALL = tag
    try:
        return R.evaluate(model, cfg, n_scen, seed)
    finally:
        R.K_SMALL = old


def lane_key(seed, arm_id=ARM_MHSA):
    return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(15), arm_id), seed)


class MHSAExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, d=64, heads=4, gru_h=32, cnn_out=128):
        super().__init__(observation_space, features_dim=2 * gru_h + cnn_out)
        self.embed = nn.Linear(7, d)
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.gru = nn.GRU(d, gru_h, batch_first=True, bidirectional=True)
        self.cnn = NatureCNN(observation_space["raster"], features_dim=cnn_out, normalized_image=True)

    def encode_tl(self, tl, mask):
        has = mask.any(1)
        e = self.embed(tl * mask[..., None])
        kpm = th.where(has[:, None], ~mask, th.zeros_like(mask))
        a, _ = self.mha(e, e, e, key_padding_mask=kpm, need_weights=False)
        y = self.ln(e + a)
        lengths = mask.sum(1).clamp(min=1).cpu()
        _, h = self.gru(pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False))
        return th.cat([h[0], h[1]], -1) * has[:, None].to(y.dtype)

    def forward(self, obs):
        tl = obs["tl"]
        mask = tl.abs().sum(-1) > 0
        return th.cat([self.encode_tl(tl, mask), self.cnn(obs["raster"])], -1)


PK = dict(features_extractor_class=MHSAExtractor, net_arch=dict(pi=[128, 128], vf=[128, 128]))
TL_MODULES = ("embed", "mha", "ln", "gru")


def ae_pretrain(ext, seed, steps=20_000, batch=256, K=17):
    g = th.Generator().manual_seed(1000 + seed)
    dec = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, K * 7))
    params = [p for m in TL_MODULES for p in getattr(ext, m).parameters()] + list(dec.parameters())
    opt = th.optim.Adam(params, lr=1e-3)
    curve = []
    for i in range(steps):
        x = th.rand((batch, K, 7), generator=g) * 2 - 1
        z = ext.encode_tl(x, th.ones(batch, K, dtype=th.bool))
        loss = ((dec(z).reshape(x.shape) - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        curve.append(float(loss))
    xe = th.rand((4096, K, 7), generator=g) * 2 - 1
    with th.no_grad():
        final = float(((dec(ext.encode_tl(xe, th.ones(4096, K, dtype=th.bool))).reshape(xe.shape) - xe) ** 2).mean())
    return np.asarray(curve).reshape(20, -1).mean(1).tolist(), final


def record_update(path, n_steps=128, n_envs=2, batch=64, epochs=2, seed=0):
    cfg = EnvConfig(gospa_metrics=False)
    venv = R.JaxVecEnv(cfg, n_envs, lane_key(seed))
    compact = []
    orig_reset, orig_wait = venv.reset, venv.step_wait

    def grab():
        o = jax.vmap(lambda st: data.observe(st, cfg))(venv.s)
        compact.append({k: np.asarray(getattr(o, k)) for k in o._fields})

    def reset():
        r = orig_reset(); grab(); return r

    def wait():
        r = orig_wait(); grab(); return r
    venv.reset, venv.step_wait = reset, wait
    model = PPO("MultiInputPolicy", venv, n_steps=n_steps, batch_size=batch, n_epochs=epochs, learning_rate=3e-4,
                seed=seed, device="cpu", policy_kwargs=PK, verbose=0)
    pol = model.policy
    p0 = {k: v.detach().numpy().copy() for k, v in pol.state_dict().items()}
    perms, steps = [], []
    orig_perm = np.random.permutation

    def perm_spy(n):
        idx = orig_perm(n); perms.append(np.asarray(idx).copy()); return idx
    np.random.permutation = perm_spy
    orig_clip = th.nn.utils.clip_grad_norm_
    names = [n for n, _ in pol.named_parameters()]

    def clip_spy(parameters, max_norm, *a, **k):
        params = list(parameters)
        steps.append(dict(before={n: p.detach().numpy().copy() for n, p in zip(names, params)},
                          pre={n: p.grad.detach().numpy().copy() for n, p in zip(names, params)}))
        return orig_clip(params, max_norm, *a, **k)
    th.nn.utils.clip_grad_norm_ = clip_spy
    model.learn(total_timesteps=n_steps * n_envs)
    np.random.permutation, th.nn.utils.clip_grad_norm_ = orig_perm, orig_clip
    buf = model.rollout_buffer
    comp = {k: np.stack([c[k] for c in compact[:n_steps]], 0) for k in compact[0]}
    comp = {k: v.swapaxes(0, 1).reshape((n_steps * n_envs,) + v.shape[2:]) for k, v in comp.items()}
    out = dict(actions=buf.actions, log_probs=buf.log_probs, values=buf.values, advantages=buf.advantages,
               returns=buf.returns, raster=buf.observations["raster"], tl_sb3=buf.observations["tl"],
               batch=np.array(batch), perms=np.stack(perms))
    out.update({"obs/" + k: v for k, v in comp.items()})
    out.update({"p0/" + k: v for k, v in p0.items()})
    keep = sorted({0, 1, len(steps) // 2, len(steps) - 1})
    for i in keep:
        out.update({f"s{i}/before/{k}": v for k, v in steps[i]["before"].items()})
        out.update({f"s{i}/pre/{k}": v for k, v in steps[i]["pre"].items()})
    out["kept_steps"] = np.array(keep)
    out["frac_rows_with_track"] = np.array((comp["tl_mask"].any(-1)).mean())
    np.savez_compressed(path, **out)
    print("[INFO] recorded", len(steps), "optimizer steps; kept", keep, "; frac obs with >=1 track",
          float(out["frac_rows_with_track"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-update", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=114_688)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--bc-train-extractor", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "official", "sb3_mhsa"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-ae", action="store_true", help="skip AE (our arm mhsa_noae, id 5: BC around a frozen random extractor)")
    a = ap.parse_args()
    th.set_num_threads(a.threads)
    if a.record_update:
        th.manual_seed(a.seed)
        record_update(a.record_update, seed=a.seed)
        return
    th.manual_seed(a.seed)
    tag = ("noae_" if a.no_ae else "") + ("bcall" if a.bc_train_extractor else "bcfrozen") + f"_seed{a.seed}"
    out_dir = os.path.join(a.out, tag)
    os.makedirs(out_dir, exist_ok=True)
    cfg = EnvConfig(gospa_metrics=False)
    k_lane = lane_key(a.seed, 5 if a.no_ae else ARM_MHSA)
    venv = R.JaxVecEnv(cfg, 8, k_lane)
    model = PPO("MultiInputPolicy", venv, n_steps=2048, batch_size=64, n_epochs=10, learning_rate=3e-4, gamma=0.99,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, seed=a.seed,
                device="cpu", policy_kwargs=PK, verbose=0)
    model.set_logger(configure(out_dir, ["csv"]))
    log = dict(arm="mhsa", seed=a.seed, bc_train_extractor=a.bc_train_extractor,
               n_params=int(sum(p.numel() for p in model.policy.parameters())), torch_threads=a.threads)
    t0 = time.time()
    log["eval_init"] = R.evaluate(model, cfg, a.n_eval, a.seed)
    print("[METRIC] eval_init", json.dumps(log["eval_init"]), flush=True)
    ext = model.policy.features_extractor
    if not a.no_ae:
        log["ae_curve"], log["ae_final"] = ae_pretrain(ext, a.seed)
    log["ae_seconds"] = time.time() - t0
    print("[METRIC] ae_final", log.get("ae_final"), "floor 0.154062", flush=True)
    trans = R.bc_transitions(cfg, k_lane, 50)
    if not a.bc_train_extractor:
        for m in TL_MODULES:
            for p in getattr(ext, m).parameters():
                p.requires_grad_(False)
    trainer = imitation_bc.BC(observation_space=venv.observation_space, action_space=venv.action_space,
                              rng=np.random.default_rng(a.seed), policy=model.policy, demonstrations=trans,
                              batch_size=32, device="cpu")
    trainer.train(n_epochs=5, log_interval=10 ** 9, progress_bar=False)
    for m in TL_MODULES:
        for p in getattr(ext, m).parameters():
            p.requires_grad_(True)
    del trans
    log["eval_after_bc"] = R.evaluate(model, cfg, a.n_eval, a.seed)
    log["valid_after_bc"] = evaluate_tag(model, cfg, 16, a.seed, K_VALID)
    print("[METRIC] valid_after_bc", json.dumps(log["valid_after_bc"]), flush=True)
    print("[METRIC] eval_after_bc", json.dumps(log["eval_after_bc"]), flush=True)
    t1 = time.time()
    model.learn(total_timesteps=a.steps)
    log["ppo_seconds"] = time.time() - t1
    log["ppo_env_steps"] = int(model.num_timesteps)
    log["eval_end"] = R.evaluate(model, cfg, a.n_eval, a.seed)
    log["valid_end"] = evaluate_tag(model, cfg, 16, a.seed, K_VALID)
    print("[METRIC] valid_end", json.dumps(log["valid_end"]), flush=True)
    model.save(os.path.join(out_dir, "model.zip"))
    print("[METRIC] eval_end", json.dumps(log["eval_end"]), flush=True)
    log["wall_seconds"] = time.time() - t0
    json.dump(log, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    print("[INFO] done", flush=True)


if __name__ == "__main__":
    main()

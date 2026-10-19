"""PPO training of one encoder family, vmapped over lanes (arm x seed) in ONE process.

Pipeline per lane: AE pre-training of the track extractor (BiGRU / BiGRU+MHSA; Flat has none) -> BC of the
uniform-random teacher (CNN, MLPs, heads; track extractor frozen) -> PPO (all parameters). Lanes of one family share
the network structure and per-lane flags: use_ae, use_bc, bc_mse (printed-MSE BC reading) and rtl_sign (+1 = flipped,
the reduction-rewarded primary; -1 = the literal printed sign). Pre-training runs for every lane and the flags select
which parameters PPO starts from.

INDEPENDENT LANES (review / review): with the default key tree the lane key includes the arm id, so two arms
at the same seed index do NOT share initialisation, AE noise, BC data, training scenarios, action draws or SGD order;
they differ in the mechanism under test AND in every random stream. Pairing by seed index is therefore nominal (every
lane is evaluated on the same 100 scenarios regardless). Config.lane_keys_shared_across_arms = True derives the lane
key from the seed alone (common random numbers across ablation arms) for future runs; small/full-scale used the default.

PRNG key tree (README):
  root = PRNGKey(seed_root)
  k_lane = fold_in(fold_in(root, arm_id), seed)      [lane_keys_shared_across_arms: fold_in(fold_in(root, 0xC12), seed)]
    k_init = fold_in(k_lane, 0)   network init
    k_ae   = fold_in(k_lane, 1)   AE noise batches + decoder init
    k_bc   = fold_in(k_lane, 2)   BC scenarios (fold_in(k_bc, 0)), teacher actions (fold_in(k_bc, 1)), batch order (2)
    k_train= fold_in(k_lane, 3)   training scenarios: env e, episode j -> fold_in(fold_in(k_train, e), j)
    k_act  = fold_in(k_lane, 4)   rollout action sampling: fold_in(k_act, global_step)
    k_sgd  = fold_in(k_lane, 5)   minibatch permutations: fold_in(k_sgd, iteration)
    k_evalpol = fold_in(k_lane, 6) policy sampling at evaluation
  validation scenarios: fold_in(root, K_VALID); the final eval scenarios fold_in(root, K_EVAL_SCEN) are never used here.

Usage:  python src/train.py --family mhsa --arms mhsa,mhsa_nobc --seeds 0,1,2 --steps 200000 --out runs/small
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common as C  # noqa: E402  (GPU bootstrap + x64 before jax)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from src import data, algorithm as alg  # noqa: E402
from src.configs.config import Config, EnvConfig, NetConfig, K_VALID, to_dict  # noqa: E402
from src.model import ActorCritic, TrackEncoder, AEDecoder, NatureCNN, raster_of, dummy_obs  # noqa: E402

# arm registry: id (key tree), encoder, flags
ARMS = {
    "flat":       dict(id=1, encoder="flat",  use_ae=0, use_bc=1, bc_mse=0, rtl_sign=1.0),
    "bigru":      dict(id=2, encoder="bigru", use_ae=1, use_bc=1, bc_mse=0, rtl_sign=1.0),
    "mhsa":       dict(id=3, encoder="mhsa",  use_ae=1, use_bc=1, bc_mse=0, rtl_sign=1.0),
    "mhsa_nobc":  dict(id=4, encoder="mhsa",  use_ae=1, use_bc=0, bc_mse=0, rtl_sign=1.0),
    "mhsa_noae":  dict(id=5, encoder="mhsa",  use_ae=0, use_bc=1, bc_mse=0, rtl_sign=1.0),
    "mhsa_lit":   dict(id=6, encoder="mhsa",  use_ae=1, use_bc=1, bc_mse=0, rtl_sign=-1.0),
    "mhsa_bcmse": dict(id=7, encoder="mhsa",  use_ae=1, use_bc=1, bc_mse=1, rtl_sign=1.0),   # pre-training only
    # planned-check verification block (README): COMMON RANDOM NUMBERS -- each arm reuses
    # the key-tree id of its control (mhsa = 3, flat = 1), so it shares the control's init, pre-training data, training
    # scenarios and action-key tree at the same seed and differs only in the mechanism under test. lr_scale multiplies
    # the Adam update (lr' = lr_scale x 3e-4, exact for Adam); unmasked / single are set per process (NetConfig.masked,
    # EnvConfig.ssv_norm) by scripts/paper_budget.py.
    "mhsa_lr1e-4":   dict(id=3, encoder="mhsa", use_ae=1, use_bc=1, bc_mse=0, rtl_sign=1.0, lr_scale=1.0 / 3.0),
    "mhsa_lr1e-3":   dict(id=3, encoder="mhsa", use_ae=1, use_bc=1, bc_mse=0, rtl_sign=1.0, lr_scale=10.0 / 3.0),
    "mhsa_lit_crn":  dict(id=3, encoder="mhsa", use_ae=1, use_bc=1, bc_mse=0, rtl_sign=-1.0),
    "mhsa_unmasked": dict(id=3, encoder="mhsa", use_ae=1, use_bc=1, bc_mse=0, rtl_sign=1.0),
    "flat_lr1e-4":   dict(id=1, encoder="flat", use_ae=0, use_bc=1, bc_mse=0, rtl_sign=1.0, lr_scale=1.0 / 3.0),
    "flat_lr1e-3":   dict(id=1, encoder="flat", use_ae=0, use_bc=1, bc_mse=0, rtl_sign=1.0, lr_scale=10.0 / 3.0),
    "flat_single":   dict(id=1, encoder="flat", use_ae=0, use_bc=1, bc_mse=0, rtl_sign=1.0),
}


class Flags(NamedTuple):
    use_ae: jax.Array
    use_bc: jax.Array
    bc_mse: jax.Array
    rtl_sign: jax.Array
    lr_scale: jax.Array = 1.0   # per-lane multiplier of the Adam update (1.0 = the SB3 default lr; every small/full-scale arm)


class Lane(NamedTuple):
    params: dict
    opt_state: tuple
    env: data.EnvState      # [n_envs]
    obs: data.Obs           # [n_envs]
    k_lane: jax.Array


def lane_key(cfg: Config, arm: str, seed: int):
    root = jax.random.PRNGKey(cfg.seed_root)
    if cfg.lane_keys_shared_across_arms:     # common random numbers across arms; not used by small/full-scale
        return jax.random.fold_in(jax.random.fold_in(root, 0xC12), seed)
    return jax.random.fold_in(jax.random.fold_in(root, ARMS[arm]["id"]), seed)


def train_scen_key(k_lane, env_i, episode):
    return jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(k_lane, 3), env_i), episode)


# ----------------------------------------------------------------------------- pre-training
def make_pretrain(cfg: Config):
    ecfg = replace(cfg.env, gospa_metrics=False)
    net = ActorCritic(cfg.net, ecfg)
    enc = TrackEncoder(cfg.net)
    K = ecfg.k_max
    dec = AEDecoder(cfg.pre.ae_dec_hidden, K * 7, init_style=cfg.net.init_style)
    pre = cfg.pre

    def ae(k_ae, enc_params):
        """Train the track extractor + decoder on x ~ U(-1,1)^{K x 7}, no mask (§2.6.2). Returns (enc_params, curve)."""
        dparams = dec.init(jax.random.fold_in(k_ae, 0), jnp.zeros((1, 64), jnp.float32))
        p = dict(enc=enc_params, dec=dparams)
        opt = optax.adam(pre.ae_lr)
        ostate = opt.init(p)
        def loss_fn(p, x):
            z = enc.apply(p["enc"], x, jnp.ones(x.shape[:-1], bool))
            xh = dec.apply(p["dec"], z).reshape(x.shape)
            return alg.ae_loss(xh, x)

        def body(c, i):
            p, ostate = c
            x = jax.random.uniform(jax.random.fold_in(k_ae, i + 1), (pre.ae_batch, K, 7), jnp.float32, -1.0, 1.0)
            l, g = jax.value_and_grad(loss_fn)(p, x)
            u, ostate = opt.update(g, ostate, p)
            return (optax.apply_updates(p, u), ostate), l

        (p, _), losses = jax.lax.scan(body, (p, ostate), jnp.arange(pre.ae_steps))
        xe = jax.random.uniform(jax.random.fold_in(k_ae, 10 ** 7), (4096, K, 7), jnp.float32, -1.0, 1.0)
        final = loss_fn(p, xe)
        return p["enc"], losses.reshape(20, -1).mean(1), final

    def bc_dataset(k_bc):
        """50 episodes of the uniform-random teacher on training-distribution scenarios (compact observations)."""
        E = pre.bc_episodes

        def episode(e):
            s = data.reset(jax.random.fold_in(jax.random.fold_in(k_bc, 0), e), ecfg)
            o = data.observe(s, ecfg)
            ka = jax.random.fold_in(jax.random.fold_in(k_bc, 1), e)

            def body(c, t):
                s, o = c
                a = jax.random.randint(jax.random.fold_in(ka, t), (2,), 0, ecfg.n_a).astype(jnp.int32)
                s2, o2, _ = data.step(s, a, ecfg)
                return (s2, o2), (o, a)

            _, (obs, act) = jax.lax.scan(body, (s, o), jnp.arange(ecfg.T))
            return obs, act

        obs, act = jax.vmap(episode)(jnp.arange(E))
        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        return jax.tree.map(flat, obs), flat(act)

    cnn = NatureCNN(cfg.net.cnn_out)

    def cnn_alive(params, o):
        """Recheck /: fraction of the 128 NatureCNN output units (z_SH) that are > 0 on at least one of
        up to 512 teacher observations. 0.0 = the scan-history stream is dead (a dead ReLU passes no gradient)."""
        z = cnn.apply({"params": params["params"]["cnn"]}, raster_of(o, ecfg).astype(jnp.float32))
        return jnp.mean(jnp.any(z > 0, axis=0).astype(jnp.float32))

    def bc(k_bc, params, mse_flag):
        """imitation-style BC (Adam 1e-3, batch 32, 5 epochs); the track extractor is frozen (zero gradient) unless
        pre.bc_train_extractor (the imitation-fork default)."""
        obs, act = bc_dataset(k_bc)
        n = act.shape[0]
        nb = n // pre.bc_batch
        opt = optax.adam(pre.bc_lr)
        ostate = opt.init(params)

        def loss_fn(p, o, a):
            la, le, _ = net.apply(p, o)
            ce = alg.bc_ce_loss(la, le, a, pre.bc_ent_weight)
            ms = alg.bc_mse_loss(la, le, a)
            return jnp.where(mse_flag > 0, ms, ce)

        def epoch(c, ep):
            perm = jax.random.permutation(jax.random.fold_in(jax.random.fold_in(k_bc, 2), ep), n)[: nb * pre.bc_batch]
            perm = perm.reshape(nb, pre.bc_batch)

            def mb(c, idx):
                p, ostate = c
                o = jax.tree.map(lambda x: x[idx], obs)
                l, g = jax.value_and_grad(loss_fn)(p, o, act[idx])
                if not pre.bc_train_extractor:       # primary: the extractor is frozen during BC
                    g = dict(params={k: (jax.tree.map(jnp.zeros_like, v) if k == "track_enc" else v)
                                     for k, v in g["params"].items()})
                u, ostate = opt.update(g, ostate, p)
                return (optax.apply_updates(p, u), ostate), l

            c, ls = jax.lax.scan(mb, c, perm)
            return c, ls.mean()

        probe = jax.tree.map(lambda x: x[:: max(1, n // 512)][:512], obs)
        alive0 = cnn_alive(params, probe)
        (params, _), ep_loss = jax.lax.scan(epoch, (params, ostate), jnp.arange(pre.bc_epochs))
        return params, ep_loss, alive0, cnn_alive(params, probe)

    def pretrain_lane(k_lane, flags: Flags):
        params0 = net.init(jax.random.fold_in(k_lane, 0), dummy_obs(ecfg, (1,)))
        out = {}
        if cfg.net.encoder != "flat":
            enc_trained, ae_curve, ae_final = ae(jax.random.fold_in(k_lane, 1), {"params": params0["params"]["track_enc"]})
            pe = jax.tree.map(lambda a, b: jnp.where(flags.use_ae > 0, a, b), enc_trained["params"],
                              params0["params"]["track_enc"])
            params1 = {"params": {**params0["params"], "track_enc": pe}}
            out.update(ae_curve=ae_curve, ae_final=ae_final)
        else:
            params1 = params0
        params_bc, bc_curve, alive0, alive1 = bc(jax.random.fold_in(k_lane, 2), params1, flags.bc_mse)
        params2 = jax.tree.map(lambda a, b: jnp.where(flags.use_bc > 0, a, b), params_bc, params1)
        out.update(bc_curve=bc_curve, cnn_alive_pre_bc=alive0,
                   cnn_alive_ppo_start=jnp.where(flags.use_bc > 0, alive1, alive0))
        return params2, out

    return net, pretrain_lane


# ----------------------------------------------------------------------------- PPO
def make_ppo(cfg: Config, net: ActorCritic, lr_scaled: bool = False):
    ecfg = replace(cfg.env, gospa_metrics=False)
    P = cfg.ppo
    nE, nS = P.n_envs, P.n_steps
    opt = alg.ppo_optimizer(P.lr, P.max_grad_norm, P.adam_eps)
    n_mb = nE * nS // P.minibatch
    T = ecfg.T

    def init_lane(k_lane, params):
        envs = jax.vmap(lambda e: data.reset(train_scen_key(k_lane, e, 0), ecfg))(jnp.arange(nE))
        obs = jax.vmap(lambda s: data.observe(s, ecfg))(envs)
        return Lane(params, opt.init(params), envs, obs, k_lane)

    def iteration(lane: Lane, flags: Flags, step0, it):
        """One PPO update: nS x nE rollout, GAE, n_epochs x n_mb minibatch steps. step0 = per-env steps taken so far
        (identical for every env and lane, unbatched under vmap -> the truncation cond stays a real branch)."""
        k_lane = lane.k_lane
        k_act = jax.random.fold_in(k_lane, 4)
        env_step = jax.vmap(lambda s, a: data.step(s, a, ecfg))

        def body(c, g):
            envs, obs = c
            la, le, v = net.apply(lane.params, obs)
            a = alg.sample_action(jax.random.fold_in(k_act, g), la, le)
            logp, _ = alg.logp_entropy(la, le, a)
            envs2, obs2, info = env_step(envs, a)
            r = info["r_sv"] + flags.rtl_sign * info["rtl_flip"]
            trunc = (g + 1) % T == 0

            def on_trunc(_):
                _, _, v_fin = net.apply(lane.params, obs2)
                ep = (g + 1) // T
                envs_r = jax.vmap(lambda e: data.reset(train_scen_key(k_lane, e, ep), ecfg))(jnp.arange(nE))
                obs_r = jax.vmap(lambda s: data.observe(s, ecfg))(envs_r)
                return v_fin, envs_r, obs_r

            def no_trunc(_):
                return jnp.zeros((nE,), jnp.float32), envs2, obs2

            v_fin, envs3, obs3 = jax.lax.cond(trunc, on_trunc, no_trunc, None)
            r_boot = r.astype(jnp.float32) + P.gamma * v_fin              # SB3 timeout bootstrap 
            done = jnp.full((nE,), trunc, jnp.float32)
            rec = dict(obs=obs, a=a, logp=logp, v=v, r=r_boot, done=done, r_sv=info["r_sv"],
                       rtl=info["rtl_flip"], cov=info["cov_sum"], ntr=info["n_tracks"], smin=info["min_S_minor"])
            return (envs3, obs3), rec

        (envs, obs), buf = jax.lax.scan(body, (lane.env, lane.obs), step0 + jnp.arange(nS))
        # env-step guard (README): rewards, covariance sums and values finite; UKF innovation S positive
        # definite (min leading principal minor > 0; +inf when no live track)
        C.nan_guard(dict(r=buf["r"], cov=buf["cov"], v=buf["v"]), dict(it=it), tag=2,
                    dump_dir=os.path.join(C.REPO, "nan_dumps"))
        C.nan_guard(dict(psd=jnp.where(jnp.min(buf["smin"]) > 0, 0.0, jnp.nan)), dict(it=it), tag=3,
                    dump_dir=os.path.join(C.REPO, "nan_dumps"))
        _, _, last_v = net.apply(lane.params, obs)
        adv, ret = alg.gae(buf["r"], buf["v"], buf["done"], last_v, P.gamma, P.gae_lambda)
        flat = lambda x: x.reshape((nS * nE,) + x.shape[2:])
        B = dict(obs=jax.tree.map(flat, buf["obs"]), a=flat(buf["a"]), logp=flat(buf["logp"]), adv=flat(adv),
                 ret=flat(ret))

        def loss_fn(p, mb):
            la, le, v = net.apply(p, mb["obs"])
            pg, vl, el, aux = alg.ppo_loss_terms(la, le, v, mb["a"], mb["logp"], mb["adv"], mb["ret"], P.clip)
            return pg + P.ent_coef * el + P.vf_coef * vl, (pg, vl, aux)

        k_sgd = jax.random.fold_in(jax.random.fold_in(k_lane, 5), it)

        def epoch(c, e):
            perm = jax.random.permutation(jax.random.fold_in(k_sgd, e), nS * nE).reshape(n_mb, P.minibatch)

            def mb_step(c, idx):
                p, os_ = c
                mb = jax.tree.map(lambda x: x[idx], B)
                (l, (pg, vl, aux)), g = jax.value_and_grad(loss_fn, has_aux=True)(p, mb)
                u, os_ = opt.update(g, os_, p)
                if lr_scaled:      # static switch: families without an lr arm compile the unchanged (bitwise) graph
                    u = jax.tree.map(lambda x: x * flags.lr_scale.astype(x.dtype), u)
                return (optax.apply_updates(p, u), os_), dict(loss=l, pg=pg, vl=vl, **aux,
                                                             gnorm=optax.global_norm(g))

            return jax.lax.scan(mb_step, c, perm)

        (params, os_), st = jax.lax.scan(epoch, (lane.params, lane.opt_state), jnp.arange(P.n_epochs))
        C.nan_guard(params, dict(it=it), tag=1, dump_dir=os.path.join(C.REPO, "nan_dumps"))
        stats = {k: v.mean() for k, v in st.items()}
        stats.update(rew=buf["r"].mean(), r_sv=buf["r_sv"].mean(), rtl=buf["rtl"].mean(),
                     cov=buf["cov"].mean(), ntr=buf["ntr"].mean(),
                     ev=1 - jnp.var(B["ret"] - flat(buf["v"])) / (jnp.var(B["ret"]) + 1e-8))
        return Lane(params, os_, envs, obs, k_lane), stats

    return init_lane, iteration


# ----------------------------------------------------------------------------- learning-curve evaluation
def make_curve_eval(cfg: Config, net: ActorCritic, deterministic: bool = False):
    from src import eval as E
    ecfg = replace(cfg.env, gospa_metrics=True)
    root = jax.random.PRNGKey(cfg.seed_root)
    sk = E.scenario_keys(root, K_VALID, cfg.ppo.curve_scenarios)

    def one(params, k_lane):
        pinit, pol = E.make_learned(net, params, deterministic)
        pk = jax.vmap(lambda i: jax.random.fold_in(jax.random.fold_in(k_lane, 6), 10_000 + i))(jnp.arange(sk.shape[0]))
        out = E.eval_many(pinit, pol, sk, pk, ecfg)
        return dict(ret_flip=out["ret_flip"].mean(), ret_lit=out["ret_lit"].mean(), sr=out["sr"].mean(),
                    m_cov=jnp.nanmean(out["m_cov"]), gospa_d=out["gospa_d"].mean(),
                    gospa_miss=out["gospa_miss"].mean(), mean_tracks=out["mean_tracks"].mean())
    return jax.jit(jax.vmap(one))


# ----------------------------------------------------------------------------- driver
def run_family(cfg: Config, arms: list, seeds: list, out_dir: str, total_steps: int, curve_every: int,
               ckpt_every_it: int = 20, lanes_meta: list | None = None, select_best_patience: int | None = None,
               curve_log: str | None = None, select_deterministic: bool = False):
    """select_best_patience (review; paper §3 'early termination if no improvement was recorded in an evaluation
    environment after 25x10^5 steps'): keep each lane's best parameters by its own validation return (ret_flip, or
    ret_lit for the literal-sign lanes; the 16 validation scenarios, stochastic, at every curve point) and stop when
    EVERY lane has gone that many env steps without improving; the returned lanes carry the per-lane BEST parameters
    (SB3 EvalCallback best_model semantics; SB3 evaluates deterministically on n_eval_episodes = 5 by default, ours is the
    curve evaluation -- a declared difference). None = off (small/full-scale).
    curve_log (review): if set, the validation-curve metrics go to this file instead of stdout, so a pending
    timing/budget decision can be taken without seeing them.
    select_deterministic (recheck): select the best model on the DETERMINISTIC (argmax) validation return,
    as SB3 EvalCallback does by default; the deterministic curve is logged next to the stochastic one. Off = full-scale."""
    enc = ARMS[arms[0]]["encoder"]
    assert all(ARMS[a]["encoder"] == enc for a in arms), "one encoder per process (vmap needs one structure)"
    cfg = replace(cfg, net=replace(cfg.net, encoder=enc), ppo=replace(cfg.ppo, total_steps=total_steps))
    lanes_meta = lanes_meta or [(a, s) for a in arms for s in seeds]
    L = len(lanes_meta)
    keys = jnp.stack([lane_key(cfg, a, s) for a, s in lanes_meta])
    flags = Flags(*(jnp.array([float(ARMS[a].get(f, 1.0)) for a, _ in lanes_meta]) for f in Flags._fields))
    C.info("family", encoder=enc, lanes=L, lanes_meta=";".join(f"{a}/{s}" for a, s in lanes_meta), steps=total_steps)
    os.makedirs(out_dir, exist_ok=True)

    net, pretrain_lane = make_pretrain(cfg)
    init_lane, iteration = make_ppo(cfg, net, lr_scaled=any(ARMS[a].get("lr_scale", 1.0) != 1.0 for a, _ in lanes_meta))
    curve_eval = make_curve_eval(cfg, net)
    curve_eval_det = make_curve_eval(cfg, net, deterministic=True) if select_deterministic else None
    curves_det = {i: [] for i in range(L)}
    tm = C.Timer()
    import pickle
    pre_cache = os.path.join(out_dir, f"pretrain_{enc}.pkl")
    if os.path.exists(pre_cache):          # resume after a crash: pre-training is deterministic per lane key
        with open(pre_cache, "rb") as f:
            pc = pickle.load(f)
        assert pc["lanes"] == [list(x) for x in lanes_meta], "pretrain cache belongs to other lanes"
        params, pre_np, t_pre = jax.tree.map(jnp.asarray, pc["params"]), pc["pre"], pc["t_pre"]
        C.info("pretrain loaded from cache", path=pre_cache)
    else:
        params, pre_out = jax.jit(jax.vmap(pretrain_lane))(keys, flags)
        jax.block_until_ready(params)
        t_pre = tm.lap()
        C.info("pretrain done", seconds=round(t_pre, 1))
        pre_np = jax.tree.map(np.asarray, pre_out)
        alive = pre_np["cnn_alive_ppo_start"]
        C.metric("cnn liveness", pre_bc=";".join(f"{x:.3f}" for x in pre_np["cnn_alive_pre_bc"]),
                 ppo_start=";".join(f"{x:.3f}" for x in alive))
        if np.any(alive == 0.0):
            C.warn("NatureCNN dead at the PPO start (all 128 z_SH units are 0 on the teacher data): those lanes are "
                   "raster-blind at the PPO start (on-policy rasters can revive units during PPO)", lanes=";".join(
                       f"{a}/{s}" for (a, s), x in zip(lanes_meta, alive) if x == 0.0))
        if ckpt_every_it:
            with open(pre_cache, "wb") as f:
                pickle.dump(dict(lanes=[list(x) for x in lanes_meta], params=jax.tree.map(np.asarray, params),
                                 pre=pre_np, t_pre=t_pre), f)
    lanes = jax.jit(jax.vmap(init_lane))(keys, params)

    it_fn = jax.jit(jax.vmap(iteration, in_axes=(0, 0, None, None)))
    P = cfg.ppo
    per_it = P.n_steps
    n_it = int(np.ceil(total_steps / (per_it * P.n_envs)))
    curve_its = max(1, int(round(curve_every / (per_it * P.n_envs))))
    curves = {i: [] for i in range(L)}
    hist = []
    ck_dir = os.path.join(out_dir, "ckpt_" + enc)
    start_it = 0
    mgr = None
    if ckpt_every_it:
        import orbax.checkpoint as ocp
        mgr = ocp.CheckpointManager(os.path.abspath(ck_dir), options=ocp.CheckpointManagerOptions(max_to_keep=2,
                                                                                                    create=True))
        if mgr.latest_step() is not None:
            st = mgr.latest_step()
            lanes = mgr.restore(st, args=ocp.args.StandardRestore(lanes))
            start_it = st
            prev = os.path.join(out_dir, f"progress_{enc}.json")
            if os.path.exists(prev):
                with open(prev) as f:
                    pj = json.load(f)
                hist, curves = pj["hist"], {int(k): v for k, v in pj["curves"].items()}
                if pj.get("curves_det"):
                    curves_det = {int(k): v for k, v in pj["curves_det"].items()}
            C.info("resumed", iteration=st)

    sign = np.array([ARMS[a]["rtl_sign"] for a, _ in lanes_meta])
    best = dict(ret=np.full(L, -np.inf), step=np.zeros(L, np.int64), params=None)
    best_file = lambda s: os.path.join(out_dir, f"best_state_{enc}_{s}.pkl")  # noqa: E731
    if start_it and select_best_patience is not None:
        # the per-lane best model is saved with every checkpoint, so selection survives a crash and resume
        if os.path.exists(best_file(start_it)):
            with open(best_file(start_it), "rb") as f:
                bs = pickle.load(f)
            best = dict(ret=np.asarray(bs["ret"]), step=np.asarray(bs["step"], np.int64),
                        params=None if bs["params"] is None else jax.tree.map(jnp.asarray, bs["params"]))
            C.info("best-model state restored", iteration=start_it, best_steps=best["step"].tolist())
        else:
            C.warn("no best-model state for the resumed checkpoint: selection restarts at the resume point",
                   iteration=start_it)

    def do_curve(it):
        res = jax.tree.map(np.asarray, curve_eval(lanes.params, keys))
        for i in range(L):
            curves[i].append(dict(env_steps=int(it * per_it * P.n_envs), **{k: float(v[i]) for k, v in res.items()}))
        a0 = {k: float(np.mean(v)) for k, v in res.items()}
        if curve_log:
            with open(curve_log, "a") as fh:
                fh.write(json.dumps(dict(it=it, env_steps=it * per_it * P.n_envs, **a0)) + "\n")
        else:
            C.metric("curve", it=it, env_steps=it * per_it * P.n_envs, **a0)
        sel = res
        if curve_eval_det is not None:
            sel = jax.tree.map(np.asarray, curve_eval_det(lanes.params, keys))
            for i in range(L):
                curves_det[i].append(dict(env_steps=int(it * per_it * P.n_envs), **{k: float(v[i]) for k, v in sel.items()}))
        if select_best_patience is not None:
            own = np.where(sign > 0, sel["ret_flip"], sel["ret_lit"])
            imp = own > best["ret"]
            best["ret"] = np.where(imp, own, best["ret"])
            best["step"] = np.where(imp, it * per_it * P.n_envs, best["step"])
            m = jnp.asarray(imp)
            best["params"] = lanes.params if best["params"] is None else jax.tree.map(
                lambda a, b: jnp.where(m.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), lanes.params, best["params"])

    if start_it == 0:
        do_curve(0)
    t_compile = None
    t_steady = 0.0
    ovr = os.path.join(out_dir, f"n_it_override_{enc}")
    it = start_it
    while it < n_it:
        if os.path.exists(ovr):            # budget control (README full-scale declaration): timing-only decision
            with open(ovr) as f:
                new_n = int(f.read().strip())
            if new_n != n_it:
                C.warn("iteration count overridden by budget control", old=n_it, new=new_n)
                n_it = max(new_n, it + 1)
                if it >= n_it:
                    break
        t0 = time.perf_counter()
        lanes, stats = it_fn(lanes, flags, it * per_it, it)
        stats = jax.tree.map(np.asarray, stats)
        dt = time.perf_counter() - t0
        if t_compile is None:
            t_compile = dt
            C.info("first iteration (compile + run)", seconds=round(dt, 1))
        else:
            t_steady += dt
        hist.append({k: [float(x) for x in v] for k, v in stats.items()})
        C.metric("iter", it=it + 1, env_steps=(it + 1) * per_it * P.n_envs, sec=round(dt, 2),
                 rew=float(stats["rew"].mean()), r_sv=float(stats["r_sv"].mean()), ntr=float(stats["ntr"].mean()),
                 cov=float(stats["cov"].mean()), ent=float(stats["entropy"].mean()), kl=float(stats["approx_kl"].mean()),
                 clip=float(stats["clip_frac"].mean()), ev=float(stats["ev"].mean()), vl=float(stats["vl"].mean()))
        if (it + 1) % curve_its == 0 or it + 1 == n_it:
            do_curve(it + 1)
            if select_best_patience is not None:
                stalled = (it + 1) * per_it * P.n_envs - best["step"] >= select_best_patience
                if np.all(stalled):
                    C.info("early stop: every lane stalled", it=it + 1, patience=select_best_patience,
                           best_steps=best["step"].tolist())
                    it += 1
                    n_it = it
                    break
        if mgr is not None and ((it + 1) % ckpt_every_it == 0) and it + 1 < n_it:
            import orbax.checkpoint as ocp
            mgr.save(it + 1, args=ocp.args.StandardSave(lanes))
            mgr.wait_until_finished()
            C.write_json(os.path.join(out_dir, f"progress_{enc}.json"), dict(hist=hist, curves=curves,
                                                                              curves_det=curves_det))
            if select_best_patience is not None:
                tmp = best_file(it + 1) + ".tmp"
                with open(tmp, "wb") as f:
                    pickle.dump(dict(it=it + 1, ret=best["ret"], step=best["step"],
                                     params=None if best["params"] is None else
                                     jax.tree.map(np.asarray, best["params"])), f)
                os.replace(tmp, best_file(it + 1))
                kept = set(mgr.all_steps())                      # keep only the states of the kept checkpoints
                for fn in os.listdir(out_dir):
                    if fn.startswith(f"best_state_{enc}_") and fn.endswith(".pkl"):
                        s = int(fn[len(f"best_state_{enc}_"):-4])
                        if s not in kept:
                            os.remove(os.path.join(out_dir, fn))
        it += 1
    n_done = n_it - start_it
    sps = (n_done - 1) * per_it * P.n_envs * L / t_steady if n_done > 1 and t_steady > 0 else float("nan")
    C.info("training done", steady_env_steps_per_sec_all_lanes=round(sps, 1), compile_s=round(t_compile or 0, 1),
           steady_s=round(t_steady, 1))
    # save final params (numpy) per lane
    with open(os.path.join(out_dir, f"params_{enc}.pkl"), "wb") as f:
        pickle.dump(dict(lanes=lanes_meta, params=jax.tree.map(np.asarray, lanes.params), cfg=to_dict(cfg)), f)
    if select_best_patience is not None and best["params"] is not None:
        with open(os.path.join(out_dir, f"params_best_{enc}.pkl"), "wb") as f:
            pickle.dump(dict(lanes=lanes_meta, params=jax.tree.map(np.asarray, best["params"]), best_ret=best["ret"],
                             best_step=best["step"], cfg=to_dict(cfg)), f)
        lanes = lanes._replace(params=best["params"])     # the evaluated policy is each lane's best model
    timing = dict(pretrain_s=t_pre, first_iter_s=t_compile, steady_s=t_steady, steady_sps_all_lanes=sps,
                  n_lanes=L, iterations=n_it)
    C.write_json(os.path.join(out_dir, f"progress_{enc}.json"), dict(hist=hist, curves=curves, timing=timing,
                                                                      curves_det=curves_det if select_deterministic else None))
    summary = []
    for i, (a, s) in enumerate(lanes_meta):
        d = dict(arm=a, seed=s, curve=curves[i], train_hist={k: [h[k][i] for h in hist] for k in (hist[0] if hist else {})},
                 pretrain={k: np.asarray(v[i]).tolist() for k, v in pre_np.items()}, timing=timing,
                 env_version=data.env_version(cfg.env), config=to_dict(cfg))
        summary.append(d)
    return net, lanes, lanes_meta, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--curve-every", type=int, default=100_000)
    ap.add_argument("--ae-steps", type=int, default=None)
    ap.add_argument("--bc-episodes", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--select-best-patience", type=int, default=None,
                    help="paper §3 early stop: env steps without validation improvement (2_500_000 in the paper)")
    ap.add_argument("--curve-log", default=None, help="write validation curves here instead of stdout")
    ap.add_argument("--select-deterministic", action="store_true",
                    help="best-model selection on the deterministic validation return (SB3 EvalCallback default)")
    args = ap.parse_args()
    C.check_device()
    cfg = Config()
    if args.ae_steps is not None:
        cfg = replace(cfg, pre=replace(cfg.pre, ae_steps=args.ae_steps))
    if args.bc_episodes is not None:
        cfg = replace(cfg, pre=replace(cfg.pre, bc_episodes=args.bc_episodes))
    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    _, _, _, summary = run_family(cfg, arms, seeds, args.out, args.steps, args.curve_every,
                                  select_best_patience=args.select_best_patience, curve_log=args.curve_log,
                                  select_deterministic=args.select_deterministic)
    for d in summary:
        C.write_json(os.path.join(args.out, d["arm"], f"seed_{d['seed']}", "train.json"), d)


if __name__ == "__main__":
    main()

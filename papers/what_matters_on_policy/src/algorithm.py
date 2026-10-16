"""The configurable on-policy agent (paper App. B; official seed_rl agents/policy_gradient), as pure JAX functions
for ONE lane (= one arm x one seed). ``train.py`` vmaps them over lanes.

Per iteration (official learner.training_step + ppo_training_step_utils, README):
 1. collect T = iteration_size / num_envs steps from N envs with the current policy and the CURRENT obs stats;
 2. update the obs statistics once with the T*N observations of the batch (bootstrap obs excluded);
 3. for each of ``epochs`` passes: if (first pass) or (recompute mode, C5): recompute targets and advantages with the
    current V on the full fragments (GAE) -- value normaliser C66 = PopArt(compensate=False): bootstrap values
    are unnormalised with the pre-update stats, targets normalised with them, advantages divided by their std
    , then the stats are updated with the targets; shuffle the T*N transitions into minibatches of B and take one
    clipped-gradient Adam step on each.

Loss (official GeneralizedOnPolicyLoss): mean policy loss + baseline_cost * mean value loss + regulariser.
  policy (PPO, B.3):  -log pi(a) * sg(rho * A) * mask,  mask = 0 where (A>0, log rho > log(1+eps)) or
                          (A<0, log rho < -log(1+eps)); this has the gradient of -min(rho A, clip(rho,1/(1+eps),1+eps) A).
  value (C11, C13):       (y - V)^2 (keras MSE, no 1/2); with PPO-style clipping max((y-V)^2, (y-clip(V, V_old+-e))^2).
  entropy penalty (C46):  coef * mean(-H) (one-sample tanh entropy).
Optimiser: keras Adam (eps-hat form), lr linear to 0 per optimiser step over the 2M-frame horizon recorded
in the authors' study rows, joint global-norm gradient clip.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from . import data as D
from . import model as M


class LaneCfg(NamedTuple):
    """Per-lane (traced) choices. Every field is a scalar array; arms differ only here and in the init."""
    obs_norm: Any          # C64 bool
    obs_clip: Any          # C65 float (inf when obs_norm off)
    tanh: Any              # C63 bool (True = tanh, False = clip)
    softplus: Any          # C60 bool (True = softplus, False = safe_exp)
    init_std: Any          # C61
    min_std: Any           # C62
    ppo_eps: Any           # C16
    vclip_eps: Any         # C13 (<= 0 means off)
    gae_lambda: Any        # C8
    gamma: Any             # C20
    lr: Any                # C24
    lr_horizon_steps: Any  # optimiser steps over which lr decays to lr * lr_end 
    lr_end: Any            # C31 (0.0)
    adam_b1: Any           # C26
    adam_b2: Any
    adam_eps: Any          # C28
    grad_clip: Any         # C68 (<= 0 means off)
    ent_coef: Any          # C46 (0 = no regulariser)
    value_norm: Any        # C66 bool
    adv_norm: Any          # C67 bool
    recompute: Any         # C5 bool ('shuffle transitions (recompute advantages)')
    shared: Any            # C47 bool
    baseline_cost: Any     # C54 (also multiplies the value loss for separate nets)
    reward_norm: Any       # A7b EXTRA bool (ppo2 VecNormalize-style return-std reward scaling; not in paper)
    logp_presquash: Any    # bool: study-era distribution semantics (seed_rl@68c15aa; store the pre-squash x, log-prob at
                           # x minus the postprocessor log-det; README). True by default; False = released-HEAD
                           # semantics, diagnostic arms X3/X4 only.


class Static(NamedTuple):
    num_envs: int = 256        # C1
    unroll: int = 8            # C2 / C1 = 2048 / 256
    epochs: int = 10           # C3
    minibatch: int = 64        # C4
    episode_length: int = 1000
    pi_depth: int = 2
    v_depth: int = 2


class Adam(NamedTuple):
    m: Any
    v: Any
    t: jax.Array


class LaneState(NamedTuple):
    params: Any
    opt: Adam
    obs_ms: D.MeanStd          # C64 stats
    val_ms: D.MeanStd          # C66 stats (size 1)
    ret_ms: D.MeanStd          # A7b reward-scaling return stats (size 1)
    ret_acc: jax.Array         # (N) discounted return accumulator for A7b
    env: Any                   # data.EnvState
    obs: jax.Array             # (N, obs_dim) current raw observation
    it: jax.Array              # iterations done
    bad: jax.Array             # NaN/Inf seen (lane frozen)


class Batch(NamedTuple):
    obs: jax.Array      # (T+1, N, d) raw obs incl. bootstrap
    act: jax.Array      # (T, N, A) stored action: pre-squash x (logp_presquash) or tanh(x)/clip(x) (HEAD)
    mu_b: jax.Array     # (T, N, A) behaviour mean
    std_b: jax.Array    # (T, N, A) behaviour std
    v_b: jax.Array      # (T, N) behaviour value output (for value clipping)
    rew: jax.Array      # (T, N) training reward (scaled only in A7b)
    done: jax.Array     # (T, N)


# ------------------------------------------------------------------ optimiser (optax transformations)
class Adam(NamedTuple):
    """State of ``keras_adam`` (optax-style): first/second moments and the step counter t (optimizer.iterations)."""
    m: Any
    v: Any
    t: jax.Array


def lr_at(cfg: LaneCfg, step):
    """keras PolynomialDecay(lr, decay_steps, end=lr*lr_end, power=1) at the step counter BEFORE the update."""
    frac = jnp.minimum(step.astype(jnp.float32), cfg.lr_horizon_steps) / cfg.lr_horizon_steps
    return cfg.lr + (cfg.lr * cfg.lr_end - cfg.lr) * frac


def keras_adam(cfg: LaneCfg) -> optax.GradientTransformation:
    """tf.keras Adam (OptimizerV2): lr_t = lr(t-1) sqrt(1-b2^t)/(1-b1^t); update = -lr_t m / (sqrt(v) + eps) -- the
    'eps-hat' form (optax.adam adds eps after bias correction instead). Per-lane lr/schedule from ``cfg``."""
    def init(params):
        z = jax.tree.map(jnp.zeros_like, params)
        return Adam(z, jax.tree.map(jnp.zeros_like, params), jnp.zeros((), jnp.int32))

    def update(grads, opt, params=None):
        lr = lr_at(cfg, opt.t)
        t = opt.t + 1
        tf_ = t.astype(jnp.float32)
        lr_t = lr * jnp.sqrt(1.0 - cfg.adam_b2 ** tf_) / (1.0 - cfg.adam_b1 ** tf_)
        m = jax.tree.map(lambda m, g: cfg.adam_b1 * m + (1.0 - cfg.adam_b1) * g, opt.m, grads)
        v = jax.tree.map(lambda v, g: cfg.adam_b2 * v + (1.0 - cfg.adam_b2) * g * g, opt.v, grads)
        upd = jax.tree.map(lambda m, v: -(lr_t * m / (jnp.sqrt(v) + cfg.adam_eps)), m, v)
        return upd, Adam(m, v, t)

    return optax.GradientTransformation(init, update)


def clip_by_global_norm_tf(max_norm) -> optax.GradientTransformation:
    """tf.clip_by_global_norm over ALL trainable parameters jointly: g * min(1, c/||g||); c <= 0 = off."""
    def update(grads, state, params=None):
        return clip_grads(grads, max_norm)[0], state

    return optax.GradientTransformation(lambda p: optax.EmptyState(), update)


def optimizer(cfg: LaneCfg) -> optax.GradientTransformation:
    return optax.chain(clip_by_global_norm_tf(cfg.grad_clip), keras_adam(cfg))


def adam_init(params) -> Adam:
    return keras_adam(None).init(params)


def adam_step(params, grads, opt: Adam, cfg: LaneCfg):
    """One keras-Adam step on already-clipped gradients (tests / official parity)."""
    upd, opt = keras_adam(cfg).update(grads, opt)
    return optax.apply_updates(params, upd), opt


def global_norm(tree):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(tree)))


def clip_grads(grads, clip):
    n = global_norm(grads)
    scale = jnp.where(clip > 0, jnp.minimum(1.0, clip / jnp.maximum(n, 1e-30)), 1.0)
    return jax.tree.map(lambda g: g * scale, grads), n


# ------------------------------------------------------------------ advantages (B.2, README)
def gae(values, rewards, done, gamma, lam):
    """values (T+1, N), rewards/done (T, N). done[t] = episode ended at step t (terminal, no bootstrap).
    Returns (targets, advantages) of shape (T, N): A_t = d_t + gamma*lam*(1-done_t)*A_{t+1}, A_T = 0."""
    nd = 1.0 - done.astype(values.dtype)
    delta = rewards + gamma * nd * values[1:] - values[:-1]

    def body(acc, x):
        d, m = x
        acc = d + gamma * lam * m * acc
        return acc, acc

    _, adv = jax.lax.scan(body, jnp.zeros_like(values[0]), (delta, nd), reverse=True)
    return adv + values[:-1], adv


def compute_advantages(params, val_ms: D.MeanStd, batch: Batch, obs_ms: D.MeanStd, cfg: LaneCfg, st: Static):
    """Official GeneralizedOnPolicyLoss.compute_advantages. Returns (norm_targets, norm_adv, new val_ms)."""
    obs_n = D.normalize_obs(obs_ms, batch.obs, cfg.obs_norm, cfg.obs_clip)
    _, _, v_out = M.forward(params, obs_n, cfg.shared, st.pi_depth, st.v_depth)        # (T+1, N)
    vm, vsd = D.meanstd_get(val_ms)
    vm, vsd = jnp.where(cfg.value_norm, vm[0], 0.0), jnp.where(cfg.value_norm, vsd[0], 1.0)
    v_un = v_out * vsd + vm                                                           # unnormalize_prediction
    targets, adv = gae(v_un, batch.rew, batch.done, cfg.gamma, cfg.gae_lambda)
    n_targets = (targets - vm) / vsd                                                  # normalize_target (old stats)
    n_adv = adv / vsd                                                                 # normalize_advantage 
    new_ms = D.meanstd_update(val_ms, targets[..., None])
    val_ms = jax.tree.map(lambda a, b: jnp.where(cfg.value_norm, a, b), new_ms, val_ms)
    return jax.lax.stop_gradient(n_targets), jax.lax.stop_gradient(n_adv), val_ms


# ------------------------------------------------------------------ loss (official GeneralizedOnPolicyLoss.__call__)
class MB(NamedTuple):
    obs_n: jax.Array   # (B, d) normalised obs
    act: jax.Array
    mu_b: jax.Array
    std_b: jax.Array
    v_b: jax.Array
    target: jax.Array  # normalised target
    adv: jax.Array     # (normalised by C66 std) advantage


def loss_fn(params, mb: MB, cfg: LaneCfg, key, st: Static, pg_form: str = "official"):
    """pg_form 'official' (trained; -log pi * sg(rho A mask)) or 'min' (the paper's min-form, a true function whose
    gradient is identical off the clip kinks; used for finite-difference checks)."""
    mu, x_rho, v = M.forward(params, mb.obs_n, cfg.shared, st.pi_depth, st.v_depth)
    shift = M.std_shift(cfg.init_std, cfg.min_std, cfg.softplus)
    std = M.std_of(x_rho, shift, cfg.min_std, cfg.softplus)
    logp = M.log_prob(mb.act, mu, std, cfg.tanh, cfg.logp_presquash)
    logp_b = M.log_prob(mb.act, mb.mu_b, mb.std_b, cfg.tanh, cfg.logp_presquash)
    adv = mb.adv
    am = adv - adv.mean()
    adv = jnp.where(cfg.adv_norm, am / (jnp.sqrt(jnp.mean(am * am)) + 1e-8), adv)   # C67: population std + 1e-8
    log_rho = jax.lax.stop_gradient(logp - logp_b)
    le = jnp.log1p(cfg.ppo_eps)
    clipped = ((adv > 0) & (log_rho > le)) | ((adv < 0) & (log_rho < -le))
    mask = 1.0 - clipped.astype(adv.dtype)
    pg = jnp.mean(-logp * jax.lax.stop_gradient(adv * mask * jnp.exp(log_rho * mask)))
    rho = jnp.exp(logp - logp_b)                       # differentiable (min-form)
    pg_minform = jnp.mean(-jnp.minimum(rho * adv, jnp.clip(rho, 1.0 / (1.0 + cfg.ppo_eps), 1.0 + cfg.ppo_eps) * adv))
    vl = jnp.square(mb.target - v)
    vc = jnp.clip(v, mb.v_b - cfg.vclip_eps, mb.v_b + cfg.vclip_eps)
    vl = jnp.where(cfg.vclip_eps > 0, jnp.maximum(vl, jnp.square(mb.target - vc)), vl)
    vloss = jnp.mean(vl)
    ent = M.entropy(key, mu, std, cfg.tanh)
    reg = cfg.ent_coef * jnp.mean(-ent)
    total = (pg if pg_form == "official" else pg_minform) + cfg.baseline_cost * vloss + reg
    aux = {"pg_loss": pg_minform, "v_loss": vloss, "entropy": jnp.mean(ent), "clipfrac": 1.0 - mask.mean(),
           "approx_kl": jnp.mean(-log_rho), "std": jnp.mean(std), "adv_absmean": jnp.mean(jnp.abs(mb.adv))}
    return total, aux


# ------------------------------------------------------------------ one training iteration
def collect(env, ls: LaneState, cfg: LaneCfg, st: Static, k_act, reset_pool):
    """T steps with the current policy and current obs stats. Returns (ls, batch, finished_returns (T,N))."""
    def step(carry, k):
        es, obs, ret_acc, ret_ms = carry
        obs_n = D.normalize_obs(ls.obs_ms, obs, cfg.obs_norm, cfg.obs_clip)
        mu, x_rho, v = M.forward(ls.params, obs_n, cfg.shared, st.pi_depth, st.v_depth)
        std = M.std_of(x_rho, M.std_shift(cfg.init_std, cfg.min_std, cfg.softplus), cfg.min_std, cfg.softplus)
        a, x = M.sample_x(k, mu, std, cfg.tanh)
        es2, obs2, r, done, fin = D.env_step(env, es, a, reset_pool, st.episode_length)
        a_store = jnp.where(cfg.logp_presquash, x, a)
        # A7b EXTRA: baselines VecNormalize reward scaling (ret <- g*ret + r; RMS update; r/sqrt(var+1e-8) clip 10; reset)
        ret_acc2 = ret_acc * cfg.gamma + r
        ret_ms2 = D.meanstd_update(ret_ms, ret_acc2[:, None])
        _, rsd = D.meanstd_get(ret_ms2)
        r_scaled = jnp.clip(r / jnp.sqrt(rsd[0] ** 2 + 1e-8), -10.0, 10.0)
        r_train = jnp.where(cfg.reward_norm, r_scaled, r)
        ret_acc2 = jnp.where(done, 0.0, ret_acc2)
        out = (obs, a_store, mu, std, v, r_train, done, fin)
        return (es2, obs2, ret_acc2, ret_ms2), out

    keys = jax.random.split(k_act, st.unroll)
    (es, obs, ret_acc, ret_ms), (o, a, mu, std, v, r, d, fin) = jax.lax.scan(
        step, (ls.env, ls.obs, ls.ret_acc, ls.ret_ms), keys)
    batch = Batch(jnp.concatenate([o, obs[None]], 0), a, mu, std, v, r, d)
    ls = ls._replace(env=es, obs=obs, ret_acc=ret_acc, ret_ms=ret_ms)
    return ls, batch, fin, d


def update(ls: LaneState, batch: Batch, cfg: LaneCfg, st: Static, k_mb, perms=None):
    """Obs-stat update, then ``epochs`` passes of shuffled minibatch Adam steps (advantages recomputed per C5).
    ``perms`` (epochs, T*N) replaces the key-derived shuffle; used ONLY by the official multi-epoch parity test."""
    T, N = batch.rew.shape
    obs_ms = jax.tree.map(lambda a, b: jnp.where(cfg.obs_norm, a, b),
                          D.meanstd_update(ls.obs_ms, batch.obs[:-1]), ls.obs_ms)
    flat = lambda x: x.reshape((T * N,) + x.shape[2:])
    obs_n = flat(D.normalize_obs(obs_ms, batch.obs[:-1], cfg.obs_norm, cfg.obs_clip))
    nmb = (T * N) // st.minibatch

    def epoch(carry, e):
        params, opt, val_ms, tgt, adv, bad = carry
        do = jnp.logical_or(e == 0, cfg.recompute)
        t2, a2, vms2 = compute_advantages(params, val_ms, batch, obs_ms, cfg, st)
        tgt = jnp.where(do, flat(t2), tgt)
        adv = jnp.where(do, flat(a2), adv)
        val_ms = jax.tree.map(lambda a, b: jnp.where(do, a, b), vms2, val_ms)
        ke = jax.random.fold_in(k_mb, e)
        perm = jax.random.permutation(jax.random.fold_in(ke, 0), T * N) if perms is None else perms[e]
        perm = perm.reshape(nmb, st.minibatch)

        def mstep(c, x):
            params, opt, bad = c
            idx, j = x
            mb = MB(obs_n[idx], flat(batch.act)[idx], flat(batch.mu_b)[idx], flat(batch.std_b)[idx],
                    flat(batch.v_b)[idx], tgt[idx], adv[idx])
            (_, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, mb, cfg, jax.random.fold_in(ke, j + 1), st)
            gn = global_norm(g)
            upd, (_, o2) = optimizer(cfg).update(g, (optax.EmptyState(), opt), params)
            p2 = optax.apply_updates(params, upd)
            ok = jnp.isfinite(gn) & jnp.isfinite(global_norm(p2))
            params = jax.tree.map(lambda a, b: jnp.where(ok, a, b), p2, params)
            opt = jax.tree.map(lambda a, b: jnp.where(ok, a, b), o2, opt)
            aux["grad_norm"] = gn
            return (params, opt, bad | ~ok), aux

        (params, opt, bad), aux = jax.lax.scan(mstep, (params, opt, bad), (perm, jnp.arange(nmb)))
        out = jax.tree.map(jnp.mean, aux)
        #  diagnostic: the first minibatch of the epoch, before any gradient step of this epoch. In epoch 0 its
        # ratio differs from 1 only through the obs-stat ordering (behaviour stats pre-update, loss post-update)
        out["clipfrac_mb0"] = aux["clipfrac"][0]
        out["approx_kl_mb0"] = aux["approx_kl"][0]
        return (params, opt, val_ms, tgt, adv, bad), out

    z = jnp.zeros((T * N,), jnp.float32)
    (params, opt, val_ms, _, adv, bad), aux = jax.lax.scan(
        epoch, (ls.params, ls.opt, ls.val_ms, z, z, ls.bad), jnp.arange(st.epochs))
    first = {k: aux[k][0] for k in ("clipfrac_mb0", "approx_kl_mb0")}      # epoch 0 only 
    aux = {k: jnp.mean(v) for k, v in aux.items() if k not in first} | first
    aux["lr"] = lr_at(cfg, ls.opt.t)          # lr at this iteration's first optimiser step (diagnostic)
    vm, vsd = D.meanstd_get(val_ms)
    aux.update(popart_mean=vm[0], popart_std=vsd[0])
    return ls._replace(params=params, opt=opt, obs_ms=obs_ms, val_ms=val_ms, bad=bad), aux


def iteration(env, ls: LaneState, cfg: LaneCfg, st: Static, keys: dict):
    """keys: per-seed base keys {'env','act','mb'}; the iteration index is folded in (README)."""
    it = ls.it
    pool = jax.vmap(env.reset)(jax.random.split(jax.random.fold_in(keys["env"], it + 1), st.num_envs))
    ls, batch, fin, done = collect(env, ls, cfg, st, jax.random.fold_in(keys["act"], it), pool)
    ls, aux = update(ls, batch, cfg, st, jax.random.fold_in(keys["mb"], it))
    aux["ep_ret_sum"] = fin.sum()
    aux["ep_count"] = done.sum().astype(jnp.float32)
    # fraction of stored action dims at the squash saturation (|tanh x| >= 0.999 or |x| >= 1 for clip): the only
    # samples on which the study-era (pre-squash) and released-HEAD log-probs differ 
    post = jnp.where(cfg.logp_presquash, jnp.where(cfg.tanh, jnp.tanh(batch.act), jnp.clip(batch.act, -1.0, 1.0)), batch.act)
    aux["sat_frac"] = jnp.mean((jnp.abs(post) >= M.TANH_THRESHOLD).astype(jnp.float32))
    return ls._replace(it=it + 1), aux


def init_lane(env, spec: D.EnvSpec, params, st: Static, k_env) -> LaneState:
    es = D.env_reset(env, jax.random.split(jax.random.fold_in(k_env, 0), st.num_envs))
    return LaneState(params=params, opt=adam_init(params), obs_ms=D.meanstd_init(spec.obs_dim),
                     val_ms=D.meanstd_init(1), ret_ms=D.meanstd_init(1),
                     ret_acc=jnp.zeros((st.num_envs,), jnp.float32), env=es, obs=es.bstate.obs,
                     it=jnp.zeros((), jnp.int32), bad=jnp.zeros((), bool))


# ------------------------------------------------------------------ evaluation (paper §2)
def evaluate(env, params, obs_ms: D.MeanStd, cfg: LaneCfg, st: Static, key, n_episodes: int, stochastic=True):
    """``n_episodes`` fresh episodes (separate env instances, frozen obs stats), cap 1000 steps. Returns per-episode
    undiscounted return and length. Runs a while-loop until every episode of every lane has ended."""
    k_reset, k_act = jax.random.split(key)
    s0 = jax.vmap(env.reset)(jax.random.split(k_reset, n_episodes))
    shift = M.std_shift(cfg.init_std, cfg.min_std, cfg.softplus)

    def body(c):
        s, ret, ln, alive, t = c
        obs_n = D.normalize_obs(obs_ms, s.obs, cfg.obs_norm, cfg.obs_clip)
        mu, x_rho, _ = M.forward(params, obs_n, cfg.shared, st.pi_depth, st.v_depth)
        std = M.std_of(x_rho, shift, cfg.min_std, cfg.softplus)
        a = M.sample(jax.random.fold_in(k_act, t), mu, std, cfg.tanh) if stochastic else M.mode(mu, cfg.tanh)
        s2 = jax.vmap(env.step)(s, a)
        ret = ret + jnp.where(alive, s2.reward, 0.0)
        ln = ln + alive.astype(jnp.int32)
        alive = alive & (s2.done < 0.5) & (t + 1 < st.episode_length)
        return s2, ret, ln, alive, t + 1

    z = jnp.zeros((n_episodes,), jnp.float32)
    c = (s0, z, jnp.zeros((n_episodes,), jnp.int32), jnp.ones((n_episodes,), bool), jnp.zeros((), jnp.int32))
    _, ret, ln, _, _ = jax.lax.while_loop(lambda c: jnp.any(c[3]), body, c)
    return ret, ln

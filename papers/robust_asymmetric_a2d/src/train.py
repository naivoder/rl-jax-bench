"""Training driver: A2D, AIL (asymmetric DAgger with a fixed expert) and the TRPO baselines (RL(MDP), RL(Compact),
RL(Image) = symmetric, RL(Asym) = asymmetric critic), as one pure jitted iteration vmapped over seeds.

Arms differ ONLY in the mechanism under test (static Config fields); everything else is shared code.
Per iteration (official step()):
  a2d : rollout 2000 steps under the per-step beta-mixture -> push (s,b) latents to the 5000 FIFO ->
        expert step [V_beta = beta V_m(s) + (1-beta) V_p(b); GAE; V_m update; (Q_m update); A *= IS weight;
        TRPO on the expert] -> learner_val step [V_beta recomputed; V_p (Q_p) update] -> 2 AIL projection epochs
        -> beta <- 0.8 beta (snap to 0 below 1e-3)
  ail : pre-train (5000 expert steps + 2 epochs), then rollout under beta (1 at iteration 1, then 0) -> push -> 2 epochs
  rl  : rollout under the actor -> GAE with the actor's critic -> value update -> TRPO (no IS)
PRNG key tree: run_key = fold_in(root, seed); init = fold_in(run_key, 0);
iter_key[n] = fold_in(fold_in(run_key, 1), n) -> {0 reset, 1 act, 2 mix, 3 obs_noise, 4 v_perm, 5 il_perm,
6 fisher, 7 aug, 8 q_perm}; eval_key[n] = fold_in(fold_in(run_key, 2), n); AIL pre-training fold_in(run_key, 3);
A2D official buffer seed fold_in(run_key, 4).

CLI: python -m src.train --exp NAME --env FL --arm a2d_compact --seeds 0-9 [--iters 200] [--set key=val ...]
"""
from __future__ import annotations

import sys
from rl_jax_bench import gpu_bootstrap  # noqa: E402
# Non-deterministic GPU reductions: 2x faster conv backward (measured); runs remain seeded / statistically paired.
gpu_bootstrap.setup(deterministic=False)

import argparse          # noqa: E402
import dataclasses       # noqa: E402
import hashlib           # noqa: E402
import json              # noqa: E402
import os                # noqa: E402
import time              # noqa: E402
from typing import NamedTuple  # noqa: E402

import numpy as np       # noqa: E402
import jax               # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rl_jax_bench import jaxlog           # noqa: E402
from . import data as D  # noqa: E402
from . import model as M  # noqa: E402
from . import algorithm as AL  # noqa: E402
from .eval import make_evaluate  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclasses.dataclass(frozen=True)
class Config:
    env: str = 'FL'
    method: str = 'a2d'              # 'a2d' | 'ail' | 'rl'
    obs: str = 'partial'             # trainee input: 'partial' | 'image' (rl on the MDP uses obs='state')
    rl_actor: str = 'learner'        # rl only: 'expert' (= RL(MDP), acts on s) | 'learner'
    asym_value: bool = False         # rl only: learner value on s (RL(Asym), Pinto et al.)
    iters: int = 200                 # official --iters 201 -> 200 iterations
    batch: int = 2000                # App. D.1
    buffer: int = 5000               # App. D.1
    gamma: float = 0.995             # App. D.1
    lam_rl: float = 0.95             # App. D.1
    lam_a2d: float = -1.0            # -1 -> env default: FL 0.9, else 0.95 (App. D.1 / official)
    ent_rl: float = 1.0              # App. D.1
    ent_a2d: float = 10.0            # App. D.1
    reg_adv: bool = True             # official regularize_adv
    value_calc: str = 'gae'          # official value-calc: 'gae' | 'time'
    max_kl: float = 0.01             # App. D.1
    beta_decay: float = 0.8          # App. D.1 (0 -> beta = 0 from the start)
    use_q: bool = False              # App. B.1
    critic: str = 'paper'            # ablation: 'paper' (beta-mixture -> V_p(b)) | 'state' (V_m(s)) | 'mix05'
    critic_targets: str = 'official' # 'official': V_m and V_p are both fit to old_V + A of the MIXTURE critic (official
    #                                  code; only the mixture is anchored while 0 < beta < 1) | 'own': each net is fit
    #                                  to its own GAE lambda-return, as App. D.1 states ("individually targeting the sum of
    #                                  rewards ahead"); the mixture is used only for the expert's advantage
    estimator: str = 'official'      # TRPO surrogate: 'official' (centered, entropy-in-advantage) | 'paper'
    #                                  (Eq. 27: IS-weighted advantage scaled by its std, no centering, no entropy)
    kl_fallback: bool = True         # TRPO freeze fix: recompute a non-finite TRPO natural-gradient step from a
    #                                  log-prob KL (same value) instead of freezing; bitwise no-op until that happens
    loglik_clamp: bool = True        # TRPO old/new log-likelihoods and both IS-weight factors
    #                                  as the official torch Categorical.log_prob = log(clip(p, 2.22e-16, 1 - 2.22e-16))
    #                                  (zero gradient outside); False = exact log_softmax (pre-fix behavior)
    official_buffer_seed: bool = False  # (a2d only) True = the official initial_log() pushes one
    #                                  2000-step rollout of the untrained expert into the AIL buffer and (a swallowed
    #                                  TypeError) never resets it, so iteration 1 projects over 4000 samples. Off by
    #                                  default: it re-draws every trajectory from iteration 1 (destroying the seed pairing
    #                                  with earlier runs) for a predicted-negligible effect, measured separately
    il_lr: float = 3e-4              # App. D.1
    val_lr: float = 7e-4             # App. D.1
    q_lr: float = 3e-4               # App. D.1
    val_epochs: int = 25             # App. D.1
    q_epochs: int = 25               # official
    n_mb: int = 32                   # App. D.1
    clip: float = 0.2                # official --clip-eps
    il_mb: int = 64                  # App. D.1
    proj_epochs: int = 2             # App. D.1 / official --max-proj-iter
    fisher_frac: float = 0.1         # official
    damping: float = 0.1             # official
    eval_every: int = 5              # App. D.1
    augment: bool = True             # official data-aug for image AIL
    expert_ckpt: str = ''            # ail: npz of the RL(MDP) expert params (per seed: {seed} placeholder)
    dtype: str = 'float32'           # 'float64' = official dtype: network params, optimizer state and hence every
    #                                  TRPO surrogate / line-search / KL computation in float64. The full-scale runs found that in
    #                                  float32 the TRPO line search of the flat (compact) policies rejects every step
    #                                  once the policy is near-deterministic (README, "dtype").

    @property
    def lam(self) -> float:
        if self.method == 'rl':
            return self.lam_rl
        if self.lam_a2d >= 0:
            return self.lam_a2d
        return 0.9 if self.env == 'FL' else 0.95

    @property
    def ent(self) -> float:
        return self.ent_a2d if self.method == 'a2d' else self.ent_rl


ARMS = {
    'rl_mdp': dict(method='rl', rl_actor='expert', obs='partial'),
    'rl_compact': dict(method='rl', obs='partial'),
    'rl_image': dict(method='rl', obs='image'),
    'rl_asym': dict(method='rl', obs='image', asym_value=True),
    'ail_compact': dict(method='ail', obs='partial'),
    'ail_image': dict(method='ail', obs='image'),
    'a2d_compact': dict(method='a2d', obs='partial'),
    'a2d_image': dict(method='a2d', obs='image'),
    # App. B.1 / Fig. B.1 (official Q_arguments: beta = 0, lambda 0.5, KL 1e-3, entropy 0.02 on the surrogate, MC values)
    'a2d_q': dict(method='a2d', obs='partial', use_q=True, beta_decay=0.0, lam_a2d=0.5, max_kl=0.001,
                  ent_a2d=0.02, ent_rl=0.02, reg_adv=False, value_calc='time'),
    'a2d_noq': dict(method='a2d', obs='partial', use_q=False, beta_decay=0.0, lam_a2d=0.5, max_kl=0.001,
                    ent_a2d=0.02, ent_rl=0.02, reg_adv=False, value_calc='time'),
}


class Buffer(NamedTuple):
    pos: jnp.ndarray
    h: jnp.ndarray
    illum: jnp.ndarray
    nk: jnp.ndarray          # per-sample observation-noise key (images are regenerated exactly)
    ptr: jnp.ndarray
    length: jnp.ndarray


class TrainState(NamedTuple):
    params: dict
    opt: dict
    beta: jnp.ndarray
    damping: jnp.ndarray
    buf: Buffer
    it: jnp.ndarray
    best_ret: jnp.ndarray
    best_expert: dict
    proj_loss: jnp.ndarray


def tree_where(c, a, b):
    return jax.tree_util.tree_map(lambda x, y: jnp.where(c, x, y), a, b)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = c = cfg
        self.spec = spec = D.get_spec(c.env)
        self.table = jnp.asarray(D.image_table(c.env)) if c.obs == 'image' else None
        self.pol_e, self.val_e = M.make_policy('flat'), M.make_value('flat')
        lk = 'image' if c.obs == 'image' else 'flat'
        self.pol_l = M.make_policy(lk)
        self.val_l = M.make_value('flat' if c.asym_value else lk)
        self.q_net = M.make_q()
        self.obs_l = D.obs_fn(spec, c.obs, self.table)
        self.state_f = lambda lat: D.state_vec(spec, lat)
        self.opt_il, self.opt_v, self.opt_q = AL.adam(c.il_lr), AL.adam(c.val_lr), AL.adam(c.q_lr)
        self.trpo_cfg = AL.TRPOCfg(max_kl=c.max_kl, ent=c.ent, reg_adv=c.reg_adv, estimator=c.estimator,
                                   kl_fallback=c.kl_fallback, loglik_clamp=c.loglik_clamp)
        self.n_fisher = int(c.batch * c.fisher_frac)
        self.n_proj = AL.n_proj_batches(c.buffer, c.il_mb)
        if c.method == 'rl':
            lanes = (('expert', False), ('expert', True)) if c.rl_actor == 'expert' else \
                (('learner', False), ('learner', True))
        else:
            lanes = (('learner', False), ('learner', True), ('expert', False), ('expert', True))
        self.lanes = lanes
        self.evaluate = make_evaluate(spec, self.obs_l, self.state_f, self.pol_e, self.pol_l, lanes,
                                      need_kl=c.method != 'rl')

    # ------------------------------------------------------------------ init
    def init(self, run_key, expert_params=None) -> TrainState:
        c, spec = self.cfg, self.spec
        ks = jax.random.split(jax.random.fold_in(run_key, 0), 6)
        s0 = jnp.zeros((spec.state_dim,))
        lat0 = D.Latent(jnp.int32(0), jnp.int32(0), jnp.array(False))
        o0 = self.obs_l(lat0, jax.random.PRNGKey(0))
        vin0 = s0 if c.asym_value else o0
        p = {'expert_pol': self.pol_e.init(ks[0], s0), 'expert_val': self.val_e.init(ks[1], s0),
             'learner_pol': self.pol_l.init(ks[2], o0), 'learner_val': self.val_l.init(ks[3], vin0)}
        if c.use_q:
            p['expert_q'] = self.q_net.init(ks[4], jnp.zeros((spec.state_dim + 4,)))
            p['learner_q'] = self.q_net.init(ks[5], jnp.zeros((o0.shape[0] + 4,)))
        if expert_params is not None:
            p['expert_pol'] = expert_params
        opt = {'learner_pol': self.opt_il.init(p['learner_pol']), 'learner_val': self.opt_v.init(p['learner_val']),
               'expert_val': self.opt_v.init(p['expert_val'])}
        if c.use_q:
            opt['expert_q'] = self.opt_q.init(p['expert_q'])
            opt['learner_q'] = self.opt_q.init(p['learner_q'])
        cap = c.buffer
        buf = Buffer(jnp.zeros(cap, jnp.int32), jnp.zeros(cap, jnp.int32), jnp.zeros(cap, bool),
                     jnp.zeros((cap, 2), jnp.uint32), jnp.int32(0), jnp.int32(0))
        if c.method == 'a2d':
            beta = 1.0 if c.beta_decay > 0 else 0.0
        elif c.method == 'ail':
            beta = 1.0
        else:
            beta = 1.0 if c.rl_actor == 'expert' else 0.0
        proj0 = jnp.float32(jnp.nan)
        if c.dtype == 'float64':
            f64 = lambda t: jax.tree_util.tree_map(lambda x: x.astype(jnp.float64) if x.dtype == jnp.float32 else x, t)
            p, opt, proj0 = f64(p), f64(opt), jnp.float64(jnp.nan)
        return TrainState(p, opt, jnp.float32(beta), jnp.float32(c.damping), buf, jnp.int32(0),
                          jnp.float32(-1e9), p['expert_pol'], proj0)

    # ------------------------------------------------------------------ rollout
    def rollout(self, params, beta, key, T: int, actor: str):
        """T env steps, one env, auto-reset (official sample_batch with exactly T samples, fresh episode first).
        actor: 'mix' (per-step Bernoulli(beta) between expert and trainee), 'expert' or 'learner'."""
        spec = self.spec
        k_reset, k_act, k_mix, k_noise = (jax.random.fold_in(key, i) for i in range(4))
        st0 = D.reset(spec, jax.random.fold_in(k_reset, 0))
        need_e, need_l = actor in ('mix', 'expert'), actor in ('mix', 'learner')

        def body(st, t):
            lat = D.latent(spec, st)
            nk = jax.random.fold_in(k_noise, t)
            ka = jax.random.fold_in(k_act, t)
            a_e = jax.random.categorical(jax.random.fold_in(ka, 0), self.pol_e.apply(params['expert_pol'],
                                                                                    self.state_f(lat))) if need_e else 0
            a_l = jax.random.categorical(jax.random.fold_in(ka, 1), self.pol_l.apply(params['learner_pol'],
                                                                                    self.obs_l(lat, nk))) if need_l else 0
            if actor == 'mix':
                coin = jax.random.bernoulli(jax.random.fold_in(k_mix, t), beta)
                a = jnp.where(coin, a_e, a_l)
            else:
                a = a_e if actor == 'expert' else a_l
            st2, r, done = D.step(spec, st, a)
            st2 = tree_where(done, D.reset(spec, jax.random.fold_in(k_reset, t + 1)), st2)
            return st2, (lat, nk, a, r, done)
        _, (lat, nk, A, R, done) = jax.lax.scan(body, st0, jnp.arange(T))
        return lat, nk, A.astype(jnp.int32), R, ~done

    def inputs(self, lat, nk):
        S = jax.vmap(self.state_f)(lat)
        O = jax.vmap(self.obs_l)(lat, nk)
        return S, O

    def push(self, buf: Buffer, lat, nk) -> Buffer:
        cap, n = self.cfg.buffer, lat.pos.shape[0]
        idx = (buf.ptr + jnp.arange(n)) % cap
        return Buffer(buf.pos.at[idx].set(lat.pos), buf.h.at[idx].set(lat.h), buf.illum.at[idx].set(lat.illum),
                      buf.nk.at[idx].set(nk), (buf.ptr + n) % cap, jnp.minimum(buf.length + n, cap))

    # ------------------------------------------------------------------ pieces
    def _perms(self, key, n_ep, N):
        return jax.vmap(lambda k: jax.random.permutation(k, N))(jax.random.split(key, n_ep))

    def mixture_value(self, params, beta, S, O):
        c = self.cfg
        vm = self.val_e.apply(params['expert_val'], S)
        vp = self.val_l.apply(params['learner_val'], S if c.asym_value else O)
        if c.critic == 'state':
            return vm
        if c.critic == 'mix05':
            return 0.5 * vm + 0.5 * vp
        return jnp.where(beta > 0, beta * vm + (1 - beta) * vp, vp)

    def projection(self, ts: TrainState, key, n_epochs: int):
        """n_epochs AIL epochs over the FIFO buffer (official mini_batch_update; forward KL, analytic)."""
        c, b = self.cfg, ts.buf
        lat = D.Latent(b.pos, b.h, b.illum)
        S = jax.vmap(self.state_f)(lat)
        p1 = AL.masked_perm(jax.random.fold_in(key, 0), b.length, c.buffer)
        p2 = AL.masked_perm(jax.random.fold_in(key, 1), b.length, c.buffer)
        idx, valid = AL.epoch_indices(p1, p2, b.length, c.il_mb, self.n_proj)
        e_params = ts.params['expert_pol']
        image = c.obs == 'image'

        def loss_fn(p, ii, k):
            e_lp = self.pol_e.apply(e_params, S[ii])
            li = D.Latent(b.pos[ii], b.h[ii], b.illum[ii])
            O = jax.vmap(self.obs_l)(li, b.nk[ii])
            if image and c.augment:
                O = AL.augment(k, O)
            return AL.kl_loss(e_lp, self.pol_l.apply(p, O))
        lp, ost = ts.params['learner_pol'], ts.opt['learner_pol']
        loss = jnp.float32(jnp.nan)
        for e in range(n_epochs):
            keys = jax.random.split(jax.random.fold_in(key, 10 + e), self.n_proj)
            lp, ost, loss = AL.projection_epoch(loss_fn, lp, ost, self.opt_il, idx[e], valid[e], keys)
        params = dict(ts.params, learner_pol=lp)
        opt = dict(ts.opt, learner_pol=ost)
        return ts._replace(params=params, opt=opt, proj_loss=loss)

    # ------------------------------------------------------------------ iterations
    def iteration(self, ts: TrainState, ikey):
        c = self.cfg
        k = lambda i: jax.random.fold_in(ikey, i)
        params, opt = dict(ts.params), dict(ts.opt)
        info = {}
        if c.method == 'rl':
            actor = c.rl_actor
            lat, nk, A, R, ND = self.rollout(params, ts.beta, k(0), c.batch, actor)
            S, O = self.inputs(lat, nk)
            pol, val = (self.pol_e, self.val_e) if actor == 'expert' else (self.pol_l, self.val_l)
            pk, vk = ('expert_pol', 'expert_val') if actor == 'expert' else ('learner_pol', 'learner_val')
            X = S if actor == 'expert' else O
            VX = S if (actor == 'expert' or c.asym_value) else O
            old_lp = AL.action_logp(pol.apply(params[pk], X), A, c.loglik_clamp)
            V = val.apply(params[vk], VX)
            adv, ret = AL.gae(R, V, ND, c.gamma, c.lam)
            params[vk], opt[vk], info['vloss'] = AL.value_update(val.apply, params[vk], opt[vk], self.opt_v, VX, adv,
                                                                 ret, ND, c.value_calc,
                                                                 self._perms(k(4), c.val_epochs, c.batch), c.n_mb,
                                                                 c.clip)
            sel = jax.random.choice(k(6), c.batch, (self.n_fisher,), replace=False)
            params[pk], tinfo = AL.trpo_step(pol.apply, params[pk], X, A, old_lp, adv, self.trpo_cfg, sel, ts.damping)
            ts = ts._replace(params=params, opt=opt, damping=tinfo['damping'])
            info['trpo_ok'] = tinfo['accepted']
            info['kl_fallback'] = tinfo['kl_fallback']
            info['damping'] = tinfo['damping']
        elif c.method == 'ail':
            lat, nk, A, R, ND = self.rollout(params, ts.beta, k(0), c.batch, 'mix')
            ts = ts._replace(buf=self.push(ts.buf, lat, nk))
            ts = self.projection(ts, k(5), c.proj_epochs)
            ts = ts._replace(beta=jnp.float32(0.0))
            info['vloss'] = jnp.float32(0)
            info['trpo_ok'] = jnp.array(False)
        else:  # a2d
            beta = ts.beta
            lat, nk, A, R, ND = self.rollout(params, beta, k(0), c.batch, 'mix')
            S, O = self.inputs(lat, nk)
            buf = self.push(ts.buf, lat, nk)
            # ---- expert step (official trpo_step(..., 'expert'))
            old_lp = AL.action_logp(self.pol_e.apply(params['expert_pol'], S), A, c.loglik_clamp)
            info['n_clamped'] = jnp.sum(old_lp <= jnp.log(AL.EPS64)).astype(jnp.float32)   # taken a with p_theta < eps
            V = self.mixture_value(params, beta, S, O)
            adv, ret = AL.gae(R, V, ND, c.gamma, c.lam)
            if c.critic_targets == 'own':      # V_m fit to its own lambda-return
                adv_m, _ = AL.gae(R, self.val_e.apply(params['expert_val'], S), ND, c.gamma, c.lam)
            else:
                adv_m = adv
            params['expert_val'], opt['expert_val'], info['vloss_m'] = AL.value_update(
                self.val_e.apply, params['expert_val'], opt['expert_val'], self.opt_v, S, adv_m, ret, ND, c.value_calc,
                self._perms(k(4), c.val_epochs, c.batch), c.n_mb, c.clip)
            oh = jax.nn.one_hot(A, 4, dtype=S.dtype)
            if c.use_q:
                params['expert_q'], opt['expert_q'], _ = AL.q_update(
                    self.q_net.apply, params['expert_q'], opt['expert_q'], self.opt_q, jnp.concatenate([S, oh], -1),
                    ret, self._perms(k(8), c.q_epochs, c.batch), c.n_mb)
                qm = self.q_net.apply(params['expert_q'], jnp.concatenate([S, oh], -1))
                qp = self.q_net.apply(params['learner_q'], jnp.concatenate([O, oh], -1))
                adv = beta * qm + (1 - beta) * qp - self.mixture_value(params, beta, S, O)
            l_lp = AL.action_logp(self.pol_l.apply(params['learner_pol'], O), A, c.loglik_clamp)
            w = AL.importance_weight(old_lp, l_lp, beta)
            sel = jax.random.choice(k(6), c.batch, (self.n_fisher,), replace=False)
            params['expert_pol'], tinfo = AL.trpo_step(self.pol_e.apply, params['expert_pol'], S, A, old_lp, adv * w,
                                                       self.trpo_cfg, sel, ts.damping)
            info['trpo_ok'] = tinfo['accepted']
            info['kl_fallback'] = tinfo['kl_fallback']
            info['damping'] = tinfo['damping']
            info['iw_mean'] = w.mean()
            # ---- learner value step (official trpo_step(..., 'learner_val'))
            if c.critic_targets == 'own':      # V_p fit to its own lambda-return
                V = self.val_l.apply(params['learner_val'], S if c.asym_value else O)
            else:
                V = self.mixture_value(params, beta, S, O)
            adv2, ret2 = AL.gae(R, V, ND, c.gamma, c.lam)
            params['learner_val'], opt['learner_val'], info['vloss'] = AL.value_update(
                self.val_l.apply, params['learner_val'], opt['learner_val'], self.opt_v, O, adv2, ret2, ND,
                c.value_calc, self._perms(jax.random.fold_in(k(4), 1), c.val_epochs, c.batch), c.n_mb, c.clip)
            if c.use_q:
                params['learner_q'], opt['learner_q'], _ = AL.q_update(
                    self.q_net.apply, params['learner_q'], opt['learner_q'], self.opt_q, jnp.concatenate([O, oh], -1),
                    ret2, self._perms(jax.random.fold_in(k(8), 1), c.q_epochs, c.batch), c.n_mb)
            ts = ts._replace(params=params, opt=opt, buf=buf, damping=tinfo['damping'])
            ts = self.projection(ts, k(5), c.proj_epochs)
            nb = beta * c.beta_decay
            ts = ts._replace(beta=jnp.where(nb < 1e-3, 0.0, nb).astype(jnp.float32))
        info['train_ret_per_step'] = R.mean()
        return ts._replace(it=ts.it + 1), info

    def ail_pretrain(self, ts: TrainState, run_key):
        """Official pretrain(): 5000 expert-driven samples + a projection step. The official code sets
        params.max_proj_iter = pretraining_projection_steps (1) but projection_step() loops over self.max_proj_iter,
        fixed at construction to --max-proj-iter = 2, so the pre-training step runs TWO epochs (found by the
        whole-iteration differential test, tests/test_official_parity.py)."""
        key = jax.random.fold_in(run_key, 3)
        lat, nk, A, R, ND = self.rollout(ts.params, 1.0, jax.random.fold_in(key, 0), self.cfg.buffer, 'expert')
        ts = ts._replace(buf=self.push(ts.buf, lat, nk))
        return self.projection(ts, jax.random.fold_in(key, 5), self.cfg.proj_epochs)

    def a2d_buffer_seed(self, ts: TrainState, run_key):
        """Official initial_log() -> logging(0) on an empty buffer: one batch of stochastic samples of the untrained
        expert is pushed, and the hard_reset() after it is skipped (mini_batch_update(-1, verbose=False, update=False)
        raises a TypeError inside a bare try/except), so the samples stay in the FIFO.
        Key: fold_in(run_key, 4), unused elsewhere, so every other draw is unchanged."""
        key = jax.random.fold_in(run_key, 4)
        lat, nk, A, R, ND = self.rollout(ts.params, 1.0, jax.random.fold_in(key, 0), self.cfg.batch, 'expert')
        return ts._replace(buf=self.push(ts.buf, lat, nk))

    def eval_metrics(self, ts: TrainState, run_key):
        ek = jax.random.fold_in(jax.random.fold_in(run_key, 2), ts.it)
        ev = self.evaluate(ts.params, ek)
        out = {}
        for i, (a, d) in enumerate(self.lanes):
            tag = f"{a}_{'det' if d else 'sto'}"
            for kk, v in ev.items():
                out[f'{tag}/{kk}'] = v[i]
        return out

    def block(self, ts: TrainState, run_key, n_iter: int):
        """n_iter iterations then one evaluation (and the RL(MDP) best-expert bookkeeping)."""
        def body(t, _):
            ikey = jax.random.fold_in(jax.random.fold_in(run_key, 1), t.it)
            t, info = self.iteration(t, ikey)
            return t, info
        ts, infos = jax.lax.scan(body, ts, None, length=n_iter)
        m = self.eval_metrics(ts, run_key)
        if self.cfg.method == 'rl' and self.cfg.rl_actor == 'expert':
            r = m['expert_sto/ret']
            better = r > ts.best_ret
            ts = ts._replace(best_ret=jnp.where(better, r, ts.best_ret),
                             best_expert=tree_where(better, ts.params['expert_pol'], ts.best_expert))
        m['proj_loss'] = ts.proj_loss
        m['beta_next'] = ts.beta
        for kk, v in infos.items():
            m[f'train/{kk}'] = jnp.mean(v.astype(jnp.float32))
        jaxlog.nan_guard({'p': ts.params}, {'it': ts.it}, tag=1, dump_dir=os.path.join(REPO, 'nan_dumps'))
        return ts, m


# ---------------------------------------------------------------------------------------- run driver
def parse_seeds(s: str):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]


def save_expert(path, best_expert, seeds):
    flat = {}
    for i, sd in enumerate(seeds):
        leaves, _ = jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: x[i], best_expert))
        for j, l in enumerate(leaves):
            flat[f's{sd}_{j}'] = np.asarray(l)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **flat)


def load_expert(path, seeds, template):
    d = np.load(path)
    leaves_t, treedef = jax.tree_util.tree_flatten(template)
    per = []
    for sd in seeds:
        per.append(jax.tree_util.tree_unflatten(treedef, [jnp.asarray(d[f's{sd}_{j}']) for j in range(len(leaves_t))]))
    return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *per)


CKPT_EVERY_BLOCKS = 10        # image arms: orbax checkpoint every 50 iterations (a 5-seed run exceeds 10 min)


class Ckpt:
    """orbax checkpoint of the full TrainState + the eval curves so far; resumed only if config + seeds match."""

    def __init__(self, run_dir, cfg, seeds):
        import orbax.checkpoint as ocp
        self.ocp = ocp
        self.dir = os.path.abspath(os.path.join(run_dir, 'ckpt'))
        self.meta = os.path.join(run_dir, 'ckpt_meta.json')
        self.sig = {'cfg': dataclasses.asdict(cfg), 'seeds': list(seeds)}
        self.mgr = ocp.CheckpointManager(self.dir, options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True))

    def save(self, ts, curves, block):
        self.mgr.save(block, args=self.ocp.args.StandardSave(ts))
        self.mgr.wait_until_finished()
        cur = [{k: (np.asarray(v).tolist() if k != 'it' else v) for k, v in c.items()} for c in curves]
        jaxlog.write_json(self.meta, dict(self.sig, block=block, curves=cur))

    def clear(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)
        if os.path.exists(self.meta):
            os.remove(self.meta)

    def restore(self, template):
        if not os.path.exists(self.meta):
            return None
        meta = json.load(open(self.meta))
        if meta['cfg'] != self.sig['cfg'] or meta['seeds'] != self.sig['seeds'] or self.mgr.latest_step() is None:
            jaxlog.warn('checkpoint signature mismatch -- starting fresh')
            return None
        ts = self.mgr.restore(meta['block'], args=self.ocp.args.StandardRestore(template))
        curves = [{k: (np.asarray(v) if k != 'it' else v) for k, v in c.items()} for c in meta['curves']]
        return ts, curves, meta['block']


def run(exp: str, arm: str, cfg: Config, seeds, top_seed: int = 0, out_root: str | None = None):
    out_root = out_root or os.path.join(REPO, 'runs', exp)
    jaxlog.check_device()
    tr = Trainer(cfg)
    root = jax.random.PRNGKey(top_seed)
    run_keys = jnp.stack([jax.random.fold_in(root, s) for s in seeds])
    expert = None
    if cfg.method == 'ail':
        tmpl = tr.pol_e.init(jax.random.PRNGKey(0), jnp.zeros((tr.spec.state_dim,)))
        expert = load_expert(cfg.expert_ckpt, seeds, tmpl)
        ts = jax.vmap(lambda k, e: tr.init(k, e))(run_keys, expert)
    else:
        ts = jax.vmap(lambda k: tr.init(k))(run_keys)
    # Image arms: lax.map over seeds (vmapped convs with per-seed weights lower to grouped convolutions that are
    # 2x slower than running the seeds one after another on the RTX 3090 — measured); flat arms: vmap.
    if cfg.obs == 'image':
        blk = jax.jit(lambda t, k: jax.lax.map(lambda x: tr.block(x[0], x[1], cfg.eval_every), (t, k)))
    else:
        blk = jax.jit(jax.vmap(lambda t, k: tr.block(t, k, cfg.eval_every)))
    ev0 = jax.jit(jax.vmap(tr.eval_metrics))
    t0 = time.time()
    if cfg.method == 'ail':
        ts = jax.jit(jax.vmap(tr.ail_pretrain))(ts, run_keys)
    if cfg.method == 'a2d' and cfg.official_buffer_seed:
        ts = jax.jit(jax.vmap(tr.a2d_buffer_seed))(ts, run_keys)
    m0 = jax.device_get(ev0(ts, run_keys))
    jax.block_until_ready(m0)
    jaxlog.info('init+eval0 compiled', exp=exp, arm=arm, env=cfg.env, secs=round(time.time() - t0, 1))
    curves = [dict(it=0, **{k: np.asarray(v) for k, v in m0.items()})]
    n_blocks = cfg.iters // cfg.eval_every
    ck = Ckpt(os.path.join(out_root, cfg.env, arm), cfg, seeds) if cfg.obs == 'image' else None
    b0 = 0
    if ck is not None:
        restored = ck.restore(ts)
        if restored is not None:
            ts, curves, b0 = restored
            jaxlog.info('resumed from checkpoint', block=b0, it=b0 * cfg.eval_every)
    t_first = None
    t1 = time.time()
    for b in range(b0, n_blocks):
        ts, m = blk(ts, run_keys)
        m = jax.device_get(m)
        if b == b0:
            t_first = time.time() - t1
            jaxlog.info('first block (compile + run)', secs=round(t_first, 1))
            t1 = time.time()
        it = (b + 1) * cfg.eval_every
        curves.append(dict(it=it, **{k: np.asarray(v) for k, v in m.items()}))
        main = 'expert_sto/ret' if (cfg.method == 'rl' and cfg.rl_actor == 'expert') else 'learner_sto/ret'
        jaxlog.metric(f'{arm} {cfg.env}', it=it, ret_mean=float(np.mean(m[main])), ret_min=float(np.min(m[main])),
                      ret_max=float(np.max(m[main])), proj=float(np.nanmean(m['proj_loss'])),
                      beta=float(np.mean(m['beta_next'])))
        if ck is not None and ((b + 1) % CKPT_EVERY_BLOCKS == 0) and b + 1 < n_blocks:
            ck.save(ts, curves, b + 1)
    steady = (time.time() - t1) / max(n_blocks - b0 - 1, 1)
    jaxlog.info('steady-state', secs_per_block=round(steady, 2),
                iters_per_sec=round(cfg.eval_every / max(steady, 1e-9), 2), seeds=len(seeds))
    if cfg.method == 'rl' and cfg.rl_actor == 'expert':
        save_expert(os.path.join(out_root, cfg.env, 'expert_best.npz'), jax.device_get(ts.best_expert), seeds)
    write_metrics(out_root, exp, arm, cfg, seeds, curves, t_first, steady)
    if ck is not None:
        ck.clear()
    return curves


def write_metrics(out_root, exp, arm, cfg, seeds, curves, t_first, steady):
    cfg_d = dataclasses.asdict(cfg)
    h = hashlib.sha256(json.dumps(cfg_d, sort_keys=True).encode()).hexdigest()[:12]
    spec = D.get_spec(cfg.env)
    pre = 5000 if cfg.method == 'ail' else 0
    for i, sd in enumerate(seeds):
        keys = sorted({k for c in curves for k in c if k != 'it'})       # it=0 eval lacks the training metrics
        cur = {k: [float(np.asarray(c[k])[i]) if (k in c and np.ndim(c[k])) else
                   (float(c[k]) if k in c else float('nan')) for c in curves] for k in keys}
        its = [c['it'] for c in curves]
        main = 'expert_sto/ret' if (cfg.method == 'rl' and cfg.rl_actor == 'expert') else 'learner_sto/ret'
        last = cur[main][-5:]
        final = {'ret_sto_last5': float(np.mean(last)), 'ret_sto_final': cur[main][-1],
                 'ret_norm_last5': float(np.mean(last)) - (spec.mdp_opt + 1.0)}
        for kk in ('learner_det/ret', 'expert_sto/ret', 'learner_sto/kl_clean', 'proj_loss',
                   'learner_sto/haz_rate', 'learner_sto/press_rate'):
            if kk in cur:
                final[kk.replace('/', '_') + '_last5'] = float(np.nanmean(cur[kk][-5:]))
        rec = {'exp': exp, 'arm': arm, 'seed': sd, 'config': cfg_d, 'config_hash': h,
               'iters': its, 'interactions': [pre + it * cfg.batch for it in its], 'curves': cur, 'final': final,
               # the paper's x-axis charges AIL the 8e4 interactions of training its MDP expert (Fig. 4
               # caption, App. D.1). Our expert is the best-eval checkpoint of a full 4e5-interaction RL(MDP) run.
               'interactions_paper_axis': [(80000 if cfg.method == 'ail' else 0) + pre + it * cfg.batch for it in its],
               'timing': {'first_block_s': t_first, 'steady_block_s': steady, 'n_seeds_vmapped': len(seeds)},
               'optima': {'mdp': spec.mdp_opt, 'pomdp': spec.pomdp_opt}}
        jaxlog.write_json(os.path.join(out_root, cfg.env, arm, f'seed_{sd}', 'metrics.json'), rec)
        # top-level per-seed index (required layout runs/seed_<n>/metrics.json)
        top = os.path.join(REPO, 'runs', f'seed_{sd}', 'metrics.json')
        idx = json.load(open(top)) if os.path.exists(top) else {}
        idx[f'{exp}/{cfg.env}/{arm}'] = {'final': final, 'config_hash': h,
                                         'detail': os.path.relpath(os.path.join(out_root, cfg.env, arm,
                                                                                f'seed_{sd}', 'metrics.json'), REPO)}
        jaxlog.write_json(top, idx)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--env', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--seeds', default='0-4')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--top-seed', type=int, default=0)
    ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args(argv)
    kw = dict(ARMS[a.arm.split('+')[0]])
    kw.update(env=a.env, iters=a.iters)
    for s in a.set:
        k, v = s.split('=')
        d = getattr(Config(), k)
        kw[k] = (v.lower() in ('1', 'true')) if isinstance(d, bool) else type(d)(v)
    if kw.get('method') == 'ail' and not kw.get('expert_ckpt'):
        kw['expert_ckpt'] = os.path.join(REPO, 'runs', a.exp, a.env, 'expert_best.npz')
    cfg = Config(**kw)
    if cfg.dtype == 'float64':
        jax.config.update('jax_enable_x64', True)      # before anything is traced
    jaxlog.info('config', exp=a.exp, arm=a.arm, **{k: v for k, v in dataclasses.asdict(cfg).items()})
    run(a.exp, a.arm, cfg, parse_seeds(a.seeds), a.top_seed)


if __name__ == '__main__':
    main()

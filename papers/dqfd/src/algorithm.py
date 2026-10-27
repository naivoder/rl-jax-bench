"""DQfD (Hester et al. 2018) and every comparator, as pure jitted single-run functions (vmapped over runs by the
caller).  Arms differ ONLY in the traced hyper-parameter dict `hp` (loss weights, replay flags, eps schedule); the
network, optimiser, replay and update code are one shared implementation.

Loss per sampled transition i targets are constants (stop_gradient):
    J_DQ_i = (r + g (1-term) Q_target(s', argmax_a Q(s',a)) - Q(s,a))^2                          1-step double Q
    J_n_i  = (G_n + g^m (1-term_n) Q_target(s_n, argmax_a Q(s_n,a)) - Q(s,a))^2                  n-step double Q
    J_E_i  = is_demo (max_a [Q(s,a) + 0.8 1(a != a_E)] - Q(s,a_E))                               large-margin loss
    CE_i   = is_demo (logsumexp_a Q(s,a) - Q(s,a_E))                                             ADET / imitation
    L = mean_i w_i [lam_td J_DQ + lam_n J_n + lam_e J_E + lam_ce CE] + lam3 * 0.5 * sum(theta^2)
"""
from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import optax

from . import replay as R
from .envs import MinAtar, pack_obs, reward_transform, unpack_obs
from .model import DuelingQNet, l2_half_sumsq

# --------------------------------------------------------------------------------------------- arms
# hp keys (all traced scalars, one value per run):
#   huber upd_every prio_total (attribution variants: Huber(1) TD loss, one update per `upd_every` env steps, DI-engine style
#   priority = sum of the per-sample losses) | lr | lam_td lam_n lam_e lam_ce lam3 margin | demo_on eps_d eps_a rbs per_on | learn_start eps0 eps1 eps_decay | nmax
ARM_NAMES = ["DQfD", "NoMargin", "NoNstep", "ADET", "IMIT", "PDD", "PDDconst", "RBS", "HER"]
PRETRAIN_ARMS = {"DQfD", "NoMargin", "NoNstep", "ADET", "IMIT"}  # BASE names of arms with a demo pre-training phase
NO_INTERACTION_ARMS = {"IMIT"}  # base names  # imitation is never updated from interaction (paper Fig. 1: flat line)


def base_arm(name: str) -> str:
    """'DQfD+eps0=0.1+eps1=0.1' -> 'DQfD' (probe arms: a base arm with labelled hp overrides)."""
    return name.split("+")[0]


def arm_hp(name: str, cfg: dict) -> dict:
    """Hyper-parameters of one arm from the config (every number is tagged in sources.yaml).
    Probe syntax: BASE+key=value+key=value overrides traced hp keys (declared attribution variants only)."""
    parts = name.split("+")
    hp = _arm_hp(parts[0], cfg)
    for kv in parts[1:]:
        k, v = kv.split("=")
        assert k in hp, f"unknown hp key {k}"
        hp[k] = float(v)
    return hp


def _arm_hp(name: str, cfg: dict) -> dict:
    d, p, per, pdd = cfg["dqfd"], cfg["per"], cfg["per"], cfg["pdd_dqn"]
    base = dict(huber=0.0, upd_every=1.0, prio_total=0.0, lr=cfg["optim"]["lr"], lam_td=1.0, lam_n=d["lambda1_nstep"], lam_e=d["lambda2_margin"], lam_ce=0.0, lam3=d["lambda3_l2"],
                margin=d["margin"], per_on=1.0, demo_on=1.0, eps_d=per["eps_demo"], eps_a=per["eps_agent"], rbs=0.0,
                learn_start=0.0, eps0=d["eps_greedy"], eps1=d["eps_greedy"], eps_decay=1.0,
                nmax=0.0 if d["nstep_bootstrap"] == "double_q" else 1.0)
    pdd_base = dict(base, lam_e=0.0, lam3=pdd["l2"], demo_on=0.0, learn_start=float(pdd["learning_starts"]),
                    eps0=1.0, eps1=pdd["eps_end"], eps_decay=float(pdd["eps_decay_steps"]))
    if name == "DQfD":
        return base
    if name == "NoMargin":
        return dict(base, lam_e=0.0)
    if name == "NoNstep":
        return dict(base, lam_n=0.0)
    if name == "ADET":
        # CHOICE: the paper is ambiguous (p.3 says original ADET does not pre-train; p.6 calls it "essentially DQfD
        # with ... cross-entropy").  We take the p.6 reading: DQfD with the margin loss replaced by CE, WITH pre-training.
        return dict(base, lam_e=0.0, lam_ce=1.0)
    if name == "IMIT":
        return dict(base, lam_td=0.0, lam_n=0.0, lam_e=0.0, lam_ce=1.0, per_on=0.0)  # uniform sampling, no IS
    if name == "PDD":
        return pdd_base
    if name == "PDDconst":
        return dict(pdd_base, eps0=pdd["eps_const_probe"], eps1=pdd["eps_const_probe"])
    if name == "RBS":
        # CHOICES: "PDD DQN with the replay buffer initially full of demonstration data" (p.6).  Our ring starts with ONE copy
        # of the ~17.5 k demo transitions (replay.init_buffer), NOT full (the ring holds 100 k; the demos are not tiled), overwritable;
        # learn_start = 0 (PDD uses 5000).  Flown on Breakout seeds 0-4 only.
        return dict(pdd_base, rbs=1.0, learn_start=0.0)
    if name == "HER":
        # CHOICES: HER = PER over demos + agent data with no demo bonus (eps_d := eps_a); no pre-training; learn_start = 0
        # (PDD uses 5000).  Flown on Breakout seeds 0-4 only.
        return dict(pdd_base, demo_on=1.0, eps_d=per["eps_agent"], learn_start=0.0)
    raise KeyError(name)


def stack_hp(names: list[str], n_seeds: int, cfg: dict) -> dict:
    """hp dict of (R,) arrays with R = len(names) * n_seeds, order = arm-major, seed-minor."""
    rows = [arm_hp(n, cfg) for n in names for _ in range(n_seeds)]
    return {k: jnp.asarray([r[k] for r in rows], jnp.float32) for k in rows[0]}


# --------------------------------------------------------------------------------------------- losses
def td_targets(q_next_online, q_next_target, rew, disc, nmax=None):
    """Bootstrap targets. disc = gamma^m * (1 - term) (0 at a game over).
    double-Q: online argmax, target evaluates;  nmax=1 -> max_a Q_target (the paper's displayed n-step form)."""
    a_star = jnp.argmax(q_next_online, axis=-1)
    boot = jnp.take_along_axis(q_next_target, a_star[..., None], axis=-1)[..., 0]
    if nmax is not None:
        boot = jnp.where(nmax > 0, q_next_target.max(axis=-1), boot)
    return rew + disc * boot


def margin_loss(q, act, margin):
    """J_E = max_a [Q(s,a) + l(a_E, a)] - Q(s, a_E), l = margin * 1[a != a_E]  (>= 0)."""
    l = margin * (1.0 - jax.nn.one_hot(act, q.shape[-1]))
    q_sa = jnp.take_along_axis(q, act[..., None], axis=-1)[..., 0]
    return jnp.max(q + l, axis=-1) - q_sa


def ce_loss(q, act):
    """Cross-entropy with the Q outputs as logits."""
    q_sa = jnp.take_along_axis(q, act[..., None], axis=-1)[..., 0]
    return jax.scipy.special.logsumexp(q, axis=-1) - q_sa


def dqfd_losses(q_s, q_s1_on, q_s1_tg, q_sn_on, q_sn_tg, batch, hp, gamma):
    """Per-sample loss terms from network outputs (no gradient flows through the *_on / *_tg arguments)."""
    act = batch["act"]
    q_sa = jnp.take_along_axis(q_s, act[:, None], axis=-1)[:, 0]
    y1 = td_targets(q_s1_on, q_s1_tg, batch["r1"], gamma * (1.0 - batch["term1"].astype(jnp.float32)))
    yn = td_targets(q_sn_on, q_sn_tg, batch["Gn"], batch["disc_n"], nmax=hp["nmax"])
    td1 = jax.lax.stop_gradient(y1) - q_sa
    tdn = jax.lax.stop_gradient(yn) - q_sa
    demo = batch["is_demo"].astype(jnp.float32)
    je = demo * margin_loss(q_s, act, hp["margin"])
    ce = demo * ce_loss(q_s, act)
    hub = lambda x: jnp.where(jnp.abs(x) <= 1.0, 0.5 * jnp.square(x), jnp.abs(x) - 0.5)   # Huber(delta = 1), optax.huber_loss
    sq = lambda x: jnp.where(hp.get("huber", 0.0) > 0, hub(x), jnp.square(x))
    return dict(j1=sq(td1), jn=sq(tdn), je=je, ce=ce, td1=td1)


def total_per_sample(terms, hp):
    return hp["lam_td"] * terms["j1"] + hp["lam_n"] * terms["jn"] + hp["lam_e"] * terms["je"] + hp["lam_ce"] * terms["ce"]


# --------------------------------------------------------------------------------------------- builder
class Fns:
    """Compiled-on-demand pure functions for one (game, layout, config)."""

    def __init__(self, env: MinAtar, L: R.Layout, cfg: dict, total_steps: int):
        self.env, self.L, self.cfg = env, L, cfg
        o = cfg["optim"]
        self.net = DuelingQNet(env.n_actions, channels=cfg["network"]["conv_channels"],
                               stream_hidden=cfg["network"]["dueling_stream_hidden"])
        # The learning rate is a TRACED per-run hyper-parameter hp["lr"] so the symmetric lr probe can share one
        # vmapped batch.  optax.adam(lr, eps) == chain(scale_by_adam(eps), scale(-lr)); the scale(-lr) is applied in `update`
        # (tests/test_full_scale_helpers.py checks the equality against optax.adam to 1e-6).
        self.tx = optax.chain(optax.clip_by_global_norm(o["grad_clip_global_norm"]),
                              optax.scale_by_adam(eps=o["adam_eps"]))
        self.B, self.tau, self.gamma = o["batch_size"], o["target_update_period"], cfg["env"]["gamma"]
        self.beta0 = cfg["per"]["beta0"]
        self.total_steps = total_steps
        self.bin_w = cfg["run"]["bin_width"]
        self.nbins = -(-total_steps // self.bin_w)
        self.ep_cap = cfg["run"]["episode_log_cap"]
        self.obs_shape = env.obs_shape

    # ---- init
    def init_params(self, key):
        x = jnp.zeros((1,) + self.obs_shape, jnp.float32)
        return self.net.init(key, x)

    def unpack(self, p):
        return unpack_obs(p, self.obs_shape)

    # ---- one gradient update (shared by pre-training and interaction)
    def update(self, params, tparams, opt_state, buf, key, beta, hp, do, cnt):
        L, B = self.L, self.B
        idx, P, total, valid = R.sample(L, buf["pa"], key, B)
        n_active = jnp.sum(buf["pa"] > 0).astype(jnp.float32)
        ok = valid & (total > 0)
        w = jnp.where(hp["per_on"] > 0, R.is_weights(P, n_active, beta, ok), ok.astype(jnp.float32))
        batch = R.gather_batch(L, buf, idx)
        s, s1, sn = (self.unpack(batch[k]) for k in ("obs", "nobs1", "nobsn"))
        x_next = jnp.concatenate([s1, sn], 0)
        q_next_tg = self.net.apply(tparams, x_next)
        q_next_on = self.net.apply(jax.lax.stop_gradient(params), x_next)

        def loss_fn(p):
            q_s = self.net.apply(p, s)
            terms = dqfd_losses(q_s, q_next_on[:B], q_next_tg[:B], q_next_on[B:], q_next_tg[B:], batch, hp, self.gamma)
            per = total_per_sample(terms, hp)
            loss = jnp.mean(w * per) + hp["lam3"] * l2_half_sumsq(p)
            return loss, terms

        (loss, terms), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        upd, opt2 = self.tx.update(grads, opt_state, params)
        upd = jax.tree_util.tree_map(lambda u: -hp["lr"] * u, upd)
        p2 = optax.apply_updates(params, upd)
        sel = lambda a, b: jax.tree_util.tree_map(lambda x, y: jnp.where(do, x, y), a, b)
        params_n, opt_n = sel(p2, params), sel(opt2, opt_state)
        sync = do & ((cnt + 1) % self.tau == 0)
        tparams_n = jax.tree_util.tree_map(lambda a, b: jnp.where(sync, a, b), params_n, tparams)
        prio = jnp.where(hp["prio_total"] > 0,
                         jnp.abs(hp["lam_n"] * terms["jn"]) + jnp.abs(terms["j1"]) + jnp.abs(hp["lam_e"] * terms["je"]), jnp.abs(terms["td1"]))
        buf_n = R.update_priorities(L, buf, idx, prio, ok & do & (hp["per_on"] > 0), hp)
        demo = batch["is_demo"].astype(jnp.float32)
        v = (valid & (total > 0)).astype(jnp.float32)
        diag = dict(loss=jnp.where(do, loss, 0.0), demo_share=jnp.sum(demo * v) / jnp.maximum(v.sum(), 1.0),
                    td_demo=jnp.sum(jnp.abs(terms["td1"]) * demo * v), n_demo=jnp.sum(demo * v),
                    td_agent=jnp.sum(jnp.abs(terms["td1"]) * (1 - demo) * v), n_agent=jnp.sum((1 - demo) * v),
                    did=do.astype(jnp.float32))
        return params_n, tparams_n, opt_n, buf_n, diag

    # ---- pre-training (demo data only, no environment)
    def pretrain_chunk(self, state, hp, key, steps: int):
        """state = dict(params, tparams, opt, buf, cnt). `steps` updates; beta = beta0 constant."""
        def body(c, k):
            p, tp, o, b, cnt = c
            p, tp, o, b, dg = self.update(p, tp, o, b, k, self.beta0, hp, jnp.array(True), cnt)
            return (p, tp, o, b, cnt + 1), dg

        keys = jax.random.split(key, steps)
        (p, tp, o, b, cnt), dg = jax.lax.scan(body, (state["params"], state["tparams"], state["opt"], state["buf"], state["cnt"]), keys)
        out = dict(state, params=p, tparams=tp, opt=o, buf=b, cnt=cnt)
        return out, jax.tree_util.tree_map(lambda x: x.mean(), dg)

    # ---- interaction
    def init_interact(self, params, tparams, opt, buf, key_env):
        obs, est = self.env.reset(key_env)
        z = lambda s, dt=jnp.float32: jnp.zeros(s, dt)
        return dict(params=params, tparams=tparams, opt=opt, buf=buf, env=est, obs=obs, ep_ret=z(()),
                    ep_count=z((), jnp.int32), t=z((), jnp.int32), cnt=z((), jnp.int32),
                    bin_sum=z((self.nbins,)), bin_cnt=z((self.nbins,)),
                    ep_ret_log=z((self.ep_cap,)), ep_step_log=z((self.ep_cap,), jnp.int32),
                    dg=self.zero_diag())

    @staticmethod
    def zero_diag():
        return {k: jnp.zeros((), jnp.float32) for k in
                ("loss", "demo_share", "td_demo", "n_demo", "td_agent", "n_agent", "did", "steps", "eps")}

    def interact_chunk(self, c, hp, key, steps: int, train_on):
        env, L = self.env, self.L

        def body(c, k):
            k_act, k_eps, k_env, k_rst, k_upd = jax.random.split(k, 5)
            t = c["t"]
            frac = jnp.clip(t.astype(jnp.float32) / hp["eps_decay"], 0.0, 1.0)
            eps = hp["eps0"] + (hp["eps1"] - hp["eps0"]) * frac
            q = self.net.apply(c["params"], c["obs"][None].astype(jnp.float32))[0]
            a_rand = jax.random.randint(k_act, (), 0, env.n_actions)
            a = jnp.where(jax.random.uniform(k_eps) < eps, a_rand, jnp.argmax(q)).astype(jnp.int32)
            obs2, st2, r, term, trunc = env.step(k_env, c["env"], a)
            end = term | trunc
            buf = R.add_transition(L, c["buf"], pack_obs(c["obs"]), pack_obs(obs2), a, reward_transform(r), term, end)
            # online episode bookkeeping (raw return, end-step stamped)
            ret = c["ep_ret"] + r
            b = jnp.clip(t // self.bin_w, 0, self.nbins - 1)
            bin_sum = c["bin_sum"].at[b].add(jnp.where(end, ret, 0.0))
            bin_cnt = c["bin_cnt"].at[b].add(end.astype(jnp.float32))
            e = jnp.minimum(c["ep_count"], self.ep_cap - 1)
            ep_ret_log = c["ep_ret_log"].at[e].set(jnp.where(end, ret, c["ep_ret_log"][e]))
            ep_step_log = c["ep_step_log"].at[e].set(jnp.where(end, t + 1, c["ep_step_log"][e]))
            obs_r, st_r = env.reset(k_rst)
            obs_n = jnp.where(end, obs_r, obs2)
            st_n = jax.tree_util.tree_map(lambda x, y: jnp.where(end, x, y), st_r, st2)
            beta = self.beta0 + (1.0 - self.beta0) * jnp.clip(t.astype(jnp.float32) / self.total_steps, 0.0, 1.0)
            do = train_on & (t.astype(jnp.float32) >= hp["learn_start"]) & (jnp.mod(t, jnp.maximum(hp["upd_every"], 1.0).astype(jnp.int32)) == 0)
            p, tp, o, buf, dg = self.update(c["params"], c["tparams"], c["opt"], buf, k_upd, beta, hp, do, c["cnt"])
            dgc = {k2: c["dg"][k2] + dg[k2] for k2 in dg}
            dgc["steps"] = c["dg"]["steps"] + 1.0
            dgc["eps"] = c["dg"]["eps"] + eps
            c2 = dict(c, params=p, tparams=tp, opt=o, buf=buf, env=st_n, obs=obs_n, ep_ret=jnp.where(end, 0.0, ret),
                      ep_count=c["ep_count"] + end.astype(jnp.int32), t=t + 1,
                      cnt=c["cnt"] + do.astype(jnp.int32), bin_sum=bin_sum, bin_cnt=bin_cnt,
                      ep_ret_log=ep_ret_log, ep_step_log=ep_step_log, dg=dgc)
            return c2, None

        keys = jax.random.split(key, steps)
        c = dict(c, dg=self.zero_diag())
        c, _ = jax.lax.scan(body, c, keys)
        return c

    # ---- evaluation of a policy by rollouts
    def greedy_returns(self, params, key, n_eps: int, eps: float):
        """Mean/all raw returns of the eps-greedy policy over n_eps fresh episodes (one run's params)."""
        env = self.env

        def one(k):
            k0, k1 = jax.random.split(k)
            obs, st = env.reset(k0)

            def body(c, kk):
                obs, st, ret, done = c
                ka, ke, ks = jax.random.split(kk, 3)
                q = self.net.apply(params, obs[None].astype(jnp.float32))[0]
                a = jnp.where(jax.random.uniform(ke) < eps, jax.random.randint(ka, (), 0, env.n_actions), jnp.argmax(q))
                obs2, st2, r, term, trunc = env.step(ks, st, a.astype(jnp.int32))
                ret = ret + jnp.where(done, 0.0, r)
                done2 = done | term | trunc
                return (obs2, st2, ret, done2), None

            (_, _, ret, _), _ = jax.lax.scan(body, (obs, st, jnp.zeros(()), jnp.array(False)),
                                             jax.random.split(k1, env.max_steps))
            return ret

        return jax.vmap(one)(jax.random.split(key, n_eps))

    def demo_gap(self, params, demo_obs_packed, demo_act, chunk: int = 1024):
        """Mean over demo states of Q(s,a_E) - max_{a != a_E} Q(s,a): the margin loss (l = 0.8) pushes it above
        ~0.8 on demonstrated states (mechanism check).  Returns (mean gap, fraction >= 0.8)."""
        n = demo_act.shape[0]
        pad = (-n) % chunk
        op = jnp.pad(demo_obs_packed, ((0, pad), (0, 0)))
        ap = jnp.pad(demo_act, (0, pad), constant_values=-1)

        def body(acc, xs):
            o, a = xs
            q = self.net.apply(params, self.unpack(o))
            valid = a >= 0
            a0 = jnp.maximum(a, 0)
            q_e = jnp.take_along_axis(q, a0[:, None], axis=-1)[:, 0]
            q_o = jnp.max(jnp.where(jax.nn.one_hot(a0, q.shape[-1]) > 0, -jnp.inf, q), axis=-1)
            g = q_e - q_o
            return (acc[0] + jnp.sum(jnp.where(valid, g, 0.0)), acc[1] + jnp.sum(valid & (g >= 0.8))), None

        acc, _ = jax.lax.scan(body, (jnp.zeros(()), jnp.zeros((), jnp.int32)),
                              (op.reshape(-1, chunk, op.shape[-1]), ap.reshape(-1, chunk)))
        return acc[0] / n, acc[1] / n

    def demo_agreement(self, params, demo_obs_packed, demo_act, chunk: int = 1024):
        """Fraction of demo states where the greedy action equals the demonstrator's label."""
        n = demo_act.shape[0]
        pad = (-n) % chunk
        op = jnp.pad(demo_obs_packed, ((0, pad), (0, 0)))
        ap = jnp.pad(demo_act, (0, pad), constant_values=-1)

        def body(acc, xs):
            o, a = xs
            q = self.net.apply(params, self.unpack(o))
            return acc + jnp.sum((jnp.argmax(q, -1) == a) & (a >= 0)), None

        acc, _ = jax.lax.scan(body, jnp.zeros((), jnp.int32),
                              (op.reshape(-1, chunk, op.shape[-1]), ap.reshape(-1, chunk)))
        return acc / n

"""JSRL (Uchendu et al. 2023) on a PDD DQN, plus the comparator arms, as pure jitted single-run functions (vmapped over
runs by the caller).  Arms differ ONLY in traced hyper-parameters `hp` (jsrl_mode, stage list, loss weights, replay flags,
eps schedule); network, optimiser, replay, update and evaluation code are one shared implementation.

PROVENANCE: the update / pre-training / PER code is our earlier DQfD reproduction's (its src/algorithm.py), copied
unchanged; JSRL-specific parts are marked 'JSRL:'.

JSRL (paper Algorithm 1 + Appendix A.4), per run:
    each episode has a switch step h_ep, fixed at its reset;  at within-episode step t_ep the guide acts iff t_ep < h_ep
    (paper: pi_{1:h} = pi_g, h guide steps), else pi_e (eps-greedy on the online Q net).
    mode 0 (PDD / DQfD / BCinit-FT):  h_ep = 0
    mode 1 (JSRL curriculum):         h_ep = h_list[stage] at the reset; stage advances by the beta rule (curriculum.py) on an
                                      evaluation of the COMBINED policy pi^{h_stage} every `chunk` env steps
    mode 2 (JSRL-Random):             h_ep ~ Uniform{h_list[0..n]} per episode (0 included)
    every transition (guide or pi_e) goes into the replay with its executed action (Algorithm 1 line 5);
    env steps are counted globally (guide steps included) for eps annealing, learning_starts and the budget.
GRADED evaluation: R_alone = mean raw return of 32 pi_e-ALONE episodes at eps 0.01 (h = 0);
    R_thr = 16 episodes of the combined policy at the current stage (curriculum input); R_greedy = 16 pi_e-alone episodes at eps 0.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax

from . import replay as R
from .curriculum import cur_init, cur_step
from .envs import MinAtar, pack_obs, reward_transform, unpack_obs
from .model import DuelingQNet, l2_half_sumsq

# --------------------------------------------------------------------------------------------- arms
ARM_NAMES = ["JSRL", "JSRL-Random", "PDD", "DQfD", "BCinit-FT", "IMIT"]
PRETRAIN_ARMS = {"DQfD", "IMIT"}          # BCinit-FT reuses IMIT's pre-trained weights (identical CE pre-training)
NO_INTERACTION_ARMS = {"IMIT"}            # evaluated only, never updated online (the zero-gradient learner floor)


def base_arm(name: str) -> str:
    """'JSRL+tol=0' -> 'JSRL' (probe arms: a base arm with labelled hp overrides)."""
    return name.split("+")[0]


def arm_hp(name: str, cfg: dict, h1: float | None = None) -> dict:
    """Hyper-parameters of one arm.  Probe syntax BASE+key=value overrides traced hp keys."""
    parts = name.split("+")
    hp = _arm_hp(parts[0], cfg, h1)
    for kv in parts[1:]:
        k, v = kv.split("=")
        assert k in hp, f"unknown hp key {k}"
        hp[k] = float(v)
    return hp


def _arm_hp(name: str, cfg: dict, h1) -> dict:
    from .curriculum import stage_list
    d, per, pdd, j = cfg["dqfd"], cfg["per"], cfg["pdd_dqn"], cfg["jsrl"]
    nst = j["n_stages"]
    hl = [float(x) for x in stage_list(h1, nst)] if h1 is not None else [0.0] * (nst + 1)
    base = dict(huber=0.0, upd_every=1.0, prio_total=0.0, lr=cfg["optim"]["lr"], lam_td=1.0, lam_n=d["lambda1_nstep"],
                lam_e=d["lambda2_margin"], lam_ce=0.0, lam3=d["lambda3_l2"], margin=d["margin"], per_on=1.0, demo_on=1.0,
                eps_d=per["eps_demo"], eps_a=per["eps_agent"], rbs=0.0, learn_start=0.0, eps0=d["eps_greedy"], eps1=d["eps_greedy"],
                eps_decay=1.0, nmax=0.0 if d["nstep_bootstrap"] == "double_q" else 1.0,
                # JSRL:
                jsrl_mode=0.0, tol=j["tolerance"], reset_on_adv=1.0 if j["ma_window_reset_per_stage"] else 0.0,
                h_list=[0.0] * (nst + 1),
                # Implementation-suspect variant: 1 = anneal eps on pi_e's OWN steps only (guide steps do not
                # advance the schedule); 0 = the default, eps annealed on the GLOBAL env-step counter incl. guide steps (as in the earlier DQfD reproduction)
                eps_own=0.0)
    pdd_base = dict(base, lam_e=0.0, lam3=pdd["l2"], demo_on=0.0, learn_start=float(pdd["learning_starts"]),
                    eps0=1.0, eps1=pdd["eps_end"], eps_decay=float(pdd["eps_decay_steps"]))
    if name == "PDD":
        return pdd_base
    if name == "JSRL":
        return dict(pdd_base, jsrl_mode=1.0, h_list=hl)
    if name == "JSRL-Random":
        return dict(pdd_base, jsrl_mode=2.0, h_list=hl)
    if name == "JSRL-h0":       # identity test: the curriculum machinery with every stage forced to h = 0 must reproduce PDD bit for bit
        return dict(pdd_base, jsrl_mode=1.0, h_list=[0.0] * (nst + 1))
    if name == "DQfD":
        return base
    if name == "IMIT":
        return dict(base, lam_td=0.0, lam_n=0.0, lam_e=0.0, lam_ce=1.0, per_on=0.0)  # uniform sampling, no IS (pre-training only)
    if name == "BCinit-FT":
        # CE-pre-trained Q net, then pure PDD TD (no demos in the buffer, no margin loss), eps constant 0.01, learning starts 5k
        b = cfg["bcinit"]
        return dict(pdd_base, eps0=b["finetune_eps"], eps1=b["finetune_eps"], learn_start=float(b["learning_starts"]))
    raise KeyError(name)


def stack_hp(names: list[str], n_seeds: int, cfg: dict, h1: float | None = None) -> dict:
    """hp dict of (R, ...) arrays with R = len(names) * n_seeds, order = arm-major, seed-minor."""
    rows = [arm_hp(n, cfg, h1) for n in names for _ in range(n_seeds)]
    return {k: jnp.asarray([r[k] for r in rows], jnp.float32) for k in rows[0]}


# --------------------------------------------------------------------------------------------- losses (the earlier DQfD reproduction, unchanged)
def td_targets(q_next_online, q_next_target, rew, disc, nmax=None):
    a_star = jnp.argmax(q_next_online, axis=-1)
    boot = jnp.take_along_axis(q_next_target, a_star[..., None], axis=-1)[..., 0]
    if nmax is not None:
        boot = jnp.where(nmax > 0, q_next_target.max(axis=-1), boot)
    return rew + disc * boot


def margin_loss(q, act, margin):
    l = margin * (1.0 - jax.nn.one_hot(act, q.shape[-1]))
    q_sa = jnp.take_along_axis(q, act[..., None], axis=-1)[..., 0]
    return jnp.max(q + l, axis=-1) - q_sa


def ce_loss(q, act):
    q_sa = jnp.take_along_axis(q, act[..., None], axis=-1)[..., 0]
    return jax.scipy.special.logsumexp(q, axis=-1) - q_sa


def dqfd_losses(q_s, q_s1_on, q_s1_tg, q_sn_on, q_sn_tg, batch, hp, gamma):
    act = batch["act"]
    q_sa = jnp.take_along_axis(q_s, act[:, None], axis=-1)[:, 0]
    y1 = td_targets(q_s1_on, q_s1_tg, batch["r1"], gamma * (1.0 - batch["term1"].astype(jnp.float32)))
    yn = td_targets(q_sn_on, q_sn_tg, batch["Gn"], batch["disc_n"], nmax=hp["nmax"])
    td1 = jax.lax.stop_gradient(y1) - q_sa
    tdn = jax.lax.stop_gradient(yn) - q_sa
    demo = batch["is_demo"].astype(jnp.float32)
    je = demo * margin_loss(q_s, act, hp["margin"])
    ce = demo * ce_loss(q_s, act)
    hub = lambda x: jnp.where(jnp.abs(x) <= 1.0, 0.5 * jnp.square(x), jnp.abs(x) - 0.5)
    sq = lambda x: jnp.where(hp.get("huber", 0.0) > 0, hub(x), jnp.square(x))
    return dict(j1=sq(td1), jn=sq(tdn), je=je, ce=ce, td1=td1)


def total_per_sample(terms, hp):
    return hp["lam_td"] * terms["j1"] + hp["lam_n"] * terms["jn"] + hp["lam_e"] * terms["je"] + hp["lam_ce"] * terms["ce"]


# --------------------------------------------------------------------------------------------- JSRL switching rule
def jsrl_act(t_ep, h_ep, a_guide, a_pi):
    """THE switching rule (paper Algorithm 1 line 4, pi_{1:h} = pi_g): the guide acts at within-episode steps t_ep < h_ep, pi_e after.
    Used by the training scan, the evaluation loop and tests/test_lock.py (switching rule) -- one implementation."""
    return jnp.where(t_ep < h_ep, a_guide, a_pi).astype(jnp.int32)


def episode_h(hp, stage, skey, ep_idx):
    """Switch step of episode `ep_idx` (0-based) of this run, drawn at its reset.  mode 1: the current stage's h; mode 2: uniform over
    the stage list (key fold_in(switch key, episode index)); mode 0: 0."""
    hl = hp["h_list"]
    j = jax.random.randint(jax.random.fold_in(skey, ep_idx), (), 0, hl.shape[0])
    h = jnp.where(hp["jsrl_mode"] == 1, hl[stage], jnp.where(hp["jsrl_mode"] == 2, hl[j], 0.0))
    return h.astype(jnp.int32)


# --------------------------------------------------------------------------------------------- builder
class Fns:
    """Compiled-on-demand pure functions for one (game, layout, config)."""

    def __init__(self, env, L: R.Layout, cfg: dict, total_steps: int, guide_fn, chunk: int):
        self.env, self.L, self.cfg, self.guide_fn = env, L, cfg, guide_fn
        o, j = cfg["optim"], cfg["jsrl"]
        self.net = DuelingQNet(env.n_actions, channels=cfg["network"]["conv_channels"],
                               stream_hidden=cfg["network"]["dueling_stream_hidden"])
        self.tx = optax.chain(optax.clip_by_global_norm(o["grad_clip_global_norm"]), optax.scale_by_adam(eps=o["adam_eps"]))
        self.B, self.tau, self.gamma = o["batch_size"], o["target_update_period"], cfg["env"]["gamma"]
        self.beta0 = cfg["per"]["beta0"]
        self.total_steps, self.chunk = total_steps, chunk
        self.n_eval = -(-total_steps // chunk)            # evaluation points 1..n_eval (point 0 = before training)
        self.nbins = self.n_eval
        self.obs_shape = env.obs_shape
        self.n_stages, self.ma_m = j["n_stages"], j["ma_horizon"]
        self.n_alone, self.n_thr, self.n_greedy = j["n_eval_alone"], j["n_thr_episodes"], j["n_eval_greedy"]
        self.eval_eps = j["eval_eps"]

    # ---- init
    def init_params(self, key):
        return self.net.init(key, jnp.zeros((1,) + self.obs_shape, jnp.float32))

    def unpack(self, p):
        return unpack_obs(p, self.obs_shape)

    # ---- one gradient update (shared by pre-training and interaction) -- the earlier DQfD reproduction, unchanged except the q-max diagnostic
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
            return loss, (terms, q_s)

        (loss, (terms, q_s)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
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
                    did=do.astype(jnp.float32), qmax=jnp.where(do, jnp.sum(q_s.max(-1) * v) / jnp.maximum(v.sum(), 1.0), 0.0))
        return params_n, tparams_n, opt_n, buf_n, diag

    # ---- pre-training (demo data only, no environment)
    def pretrain_chunk(self, state, hp, key, steps: int):
        def body(c, k):
            p, tp, o, b, cnt = c
            p, tp, o, b, dg = self.update(p, tp, o, b, k, self.beta0, hp, jnp.array(True), cnt)
            return (p, tp, o, b, cnt + 1), dg

        keys = jax.random.split(key, steps)
        (p, tp, o, b, cnt), dg = jax.lax.scan(body, (state["params"], state["tparams"], state["opt"], state["buf"], state["cnt"]), keys)
        out = dict(state, params=p, tparams=tp, opt=o, buf=b, cnt=cnt)
        return out, jax.tree_util.tree_map(lambda x: x.mean(), dg)

    # ---- interaction
    @staticmethod
    def zero_diag():
        return {k: jnp.zeros((), jnp.float32) for k in
                ("loss", "demo_share", "td_demo", "n_demo", "td_agent", "n_agent", "did", "qmax", "steps", "eps", "guide")}

    def init_interact(self, params, tparams, opt, buf, key_env, skey, hp):
        obs, est = self.env.reset(key_env)
        z = lambda s, dt=jnp.float32: jnp.zeros(s, dt)
        ne = self.n_eval + 1
        cur = cur_init(self.ma_m)
        return dict(params=params, tparams=tparams, opt=opt, buf=buf, env=est, obs=obs, ep_ret=z(()),
                    ep_count=z((), jnp.int32), t=z((), jnp.int32), tpi=z((), jnp.int32), cnt=z((), jnp.int32),
                    t_ep=z((), jnp.int32), h_ep=episode_h(hp, cur["stage"], skey, 0), skey=skey, cur=cur,
                    guide_steps=z((), jnp.int32), neval=z((), jnp.int32),
                    bin_sum=z((self.nbins,)), bin_cnt=z((self.nbins,)),
                    ev_alone=z((ne,)), ev_thr=z((ne,)), ev_greedy=z((ne,)), ev_stage=z((ne,), jnp.int32), ev_gsteps=z((ne,), jnp.int32),
                    ev_alone_sd=z((ne,)), ev_cap=z((ne,)), dg=self.zero_diag())

    def interact_chunk(self, c, hp, key, steps: int, train_on):
        env, L = self.env, self.L

        def body(c, k):
            k_act, k_eps, k_env, k_rst, k_upd = jax.random.split(k, 5)
            t = c["t"]
            t_eps = jnp.where(hp["eps_own"] > 0, c["tpi"], t)     # eps_own = 1: the schedule counts pi_e steps only (variant); default = global t
            frac = jnp.clip(t_eps.astype(jnp.float32) / hp["eps_decay"], 0.0, 1.0)
            eps = hp["eps0"] + (hp["eps1"] - hp["eps0"]) * frac
            q = self.net.apply(c["params"], c["obs"][None].astype(jnp.float32))[0]
            a_rand = jax.random.randint(k_act, (), 0, env.n_actions)
            a_pi = jnp.where(jax.random.uniform(k_eps) < eps, a_rand, jnp.argmax(q)).astype(jnp.int32)
            # JSRL: the guide's action (own noise eps_g, key disjoint from the learner's 5-way split) and the switching rule
            a_g = self.guide_fn(c["obs"], jax.random.fold_in(k, 11))
            use_g = c["t_ep"] < c["h_ep"]
            a = jsrl_act(c["t_ep"], c["h_ep"], a_g, a_pi)
            obs2, st2, r, term, trunc = env.step(k_env, c["env"], a)
            end = term | trunc
            buf = R.add_transition(L, c["buf"], pack_obs(c["obs"]), pack_obs(obs2), a, reward_transform(r), term, end)
            ret = c["ep_ret"] + r
            b = jnp.clip(t // self.chunk, 0, self.nbins - 1)
            bin_sum = c["bin_sum"].at[b].add(jnp.where(end, ret, 0.0))
            bin_cnt = c["bin_cnt"].at[b].add(end.astype(jnp.float32))
            obs_r, st_r = env.reset(k_rst)
            obs_n = jnp.where(end, obs_r, obs2)
            st_n = jax.tree_util.tree_map(lambda x, y: jnp.where(end, x, y), st_r, st2)
            ep_count = c["ep_count"] + end.astype(jnp.int32)
            h_next = episode_h(hp, c["cur"]["stage"], c["skey"], ep_count)
            beta = self.beta0 + (1.0 - self.beta0) * jnp.clip(t.astype(jnp.float32) / self.total_steps, 0.0, 1.0)
            do = train_on & (t.astype(jnp.float32) >= hp["learn_start"]) & (jnp.mod(t, jnp.maximum(hp["upd_every"], 1.0).astype(jnp.int32)) == 0)
            p, tp, o, buf, dg = self.update(c["params"], c["tparams"], c["opt"], buf, k_upd, beta, hp, do, c["cnt"])
            dgc = {k2: c["dg"][k2] + dg[k2] for k2 in dg}
            dgc["steps"] = c["dg"]["steps"] + 1.0
            dgc["eps"] = c["dg"]["eps"] + eps
            dgc["guide"] = c["dg"]["guide"] + use_g.astype(jnp.float32)
            c2 = dict(c, params=p, tparams=tp, opt=o, buf=buf, env=st_n, obs=obs_n, ep_ret=jnp.where(end, 0.0, ret),
                      ep_count=ep_count, t=t + 1, tpi=c["tpi"] + (~use_g).astype(jnp.int32), cnt=c["cnt"] + do.astype(jnp.int32), bin_sum=bin_sum, bin_cnt=bin_cnt,
                      t_ep=jnp.where(end, 0, c["t_ep"] + 1), h_ep=jnp.where(end, h_next, c["h_ep"]),
                      guide_steps=c["guide_steps"] + use_g.astype(jnp.int32), dg=dgc)
            return c2, None

        keys = jax.random.split(key, steps)
        c = dict(c, dg=self.zero_diag())
        c, _ = jax.lax.scan(body, c, keys)
        return c

    # ---- evaluation of a policy by rollouts -------------------------------------------------------------------------------
    def evaluate(self, params, h_thr, key):
        """One run's evaluation point.  Three batches of episodes stepped in lock-step from fresh resets (no replay writes):
            n_alone  pi_e alone (h = 0), eps = eval_eps        -> R_alone  (the GRADED metric)
            n_thr    combined policy pi^{h_thr}, eps = eval_eps on pi_e, the guide at its own eps_g   -> R_thr (curriculum input)
            n_greedy pi_e alone, eps = 0                       -> R_greedy (descriptor)
        Returns dict(alone, thr, greedy, alone_sd) of mean raw returns."""
        env = self.env
        na, nt, ng = self.n_alone, self.n_thr, self.n_greedy
        N = na + nt + ng
        h = jnp.concatenate([jnp.zeros(na, jnp.int32), jnp.full((nt,), h_thr, jnp.int32), jnp.zeros(ng, jnp.int32)])
        eps = jnp.concatenate([jnp.full((na + nt,), self.eval_eps, jnp.float32), jnp.zeros(ng, jnp.float32)])
        k0, k1 = jax.random.split(key)
        obs, st = jax.vmap(env.reset)(jax.random.split(k0, N))
        gv = jax.vmap(self.guide_fn)

        def cond(c):
            t, _, _, _, done, _ = c
            return (t < env.max_steps) & ~jnp.all(done)

        def body(c):
            t, obs, st, ret, done, capd = c
            kt = jax.random.fold_in(k1, t)
            ka, ke, ks, kg = jax.random.split(kt, 4)
            q = self.net.apply(params, obs.astype(jnp.float32))
            a_rand = jax.random.randint(ka, (N,), 0, env.n_actions)
            a_pi = jnp.where(jax.random.uniform(ke, (N,)) < eps, a_rand, jnp.argmax(q, -1)).astype(jnp.int32)
            a_g = gv(obs, jax.random.split(kg, N))
            a = jsrl_act(t, h, a_g, a_pi)
            obs2, st2, r, term, trunc = jax.vmap(env.step)(jax.random.split(ks, N), st, a)
            ret = ret + jnp.where(done, 0.0, r)
            capd = capd | (trunc & ~done)          # the episode ended at the time cap (counts only the first ending)
            return t + 1, obs2, st2, ret, done | term | trunc, capd

        _, _, _, ret, _, capd = jax.lax.while_loop(cond, body, (jnp.zeros((), jnp.int32), obs, st, jnp.zeros(N), jnp.zeros(N, bool), jnp.zeros(N, bool)))
        ra = ret[:na]
        return dict(alone=ra.mean(), alone_sd=ra.std(), thr=ret[na:na + nt].mean(), greedy=ret[na + nt:].mean(),
                    cap=capd[:na].astype(jnp.float32).mean())

    def h_eval(self, hp, stage):
        """Switch step of the threshold-evaluation episodes: the current stage's h for the curriculum arm, 0 otherwise."""
        return jnp.where(hp["jsrl_mode"] == 1, hp["h_list"][stage], 0.0).astype(jnp.int32)

    def eval_point0(self, c, hp, key):
        """Evaluation point 0 (before any interaction); logged only, never fed to the curriculum."""
        ev = self.evaluate(c["params"], self.h_eval(hp, c["cur"]["stage"]), key)
        return dict(c, ev_alone=c["ev_alone"].at[0].set(ev["alone"]), ev_alone_sd=c["ev_alone_sd"].at[0].set(ev["alone_sd"]),
                    ev_thr=c["ev_thr"].at[0].set(ev["thr"]), ev_greedy=c["ev_greedy"].at[0].set(ev["greedy"]),
                    ev_cap=c["ev_cap"].at[0].set(ev["cap"]))

    def chunk_and_eval(self, c, hp, key, ekey, train_on):
        """`chunk` interaction steps, then evaluation point j = neval + 1 and (curriculum arm) the beta rule."""
        c = self.interact_chunk(c, hp, key, self.chunk, train_on)
        stage = c["cur"]["stage"]
        ev = self.evaluate(c["params"], self.h_eval(hp, stage), ekey)
        cur, adv = cur_step(c["cur"], ev["thr"], hp["tol"], self.n_stages, hp["reset_on_adv"], hp["jsrl_mode"] == 1)
        j = jnp.minimum(c["neval"] + 1, self.n_eval)
        return dict(c, cur=cur, neval=c["neval"] + 1,
                    ev_alone=c["ev_alone"].at[j].set(ev["alone"]), ev_alone_sd=c["ev_alone_sd"].at[j].set(ev["alone_sd"]),
                    ev_thr=c["ev_thr"].at[j].set(ev["thr"]), ev_greedy=c["ev_greedy"].at[j].set(ev["greedy"]),
                    ev_stage=c["ev_stage"].at[j].set(stage), ev_gsteps=c["ev_gsteps"].at[j].set(c["guide_steps"]),
                    ev_cap=c["ev_cap"].at[j].set(ev["cap"]))

    # ---- warm-start pre-flights (logged diagnostics)
    def preflight(self, params, demo_obs_packed, demo_act, n: int = 2048):
        """Alive fraction of each hidden layer (unit alive iff pre-activation > 0 on at least one of the first n demo states), and the greedy
        agreement with the guide's labels on those states.  Pure jnp, jittable."""
        x = self.unpack(demo_obs_packed[:n])
        q, inter = self.net.apply(params, x, capture_intermediates=lambda m, name: m.name in ("conv", "v_hidden", "a_hidden"), mutable=["intermediates"])
        alive = lambda h: jnp.mean(jnp.any(h.reshape((-1, h.shape[-1])) > 0, axis=0).astype(jnp.float32))
        it = inter["intermediates"]
        return dict(alive_conv=alive(it["conv"]["__call__"][0]), alive_v=alive(it["v_hidden"]["__call__"][0]), alive_a=alive(it["a_hidden"]["__call__"][0]),
                    agreement=jnp.mean((jnp.argmax(q, -1) == demo_act[:n]).astype(jnp.float32)), q_norm=jnp.median(jnp.linalg.norm(q, axis=-1)))

    # ---- diagnostics used by the warm-start pre-flights (the earlier DQfD reproduction)
    def demo_agreement(self, params, demo_obs_packed, demo_act, chunk: int = 1024):
        n = demo_act.shape[0]
        pad = (-n) % chunk
        op = jnp.pad(demo_obs_packed, ((0, pad), (0, 0)))
        ap = jnp.pad(demo_act, (0, pad), constant_values=-1)

        def body(acc, xs):
            o, a = xs
            q = self.net.apply(params, self.unpack(o))
            return acc + jnp.sum((jnp.argmax(q, -1) == a) & (a >= 0)), None

        acc, _ = jax.lax.scan(body, jnp.zeros((), jnp.int32), (op.reshape(-1, chunk, op.shape[-1]), ap.reshape(-1, chunk)))
        return acc / n

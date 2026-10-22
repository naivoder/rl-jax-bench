"""DQSA training algorithm (paper §IV-B box), reward variants (§V), and every baseline (§VI), as pure jitted functions.

Learner: rl_jax_bench.drqn supplies the masked sequence TD loss (`sequence_loss`), the optimizer step (`update`), the hard
target sync (`sync` / `sync_due`) and `td_target`/`bootstrap_value` (double-Q).  The dueling LSTM network is model.py's, passed
through drqn's `unroll_fn` hook.  The on-policy driver below is new: M fresh episodes per iteration, ALL users acting with the
same network (common training, Def. 5), NO replay buffer, one gradient step per iteration, target copy every l iterations.

Arms differ ONLY in the mechanism under test: `reward` in {competitive, sum_rate, prop_fair} is the sole difference between the
three DQSA arms; baselines share the world generator (data.py) and the world keys.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import data as D  # noqa: E402
import model as M  # noqa: E402
from rl_jax_bench import drqn  # noqa: E402
import optax  # noqa: E402


def optax_norm(tree):
    return optax.global_norm(tree)


# ----------------------------------------------------------------------------------------------- rewards (paper §V)
# paper §III says the reward "can be viewed as a function of the achievable data rate B log2(1+SNR)".  The *_rate kinds weight every
# success by the delivered rate of that slot (computed after the rollout from the per-slot delivered Mbps, `rate`):
#   rate_comp : own delivered rate / C_ERGODIC                                         (Eq. 13 with rate-valued successes)
#   rate_sum  : sum over active users of delivered rate / C_ERGODIC                    (sum of rates, alpha-fair alpha = 0)
#   rate_pf   : sum over active users of x_n(t) / X_n(t), x = delivered rate, X = cumulative delivered rate incl. slot t
#               (the harmonic / log-increment shaping of Eq. 23-24 applied to rate instead of packet counts)
# When every success delivers rate 1 (no fading): rate_pf == prop_fair exactly and rate_comp / rate_sum == competitive / sum_rate / C_ERGODIC (tests/test_review_fixes.py).
RATE_KINDS = ("rate_comp", "rate_sum", "rate_pf")
# (recheck 2026-10-20): the v1/v2 arXiv versions define the cooperative reward as EPISODIC (Eq. 19-21 of v2): r_n(t) = 0 for t < T and
# r_n(T) = sum_n f(sum_t 1_n(t-1)), f(x) = x^{1-a}/(1-a).  a = 0 -> total successes of the episode ("sum_rate_episodic"); a = 1 -> sum_n ln(max(x_n, 0.01))
# ("pf_episodic", floor LOG_FLOOR_EP = ln 0.01 declared for x = 0).  Delivered ONCE, at the last slot, and the episode is TERMINAL there
# (train.build forces bootstrap_at_T = False for these kinds).  Default-off arms; the v3 shaped rewards stay the primary.
EPISODIC_KINDS = ("sum_rate_episodic", "pf_episodic")
LOG_FLOOR_EP = -4.605170185988091


def reward_episodic(kind: str, success, active):
    """success [T,N] bool -> rewards [T,N]: zero except the last row, which holds the episode's global reward for every user."""
    cnt = (success & active[None, :]).astype(jnp.float32).sum(0)                       # [N] successes of each user
    if kind == "sum_rate_episodic":
        tot = cnt.sum()
    elif kind == "pf_episodic":
        tot = jnp.where(active, jnp.maximum(jnp.log(jnp.maximum(cnt, 1e-30)), LOG_FLOOR_EP), 0.0).sum()
    else:
        raise ValueError(kind)
    T, N = success.shape
    return jnp.zeros((T, N), jnp.float32).at[-1].set(tot)


def reward_rate(kind: str, rate, active):
    """rate [T,N] delivered Mbps per slot -> rewards [T,N] (see RATE_KINDS)."""
    x = jnp.where(active[None, :], rate, 0.0)
    if kind == "rate_comp":
        return x / D.C_ERGODIC_MBPS
    if kind == "rate_sum":
        return jnp.broadcast_to(x.sum(1, keepdims=True) / D.C_ERGODIC_MBPS, x.shape)
    if kind == "rate_pf":
        cum = jnp.cumsum(x, axis=0)
        term = jnp.where(cum > 0, x / jnp.maximum(cum, 1e-30), 0.0)
        return jnp.broadcast_to(term.sum(1, keepdims=True), x.shape)
    raise ValueError(kind)


def reward_vec(kind: str, success, active, succ_after):
    """Per-user reward r_n(t+1) from the outcomes of slot t.
    competitive : own success                                   (Eq. 13)
    sum_rate    : sum_{n'} 1_{n'}(t), same for every user       (§V-C; alpha-fair alpha = 0)
    prop_fair   : sum_{n'} 1_{n'}(t) / M_{n'}(t)                (Eq. 23-24; M includes slot t; harmonic-number shaping)
    The sum runs over the active users of the episode's interference domain (clique / whole network, A15)."""
    s = (success & active).astype(jnp.float32)
    if kind in RATE_KINDS or kind in EPISODIC_KINDS:                    # rewards are computed from the rate / success sequence after the scan
        return jnp.zeros_like(s)
    if kind == "competitive":
        return s
    if kind == "sum_rate":
        return jnp.broadcast_to(s.sum(), s.shape)
    if kind == "prop_fair":
        return jnp.broadcast_to((s / jnp.maximum(succ_after, 1.0)).sum(), s.shape)
    raise ValueError(kind)


# ----------------------------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class TrainCfg:
    reward: str = "competitive"
    iterations: int = 10_000        # R   paper §VI
    episodes: int = 16              # M   paper §VI (minibatch of 16 episodes)
    horizon: int = 50               # T   paper §VI (50 time steps each)
    gamma: float = 0.95             # paper §VI (in the target: ASSUMED)
    target_every: int = 5           # l   paper §IV-B
    lr: float = 3e-4                # ASSUMED
    adam_eps: float = 1e-8          # ASSUMED
    grad_steps: int = 1             # ASSUMED epochs of the full-vector regression (full-vector regression) per iteration; 1 == scalar TD loss (scalar TD loss)
    double: bool = True             # paper §IV-A.5
    bootstrap_at_T: bool = True     # ASSUMED (False = terminal at T, ablation)
    reward_scale: float = 1.0       # ASSUMED (1.0 literal; 1/N ablation divides cooperative rewards)
    dueling: bool = True
    lstm: int = 100
    hid: int = 10
    alpha0: float = 0.05
    alpha1: float = 0.0
    beta0: float = 1.0
    beta1: float = 20.0
    max_grad_norm: float | None = None   # ASSUMED none
    act_rule: str = "eq11"             # training acting rule. "eq11" = paper (sample Eq. 11). "argmax_eps" = third-party shkrwnd rule, ablation ONLY: argmax_a Q, and with prob act_eps ALL users act uniformly at random that slot
    act_eps: float = 0.02                # shkrwnd's epsilon (0.02 -> 0.01 decay; constant 0.02 here)
    fresh_opt: bool = False              # (recheck): re-initialise the Adam moments at the start of EVERY iteration (a literal Matlab trainNetwork call restarts its solver); default = carried moments (primary)
    l2: float = 0.0                      # L2 regularisation factor on the weight matrices (Matlab trainingOptions default 1e-4); gradient += l2 * w on kernels, biases excluded; 0 = off
    rescale_beta: bool = False           # divide the Eq. 11 beta (train ramp and test) by reward_scale so that a reward rescaling leaves the policy semantics unchanged
    sched_iters: int | None = None       # length of the (alpha, beta) ramps; None = `iterations`.  10000 + --iters 20000 CONTINUES training past R at the final alpha=0, beta=20 (under-training check)


def make_qcfg(spec: D.EnvSpec, tc: TrainCfg) -> drqn.QConfig:
    return drqn.QConfig(n_actions=spec.n_actions, obs_dim=spec.obs_dim, hidden=(), lstm=tc.lstm, double=tc.double,
                        gamma=tc.gamma, lr=tc.lr, adam_eps=tc.adam_eps, max_grad_norm=tc.max_grad_norm,
                        target_period=tc.target_every)


# ----------------------------------------------------------------------------------------------- rollouts
def _episode_learned(spec, net, params, wkey, akey, T, alpha, beta, reward, greedy, rscale=1.0, act_rule="eq11", act_eps=0.0):
    """One episode, every active user acts with `params`.  Returns per-user sequences (leading axis N)."""
    wk, ik, sk = D.split_world_key(wkey)
    world = D.sample_world(spec, wk)
    state = D.init_state(spec, ik)
    carry = M.zero_carry(net, (spec.n_max,))

    def body(c, t):
        state, carry = c
        x = D.observe(spec, world, state)
        q, carry = M.act_step(net, params, x, carry)
        if act_rule == "argmax_eps":
            ke, kr = jax.random.split(jax.random.fold_in(akey, t))
            rand = jax.random.randint(kr, (spec.n_max,), 0, spec.n_actions)
            a = jnp.where(jax.random.uniform(ke) < act_eps, rand, M.greedy_action(q)).astype(jnp.int32)
        elif greedy:
            a = M.greedy_action(q)
        else:
            a = M.sample_eq11(jax.random.fold_in(akey, t), q, alpha, beta)
        state, info = D.step_local(spec, world, state, a, jax.random.fold_in(sk, t))
        r = reward_vec(reward, info["success"], world.active, state.succ) * rscale
        return (state, carry), (x, a, r, info["success"], info["tx"], info["g"], info["rate"], info["pu_free"])

    (state, _), (xs, acts, rews, succ, tx, g, rate, pu) = jax.lax.scan(body, (state, carry), jnp.arange(T))
    if reward in RATE_KINDS:
        rews = reward_rate(reward, rate, world.active) * rscale
    elif reward in EPISODIC_KINDS:
        rews = reward_episodic(reward, succ, world.active) * rscale
    x_last = D.observe(spec, world, state)
    xs = jnp.concatenate([xs, x_last[None]], axis=0)                    # [T+1,N,D]
    return dict(xs=jnp.swapaxes(xs, 0, 1), acts=acts.T, rews=rews.T, active=world.active, n_active=world.n_active,
                success=succ, tx=tx, g=g, rate=rate, subset=world.subset, pu_free=pu)      # success/tx/g/rate: [T,N]; pu_free [T,K]


def rollout_learned(spec, net, params, wkeys, akeys, T, alpha, beta, reward, greedy=False, rscale=1.0, act_rule="eq11", act_eps=0.0):
    """vmap over the M episodes of `wkeys`/`akeys`."""
    f = lambda wk, ak: _episode_learned(spec, net, params, wk, ak, T, alpha, beta, reward, greedy, rscale, act_rule, act_eps)
    return jax.vmap(f)(wkeys, akeys)


def _episode_policy(spec, policy, wkey, akey, T):
    """One episode for a stateless/state-feedback baseline policy(key, spec, world, state, t) -> (tx, g)."""
    wk, ik, sk = D.split_world_key(wkey)
    world = D.sample_world(spec, wk)
    state = D.init_state(spec, ik)

    def body(state, t):
        ret = policy(jax.random.fold_in(akey, t), spec, world, state, t)
        tx, g = ret[0] & world.active, ret[1]
        ann = ret[2] if len(ret) > 2 else g                              # what the user announces to its neighbours (default: its channel)
        state, info = D.step_global(spec, world, state, tx, g, jax.random.fold_in(sk, t))
        state = state._replace(choice=jnp.where(world.active, ann, -1).astype(jnp.int32))
        return state, (info["success"], info["tx"], info["g"], info["rate"], info["pu_free"])

    _, (succ, tx, g, rate, pu) = jax.lax.scan(body, state, jnp.arange(T))
    return dict(active=world.active, n_active=world.n_active, success=succ, tx=tx, g=g, rate=rate, subset=world.subset, pu_free=pu)


def rollout_policy(spec, policy, wkeys, akeys, T):
    return jax.vmap(lambda wk, ak: _episode_policy(spec, policy, wk, ak, T))(wkeys, akeys)


# ----------------------------------------------------------------------------------------------- baselines (paper §VI)
def aloha(p=None):
    """Slotted Aloha (Fig. 3 benchmark): transmit w.p. p on a uniformly random channel.  Default p = min(1, K/n_active),
    which is the paper's optimal p = 1/n_j for a clique (K = 1) and the multichannel generalisation (README)."""
    def policy(key, spec, world, state, t):
        k1, k2 = jax.random.split(key)
        pp = jnp.minimum(1.0, spec.n_channels / world.n_active) if p is None else p
        tx = jax.random.uniform(k1, (spec.n_max,)) < pp
        g = jax.random.randint(k2, (spec.n_max,), 0, spec.n_channels)
        return tx, g
    return policy


def tdma_genie():
    """TDMA GENIE REFERENCE (not an upper bound): centralised collision-free, idle-free schedule (Thm 2/3 genie): in slot t users
    (tK + j) mod n transmit on channel j.  It ignores the users' channel subsets, the PUs and the fading state, so under the 2-channel
    restriction / PUs it is NOT feasible, and under fading (no CSI) it is NOT an upper bound on the rate.  The achievable per-slot
    success bound for the subsets and PU state is eval.reduce_one's `feasible_t` (data.max_feasible)."""
    def policy(key, spec, world, state, t):
        i = jnp.arange(spec.n_max)
        off = jnp.mod(i - t * spec.n_channels, world.n_active)
        return off < spec.n_channels, off.astype(jnp.int32)
    return policy


def stay_redraw():
    """Toy T2 scripted policy (README): always transmit; keep the channel after a success, redraw uniformly otherwise."""
    def policy(key, spec, world, state, t):
        redraw = jax.random.randint(key, (spec.n_max,), 0, spec.n_channels)
        keep = (state.prev_ack > 0.5) & (state.prev_g >= 0)
        return jnp.ones((spec.n_max,), bool), jnp.where(keep, state.prev_g, redraw)
    return policy


def _announced_others(spec, world, state, k):
    """For every user n and every channel k: number of OTHER users that announced channel k in the previous slot (state.choice;
    the paper's message exchange 'sent to its neighbors ... used in future iterations', §VI-A)."""
    K = spec.n_channels
    ch = state.choice
    ok = (ch >= 0) & world.active
    occ = jnp.zeros((K,), jnp.float32).at[jnp.maximum(ch, 0)].add(ok.astype(jnp.float32))
    own = jax.nn.one_hot(jnp.maximum(ch, 0), K) * ok[:, None]
    return occ[None, :] - own                                            # [N,K]


def oca(same_slot: bool = False, pu_aware: bool = True):
    """Opportunistic Channel Aware access -- RECONSTRUCTED (refs [36],[57]): every user takes the channel with the highest
    instantaneous achievable rate over ALL K channels ("irrespective of the collision rate") and transmits with probability
    1 / (1 + number of OTHER users that announced that channel in the PREVIOUS slot) -- the message-exchange information (the paper
    sends parameter updates 'to its neighbors ... used in future iterations').  pu_aware (ref [57] exploits idle channels): the
    ranking uses rate x 1{PU idle} so a user does not pick a PU-occupied channel while an idle one exists.
    same_slot=True, pu_aware=False is the pre-review (v1) oracle version: p = 1 / (users that picked the channel THIS slot), no PU
    information -- kept as arm `oca_sameslot` for comparison only."""
    def policy(key, spec, world, state, t):
        r = D.all_rates(spec, state)                                    # [N,K] genie CSI
        if pu_aware:
            r = jnp.where(state.pu_free[None, :], r, 0.0)
        best = jnp.argmax(r, axis=1).astype(jnp.int32)
        if same_slot:
            cnt = jnp.zeros((spec.n_channels,), jnp.float32).at[best].add(world.active.astype(jnp.float32))
            p = 1.0 / jnp.maximum(cnt[best], 1.0)
        else:
            o = jnp.take_along_axis(_announced_others(spec, world, state, best), best[:, None], axis=1)[:, 0]
            p = 1.0 / (o + 1.0)
        tx = jax.random.uniform(key, (spec.n_max,)) < p
        return tx, best
    return policy


P_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)       # ASSUMED : transmission-probability grid of the Gibbs sampler


def dp(tau: float = 0.1, p_update: float = 1.0, tx_rule: str = "gibbs"):
    """Distributed Protocol (Hou & Gupta [58]) -- RECONSTRUCTED.  Paper §VI: DP uses a 'Gibbs sampler when selecting
    channels AND transmission probabilities'.
    tx_rule="gibbs" (default): every slot each user samples the JOINT pair (channel k, transmission probability p in P_GRID) from the
      Gibbs pmf ~ exp(log(rate_{n,k} * p * prod_{o != n announced on k} (1 - p_o)) / tau), where (k_o, p_o) are the pairs the other
      users announced in the PREVIOUS slot (neighbour messages, packed into state.choice as k * len(P_GRID) + j), then transmits w.p. p.
    tx_rule="count" (pre-review v1, arm `dp_sameslot`): channel only, Psucc_{n,k} = (1 - 1/(o+1))^o with o = others on k last slot, and the
      transmission probability is the ORACLE 1 / (number of users that picked this channel THIS slot).
    p_update < 1: each user re-samples only w.p. p_update per slot (asynchronous Gibbs sweep; the sampler converges when neighbours do
    NOT update simultaneously).  p_update = 1 is the 'literal' version; tau is ASSUMED (the paper gives no temperature)."""
    def policy_count(key, spec, world, state, t):
        k1, k2 = jax.random.split(key)
        K = spec.n_channels
        r = D.all_rates(spec, state)                                    # [N,K]
        ch = state.choice
        o = _announced_others(spec, world, state, None)
        psucc = (1.0 - 1.0 / (o + 1.0)) ** o
        logits = jnp.log(r * psucc + 1e-30) / tau
        pick = jax.random.categorical(k1, logits, axis=-1).astype(jnp.int32)
        if p_update < 1.0:
            upd = (jax.random.uniform(jax.random.fold_in(k1, 7), (spec.n_max,)) < p_update) | (ch < 0)
            pick = jnp.where(upd, pick, ch)
        cnt = jnp.zeros((K,), jnp.float32).at[pick].add(world.active.astype(jnp.float32))
        tx = jax.random.uniform(k2, (spec.n_max,)) < 1.0 / jnp.maximum(cnt[pick], 1.0)
        return tx, pick

    PG = jnp.asarray(P_GRID, jnp.float32)
    nP = len(P_GRID)

    def policy_gibbs(key, spec, world, state, t):
        k1, k2, k3 = jax.random.split(key, 3)
        K = spec.n_channels
        r = D.all_rates(spec, state)                                    # [N,K]
        ch = state.choice
        has = (ch >= 0) & world.active
        kp, jp = jnp.maximum(ch, 0) // nP, jnp.maximum(ch, 0) % nP
        lq = jnp.log(jnp.maximum(1.0 - PG[jp], 1e-6)) * has             # log(1 - p_o) of each announcer
        S = jnp.zeros((K,), jnp.float32).at[kp].add(lq)                 # per channel
        logps = S[None, :] - jax.nn.one_hot(kp, K) * lq[:, None]        # others only, [N,K]
        score = jnp.log(r + 1e-30)[:, :, None] + logps[:, :, None] + jnp.log(PG)[None, None, :]
        pick = jax.random.categorical(k1, (score / tau).reshape(spec.n_max, K * nP), axis=-1).astype(jnp.int32)
        if p_update < 1.0:
            upd = (jax.random.uniform(k3, (spec.n_max,)) < p_update) | (ch < 0)
            pick = jnp.where(upd, pick, ch)
        k, j = pick // nP, pick % nP
        tx = jax.random.uniform(k2, (spec.n_max,)) < PG[j]
        return tx, k.astype(jnp.int32), pick

    if tx_rule not in ("gibbs", "count"):
        raise ValueError(tx_rule)
    return policy_gibbs if tx_rule == "gibbs" else policy_count


def scripted_e6():
    """Metric-definition oracle for E6 (v1/v2): users 0..K-1 always transmit on their own channel, the others never do."""
    def policy(key, spec, world, state, t):
        i = jnp.arange(spec.n_max)
        return i < spec.n_channels, jnp.minimum(i, spec.n_channels - 1).astype(jnp.int32)
    return policy


def scripted_e7():
    """Metric-definition oracle for E7 (N=3, K=2): user 0 alone on channel 0, users 1 and 2 alternate on channel 1 (TDMA 50/50)."""
    def policy(key, spec, world, state, t):
        i = jnp.arange(spec.n_max)
        tx = (i == 0) | ((i == 1) & (t % 2 == 0)) | ((i == 2) & (t % 2 == 1))
        return tx, jnp.where(i == 0, 0, 1).astype(jnp.int32)
    return policy


def oca_sameslot():
    return oca(same_slot=True, pu_aware=False)


def dp_sameslot(tau: float = 0.1, p_update: float = 1.0):
    return dp(tau=tau, p_update=p_update, tx_rule="count")


BASELINES: dict[str, Callable] = dict(aloha=aloha, tdma_genie=tdma_genie, stay_redraw=stay_redraw, oca=oca, dp=dp,
                                      oca_sameslot=oca_sameslot, dp_sameslot=dp_sameslot,
                                      scripted_e6=scripted_e6, scripted_e7=scripted_e7)


# ----------------------------------------------------------------------------------------------- training
class TrainFns(NamedTuple):
    net: M.DQSANet
    qcfg: drqn.QConfig
    init: Callable          # key -> QState
    iterate: Callable       # (QState, it, train_key) -> (QState, metrics)
    run_chunk: Callable     # (QState, start_it, n, train_key) -> (QState, metrics[n])


def make_train_fns(spec: D.EnvSpec, tc: TrainCfg) -> TrainFns:
    net = M.make_net(spec.n_actions, tc.lstm, tc.hid, tc.dueling)
    qcfg = make_qcfg(spec, tc)
    unroll_fn = M.make_unroll_fn(net)
    opt = drqn.make_optimizer(qcfg)
    B = tc.episodes * spec.n_max

    def init(key):
        params = M.init_params(net, key, spec.obs_dim)
        return drqn.QState(params, jax.tree.map(jnp.copy, params), opt.init(params), jnp.zeros((), jnp.int32))

    def make_batch(out):
        T = tc.horizon
        mask = jnp.broadcast_to(out["active"].reshape(B, 1), (B, T)).astype(jnp.float32)
        disc = jnp.full((B, T), tc.gamma, jnp.float32)
        if not tc.bootstrap_at_T:                                        # ablation: terminal at T
            disc = disc.at[:, -1].set(0.0)
        return dict(inputs=out["xs"].reshape(B, T + 1, spec.obs_dim), actions=out["acts"].reshape(B, T),
                    rewards=out["rews"].reshape(B, T), discounts=disc, mask=mask)

    def d6_epochs(qs, batch, epochs):
        """Paper box steps 9-14 read literally with several epochs: Qhat = Q_theta1 with the taken entry replaced by the double-Q
        target y (both frozen at the start of the iteration), then `epochs` Adam steps on the full-vector MSE (full-vector regression).  Untaken
        entries are anchored to their old predictions; epochs == 1 reproduces the scalar TD gradient (tests/test_grads.py)."""
        inputs, act, m = batch["inputs"], batch["actions"], batch["mask"]
        q1, _ = unroll_fn(qs.params, inputs, None)
        qt, _ = unroll_fn(qs.target_params, inputs, None)
        y = drqn.td_target(q1[:, 1:], qt[:, 1:], batch["rewards"], batch["discounts"], qcfg.double)
        oh = jax.nn.one_hot(act, qcfg.n_actions)
        qhat = jax.lax.stop_gradient(q1[:, :-1] * (1 - oh) + y[..., None] * oh)

        def loss_fn(params):
            q = unroll_fn(params, inputs, None)[0][:, :-1]
            loss = (m[..., None] * (q - qhat) ** 2).sum() / (jnp.maximum(m.sum(), 1.0) * qcfg.n_actions)
            if tc.l2 > 0.0:                                              # Matlab L2Regularization: gradient += l2 * w on weights (not biases)
                loss = loss + 0.5 * tc.l2 * sum((w ** 2).sum() for path, w in jax.tree_util.tree_flatten_with_path(params)[0]
                                                if "bias" not in str(path[-1]).lower())
            return loss

        def body(c, _):
            params, opt_state = c
            loss, g = jax.value_and_grad(loss_fn)(params)
            upd, opt_state = opt.update(g, opt_state, params)
            return (jax.tree.map(lambda a, b: a + b, params, upd), opt_state), (loss, optax_norm(g))

        opt0 = opt.init(qs.params) if tc.fresh_opt else qs.opt_state                  # fresh_opt: solver state restarts every iteration
        (params, opt_state), (losses, gn) = jax.lax.scan(body, (qs.params, opt0), None, length=epochs)
        if tc.fresh_opt:
            opt_state = qs.opt_state                                     # carried slot unused; keep the pytree shape
        qs = drqn.QState(params, qs.target_params, opt_state, qs.n_updates + 1)   # one iteration == one counted update (target sync period l)
        qmean = (jnp.take_along_axis(q1[:, :-1], act[..., None], -1)[..., 0] * m).sum() / jnp.maximum(m.sum(), 1.0)
        return qs, dict(loss=losses[-1], q_mean=qmean, maxq_mean=qmean, grad_norm=gn[-1])

    def iterate(qs, it, train_key):
        bsc = (1.0 / tc.reward_scale) if tc.rescale_beta else 1.0
        alpha, beta = M.schedule(it, tc.sched_iters or tc.iterations, tc.alpha0, tc.alpha1, tc.beta0 * bsc, tc.beta1 * bsc)
        k = jax.random.fold_in(train_key, it)
        ek, ak = jax.random.split(k)
        out = rollout_learned(spec, net, qs.params, D.world_keys(ek, tc.episodes), D.world_keys(ak, tc.episodes),
                              tc.horizon, alpha, beta, tc.reward, False, tc.reward_scale, tc.act_rule, tc.act_eps)
        batch = make_batch(out)
        if tc.grad_steps == 1 and not tc.fresh_opt and tc.l2 == 0.0:     # primary: one step, full-vector gradient == scalar TD gradient
            qs, info = drqn.update(qs, batch, qcfg, unroll_fn)
        else:                                                            # ablation: E epochs of full-vector regression on fixed targets Qhat
            qs, info = d6_epochs(qs, batch, tc.grad_steps)
        qs = drqn.sync(qs, drqn.sync_due(qs, tc.target_every))
        act = out["active"][:, None, :]
        util = (out["success"] & act).sum((1, 2)).mean() / (tc.horizon * spec.n_channels)
        met = dict(loss=info["loss"], q_mean=info["q_mean"], maxq_mean=info["maxq_mean"], grad_norm=info["grad_norm"],
                   train_util=util, alpha=alpha, beta=beta)
        return qs, met

    def run_chunk(qs, start, n, train_key):
        def body(qs, i):
            return iterate(qs, start + i, train_key)
        return jax.lax.scan(body, qs, jnp.arange(n))

    return TrainFns(net, qcfg, init, iterate, run_chunk)

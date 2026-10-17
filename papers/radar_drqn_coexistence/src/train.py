"""Training / evaluation driver: every arm and baseline, vmapped over lanes = (experiment, reading, seed).

Protocol (paper §III-C, §IV-B): train_cpis x 1000 pulses with uniform-random actions and one
gradient step per pulse once the buffer holds a batch; then eval_cpis x 1000 pulses greedy with online
updates continuing, on an independent evaluation stream that starts OFF. A frozen-network evaluation
(no updates) on the same stream is run descriptively. Baselines (pi*, SAA, random, policy iteration) act on the SAME
evaluation stream of each lane (common random numbers).

PRNG key tree:
    root = PRNGKey(seed_root)
    k_lane      = fold_in(fold_in(fold_in(root, exp_id), reading_id), seed)    exp_id/reading_id = global indices
    k_env_train = fold_in(k_lane, 0)   interference uniforms fold_in(fold_in(k, 0), t), target fold_in(fold_in(k, 1), .)
    k_env_eval  = fold_in(k_lane, 1)   same layout; t restarts at 0 at the start of evaluation
    k_init      = fold_in(k_lane, 2)   network init
    k_explore   = fold_in(k_lane, 3)   uniform actions of the offline phase: randint(fold_in(k_explore, t))
    k_replay    = fold_in(k_lane, 4)   minibatch draw at global step t: fold_in(k_replay, t)
    k_random    = fold_in(k_lane, 5)   the random baseline's actions at eval step t
All arms of a lane share k_env_*, k_explore (identical training data, also used by policy iteration) and the eval stream.

Usage:  python src/train.py --config src/configs/small.yaml --out runs/small [--arms dqn ddrqn] [--skip-baselines]
"""
from __future__ import annotations

import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Any, NamedTuple  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import algorithm as alg  # noqa: E402
import baselines as bl  # noqa: E402
import data as D  # noqa: E402
from rl_jax_bench.jaxlog import Timer, check_device, info, metric, nan_guard, warn, write_json  # noqa: E402
from model import act_reset_period, arm_config, is_window  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- lanes
class Lane(NamedTuple):
    params: D.EnvParams
    k_env_train: jax.Array
    k_env_eval: jax.Array
    k_init: jax.Array
    k_explore: jax.Array
    k_replay: jax.Array
    k_random: jax.Array


def lane_key(seed_root: int, exp: str, reading: str, seed: int) -> jax.Array:
    k = jax.random.PRNGKey(seed_root)
    k = jax.random.fold_in(k, D.EXPERIMENTS.index(exp))
    k = jax.random.fold_in(k, D.READINGS.index(reading))
    return jax.random.fold_in(k, seed)


def build_lanes(cfg: dict, exps, readings, seeds, p_move: float = 0.1) -> tuple[Lane, list[tuple]]:
    ids, lanes = [], []
    for e in exps:
        for r in readings:
            for s in seeds:
                kl = lane_key(cfg["seed_root"], e, r, s)
                lanes.append(Lane(D.env_params(e, r, cfg["eval_cpis"], p_move), *[jax.random.fold_in(kl, i) for i in range(6)]))
                ids.append((e, r, s))
    return jax.tree.map(lambda *xs: jnp.stack(xs), *lanes), ids


# ----------------------------------------------------------------------------- learner loop
class Carry(NamedTuple):
    env: D.EnvState
    q: alg.QState
    buf: alg.TraceBuffer
    wr: alg.TraceWriter
    ac: Any            # actor LSTM carry
    prev_a: jax.Array


def init_carry(lane: Lane, qcfg: alg.QConfig, window: bool = False) -> Carry:
    ac = jnp.zeros((qcfg.trace_len, qcfg.obs_dim), jnp.float32) if window else alg.initial_carry(qcfg)
    return Carry(D.env_reset(lane.k_env_train, lane.params), alg.init(lane.k_init, qcfg), alg.buffer_init(qcfg),
                 alg.writer_init(qcfg), ac, jnp.zeros((), jnp.int32))


def _sel(flag, a, b):
    return jax.tree.map(lambda x, y: jnp.where(flag, x, y), a, b)


def lane_step(c: Carry, t, T_cur, lane: Lane, qcfg: alg.QConfig, n_train: int, online: bool, reset_period: int = 0):
    """One pulse of one lane. t (global step) is unbatched, so the learning cond stays a real cond under vmap.
    reset_period: pulses between act-time carry resets (0 -> trace_len, the shipped reading)."""
    net = alg.make_network(qcfg)
    L = qcfg.trace_len
    is_train = t < n_train
    t_stream = jnp.where(is_train, t, t - n_train)
    k_stream = jnp.where(is_train, lane.k_env_train, lane.k_env_eval)
    env = _sel(t == n_train, D.env_reset(lane.k_env_eval, lane.params), c.env)   # eval stream starts fresh
    obs = D.obs_of(env, lane.params)
    P = reset_period or L
    reset = t % P == 0                                                            # act-time reset (P = L shipped)
    q, ac = alg.act_q(net, qcfg, c.q.params, obs, c.ac, reset)
    a_rand = jax.random.randint(jax.random.fold_in(lane.k_explore, t), (), 0, D.N_ACT)
    a = jnp.where(is_train, a_rand, alg.greedy(q))
    env2, r, (coll, miss, size, th2) = D.env_step(env, a, t_stream, T_cur, k_stream, lane.params)
    obs2 = D.obs_of(env2, lane.params)
    wr, buf, _ = alg.writer_step(c.wr, c.buf, obs, a, r, qcfg.gamma, obs2, reset, (t + 1) % P == 0)
    can_learn = (t + 1) // L >= qcfg.batch_size
    do_learn = can_learn & (is_train | online)
    if qcfg.update_every > 1:                                                     # audit arm only (shipped: 1)
        do_learn = do_learn & ((t + 1) % qcfg.update_every == 0)

    def learn(qs):
        batch = alg.buffer_sample(buf, jax.random.fold_in(lane.k_replay, t), qcfg.batch_size)
        qs, inf = alg.update(qs, batch, qcfg)
        qs = alg.sync(qs, alg.sync_due(qs, qcfg.target_period))
        return qs, inf["loss"], inf["maxq_mean"]

    def skip(qs):
        return qs, jnp.zeros((), jnp.float32), jnp.zeros((), jnp.float32)

    qs, loss, maxq = jax.lax.cond(do_learn, learn, skip, c.q)
    adapt = (a != c.prev_a) & (t != n_train)                                       # Def. 3, eval pulses only
    th_obs = lane.params.pattern[env.z]
    out = jnp.stack([r, coll, miss, size, adapt.astype(jnp.float32), loss, maxq, do_learn.astype(jnp.float32)])
    fp = jnp.stack([th2, th2 * ((t_stream % CPI_FP) + 1)]).astype(jnp.int32)       # stream fingerprint terms
    return Carry(env2, qs, buf, wr, ac, a), out, fp, th_obs


CPI_FP = 1009   # fingerprint modulus (prime), only for the stream-identity check


# ----------------------------------------------------------------------------- T-A: Keras sliding-window DRQN (audit only)
def window_loss(params, target_params, batch: dict, cfg: alg.QConfig):
    """Sliding-window reading of "Episode Length (DDRQN) 10": inputs [B, L+1, D] = window_t (rows 0..L-1) + o_{t+1}.
    Q(window_t) = LSTM over rows 0..L-1 from a zero state, read at the last step; the double-Q target reads the next
    window (rows 1..L); one TD error per stored transition (action/reward/discount at position L-1)."""
    net = alg.make_network(cfg)
    inp = batch["inputs"]
    B, L = inp.shape[0], cfg.trace_len
    z = alg.initial_carry(cfg, (B,))
    q_cur, _ = alg.unroll(net, cfg, params, inp[:, :L], z)
    q_nxt_on, _ = alg.unroll(net, cfg, params, inp[:, 1:], z)
    q_nxt_tg, _ = alg.unroll(net, cfg, target_params, inp[:, 1:], z)
    a = batch["actions"][:, L - 1]
    q_sa = jnp.take_along_axis(q_cur[:, -1], a[:, None], axis=-1)[:, 0]
    y = jax.lax.stop_gradient(alg.td_target(q_nxt_on[:, -1], q_nxt_tg[:, -1], batch["rewards"][:, L - 1],
                                            batch["discounts"][:, L - 1], cfg.double))
    return ((q_sa - y) ** 2).mean(), q_cur[:, -1].max(-1).mean()


def window_update(state: alg.QState, batch: dict, cfg: alg.QConfig):
    (loss, maxq), g = jax.value_and_grad(window_loss, has_aux=True)(state.params, state.target_params, batch, cfg)
    upd, opt = alg.make_optimizer(cfg).update(g, state.opt_state, state.params)
    return alg.QState(alg.optax.apply_updates(state.params, upd), state.target_params, opt, state.n_updates + 1), \
        dict(loss=loss, maxq_mean=maxq)


def lane_step_window(c: Carry, t, T_cur, lane: Lane, qcfg: alg.QConfig, n_train: int, online: bool):
    """As lane_step, for the window arm: c.ac is the last L observations; acting unrolls them from a zero state
    (no act-time reset); every pulse stores one window transition; learning starts at batch_size windows."""
    net = alg.make_network(qcfg)
    L = qcfg.trace_len
    is_train = t < n_train
    t_stream = jnp.where(is_train, t, t - n_train)
    k_stream = jnp.where(is_train, lane.k_env_train, lane.k_env_eval)
    env = _sel(t == n_train, D.env_reset(lane.k_env_eval, lane.params), c.env)
    obs = D.obs_of(env, lane.params)
    win = jnp.concatenate([c.ac[1:], obs[None]], 0)                               # [L, D], zero-padded at the start
    q = alg.unroll(net, qcfg, c.q.params, win[None], alg.initial_carry(qcfg, (1,)))[0][0, -1]
    a_rand = jax.random.randint(jax.random.fold_in(lane.k_explore, t), (), 0, D.N_ACT)
    a = jnp.where(is_train, a_rand, alg.greedy(q))
    env2, r, (coll, miss, size, th2) = D.env_step(env, a, t_stream, T_cur, k_stream, lane.params)
    obs2 = D.obs_of(env2, lane.params)
    z = jnp.zeros((L,), jnp.float32)
    trace = dict(obs=jnp.concatenate([win, obs2[None]], 0), act=jnp.zeros((L,), jnp.int32).at[L - 1].set(a),
                 rew=z.at[L - 1].set(r), disc=z.at[L - 1].set(qcfg.gamma), resets=jnp.zeros((L + 1,), bool), carry0=())
    buf = alg.buffer_add(c.buf, trace, True)
    do_learn = ((t + 1) >= qcfg.batch_size) & (is_train | online)

    def learn(qs):
        batch = alg.buffer_sample(buf, jax.random.fold_in(lane.k_replay, t), qcfg.batch_size)
        qs, inf = window_update(qs, batch, qcfg)
        qs = alg.sync(qs, alg.sync_due(qs, qcfg.target_period))
        return qs, inf["loss"], inf["maxq_mean"]

    def skip(qs):
        return qs, jnp.zeros((), jnp.float32), jnp.zeros((), jnp.float32)

    qs, loss, maxq = jax.lax.cond(do_learn, learn, skip, c.q)
    adapt = (a != c.prev_a) & (t != n_train)
    th_obs = lane.params.pattern[env.z]
    out = jnp.stack([r, coll, miss, size, adapt.astype(jnp.float32), loss, maxq, do_learn.astype(jnp.float32)])
    fp = jnp.stack([th2, th2 * ((t_stream % CPI_FP) + 1)]).astype(jnp.int32)
    return Carry(env2, qs, buf, c.wr, win, a), out, fp, th_obs


def make_chunk_fn(qcfg: alg.QConfig, n_train_cpis: int, chunk_cpis: int, online: bool, window: bool = False,
                  reset_period: int = 0):
    n_train = n_train_cpis * D.CPI
    if window:
        step_fn = lane_step_window
    else:
        def step_fn(c, t, T_cur, lane, qcfg, n_train, online):
            return lane_step(c, t, T_cur, lane, qcfg, n_train, online, reset_period)

    def lane_chunk(c: Carry, lane: Lane, cpi0):
        def cpi_body(c, j):
            cpi = cpi0 + j
            T_cur = D.T_for_cpi(lane.params, cpi, n_train_cpis)

            def step_body(c, i):
                c, out, fp, _ = step_fn(c, cpi * D.CPI + i, T_cur, lane, qcfg, n_train, online)
                return c, (out, fp)

            c, (outs, fps) = jax.lax.scan(step_body, c, jnp.arange(D.CPI))
            return c, (outs.sum(0), fps.sum(0))

        return jax.lax.scan(cpi_body, c, jnp.arange(chunk_cpis))

    vm = jax.vmap(lane_chunk, in_axes=(0, 0, None))

    @jax.jit
    def chunk(c, lanes, cpi0):
        c2, (outs, fps) = vm(c, lanes, cpi0)
        nan_guard((c2.q.params, outs), (outs,), tag=0, dump_dir=os.path.join(HERE, "..", "runs", "nan_dumps"))
        return c2, outs, fps

    return chunk


def probe_q(qcfg: alg.QConfig):
    """Q on probe observations (zero carry, one step), averaged over a 5x2 (x, v) grid -> [Z_MAX, 15] per lane."""
    net = alg.make_network(qcfg)
    xs = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    vs = jnp.array([0.0, 1.0])

    def one(params, pattern):
        th = D.THETA_BITS_J[pattern]                                         # [Z, 5]
        X, V = jnp.meshgrid(xs, vs)
        xv = jnp.stack([X.ravel(), V.ravel()], -1)                         # [10, 2]
        obs = jnp.concatenate([jnp.broadcast_to(xv[None], (D.Z_MAX, 10, 2)),
                               jnp.broadcast_to(th[:, None], (D.Z_MAX, 10, D.N_SUB))], -1)
        q, _ = alg.act_q(net, qcfg, params, obs, alg.initial_carry(qcfg, (D.Z_MAX, 10)))
        return q.mean(1)

    return jax.jit(jax.vmap(one))


# ----------------------------------------------------------------------------- checkpointing (orbax)
def _ckpt_mgr(path):
    import orbax.checkpoint as ocp
    return ocp.CheckpointManager(os.path.abspath(path), options=ocp.CheckpointManagerOptions(max_to_keep=2, create=True))


def save_ckpt(mgr, step, state):
    import orbax.checkpoint as ocp
    mgr.save(step, args=ocp.args.StandardSave(state))
    mgr.wait_until_finished()


def restore_ckpt(mgr, template):
    import orbax.checkpoint as ocp
    step = mgr.latest_step()
    if step is None:
        return None, None
    return step, mgr.restore(step, args=ocp.args.StandardRestore(template))


def run_arm(cfg: dict, arm: str, lanes: Lane, ids: list, out_dir: str, qstar: np.ndarray) -> dict:
    """Train + online eval (+ frozen eval) for every lane of one arm. Returns per-CPI arrays [n_lanes, n_cpi, 8]."""
    qcfg = arm_config(arm, cfg)
    win = is_window(arm)
    ntr, nev, ch = cfg["train_cpis"], cfg["eval_cpis"], cfg["chunk_cpis"]
    assert ntr % ch == 0 and nev % ch == 0, "chunk_cpis must divide train_cpis and eval_cpis"
    total = ntr + nev
    nl = len(ids)
    rp = act_reset_period(arm)
    chunk_on = make_chunk_fn(qcfg, ntr, ch, online=bool(cfg["eval_online_updates"]), window=win, reset_period=rp)
    probe = probe_q(qcfg)   # window arm: the probe is a 1-observation window (monitoring only, never a claim)
    carry = jax.jit(jax.vmap(lambda ln: init_carry(ln, qcfg, win)))(lanes)
    outs = np.zeros((nl, total, 8), np.float64)
    fps = np.zeros((nl, total, 2), np.int64)
    probe_err = np.full((nl, total // ch), np.nan)
    probe_agree = np.full((nl, total // ch), np.nan)
    ck_dir = os.path.join(out_dir, "ckpt", arm)
    mgr = _ckpt_mgr(ck_dir)
    start = 0
    frozen_carry = None
    # the frozen-eval snapshot (carry at the train/eval boundary) is checkpointed too; before the boundary the slot
    # holds a placeholder copy and has_frozen = 0
    tmpl = dict(carry=carry, frozen=carry, has_frozen=np.zeros((), np.int64), outs=outs, fps=fps, probe_err=probe_err,
                probe_agree=probe_agree, cpi=np.zeros((), np.int64))
    step, st = restore_ckpt(mgr, tmpl)
    if st is not None:
        carry, outs, fps = st["carry"], np.asarray(st["outs"]), np.asarray(st["fps"])
        probe_err, probe_agree, start = np.asarray(st["probe_err"]), np.asarray(st["probe_agree"]), int(st["cpi"])
        if int(st["has_frozen"]):
            frozen_carry = st["frozen"]
        info("resumed", arm=arm, cpi=start, has_frozen=int(st["has_frozen"]))
    tm = Timer()
    last_ck = time.time()
    first = True
    for cpi0 in range(start, total, ch):
        if cpi0 == ntr and cfg.get("frozen_eval", True):
            frozen_carry = jax.tree.map(jnp.copy, carry)
        carry, o, f = chunk_on(carry, lanes, jnp.int32(cpi0))
        o = np.asarray(o)
        dt = tm.lap()
        if first:
            info("first chunk (includes compile)", arm=arm, seconds=round(dt, 2), lanes=nl, cpis=ch)
            first = False
        else:
            metric("chunk", arm=arm, cpi=cpi0 + ch, sec=round(dt, 3), steps_per_s=round(nl * ch * D.CPI / dt, 1))
        outs[:, cpi0:cpi0 + ch] = o
        fps[:, cpi0:cpi0 + ch] = np.asarray(f)
        qp = np.asarray(probe(carry.q.params, lanes.params.pattern))          # [nl, Z, 15]
        seg = (cpi0 + ch - 1 - ntr) if cpi0 + ch > ntr else -1
        if seg < 0:
            qs = qstar[:, 0]
        else:
            sidx = np.minimum((seg >= np.asarray(lanes.params.seg_end)).sum(1), D.N_SEG - 1)
            qs = qstar[np.arange(nl), 1 + sidx]
        nz = np.array([2 if e != "fh" else 5 for e, _, _ in ids])
        err, agr = [], []
        for li in range(nl):
            z = nz[li]
            err.append(np.abs(qp[li, :z] - qs[li, :z]).max())
            agr.append(np.mean(qp[li, :z].argmax(-1) == qs[li, :z].argmax(-1)))
        probe_err[:, (cpi0 + ch) // ch - 1], probe_agree[:, (cpi0 + ch) // ch - 1] = err, agr
        if ((cpi0 + ch) % 50 == 0) or cpi0 + ch == total:
            phase = "train" if cpi0 + ch <= ntr else "eval"
            metric("monitor", arm=arm, cpi=cpi0 + ch, phase=phase, td_loss=float(o[:, :, 5].sum() / max(o[:, :, 7].sum(), 1)),
                   maxq=float(o[:, :, 6].sum() / max(o[:, :, 7].sum(), 1)), reward=float(o[:, :, 0].mean() / D.CPI),
                   qerr_vs_qstar=float(np.mean(err)), greedy_agree_pistar=float(np.mean(agr)))
        if time.time() - last_ck > 60 * cfg.get("ckpt_minutes", 5) and cpi0 + ch < total:
            save_ckpt(mgr, cpi0 + ch, dict(carry=carry, frozen=frozen_carry if frozen_carry is not None else carry,
                                           has_frozen=np.array(int(frozen_carry is not None), np.int64), outs=outs,
                                           fps=fps, probe_err=probe_err, probe_agree=probe_agree,
                                           cpi=np.array(cpi0 + ch, np.int64)))
            last_ck = time.time()
    res = dict(outs=outs, fps=fps, probe_err=probe_err, probe_agree=probe_agree)
    if frozen_carry is not None:
        chunk_fr = make_chunk_fn(qcfg, ntr, ch, online=False, window=win, reset_period=rp)
        fo = np.zeros((nl, nev, 8))
        c = frozen_carry
        for k, cpi0 in enumerate(range(ntr, total, ch)):
            c, o, _ = chunk_fr(c, lanes, jnp.int32(cpi0))
            fo[:, k * ch:(k + 1) * ch] = np.asarray(o)
        res["frozen_outs"] = fo
    return res


# ----------------------------------------------------------------------------- baselines on the eval stream
def make_policy_eval(n_eval_cpis: int, mode: str):
    """mode in {table, full, random}: table = per-segment theta_obs table [N_SEG, 32] (pi*, SAA);
    full = 16,000-state table (PI); random = uniform from k_random."""

    def lane_eval(lane: Lane, tab_seg, tab_full):
        env0 = D.env_reset(lane.k_env_eval, lane.params)

        def cpi_body(carry, c):
            env, prev = carry
            T = lane.params.T_eval[D.segment_of(lane.params, c)]
            seg = D.segment_of(lane.params, c)

            def step(carry, i):
                env, prev = carry
                t = c * D.CPI + i
                th = lane.params.pattern[env.z]
                if mode == "table":
                    a = tab_seg[seg, th]
                elif mode == "full":
                    a = tab_full[D.full_state_index(env, lane.params)]
                else:
                    a = jax.random.randint(jax.random.fold_in(lane.k_random, t), (), 0, D.N_ACT)
                env2, r, (coll, miss, size, th2) = D.env_step(env, a, t, T, lane.k_env_eval, lane.params)
                adapt = ((a != prev) & (t > 0)).astype(jnp.float32)
                out = jnp.stack([r, coll, miss, size, adapt, 0.0, 0.0, 0.0])
                fp = jnp.stack([th2, th2 * ((t % CPI_FP) + 1)]).astype(jnp.int32)
                return (env2, a), (out, fp, th2.astype(jnp.int8))

            carry, (o, fp, th) = jax.lax.scan(step, (env, prev), jnp.arange(D.CPI))
            return carry, (o.sum(0), fp.sum(0), th)

        _, (o, fp, th) = jax.lax.scan(cpi_body, (env0, jnp.zeros((), jnp.int32)), jnp.arange(n_eval_cpis))
        return o, fp, th.reshape(-1)

    return jax.jit(jax.vmap(lane_eval))


def collect_training(n_train_cpis: int):
    """The learners' training stream (k_env_train + k_explore uniform actions), as full-state transitions for PI."""

    def lane_collect(lane: Lane):
        env0 = D.env_reset(lane.k_env_train, lane.params)
        T = lane.params.T_train

        def step(env, t):
            s = D.full_state_index(env, lane.params)
            a = jax.random.randint(jax.random.fold_in(lane.k_explore, t), (), 0, D.N_ACT)
            env2, r, (_, _, _, th2) = D.env_step(env, a, t, T, lane.k_env_train, lane.params)
            return env2, (s, a, r, D.full_state_index(env2, lane.params), th2)

        _, out = jax.lax.scan(step, env0, jnp.arange(n_train_cpis * D.CPI))
        return out

    return jax.jit(jax.vmap(lane_collect))


def exact_tables(ids, cfg) -> tuple[np.ndarray, np.ndarray, list]:
    """Per lane: pi* tables per eval segment [N_SEG, 32] and Q*(z, a) for [train, seg0..seg4] -> [1+N_SEG, Z, 15]."""
    tabs, qst, ex_info = [], [], []
    for e, r, _ in ids:
        ex = D.experiment(e)
        R = D.reward_table(r)
        tab = np.tile(D.ASTAR_ACTION, (D.N_SEG, 1))
        q = np.zeros((1 + D.N_SEG, D.Z_MAX, D.N_ACT))
        sol = bl.exact_solve(ex["train"], R, cfg["gamma"])
        q[0, :ex["train"].nz] = sol["Q"]
        for k in range(D.N_SEG):
            ch = ex["eval"][min(k, len(ex["eval"]) - 1)]
            sol = bl.exact_solve(ch, R, cfg["gamma"])
            tab[k, ch.pattern] = sol["pi"]
            q[1 + k, :ch.nz] = sol["Q"]
        tabs.append(tab)
        qst.append(q)
    return np.stack(tabs).astype(np.int32), np.stack(qst), ex_info


def run_baselines(cfg: dict, lanes: Lane, ids: list, lanes_static: Lane | None) -> dict:
    nev = cfg["eval_cpis"]
    nl = len(ids)
    tab_pi, qstar, _ = exact_tables(ids, cfg)
    tab_saa = np.tile(D.ASTAR_ACTION, (nl, D.N_SEG, 1)).astype(np.int32)
    dummy_full = jnp.zeros((nl, D.N_POS * D.N_VEL * D.N_THETA), jnp.int32)
    res = {}
    tm = Timer()
    ev_tab = make_policy_eval(nev, "table")
    tab_constb = np.full((nl, D.N_SEG, D.N_THETA), D.BLOCK_B, np.int32)   # constant 00111 (descriptive)
    for name, tab in (("pistar", tab_pi), ("saa", tab_saa), ("constb", tab_constb)):
        o, fp, th = ev_tab(lanes, jnp.asarray(tab), dummy_full)
        res[name] = dict(outs=np.asarray(o), fps=np.asarray(fp))
        if name == "pistar":
            th = np.asarray(th)
            res["eval_stream_sha256"] = [hashlib.sha256(th[i].tobytes()).hexdigest() for i in range(nl)]
    o, fp, _ = make_policy_eval(nev, "random")(lanes, jnp.asarray(tab_saa), dummy_full)
    res["random"] = dict(outs=np.asarray(o), fps=np.asarray(fp))
    info("baselines pi*/SAA/random evaluated", seconds=round(tm.lap(), 2))
    # policy iteration on the learners' training data (Markov experiments; FH too, it is cheap)
    coll = collect_training(cfg["train_cpis"])
    ev_full = make_policy_eval(nev, "full")
    for tag, ln in (("pi", lanes), ("pi_static", lanes_static)):
        if ln is None:
            continue
        tables, meta, train_fp = [], [], []
        for g0 in range(0, nl, 20):
            sl = jax.tree.map(lambda x: x[g0:g0 + 20], ln)
            s, a, r, s2, th2 = [np.asarray(x) for x in coll(sl)]
            for i in range(s.shape[0]):
                out = bl.policy_iteration(s[i], a[i], r[i], s2[i], cfg["gamma"],
                                          strict_improvement=bool(cfg.get("pi_strict_improvement", False)))
                tables.append(out["table"])
                meta.append(dict(iters=out["iters"], n_seen_states=out["n_seen_states"], n_seen_sa=out["n_seen_sa"],
                                 converged=out["converged"]))
                train_fp.append(hashlib.sha256(th2[i].astype(np.int8).tobytes()).hexdigest())
        o, fp, _ = ev_full(ln, jnp.asarray(tab_saa), jnp.asarray(np.stack(tables)))
        res[tag] = dict(outs=np.asarray(o), fps=np.asarray(fp), meta=meta, train_stream_sha256=train_fp)
        info("policy iteration done", variant=tag, seconds=round(tm.lap(), 2),
             mean_seen_states=float(np.mean([m["n_seen_states"] for m in meta])))
    res["qstar"] = qstar
    return res


# ----------------------------------------------------------------------------- main
def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "default.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--experiments", nargs="*")
    ap.add_argument("--readings", nargs="*")
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--train-cpis", type=int)
    ap.add_argument("--eval-cpis", type=int)
    ap.add_argument("--chunk-cpis", type=int)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--baselines-only", action="store_true", help="run the baselines (incl. PI) and no learner arm")
    ap.add_argument("--seed-root", type=int, help="override seed_root (fresh seed pool for a confirmatory launch)")
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    for k, v in (("arms", args.arms), ("experiments", args.experiments), ("readings", args.readings),
                 ("seeds", args.seeds), ("train_cpis", args.train_cpis), ("eval_cpis", args.eval_cpis),
                 ("chunk_cpis", args.chunk_cpis), ("seed_root", args.seed_root)):
        if v:
            cfg[k] = v
    if args.baselines_only:
        cfg["arms"] = []
    os.makedirs(args.out, exist_ok=True)
    check_device()
    info("config", **{k: cfg[k] for k in ("experiments", "readings", "arms", "seeds", "train_cpis", "eval_cpis")})
    seeds = list(range(cfg["seeds"]))
    lanes, ids = build_lanes(cfg, cfg["experiments"], cfg["readings"], seeds)
    lanes_static = None
    if cfg.get("pi_static_target_sensitivity"):   # descriptive; built for every lane (reported for E1/E2)
        lanes_static, _ = build_lanes(cfg, cfg["experiments"], cfg["readings"], seeds, p_move=0.0)
    write_json(os.path.join(args.out, "lanes.json"), dict(ids=ids, config=cfg))
    tm = Timer()
    if not args.skip_baselines:
        res = run_baselines(cfg, lanes, ids, lanes_static)
        np.savez_compressed(os.path.join(args.out, "baselines.npz"),
                            **{f"{k}__{kk}": np.asarray(vv) for k, v in res.items() if isinstance(v, dict)
                               for kk, vv in v.items() if kk in ("outs", "fps")},
                            qstar=res["qstar"])
        write_json(os.path.join(args.out, "baselines_meta.json"),
                   dict(eval_stream_sha256=res["eval_stream_sha256"],
                        pi_meta=res["pi"]["meta"], pi_train_stream_sha256=res["pi"]["train_stream_sha256"],
                        pi_static_meta=res.get("pi_static", {}).get("meta")))
        qstar = res["qstar"]
    else:
        qstar = exact_tables(ids, cfg)[1]
    info("baselines total", seconds=round(tm.lap(), 2))
    for arm in cfg["arms"]:
        t0 = time.time()
        r = run_arm(cfg, arm, lanes, ids, args.out, qstar)
        np.savez_compressed(os.path.join(args.out, f"arm_{arm}.npz"), **r)
        info("arm done", arm=arm, seconds=round(time.time() - t0, 1))
    info("all done", seconds=round(tm.lap(), 1))


if __name__ == "__main__":
    main()

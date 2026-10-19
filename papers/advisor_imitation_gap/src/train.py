"""One on-policy trainer for every arm (PPO, BC, DAgger, BC^tf=1, BC+PPO, IL->PPO pipelines, ADVISOR, IL->ADVISOR),
written as a pure jitted update vmapped over runs (arm x seed x hyperparameter x env-parameter). Arms are DATA
(a per-run stage table), so one compiled program trains all arms side by side and arms differ only in the loss mode,
the teacher-forcing schedule and the stage boundaries.

Rollout/update semantics follow AllenAct v0.5.0:
  * 20 parallel envs x 100-step rollouts (= 2000 steps per update); envs auto-reset; the LSTM state is reset at
    episode starts to a trainable initial state; the update re-unrolls each env's 100 steps from the stored carry.
  * GAE(gamma 0.99, lambda 1); advantages normalised over the whole rollout; 4 epochs x 2 minibatches of 10 whole
    env sequences (contiguous sampler groups, order shuffled per epoch).
  * Adam(0.9, 0.999, eps 1e-8), grad-norm clip 0.5, lr linearly decayed to 0 over the total step budget (LambdaLR);
    PPO clip 0.1 linearly decayed to 0 over the stage that contains PPO (step count = steps taken in the stage).
  * teacher forcing: with prob tf(stage steps) the teacher's action is executed (DAgger 1->0, BC^tf=1 = 1).
  * stages end after max_stage_steps (checked after each update) or on early stopping (2D-LH StopIfNearOptimal).

PRNG key tree (one top-level seed per run):
  root = PRNGKey(seed); k_init = fold_in(root, 0); k_env = fold_in(root, 1); k_roll = fold_in(root, 2);
  k_mb = fold_in(root, 3); k_eval = fold_in(root, 4).
  update u, rollout step t: fold_in(fold_in(k_roll, u), t) -> (action sample, tf coins, env resets) per env;
  update u, epoch e: fold_in(fold_in(k_mb, u), e) -> minibatch order; eval c: fold_in(k_eval, c) -> eval sampling.
  Arms with the same seed share k_init and the initial env draws (paired by construction).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import algorithm as A
from .data import EnvSpec

# ------------------------------------------------------------------ arms (stage tables)
# (mode, fraction of total steps or 'rest', tf_start, tf_end). Fractions use stage_split (TF_RATIO).
ARMS: Dict[str, List[Tuple[str, str, float, float]]] = {
    "ppo": [("ppo", "all", 0.0, 0.0)],
    "bc": [("imitation", "all", 0.0, 0.0)],
    "dagger": [("imitation", "split", 1.0, 0.0), ("imitation", "rest", 0.0, 0.0)],
    "bc_tf1": [("imitation", "all", 1.0, 1.0)],
    "bc_ppo": [("bc_ppo", "all", 0.0, 0.0)],
    "bc_then_ppo": [("imitation", "split", 0.0, 0.0), ("ppo", "rest", 0.0, 0.0)],
    "dagger_then_ppo": [("imitation", "split", 1.0, 0.0), ("ppo", "rest", 0.0, 0.0)],
    "bc_tf1_then_ppo": [("imitation", "split", 1.0, 1.0), ("ppo", "rest", 0.0, 0.0)],
    "advisor": [("advisor", "all", 0.0, 0.0)],
    # official X->ADV warm-up = AdvisorImitationStage (main AND aux CE); an earlier version trained the main head only
    "dagger_then_advisor": [("advisor_imitation", "split", 1.0, 0.0), ("advisor", "rest", 0.0, 0.0)],
    "bc_tf1_then_advisor": [("advisor_imitation", "split", 1.0, 1.0), ("advisor", "rest", 0.0, 0.0)],
}
MAX_STAGES = 2


def stage_table(arm: str, total_steps: int, split: float):
    """-> (mode[S], length[S], tf0[S], tf1[S], n_stages) with S = MAX_STAGES (official int(total * TF_RATIO))."""
    st = ARMS[arm]
    n_first = int(total_steps * split)
    mode, length, tf0, tf1 = [], [], [], []
    for m, frac, a, b in st:
        L = {"all": total_steps, "split": n_first, "rest": total_steps - n_first}[frac]
        mode.append(A.MODE_ID[m]); length.append(L); tf0.append(a); tf1.append(b)
    n = len(st)
    while len(mode) < MAX_STAGES:
        mode.append(0); length.append(1); tf0.append(0.0); tf1.append(0.0)
    return (np.array(mode, np.int32), np.array(length, np.int32), np.array(tf0, np.float32),
            np.array(tf1, np.float32), np.int32(n))


@dataclass(frozen=True)
class TrainCfg:
    env: str
    total_steps: int                 # Table A.3 (3e5 PD/2D-LH, 1e6 MiniGrid)
    n_envs: int = 20                 # Table A.3 (# processes)
    T: int = 100                     # Table A.3 rollout length
    epochs: int = 4                  # Table A.3
    n_mb: int = 2                    # official num_mini_batch (= 'rollouts per batch 10')
    gamma: float = 0.99              # Table A.3
    lam: float = 1.0                 # Table A.3
    clip: float = 0.1                # Table A.3
    max_grad_norm: float = 0.5       # Table A.3
    value_coef: float = 0.5          # Table A.3
    entropy_coef: float = 0.01       # official PPOConfig
    adam_eps: float = 1e-8           # torch default (official)
    early_stop: bool = False         # 2D-LH only (official StopIfNearOptimal)
    es_every_updates: int = 5        # official metric_accumulate_interval = 10 * MAX_STEPS = 10000 steps
    es_memory: int = 50              # official min_memory_size
    es_max_memory: int = 1024        # buffer for the official GROWING memory (>= episodes per 10k-step interval)
    es_tol: float = 0.05             # official deviation = 5 % of optimal

    @property
    def steps_per_update(self):
        return self.n_envs * self.T

    @property
    def n_updates(self):
        return math.ceil(self.total_steps / self.steps_per_update)


# ------------------------------------------------------------------ state
def _tree_where(c, a, b):
    return jax.tree_util.tree_map(lambda x, y: jnp.where(c, x, y), a, b)


def es_step(state, count, ave):
    """One call of the official StopIfNearOptimal memory update (allenact_plugins/lighthouse_plugin/lighthouse_util.py)
    with a tracker holding `count` > 0 episodes of mean length `ave`. Ported literally, including its quirks: the
    memory GROWS to `count` when one interval has more episodes than it holds; in the partial branch `current_pos` is
    never advanced unless the write overflows, and then becomes `pos + count % n` (operator precedence).
    state = (mem[es_max_memory], n, pos, filled); returns the new state."""
    mem, n, pos, filled = state
    idx = jnp.arange(mem.shape[0])
    count = jnp.minimum(count, mem.shape[0])
    big = count >= n
    n_big = jnp.maximum(n, count)
    mem_big = jnp.where(idx < n_big, ave, mem)
    over = pos + count > n
    mem_s = jnp.where((idx >= pos) & (idx < jnp.minimum(pos + count, n)), ave, mem)
    pos_s = jnp.where(over, pos + count % jnp.maximum(n, 1), pos)
    mem_s = jnp.where(over & (idx < pos_s), ave, mem_s)
    return (jnp.where(big, mem_big, mem_s), jnp.where(big, n_big, n), jnp.where(big, 0, pos_s),
            filled | big | over)


def es_mean(mem, n):
    return jnp.sum(jnp.where(jnp.arange(mem.shape[0]) < n, mem, 0.0)) / n


def es_update(state, count, ave, optimal, tol=0.05, min_memory=50, max_memory=1024):
    """Host-side driver of es_step with the official call semantics (no update when the interval had no episode);
    returns (state, stop, memory mean). Used by tests/test_official_parity.py."""
    if state is None:
        state = (jnp.zeros(max_memory, jnp.float64 if jax.config.jax_enable_x64 else jnp.float32),
                 jnp.int32(min_memory), jnp.int32(0), jnp.bool_(False))
    if count > 0:
        state = es_step(state, jnp.int32(count), ave)
    mean = float(es_mean(state[0], state[1]))
    return state, bool(state[3]) and mean < optimal + optimal * tol, mean


def clip_grad_norm_torch(max_norm: float) -> optax.GradientTransformation:
    """torch.nn.utils.clip_grad_norm_: g * min(1, max_norm / (||g|| + 1e-6)) (optax's version has no 1e-6)."""
    def update(g, state, params=None):
        coef = jnp.minimum(1.0, max_norm / (optax.global_norm(g) + 1e-6))
        return jax.tree_util.tree_map(lambda x: x * coef.astype(x.dtype), g), state
    return optax.GradientTransformation(lambda p: optax.EmptyState(), update)


def make_optimizer(cfg: TrainCfg):
    return optax.chain(clip_grad_norm_torch(cfg.max_grad_norm),
                       optax.scale_by_adam(b1=0.9, b2=0.999, eps=cfg.adam_eps))


def init_run(cfg: TrainCfg, spec: EnvSpec, net, run):
    """run: dict(seed key 'key', 'ep' env params, ...). Returns the per-run training state."""
    root = run["key"]
    k_init, k_env = jax.random.fold_in(root, 0), jax.random.fold_in(root, 1)
    env_state = jax.vmap(lambda k: spec.reset(k, run["ep"]))(jax.random.split(k_env, cfg.n_envs))
    obs0 = jax.vmap(lambda s: spec.obs(s, run["ep"]))(env_state)
    carry = net.initial_carry(cfg.n_envs)
    params = net.init(k_init, obs0, carry, jnp.ones(cfg.n_envs, bool))
    opt_state = make_optimizer(cfg).init(params)
    z = jnp.zeros((), jnp.int32)
    return dict(params=params, opt=opt_state, env=env_state, carry=carry, start=jnp.ones(cfg.n_envs, bool),
                u=z, stage=z, stage_steps=z, total=z, finished=jnp.zeros((), bool), finish_steps=z - 1,
                es_mem=jnp.zeros(cfg.es_max_memory), es_n=z + cfg.es_memory, es_pos=z, es_filled=jnp.zeros((), bool),
                es_sum=jnp.zeros(()), es_cnt=jnp.zeros(()))


# ------------------------------------------------------------------ one update
def schedules(cfg: TrainCfg, lr0, total_before, stage_steps_before, L):
    """AllenAct LambdaLR / PPO clip decay for the update that follows a rollout. lr: linear decay to 0 over the total
    budget, evaluated at the steps taken BEFORE this rollout; clip: linear decay over the stage, evaluated at the stage
    steps INCLUDING this rollout (official PPO clip LinearDecay at loss time). Shared by make_update and the official-code
    full-update test (tests/test_official_parity.py), so the formulas exist once."""
    stage_steps_after = stage_steps_before + cfg.steps_per_update
    clip = cfg.clip * (1.0 - jnp.minimum(stage_steps_after, L) / L)
    lr = lr0 * (1.0 - jnp.minimum(total_before, cfg.total_steps) / cfg.total_steps)
    return clip, lr


def make_learner(cfg: TrainCfg, net):
    """The learning half of one update: epochs x minibatches of whole env sequences from a finished rollout.
    learn(params, opt_state, roll, carry0, orders, clip, lr, mode, alpha, beta) -> (params, opt_state, per-minibatch
    stats). roll: dict of [T, B, ...] arrays (obs, start, action, logp, value, ret, adv_norm, label, label_valid);
    orders: [epochs * n_mb] minibatch indices (contiguous groups of B / n_mb envs, as the official sampler)."""
    opt = make_optimizer(cfg)
    mb = cfg.n_envs // cfg.n_mb

    def unroll(params, obs, start, carry0):
        def body(c, x):
            o, s = x
            lg, ax, v, c = net.apply(params, o, c, s)
            return c, (lg, ax, v)
        _, (lg, ax, v) = jax.lax.scan(body, carry0, (obs, start))
        return lg, ax, v

    def learn(params, opt_state, roll, carry0, orders, clip, lr, mode, alpha, beta):
        def loss_fn(p, batch, c0):
            lg, ax, v = unroll(p, batch["obs"], batch["start"], c0)
            terms = A.per_step_terms(lg, ax, v, batch, clip)
            return A.loss_from_terms(terms, mode, alpha, beta, cfg.value_coef, cfg.entropy_coef)

        def mb_step(c, x):
            p, o_s = c
            g_idx = x
            sl = lambda a: jax.lax.dynamic_slice_in_dim(a, g_idx * mb, mb, axis=1)
            batch = jax.tree_util.tree_map(sl, roll)
            c0 = jax.tree_util.tree_map(lambda a: jax.lax.dynamic_slice_in_dim(a, g_idx * mb, mb, axis=0), carry0)
            (loss, stats), g = jax.value_and_grad(loss_fn, has_aux=True)(p, batch, c0)
            upd, o_s = opt.update(g, o_s, p)
            p = jax.tree_util.tree_map(lambda w, d_: w - lr * d_, p, upd)
            stats["loss"] = loss
            stats["grad_norm"] = optax.global_norm(g)
            return (p, o_s), stats

        (p, o_s), stats = jax.lax.scan(mb_step, (params, opt_state), orders)
        return p, o_s, stats

    return learn


def make_update(cfg: TrainCfg, spec: EnvSpec, net):
    B, T = cfg.n_envs, cfg.T
    spu = cfg.steps_per_update
    learn = make_learner(cfg, net)

    def update(st, run):
        ep = run["ep"]
        params = st["params"]
        mode = run["mode"][st["stage"]]
        L = run["length"][st["stage"]]
        tf0, tf1 = run["tf0"][st["stage"]], run["tf1"][st["stage"]]
        k_roll = jax.random.fold_in(jax.random.fold_in(run["key"], 2), st["u"])
        k_mb = jax.random.fold_in(jax.random.fold_in(run["key"], 3), st["u"])

        # ---------------- rollout
        pool = None
        if spec.pool_size:
            kp = jax.random.split(jax.random.fold_in(k_roll, 1_000_003), B * spec.pool_size)
            pool = jax.vmap(lambda kk: spec.reset(kk, ep))(kp)
            pool = jax.tree_util.tree_map(lambda x: x.reshape((B, spec.pool_size) + x.shape[1:]), pool)

        def step_fn(c, t):
            env, carry, start, pool_cnt = c
            k = jax.random.fold_in(k_roll, t)
            k_a, k_tf, k_r = jax.random.split(k, 3)
            o = jax.vmap(lambda s: spec.obs(s, ep))(env)
            lg, _, v, carry_n = net.apply(params, o, carry, start)
            label = jax.vmap(lambda s: spec.teacher(s, ep))(env)
            a_pi = jax.random.categorical(k_a, lg)
            tf = tf0 + (tf1 - tf0) * jnp.minimum(st["stage_steps"] + B * t, L) / L
            force = jax.random.uniform(k_tf, (B,)) < tf
            a = jnp.where(force, label, a_pi)
            logp = A.take(A.log_softmax(lg), a)
            env_n, r, d = jax.vmap(lambda s, aa: spec.step(s, aa, ep))(env, a)
            info = jax.vmap(spec.info)(env_n)
            if spec.pool_size == 0:
                nxt = jax.vmap(lambda kk: spec.reset(kk, ep))(jax.random.split(k_r, B))
                used = jnp.zeros(B, jnp.int32)
            else:
                # fresh states come from a per-update pool (expensive resets, e.g. MiniGrid BFS), consumed in order
                fresh = jax.vmap(lambda p_, c_: jax.tree_util.tree_map(lambda x: x[c_], p_))(
                    pool, jnp.minimum(pool_cnt, spec.pool_size - 1))
                nxt, consumed = jax.vmap(lambda s_, f_: spec.next_episode(s_, f_, ep))(env_n, fresh)
                used = (d & consumed).astype(jnp.int32)
            env_n = _tree_where_b(d, nxt, env_n)
            rec = dict(obs=o, start=start, action=a, logp=logp, value=v, reward=r, done=d, label=label,
                       label_valid=jnp.ones(B, bool))
            return (env_n, carry_n, d, pool_cnt + used), (rec, info)

        carry0 = st["carry"]
        (env, carry, start, pool_cnt), (roll, infos) = jax.lax.scan(
            step_fn, (st["env"], carry0, st["start"], jnp.zeros(B, jnp.int32)), jnp.arange(T))
        o_last = jax.vmap(lambda s: spec.obs(s, ep))(env)
        _, _, v_last, _ = net.apply(params, o_last, carry, start)
        adv, ret = A.gae(roll["reward"], roll["value"], roll["done"], v_last, cfg.gamma, cfg.lam)
        roll["adv_norm"] = A.normalise(adv)
        roll["ret"] = ret

        clip, lr = schedules(cfg, run["lr"], st["total"], st["stage_steps"], L)
        orders = jnp.stack([jax.random.permutation(jax.random.fold_in(k_mb, e), cfg.n_mb) for e in range(cfg.epochs)])
        new_params, new_opt, stats = learn(params, st["opt"], roll, carry0, orders.reshape(-1), clip, lr, mode,
                                           run["alpha"], run["beta"])
        stats = jax.tree_util.tree_map(lambda x: x.mean(), stats)

        # ---------------- bookkeeping
        frozen = st["finished"]
        new_params = _tree_where(frozen, params, new_params)
        new_opt = _tree_where(frozen, st["opt"], new_opt)
        n_ep = roll["done"].sum()
        ep_ret = jnp.where(roll["done"], infos["ret"], 0.0).sum()
        ep_len = jnp.where(roll["done"], infos["len"], 0.0).sum()
        total = st["total"] + jnp.where(frozen, 0, spu)
        stage_steps = st["stage_steps"] + spu

        # early stopping (official StopIfNearOptimal, checked every es_every_updates updates = every metric interval)
        es_sum, es_cnt = st["es_sum"] + ep_len, st["es_cnt"] + n_ep
        check = cfg.early_stop & (((st["u"] + 1) % cfg.es_every_updates) == 0) & (es_cnt > 0)
        es = es_step((st["es_mem"], st["es_n"], st["es_pos"], st["es_filled"]), es_cnt.astype(jnp.int32),
                     es_sum / jnp.maximum(es_cnt, 1.0))
        es_mem, es_n, es_pos, es_filled = (jnp.where(check, a, b) for a, b in
                                           zip(es, (st["es_mem"], st["es_n"], st["es_pos"], st["es_filled"])))
        # official 2D-LH configs attach StopIfNearOptimal only to stages WITHOUT teacher forcing (dagger,
        # bc_teacher_forcing[_then_ppo], dagger_then_ppo: stage 1 has no criterion; every tf = 0 stage has one).
        # Found in the full-scale DAgger no-gap cells; before this fix ES also ended the teacher-forced stage.
        es_stage = run["tf0"][st["stage"]] == 0.0
        es_stop = check & es_stage & es_filled & (es_mean(es_mem, es_n) < run["opt_len"] * (1.0 + cfg.es_tol))
        es_sum = jnp.where(check, 0.0, es_sum) if cfg.early_stop else jnp.zeros(())
        es_cnt = jnp.where(check, 0.0, es_cnt) if cfg.early_stop else jnp.zeros(())

        stage_done = (stage_steps >= L) | es_stop
        next_stage = st["stage"] + 1
        finished_now = stage_done & (next_stage >= run["n_stages"])
        finished = frozen | finished_now | (total >= cfg.total_steps)
        stage = jnp.where(stage_done & ~finished, next_stage, st["stage"])
        stage_steps = jnp.where(stage_done, 0, stage_steps)
        reset_es = stage_done
        es_mem = jnp.where(reset_es, 0.0, es_mem)
        es_n = jnp.where(reset_es, cfg.es_memory, es_n)
        es_filled = jnp.where(reset_es, False, es_filled)
        es_pos = jnp.where(reset_es, 0, es_pos)
        es_sum = jnp.where(reset_es, 0.0, es_sum)    # official logs (and clears) the tracker at a stage end
        es_cnt = jnp.where(reset_es, 0.0, es_cnt)
        finish_steps = jnp.where(finished & ~frozen, total, st["finish_steps"])

        new = dict(params=new_params, opt=new_opt, env=env, carry=carry, start=start, u=st["u"] + 1,
                   stage=stage, stage_steps=stage_steps, total=total, finished=finished, finish_steps=finish_steps,
                   es_mem=es_mem, es_n=es_n, es_pos=es_pos, es_filled=es_filled, es_sum=es_sum, es_cnt=es_cnt)
        m = dict(pool_overflow=jnp.sum(pool_cnt > spec.pool_size), n_ep=n_ep, ep_ret=ep_ret / jnp.maximum(n_ep, 1), ep_len=ep_len / jnp.maximum(n_ep, 1),
                 mode=mode, clip=clip, lr=lr, total=total, finished=finished, **stats)
        return new, m

    return update


def _tree_where_b(mask, a, b):
    """Per-env select for env-state pytrees with a leading batch axis."""
    def sel(x, y):
        m = mask.reshape(mask.shape + (1,) * (x.ndim - 1))
        return jnp.where(m, x, y)
    return jax.tree_util.tree_map(sel, a, b)


# ------------------------------------------------------------------ evaluation
def make_eval(cfg: TrainCfg, spec: EnvSpec, net, n_eval: int):
    """n_eval fixed evaluation tasks, one episode each, SAMPLED actions (official deterministic_agents=False)."""
    def evaluate(params, run, ckpt_idx):
        ep = run["ep"]
        k = jax.random.fold_in(jax.random.fold_in(run["key"], 4), ckpt_idx)
        ks = jax.random.split(k, n_eval)
        env = jax.vmap(lambda i, kk: spec.eval_reset(i, kk, ep))(jnp.arange(n_eval), ks)
        carry = net.initial_carry(n_eval)

        def body(c, t):
            env, carry, done, final = c
            o = jax.vmap(lambda s: spec.obs(s, ep))(env)
            lg, _, _, carry = net.apply(params, o, carry, jnp.full(n_eval, t == 0))
            a = jax.random.categorical(jax.random.fold_in(k, t + 1), lg)
            env_n, _, d = jax.vmap(lambda s, aa: spec.step(s, aa, ep))(env, a)
            env_n = _tree_where_b(done, env, env_n)
            just = d & ~done
            final = _tree_where_b(just, env_n, final)
            return (env_n, carry, done | d, final), None

        (env, _, done, final), _ = jax.lax.scan(body, (env, carry, jnp.zeros(n_eval, bool), env),
                                                jnp.arange(spec.max_steps))
        info = jax.vmap(spec.info)(final)
        return info
    return evaluate


# ------------------------------------------------------------------ driver
def build_runs(cfg: TrainCfg, specs: Sequence[dict]):
    """specs: list of dict(arm, seed, lr, alpha, beta, split, ep={...}, opt_len). -> stacked run pytree."""
    cols = {k: [] for k in ("key", "lr", "alpha", "beta", "mode", "length", "tf0", "tf1", "n_stages", "opt_len")}
    eps = []
    for r in specs:
        mode, length, tf0, tf1, n = stage_table(r["arm"], cfg.total_steps, r.get("split", 0.5))
        cols["key"].append(jax.random.PRNGKey(r["seed"]))
        cols["lr"].append(np.float32(r["lr"])); cols["alpha"].append(np.float32(r.get("alpha", 8.0)))
        cols["beta"].append(np.float32(r.get("beta", 0.0)))
        cols["mode"].append(mode); cols["length"].append(length); cols["tf0"].append(tf0); cols["tf1"].append(tf1)
        cols["n_stages"].append(n); cols["opt_len"].append(np.float32(r.get("opt_len", 0.0)))
        eps.append(r.get("ep", {}))
    run = {k: jnp.stack([jnp.asarray(x) for x in v]) for k, v in cols.items()}
    keys = sorted(eps[0])
    run["ep"] = {k: jnp.asarray(np.array([e[k] for e in eps])) for k in keys}
    return run


def run_experiment(cfg: TrainCfg, spec: EnvSpec, net, specs: Sequence[dict], *, eval_at: Sequence[int],
                   n_eval: int, ckpt_dir: str | None = None, log=print, chunk: int | None = None,
                   n_eval_mid: int | None = None):
    """Train all runs in one vmapped program. eval_at: update counts after which to evaluate (0 = before training).
    n_eval_mid: episodes for the non-final checkpoints (curves only); default n_eval. The final checkpoint always
    uses n_eval (the headline).
    Returns (per-update metrics [runs, U, ...] as numpy, list of eval dicts per checkpoint, final state)."""
    import time
    from .common import info, metric, nan_guard
    runs = build_runs(cfg, specs)
    n_runs = len(specs)
    init_v = jax.jit(jax.vmap(lambda r: init_run(cfg, spec, net, r)))
    update = make_update(cfg, spec, net)
    evaluate = make_eval(cfg, spec, net, n_eval)
    evaluate_mid = make_eval(cfg, spec, net, n_eval_mid) if n_eval_mid else evaluate

    def seg(st, runs, n):
        def body(s, _):
            s2, m = jax.vmap(update)(s, runs)
            nan_guard(s2["params"], {"m": m}, tag=0)
            return s2, m
        return jax.lax.scan(body, st, None, length=n)

    seg_j = jax.jit(seg, static_argnums=2)
    eval_final_j = jax.jit(jax.vmap(evaluate, in_axes=(0, 0, None)), static_argnums=2)
    eval_mid_j = jax.jit(jax.vmap(evaluate_mid, in_axes=(0, 0, None)), static_argnums=2)

    t0 = time.perf_counter()
    st = init_v(runs)
    start_u = 0
    if ckpt_dir is not None:
        st, start_u = _maybe_restore(ckpt_dir, st)
        if start_u:
            info("resumed from checkpoint", update=start_u)
    jax.block_until_ready(st)
    evals, metrics = [], []
    points = sorted(set(int(x) for x in eval_at))
    last = max(points)

    def eval_j(params, runs_, c, at):
        return (eval_final_j if at == last else eval_mid_j)(params, runs_, c)
    u = start_u
    if u == 0 and 0 in points:
        ev = eval_j(st["params"], runs, 0, 0)
        evals.append((0, jax.device_get(ev)))
    first = True
    for ci, target in enumerate([p for p in points if p > 0]):
        if target <= u:
            continue
        n = target - u
        t1 = time.perf_counter()
        st, m = seg_j(st, runs, n)
        jax.block_until_ready(st)
        dt = time.perf_counter() - t1
        if first:
            info("first segment (includes compile)", seconds=round(dt, 2), updates=n, runs=n_runs)
            first = False
        else:
            info("segment steady-state", seconds=round(dt, 2), updates=n, runs=n_runs,
                 env_steps_per_s=round(n * cfg.steps_per_update * n_runs / dt))
        metrics.append(jax.device_get(m))
        u = target
        ev = eval_j(st["params"], runs, ci + 1, u)
        evals.append((u, jax.device_get(ev)))
        mm = jax.device_get(m)
        metric("segment", update=u, steps=u * cfg.steps_per_update,
               train_ret_mean=float(np.mean(mm["ep_ret"][:, -1])),
               eval_ret_mean=float(np.mean(ev["ret"])))
        if ckpt_dir is not None:
            _save(ckpt_dir, st, u)
    info("experiment done", seconds=round(time.perf_counter() - t0, 1))
    met = ({k: np.swapaxes(np.concatenate([np.asarray(m[k]) for m in metrics], axis=0), 0, 1) for k in metrics[0]}
           if metrics else {})   # [runs, updates]
    return met, evals, st


def _save(ckpt_dir, st, u):
    import orbax.checkpoint as ocp
    import os
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(os.path.abspath(ckpt_dir), f"u{u:06d}")
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(path, jax.device_get(st), force=True)
    ckptr.wait_until_finished()


def _maybe_restore(ckpt_dir, st):
    import os
    import orbax.checkpoint as ocp
    if not os.path.isdir(ckpt_dir):
        return st, 0
    done = sorted(d for d in os.listdir(ckpt_dir) if d.startswith("u") and not d.endswith(".tmp"))
    if not done:
        return st, 0
    path = os.path.join(os.path.abspath(ckpt_dir), done[-1])
    restored = ocp.StandardCheckpointer().restore(path, jax.device_get(st))
    return jax.tree_util.tree_map(jnp.asarray, restored), int(done[-1][1:])


def _maybe_restore_at(ckpt_dir, st, u):
    """Restore the checkpoint written after update u (used by scripts/reeval_pd.py)."""
    import os
    import orbax.checkpoint as ocp
    path = os.path.join(os.path.abspath(ckpt_dir), f"u{u:06d}")
    restored = ocp.StandardCheckpointer().restore(path, jax.device_get(st))
    return jax.tree_util.tree_map(jnp.asarray, restored), u

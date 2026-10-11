"""A2C on the load-balancing environment with four baseline arms (none / state / multi / maml).

Usage:
    python -m src.train --config src/configs/small.yaml --arm maml --seed 0 --out runs/small
    python -m src.train ... --set iterations=10 --set meta_inner_reduction=sum

The jitted train step is pure: (train_state, consts) -> (train_state, metrics); all randomness
comes from the key tree in ``consts`` folded with the iteration counter stored in the state,
so a resumed run reproduces the uninterrupted one exactly.
"""
from __future__ import annotations

import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()

import argparse  # noqa: E402
import copy  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from functools import partial  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import yaml  # noqa: E402

from . import algorithm as alg  # noqa: E402
from .common import (Timer, check_device, git_dirty, git_sha, info, metric, nan_guard, warn,  # noqa: E402
                     write_json)
from .data import (KeyTree, LBInputs, fingerprint, full_sha, key_tree, lb_input_kwargs, sample_lb_batch,  # noqa: E402
                   seq_hashes, test_key, val_key)
from .envs import lb_heuristic_rollout, lb_rollout, make_lb_spec  # noqa: E402
from .model import init_actor, init_critic, init_critic_stack, n_params  # noqa: E402

ARMS = ("none", "state", "multi", "maml", "multi_shuffled")
# multi_shuffled = ABLATION (full-scale): identical to multi (net i is still trained on fixed sequence i), but the
# baseline of a rollout on sequence i is read from net (i+1) mod N -- a head fitted to a DIFFERENT input
# sequence.  It removes the input-matching of the baseline and keeps everything else (compute, nets, data).
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


# ---------------------------------------------------------------- config
def load_config(path: str, overrides=()) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base = load_config(os.path.join(os.path.dirname(path), cfg.pop("base")))
        base.update(cfg)
        cfg = base
    for ov in overrides:
        k, v = ov.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def batch_shape(cfg: dict, arm: str):
    """(groups, rollouts per group): rollouts in a group share one input sequence."""
    if arm in ("none", "state"):
        return int(cfg["n_envs"]), 1
    if arm in ("multi", "multi_shuffled"):
        if _mv_official(cfg):  # one rollout per agent, its net index drawn with replacement (official code)
            return int(cfg["n_envs"]), 1
        return int(cfg["mv_groups"]), int(cfg["mv_k"])
    if arm == "maml":
        return int(cfg["meta_z_per_iter"]), int(cfg["meta_k"])
    raise ValueError(arm)


def _mv_official(cfg: dict) -> bool:
    """mv_schedule 'groups' (default, pre-fix): mv_groups nets without replacement x mv_k
    rollouts, one Adam step per sampled net.  'official' (load_balance_actor_multi_critic_train.py): each of the n_envs
    agents draws a net index WITH replacement (np.random.choice(num_models, num_agents)) and its critic gradient, taken at
    the pre-iteration parameters, is applied as a separate Adam step on that net (critic_agents[m_i].apply_gradients
    inside the per-agent loop), so a net drawn twice takes two steps."""
    sch = str(cfg.get("mv_schedule", "groups"))
    if sch not in ("groups", "official"):
        raise ValueError(sch)
    return sch == "official"


# ---------------------------------------------------------------- build
def build(cfg: dict, arm: str, seed: int):
    spec = make_lb_spec(cfg)
    keys = key_tree(seed)
    k_actor, k_critic = jax.random.split(keys.init)
    hidden = tuple(cfg["hidden"])
    actor_m, actor_p = init_actor(k_actor, hidden, spec.n_servers, spec.obs_dim, cfg["activation"])
    if arm in ("multi", "multi_shuffled"):
        critic_m, critic_p = init_critic_stack(k_critic, hidden, spec.critic_dim, int(cfg["n_value_nets"]),
                                               cfg["activation"])
    else:
        critic_m, critic_p = init_critic(k_critic, hidden, spec.critic_dim, cfg["activation"])
    tx_a = optax.adam(float(cfg["lr_actor"]), eps=float(cfg["adam_eps"]))
    lr_c = float(cfg["meta_outer_lr"]) if arm == "maml" else float(cfg["lr_critic"])
    tx_c = optax.adam(lr_c, eps=float(cfg["adam_eps"]))
    critic_opt = jax.vmap(tx_c.init)(critic_p) if arm in ("multi", "multi_shuffled") else tx_c.init(critic_p)
    ts = {"actor": actor_p, "actor_opt": tx_a.init(actor_p), "critic": critic_p, "critic_opt": critic_opt,
          "it": jnp.int32(0)}
    inkw = lb_input_kwargs(cfg)
    fixed = sample_lb_batch(keys.inputs_fixedN, int(cfg["n_value_nets"]), **inkw)
    consts = {"keys": jnp.stack(list(keys)), "fixed": fixed}
    models = dict(spec=spec, actor=actor_m, critic=critic_m, tx_a=tx_a, tx_c=tx_c, inkw=inkw)
    return ts, consts, models


# ---------------------------------------------------------------- train step
def make_train_step(cfg: dict, arm: str, models: dict, pg_var: bool = True):
    spec, actor_m, critic_m, tx_a, tx_c, inkw = (models[k] for k in ("spec", "actor", "critic", "tx_a", "tx_c", "inkw"))
    G, k = batch_shape(cfg, arm)
    E = G * k
    gamma, scale = float(cfg["gamma"]), float(cfg["reward_scale"])
    n_actions = spec.n_servers
    pg_every = int(cfg.get("pg_var_every", 5))
    capply = critic_m.apply
    mv_official = arm in ("multi", "multi_shuffled") and _mv_official(cfg)

    def step(ts, consts):
        it = ts["it"]
        keys = KeyTree(*consts["keys"])
        k_in = jax.random.fold_in(keys.inputs_train, it)
        k_ro = jax.random.fold_in(keys.rollout, it)
        k_sc = jax.random.fold_in(keys.sched, it)
        if arm in ("multi", "multi_shuffled") and mv_official:
            idx = jax.random.choice(k_sc, int(cfg["n_value_nets"]), (G,), replace=True)
            group_inputs = jax.tree_util.tree_map(lambda x: x[idx], consts["fixed"])
        elif arm in ("multi", "multi_shuffled"):
            idx = jax.random.permutation(k_sc, int(cfg["n_value_nets"]))[:G]
            group_inputs = jax.tree_util.tree_map(lambda x: x[idx], consts["fixed"])
        else:
            idx = jnp.zeros((G,), jnp.int32)
            group_inputs = sample_lb_batch(k_in, G, **inkw)
        env_inputs = jax.tree_util.tree_map(lambda x: jnp.repeat(x, k, axis=0), group_inputs)
        policy = lambda o: actor_m.apply(ts["actor"], o)
        traj = jax.vmap(lambda inp, kk: lb_rollout(spec, policy, inp, kk))(env_inputs, jax.random.split(k_ro, E))
        rew = traj.rewards * scale
        ret = alg.discounted_returns(rew, gamma)  # [E, T]
        x = traj.cobs  # [E, T, dc]
        m = {}

        # ---------------- baseline (computed from PRE-update critic parameters), then critic step
        if arm == "none":
            b = jnp.zeros_like(ret)
            critic, critic_opt = ts["critic"], ts["critic_opt"]
        elif arm == "state":
            b = capply(ts["critic"], x)
            vl, g = jax.value_and_grad(lambda p: alg.value_loss(capply, p, x, ret, "mean"))(ts["critic"])
            upd, critic_opt = tx_c.update(g, ts["critic_opt"], ts["critic"])
            critic = optax.apply_updates(ts["critic"], upd)
        elif arm in ("multi", "multi_shuffled"):
            net_of_env = jnp.repeat(idx, k)
            n_nets = int(cfg["n_value_nets"])
            base_net = (net_of_env + 1) % n_nets if arm == "multi_shuffled" else net_of_env  # ablation: mismatched head
            per_env = jax.tree_util.tree_map(lambda p: p[base_net], ts["critic"])
            b = jax.vmap(capply)(per_env, x)

            def mv_loss(stack):
                pe = jax.tree_util.tree_map(lambda p: p[net_of_env], stack)
                err = jax.vmap(capply)(pe, x) - ret  # [E, T]
                per_group = jnp.mean((err * err).reshape(G, k, -1), axis=(1, 2))
                return jnp.sum(per_group)  # each sampled net: mean squared error over its own rollouts

            if mv_official:
                # per-agent gradient at the PRE-iteration parameters of its own net, then one Adam step per agent,
                # applied in agent order on that net with the net's own optimizer state
                def agent_loss(p_i, x_i, r_i):
                    err = capply(p_i, x_i) - r_i
                    return jnp.mean(err * err)

                own = jax.tree_util.tree_map(lambda p: p[net_of_env], ts["critic"])
                g_env = jax.vmap(jax.grad(agent_loss))(own, x, ret)

                def apply_one(carry, inp):
                    P, O = carry
                    g_i, m_i = inp
                    p_i = jax.tree_util.tree_map(lambda a: a[m_i], P)
                    o_i = jax.tree_util.tree_map(lambda a: a[m_i], O)
                    u, o_i = tx_c.update(g_i, o_i, p_i)
                    p_i = optax.apply_updates(p_i, u)
                    P = jax.tree_util.tree_map(lambda a, v: a.at[m_i].set(v), P, p_i)
                    O = jax.tree_util.tree_map(lambda a, v: a.at[m_i].set(v), O, o_i)
                    return (P, O), None

                (critic, critic_opt), _ = jax.lax.scan(apply_one, (ts["critic"], ts["critic_opt"]), (g_env, net_of_env))
            else:
                g = jax.grad(mv_loss)(ts["critic"])
                upd, new_opt = jax.vmap(tx_c.update)(g, ts["critic_opt"], ts["critic"])
                new_p = optax.apply_updates(ts["critic"], upd)
                mask = jnp.zeros((int(cfg["n_value_nets"]),), bool).at[idx].set(True)
                critic = alg.tree_select(mask, new_p, ts["critic"])
                critic_opt = alg.tree_select(mask, new_opt, ts["critic_opt"])
        elif arm == "maml":
            h = k // 2
            xs = x.reshape(G, k, *x.shape[1:])
            rs = ret.reshape(G, k, -1)
            fn = partial(alg.maml_crossfit, capply, steps=int(cfg["meta_inner_steps"]), lr=float(cfg["meta_inner_lr"]),
                         inner_reduction=str(cfg["meta_inner_reduction"]),
                         first_order=bool(cfg["meta_first_order"]), clip=float(cfg.get("meta_inner_clip", 0.0)))
            b1, b2, mg, aux = jax.vmap(lambda xx, rr: fn(ts["critic"], xx[:h], rr[:h], xx[h:], rr[h:]))(xs, rs)
            b = jnp.concatenate([b1, b2], axis=1).reshape(E, -1)
            mg = jax.tree_util.tree_map(lambda a: jnp.mean(a, axis=0), mg)
            upd, critic_opt = tx_c.update(mg, ts["critic_opt"], ts["critic"])
            critic = optax.apply_updates(ts["critic"], upd)
            m["maml_mse_unadapted"] = jnp.mean(aux["mse_unadapted"])
            m["maml_mse_crossfit"] = jnp.mean(aux["mse_crossfit"])
        else:
            raise ValueError(arm)

        adv = ret - b
        # ---------------- actor step
        ec = alg.entropy_coef(it, float(cfg["entropy_init"]), float(cfg["entropy_final"]), int(cfg["entropy_decay_iters"]))
        (aloss, aaux), ga = jax.value_and_grad(
            lambda p: alg.actor_loss(actor_m.apply, p, traj.obs, traj.actions, adv, ec, n_actions), has_aux=True)(ts["actor"])
        upd, actor_opt = tx_a.update(ga, ts["actor_opt"], ts["actor"])
        actor = optax.apply_updates(ts["actor"], upd)

        if pg_var:
            var_pg = jax.lax.cond(
                it % pg_every == 0,
                lambda: alg.pg_variance(alg.per_rollout_pg(actor_m.apply, ts["actor"], traj.obs, traj.actions, adv)),
                lambda: jnp.float32(jnp.nan))
        else:
            var_pg = jnp.float32(jnp.nan)
        m.update({
            "train_return": jnp.mean(jnp.sum(rew, axis=1)),  # undiscounted, scaled (same units as test)
            "baseline_mse": jnp.mean((b - ret) ** 2),
            "adv_std": jnp.std(adv),
            "ret_std": jnp.std(ret),
            "entropy": aaux["entropy"],
            "ent_coef": ec,
            "actor_loss": aloss,
            "var_pg": var_pg,
        })
        new = {"actor": actor, "actor_opt": actor_opt, "critic": critic, "critic_opt": critic_opt, "it": it + 1}
        nan_guard((new["actor"], new["critic"], m["actor_loss"]),
                  {"obs": traj.obs, "actions": traj.actions, "rewards": traj.rewards, "it": it}, tag=1)
        return new, m

    return step


# ---------------------------------------------------------------- evaluation
def make_eval(cfg: dict, models: dict):
    spec, actor_m = models["spec"], models["actor"]
    scale = float(cfg["reward_scale"])

    @jax.jit
    def evaluate(actor_p, test_inputs, key):
        n = test_inputs.arrivals.shape[0]
        policy = lambda o: actor_m.apply(actor_p, o)
        keys = jax.random.split(key, n)
        sto = jax.vmap(lambda inp, kk: lb_rollout(spec, policy, inp, kk).rewards)(test_inputs, keys)
        gre = jax.vmap(lambda inp, kk: lb_rollout(spec, policy, inp, kk, greedy=True).rewards)(test_inputs, keys)
        return jnp.sum(sto, axis=1) * scale, jnp.sum(gre, axis=1) * scale

    return evaluate


def heuristic_returns(cfg: dict, spec, test_inputs, names=("spt", "lrt", "eft", "jsq")) -> dict:
    out = {}
    for nm in names:
        r = jax.jit(jax.vmap(lambda inp: lb_heuristic_rollout(spec, nm, inp)))(test_inputs)
        out[nm] = np.asarray(jnp.sum(r, axis=1)) * float(cfg["reward_scale"])
    return out


def test_inputs_for(cfg: dict) -> LBInputs:
    """The held-out evaluation set.  eval_split='test' (default): the fixed test set (k_inputs_test);
    eval_split='val': a disjoint validation set (VAL_SEED) used ONLY for hyper-parameter selection."""
    key = val_key() if str(cfg.get("eval_split", "test")) == "val" else test_key()
    return sample_lb_batch(key, int(cfg["test_episodes"]), **lb_input_kwargs(cfg))


# ---------------------------------------------------------------- checkpointing
def _ckpt_manager(path: str):
    import orbax.checkpoint as ocp

    return ocp.CheckpointManager(os.path.abspath(path), options=ocp.CheckpointManagerOptions(max_to_keep=2, create=True))


def save_ckpt(mgr, ts):
    import orbax.checkpoint as ocp

    mgr.save(int(ts["it"]), args=ocp.args.StandardSave(ts))
    mgr.wait_until_finished()


def restore_ckpt(mgr, template):
    import orbax.checkpoint as ocp

    step = mgr.latest_step()
    if step is None:
        return None
    return mgr.restore(step, args=ocp.args.StandardRestore(template))


# ---------------------------------------------------------------- driver
def run(cfg: dict, arm: str, seed: int, out_dir: str, log_every: int | None = None) -> dict:
    check_device()
    run_dir = os.path.join(out_dir, arm, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    ts, consts, models = build(cfg, arm, seed)
    spec = models["spec"]
    info("build", arm=arm, seed=seed, actor_params=n_params(ts["actor"]), critic_params=n_params(ts["critic"]),
         batch=batch_shape(cfg, arm), run_dir=run_dir)
    test_inputs = test_inputs_for(cfg)
    fp = fingerprint(test_inputs)
    tsha = full_sha(test_inputs)
    _record_test_sha(cfg, tsha)
    test_hashes = set(seq_hashes(test_inputs))
    overlap = len(test_hashes & set(seq_hashes(consts["fixed"])))
    # start-up spot check on the sequences THIS arm draws at iteration 0 (G groups, as in the train step; the
    # full check over every iteration and both G=16 / G=1 splits is scripts/check_overlap.py)
    G0 = batch_shape(cfg, arm)[0]
    probe_train = sample_lb_batch(jax.random.fold_in(KeyTree(*consts["keys"]).inputs_train, 0), G0,
                                  **models["inkw"])
    overlap += len(test_hashes & set(seq_hashes(probe_train)))
    if str(cfg.get("eval_split", "test")) == "val":  # validation must also be disjoint from the test set
        overlap += len(test_hashes & set(seq_hashes(sample_lb_batch(test_key(), int(cfg["test_episodes"]),
                                                                     **models["inkw"]))))
    info("test inputs", split=cfg.get("eval_split", "test"), fingerprint=fp, sha256=tsha[:16],
         train_test_hash_overlap=overlap)
    if overlap:
        raise RuntimeError("test input sequence found in training inputs")

    step_fn = jax.jit(make_train_step(cfg, arm, models), donate_argnums=(0,))
    evaluate = make_eval(cfg, models)
    keys = KeyTree(*consts["keys"])

    mgr = _ckpt_manager(os.path.join(run_dir, "ckpt"))
    metrics_path = os.path.join(run_dir, "metrics.json")
    cfg_hash_path = os.path.join(run_dir, "ckpt.config.sha256")  # next to ckpt/, outside orbax's directory
    chash = config_hash(cfg)
    history, evals = [], []
    if mgr.latest_step() is not None:
        _check_resume_config(cfg, chash, cfg_hash_path, metrics_path)
    restored = restore_ckpt(mgr, ts)
    if restored is not None:
        ts = restored
        import json

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                old = json.load(f)
            history = [h for h in old.get("history", []) if h["it"] < int(ts["it"])]
            evals = [e for e in old.get("evals", []) if e["it"] <= int(ts["it"])]
        info("resumed from checkpoint", it=int(ts["it"]), config_sha256=chash[:16])
    with open(cfg_hash_path, "w") as f:
        f.write(chash + "\n")

    n_iter = int(cfg["iterations"])
    eval_every = int(cfg["eval_every"])
    eval_start = int(cfg.get("eval_start", 0))  # test evals only at it >= eval_start (and it = 0); training unaffected
    log_every = log_every or eval_every
    steps_per_iter = int(cfg["n_envs"]) * spec.n_jobs
    assert np.prod(batch_shape(cfg, arm)) == int(cfg["n_envs"]), "every arm must use the same number of rollouts"

    wall0 = time.perf_counter()
    t = Timer()
    compiled = step_fn.lower(ts, consts).compile()
    compile_s = t.lap()
    flops = _flops(compiled)
    info("first compile (train step)", seconds=round(compile_s, 2), flops_per_step=flops)
    t_eval0 = time.perf_counter()
    _ = evaluate(ts["actor"], test_inputs, jax.random.fold_in(keys.eval, 0))
    jax.block_until_ready(_)
    info("first compile+run (eval)", seconds=round(time.perf_counter() - t_eval0, 2))

    def do_eval(it):
        sto, gre = evaluate(ts["actor"], test_inputs, jax.random.fold_in(keys.eval, it))
        sto, gre = np.asarray(sto), np.asarray(gre)
        e = {"it": it, "timesteps": it * steps_per_iter, "test_mean": float(sto.mean()),
             "test_std": float(sto.std()), "test_se": float(sto.std(ddof=1) / np.sqrt(len(sto))),
             "greedy_mean": float(gre.mean())}
        metric("eval", arm=arm, seed=seed, **e)
        return e, sto

    last_ckpt = time.perf_counter()
    pending = []
    steady_t0, steady_it0 = None, None
    it = int(ts["it"])
    if it == 0 and not evals:
        e, _ = do_eval(0)
        evals.append(e)
    last_sto = None
    while it < n_iter:
        ts, m = compiled(ts, consts)
        pending.append(m)
        it += 1
        if steady_t0 is None:  # steady-state clock starts after the first (warm) call
            jax.block_until_ready(ts["it"])
            steady_t0, steady_it0 = time.perf_counter(), it
        ckpt_due = time.perf_counter() - last_ckpt > 60 * float(cfg["ckpt_minutes"])
        if it % log_every == 0 or it == n_iter or (ckpt_due and pending):
            host = jax.device_get(pending)  # one host sync per logging boundary (or checkpoint)
            for j, mm in enumerate(host):
                row = {kk: float(v) for kk, v in mm.items()}
                row["it"] = it - len(host) + j
                history.append(row)
            pending = []
            last = history[-1]
            rate = (it - steady_it0) / max(time.perf_counter() - steady_t0, 1e-9)
            metric("train", arm=arm, seed=seed, it=it, train_return=last["train_return"],
                   baseline_mse=last["baseline_mse"], entropy=last["entropy"], ent_coef=last["ent_coef"],
                   iters_per_s=round(rate, 2), env_steps_per_s=round(rate * steps_per_iter))
        if (it % eval_every == 0 and it >= eval_start) or it == n_iter:
            e, last_sto = do_eval(it)
            evals.append(e)
            write_json(metrics_path, _metrics_blob(cfg, arm, seed, fp, history, evals, compile_s,
                                                   steady_t0, steady_it0, it, steps_per_iter, last_sto, tsha, flops,
                                                   wall0))
        if ckpt_due:
            # history is flushed above, so metrics.json covers every iteration the checkpoint covers (no hole
            # between the last eval and the checkpoint after a resume)
            write_json(metrics_path, _metrics_blob(cfg, arm, seed, fp, history, evals, compile_s,
                                                   steady_t0, steady_it0, it, steps_per_iter, last_sto, tsha, flops,
                                                   wall0))
            save_ckpt(mgr, ts)
            last_ckpt = time.perf_counter()
    save_ckpt(mgr, ts)
    if last_sto is None:
        e, last_sto = do_eval(it)
    blob = _metrics_blob(cfg, arm, seed, fp, history, evals, compile_s, steady_t0, steady_it0, it, steps_per_iter,
                         last_sto, tsha, flops, wall0)
    write_json(metrics_path, blob)
    info("done", arm=arm, seed=seed, final_last5=blob["final"]["test_mean_last5"])
    return blob


# keys that may legitimately differ between the interrupted and the resumed invocation (extending a run, or
# changing the checkpoint cadence); every other key must match or the resume is refused
RESUME_FREE_KEYS = ("iterations", "ckpt_minutes")


def config_hash(cfg: dict) -> str:
    import hashlib
    import json

    c = {k: v for k, v in cfg.items() if k not in RESUME_FREE_KEYS}
    return hashlib.sha256(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()


def _check_resume_config(cfg: dict, chash: str, hash_path: str, metrics_path: str) -> None:
    """Refuse to resume from a checkpoint written under a different config."""
    import json

    if os.path.exists(hash_path):
        with open(hash_path) as f:
            old = f.read().split()[0]
    elif os.path.exists(metrics_path):  # legacy run directory without a hash file: use its stored config
        with open(metrics_path) as f:
            old = config_hash(json.load(f)["config"])
    else:
        raise RuntimeError(f"checkpoint in {os.path.dirname(hash_path)} has no config record; refusing to resume")
    if old != chash:
        raise RuntimeError(f"refusing to resume: checkpoint config sha {old[:16]} != current {chash[:16]} "
                           f"({os.path.dirname(hash_path)})")


def _flops(compiled) -> float | None:
    """XLA's static FLOP estimate of one compiled train step (compiled.cost_analysis())."""
    try:
        ca = compiled.cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0]
        return float(ca.get("flops")) if ca and ca.get("flops") is not None else None
    except Exception as e:  # pragma: no cover
        warn("cost_analysis failed", err=str(e))
        return None


def _record_test_sha(cfg: dict, sha: str) -> None:
    """runs/test_inputs_lb.sha256: written once, every later run must match it."""
    if str(cfg.get("eval_split", "test")) != "test" or str(cfg.get("env", "load_balance")) != "load_balance" \
            or int(cfg["jobs_per_episode"]) != 500 or int(cfg["test_episodes"]) != 100:
        return
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "test_inputs_lb.sha256")
    if os.path.exists(path):
        with open(path) as f:
            old = f.read().split()[0]
        if old != sha:
            raise RuntimeError(f"test-input SHA-256 changed: {old} != {sha}")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(sha + "  load_balance test inputs (PRNGKey(100000000), 100 x 500 jobs)\n")


def _metrics_blob(cfg, arm, seed, fp, history, evals, compile_s, steady_t0, steady_it0, it, steps_per_iter, last_sto,
                  tsha=None, flops=None, wall0=None):
    dt = time.perf_counter() - steady_t0 if steady_t0 else float("nan")
    rate = (it - steady_it0) / dt if steady_t0 and dt > 0 else float("nan")
    last5 = [e["test_mean"] for e in evals[-5:]]
    return {
        "arm": arm, "seed": seed, "git_sha": git_sha(), "git_dirty_src": git_dirty(), "config": cfg,
        "test_fingerprint": fp, "test_sha256": tsha, "devices": str(jax.devices()),
        "timing": {"first_compile_s": compile_s, "steady_iters_per_s": rate,
                   "steady_env_steps_per_s": rate * steps_per_iter, "flops_per_train_step": flops,
                   "wall_s_total": (time.perf_counter() - wall0) if wall0 else None},
        "history": history, "evals": evals,
        "final": {"it": it, "test_mean_last5": float(np.mean(last5)) if last5 else None,
                  "test_mean_last": evals[-1]["test_mean"] if evals else None,
                  "test_per_episode_last": [float(v) for v in last_sto] if last_sto is not None else None},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(CONFIG_DIR, "load_balance.yaml"))
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=str, default=None, help="comma list; runs sequentially in one process")
    ap.add_argument("--out", default="runs/lb")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [a.seed]
    for s in seeds:
        run(copy.deepcopy(cfg), a.arm, s, a.out)


if __name__ == "__main__":
    main()

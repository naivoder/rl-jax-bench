"""Oracle harness: CleanRL dqn_jax.py @ fe8d8a0 on gymnax CartPole-v1 -> differential fixture.

What runs UNMODIFIED from cleanrl/cleanrl/dqn_jax.py (extracted verbatim with `ast`, exec'd in a namespace):
  QNetwork, TrainState, linear_schedule, the jitted `update` function and the target-update statement.
Labelled harness parts (not CleanRL code): the main-loop transcription with gymnax CartPole-v1 in place of the
gymnasium SyncVectorEnv (one env), and a minimal SB3-ReplayBuffer stand-in with the same sampling rule
(np.random.randint(0, upper_bound, batch_size)); `dones` = gymnax `terminated` (CleanRL stores terminations).
Variant "double": ONE labelled source patch of `update`, the line
    q_next_target = jnp.max(q_next_target, axis=-1)
replaced by the double-Q selection (online argmax, target evaluation). Everything else identical.

Writes $OFFICIAL_ROOT/fixtures/cleanrl_{max,double}.npz with: init params, every update's batch, loss, q_pred, the params after
every 25th update and at the end, the target-sync global steps, epsilon per step and greedy actions.
Run: JAX_PLATFORMS=cpu uv run python scripts/official/cleanrl_harness.py (needs $OFFICIAL_ROOT/cleanrl and gymnax)
"""
from __future__ import annotations

import ast
import os
import random
import sys

import numpy as np

HERE = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external", "official"))   # see setup.sh
SRC = os.path.join(HERE, "cleanrl", "cleanrl", "dqn_jax.py")
COMMIT = "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"
MAX_LINE = "q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)"
DOUBLE_LINE = ("q_next_target = jnp.take_along_axis(q_next_target, jnp.argmax(q_network.apply(q_state.params, "
               "next_observations), axis=-1)[:, None], axis=-1)[:, 0]  # HARNESS PATCH: double-Q")


def extract(variant: str) -> dict:
    src = open(SRC).read()
    tree = ast.parse(src)
    seg = {}
    main_if = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ("QNetwork", "TrainState"):
            seg[node.name] = ast.get_source_segment(src, node)
        if isinstance(node, ast.FunctionDef) and node.name == "linear_schedule":
            seg["linear_schedule"] = ast.get_source_segment(src, node)
        if isinstance(node, ast.If) and "__main__" in ast.get_source_segment(src, node.test):
            main_if = node
    for node in ast.walk(main_if):
        if isinstance(node, ast.FunctionDef) and node.name == "update":
            lines = src.splitlines()[node.lineno - 2:node.end_lineno]      # include the @jax.jit decorator line
            ind = len(lines[0]) - len(lines[0].lstrip())
            seg["update"] = "\n".join(l[ind:] for l in lines)
    assert MAX_LINE in seg["update"]
    if variant == "double":
        seg["update"] = seg["update"].replace(MAX_LINE, DOUBLE_LINE)
    return seg


def run(variant: str, total=3000, learning_starts=500, train_frequency=10, target_freq=500, batch=128,
        buffer_size=10000, lr=2.5e-4, gamma=0.99, seed=1):
    import flax
    import flax.linen as nn
    import gymnax
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState as _TS

    seg = extract(variant)
    ns = dict(flax=flax, nn=nn, jax=jax, jnp=jnp, optax=optax, np=np, TrainState=_TS)
    exec(seg["QNetwork"], ns)
    exec(seg["TrainState"], ns)
    exec(seg["linear_schedule"], ns)

    class Args:
        pass

    args = Args()
    args.gamma, args.learning_rate, args.tau = gamma, lr, 1.0
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    key, q_key = jax.random.split(key, 2)
    env, env_params = gymnax.make("CartPole-v1")
    ekey = jax.random.PRNGKey(seed + 1000)
    ekey, rk = jax.random.split(ekey)
    obs, state = env.reset(rk, env_params)
    obs = np.asarray(obs)[None]
    q_network = ns["QNetwork"](action_dim=2)
    q_state = ns["TrainState"].create(apply_fn=q_network.apply, params=q_network.init(q_key, obs),
                                      target_params=q_network.init(q_key, obs), tx=optax.adam(learning_rate=lr))
    q_network.apply = jax.jit(q_network.apply)
    q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))
    ns.update(q_network=q_network, args=args)
    exec(seg["update"], ns)
    update = ns["update"]
    init_params = jax.device_get(q_state.params)
    B_obs = np.zeros((buffer_size, 4), np.float32)
    B_nobs = np.zeros((buffer_size, 4), np.float32)
    B_act = np.zeros((buffer_size, 1), np.int64)
    B_rew = np.zeros((buffer_size, 1), np.float32)
    B_done = np.zeros((buffer_size, 1), np.float32)
    pos, full = 0, False
    rec = dict(eps=[], actions=[], greedy_obs=[], greedy_act=[], greedy_step=[], b_obs=[], b_act=[], b_nobs=[], b_rew=[], b_done=[],
               loss=[], q_pred=[], upd_step=[], sync_step=[], snap_idx=[], snaps=[])
    for global_step in range(total):
        epsilon = ns["linear_schedule"](1.0, 0.05, 0.5 * total, global_step)
        rec["eps"].append(epsilon)
        if random.random() < epsilon:
            actions = np.array([random.randrange(2)])
        else:
            q_values = q_network.apply(q_state.params, obs)
            actions = np.asarray(jax.device_get(q_values.argmax(axis=-1)))
            rec["greedy_obs"].append(obs[0].copy())
            rec["greedy_act"].append(int(actions[0]))
            rec["greedy_step"].append(global_step)
        rec["actions"].append(int(actions[0]))
        ekey, sk = jax.random.split(ekey)
        nobs, state, r, done, _trunc, _ = env.step(sk, state, int(actions[0]), env_params)   # done = terminations (CleanRL rb.add)
        nobs = np.asarray(nobs)[None]
        real_next = nobs.copy()   # gymnax auto-resets; on done the stored next obs is the reset obs (harness label)
        B_obs[pos], B_nobs[pos], B_act[pos], B_rew[pos], B_done[pos] = obs[0], real_next[0], actions[0], float(r), float(done)
        pos += 1
        if pos == buffer_size:
            full, pos = True, 0
        obs = nobs
        if global_step > learning_starts:
            if global_step % train_frequency == 0:
                upper = buffer_size if full else pos
                idx = np.random.randint(0, upper, size=batch)
                o, a, no, rw, d = B_obs[idx], B_act[idx], B_nobs[idx], B_rew[idx].flatten(), B_done[idx].flatten()
                loss, old_val, q_state = update(q_state, o, a, no, rw, d)
                for k, v in (("b_obs", o), ("b_act", a[:, 0]), ("b_nobs", no), ("b_rew", rw), ("b_done", d),
                             ("loss", float(loss)), ("q_pred", np.asarray(old_val)), ("upd_step", global_step)):
                    rec[k].append(v)
                n = len(rec["loss"])
                if n % 25 == 0:
                    rec["snap_idx"].append(n)
                    rec["snaps"].append(jax.device_get(q_state.params))
            if global_step % target_freq == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau))
                rec["sync_step"].append(global_step)
    rec["final"] = jax.device_get(q_state.params)
    return init_params, rec


def flat(params) -> dict:
    import jax
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        out["/".join(str(getattr(p, "key", p)) for p in path)] = np.asarray(leaf)
    return out


def main():
    os.makedirs(os.path.join(HERE, "fixtures"), exist_ok=True)
    for variant in ("max", "double"):
        init_params, rec = run(variant)
        arr = {f"init/{k}": v for k, v in flat(init_params).items()}
        arr.update({f"final/{k}": v for k, v in flat(rec["final"]).items()})
        for i, sp in zip(rec["snap_idx"], rec["snaps"]):
            arr.update({f"snap{i}/{k}": v for k, v in flat(sp).items()})
        for k in ("eps", "actions", "greedy_obs", "greedy_act", "greedy_step", "b_obs", "b_act", "b_nobs", "b_rew", "b_done", "loss",
                  "q_pred", "upd_step", "sync_step", "snap_idx"):
            arr[k] = np.asarray(rec[k])
        arr["config"] = np.array([3000, 500, 10, 500, 128, 10000, 2.5e-4, 0.99, 1.0, 0.05, 0.5])
        out = os.path.join(HERE, "fixtures", f"cleanrl_{variant}.npz")
        np.savez_compressed(out, **arr)
        print(f"[INFO] wrote {out} updates={len(rec['loss'])} syncs={rec['sync_step']} commit={COMMIT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

"""Dump the exogenous env stream, explore actions, replay indices and init params of one lane from OUR key tree,
and (optionally) our own per-step trajectory through src/train.lane_step, for the Keras reference (keras_ref.py)."""
import sys, os, argparse
PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the paper folder
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from rl_jax_bench import gpu_bootstrap
gpu_bootstrap.setup()
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.path.join(PAPER, 'src'))
import algorithm as alg, data as D
import train as TR
from model import arm_config

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--exp", default="mc_p04")
ap.add_argument("--reading", default="eq3")
ap.add_argument("--seed-root", type=int, default=20261016)
ap.add_argument("--train-cpis", type=int, default=2)
ap.add_argument("--eval-cpis", type=int, default=1)
ap.add_argument("--ours", action="store_true", help="also run our lane_step and record the trajectory")
ap.add_argument("--out", required=True)
a = ap.parse_args()

cfg = TR.load_cfg(os.path.join(PAPER, 'src', 'configs', 'full.yaml'))
cfg["seed_root"] = a.seed_root
cfg["eval_cpis"] = 70  # segment layout only (E1 has one segment)
lanes, ids = TR.build_lanes(cfg, [a.exp], [a.reading], [a.seed])
lane = jax.tree.map(lambda x: x[0], lanes)
ntr = a.train_cpis * D.CPI
ntot = (a.train_cpis + a.eval_cpis) * D.CPI

# ---- env stream (action-independent dynamics), mirrored from lane_step's env handling
def env_scan(env, t):
    is_train = t < ntr
    t_stream = jnp.where(is_train, t, t - ntr)
    k_stream = jnp.where(is_train, lane.k_env_train, lane.k_env_eval)
    env = TR._sel(t == ntr, D.env_reset(lane.k_env_eval, lane.params), env)
    obs = D.obs_of(env, lane.params)
    T_cur = D.T_for_cpi(lane.params, t // D.CPI, a.train_cpis)
    env2, r, (coll, miss, size, th2) = D.env_step(env, jnp.int32(0), t_stream, T_cur, k_stream, lane.params)
    obs2 = D.obs_of(env2, lane.params)
    a_rand = jax.random.randint(jax.random.fold_in(lane.k_explore, t), (), 0, D.N_ACT)
    return env2, (obs, obs2, th2, a_rand)

env0 = D.env_reset(lane.k_env_train, lane.params)
_, (obs, obs2, th2, a_rand) = jax.jit(lambda e: jax.lax.scan(env_scan, e, jnp.arange(ntot)))(env0)
out = dict(obs=np.asarray(obs), obs2=np.asarray(obs2), th2=np.asarray(th2), a_rand=np.asarray(a_rand),
           R=np.asarray(lane.params.R), ntr=ntr, ntot=ntot)

# ---- replay indices per arm (count after the buffer write at step t, as in lane_step)
for arm in ("dqn", "ddrqn"):
    q = arm_config(arm, cfg)
    L, N = q.trace_len, q.buffer_traces
    def idx_t(t):
        count = jnp.minimum((t + 1) // L, N)
        return jax.random.randint(jax.random.fold_in(lane.k_replay, t), (q.batch_size,), 0, jnp.maximum(count, 1))
    out[f"idx_{arm}"] = np.asarray(jax.jit(jax.vmap(idx_t))(jnp.arange(ntot))).astype(np.int32)
    p0 = alg.init(lane.k_init, q).params
    flat, _ = jax.tree_util.tree_flatten_with_path(p0)
    for path, v in flat:
        out[f"init_{arm}__" + "/".join(str(getattr(k, 'key', k)) for k in path)] = np.asarray(v)

    if a.ours:
        net_step = lambda c, t: TR.lane_step(c, t, D.T_for_cpi(lane.params, t // D.CPI, a.train_cpis), lane, q, ntr, True, 0)
        def body(c, t):
            c2, o, fp, th = net_step(c, t)
            return c2, (o, c2.prev_a)
        c0 = TR.init_carry(lane, q)
        cT, (o, acts) = jax.jit(lambda c: jax.lax.scan(body, c, jnp.arange(ntot)))(c0)
        out[f"ours_{arm}_out"] = np.asarray(o)
        out[f"ours_{arm}_act"] = np.asarray(acts)
        flat, _ = jax.tree_util.tree_flatten_with_path(cT.q.params)
        for path, v in flat:
            out[f"final_{arm}__" + "/".join(str(getattr(k, 'key', k)) for k in path)] = np.asarray(v)

np.savez_compressed(a.out, **out)
print("saved", a.out, {k: v.shape for k, v in out.items() if hasattr(v, 'shape') and not k.startswith(('init', 'final'))})
print([k for k in out if k.startswith("init_ddrqn")])

"""official-code check reference curve (labelled second-best oracle): OUR DQN core on CleanRL's headline task.

No author code exists for Thornton et al. 2020 (see the README).
The labelled second-best oracle is CleanRL dqn_jax.py @ fe8d8a0, whose own headline experiment is CartPole-v1 at its
defaults (CleanRL's documented benchmark: 498.38 +- 2.29). That file is run UNMODIFIED in its own venv
($OFFICIAL_ROOT/.venv-cleanrl, jax 0.4.8 / flax 0.6.8 / gymnasium 0.29.1) by the oracle side; this
script runs OUR core (src/algorithm.py: QNetwork, sequence_loss, update, sync, TraceBuffer, linear_epsilon) through
gymnax CartPole-v1 with CleanRL's defaults:
  total 500k env steps, lr 2.5e-4, Adam eps 1e-8 (optax default, CleanRL), buffer 10000, gamma 0.99,
  hard target copy every 500 env steps (after learning_starts), batch 128, eps 1 -> 0.05 over 250k steps,
  learning_starts 10000, one update every 10 env steps, MLP 120-84, max target (CleanRL dqn) AND double target.
Labelled differences that remain (ours vs CleanRL): glorot_uniform kernels (our Keras-faithful init) vs flax
lecun_normal; gymnax vs gymnasium CartPole (same dynamics, uniform(-0.05, 0.05) reset, 500-step limit);
truncation bootstraps through info['final_observation'] exactly as CleanRL's real_next_obs.
Metric (identical to the oracle side, computed from the per-episode (global_step, return) stream):
  final = mean return of episodes completing in the last 10 % of steps (global_step >= 450000);
  auc   = mean return of all episodes completing at global_step >= learning_starts.
Key tree: root PRNGKey(seed) per seed (seeds 1..5); fold_in(root, 0) init, fold_in(root, 1) env reset/step,
fold_in(fold_in(root, 2), t) exploration, fold_in(fold_in(root, 3), t) replay sample.
Writes runs/official_ref/cartpole_ours{--tag}.json. --init lecun = the pre-declared our-code-first rerun.
"""
import os
import sys
import time

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import gymnax  # noqa: E402
from rl_jax_bench import jaxlog as L  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import algorithm as alg  # noqa: E402

TOTAL, LS, TF, TGT, BATCH, BUF, LR, GAMMA, S_E, E_E, FRAC = 500_000, 10_000, 10, 500, 128, 10_000, 2.5e-4, 0.99, 1.0, 0.05, 0.5
CHUNK = 10_000


def lecun_like_flax(key, params):
    """Standing-rule-1 option: replace every Dense kernel with flax's default lecun_normal draw (CleanRL's QNetwork
    init); biases stay zero. Changes only the initial weights, nothing in the core."""
    leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    keys = jax.random.split(key, len(leaves))
    init = jax.nn.initializers.lecun_normal()
    new = [init(k, x.shape, x.dtype) if str(getattr(path[-1], "key", "")) == "kernel" else x
           for k, (path, x) in zip(keys, leaves)]
    return jax.tree_util.tree_unflatten(treedef, new)


def make(double: bool, init_name: str):
    cfg = alg.QConfig(n_actions=2, obs_dim=4, hidden=(120, 84), lstm=0, double=double, gamma=GAMMA, lr=LR, adam_eps=1e-8,
                      trace_len=1, batch_size=BATCH, buffer_traces=BUF)
    env, ep = gymnax.make("CartPole-v1")
    net = alg.make_network(cfg)

    def init_one(seed):
        root = jax.random.PRNGKey(seed)
        q = alg.init(jax.random.fold_in(root, 0), cfg)
        if init_name == "lecun":
            p = lecun_like_flax(jax.random.fold_in(root, 4), q.params)
            q = alg.QState(p, jax.tree.map(jnp.copy, p), alg.make_optimizer(cfg).init(p), q.n_updates)
        ek = jax.random.fold_in(root, 1)
        obs, es = env.reset(jax.random.fold_in(ek, 0), ep)
        return dict(q=q, buf=alg.buffer_init(cfg), obs=obs, es=es, ret=jnp.zeros(()), root=root)

    def step_one(c, t):
        root = c["root"]
        eps = alg.linear_epsilon(t, S_E, E_E, FRAC * TOTAL)
        kx = jax.random.fold_in(jax.random.fold_in(root, 2), t)
        k1, k2 = jax.random.split(kx)
        qv, _ = alg.act_q(net, cfg, c["q"].params, c["obs"], ())
        a = jnp.where(jax.random.uniform(k1) < eps, jax.random.randint(k2, (), 0, 2), alg.greedy(qv))
        ks = jax.random.fold_in(jax.random.fold_in(root, 1), t + 1)
        nobs, es, r, term, trunc, info = env.step(ks, c["es"], a, ep)
        real_next = info["final_observation"]
        trace = dict(obs=jnp.stack([c["obs"], real_next]), act=a[None].astype(jnp.int32), rew=r[None].astype(jnp.float32),
                     disc=(GAMMA * (1.0 - term.astype(jnp.float32)))[None], resets=jnp.zeros((2,), bool))
        buf = alg.buffer_add(c["buf"], trace)
        done = jnp.logical_or(term, trunc)
        ret = c["ret"] + r
        out = (done, ret)

        def do_update(q):
            b = alg.buffer_sample(buf, jax.random.fold_in(jax.random.fold_in(root, 3), t), BATCH)
            q2, _ = alg.update(q, b, cfg)
            return q2

        q = jax.lax.cond((t > LS) & (t % TF == 0), do_update, lambda q: q, c["q"])
        q = alg.sync(q, (t > LS) & (t % TGT == 0))
        return dict(q=q, buf=buf, obs=nobs, es=es, ret=jnp.where(done, 0.0, ret), root=root), out

    vstep = jax.vmap(step_one, in_axes=(0, None))

    @jax.jit
    def run_chunk(c, t0):
        return jax.lax.scan(lambda cc, t: vstep(cc, t), c, t0 + jnp.arange(CHUNK))

    return jax.vmap(init_one), run_chunk


def summarize(steps, rets):
    steps, rets = np.asarray(steps), np.asarray(rets)
    return dict(final=float(rets[steps >= 0.9 * TOTAL].mean()), auc=float(rets[steps >= LS].mean()),
                n_episodes=int(len(rets)))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="glorot", choices=["glorot", "lecun"])
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--variants", default="max,double")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    L.check_device()
    seeds = np.asarray([int(x) for x in args.seeds.split(",")])
    out = dict(config=dict(total=TOTAL, learning_starts=LS, train_frequency=TF, target_freq=TGT, batch=BATCH, buffer=BUF,
                           lr=LR, gamma=GAMMA, start_e=S_E, end_e=E_E, exploration_fraction=FRAC, hidden=[120, 84],
                           adam_eps=1e-8, init=args.init), device=str(jax.devices()[0]), seeds=seeds.tolist())
    for variant in args.variants.split(","):
        init_fn, run_chunk = make(variant == "double", args.init)
        c = init_fn(jnp.asarray(seeds))
        t0 = time.time()
        c, (d, r) = run_chunk(c, jnp.int32(0))
        jax.block_until_ready(d)
        L.info("first chunk (compile + run)", variant=variant, secs=round(time.time() - t0, 2))
        dones, rets = [np.asarray(d)], [np.asarray(r)]
        t1 = time.time()
        for k in range(1, TOTAL // CHUNK):
            c, (d, r) = run_chunk(c, jnp.int32(k * CHUNK))
            dones.append(np.asarray(d))
            rets.append(np.asarray(r))
            if not all(bool(np.isfinite(np.asarray(x)).all()) for x in jax.tree_util.tree_leaves(c["q"].params)):
                L.error("NaN/Inf in params -- aborting", variant=variant, chunk=k)    # NaN/Inf guard (once per chunk)
                sys.exit(3)
        dt = time.time() - t1
        L.info("steady state", variant=variant, secs=round(dt, 1), env_steps_per_sec=round((TOTAL - CHUNK) * len(seeds) / dt))
        D, R = np.concatenate(dones), np.concatenate(rets)       # [TOTAL, n_seeds]
        per = {}
        for j, s in enumerate(seeds):
            st = np.nonzero(D[:, j])[0]
            per[int(s)] = dict(**summarize(st, R[st, j]), curve_steps=st.tolist()[::5], curve_returns=R[st, j].tolist()[::5])
            L.metric("cartpole ours", variant=variant, seed=int(s), final=round(per[int(s)]["final"], 2),
                     auc=round(per[int(s)]["auc"], 2), episodes=per[int(s)]["n_episodes"])
        out[variant] = per
    os.makedirs(os.path.join(ROOT, "runs", "official_ref"), exist_ok=True)
    L.write_json(os.path.join(ROOT, "runs", "official_ref", f"cartpole_ours{args.tag}.json"), out)


if __name__ == "__main__":
    main()

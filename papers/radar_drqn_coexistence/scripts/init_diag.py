"""official-code check follow-up (our-code-first rule, descriptive): the CleanRL reference run showed our core is init-sensitive
(glorot_uniform vs flax lecun_normal) on CartPole. Is the toy check 6b DDRQN shortfall on the one-hot toy also an
init effect? Same toy, Table-I hypers, R_lin, p = 0.4, 100k steps, 10 seeds; Dense kernels (torso, head, LSTM input
kernel) redrawn lecun_normal, the LSTM recurrent kernel stays orthogonal. Writes runs/toys/init_diag.json.
Run on CPU (JAX_PLATFORMS=cpu)."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import toys as G  # bootstraps (GPU if visible)
import jax, jax.numpy as jnp
import algorithm as alg
import train as TR

_orig = alg.init


def lecun_init(key, cfg):
    st = _orig(key, cfg)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(st.params)
    keys = jax.random.split(jax.random.fold_in(key, 77), len(leaves))
    f = jax.nn.initializers.lecun_normal()
    names = lambda path: [str(getattr(p, "key", p)) for p in path]
    new = [f(k, x.shape, x.dtype) if names(path)[-1] == "kernel" and "recurrent" not in names(path) else x
           for k, (path, x) in zip(keys, leaves)]
    p = jax.tree_util.tree_unflatten(treedef, new)
    return alg.QState(p, jax.tree.map(jnp.copy, p), alg.make_optimizer(cfg).init(p), st.n_updates)


cfg = TR.load_cfg(os.path.join(G.ROOT, "src", "configs", "default.yaml"))
rows = []
# glorot rows are toy checks's toy check 6b Table-I rows (runs/toys/toys.json): DQN 0.978 (re-run here once and reproduced
# exactly before the glorot DDRQN re-run was cut for CPU time), DDRQN 0.350, 3/10. Only lecun is run.
for init in ("lecun",):
    alg.init = lecun_init if init == "lecun" else _orig
    for arm in ("ddrqn", "dqn"):
        r = G.core_on_toy(arm, cfg, "lin", n_steps=100_000)
        r.update(init=init)
        rows.append(r)
        G.metric("init diag", arm=arm, init=init, act_agree=round(r["act_agree_mean"], 3),
                 final_policy=str(r["final_policy"]))
alg.init = _orig
json.dump(rows, open(os.path.join(G.ROOT, "runs", "toys", "init_diag.json"), "w"), indent=1, default=float)

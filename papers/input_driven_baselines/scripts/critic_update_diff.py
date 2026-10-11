"""Multi-critic update differential, OUR side.  One iteration of OUR multi-value critic update at the README config through the REAL
jitted train step (src.train.make_train_step, mv_schedule=official and groups), on CPU.  The batch the step used is
re-derived from the same key tree (same calls as src/train.py:123-142) and saved together with the pre-update critic
stack, the 16 net indices, and the post-update stacks of both schedules.  scripts/official/critic_update_official.py replays the authors' per-agent
CriticAgent update on that batch.

    JAX_PLATFORMS=cpu python scripts/critic_update_diff.py make <out.npz>
    JAX_PLATFORMS=cpu python scripts/critic_update_diff.py diff <out.npz>
"""
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import src.data as D  # noqa: E402
import src.train as T  # noqa: E402
from src.envs import lb_rollout  # noqa: E402
from src import algorithm as alg  # noqa: E402


def sample_uniform_inputs(key, n_jobs, mean_interarrival, pareto_shape=None, pareto_scale=None):
    k_size, k_gap = jax.random.split(key)
    sizes = jnp.floor(jax.random.uniform(k_size, (n_jobs,), minval=100.0, maxval=1000.0)).astype(jnp.int32)
    gaps = jnp.floor(jax.random.exponential(k_gap, (n_jobs,)) * mean_interarrival).astype(jnp.int32)
    return D.LBInputs(jnp.cumsum(gaps).astype(jnp.int32), sizes)


T.sample_lb_batch = lambda key, n, **kw: jax.vmap(lambda k: sample_uniform_inputs(k, **kw))(jax.random.split(key, n))
D.sample_lb_batch = T.sample_lb_batch

mode, path = sys.argv[1], sys.argv[2]
cfg = T.load_config(os.path.join(REPO, "src/configs/load_balance.yaml"))
cfg.update({"jobs_per_episode": 1000, "mean_interarrival": 100.0, "gamma": 1.0, "hidden": [200, 128],
            "activation": "leaky_relu", "lr_actor": 1e-3, "lr_critic": 1e-3, "entropy_init": 1.0,
            "entropy_final": 1e-4, "entropy_decay_iters": 1000, "pg_var_every": 1000000})
N = 10


def flat_stack(p):  # [N] x (W0, b0, W1, b1, W2, b2) in the authors' variable order
    d = p["params"]["MLP_0"]
    return [[np.asarray(d[f"Dense_{i}"][k][n]) for i in range(3) for k in ("kernel", "bias")] for n in range(N)]


def one_iteration(sched):
    c = dict(cfg, mv_schedule=sched)
    ts, consts, models = T.build(c, "multi", 0)
    # non-zero biases and a non-trivial start (zero biases would hide bias bugs); identical for both schedules
    ts["critic"] = jax.tree_util.tree_map(lambda x: x + 0.05 * jax.random.normal(jax.random.PRNGKey(4), x.shape),
                                          ts["critic"])
    pre = jax.tree_util.tree_map(np.asarray, ts["critic"])
    # the batch exactly as src/train.py:123-142 draws it at it = 0 (official: G = 16 groups of k = 1)
    G, k = T.batch_shape(c, "multi")
    keys = D.KeyTree(*consts["keys"])
    k_ro, k_sc = jax.random.fold_in(keys.rollout, 0), jax.random.fold_in(keys.sched, 0)
    if sched == "official":
        idx = jax.random.choice(k_sc, N, (G,), replace=True)
    else:
        idx = jax.random.permutation(k_sc, N)[:G]
    env_inputs = jax.tree_util.tree_map(lambda x: jnp.repeat(x[idx], k, axis=0), consts["fixed"])
    policy = lambda o: models["actor"].apply(ts["actor"], o)
    traj = jax.vmap(lambda inp, kk: lb_rollout(models["spec"], policy, inp, kk))(env_inputs, jax.random.split(k_ro, G * k))
    ret = alg.discounted_returns(traj.rewards * float(c["reward_scale"]), float(c["gamma"]))
    step = jax.jit(T.make_train_step(c, "multi", models, pg_var=False))
    new, m = step(ts, consts)
    b = jax.vmap(models["critic"].apply)(jax.tree_util.tree_map(lambda p: p[jnp.repeat(idx, k)], ts["critic"]), traj.cobs)
    recon_mse = float(jnp.mean((b - ret) ** 2))
    return dict(pre=pre, post=jax.tree_util.tree_map(np.asarray, new["critic"]), idx=np.asarray(jnp.repeat(idx, k)),
                x=np.asarray(traj.cobs), ret=np.asarray(ret), step_mse=float(m["baseline_mse"]), recon_mse=recon_mse,
                train_return=float(m["train_return"]), recon_return=float(jnp.mean(jnp.sum(traj.rewards, 1))) * 1e-4)


if mode == "make":
    o = one_iteration("official")
    g = one_iteration("groups")
    assert np.array_equal(np.asarray(o["pre"]["params"]["MLP_0"]["Dense_0"]["kernel"]),
                          np.asarray(g["pre"]["params"]["MLP_0"]["Dense_0"]["kernel"]))
    out = {"idx": o["idx"], "x": o["x"], "ret": o["ret"], "g_idx": g["idx"], "g_x": g["x"], "g_ret": g["ret"]}
    for n, arrs in enumerate(flat_stack(o["pre"])):
        for j, a in enumerate(arrs):
            out[f"pre_{n}_{j}"] = a
    for tag, r in (("off", o), ("grp", g)):
        for n, arrs in enumerate(flat_stack(r["post"])):
            for j, a in enumerate(arrs):
                out[f"ours_{tag}_post_{n}_{j}"] = a
    np.savez(path, **out)
    print(json.dumps({"reconstruction check (official)": [o["step_mse"], o["recon_mse"], o["train_return"], o["recon_return"]],
                      "reconstruction check (groups)": [g["step_mse"], g["recon_mse"], g["train_return"], g["recon_return"]],
                      "official idx": o["idx"].tolist(), "groups idx": g["idx"].tolist()}))
else:
    b = np.load(path)
    off = np.load(path.replace(".npz", "_official.npz"))
    rep = {"idx": b["idx"].tolist(), "T": int(b["x"].shape[1])}
    # (1) per-agent loss and gradient at the pre-update params: authors' sum over T = ours (mean) x T
    from src.model import Critic
    cm = Critic((200, 128), "leaky_relu")

    def params_of(n, prefix="pre"):
        L = [jnp.asarray(b[f"{prefix}_{n}_{j}"]) for j in range(6)]
        return {"params": {"MLP_0": {f"Dense_{i}": {"kernel": L[2 * i], "bias": L[2 * i + 1]} for i in range(3)}}}

    loss_rel, grad_rel = [], []
    for i, n in enumerate(b["idx"]):
        x, r = jnp.asarray(b["x"][i]), jnp.asarray(b["ret"][i])
        l, g = jax.value_and_grad(lambda p: jnp.mean((cm.apply(p, x) - r) ** 2))(params_of(int(n)))
        Tn = x.shape[0]
        loss_rel.append(abs(float(l) * Tn - float(off[f"loss_{i}"])) / abs(float(off[f"loss_{i}"])))
        gl = [g["params"]["MLP_0"][f"Dense_{k}"][w] for k in range(3) for w in ("kernel", "bias")]
        grad_rel.append(max(float(np.linalg.norm(np.asarray(a, np.float64) * Tn - off[f"grad_{i}_{j}"]) /
                                  max(np.linalg.norm(off[f"grad_{i}_{j}"]), 1e-30)) for j, a in enumerate(gl)))
    rep["per-agent loss rel err (ours mean x T vs authors' sum), max"] = max(loss_rel)
    rep["per-agent gradient rel err (ours x T vs authors'), max over agents and tensors"] = max(grad_rel)
    # (2) post-update parameters of all 10 nets: ours (mv_schedule official, the real jitted step) vs the authors'
    upd_rel, untouched = [], []
    for n in range(N):
        for j in range(6):
            pre = b[f"pre_{n}_{j}"].astype(np.float64)
            d_ours = b[f"ours_off_post_{n}_{j}"] - pre
            d_off = off[f"post_{n}_{j}"] - pre
            if np.linalg.norm(d_off) == 0:
                untouched.append(float(np.abs(d_ours).max()))
            else:
                upd_rel.append(float(np.linalg.norm(d_ours - d_off) / np.linalg.norm(d_off)))
    rep["update (post - pre) rel err, ours official vs authors', max over nets and tensors"] = max(upd_rel)
    rep["update (post - pre) rel err, median"] = float(np.median(upd_rel))
    rep["nets not drawn: max |our update|"] = max(untouched) if untouched else None
    rep["n nets drawn"] = len(set(b["idx"].tolist()))
    rep["steps per drawn net"] = {int(n): int((b["idx"] == n).sum()) for n in sorted(set(b["idx"].tolist()))}
    # (3) the same quantity for our DEFAULT schedule (groups) on its own batch: how far its update is from the
    # authors' rule applied to the same 16 rollouts is not defined (different draw); report its step count instead
    rep["groups: nets updated, rollouts per net, Adam steps per updated net"] = [len(set(b["g_idx"].tolist())), 4, 1]
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(path.replace(".npz", "_diff.json"), "w"), indent=1)

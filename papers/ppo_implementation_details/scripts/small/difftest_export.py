"""small-scale, our side of the differential tests vs CleanRL @fe8d8a0 (identical inputs, copied weights).

Writes runs/small/difftest/ours_<case>.npz: inputs, our outputs and our weights in torch state_dict layout.
The oracle (torch, CleanRL code) is scripts/official/oracle_difftest.py;
the comparison is scripts/small/difftest_compare.py.
Run (CPU is fine): uv run python -m scripts.small.difftest_export
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src.train import CONFIG_DIR, PPO, load_config  # noqa: E402  (bootstraps the GPU env + matmul precision)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from src import algorithm as alg  # noqa: E402
from src import model as mdl  # noqa: E402

OUT = os.path.join(REPO, "runs", "small", "difftest")


def torch_sd(params, arch: str, cont: bool) -> dict:
    p = jax.tree.map(np.asarray, params)["params"]
    sd = {}

    def lin(prefix, d):
        sd[prefix + ".weight"] = d["kernel"].T
        sd[prefix + ".bias"] = d["bias"]

    if arch == "separate":
        a = "actor_mean" if cont else "actor"
        lin(f"{a}.0", p["actor"]["l0"]); lin(f"{a}.2", p["actor"]["l1"]); lin(f"{a}.4", p["actor_head"])
        lin("critic.0", p["critic"]["l0"]); lin("critic.2", p["critic"]["l1"]); lin("critic.4", p["critic_head"])
        if cont:
            sd["actor_logstd"] = p["log_std"][None]
    elif arch == "shared":
        lin("network.0", p["torso"]["l0"]); lin("network.2", p["torso"]["l1"])
        lin("actor", p["actor_head"]); lin("critic", p["critic_head"])
    elif arch == "lstm":
        lin("network.0", p["encoder"])
        c = p["lstm"]["cell"]
        sd["lstm.weight_ih_l0"], sd["lstm.weight_hh_l0"] = c["weight_ih"], c["weight_hh"]
        sd["lstm.bias_ih_l0"], sd["lstm.bias_hh_l0"] = c["bias_ih"], c["bias_hh"]
        lin("actor", p["actor_head"]); lin("critic", p["critic_head"])
    return {f"sd/{k}": np.asarray(v, np.float32) for k, v in sd.items()}


def grads_flat(g, arch, cont):
    return {k.replace("sd/", "grad/"): v for k, v in torch_sd(g, arch, cont).items()}


def params_flat(p, arch, cont, tag):
    return {k.replace("sd/", f"{tag}/"): v for k, v in torch_sd(p, arch, cont).items()}


def perturb(params, key, scale=0.3):
    leaves, tdef = jax.tree.flatten(params)
    ks = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(tdef, [x + scale * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, ks)])


def ff_case(name, config, arm_over, seed):
    cfg = load_config(os.path.join(CONFIG_DIR, config))
    cfg.update(arm_over)
    cfg.update(num_envs=4, num_steps=128, num_minibatches=4)  # the classic-control batch shape for every FF case
    ppo = PPO(cfg)
    cont, arch = ppo.cont, cfg["architecture"]
    key = jax.random.PRNGKey(seed)
    ks = jax.random.split(key, 12)
    runner, _ = ppo.init(jnp.int32(seed))
    params = perturb(runner.params, ks[0])      # away from init so every term is non-trivial
    d, A, M, N = ppo.spec.obs_dim, ppo.spec.n_out, 128, 4
    out = {"meta/arch": np.array(arch), "meta/cont": np.array(cont), "meta/n_out": np.array(A),
           "meta/obs_dim": np.array(d)}
    out.update(torch_sd(params, arch, cont))
    # (1) forward on 256 random observations
    obs256 = jax.random.normal(ks[1], (256, d)) * 2.0
    dist, v = ppo.net.apply(params, obs256)
    act256 = (jax.random.normal(ks[2], (256, A)) * 1.5 if cont else jax.random.randint(ks[2], (256,), 0, A))
    out.update({"fwd/obs": obs256, "fwd/act": act256, "fwd/value": v, "fwd/logp": mdl.log_prob(dist, act256, cont),
                "fwd/entropy": mdl.entropy(dist, cont)})
    out["fwd/out0"] = dist[0] if cont else dist
    # (2) GAE on a rollout batch with mid-segment dones and next_done = 1 for two envs
    rewards = jnp.round(jax.random.normal(ks[3], (M, N)) * 2.0, 2)
    values = jax.random.normal(ks[4], (M, N))
    dones = (jax.random.uniform(ks[5], (M, N)) < 0.05).astype(jnp.float32).at[0].set(0.0)
    next_obs = jax.random.normal(ks[6], (N, d))
    next_done = jnp.array([0.0, 1.0, 0.0, 1.0])
    next_value = ppo.net.apply(params, next_obs)[1]
    adv, ret = alg.gae(rewards, values, dones, next_value, next_done, 0.99, 0.95)
    out.update({"gae/rewards": rewards, "gae/values": values, "gae/dones": dones, "gae/next_obs": next_obs,
                "gae/next_done": next_done, "gae/adv": adv, "gae/ret": ret})
    # (3) two consecutive minibatch SGD steps on the flattened batch
    b_obs = jax.random.normal(ks[7], (M * N, d)) * 2.0
    b_act = (jax.random.normal(ks[8], (M * N, A)) * 1.5 if cont else jax.random.randint(ks[8], (M * N,), 0, A))
    d0, v0 = ppo.net.apply(params, b_obs)
    b_logp = mdl.log_prob(d0, b_act, cont) + 0.3 * jax.random.normal(ks[9], (M * N,))  # ratios away from 1
    b_values = v0 + 0.5 * jax.random.normal(ks[10], (M * N,))
    perm = jax.random.permutation(ks[11], M * N)
    lr = 2.5e-4 * (1.0 - 3 / 10)  # an annealed value (update 4 of 10)
    opt_state = alg.set_lr(ppo.tx.init(params), lr)

    class MB(dict):
        __getattr__ = dict.__getitem__

    out.update({"b/obs": b_obs, "b/actions": b_act, "b/logprobs": b_logp, "b/values": b_values,
                "b/advantages": adv.reshape(-1), "b/returns": ret.reshape(-1), "b/perm": perm, "b/lr": np.float32(lr),
                "meta/clip_coef": cfg["clip_coef"], "meta/ent_coef": cfg["ent_coef"], "meta/vf_coef": cfg["vf_coef"],
                "meta/clip_vloss": cfg["clip_vloss"]})
    p = params
    for step in range(2):
        ids = perm[step * 128:(step + 1) * 128]
        mb = MB(obs=b_obs[ids], action=b_act[ids], logp=b_logp[ids], value=b_values[ids], ret=ret.reshape(-1)[ids],
                adv=adv.reshape(-1)[ids], done=None)
        (loss, aux), g = jax.value_and_grad(ppo._loss, has_aux=True)(p, mb)
        gc, _ = optax.clip_by_global_norm(cfg["max_grad_norm"]).update(g, None)
        p, opt_state, aux2 = ppo.sgd_step(p, opt_state, mb)
        pre = f"step{step}/"
        out.update({pre + k: np.asarray(v) for k, v in aux.items()})
        out[pre + "loss"] = loss
        out.update({pre + k: v for k, v in grads_flat(g, arch, cont).items()})
        out.update({pre + k.replace("grad/", "gradclip/"): v for k, v in grads_flat(gc, arch, cont).items()})
        out.update(params_flat(p, arch, cont, pre + "params_after"))
    # (4) lr schedule over 10 updates
    out["lr/ours"] = np.array([float(alg.lr_at(jnp.int32(u), 10, 2.5e-4, True)) for u in range(10)])
    np.savez(os.path.join(OUT, f"ours_{name}.npz"), **{k: np.asarray(v) for k, v in out.items()})
    print(f"[INFO] wrote ours_{name}.npz keys={len(out)}")


def lstm_case(seed=5):
    cfg = load_config(os.path.join(CONFIG_DIR, "lstm_masked_cartpole.yaml"))
    cfg.update(num_envs=4, num_steps=16, num_minibatches=2)
    ppo = PPO(cfg)
    ks = jax.random.split(jax.random.PRNGKey(seed), 10)
    runner, _ = ppo.init(jnp.int32(seed))
    params = perturb(runner.params, ks[0], 0.2)
    M, N, d, H = 16, 4, 4, cfg["lstm_hidden"]
    out = torch_sd(params, "lstm", False)
    obs = jax.random.normal(ks[1], (M, N, d))
    dones = jnp.zeros((M, N)).at[5, 1].set(1.0).at[9, 3].set(1.0).at[0, 2].set(1.0).at[12, 1].set(1.0)
    h0 = (jax.random.normal(ks[2], (N, H)) * 0.5, jax.random.normal(ks[3], (N, H)) * 0.5)
    logits, v, (hT, cT) = ppo.net.apply(params, obs, dones, h0)
    act = jax.random.randint(ks[4], (M, N), 0, 2)
    lp = mdl.log_prob(logits, act, False)
    b_logp = lp + 0.3 * jax.random.normal(ks[5], (M, N))
    b_values = v + 0.5 * jax.random.normal(ks[6], (M, N))
    adv = jax.random.normal(ks[7], (M, N))
    ret = b_values + adv
    out.update({"seq/obs": obs, "seq/dones": dones, "seq/h0": h0[0], "seq/c0": h0[1], "seq/logits": logits,
                "seq/value": v, "seq/hT": hT, "seq/cT": cT})
    mbenvinds = jnp.array([3, 1])

    class MB(dict):
        __getattr__ = dict.__getitem__

    mb = MB(obs=obs[:, mbenvinds], action=act[:, mbenvinds], logp=b_logp[:, mbenvinds], value=b_values[:, mbenvinds],
            ret=ret[:, mbenvinds], adv=adv[:, mbenvinds], done=dones[:, mbenvinds])
    ic = (h0[0][mbenvinds], h0[1][mbenvinds])
    (loss, aux), g = jax.value_and_grad(ppo._loss, has_aux=True)(params, mb, ic)
    lr = 2.5e-4
    p2, _, _ = ppo.sgd_step(params, alg.set_lr(ppo.tx.init(params), lr), mb, ic)
    out.update({"b/actions": act, "b/logprobs": b_logp, "b/values": b_values, "b/advantages": adv, "b/returns": ret,
                "b/mbenvinds": mbenvinds, "b/lr": np.float32(lr), "step0/loss": loss})
    out.update({"step0/" + k: np.asarray(v) for k, v in aux.items()})
    out.update({"step0/" + k: v for k, v in grads_flat(g, "lstm", False).items()})
    out.update(params_flat(p2, "lstm", False, "step0/params_after"))
    np.savez(os.path.join(OUT, "ours_lstm.npz"), **{k: np.asarray(v) for k, v in out.items()})
    print(f"[INFO] wrote ours_lstm.npz keys={len(out)}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ff_case("separate", "classic_control.yaml", {"architecture": "separate"}, 1)
    ff_case("shared", "classic_control.yaml", {"architecture": "shared"}, 2)
    ff_case("separate_novclip", "classic_control.yaml", {"architecture": "separate", "clip_vloss": False}, 3)
    ff_case("gaussian", "pendulum.yaml", {"ent_coef": 0.01}, 4)  # ent_coef > 0 so the log-std entropy path is tested
    lstm_case()

"""Real-data multi-iteration parity: the authors' modules (study-era distribution, REC and
REC + value-norm None, full widths 64/256, 256 envs x 8 steps, 10 epochs x 32 minibatches) ran on gymnasium
HalfCheetah-v4 and dumped every batch, permutation and pre/post state (runs/recheck/official-oracle/
official_dump.py). Our ``algorithm.update`` replays the same batches and permutations from the official pre-update
state; the fixture keeps iterations 0-1 (REC) and 0 (value-norm None) of those dumps (make_slim_replay.py).
Measured in the audit: weights <= 6.1e-7, stats <= 4e-6 relative, against official per-iteration steps of 3-15 %."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src import algorithm as A
from src import data as D
from src import model as M

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
SEP_PATHS = [("std",), ("pi", "h0", "kernel"), ("pi", "h0", "bias"), ("pi", "h1", "kernel"), ("pi", "h1", "bias"),
             ("pi", "out", "kernel"), ("pi", "out", "bias"), ("v", "out", "kernel"), ("v", "out", "bias"),
             ("v", "h0", "kernel"), ("v", "h0", "bias"), ("v", "h1", "kernel"), ("v", "h1", "bias")]


def _get(t, p):
    for k in p:
        t = t[k]
    return t


def _set(t, p, v):
    if len(p) == 1:
        t[p[0]] = v
    else:
        _set(t[p[0]], p[1:], v)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))


def ms_from(mean, std, count):
    c = jnp.full(mean.shape, float(count), jnp.float32)
    return D.MeanStd(c, jnp.asarray(mean, jnp.float32), jnp.asarray(std, jnp.float32) ** 2 * c)


@pytest.mark.parametrize("name,vnorm", [("rec_pre", True), ("rec_pre_vnone", False)])
def test_replay_parity_real_data(name, vnorm):
    jax.config.update("jax_default_matmul_precision", "highest")
    z = np.load(os.path.join(FIX, f"replay_slim_{name}.npz"))
    K = max(int(k.split("/")[0][2:]) for k in z.files if k.startswith("it")) + 1
    T1, N, d = z["it0/obs"].shape
    T, Adim, wv = T1 - 1, z["it0/act"].shape[-1], z["w0_12"].shape[0]
    cfg = A.LaneCfg(obs_norm=jnp.asarray(True), obs_clip=jnp.float32(10.0), tanh=jnp.asarray(True),
                    softplus=jnp.asarray(True), init_std=jnp.float32(0.5), min_std=jnp.float32(1e-3),
                    ppo_eps=jnp.float32(0.25), vclip_eps=jnp.float32(0.0), gae_lambda=jnp.float32(0.9),
                    gamma=jnp.float32(0.99), lr=jnp.float32(3e-4), lr_horizon_steps=jnp.float32(977 * 10 * 32),
                    lr_end=jnp.float32(0.0), adam_b1=jnp.float32(0.9), adam_b2=jnp.float32(0.999),
                    adam_eps=jnp.float32(1e-7), grad_clip=jnp.float32(0.5), ent_coef=jnp.float32(0.0),
                    value_norm=jnp.asarray(vnorm), adv_norm=jnp.asarray(False), recompute=jnp.asarray(True),
                    shared=jnp.asarray(False), baseline_cost=jnp.float32(1.0), reward_norm=jnp.asarray(False),
                    logp_presquash=jnp.asarray(True))
    st = A.Static(num_envs=N, unroll=T, epochs=10, minibatch=64)
    shift = M.std_shift(cfg.init_std, cfg.min_std, cfg.softplus)

    def params_from(prefix):
        p = M.init_params(jax.random.PRNGKey(0), d, Adim, pi_width=64, v_width=wv)
        for i, pa in enumerate(SEP_PATHS):
            _set(p, pa, jnp.asarray(z[f"{prefix}{i}"]))
        return p

    upd = jax.jit(lambda ls, b, perms: A.update(ls, b, cfg, st, jax.random.PRNGKey(0), perms=perms))
    opt = None
    for it in range(K):
        lb = jnp.asarray(z[f"it{it}/logits"])
        b = A.Batch(obs=jnp.asarray(z[f"it{it}/obs"]), act=jnp.asarray(z[f"it{it}/act"])[:-1], mu_b=lb[:-1, :, :Adim],
                    std_b=M.std_of(lb[:-1, :, Adim:], shift, cfg.min_std, cfg.softplus),
                    v_b=jnp.asarray(z[f"it{it}/base"])[:-1], rew=jnp.asarray(z[f"it{it}/rew"])[1:],
                    done=jnp.asarray(z[f"it{it}/done"])[1:])
        ppre = params_from(f"it{it}/wpre_")
        ms_pre = ms_from(z[f"it{it}/obs_mean_pre"], z[f"it{it}/obs_std_pre"], it * T * N)
        # actor path: our forward on the official pre-update state reproduces the stored behaviour outputs
        o = D.normalize_obs(ms_pre, jnp.asarray(z[f"it{it}/obs"]), cfg.obs_norm, cfg.obs_clip)
        mu, x_rho, v = M.forward(ppre, o, cfg.shared)
        assert rel(mu, z[f"it{it}/logits"][..., :Adim]) < 1e-5
        assert rel(x_rho, z[f"it{it}/logits"][..., Adim:]) < 1e-5
        assert rel(v, z[f"it{it}/base"]) < 1e-5
        pa = D.meanstd_init(1)
        if it > 0 and vnorm:
            pa = ms_from(z[f"it{it - 1}/pa_mean"], z[f"it{it - 1}/pa_std"], it * 10 * T * N)
        ls = A.LaneState(params=ppre, opt=opt if opt is not None else A.adam_init(ppre), obs_ms=ms_pre, val_ms=pa,
                         ret_ms=D.meanstd_init(1), ret_acc=jnp.zeros((N,)), env=None, obs=None,
                         it=jnp.zeros((), jnp.int32), bad=jnp.zeros((), bool))
        ls2, _ = upd(ls, b, jnp.asarray(z[f"it{it}/perms"]))
        opt = ls2.opt
        assert int(ls2.opt.t) == int(z[f"it{it}/opt_iter"])
        step = max(rel(z[f"it{it}/wpost_{i}"], z[f"it{it}/wpre_{i}"]) for i in range(len(SEP_PATHS)))
        gap = max(rel(_get(ls2.params, p), z[f"it{it}/wpost_{i}"]) for i, p in enumerate(SEP_PATHS))
        assert step > 1e-2 and gap < 1e-5, (it, gap, step)
        m, s = D.meanstd_get(ls2.obs_ms)
        assert rel(m, z[f"it{it}/obs_mean"]) < 2e-5 and rel(s, z[f"it{it}/obs_std"]) < 2e-5
        if vnorm:
            m, s = D.meanstd_get(ls2.val_ms)
            assert rel(m, z[f"it{it}/pa_mean"]) < 2e-5 and rel(s, z[f"it{it}/pa_std"]) < 2e-5

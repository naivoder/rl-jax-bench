"""Differential test against the official PyTorch code .

Fixtures: tests/fixtures/nets_<algo>_<cell>_<arch>.npz, produced by
scripts/official/dump_fixtures.py from the UNMODIFIED official modules
(repo @ e7c19c3, torch 1.13.1 CPU) with injected noise.  We load the official initial weights into our parameter
trees, feed the same batches and noise, and compare: update-0 critic/actor losses and every parameter gradient
(official loss functions + autograd), three full official update steps (losses, alpha, and ALL parameters after
each step: Adam, Polyak, update order, shared joint loss), and the official grad-norm metric.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from src import common  # noqa: F401
from src.algorithm import Agent
from src.model import Nets
from tests import oracle_map as M

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
CASES = [(a, c, r) for a in ("sac", "td3") for c in ("lstm", "gru") for r in ("separate", "shared")]


def _setup(algo, cell, arch):
    z = np.load(os.path.join(FIX, f"nets_{algo}_{cell}_{arch}.npz"))
    cfg = dict(M.ORACLE_SIZES, algo=algo, cell=cell, arch=arch)
    nets = Nets(cfg, obs_dim=1, act_dim=1, discrete=False)
    ag = Agent(cfg, nets)
    sd0 = M.state_dict(z, "init/")
    params = M.separate_params(sd0, algo) if arch == "separate" else M.shared_params(sd0, algo)
    if arch == "separate":
        opt = {"actor": ag.opt.init(params["actor"]), "critic": ag.opt.init(params["critic"])}
    else:
        opt = {"main": ag.opt.init(params["main"])}
    la = jnp.zeros((), jnp.float32)
    learner = {"params": params, "opt": opt, "log_alpha": la, "alpha_opt": ag.alpha_opt.init(la)}
    return z, ag, learner


def _close(a, b, rtol=2e-4, atol=1e-6, what=""):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    err = np.max(np.abs(a - b) - (atol + rtol * np.abs(b))) if a.size else -1
    assert err <= 0, f"{what}: max|diff|={np.max(np.abs(a - b)):.3g} (scale {np.max(np.abs(b)):.3g})"


@pytest.mark.parametrize("algo,cell,arch", [c for c in CASES if c[2] == "separate"])
def test_separate_losses_and_grads(algo, cell, arch):
    z, ag, learner = _setup(algo, cell, arch)
    P = learner["params"]
    b, noise = M.batch_from(z, 0)
    actor_x = P["actor_t"] if algo == "td3" else P["actor"]
    (lq, _), gq = jax.value_and_grad(ag.sep_critic_loss, has_aux=True)(P["critic"], actor_x, P["critic_t"], b, noise,
                                                                        jnp.float32(1.0))
    _close(lq, z["g0/critic_loss"], what="critic loss")
    (lpi, _), ga = jax.value_and_grad(ag.sep_actor_loss, has_aux=True)(P["actor"], P["critic"], b, noise,
                                                                       jnp.float32(1.0))
    _close(lpi, z["g0/actor_loss"], what="actor loss")
    flat = M.flatten_separate({"actor": ga, "critic": gq, "actor_t": ga, "critic_t": gq})
    n = 0
    for k in z.files:
        if k.startswith("g0/critic/") or k.startswith("g0/actor/"):
            mod, name = k.split("/")[1], k.split("/", 2)[2]
            _close(flat[f"{mod}.{name}"], z[k], rtol=1e-3, atol=1e-6, what=k)
            n += 1
    assert n > 30


@pytest.mark.parametrize("algo,cell,arch", CASES)
def test_three_official_updates(algo, cell, arch):
    z, ag, learner = _setup(algo, cell, arch)
    coef = {"leak": jnp.float32(1.0), "enc": jnp.float32(1.0)}
    flatten = M.flatten_separate if arch == "separate" else M.flatten_shared
    for u in range(3):
        b, noise = M.batch_from(z, u)
        learner, inf = ag.update(learner, b, noise, coef, diag=(u == 0))
        for k in ("qf1_loss", "qf2_loss", "policy_loss"):
            _close(inf[k], z[f"u{u}/info/{k}"], rtol=1e-3, atol=1e-6, what=f"u{u} {k}")
        if algo == "sac":
            _close(-inf["mean_logp"], z[f"u{u}/info/policy_entropy"], rtol=1e-3, atol=1e-6, what="entropy")
            _close(jnp.exp(learner["log_alpha"]), z[f"u{u}/info/alpha"], rtol=1e-6, what="alpha after step")
        flat = flatten(learner["params"])
        ref = M.state_dict(z, f"u{u}/params/")
        assert set(ref) == set(flat), set(ref) ^ set(flat)
        for k in ref:
            # Adam steps are lr-sized (3e-4); a wrong gradient sign or update order shows up as ~1e-4 errors
            _close(flat[k], ref[k], rtol=1e-4, atol=3e-6, what=f"u{u} param {k}")
        if u == 0:
            g = M.state_dict(z, "u0/gradnorm/")
            pairs = ({"q_grad_norm": "off_q", "q_rnn_grad_norm": "off_q_rnn", "pi_grad_norm": "off_pi",
                      "pi_rnn_grad_norm": "off_pi_rnn"} if arch == "separate" else
                     {"rnn_grad_norm": "off_rnn", "q_grad_norm": "off_q", "pi_grad_norm": "off_pi"})
            for tk, ok in pairs.items():
                _close(inf[ok], g[tk], rtol=2e-3, atol=1e-7, what=f"grad-norm {tk}")


def test_shared_variant_coefficients_change_only_actor_routing():
    """noleak / detach: identical forward values and critic-loss gradients; actor-loss gradients removed exactly where
    declared (Q heads + obs-action shortcut for noleak; additionally the encoder for detach)."""
    z, ag, learner = _setup("td3", "lstm", "shared")
    p, tgt = learner["params"]["main"], learner["params"]["tgt"]
    b, noise = M.batch_from(z, 0)
    a1 = jnp.float32(1.0)

    def grads(leak, enc):
        coef = {"leak": jnp.float32(leak), "enc": jnp.float32(enc)}
        f = lambda q: ag.shared_losses(q, tgt, b, noise, a1, coef)[1]  # noqa: E731
        return jax.grad(f)(p), ag.shared_losses(p, tgt, b, noise, a1, coef)
    g_off, l_off = grads(1, 1)
    g_nl, l_nl = grads(0, 1)
    g_dt, l_dt = grads(0, 0)
    for la, lb in ((l_off, l_nl), (l_off, l_dt)):
        assert float(la[0]) == float(lb[0]) and float(la[1]) == float(lb[1])
    zero = lambda t: all(float(jnp.max(jnp.abs(x))) == 0.0 for x in jax.tree_util.tree_leaves(t))  # noqa: E731
    qh = lambda g: {"q1": g["q1"], "q2": g["q2"]}  # noqa: E731  (min(Q1,Q2) may route through one head only)
    assert not zero(qh(g_off)) and not zero(g_off["short_oa"])
    assert zero(qh(g_nl)) and zero(g_nl["short_oa"]) and zero(qh(g_dt)) and zero(g_dt["short_oa"])
    assert not zero(g_nl["enc"]) and zero(g_dt["enc"])
    for g in (g_off, g_nl, g_dt):
        assert not zero(g["pi"]) and not zero(g["short_o"])
    _close(optax.tree.norm(g_nl["pi"]), optax.tree.norm(g_off["pi"]), what="policy grads unchanged")

"""Backprop (BP), L2, Shrink-and-Perturb (S&P), Adam and Continual Backpropagation (CBP) as ONE pure step.

Every method is the same jitted function; the arms differ only in hyper-parameter values (vmappable scalars):

    BP   : wd = 0, sigma = 0, rho = 0
    L2   : wd = lambda                                    (grad += lambda * theta on ALL params, torch SGD weight_decay)
    S&P  : wd = lambda, sigma = noise std                 (N(0, sigma^2) added to every weight+bias after the step)
    CBP  : rho > 0 (+ util_id, mt, eta, bias_transfer)    (generate-and-test after the step; Algorithm 1 / 3)

Derivation: the derivation summarized in the README.  The reference
implementation used as oracle is lop/algos/gnt.py, lop/utils/AdamGnT.py, lop/algos/bp.py of
github.com/shibhansh/loss-of-plasticity @ a6b7958 -- re-implemented here, never copied.

Order inside one step (official code ContinualBackprop.learn / Backprop.learn):
  1. forward on x (pre-update) -> loss, output, hidden activations h_l
  2. gradient step (SGD or per-element Adam) incl. L2 term
  3. S&P perturbation (if sigma > 0)
  4. CBP: for EVERY hidden layer first age+1 / utility update / selection (using the pre-update h_l and the
     post-update weights), THEN replacements layer by layer in increasing order (re-init incoming, zero own bias,
     [optional bias transfer], zero outgoing, reset Adam state of all touched weights + the unit's bias).

PRNG: the caller passes one key per step (fold_in(k_algo, step)); it is split here into the S&P noise key and one
CBP key per layer (selection coin, random utility, re-init draws).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp

from .model import make_model

# Utility ids (paper App. D names -> official util_type names)
UTIL_NAMES = ("overall",                       # 'adaptable_contribution'  Eqs. 6-8
              "contribution",                  # 'contribution'            Eq. 2
              "mean_corrected_contribution",   # 'zero_contribution'       Eq. 5
              "weight_magnitude",              # 'weight'                  (outgoing weight magnitude)
              "adaptation",                    # 'adaptation'              1 / incoming weight magnitude
              "random")                        # 'random'
UTIL_ID = {n: i for i, n in enumerate(UTIL_NAMES)}


@dataclass(frozen=True)
class Static:
    """Compile-time configuration (hashable; one jit per Static)."""
    n_in: int
    widths: tuple
    n_out: int
    loss: str = "mse"                 # 'mse' -> (yhat - y)^2 (F.mse_loss) | 'ce' -> softmax cross-entropy
    opt: str = "sgd"                  # 'sgd' | 'adam' (AdamGnT: per-element step counter)
    replace_mode: str = "accumulate"  # 'accumulate' (MNIST) | 'bernoulli' (SCR expr.py default)
    k_max: int = 1                    # static upper bound on replacements per layer per step
    beta1: float = 0.9                # paper §5 'default values'; official expr.py
    beta2: float = 0.999
    eps: float = 1e-8                 # AdamGnT default
    weight_reduction: str = "mean"    # 'mean' (official code) | 'sum' (paper Eq. 6) -- ranking-identical
    convention: str = "C"             # bias-correction convention: 'C' (official) | 'P' (paper literal) | 'E' (exact)
    track_wmag: bool = True           # log the per-step average |w| (weights only)
    perturb: bool = True              # draw S&P noise; False is exact when every arm has sigma = 0 (p + 0*N == p)
    count_base: str = "eligible"      # replacements per step = rho * n_eligible ('eligible', official gnt.py) |
                                      # rho * n_l ('layer', paper Alg. 1 'n_l * rho of eligible units'), capped at
                                      # n_eligible


class HP(NamedTuple):
    """Per-run hyper-parameters (each a scalar; vmapped as arrays)."""
    lr: jnp.ndarray
    wd: jnp.ndarray            # L2 / S&P shrink
    sigma: jnp.ndarray         # S&P noise std
    rho: jnp.ndarray           # CBP replacement rate (0 = no replacement)
    util_id: jnp.ndarray       # int
    mt: jnp.ndarray            # maturity threshold
    eta: jnp.ndarray           # utility decay rate
    bias_transfer: jnp.ndarray  # 0/1 -- paper §6 text (official code: no-op)
    act: jnp.ndarray           # activation id
    reinit_gain: jnp.ndarray   # gain of the re-init bound U(-g sqrt(3/fan_in), +)
    adam_rule: jnp.ndarray     # int: 0 = 'standard' AdamGnT (official a6b7958, post-f00ca40; paper Alg. 3 text)
                               #      1 = 'paper_era' AdamGnT (official dcc3619 = the code at the paper's date)


ADAM_RULES = ("standard", "paper_era")


def make_hp(lr, wd=0.0, sigma=0.0, rho=0.0, util="overall", mt=100, eta=0.99, bias_transfer=0.0, act=0,
            reinit_gain=2 ** 0.5, adam_rule="standard") -> HP:
    uid = UTIL_ID[util] if isinstance(util, str) else util
    rid = ADAM_RULES.index(adam_rule) if isinstance(adam_rule, str) else adam_rule
    f = lambda v: jnp.asarray(v, jnp.float32)
    return HP(f(lr), f(wd), f(sigma), f(rho), jnp.asarray(uid, jnp.int32), f(mt), f(eta), f(bias_transfer),
              jnp.asarray(act, jnp.int32), f(reinit_gain), jnp.asarray(rid, jnp.int32))


def stack_hp(hps: Sequence[HP]) -> HP:
    return jax.tree.map(lambda *xs: jnp.stack(xs), *hps)


# ----------------------------------------------------------------------------------------------- state
def init_state(static: Static, params) -> dict:
    zeros = lambda: jax.tree.map(jnp.zeros_like, params)
    opt = {"m": zeros(), "v": zeros(), "t": zeros()} if static.opt == "adam" else {}
    cbp = {k: [jnp.zeros((w,), jnp.float32) for w in static.widths] for k in ("util", "bcu", "f", "age")}
    cbp["acc"] = [jnp.zeros((), jnp.float32) for _ in static.widths]
    return {"params": params, "opt": opt, "cbp": cbp}


def _names(static):
    return [f"layer_{i}" for i in range(len(static.widths) + 1)]


def loss_and_out(static: Static, params, x, y, act):
    out, hs = make_model(static.widths, static.n_out).apply(params, x, act)
    if static.loss == "mse":
        loss = jnp.mean((out - y) ** 2)                               # F.mse_loss, no 1/2
    else:
        logp = jax.nn.log_softmax(out, axis=-1)
        loss = -jnp.mean(jnp.take_along_axis(logp, y[:, None].astype(jnp.int32), axis=-1))
    return loss, (out, hs)


def mean_abs_weight(static: Static, params):
    """Average |w| over all weight matrices (biases excluded) -- the paper's weight-magnitude diagnostic."""
    ks = [params["params"][n]["kernel"] for n in _names(static)]
    return sum(jnp.sum(jnp.abs(k)) for k in ks) / sum(k.size for k in ks)


# ----------------------------------------------------------------------------------------------- optimisers
def sgd_update(params, grads, hp: HP):
    return jax.tree.map(lambda p, g: p - hp.lr * (g + hp.wd * p), params, grads)


def adam_update(static: Static, params, grads, opt, hp: HP):
    """AdamGnT: per-element step counter t (reset to 0 by a CBP replacement); denominator sqrt(v) + eps.

    hp.adam_rule = 0 ('standard'): step = lr * sqrt(1 - b2^t) / (1 - b1^t)  -- official lop/utils/AdamGnT.py after
        f00ca40 (2025-09-01, a third-party "match standard Adam" fix; a6b7958 is our oracle checkout) and the paper's
        Algorithm 3 text (bias-corrected Adam).
    hp.adam_rule = 1 ('paper_era'): step = lr * (1 - b2^t)  -- official AdamGnT.py as of dcc3619 (2023-05-20), the
        code that existed when the paper's Adam figures were produced: `bias_correction.sqrt().div_(bias_correction1)`
        writes into a discarded temporary, so the update is p -= lr * (1 - b2^t) * m / (sqrt(v) + eps).  Identical to
        the standard rule for large t; after a reset the reborn weights move by ~0.003 lr (t=1), ~0.31 lr (t=100).
    """
    b1, b2, eps = static.beta1, static.beta2, static.eps

    def upd(p, g, m, v, t):
        g = g + hp.wd * p
        t = t + 1.0
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        step = jnp.where(hp.adam_rule == 1, hp.lr * (1 - b2 ** t), hp.lr * jnp.sqrt(1 - b2 ** t) / (1 - b1 ** t))
        return p - step * m / (jnp.sqrt(v) + eps), m, v, t

    out = jax.tree.map(upd, params, grads, opt["m"], opt["v"], opt["t"])
    istup = lambda x: isinstance(x, tuple)
    pick = lambda i: jax.tree.map(lambda o: o[i], out, is_leaf=istup)
    return pick(0), {"m": pick(1), "v": pick(2), "t": pick(3)}


def perturb(params, sigma, key):
    leaves, tdef = jax.tree.flatten(params)
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(tdef, [p + sigma * jax.random.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)])


# ----------------------------------------------------------------------------------------------- CBP
def utility_step(static: Static, h, w_in, w_out, util, f, age, hp: HP, key):
    """One utility update for one hidden layer.  Returns (util, bcu, f, fhat, age).

    h: (B, n) activations of the current example(s) (pre-update forward pass); w_in (fan_in, n), w_out (n, n_next)
    are the post-update kernels.  Convention 'C' = official gnt.py update_utility; 'P' = paper Eqs. 3-8 literally;
    'E' = exact-EMA reading.
    """
    eta = hp.eta
    age = age + 1.0
    bc = 1.0 - eta ** age
    hbar = jnp.mean(h, axis=0)
    red = jnp.mean if static.weight_reduction == "mean" else jnp.sum
    out_mag = red(jnp.abs(w_out), axis=1)
    in_mag = red(jnp.abs(w_in), axis=0)
    if static.convention == "C":
        f_new = eta * f + (1 - eta) * hbar
        fhat = f_new / bc
    elif static.convention == "P":
        fhat = f / bc
        f_new = eta * f + (1 - eta) * hbar
    elif static.convention == "E":
        bc_prev = 1.0 - eta ** (age - 1.0)
        fhat = jnp.where(age > 1.0, f / jnp.where(bc_prev > 0, bc_prev, 1.0), 0.0)
        f_new = eta * f + (1 - eta) * hbar
    else:
        raise ValueError(static.convention)
    dev = jnp.mean(jnp.abs(h - fhat), axis=0)
    habs = jnp.mean(jnp.abs(h), axis=0)
    cand = jnp.stack([out_mag * dev / in_mag,       # overall
                      out_mag * habs,               # contribution
                      out_mag * dev,                # mean-corrected contribution
                      out_mag,                      # weight magnitude
                      1.0 / in_mag,                 # adaptation
                      jnp.zeros_like(out_mag)])     # random (EMA unused)
    y = cand[hp.util_id]
    u_new = eta * util + (1 - eta) * y
    if static.convention == "P":
        bcu = util / bc
    else:
        bcu = u_new / bc
    bcu = jnp.where(hp.util_id == UTIL_ID["random"], jax.random.uniform(key, bcu.shape), bcu)
    return u_new, bcu, f_new, fhat, age


def select_units(static: Static, bcu, age, acc, hp: HP, key):
    """Choose up to k_max eligible (age > mt) units with the smallest bias-corrected utility.

    Returns (idx (k_max,), active (k_max,) bool, acc, overflow flag).
    """
    elig = age > hp.mt
    n_elig = jnp.sum(elig).astype(jnp.float32)
    if static.count_base == "eligible":
        n_base = n_elig
    elif static.count_base == "layer":
        n_base = jnp.asarray(age.shape[0], jnp.float32)
    else:
        raise ValueError(static.count_base)
    num = hp.rho * n_base
    if static.replace_mode == "accumulate":
        # gnt.py accumulates rho * n_eligible in a Python (float64) number.  float32 drifts (0.2 x 5 != 1), so when
        # 1/rho is an integer Q (every rate we use) the residual is kept EXACTLY as an integer count of
        # eligible-unit-steps modulo Q; otherwise a float32 residual is used.
        Q = jnp.round(1.0 / jnp.where(hp.rho > 0, hp.rho, 1.0))
        exact = (hp.rho > 0) & (jnp.abs(Q * hp.rho - 1.0) < 1e-6)
        Qi = jnp.maximum(Q, 1.0).astype(jnp.int32)
        acc_i = jnp.round(acc).astype(jnp.int32) + n_base.astype(jnp.int32)
        k_e = (acc_i // Qi).astype(jnp.float32)
        acc_e = (acc_i % Qi).astype(jnp.float32)
        acc_f = acc + num
        k_f = jnp.floor(acc_f)
        k = jnp.where(exact, k_e, k_f)
        acc = jnp.where(exact, acc_e, acc_f - k_f)
    else:  # bernoulli: if num < 1 replace one unit w.p. num, else floor(num)  (gnt.py accumulate=False)
        u = jax.random.uniform(key)
        k = jnp.where(num < 1.0, jnp.where(u < num, 1.0, 0.0), jnp.floor(num))
    k = jnp.minimum(k, n_elig)
    overflow = k > static.k_max
    score = jnp.where(elig, bcu, jnp.inf)
    _, idx = jax.lax.top_k(-score, static.k_max)
    active = (jnp.arange(static.k_max) < k) & (hp.rho > 0)
    return idx, active, acc, overflow


def cbp_update(static: Static, params, opt, cbp, hs, hp: HP, key):
    """Generate-and-test after the gradient step (official gnt.py gen_and_test)."""
    names = _names(static)
    L = len(static.widths)
    P = {n: dict(params["params"][n]) for n in names}
    O = None
    if static.opt == "adam":
        O = {s: {n: dict(opt[s]["params"][n]) for n in names} for s in ("m", "v", "t")}
    new = {k: list(v) for k, v in cbp.items()}
    sel = []
    n_rep = []
    overflow = jnp.array(False)
    # phase 1 -- utilities and selection for every layer, with the post-update weights
    for l in range(L):
        kl = jax.random.fold_in(key, l)
        k_util, k_sel, k_new = jax.random.split(kl, 3)
        util, bcu, f, fhat, age = utility_step(static, hs[l], P[names[l]]["kernel"], P[names[l + 1]]["kernel"],
                                               cbp["util"][l], cbp["f"][l], cbp["age"][l], hp, k_util)
        idx, active, acc, ovf = select_units(static, bcu, age, cbp["acc"][l], hp, k_sel)
        overflow = overflow | ovf
        rep = jnp.zeros(age.shape, jnp.int32).at[idx].max(active.astype(jnp.int32)) > 0
        new["util"][l] = jnp.where(rep, 0.0, util)
        new["f"][l] = jnp.where(rep, 0.0, f)
        new["bcu"][l] = bcu
        new["age"][l] = age  # zeroed in phase 2 (after the bias transfer reads it, as in gnt.py)
        new["acc"][l] = acc
        sel.append((idx, active, fhat, k_new, rep))
        n_rep.append(jnp.sum(active))
    # phase 2 -- replacements, layer by layer in increasing order
    for l in range(L):
        idx, active, fhat, k_new, rep = sel[l]
        cur, nxt = names[l], names[l + 1]
        w_in, b_in = P[cur]["kernel"], P[cur]["bias"]
        w_out, b_out = P[nxt]["kernel"], P[nxt]["bias"]
        fan_in = w_in.shape[0]
        bound = hp.reinit_gain * jnp.sqrt(3.0 / fan_in)            # gnt.py compute_bounds (kaiming)
        draws = jax.random.uniform(k_new, (fan_in, static.k_max), w_in.dtype, -1.0, 1.0) * bound
        for j in range(static.k_max):
            r, a = idx[j], active[j]
            w_in = w_in.at[:, r].set(jnp.where(a, draws[:, j], w_in[:, r]))
            b_in = b_in.at[r].set(jnp.where(a, 0.0, b_in[r]))
            b_out = b_out + jnp.where(a, hp.bias_transfer * w_out[r, :] * fhat[r], 0.0)
            w_out = w_out.at[r, :].set(jnp.where(a, 0.0, w_out[r, :]))
            if O is not None:
                for s in ("m", "v", "t"):
                    O[s][cur]["kernel"] = O[s][cur]["kernel"].at[:, r].set(jnp.where(a, 0.0, O[s][cur]["kernel"][:, r]))
                    O[s][cur]["bias"] = O[s][cur]["bias"].at[r].set(jnp.where(a, 0.0, O[s][cur]["bias"][r]))
                    O[s][nxt]["kernel"] = O[s][nxt]["kernel"].at[r, :].set(jnp.where(a, 0.0, O[s][nxt]["kernel"][r, :]))
        P[cur] = {"kernel": w_in, "bias": b_in}
        P[nxt] = {"kernel": w_out, "bias": b_out}
        new["age"][l] = jnp.where(rep, 0.0, new["age"][l])
    params = {"params": P}
    if O is not None:
        opt = {s: {"params": O[s]} for s in ("m", "v", "t")}
    return params, opt, new, jnp.stack(n_rep), overflow


# ----------------------------------------------------------------------------------------------- one step
def learner_step(static: Static, state: dict, hp: HP, x, y, key):
    """One online update on (x, y) (x: (B, n_in)).  Returns (state, info)."""
    params = state["params"]
    (loss, (out, hs)), grads = jax.value_and_grad(loss_and_out, argnums=1, has_aux=True)(static, params, x, y, hp.act)
    opt = state["opt"]
    if static.opt == "sgd":
        params = sgd_update(params, grads, hp)
    else:
        params, opt = adam_update(static, params, grads, opt, hp)
    k_noise, k_cbp = jax.random.split(key)
    if static.perturb:
        params = perturb(params, hp.sigma, k_noise)  # sigma = 0 -> exact no-op
    cbp = state["cbp"]
    n_rep = jnp.zeros((len(static.widths),), jnp.int32)
    overflow = jnp.array(False)
    if len(static.widths) > 0:
        params, opt, cbp, n_rep, overflow = cbp_update(static, params, opt, cbp, hs, hp, k_cbp)
    info = {"loss": loss, "out": out, "n_rep": n_rep.astype(jnp.int32), "overflow": overflow}
    if static.loss == "ce":
        info["correct"] = jnp.mean((jnp.argmax(out, -1) == y).astype(jnp.float32))
    if static.track_wmag:
        info["wmag"] = mean_abs_weight(static, params)
    return {"params": params, "opt": opt, "cbp": cbp}, info

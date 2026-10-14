"""Pure JAX building blocks of A2D, AIL (asymmetric DAgger) and TRPO, re-implemented from the paper (Alg. 1,
Eqs. 11, 17, 27-29, App. D.1) with the official code's numerical conventions (README, "What differs").
Every function is jit/vmap friendly; randomness enters only through explicit index arrays / keys.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree


def adam(lr: float):
    """torch.optim.Adam(lr, eps=1e-5): m_hat / (sqrt(v_hat) + eps) — identical update rule to optax.adam."""
    return optax.adam(lr, eps=1e-5)


# ----------------------------------------------------------------------------------------------- returns / GAE
def gae(R, V, ND, gamma: float, lam: float):
    """Official advantage_and_return: paths end at done steps; the cut final step bootstraps V_{t+1} := V_t.
    delta_t = r_t + gamma * V_{t+1} * notdone_t - V_t ; A_t = delta_t + gamma*lam*notdone_t*A_{t+1};
    returns = discounted reward-to-go within the path (no bootstrap)."""
    nd = ND.astype(V.dtype)
    v_next = jnp.concatenate([V[1:], V[-1:]]) * nd
    delta = R + gamma * v_next - V

    def back(carry, x):
        a_n, g_n = carry
        d, r, m = x
        a = d + gamma * lam * m * a_n
        g = r + gamma * m * g_n
        return (a, g), (a, g)
    z = jnp.zeros((), V.dtype)
    _, (adv, ret) = jax.lax.scan(back, (z, z), (delta, R.astype(V.dtype), nd), reverse=True)
    return adv, ret


# ----------------------------------------------------------------------------------------------- value / Q losses
def value_loss_gae(vs, old_vs, adv, nd, mask, clip: float):
    """Official value_loss_gae: target = old_V + A (lambda-return); residual = max(v - t, v_clip - t) taken on
    the SIGNED residuals, then squared; terminal-step samples excluded; mean over the remaining samples."""
    targ = old_vs + adv
    v_clip = old_vs + jnp.clip(vs - old_vs, -clip, clip)
    r = jnp.maximum(vs - targ, v_clip - targ)
    sel = (nd.astype(bool) & mask.astype(bool)).astype(vs.dtype)
    return jnp.sum(sel * r ** 2) / jnp.maximum(jnp.sum(sel), 1.0)


def value_loss_returns(vs, ret, nd, mask):
    """Official value_loss_returns ('time' value-calc, Q-args): MSE to the MC reward-to-go, terminal excluded."""
    sel = (nd.astype(bool) & mask.astype(bool)).astype(vs.dtype)
    return jnp.sum(sel * (vs - ret) ** 2) / jnp.maximum(jnp.sum(sel), 1.0)


def split_layout(n: int, n_mb: int):
    """np.array_split(perm, n_mb) as a padded (n_mb, ceil(n/n_mb)) position table + validity mask (static)."""
    import numpy as np
    sizes = [n // n_mb + (1 if j < n % n_mb else 0) for j in range(n_mb)]
    m = max(sizes)
    pos = np.zeros((n_mb, m), np.int32)
    msk = np.zeros((n_mb, m), bool)
    s = 0
    for j, z in enumerate(sizes):
        pos[j, :z] = np.arange(s, s + z)
        msk[j, :z] = True
        s += z
    return jnp.asarray(pos), jnp.asarray(msk)


def value_update(apply, params, opt_state, opt, X, adv, ret, nd, calc: str, perms, n_mb: int, clip: float):
    """Official value_step: old_vs from the pre-update net; len(perms) epochs x n_mb array_split minibatches, Adam."""
    old_vs = jax.lax.stop_gradient(apply(params, X))
    pos, msk = split_layout(X.shape[0], n_mb)

    def mb_loss(p, ii, m):
        vs = apply(p, X[ii])
        if calc == 'gae':
            return value_loss_gae(vs, old_vs[ii], adv[ii], nd[ii], m, clip)
        return value_loss_returns(vs, ret[ii], nd[ii], m)

    def epoch(carry, perm):
        def mb(c, j):
            p, s = c
            ii = perm[pos[j]]
            loss, g = jax.value_and_grad(mb_loss)(p, ii, msk[j])
            u, s = opt.update(g, s, p)
            return (optax.apply_updates(p, u), s), loss
        carry, losses = jax.lax.scan(mb, carry, jnp.arange(n_mb))
        return carry, losses[-1]
    (params, opt_state), last = jax.lax.scan(epoch, (params, opt_state), perms)
    return params, opt_state, last[-1]


def q_update(apply, params, opt_state, opt, X, target, perms, n_mb: int):
    """Official q_step: MSE of Q(x, onehot a) to the MC reward-to-go over ALL samples (no terminal exclusion)."""
    pos, msk = split_layout(X.shape[0], n_mb)

    def mb_loss(p, ii, m):
        r = apply(p, X[ii]) - target[ii]
        return jnp.sum(m * r ** 2) / jnp.sum(m)

    def epoch(carry, perm):
        def mb(c, j):
            p, s = c
            loss, g = jax.value_and_grad(mb_loss)(p, perm[pos[j]], msk[j].astype(X.dtype))
            u, s = opt.update(g, s, p)
            return (optax.apply_updates(p, u), s), loss
        carry, losses = jax.lax.scan(mb, carry, jnp.arange(n_mb))
        return carry, losses[-1]
    (params, opt_state), last = jax.lax.scan(epoch, (params, opt_state), perms)
    return params, opt_state, last[-1]


# ----------------------------------------------------------------------------------------------- policy objectives
EPS64 = 2.220446049250313e-16   # finfo(float64).eps: torch clamp_probs eps of the official (float64) code


def action_logp(logp_all, A, clamp: bool = True):
    """Log-likelihood of the taken actions. clamp=True is the official get_loglikelihood (models.DiscPolicy):
    torch Categorical(probs=softmax(z)).log_prob = log(clamp(p, eps, 1 - eps)) with eps = 2.22e-16 (float64 finfo,
    kept fixed also in float32 runs because the official code is float64). jnp.clip has zero gradient outside
    [eps, 1 - eps], as torch.clamp does, so a taken action with p < eps carries no surrogate gradient.
    clamp=False is the exact log_softmax used before the clamp fix (pre-fix behavior)."""
    if not clamp:
        return jnp.take_along_axis(jax.nn.log_softmax(logp_all, -1), A[:, None], 1)[:, 0]
    p = jax.nn.softmax(logp_all, -1)
    pa = jnp.take_along_axis(p, A[:, None], 1)[:, 0]
    return jnp.log(jnp.clip(pa, EPS64, 1.0 - EPS64))


def importance_weight(e_lp_a, l_lp_a, beta):
    """A2D IS weight (Eq. 27): pi_theta(a|s) / (beta pi_theta(a|s) + (1 - beta) pi_psi(a|b)), + 1e-12 (official)."""
    num = jnp.exp(e_lp_a)
    den = beta * num + (1.0 - beta) * jnp.exp(l_lp_a)
    return num / (den + 1e-12)


def entropy_of(logp_all, safe: bool = False):
    """Official -sum log(p) p. safe=True (used only as the TRPO fallback): an action whose probability
    underflows to exactly 0 contributes its limit 0 instead of 0 * log 0 = NaN; the double where keeps the gradient
    finite. Identical in value wherever all p > 0."""
    p = jax.nn.softmax(logp_all, -1)
    if not safe:
        return -jnp.sum(jnp.log(p) * p, -1)
    pos = p > 0
    ps = jnp.where(pos, p, 1.0)
    return -jnp.sum(jnp.where(pos, jnp.log(ps) * ps, 0.0), -1)


def surrogate(logp_all, A, old_lp, adv_in, ent: float, reg_adv: bool, estimator: str = 'official',
              safe_entropy: bool = False, clamp: bool = False):
    """Official surrogate_reward(...).mean(): with reg_adv the entropy of the CURRENT policy is added to the
    (IS-weighted) advantage, the sum is normalized (mean / unbiased std + 1e-3), times the ratio pi/pi_old;
    without reg_adv the advantage is normalized and ent * mean(H) is added to the surrogate.

    estimator='paper' (a labeled non-official arm): the official mean subtraction is NOT a valid baseline
    when the actions come from another policy (A2D: a ~ pi_beta, adv_in = w * A). Its expected gradient adds
    m * (pi_psi(.|b) - pi_theta(.|s)) in logit space, and the un-weighted entropy term adds c * H * (pi_psi - pi_theta).
    'paper' keeps only Eq. 27: the IS-weighted advantage, scaled (not centered) by its std, no entropy term. A positive
    rescaling does not change the TRPO step (the step length is set by the KL radius), so this is Eq. 27's direction.

    clamp: taken-action log-likelihood as the official Categorical.log_prob (action_logp); entropy stays unclamped."""
    lp = action_logp(logp_all, A, clamp)
    if estimator == 'paper':
        n_adv = adv_in / (jnp.std(adv_in, ddof=1) + 1e-3)
        return jnp.mean(jnp.exp(lp - old_lp) * n_adv)
    H = entropy_of(logp_all, safe_entropy)
    adv = adv_in + ent * H if (reg_adv and ent > 0) else adv_in
    n_adv = (adv - adv.mean()) / (jnp.std(adv, ddof=1) + 1e-3)
    s = jnp.exp(lp - old_lp) * n_adv
    if (not reg_adv) and ent > 0:
        s = s + jnp.mean(ent * H)
    return jnp.mean(s)


def kl_probs(p, q):
    return jnp.sum(p * (jnp.log(p) - jnp.log(q)), -1)


def kl_logp(lp, lq):
    """KL from log-probabilities: identical in value to kl_probs(exp lp, exp lq), but its second derivative stays
    finite when some probabilities are ~1e-150 (kl_probs differentiates log(softmax(.)), whose Hessian scales as 1/q^2
    and overflows)."""
    lp_safe = jnp.where(jnp.isfinite(lp), lp, 0.0)          # p = 0 entries contribute exactly 0
    return jnp.sum(jnp.exp(lp) * (lp_safe - lq), -1)


def kl_loss(expert_logp, learner_logp):
    """AIL loss (Eq. 11/17, official discrete_kl_loss): mean_x KL[pi_expert(.|s) || pi_learner(.|b)], analytic over
    actions, expert held fixed (forward KL — Thm 1 needs this direction)."""
    e = jax.lax.stop_gradient(expert_logp)
    return jnp.mean(jnp.sum(jax.nn.softmax(e, -1) * (e - learner_logp), -1))


class TRPOCfg(NamedTuple):
    max_kl: float
    ent: float
    reg_adv: bool
    cg_steps: int = 10
    max_backtrack: int = 10
    accept_ratio: float = 0.1
    estimator: str = 'official'     # 'official' | 'paper' (see surrogate)
    kl_fallback: bool = True        # recompute a non-finite natural-gradient step from log-probs
    loglik_clamp: bool = True       # official Categorical clamp on the taken-action log-likelihood


def _cg(fvp, b, n):
    def body(c, _):
        x, r, p, rn = c
        Ap = fvp(p)
        alpha = rn / jnp.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        nrn = jnp.dot(r, r)
        p = r + (nrn / rn) * p
        return (x, r, p, nrn), None
    (x, _, _, _), _ = jax.lax.scan(body, (jnp.zeros_like(b), b, b, jnp.dot(b, b)), None, length=n)
    return x


def trpo_step(apply, params, X, A, old_lp, adv, cfg: TRPOCfg, sel, damping):
    """Official TRPO step: natural gradient by 10 CG steps on the Fisher of KL(old || new) over the subsample
    `sel` (10 %), step sqrt(2 delta / s^T F s) s (F incl. damping), backtracking x1/2 up to 10 times accepting the
    first step with surrogate improvement > 0, improvement / expected > 0.1 and mean full-batch KL <= delta.
    If the step is NaN, damping is doubled (persistently, as in the code) and the step recomputed; if still NaN no
    step is taken. Returns (new_params, info) with info['damping'] the damping to carry forward."""
    flat0, unravel = ravel_pytree(params)
    logp = lambda f, x: apply(unravel(f), x)
    surr = lambda f: surrogate(logp(f, X), A, old_lp, adv, cfg.ent, cfg.reg_adv, cfg.estimator, False,
                               cfg.loglik_clamp)
    s0, g = jax.value_and_grad(surr)(flat0)
    if cfg.kl_fallback:
        # TRPO freeze, entropy route: once an expert probability underflows to exactly 0, the official entropy
        # -sum log(p) p is NaN, and so is the surrogate (entropy-in-advantage or + c mean H). Recompute with the
        # 0 log 0 = 0 limit; identical wherever all p > 0, so runs that never get there are unchanged bit for bit.
        surr_safe = lambda f: surrogate(logp(f, X), A, old_lp, adv, cfg.ent, cfg.reg_adv, cfg.estimator, True,
                                        cfg.loglik_clamp)
        g_bad = ~(jnp.isfinite(s0) & jnp.all(jnp.isfinite(g)))
        s0, g = jax.lax.cond(g_bad, lambda: jax.value_and_grad(surr_safe)(flat0), lambda: (s0, g))
    p0 = jax.nn.softmax(logp(flat0, X), -1)
    lp0 = jax.lax.stop_gradient(jnp.log(p0))        # log of the official p0 (not a new log_softmax: keeps XLA's
    #                                                   fusion of p0, hence every non-fallback run, bitwise unchanged)
    Xs = X[sel]
    p0s = jax.lax.stop_gradient(jax.nn.softmax(logp(flat0, Xs), -1))
    lp0s = jnp.log(p0s)
    kl_s = lambda f: jnp.mean(kl_probs(p0s, jax.nn.softmax(logp(f, Xs), -1)))          # official calc_kl
    kl_s_lp = lambda f: jnp.mean(kl_logp(lp0s, jax.nn.log_softmax(logp(f, Xs), -1)))   # same value, stable Hessian

    def solve(damp, kl_fn):
        gkl = jax.grad(kl_fn)
        fvp = lambda v: jax.jvp(gkl, (flat0,), (v,))[1] + damp * v
        stp = _cg(fvp, g, cfg.cg_steps)
        return jnp.sqrt(2.0 * cfg.max_kl / jnp.dot(stp, fvp(stp))) * stp

    full = solve(damping, kl_s)
    if cfg.kl_fallback:
        # TRPO freeze: once the policy is near-deterministic (min prob ~1e-150 in float64, ~1e-38 in float32)
        # the Fisher-vector product of the official KL form is non-finite. The official code then doubles the damping
        # persistently and takes no step, so the policy freezes for good. We recompute the same Fisher (same KL value)
        # from log-probs instead. Runs that never reach that regime are unchanged bit for bit.
        fell = ~jnp.all(jnp.isfinite(full))
        full = jax.lax.cond(fell, lambda: solve(damping, kl_s_lp), lambda: full)
    else:
        fell = jnp.array(False)
    kl_alt = kl_s_lp if cfg.kl_fallback else kl_s
    bad = jnp.any(jnp.isnan(full))
    damping = jnp.where(bad, 2.0 * damping, damping)
    full = jax.lax.cond(bad, lambda: solve(damping, kl_alt), lambda: full)
    still_bad = jnp.any(jnp.isnan(full))
    full = jnp.where(still_bad, jnp.zeros_like(full), full)
    expected = jnp.dot(g, full)

    def try_scale(i):
        sc = 2.0 ** (-i.astype(flat0.dtype))
        f = flat0 + full * sc
        lp = logp(f, X)
        new = surrogate(lp, A, old_lp, adv, cfg.ent, cfg.reg_adv, cfg.estimator, False, cfg.loglik_clamp)
        kl = jnp.mean(kl_probs(p0, jax.nn.softmax(lp, -1)))
        if cfg.kl_fallback:      # official form wherever it is finite (bitwise unchanged), safe form otherwise
            new = jnp.where(jnp.isfinite(new), new,
                            surrogate(lp, A, old_lp, adv, cfg.ent, cfg.reg_adv, cfg.estimator, True,
                                      cfg.loglik_clamp))
            kl = jnp.where(jnp.isfinite(kl), kl, jnp.mean(kl_logp(lp0, jax.nn.log_softmax(lp, -1))))
        improve = jnp.where((new <= s0) | (kl > cfg.max_kl), -jnp.inf, new - s0)
        ok = (improve / (expected * sc) > cfg.accept_ratio) & (improve > 0)
        return ok

    def cond(c):
        i, found = c
        return (~found) & (i < cfg.max_backtrack)

    def body(c):
        i, _ = c
        ok = try_scale(i)
        return jnp.where(ok, i, i + 1), ok
    i_acc, found = jax.lax.while_loop(cond, body, (jnp.int32(0), jnp.array(False)))
    found = found & (~still_bad)
    scale = jnp.where(found, 2.0 ** (-i_acc.astype(flat0.dtype)), 0.0)
    new_flat = flat0 + full * scale
    info = {'damping': damping, 'surr0': s0, 'accepted': found, 'n_backtrack': i_acc, 'kl_fallback': fell,
            'step_norm': jnp.linalg.norm(full * scale)}
    return unravel(new_flat), info


# ----------------------------------------------------------------------------------------------- AIL projection
def n_proj_batches(cap: int, mb: int) -> int:
    return (cap - 1) // mb


def epoch_indices(perm1, perm2, L, mb: int, max_batches: int):
    """The official ReplayBuffer epoch sampler for two consecutive projection epochs after a push (verified against
    the official code). A minibatch is served only while strictly more than mb indices remain (`n_samp < len(order)`),
    so epoch 1 = the first k1 = floor((L-1)/mb) minibatches of a fresh permutation perm1 (of 0..L-1); the
    wrap-around batch (the r = L - mb*k1 leftovers + the head of a fresh perm2) is drawn and DISCARDED ('STOP'), and
    the sampler then drops r entries of perm2 (it slices `order[len(old_idx):]`). Epoch 2 = floor((L-r-1)/mb)
    minibatches from perm2[r:]. Returns idx (2, max_batches, mb), valid (2, max_batches)."""
    b = jnp.arange(max_batches)
    j = jnp.arange(mb)
    k1 = (L - 1) // mb
    c = L - mb * k1
    k2 = (L - c - 1) // mb
    i1 = jnp.clip(b[:, None] * mb + j[None], 0, perm1.shape[0] - 1)
    i2 = jnp.clip(c + b[:, None] * mb + j[None], 0, perm2.shape[0] - 1)
    idx = jnp.stack([perm1[i1], perm2[i2]])
    valid = jnp.stack([b < k1, b < k2])
    return idx, valid


def projection_epoch(loss_fn: Callable, params, opt_state, opt, idx, valid, keys=None):
    """One official projection epoch: Adam step per valid minibatch; returns the mean pre-update minibatch loss
    (the official `current_projection_loss`, Fig. 5's F(psi)). loss_fn(params, idx_batch, key)."""
    if keys is None:
        keys = jax.random.split(jax.random.PRNGKey(0), idx.shape[0])

    def mb(c, x):
        p, s, tot, n = c
        ii, v, k = x
        loss, g = jax.value_and_grad(loss_fn)(p, ii, k)
        u, s2 = opt.update(g, s, p)
        p2 = optax.apply_updates(p, u)
        p = jax.tree_util.tree_map(lambda a, b_: jnp.where(v, a, b_), p2, p)
        s = jax.tree_util.tree_map(lambda a, b_: jnp.where(v, a, b_), s2, s)
        return (p, s, tot + jnp.where(v, loss, 0.0), n + v), None
    z = jnp.zeros((), jax.tree_util.tree_leaves(params)[0].dtype)
    (params, opt_state, tot, n), _ = jax.lax.scan(mb, (params, opt_state, z, jnp.int32(0)), (idx, valid, keys))
    return params, opt_state, tot / jnp.maximum(n, 1)


def masked_perm(key, L, cap: int):
    """A uniform permutation of 0..L-1 in the first L slots of a length-cap array (L dynamic)."""
    u = jax.random.uniform(key, (cap,))
    u = jnp.where(jnp.arange(cap) < L, u, 2.0 + jnp.arange(cap) / cap)
    return jnp.argsort(u).astype(jnp.int32)


# ----------------------------------------------------------------------------------------------- image augmentation
def crop_resize(x, cf, orig, noise, pad: int = 2, out: int = 42):
    """Official image-AIL augmentation for a minibatch x (B, A1, A2, C): zero-pad `pad` px on both spatial
    dims, crop a (int(46 cf0), int(46 cf1)) window at `orig` (one crop per minibatch), add speckle `noise` (given on
    the padded grid) and clamp to [0,1], then bilinear-resize (align_corners=False, no antialias) back to out x out."""
    xp = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    P1, P2 = xp.shape[1], xp.shape[2]
    xp = jnp.clip(xp + noise, 0.0, 1.0)
    ts = jnp.stack([jnp.floor(P1 * cf[0]), jnp.floor(P2 * cf[1])]).astype(jnp.int32)

    def coords(t, o):
        o_ = jnp.arange(out, dtype=x.dtype)
        src = jnp.maximum((o_ + 0.5) * (t.astype(x.dtype) / out) - 0.5, 0.0)
        i0 = jnp.floor(src).astype(jnp.int32)
        i1 = i0 + (i0 < t - 1).astype(jnp.int32)
        l1 = src - i0.astype(x.dtype)
        return o + i0, o + i1, 1.0 - l1, l1
    a0, a1, wa0, wa1 = coords(ts[0], orig[0])
    b0, b1, wb0, wb1 = coords(ts[1], orig[1])
    g = lambda ia, ib: xp[:, ia][:, :, ib]
    return (wa0[None, :, None, None] * (wb0[None, None, :, None] * g(a0, b0) + wb1[None, None, :, None] * g(a0, b1))
            + wa1[None, :, None, None] * (wb0[None, None, :, None] * g(a1, b0) + wb1[None, None, :, None] * g(a1, b1)))


def augment(key, x, pad: int = 2):
    """Draw the official crop/speckle randomness and apply crop_resize to a minibatch of HWC images."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    P = x.shape[1] + 2 * pad
    cmin = 0.9 / 1.1
    cf = jax.random.uniform(k1, (2,), x.dtype, cmin, 1.0)
    ts = jnp.floor(P * cf).astype(jnp.int32)
    o1 = jax.random.randint(k2, (), 0, P - ts[0])
    o2 = jax.random.randint(k3, (), 0, P - ts[1])
    noise = 0.025 * jax.random.normal(k4, (x.shape[0], P, P, x.shape[3]), x.dtype)
    return crop_resize(x, cf, jnp.stack([o1, o2]), noise, pad, x.shape[1])

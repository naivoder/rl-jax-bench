"""PPO / TRPO updates and the paper's five instruments, as pure jittable JAX functions.

Re-derived from the paper, with the official companion code's semantics where the paper is silent;
nothing is copied from the official PyTorch code.

Instruments (the deliverable):
  * pg_vector            -- eq. (1) policy-gradient estimate over all policy params incl. log-std
  * cos_stats/bootstrap  -- pairwise agreement and cosine-to-reference, with a self-pair-free bootstrap
  * snr_* helpers        -- the signal-to-noise model and variance-reduction ratio
  * value_errors         -- Returns-MRE, GAE-MRE (true |.|/|.| and the official code's squared/|.|), MSEs
  * surrogate_grid       -- landscape slices of the algorithm's own surrogate (eqs 14/17) on a fixed sample
  * fit_value            -- the V* "true value" fit used as a baseline (Fig. 4)
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .model import gauss_kl


# ======================================================================== advantages
def gae(rew, values, not_done, gamma: float, lam: float):
    """GAE + truncated discounted returns for ONE actor's [T] arrays, official semantics:
    V(s_{t+1}) at the last index := V(s_t) of that index, times not_done; paths split after done steps;
    returns are not bootstrapped at the batch end . Returns (adv, ret)."""
    v_tp1 = jnp.concatenate([values[1:], values[-1:]]) * not_done
    delta = rew + gamma * v_tp1 - values

    def back(carry, x):
        a_next, g_next = carry
        d, r, nd = x
        a = d + gamma * lam * nd * a_next
        g = r + gamma * nd * g_next
        return (a, g), (a, g)

    _, (adv, ret) = jax.lax.scan(back, (jnp.float32(0), jnp.float32(0)), (delta, rew, not_done), reverse=True)
    return adv, ret


def lambda_return(rew, values, not_done, gamma: float, lam: float):
    """Explicit TD(lambda) return recursion G^l_t = r_t + g*nd_t[(1-l)V(s_{t+1}) + l G^l_{t+1}] with the same
    last-index convention as gae(); used only to test (V_old + A^GAE == G^lambda)."""
    v_tp1 = jnp.concatenate([values[1:], values[-1:]])
    T = rew.shape[0]

    def back(g_next, i):
        g_next = jnp.where(i == T - 1, v_tp1[i], g_next)  # last index: no G_{T}; bootstrap with v_tp1
        g = rew[i] + gamma * not_done[i] * ((1 - lam) * v_tp1[i] + lam * g_next)
        return g, g

    _, out = jax.lax.scan(back, jnp.float32(0), jnp.arange(T), reverse=True)
    return out


def adv_normalize(a, eps: float = 1e-8):
    """(A - mean)/(unbiased std + eps), eq. (15) as the official surrogate_reward computes it (torch .std())."""
    return (a - a.mean()) / (jnp.std(a, ddof=1) + eps)


def complete_episode_mask(not_done):
    """1 for steps whose episode terminated inside the [.., T] rollout (its MC return is not truncated)."""
    done = (not_done < 0.5).astype(jnp.int32)
    later_done = jnp.flip(jnp.cumsum(jnp.flip(done, -1), -1), -1)   # #dones at or after t
    return later_done > 0


# ======================================================================== losses
def masked_mean(x, m):
    m = m.astype(x.dtype)
    return jnp.sum(x * m) / jnp.maximum(jnp.sum(m), 1.0)


def value_loss(v_new, target, v_old, not_done, clip: bool, clip_eps: float):
    """Official value_loss_gae: squared error on not_done steps; with clipping, max(unclipped, clipped)."""
    sq = (v_new - target) ** 2
    if clip:
        v_c = v_old + jnp.clip(v_new - v_old, -clip_eps, clip_eps)
        sq = jnp.maximum(sq, (v_c - target) ** 2)
    return masked_mean(sq, not_done), sq


def ppo_surrogate(logp_new, logp_old, nadv, clip_eps):
    ratio = jnp.exp(logp_new - logp_old)
    return jnp.mean(jnp.minimum(ratio * nadv, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * nadv))


# ======================================================================== train state + updates
class TrainState(NamedTuple):
    pi: jax.Array
    v: jax.Array
    pi_opt: optax.OptState
    v_opt: optax.OptState


class Batch(NamedTuple):
    obs: jax.Array
    act: jax.Array
    logp: jax.Array
    adv: jax.Array
    ret: jax.Array
    values: jax.Array
    not_done: jax.Array
    vadv: jax.Array | None = None   # GAE advantage for the VALUE target when `adv` (policy) is not GAE (E5 arms)


def make_optimizers(cfg):
    pi_tx = optax.inject_hyperparams(optax.adam)(learning_rate=cfg.get("policy_lr", 0.0), eps=cfg.get("adam_eps", 1e-5))
    v_tx = optax.inject_hyperparams(optax.adam)(learning_rate=cfg["value_lr"], eps=cfg["value_adam_eps"])
    return pi_tx, v_tx


def lr_scale(cfg, it):
    """LambdaLR(1 - it/train_steps) when anneal_lr (official agent.py); constant otherwise."""
    if cfg.get("anneal_lr", False):
        return 1.0 - it / float(cfg["train_iterations"])
    return 1.0


def _set_lr(opt_state, lr):
    hp = dict(opt_state.hyperparams)
    hp["learning_rate"] = jnp.asarray(lr, jnp.float32)
    return opt_state._replace(hyperparams=hp)


def _minibatch_perms(key, n, epochs, n_mb):
    assert n % n_mb == 0, (n, n_mb)
    ks = jax.random.split(key, epochs)
    perms = jax.vmap(lambda k: jax.random.permutation(k, n))(ks)
    return perms.reshape(epochs * n_mb, n // n_mb)


def value_update(nets, cfg, v, v_opt, v_tx, b: Batch, lr, key):
    """10 epochs x 32 minibatches of Adam on the (clipped) squared error to V_old + A^GAE (eq. 4)."""
    target = b.values + (b.adv if b.vadv is None else b.vadv)
    clip = bool(cfg.get("value_clipping", False))
    v_opt = _set_lr(v_opt, lr)
    idx = _minibatch_perms(key, b.obs.shape[0], cfg["value_epochs"], cfg["value_minibatches"])

    def step(carry, ix):
        v, opt = carry
        def loss(p):
            return value_loss(nets.v_apply(p, b.obs[ix]), target[ix], b.values[ix], b.not_done[ix], clip,
                              cfg.get("clip_val_eps", 0.2))[0]
        l, g = jax.value_and_grad(loss)(v)
        upd, opt = v_tx.update(g, opt, v)
        return (optax.apply_updates(v, upd), opt), l

    (v, v_opt), losses = jax.lax.scan(step, (v, v_opt), idx)
    return v, v_opt, losses[-1]


def ppo_policy_update(nets, cfg, pi, pi_opt, pi_tx, b: Batch, lr, key):
    """10 epochs x 32 minibatches of Adam ascent on the clipped surrogate (eq. 17), per-minibatch adv norm ."""
    pi_opt = _set_lr(pi_opt, lr)
    idx = _minibatch_perms(key, b.obs.shape[0], cfg["policy_epochs"], cfg["minibatches"])
    eps = cfg["clip_eps"]

    def step(carry, ix):
        p, opt = carry
        nadv = adv_normalize(b.adv[ix]) if cfg.get("adv_center", True) else \
            b.adv[ix] / (jnp.std(b.adv[ix], ddof=1) + 1e-8)          # diagnostic: scale only, no mean baseline
        def loss(p):
            return -ppo_surrogate(nets.logp(p, b.obs[ix], b.act[ix]), b.logp[ix], nadv, eps)
        l, g = jax.value_and_grad(loss)(p)
        upd, opt = pi_tx.update(g, opt, p)
        p_new = optax.apply_updates(p, upd)
        return (p_new, opt), l

    (pi_new, pi_opt), losses = jax.lax.scan(step, (pi, pi_opt), idx)
    mean0, ls0 = nets.pi_apply(pi, b.obs)
    mean1, ls1 = nets.pi_apply(pi_new, b.obs)
    ratio = jnp.exp(nets.logp(pi_new, b.obs, b.act) - b.logp)
    stats = dict(approx_kl=jnp.mean(gauss_kl(mean0, ls0, mean1, ls1)),
                 clip_frac=jnp.mean(jnp.abs(ratio - 1.0) > eps), max_ratio=jnp.max(ratio))
    return pi_new, pi_opt, stats


def cg_solve(fvp, b, iters: int):
    """Conjugate gradients from x=0 (official torch_utils.cg_solve semantics)."""
    def body(c, _):
        x, r, p, rr = c
        Ap = fvp(p)
        alpha = rr / jnp.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = jnp.dot(r, r)
        p = r + (rr_new / rr) * p
        return (x, r, p, rr_new), None
    (x, *_), _ = jax.lax.scan(body, (jnp.zeros_like(b), b, b, jnp.dot(b, b)), None, length=iters)
    return x


def trpo_policy_update(nets, cfg, pi, b: Batch, key):
    """Natural-gradient step (eq. 16): CG on (F + damping I) with F = HVP of mean KL on a random 10% subset,
    max step sqrt(2 delta / s^T(F+cI)s) s, backtracking line search (10 halvings, accept ratio 0.1, KL<=delta)."""
    nadv = adv_normalize(b.adv)
    delta = cfg["kl_delta"]
    n = b.obs.shape[0]

    def surr(p):
        return jnp.mean(jnp.exp(nets.logp(p, b.obs, b.act) - b.logp) * nadv)

    s0, g = jax.value_and_grad(surr)(pi)
    n_f = int(n * cfg["fisher_fraction"])
    sel = jax.random.choice(key, n, (n_f,), replace=False)
    obs_f = b.obs[sel]
    m_old, ls_old = jax.lax.stop_gradient(nets.pi_apply(pi, obs_f))

    def kl_f(p):
        m, ls = nets.pi_apply(p, obs_f)
        return jnp.mean(gauss_kl(m_old, ls_old, m, ls))

    grad_kl = jax.grad(kl_f)

    def fvp(v):
        return jax.jvp(grad_kl, (pi,), (v,))[1] + cfg["cg_damping"] * v

    step = cg_solve(fvp, g, cfg["cg_steps"])
    shs = jnp.dot(step, fvp(step))
    full = jnp.sqrt(2 * delta / shs) * step
    expected = jnp.dot(g, full)
    m_all, ls_all = nets.pi_apply(pi, b.obs)

    def try_step(i):
        sc = cfg["backtrack_factor"] ** i
        p = pi + sc * full
        s_new = surr(p)
        m1, ls1 = nets.pi_apply(p, b.obs)
        kl = jnp.mean(gauss_kl(m_all, ls_all, m1, ls1))
        improve = jnp.where((s_new <= s0) | (kl > delta), -jnp.inf, s_new - s0)
        ok = (improve / (expected * sc) > cfg["backtrack_accept_ratio"]) & (improve > 0)
        return ok, sc, kl

    oks, scs, kls = jax.vmap(try_step)(jnp.arange(cfg["backtrack_steps"]))
    first = jnp.argmax(oks)
    any_ok = jnp.any(oks)
    scale = jnp.where(any_ok, scs[first], 0.0)
    pi_new = pi + scale * full
    stats = dict(ls_scale=scale, ls_accepted=any_ok.astype(jnp.float32),
                 post_kl=jnp.where(any_ok, kls[first], 0.0), step_norm=jnp.linalg.norm(scale * full),
                 surr_gain=surr(pi_new) - s0)
    return pi_new, stats


def make_train_update(nets, cfg):
    """One training iteration's parameter update on a 2048-pair batch: value step, then policy step
    (official take_steps order). The value and policy LR schedules step once per iteration."""
    pi_tx, v_tx = make_optimizers(cfg)

    def update(ts: TrainState, b: Batch, it, key):
        k_v, k_p = jax.random.split(key)
        sc = lr_scale(cfg, it)
        v, v_opt, vloss = value_update(nets, cfg, ts.v, ts.v_opt, v_tx, b, cfg["value_lr"] * sc, k_v)
        if cfg["alg"] == "ppo":
            pi, pi_opt, st = ppo_policy_update(nets, cfg, ts.pi, ts.pi_opt, pi_tx, b, cfg["policy_lr"] * sc, k_p)
        else:
            pi, st = trpo_policy_update(nets, cfg, ts.pi, b, k_p)
            pi_opt = ts.pi_opt
        st["value_loss"] = vloss
        return TrainState(pi, v, pi_opt, v_opt), st

    def init_state(key):
        pi, v = nets.init(key)
        return TrainState(pi, v, pi_tx.init(pi), v_tx.init(v))

    return update, init_state


# ======================================================================== gradient instruments
def pg_vector(nets, pi_flat, obs, act, adv, normalize: bool = True):
    """Eq. (1): grad_theta mean(pi_theta/pi_theta_k * A_hat) at theta = theta_k (= mean(psi * A_hat))."""
    a = adv_normalize(adv) if normalize else adv

    def surr(p):
        lp = nets.logp(p, obs, act)
        return jnp.mean(jnp.exp(lp - jax.lax.stop_gradient(lp)) * a)

    return jax.grad(surr)(pi_flat)


def cosine(x, y):
    return jnp.dot(x, y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y))


def cos_matrix(G):
    G = np.asarray(G, np.float64)
    Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
    return Gn @ Gn.T


def pairwise_mean(C, idx=None):
    """Mean cosine over pairs i<j of the (resampled) estimates; pairs of the SAME original estimate excluded."""
    m = C.shape[0]
    idx = np.arange(m) if idx is None else np.asarray(idx)
    sub = C[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), 1)
    keep = idx[iu[0]] != idx[iu[1]]
    return float(sub[iu][keep].mean()) if keep.any() else float("nan")


def cos_stats(G, g_ref=None, n_boot: int = 1000, seed: int = 0) -> dict:
    """ pairwise / to_true with percentile-bootstrap 95% CIs over the m estimates ."""
    G = np.asarray(G, np.float64)
    C = cos_matrix(G)
    out = {"pairwise": pairwise_mean(C), "m": int(G.shape[0])}
    t = None
    if g_ref is not None:
        r = np.asarray(g_ref, np.float64)
        t = (G @ r) / (np.linalg.norm(G, axis=1) * np.linalg.norm(r))
        out["to_true"] = float(t.mean())
        out["to_true_each"] = t.tolist()
    rng = np.random.default_rng(seed)
    pw, tt = [], []
    for _ in range(n_boot):
        ix = rng.integers(0, G.shape[0], G.shape[0])
        pw.append(pairwise_mean(C, ix))
        if t is not None:
            tt.append(t[ix].mean())
    pw = np.asarray(pw)
    out["pairwise_ci"] = np.nanpercentile(pw, [2.5, 97.5]).tolist()
    if t is not None:
        out["to_true_ci"] = np.percentile(tt, [2.5, 97.5]).tolist()
    # direct variance read-out: n * trace(empirical covariance) / ||mean||^2 (for cross-checks)
    out["mean_norm2"] = float(np.sum(G.mean(0) ** 2))
    out["trace_cov"] = float(np.sum(G.var(0, ddof=1)))
    return out


def snr_pairwise(n, n_half):
    """: E pairwise(n) ~ n / (n + n_half), n_half = tr(Sigma)/||g||^2."""
    return n / (n + n_half)


def snr_to_true(n, n_half, N=np.inf):
    """: E to_true(n) ~ 1 / sqrt((1 + n_half/n)(1 + n_half/N))."""
    return 1.0 / np.sqrt((1 + n_half / n) * (1 + (0 if np.isinf(N) else n_half / N)))


def variance_ratio_from_cos(c_b, c_0):
    """: tr Sigma_b / tr Sigma_0 = [(1-c_b)/c_b] / [(1-c_0)/c_0] at equal n (same mean gradient)."""
    return ((1 - c_b) / c_b) / ((1 - c_0) / c_0)


# ======================================================================== value instruments
def value_errors(v_new, v_old, adv, ret, not_done, clip: bool = False, clip_eps: float = 0.2) -> dict:
    """ + official-code variants, over not_done steps : y = V_old + A^GAE, G = truncated MC return."""
    y = v_old + adv
    nd = not_done
    sq_code = value_loss(v_new, y, v_old, nd, clip, clip_eps)[1]
    return dict(
        returns_mre=masked_mean(jnp.abs(v_new - ret) / jnp.abs(ret), nd),
        gae_mre_true=masked_mean(jnp.abs(v_new - y) / jnp.abs(y), nd),
        gae_mre_code=masked_mean(sq_code / jnp.abs(y), nd),
        returns_mse=masked_mean((v_new - ret) ** 2, nd),
        gae_mse=masked_mean((v_new - y) ** 2, nd),
    )


def mre(v, target, mask=None):
    """Plain mean relative error mean |v - t| / |t| (two-point toy)."""
    r = jnp.abs(v - target) / jnp.abs(target)
    return r.mean() if mask is None else masked_mean(r, mask)


def fit_value(nets, key, obs, target, mask, lr=1e-3, steps=4000, batch=4096, eval_every=100, init_flat=None):
    """V*: a fresh [64,64] value net fit by Adam to MC returns (mask = usable steps), early-stopped on a random 10%
    held-out split . Targets are STANDARDISED with the training-split mean/std and the net predicts
    (G - mu)/sd; V*(s) = mu + sd * net(s). Without this, raw-reward (TRPO) returns in the hundreds cannot be reached
    by Adam at lr ~1e-4..1e-3 within ~1e3 steps (each step moves the output by ~lr * width), and V* ended up worse
    than the agent's own V_theta (first small-scale pass). Returns (flat params of the standardised net, mu, sd, info)."""
    k_init, k_split, k_mb = jax.random.split(key, 3)
    n = obs.shape[0]
    v0 = nets.init(k_init)[1] if init_flat is None else init_flat
    held = jax.random.uniform(k_split, (n,)) < 0.1
    w_tr = (mask & ~held).astype(jnp.float32)
    w_ho = (mask & held).astype(jnp.float32)
    mu = jnp.sum(w_tr * target) / jnp.sum(w_tr)
    sd = jnp.sqrt(jnp.sum(w_tr * (target - mu) ** 2) / jnp.sum(w_tr)) + 1e-8
    z = (target - mu) / sd
    sched = optax.cosine_decay_schedule(lr, steps, alpha=0.05)
    tx = optax.adam(sched, eps=1e-5)
    p_tr = w_tr / jnp.sum(w_tr)
    cs = next(c for c in (100_000, 50_000, 20_000, 10_000, 5000, 2000, 1000, 500, 100, 1) if n % c == 0)

    def ho_mse(v):
        def chunk(i):
            sl = jax.lax.dynamic_slice_in_dim
            o = sl(obs, i * cs, cs); t = sl(z, i * cs, cs); w = sl(w_ho, i * cs, cs)
            return jnp.sum(w * (nets.v_apply(v, o) - t) ** 2), jnp.sum(w)
        s_, c_ = jax.lax.map(chunk, jnp.arange(n // cs))
        return s_.sum() / jnp.maximum(c_.sum(), 1.0)

    def train_chunk(carry, k):
        v, opt = carry
        def one(c, kk):
            v, opt = c
            ix = jax.random.choice(kk, n, (batch,), p=p_tr)
            l, g = jax.value_and_grad(lambda p: jnp.mean((nets.v_apply(p, obs[ix]) - z[ix]) ** 2))(v)
            u, opt = tx.update(g, opt, v)
            return (optax.apply_updates(v, u), opt), l
        (v, opt), ls = jax.lax.scan(one, (v, opt), jax.random.split(k, eval_every))
        return (v, opt), (v, ho_mse(v), ls[-1])

    (_, _), (vs, ho, tr) = jax.lax.scan(train_chunk, (v0, tx.init(v0)), jax.random.split(k_mb, steps // eval_every))
    best = jnp.argmin(ho)
    # held-out R^2 in standardised units (= 1 - MSE / Var) -- 1 - ho since Var(z) ~ 1 on the held-out split
    return vs[best], mu, sd, dict(heldout_mse_std=ho, train_mse_std=tr, best_step=(best + 1) * eval_every,
                                  heldout_r2=1.0 - ho[best], target_mu=mu, target_sd=sd)


# ======================================================================== landscape instrument
def surrogate_grid(nets, alg, clip_eps, pi_flat, delta, u, alphas, betas, b: Batch, mb_idx=None):
    """S(alpha, beta) = the algorithm's own objective on the step's FIXED sample with that sample's normalised
    advantages (PPO eq. 17 clipped; TRPO eq. 14), at theta_k + alpha*Delta + beta*u. S(0,0) = 0 .

    Advantages are normalised over the WHOLE batch. PPO's update normalises per MINIBATCH, so for PPO this is an
    approximation of the optimised objective . With mb_idx [epochs*n_mb, mb] (the update's own
    minibatch partitions, from _minibatch_perms with the update key) the PPO surface is instead the mean over every
    (epoch, minibatch) of the clipped surrogate with per-minibatch normalised advantages: the objective PPO's 10x32
    Adam steps actually ascend, evaluated as a function of theta."""
    nadv = adv_normalize(b.adv)
    if mb_idx is not None:
        nadv_mb = jax.vmap(lambda ix: adv_normalize(b.adv[ix]))(mb_idx)            # [epochs*n_mb, mb]
        lp_old_mb = b.logp[mb_idx]

    def at(a, bb):
        p = pi_flat + a * delta + bb * u
        lp = nets.logp(p, b.obs, b.act)
        if alg == "ppo":
            if mb_idx is not None:
                return jnp.mean(jax.vmap(lambda l, lo, na: ppo_surrogate(l, lo, na, clip_eps))(
                    lp[mb_idx], lp_old_mb, nadv_mb))
            return ppo_surrogate(lp, b.logp, nadv, clip_eps)
        return jnp.mean(jnp.exp(lp - b.logp) * nadv)

    A, B = jnp.meshgrid(alphas, betas, indexing="ij")
    return jax.vmap(jax.vmap(at))(A, B)


def landscape_stats(S, T, T_se, alphas) -> dict:
    """Summary statistics of the 2-D slices (axis 0 = alpha along the step, axis 1 = beta)."""
    from scipy.stats import spearmanr
    S, T, T_se = map(np.asarray, (S, T, T_se))
    da = float(alphas[1] - alphas[0])
    i1 = int(np.argmin(np.abs(np.asarray(alphas) - 1.0)))
    d2a = T[2:, :] - 2 * T[1:-1, :] + T[:-2, :]
    d2b = T[:, 2:] - 2 * T[:, 1:-1] + T[:, :-2]
    rough = np.sqrt(np.mean(np.concatenate([d2a.ravel(), d2b.ravel()]) ** 2))
    return dict(dT_dalpha0=float((T[1, 0] - T[0, 0]) / da), true_gain_step=float(T[i1, 0] - T[0, 0]),
                true_gain_se=float(np.hypot(T_se[i1, 0], T_se[0, 0])),
                surr_gain_step=float(S[i1, 0] - S[0, 0]),
                spearman_S_T=float(spearmanr(S.ravel(), T.ravel()).correlation),
                # recheck: agreement ALONG THE STEP only (beta = 0 column), robust to the u scale
                spearman_S_T_step=float(spearmanr(S[:, 0], T[:, 0]).correlation),
                roughness_rel_se=float(rough / max(np.mean(T_se), 1e-12)), roughness=float(rough))

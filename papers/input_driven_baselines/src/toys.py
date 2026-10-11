"""toy: the App. A 1-D toy with a known ground-truth policy gradient.

s_{t+1} = s_t + a_t + z_t, r_t = a_t + z_t, a_t in {-1,+1}, z_t iid uniform on {-sigma,+sigma},
pi_theta(+1) = sigmoid(theta) (state independent). Estimator, paper convention:

    g = sum_t psi_t (R_t - b_t), R_t = sum_{t'>=t} gamma^t' r_t', psi_t = 1[a_t=+1] - sigmoid(theta)

Parts:
  A theta=0 Monte Carlo (1e6 trajectories) of the state and input baselines vs the closed forms,
     for gamma in {0.9,0.99}, T in {20,200}, sigma^2 in {1,25}; plus the practical gamma^(t'-t) convention.
  B theta != 0: exact enumeration over all (a,z) in {+-1}^{2T} (T=8) of mean and variance under
     V(s), V(s,z) and the variance-optimal b* (P-4), cross-checked by Monte Carlo.
  C negative controls: baselines that read the rollout's own future actions / next state must be biased;
     in-sample (non-cross-fitted) per-input mean is biased, leave-one-out (cross-fitted) is not.
  D learned baselines through the production code paths (model.Critic, algorithm.value_loss /
     maml_crossfit): state net, 10 per-input nets, MAML meta-net; unbiasedness + variance vs V1/V2.
  E learning: A2C on theta with each baseline; return increases / actor loss decreases.
"""
from __future__ import annotations

import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import algorithm as alg
from .common import info, metric, write_json
from .data import sample_toy_inputs
from .model import Critic

N_MC = 1_000_000
CHUNK = 125_000


# ---------------------------------------------------------------- closed forms
def analytic_theta0(gamma: float, T: int, s2: float) -> dict:
    t = np.arange(T, dtype=np.float64)
    g2 = gamma ** (2 * t)
    V2 = 0.25 * np.sum(t * g2)
    V1 = V2 + 0.25 * s2 * np.sum((t + 1) * g2)
    mean = (1 - gamma ** T) / (2 * (1 - gamma))
    # practical convention G_t = sum gamma^(t'-t) r_t'
    d = np.arange(1, T, dtype=np.float64)
    V2p = 0.25 * np.sum((T - d) * gamma ** (2 * d))
    V1p = V2p + 0.25 * s2 * np.sum([np.sum(gamma ** (2 * np.arange(tp + 1))) for tp in range(T)])
    return {"mean": mean, "V1": V1, "V2": V2, "ratio": V1 / V2 if V2 > 0 else np.inf,
            "mean_practical": T / 2.0, "V1_practical": V1p, "V2_practical": V2p,
            "V1_minus_V2_inf": s2 / (4 * (1 - gamma ** 2) ** 2)}


# ---------------------------------------------------------------- toy rollouts
def toy_rollout(theta, z, key):
    """z [n,T] -> a [n,T] in {-1,1}, s_t [n,T] (state BEFORE step t, s_0=0), r [n,T]."""
    p = jax.nn.sigmoid(theta)
    a = 2.0 * jax.random.bernoulli(key, p, z.shape).astype(jnp.float32) - 1.0
    r = a + z
    s = jnp.cumsum(r, axis=1) - r
    return a, s, r


def rev_disc_cumsum(x, gamma, paper: bool = True):
    """paper: sum_{t'>=t} gamma^t' x_t'; practical: sum_{t'>=t} gamma^(t'-t) x_t'."""
    T = x.shape[-1]
    if paper:
        w = gamma ** jnp.arange(T, dtype=jnp.float32)
        return jnp.flip(jnp.cumsum(jnp.flip(x * w, -1), -1), -1)
    return alg.discounted_returns(x, gamma)


def psi(a, theta):
    return (a + 1.0) / 2.0 - jax.nn.sigmoid(theta)


def _moments(g: np.ndarray) -> dict:
    g = g.astype(np.float64)
    n = g.size
    m = g.mean()
    c = g - m
    var = np.mean(c ** 2) * n / (n - 1)
    m4 = np.mean(c ** 4)
    return {"mean": m, "mean_se": np.sqrt(var / n), "var": var, "var_se": np.sqrt(max(m4 - var ** 2, 0) / n), "n": n}


# ---------------------------------------------------------------- Part A
def part_a(seed: int = 0) -> dict:
    out = {}

    from functools import partial

    @partial(jax.jit, static_argnums=(1,))
    def chunk(key, T, gamma, sigma, theta):
        kz, ka = jax.random.split(key)
        z = sample_toy_inputs(kz, CHUNK, T, sigma)
        a, s, r = toy_rollout(theta, z, ka)
        ps = psi(a, theta)
        R = rev_disc_cumsum(r, gamma)
        Bz = rev_disc_cumsum(z, gamma)
        Rp = rev_disc_cumsum(r, gamma, paper=False)
        Bzp = rev_disc_cumsum(z, gamma, paper=False)
        return (jnp.sum(ps * R, 1), jnp.sum(ps * (R - Bz), 1), jnp.sum(ps * Rp, 1), jnp.sum(ps * (Rp - Bzp), 1))

    for gamma, T, s2 in itertools.product((0.9, 0.99), (20, 200), (1.0, 25.0)):
        key = jax.random.fold_in(jax.random.PRNGKey(seed), int(gamma * 1000) * 1000 + T * 10 + int(s2))
        outs = [[], [], [], []]
        for c in range(N_MC // CHUNK):
            res = chunk(jax.random.fold_in(key, c), T, gamma, float(np.sqrt(s2)), 0.0)
            for i in range(4):
                outs[i].append(np.asarray(res[i]))
        g1, g2, g1p, g2p = (np.concatenate(o) for o in outs)
        an = analytic_theta0(gamma, T, s2)
        m1, m2, m1p, m2p = _moments(g1), _moments(g2), _moments(g1p), _moments(g2p)
        cell = {
            "analytic": an,
            "state": m1, "input": m2, "state_practical": m1p, "input_practical": m2p,
            "z_mean_state": (m1["mean"] - an["mean"]) / m1["mean_se"],
            "z_mean_input": (m2["mean"] - an["mean"]) / m2["mean_se"],
            "z_var_state": (m1["var"] - an["V1"]) / m1["var_se"],
            "z_var_input": (m2["var"] - an["V2"]) / m2["var_se"],
            "z_mean_state_practical": (m1p["mean"] - an["mean_practical"]) / m1p["mean_se"],
            "z_mean_input_practical": (m2p["mean"] - an["mean_practical"]) / m2p["mean_se"],
            "z_var_state_practical": (m1p["var"] - an["V1_practical"]) / m1p["var_se"],
            "z_var_input_practical": (m2p["var"] - an["V2_practical"]) / m2p["var_se"],
            "ratio_empirical": m1["var"] / m2["var"],
        }
        zs = [abs(v) for k, v in cell.items() if k.startswith("z_")]
        cell["pass"] = bool(max(zs) < 3.0)
        name = f"g{gamma}_T{T}_s2{int(s2)}"
        out[name] = cell
        metric("toys.A", cell=name, V1=m1["var"], V1_an=an["V1"], V2=m2["var"], V2_an=an["V2"],
               mean=m2["mean"], mean_an=an["mean"], ratio=cell["ratio_empirical"], ratio_an=an["ratio"],
               max_abs_z=max(zs), ok=cell["pass"])
    return out


# ---------------------------------------------------------------- Part B
def part_b(T: int = 8, gamma: float = 0.9, thetas=(-1.0, 0.5, 1.5), seed: int = 1) -> dict:
    out = {}
    paths = np.array(list(itertools.product((-1.0, 1.0), repeat=T)))  # 2^T x T
    A = np.repeat(paths, len(paths), axis=0)  # actions
    Z = np.tile(paths, (len(paths), 1))  # inputs (sigma = 1)
    w = gamma ** np.arange(T)
    for theta in thetas:
        p = 1 / (1 + np.exp(-theta))
        prob = np.prod(np.where(A > 0, p, 1 - p), axis=1) * 0.5 ** T
        ps = (A + 1) / 2 - p
        r = A + Z
        R = np.flip(np.cumsum(np.flip(r * w, 1), 1), 1)
        Ea = 2 * p - 1
        tail = lambda x: np.flip(np.cumsum(np.flip(x * w, 1), 1), 1)
        bV = tail(np.full_like(Z, Ea))  # V(s_t): E[R_t] (policy ignores state; E z = 0)
        bVz = tail(Ea + Z)  # V(s_t, z_{t:})
        # b* (P-4): E_a[psi^2 Q]/E_a[psi^2]; E[psi^2 a]/E[psi^2] = 1-2p
        bstar = w * (1 - 2 * p) + w * Z + tail(np.full_like(Z, Ea) + Z) - w * (Ea + Z)
        bstar_S = w * (1 - 2 * p) + tail(np.full_like(Z, Ea)) - w * Ea
        true_grad = np.sum(w) * 2 * p * (1 - p)
        res = {"true_grad": true_grad}
        for name, b in (("none", 0 * R), ("V_s", bV), ("V_sz", bVz), ("bstar_S", bstar_S), ("bstar", bstar)):
            per_step = ps * (R - b)
            g = per_step.sum(1)
            mean = float(np.sum(prob * g))
            var = float(np.sum(prob * (g - mean) ** 2))
            # per-step (Thm 2) objective: sum_t E[psi_t^2 (R_t - b_t)^2] - ||E||^2 summed over t
            ps_var = float(np.sum(prob[:, None] * per_step ** 2, axis=0).sum()
                           - np.sum(np.sum(prob[:, None] * per_step, axis=0) ** 2))
            res[name] = {"mean": mean, "var_traj": var, "var_per_step_sum": ps_var}
        # Monte Carlo cross-check of the trajectory variance (1e6 samples, sigma=1)
        key = jax.random.fold_in(jax.random.PRNGKey(seed), int(theta * 100) + 1000)
        kz, ka = jax.random.split(key)
        z = sample_toy_inputs(kz, N_MC, T, 1.0)
        a, s, rr = toy_rollout(jnp.float32(theta), z, ka)
        RR = rev_disc_cumsum(rr, gamma)
        wj = jnp.asarray(w, jnp.float32)
        tailj = lambda x: jnp.flip(jnp.cumsum(jnp.flip(x * wj, 1), 1), 1)
        bVzj = tailj(Ea + z)
        bstarj = wj * (1 - 2 * p) + wj * z + tailj(Ea + z) - wj * (Ea + z)
        pj = psi(a, jnp.float32(theta))
        for name, b in (("V_s", tailj(jnp.full_like(z, Ea))), ("V_sz", bVzj), ("bstar", bstarj)):
            m = _moments(np.asarray(jnp.sum(pj * (RR - b), 1)))
            res[name]["mc"] = m
            res[name]["z_var"] = (m["var"] - res[name]["var_traj"]) / m["var_se"]
            res[name]["z_mean"] = (m["mean"] - true_grad) / m["mean_se"]
        exact_unbiased = all(abs(res[n]["mean"] - true_grad) < 1e-9 * max(1, abs(true_grad))
                             for n in ("none", "V_s", "V_sz", "bstar_S", "bstar"))
        order_ps = res["bstar"]["var_per_step_sum"] <= res["V_sz"]["var_per_step_sum"] + 1e-9 <= \
            res["V_s"]["var_per_step_sum"] + 2e-9
        order_traj = res["bstar"]["var_traj"] <= res["V_sz"]["var_traj"] + 1e-9 and \
            res["V_sz"]["var_traj"] <= res["V_s"]["var_traj"] + 1e-9
        mc_ok = all(abs(res[n]["z_var"]) < 3 and abs(res[n]["z_mean"]) < 3 for n in ("V_s", "V_sz", "bstar"))
        res.update({"exact_unbiased": exact_unbiased, "order_per_step": bool(order_ps),
                    "order_traj": bool(order_traj), "mc_ok": bool(mc_ok)})
        res["pass"] = bool(exact_unbiased and order_ps and mc_ok)
        out[f"theta{theta}"] = res
        metric("toys.B", theta=theta, true_grad=true_grad, V_s=res["V_s"]["var_traj"], V_sz=res["V_sz"]["var_traj"],
               bstar=res["bstar"]["var_traj"], bstar_S=res["bstar_S"]["var_traj"],
               ps_V_s=res["V_s"]["var_per_step_sum"], ps_V_sz=res["V_sz"]["var_per_step_sum"],
               ps_bstar=res["bstar"]["var_per_step_sum"], exact_unbiased=exact_unbiased,
               order_per_step=bool(order_ps), order_traj=bool(order_traj), mc_ok=bool(mc_ok))
    return out


# ---------------------------------------------------------------- Part C
def part_c(T: int = 20, gamma: float = 0.9, sigma: float = 1.0, k: int = 2, seed: int = 2) -> dict:
    n_z = N_MC // k
    key = jax.random.PRNGKey(seed)
    kz, ka = jax.random.split(key)
    z = sample_toy_inputs(kz, n_z, T, sigma)
    zr = jnp.repeat(z, k, axis=0)
    a, s, r = toy_rollout(jnp.float32(0.0), zr, ka)
    ps = psi(a, 0.0)
    R = rev_disc_cumsum(r, gamma)
    true = analytic_theta0(gamma, T, sigma ** 2)["mean"]
    Rz = R.reshape(n_z, k, T)
    loo = ((jnp.sum(Rz, 1, keepdims=True) - Rz) / (k - 1)).reshape(R.shape)
    insample = jnp.repeat(jnp.mean(Rz, 1), k, axis=0)
    s_next = s + r
    bases = {
        "input_exact": rev_disc_cumsum(zr, gamma),
        "crossfit_loo_mean": loo,  # baseline from the OTHER rollouts on the same z (Alg. 1 cross-fitting)
        "NEG_own_future_actions": rev_disc_cumsum(a, gamma),
        "NEG_next_state": 0.5 * s_next,
        "NEG_insample_mean": insample,  # same-half fit including the rollout itself (no cross-fitting)
    }
    out = {"true_grad": true, "k": k}
    for name, b in bases.items():
        m = _moments(np.asarray(jnp.sum(ps * (R - b), 1)))
        zb = (m["mean"] - true) / m["mean_se"]
        out[name] = {"mean": m["mean"], "mean_se": m["mean_se"], "var": m["var"], "bias_z": zb}
        metric("toys.C", baseline=name, mean=m["mean"], true=true, bias_z=zb, var=m["var"])
    neg_ok = all(abs(out[n]["bias_z"]) > 5 for n in out if isinstance(out[n], dict) and n.startswith("NEG"))
    pos_ok = all(abs(out[n]["bias_z"]) < 3 for n in ("input_exact", "crossfit_loo_mean"))
    out["pass"] = bool(neg_ok and pos_ok)
    return out


# ---------------------------------------------------------------- Part D: learned baselines via production code
def part_d(T: int = 20, gamma: float = 0.9, sigma: float = 5.0, seed: int = 3) -> dict:
    """Linear value nets (model.Critic, no hidden layer, no bias) on one-hot(t) features -- the toy's V(s_t) at
    theta=0 is a function of t only, so this class contains the exact state baseline and, per input,
    the exact input baseline."""
    an = analytic_theta0(gamma, T, sigma ** 2)
    feats = jnp.eye(T, dtype=jnp.float32)  # [T, T]
    cm = Critic(hidden=(), use_bias=False)  # bias is redundant with one-hot(t) and would make the inner SGD ill-conditioned
    key = jax.random.PRNGKey(seed)
    k_init, k_state, k_mv, k_meta, k_eval = jax.random.split(key, 5)
    phi0 = cm.init(k_init, feats[0])
    tx = optax.adam(1e-2)
    n_batch = 256

    def batch(key, z):
        a, s, r = toy_rollout(jnp.float32(0.0), z, key)
        return a, rev_disc_cumsum(r, gamma)

    X = lambda n: jnp.broadcast_to(feats, (n, T, T))

    # state net: fresh z every step
    @jax.jit
    def fit_state(phi):
        def body(c, kk):
            p, o = c
            k1, k2 = jax.random.split(kk)
            _, R = batch(k2, sample_toy_inputs(k1, n_batch, T, sigma))
            g = jax.grad(lambda q: alg.value_loss(cm.apply, q, X(n_batch), R, "mean"))(p)
            u, o = tx.update(g, o, p)
            return (optax.apply_updates(p, u), o), None

        (p, _), _ = jax.lax.scan(body, (phi, tx.init(phi)), jax.random.split(k_state, 2000))
        return p

    phi_state = fit_state(phi0)

    # multi-value: 10 fixed inputs, one net each
    z_fixed = sample_toy_inputs(jax.random.fold_in(k_mv, 0), 10, T, sigma)

    @jax.jit
    def fit_one(phi, zi, kk):
        def body(c, kb):
            p, o = c
            _, R = batch(kb, jnp.broadcast_to(zi, (n_batch, T)))
            g = jax.grad(lambda q: alg.value_loss(cm.apply, q, X(n_batch), R, "mean"))(p)
            u, o = tx.update(g, o, p)
            return (optax.apply_updates(p, u), o), None

        (p, _), _ = jax.lax.scan(body, (phi, tx.init(phi)), jax.random.split(kk, 2000))
        return p

    phi_mv = jax.vmap(fit_one, in_axes=(None, 0, 0))(phi0, z_fixed, jax.random.split(jax.random.fold_in(k_mv, 1), 10))

    # MAML meta-net: FOMAML over fresh z, k=16 rollouts per z (halves of 8), inner SGD on sum_time loss
    k_roll, h = 16, 8
    inner = dict(steps=5, lr=0.25, inner_reduction="sum_time", first_order=True)
    txm = optax.adam(1e-2)

    @jax.jit
    def fit_meta(phi):
        def body(c, kk):
            p, o = c
            k1, k2 = jax.random.split(kk)
            zz = jnp.repeat(sample_toy_inputs(k1, 1, T, sigma), k_roll, axis=0)
            _, R = batch(k2, zz)
            _, _, g, aux = alg.maml_crossfit(cm.apply, p, X(h), R[:h], X(h), R[h:], **inner)
            u, o = txm.update(g, o, p)
            return (optax.apply_updates(p, u), o), aux["mse_crossfit"]

        (p, _), losses = jax.lax.scan(body, (phi, txm.init(phi)), jax.random.split(k_meta, 2000))
        return p, losses

    phi_meta, meta_losses = fit_meta(phi0)

    # ---- evaluation: gradient estimator mean/var at theta=0 with each learned baseline
    n_eval_z = 50_000

    @jax.jit
    def eval_all(kk):
        k1, k2, k3, k4 = jax.random.split(kk, 4)
        # state net and MAML on fresh inputs, k_roll rollouts per input (MAML needs the shared-input halves)
        zz = jnp.repeat(sample_toy_inputs(k1, n_eval_z, T, sigma), k_roll, axis=0)
        a, R = batch(k2, zz)
        ps = psi(a, 0.0)
        b_state = cm.apply(phi_state, X(zz.shape[0]))
        g_state = jnp.sum(ps * (R - b_state), 1)
        Rz = R.reshape(n_eval_z, k_roll, T)
        b1, b2, _, _ = jax.vmap(lambda RR: alg.maml_crossfit(cm.apply, phi_meta, X(h), RR[:h], X(h), RR[h:], **inner))(Rz)
        b_meta = jnp.concatenate([b1, b2], axis=1).reshape(R.shape)
        g_meta = jnp.sum(ps * (R - b_meta), 1)
        g_meta_unadapted = jnp.sum(ps * (R - cm.apply(phi_meta, X(zz.shape[0]))), 1)
        # multi-value: inputs drawn uniformly from the 10 fixed ones
        idx = jax.random.randint(k3, (n_eval_z * 4,), 0, 10)
        a2, R2 = batch(k4, z_fixed[idx])
        ps2 = psi(a2, 0.0)
        b_mv = jax.vmap(lambda i: cm.apply(jax.tree_util.tree_map(lambda p: p[i], phi_mv), feats))(idx)
        g_mv = jnp.sum(ps2 * (R2 - b_mv), 1)
        return g_state, g_meta, g_meta_unadapted, g_mv

    gs, gm, gmu, gmv = (np.asarray(x) for x in eval_all(k_eval))
    out = {"analytic": an, "meta_crossfit_mse_first100": float(np.mean(np.asarray(meta_losses[:100]))),
           "meta_crossfit_mse_last100": float(np.mean(np.asarray(meta_losses[-100:])))}
    for name, g in (("state_net", gs), ("maml_crossfit", gm), ("maml_unadapted", gmu), ("multi_value_10", gmv)):
        m = _moments(g)
        m["bias_z"] = (m["mean"] - an["mean"]) / m["mean_se"]
        m["var_over_V1"] = m["var"] / an["V1"]
        m["var_over_V2"] = m["var"] / an["V2"]
        out[name] = m
        metric("toys.D", baseline=name, mean=m["mean"], mean_an=an["mean"], bias_z=m["bias_z"], var=m["var"],
               V1=an["V1"], V2=an["V2"])
    unb = all(abs(out[n]["bias_z"]) < 3 for n in ("state_net", "maml_crossfit", "multi_value_10"))
    # per-z, k-rollout cross-fitting leaves extra variance from the other half's own action noise; require MAML
    # and multi-value to sit closer to V2 than to V1 and the state net to sit within 5% of V1
    order = (out["multi_value_10"]["var"] < out["maml_crossfit"]["var"] < out["state_net"]["var"])
    near = (abs(out["state_net"]["var_over_V1"] - 1) < 0.05 and abs(out["multi_value_10"]["var_over_V2"] - 1) < 0.05
            and out["maml_crossfit"]["var"] < 0.5 * (an["V1"] + an["V2"]))
    out["pass"] = bool(unb and order and near)
    return out


# ---------------------------------------------------------------- Part E: learning on the toy
def part_e(T: int = 20, gamma: float = 0.9, sigma: float = 5.0, iters: int = 300, seed: int = 4) -> dict:
    """A2C on the scalar logit theta with the analytic per-arm baselines; J(theta) = sum gamma^t (2 sigmoid - 1)."""
    out = {}
    w = gamma ** jnp.arange(T, dtype=jnp.float32)
    J = lambda th: float(jnp.sum(w) * (2 * jax.nn.sigmoid(th) - 1))
    for arm in ("none", "state", "input"):
        tx = optax.adam(0.02)
        th = jnp.float32(-2.0)
        o = tx.init(th)

        @jax.jit
        def step(th, o, kk):
            k1, k2 = jax.random.split(kk)
            z = sample_toy_inputs(k1, 16, T, sigma)
            a, s, r = toy_rollout(th, z, k2)
            R = rev_disc_cumsum(r, gamma)
            Ea = 2 * jax.nn.sigmoid(th) - 1
            b = {"none": 0.0 * R, "state": rev_disc_cumsum(jnp.full_like(z, 1.0) * Ea, gamma),
                 "input": rev_disc_cumsum(Ea + z, gamma)}[arm]
            adv = jax.lax.stop_gradient(R - b)

            def loss(t):
                logp = jnp.where(a > 0, jax.nn.log_sigmoid(t), jax.nn.log_sigmoid(-t))
                return -jnp.mean(jnp.sum(logp * adv, 1))

            l, g = jax.value_and_grad(loss)(th)
            u, o = tx.update(g, o, th)
            return optax.apply_updates(th, u), o, l

        Js, ls = [J(th)], []
        for i in range(iters):
            th, o, l = step(th, o, jax.random.fold_in(jax.random.PRNGKey(seed), i))
            ls.append(float(l))
            Js.append(J(th))
        out[arm] = {"J_start": Js[0], "J_end": Js[-1], "J_opt": float(jnp.sum(w)), "theta_end": float(th),
                    "loss_first20": float(np.mean(ls[:20])), "loss_last20": float(np.mean(ls[-20:]))}
        metric("toys.E", arm=arm, J_start=Js[0], J_end=Js[-1], J_opt=float(jnp.sum(w)))
    out["pass"] = bool(all(out[a]["J_end"] > out[a]["J_start"] + 0.5 * (out[a]["J_opt"] - out[a]["J_start"])
                           for a in ("none", "state", "input")))
    return out


def run_toys(out_dir: str = "runs/toys") -> dict:
    t0 = time.perf_counter()
    res = {}
    for name, fn in (("A_theta0_closed_form", part_a), ("B_theta_neq0_exact", part_b),
                     ("C_negative_controls", part_c), ("D_learned_baselines", part_d), ("E_learning", part_e)):
        t = time.perf_counter()
        res[name] = fn()
        if name == "A_theta0_closed_form":
            res[name]["pass"] = all(v["pass"] for v in res[name].values() if isinstance(v, dict))
        elif name == "B_theta_neq0_exact":
            res[name]["pass"] = all(v["pass"] for v in res[name].values() if isinstance(v, dict))
        info(f"toys part done", part=name, ok=res[name]["pass"], seconds=round(time.perf_counter() - t, 1))
    res["pass"] = all(res[k]["pass"] for k in res)
    res["seconds"] = time.perf_counter() - t0
    write_json(f"{out_dir}/metrics.json", res)
    metric("toys", ok=res["pass"], seconds=round(res["seconds"], 1))
    return res

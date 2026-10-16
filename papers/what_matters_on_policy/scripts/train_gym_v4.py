"""Faithful-env path (promoted from runs/recheck/official-oracle/ours_on_v4.py): OUR learner (src.algorithm.update, src.model sampling/log-prob, src.data
AverageMeanStd) driven by a host loop on gymnasium HalfCheetah-v4 -- the simulator of the authors'-module reference
runs (scripts/official/official_loop.py) -- so ours and theirs can be compared on the SAME simulator. Protocol mirrors
official_loop.py: 256 envs x 8 steps, terminal time-limit cut (C22 = False), SEED reset convention (the obs after a
done is the reset obs), 10 stochastic eval episodes at step 0 and at the first iteration boundary >= k*100k
(k = 1..10), score = mean of the 10 points after step 0. CPU only.

  python scripts/train_gym_v4.py <arm> <seed> <out.json> [--overlap] [--total N] [--lr_horizon_frames F]

--lr_horizon_frames F overrides static.lr_horizon_frames (: default = training length 1001472; 2000000
reproduces the pre-fix  behaviour). Run it in the repository environment; the env is driven unwrapped with our own reset and
1000-step cut, so the gymnasium autoreset mode (1.x next-step vs 0.29 same-step) never enters.

<arm> is any arm of src/configs/agent.yaml (REC, A2b_initstd_1, A5a_lr_1em4, A7_valuenorm_none, X5_..., X6_...).
--overlap: emulate the released SEED unroll store (common/utils.py UnrollStore._complete_unrolls copies the last row of
an unroll -- incl. the action and behaviour logits computed with the PRE-update parameters -- to row 0 of the next
unroll; the actor executes that action). Off by default (the paper's B.1 text: synchronous, current policy).
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import algorithm as A  # noqa: E402
from src import configs as C  # noqa: E402
from src import data as D  # noqa: E402
from src import model as M  # noqa: E402

BOOL = ("obs_norm", "tanh", "softplus", "value_norm", "adv_norm", "recompute", "shared", "reward_norm", "logp_presquash")


def lane_cfg(r, static):
    v = {}
    for f in A.LaneCfg._fields:
        if f == "lr_horizon_steps":
            v[f] = jnp.float32(C.lr_horizon_steps(static, r.get("lr_horizon_frames")))
        elif f in BOOL:
            v[f] = jnp.bool_(r[f])
        else:
            v[f] = jnp.float32(r[f])
    return A.LaneCfg(**v)


class Envs:
    """N single gymnasium envs with the SEED reset convention and our own 1000-step cut."""

    def __init__(self, n, seed):
        self.e = [gym.make("HalfCheetah-v4", disable_env_checker=True).unwrapped for _ in range(n)]
        self.obs = np.stack([e.reset(seed=seed * 100_000 + i)[0] for i, e in enumerate(self.e)]).astype(np.float32)
        self.t = np.zeros(n, np.int64)
        self.ret = np.zeros(n)

    def step(self, a):
        o2, r, d = np.empty_like(self.obs), np.empty(len(self.e), np.float32), np.zeros(len(self.e), bool)
        fin = []
        for i, e in enumerate(self.e):
            o, ri, term, _, _ = e.step(a[i])
            self.t[i] += 1
            self.ret[i] += ri
            r[i] = ri
            if term or self.t[i] >= 1000:
                d[i] = True
                fin.append(self.ret[i])
                o, _ = e.reset()
                self.t[i], self.ret[i] = 0, 0.0
            o2[i] = o
        self.obs = o2
        return o2, r, d, fin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("seed", type=int)
    ap.add_argument("out")
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--total", type=int, default=None)
    ap.add_argument("--eval_episodes", type=int, default=10)
    ap.add_argument("--lr_horizon_frames", type=int, default=None)
    a = ap.parse_args()
    cfgall = C.load()
    static = dict(cfgall["static"])
    if a.lr_horizon_frames is not None:
        static["lr_horizon_frames"] = a.lr_horizon_frames
    r = C.resolve(cfgall, a.arm)
    cfg = lane_cfg(r, static)
    st = A.Static(num_envs=static["num_envs"], unroll=static["iteration_size"] // static["num_envs"],
                  epochs=static["epochs"], minibatch=static["minibatch"], episode_length=1000)
    N, T = st.num_envs, st.unroll
    total = a.total or static["total_env_steps"]
    it_total = math.ceil(total / static["iteration_size"])
    root = jax.random.PRNGKey(20210101)
    f = lambda i: jax.random.fold_in(jax.random.fold_in(root, i), a.seed)
    keys = {"init": f(1), "act": f(2), "mb": f(3), "eval": f(4)}
    envs = Envs(N, a.seed)
    obs_dim, act_dim = envs.obs.shape[1], 6
    params = M.init_params(keys["init"], obs_dim, act_dim, pi_width=static["pi_width"], pi_depth=2,
                           v_width=r["v_width"], v_depth=2, pi_last_gain=r["pi_last_gain"])
    ls = A.LaneState(params=params, opt=A.adam_init(params), obs_ms=D.meanstd_init(obs_dim), val_ms=D.meanstd_init(1),
                     ret_ms=D.meanstd_init(1), ret_acc=jnp.zeros((N,), jnp.float32), env=None,
                     obs=jnp.asarray(envs.obs), it=jnp.zeros((), jnp.int32), bad=jnp.zeros((), bool))
    shift = M.std_shift(cfg.init_std, cfg.min_std, cfg.softplus)

    @jax.jit
    def act(params, obs_ms, obs, key):
        o = D.normalize_obs(obs_ms, obs, cfg.obs_norm, cfg.obs_clip)
        mu, x_rho, v = M.forward(params, o, cfg.shared)
        std = M.std_of(x_rho, shift, cfg.min_std, cfg.softplus)
        env_a, x = M.sample_x(key, mu, std, cfg.tanh)
        return env_a, jnp.where(cfg.logp_presquash, x, env_a), mu, std, v

    upd = jax.jit(lambda ls, b, k: A.update(ls, b, cfg, st, k))

    def evaluate(params, obs_ms, p):
        E = a.eval_episodes
        ev = [gym.make("HalfCheetah-v4", disable_env_checker=True).unwrapped for _ in range(E)]
        o = np.stack([e.reset(seed=10_000 + a.seed * 1000 + p * 37 + i)[0] for i, e in enumerate(ev)]).astype(np.float32)
        ret = np.zeros(E)
        ke = jax.random.fold_in(keys["eval"], p)
        for t in range(1000):
            env_a, *_ = act(params, obs_ms, jnp.asarray(o), jax.random.fold_in(ke, t))
            env_a = np.asarray(env_a)
            for i, e in enumerate(ev):
                oi, ri, term, _, _ = e.step(env_a[i])
                o[i] = oi
                ret[i] += ri
        return float(ret.mean())

    pts = [0]
    k = 1
    while k * static["eval_every"] <= it_total * static["iteration_size"]:
        pts.append(math.ceil(k * static["eval_every"] / static["iteration_size"]))
        k += 1
    curve, diag, t0 = [(0, evaluate(ls.params, ls.obs_ms, 0))], [], time.time()
    carry_row = None            # --overlap: (env_a, stored_a, mu, std, v) of the previous bootstrap row
    fins = []
    for it in range(it_total):
        O, AC, MU, SD, V, R, DN = [envs.obs.copy()], [], [], [], [], [], []
        ka = jax.random.fold_in(keys["act"], it)
        for t in range(T):
            if t == 0 and carry_row is not None:
                env_a, sa, mu, sd, v = carry_row
            else:
                env_a, sa, mu, sd, v = act(ls.params, ls.obs_ms, jnp.asarray(envs.obs), jax.random.fold_in(ka, t))
            o2, rr, dd, fin = envs.step(np.asarray(env_a))
            fins += fin
            AC.append(np.asarray(sa)); MU.append(np.asarray(mu)); SD.append(np.asarray(sd)); V.append(np.asarray(v))
            R.append(rr); DN.append(dd); O.append(o2.copy())
        if a.overlap:   # bootstrap row computed with the pre-update params/obs stats, executed next iteration
            carry_row = act(ls.params, ls.obs_ms, jnp.asarray(envs.obs), jax.random.fold_in(ka, T))
        batch = A.Batch(jnp.asarray(np.stack(O)), jnp.asarray(np.stack(AC)), jnp.asarray(np.stack(MU)),
                        jnp.asarray(np.stack(SD)), jnp.asarray(np.stack(V)), jnp.asarray(np.stack(R)),
                        jnp.asarray(np.stack(DN)))
        ls, aux = upd(ls, batch, jax.random.fold_in(keys["mb"], it))
        if it + 1 in pts:
            p = pts.index(it + 1)
            curve.append(((it + 1) * static["iteration_size"], evaluate(ls.params, ls.obs_ms, p)))
            dg = {k2: float(v2) for k2, v2 in aux.items()}
            dg["lr_next"] = float(A.lr_at(cfg, ls.opt.t))
            dg.update(steps=(it + 1) * static["iteration_size"], train_ret=float(np.mean(fins)) if fins else None,
                      wall=time.time() - t0, bad=bool(ls.bad))
            fins = []
            diag.append(dg)
            print(f"[METRIC] ours_on_v4 arm={a.arm} seed={a.seed} overlap={a.overlap} steps={dg['steps']} "
                  f"eval={curve[-1][1]:.1f} std={dg['std']:.3f} kl={dg['approx_kl']:.4f} clip={dg['clipfrac']:.3f} "
                  f"wall={dg['wall']:.0f}", flush=True)
            json.dump({"env": "HalfCheetah-v4", "gymnasium": gym.__version__, "mujoco": __import__("mujoco").__version__,
                       "arm": a.arm, "seed": a.seed, "overlap": a.overlap, "curve": curve, "diag": diag,
                       "score": float(np.mean([c[1] for c in curve[1:]])), "resolved": r, "eval_episodes": a.eval_episodes,
                       "lr_horizon_frames": static["lr_horizon_frames"], "lr_horizon_steps": float(cfg.lr_horizon_steps)},
                      open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

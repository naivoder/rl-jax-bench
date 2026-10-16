"""Training drivers: every (arm, seed) lane of one experiment is ONE vmapped lax.scan program (chunks of steps),
with orbax checkpoint/resume per chunk group, NaN guard, compile-vs-steady timing and metrics.json writers.

Usage:
    python -m src.train --config src/configs/e3_minatar.yaml --tag small [--set key=value ...] [--no-ckpt]

PRNG key tree:
    root      = PRNGKey(master_seed)
    run_key   = fold_in(root, seed)                  # arm-independent => arms are paired (common random numbers)
      init    = fold_in(run_key, 0) -> fold_in(., net_id) -> fold_in(., layer_idx) -> split = (kernel, bias)
                (SparseInit kernel: split = (uniform values, permutation keys))
      env     = fold_in(run_key, 1) -> fold_in(., 0) = first reset
      step    = fold_in(run_key, 2) -> fold_in(., t) -> split(5) = (eps coin | Gaussian noise, random action,
                sticky coin, env step, env reset)                     # t = 0-based global step
A resumed run reproduces the uninterrupted one (keys depend only on (seed, t)).
"""
from __future__ import annotations

import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()

import argparse  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from functools import partial  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

# full-precision float32 matmuls (TF32 on the 3090 would break the 1e-5 oracle differential tests)
jax.config.update("jax_default_matmul_precision", "highest")
from rl_jax_bench.jaxlog import Timer, check_device, info, metric, nan_guard, warn, write_json  # noqa: E402

from . import agents as ag  # noqa: E402
from . import data as dt  # noqa: E402
from . import eval as ev  # noqa: E402
from . import model as mdl  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO, "src", "configs")

# ------------------------------------------------------------------ arms (ONLY the switched mechanism differs)
S, D0, TI = mdl.INIT_SPARSE, mdl.INIT_DENSE0, mdl.INIT_TORCH
Q_STREAM = dict(use_ln=1, init=S, obs_norm=1, rew_scale=1, opt=ag.OPT_OBGD, lr=1.0, kappa=2.0, lam=0.8,
                eps_end=0.01, eps_frac=0.2, use_target=0, target_period=1, loss_mult=1.0, watkins_paper=0,
                adam_eps=1e-4)                                                      # App. F.2
Q_ARMS = {
    "stream_q": {},                                                                     # Alg. 8 + official
    "classic_q": dict(use_ln=0, init=TI, obs_norm=0, rew_scale=0, opt=ag.OPT_ADAM, lr=1e-5),   # App. F.2 + B1/
    "dqn1": dict(use_ln=0, init=TI, obs_norm=0, rew_scale=0, opt=ag.OPT_ADAM, lr=2.5e-4, lam=0.0, eps_end=0.05,
                 eps_frac=0.5, use_target=1, target_period=500, loss_mult=2.0,
                 adam_eps=1e-8),   # App. F.2 + B7 (CleanRL dqn.py: optim.Adam default eps 1e-8; the full-scale run used 1e-4)
    "stream_q_eps05": dict(eps_frac=0.05),                                              # paper F.2 value
    "stream_q_watkins_paper": dict(watkins_paper=1),                                    # paper Alg. 8 cut
    # Sensitivity arm (B1): classic Q that KEEPS LayerNorm, SparseInit, obs norm and reward scaling and swaps ONLY the
    # optimizer (ObGD -> Adam 1e-5 on -delta*z, App. F.2 lr/eps). Not a paper arm.
    "classic_q_keepnorm": dict(opt=ag.OPT_ADAM, lr=1e-5),
}
AC_STREAM = dict(use_ln=1, init=S, norm=1, opt=ag.OPT_OBGD, lr=1.0, kappa_pi=3.0, kappa_v=2.0, lam=0.8, tau=0.01,
                 adam_eps=1e-4)                                                      # App. F.4
AC_ARMS = {
    "stream_ac": {},                                                                    # Alg. 7, App. F.4
    "no_obgd": dict(opt=ag.OPT_ADAM, lr=1e-5),                                          # Fig. 7; lr = B3 (cfg)
    "no_norm": dict(norm=0),                                                            # Fig. 7 (: both)
    "no_ln": dict(use_ln=0),                                                            # Fig. 7
    "no_sparse": dict(init=D0),                                                         # Fig. 7
    "classic_ac": dict(use_ln=0, init=TI, norm=0, opt=ag.OPT_ADAM, lr=1e-7),            # Fig. 7/8, F.4 + B1/
    "stream_ac0": dict(lam=0.0),                                                        # Fig. 8
    "classic_ac0": dict(use_ln=0, init=TI, norm=0, opt=ag.OPT_ADAM, lr=1e-7, lam=0.0),  # Fig. 8
}
TD_STREAM = dict(use_ln=1, init=S, norm=1, opt=ag.OPT_OBGD, lr=1.0, kappa=2.0, lam=0.8, adam_eps=1e-4)  # F.1
TD_ARMS = {
    "stream_td": {},                                                                    # Alg. 9, App. F.1
    "classic_td": dict(use_ln=0, init=TI, norm=0, opt=ag.OPT_ADAM, lr=3e-4),            # App. F.1 + B1/
    # Sensitivity arm (B1): classic TD that KEEPS LayerNorm, SparseInit, obs norm and reward scaling and swaps ONLY the
    # optimizer (ObGD -> Adam 3e-4 on -delta*z, App. F.1 lr/eps). Not a paper arm.
    "classic_td_keepnorm": dict(opt=ag.OPT_ADAM, lr=3e-4),
}


def lane_cfg(kind: str, arm: str, overrides: dict | None = None) -> dict:
    """Arm name = '<base arm>' or '<base arm>@k=v@k=v' (e.g. 'no_obgd@lr=3e-5' for the B3 lr sweep)."""
    base, arms = {"q": (Q_STREAM, Q_ARMS), "ac": (AC_STREAM, AC_ARMS), "td": (TD_STREAM, TD_ARMS)}[kind]
    name, *kv = arm.split("@")
    d = dict(base, **arms[name])
    for item in kv:
        k, v = item.split("=", 1)
        assert k in d, f"unknown lane field {k}"
        v = yaml.safe_load(v)
        d[k] = float(v) if isinstance(v, str) else v   # PyYAML reads '1e-4' (no dot) as a string; stack_lanes'
        # float32 cast parsed it correctly in every run so far (checked in the review pass), the dict now holds the number
    d.update((overrides or {}).get(arm, {}))
    return d


def stack_lanes(kind: str, cfgs: list[dict]):
    cls = {"q": ag.QLane, "ac": ag.ACLane, "td": ag.TDLane}[kind]
    ints = {"init", "opt"}
    return cls(**{f: jnp.asarray([c[f] for c in cfgs], jnp.int32 if f in ints else jnp.float32)
                  for f in cls._fields})


# ------------------------------------------------------------------ config
def load_config(path: str, overrides=()) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        k, v = ov.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "nogit"


# ------------------------------------------------------------------ checkpointing
def _save(ckpt_dir, carry, step, host):
    import orbax.checkpoint as ocp
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(ckpt_dir, f"s{step:010d}"))
    with ocp.StandardCheckpointer() as cp:
        cp.save(path, carry)
    np.savez(os.path.join(ckpt_dir, f"s{step:010d}_host.npz"), **host)
    for f in sorted(os.listdir(ckpt_dir)):
        if not f.startswith(f"s{step:010d}"):
            p = os.path.join(ckpt_dir, f)
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)


def _restore(ckpt_dir, carry_like):
    import orbax.checkpoint as ocp
    dirs = sorted(d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d)))
    if not dirs:
        return carry_like, 0, None
    last = dirs[-1]
    with ocp.StandardCheckpointer() as cp:
        carry = cp.restore(os.path.abspath(os.path.join(ckpt_dir, last)), carry_like)
    z = np.load(os.path.join(ckpt_dir, f"{last}_host.npz"), allow_pickle=False)
    return carry, int(last[1:]), {k: z[k] for k in z.files}


# ------------------------------------------------------------------ RL experiments (MinAtar Q, Brax AC)
def build(cfg: dict):
    kind = cfg["kind"]
    if kind == "q":
        env = dt.minatar_env(cfg["env"])
        # schedule_steps: the run length the epsilon schedule is defined on (official: exploration_fraction *
        # total_steps). Default = total_steps; full-scale sets 5e6 (the paper's run) and stops earlier, so a
        # truncated run is the paper's run observed up to total_steps (comparable to Figure 4 at that step).
        agent = ag.QAgent(env, int(cfg.get("schedule_steps", cfg["total_steps"])), gamma=cfg.get("gamma", 0.99),
                          overshoot_diag=cfg.get("overshoot_diag", True))
    elif kind == "ac":
        backend = cfg.get("brax_backend", dt.BRAX_BACKEND)
        if backend in dt.PROXY_GATE_FAILED and not cfg.get("allow_failed_proxy_gate", False):
            # This backend FAILS the passive-policy proxy gate (scripts/
            # hopper_passive_gate.py); set allow_failed_proxy_gate only to reproduce the original full-scale runs.
            raise ValueError(f"Brax backend {backend!r} failed the hopper passive-policy proxy gate ()")
        env = dt.brax_env(cfg["env"], backend)
        agent = ag.ACAgent(env, gamma=cfg.get("gamma", 0.99), episode_limit=cfg.get("episode_limit", 1000),
                           overshoot_diag=cfg.get("overshoot_diag", True))
    else:
        raise ValueError(kind)
    return env, agent


def run_rl(cfg: dict, tag: str, ckpt: bool = True, max_chunks: int | None = None) -> dict | None:
    kind, T, C = cfg["kind"], int(cfg["total_steps"]), int(cfg["chunk_steps"])
    assert T % C == 0, "total_steps must be a multiple of chunk_steps"
    env, agent = build(cfg)
    arms, seeds = list(cfg["arms"]), list(cfg["seeds"])
    arm_seeds = {a: list((cfg.get("arm_seeds") or {}).get(a, seeds)) for a in arms}   # optional per-arm seed list
    lanes = [(a, s) for a in arms for s in arm_seeds[a]]
    lane = stack_lanes(kind, [lane_cfg(kind, a, cfg.get("arm_overrides")) for a, _ in lanes])
    root = jax.random.PRNGKey(int(cfg.get("master_seed", 0)))
    run_keys = jnp.stack([jax.random.fold_in(root, s) for _, s in lanes])
    step_keys = jax.vmap(lambda k: jax.random.fold_in(k, 2))(run_keys)
    exp = f"{cfg['exp']}_{cfg['env']}"
    run_dir = os.path.join(REPO, "runs", tag, exp)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    info("build", exp=exp, kind=kind, lanes=len(lanes), arms=",".join(arms), seeds=len(seeds), total_steps=T,
         chunk=C, env=env.name, obs=str(env.obs_shape), n_act=env.n_actions)
    t = Timer()
    carry = jax.jit(jax.vmap(agent.init))(run_keys, lane)
    jax.block_until_ready(carry)
    info("init done", seconds=t.lap())
    zero_diag = ag.qdiag_zero() if kind == "q" else ag.acdiag_zero()
    dump = os.path.join(REPO, "runs", "nan_dumps")

    # nan_policy: 'abort' (default; any non-finite parameter in any lane dumps and exits) or 'mark' (full-scale: a lane
    # that diverges is recorded with its step and keeps running isolated -- lanes never interact -- so one diverging
    # ablation arm, e.g. the paper's own no-LayerNorm Hopper curve, cannot kill the other lanes of the program).
    nan_policy = str(cfg.get("nan_policy", "abort"))

    def chunk_fn(c, t0, ln, sk):
        c = c._replace(diag=zero_diag)
        c, outs = jax.lax.scan(lambda cc, i: agent.step(cc, t0 + i, ln, sk), c, jnp.arange(C, dtype=jnp.int32),
                                unroll=int(cfg.get("scan_unroll", 1)))
        ptree = {"p": c.params if kind == "q" else (c.pi, c.v)}
        if nan_policy == "abort":
            nan_guard(ptree, {"t0": t0}, tag=0, dump_dir=dump)
        ok = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(ptree)]))
        return c, (outs, ok)

    run_chunk = jax.jit(jax.vmap(chunk_fn, in_axes=(0, None, 0, 0)))
    done_steps, host = 0, None
    if ckpt and os.path.isdir(ckpt_dir):
        carry, done_steps, host = _restore(ckpt_dir, carry)
        if done_steps:
            info("resumed from checkpoint", steps_done=done_steps)
    L = len(lanes)
    ep = host["ep"].tolist() if host is not None else []          # rows: lane, end_step, return, length
    diags = host["diag"].tolist() if host is not None else []     # rows: lane, chunk_end_step, *diag fields
    compile_s, run_s, first = 0.0, 0.0, True
    diverged = {int(k): int(v) for k, v in (host["diverged"].tolist() if host is not None and "diverged" in host
                                            else [])}
    ck_every = int(cfg.get("ckpt_every_chunks", 10))
    n_chunks = 0
    while done_steps < T:
        t.lap()
        carry, ((rets, lens), ok) = run_chunk(carry, jnp.int32(done_steps), lane, step_keys)
        rets, lens, ok = np.asarray(rets), np.asarray(lens), np.asarray(ok)
        for i in np.nonzero(~ok)[0]:
            if i not in diverged:
                diverged[int(i)] = done_steps + C
                warn("lane diverged (non-finite parameters); marked, kept isolated", lane=int(i), arm=lanes[i][0],
                     seed=lanes[i][1], by_step=done_steps + C)
        dt_ = t.lap()
        if first:
            compile_s, first = dt_, False
            info("first chunk (compile + run)", seconds=dt_)
        else:
            run_s += dt_
        li, si = np.nonzero(~np.isnan(rets))
        ep.extend(np.stack([li, done_steps + si + 1, rets[li, si], lens[li, si]], 1).tolist())
        dg = jax.tree.map(np.asarray, carry.diag)
        done_steps += C
        n_chunks += 1
        for i in range(L):
            diags.append([i, done_steps] + [float(x[i]) for x in dg])
        epa = np.asarray(ep[-50000:]) if ep else np.zeros((0, 4))
        recent = []
        for ai, a in enumerate(arms):
            m = np.isin(epa[:, 0], [i for i, (aa, _) in enumerate(lanes) if aa == a]) & \
                (epa[:, 1] > done_steps - C) if len(epa) else np.zeros(0, bool)
            recent.append(f"{a}:{np.mean(epa[m, 2]) if m.any() else float('nan'):.3g}")
        metric("chunk", exp=exp, step=done_steps, sps_total=(C * L / dt_), ret_last_chunk=" ".join(recent),
               bound_active=float(np.mean(dg.n_active / np.maximum(dg.n, 1))) if kind == "q" else
               float(np.mean(dg.n_active_v / np.maximum(dg.n, 1))),
               overshoot=float(np.mean(dg.n_overshoot / np.maximum(dg.n, 1))), max_viol=float(np.max(dg.max_viol)))
        if float(np.max(dg.max_viol)) > 1e-4:
            warn("ObGD L1 bound violated (P1)", max_viol=float(np.max(dg.max_viol)))
        if ckpt and done_steps < T and n_chunks % ck_every == 0:
            _save(ckpt_dir, carry, done_steps, {"ep": np.asarray(ep, np.float64).reshape(-1, 4),
                                                "diag": np.asarray(diags, np.float64),
                                                "diverged": np.asarray(sorted(diverged.items()), np.int64).reshape(-1, 2)})
        if max_chunks is not None and n_chunks >= max_chunks and done_steps < T:
            if ckpt:
                _save(ckpt_dir, carry, done_steps, {"ep": np.asarray(ep, np.float64).reshape(-1, 4),
                                                    "diag": np.asarray(diags, np.float64),
                                                    "diverged": np.asarray(sorted(diverged.items()),
                                                                           np.int64).reshape(-1, 2)})
            warn("stopping early (max_chunks, simulated interruption)", steps_done=done_steps)
            return None
    steady_steps = T - C * (1 if compile_s else 0)
    sps = steady_steps * L / max(run_s, 1e-9)
    info("timing", exp=exp, compile_plus_first_chunk_s=compile_s, steady_seconds=run_s, sps_total_steady=sps,
         sps_per_lane=sps / L, device=str(jax.devices()[0]))
    ep = np.asarray(ep, np.float64).reshape(-1, 4)
    diags = np.asarray(diags, np.float64)
    dfields = list(zero_diag._fields)
    stride = int(cfg.get("bin_stride", 50_000))
    out = {}
    for i, (a, s) in enumerate(lanes):
        e = ep[ep[:, 0] == i][:, 1:]
        dd = diags[diags[:, 0] == i][:, 1:]
        summ = ev.summarize_run(e, T, stride)
        rec = dict(exp=exp, env=cfg["env"], env_impl=env.name, arm=a, seed=s, git=git_sha(),
                   config=cfg, lane=lane_cfg(kind, a, cfg.get("arm_overrides")), summary=summ,
                   diverged_by_step=diverged.get(i),
                   episodes_cols=["end_step", "return", "length"], episodes=e.tolist(),
                   diag_cols=["chunk_end_step"] + dfields, diag=dd.tolist(),
                   timing=dict(compile_plus_first_chunk_s=compile_s, steady_seconds=run_s, lanes_vmapped=L,
                               sps_per_lane=sps / L, device=str(jax.devices()[0])))
        write_json(os.path.join(run_dir, a, f"seed_{s}", "metrics.json"), rec)
        out.setdefault(a, {})[s] = summ
    for a in arms:
        fb = [out[a][s]["final_bin"] for s in arm_seeds[a]]
        m, se = ev.mean_se(fb)
        metric("final", exp=exp, arm=a, final_bin_mean=m, final_bin_se=se, n=len(fb))
    if ckpt and os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    return dict(exp=exp, compile_s=compile_s, steady_s=run_s, sps_total=sps, lanes=L,
                final={a: {s: out[a][s]["final_bin"] for s in arm_seeds[a]} for a in arms})


# ------------------------------------------------------------------ ETTm2 stream TD (E8)
def ema_traces(x: np.ndarray, beta: float) -> np.ndarray:
    """Official Trace: m <- beta m + (1-beta) x; return m / (1 - beta^count) (float64, bias-corrected)."""
    out = np.empty_like(x)
    m = np.zeros(x.shape[1])
    for n in range(x.shape[0]):
        m = beta * m + (1 - beta) * x[n]
        out[n] = m / (1 - beta ** (n + 1))
    return out


def run_td(cfg: dict, tag: str) -> dict:
    T = int(cfg["total_steps"])
    d = dt.load_ett()
    info("ETTm2", rows=d["rows"], sha256=d["sha256"], sha_ok=d["sha_ok"])
    if not d["sha_ok"]:
        warn("ETTm2 checksum differs from the recorded one", sha=d["sha256"])
    X = ema_traces(d["obs"][: T + 1], float(cfg.get("memory_beta", 0.999))).astype(np.float32)
    cum = d["cumulant"][2: T + 2].astype(np.float32)          # transition k (1..T) has cumulant c[k+1]
    done = np.zeros(T, bool)
    agent = ag.TDAgent(X.shape[1], gamma=cfg.get("gamma", 0.99))
    arms, seeds = list(cfg["arms"]), list(cfg["seeds"])
    lanes = [(a, s) for a in arms for s in seeds]
    lane = stack_lanes("td", [lane_cfg("td", a, cfg.get("arm_overrides")) for a, _ in lanes])
    root = jax.random.PRNGKey(int(cfg.get("master_seed", 0)))
    run_keys = jnp.stack([jax.random.fold_in(root, s) for _, s in lanes])
    t = Timer()
    carry = jax.jit(jax.vmap(agent.init, in_axes=(0, 0, None)))(run_keys, lane, jnp.asarray(X[0]))

    def run(c, ln):
        return jax.lax.scan(lambda cc, x: agent.step(cc, x[0], x[1], x[2], ln), c,
                            (jnp.asarray(X[1:]), jnp.asarray(cum), jnp.asarray(done)))

    carry, (pred, cumr) = jax.jit(jax.vmap(run))(carry, lane)
    pred, cumr = np.asarray(pred, np.float64), np.asarray(cumr, np.float64)
    secs = t.lap()
    info("timing", exp="e8_ett", seconds_incl_compile=secs, lanes=len(lanes))
    gamma = cfg.get("gamma", 0.99)
    run_dir = os.path.join(REPO, "runs", tag, "e8_ETTm2")
    res = {}
    os.makedirs(run_dir, exist_ok=True)   # full per-step series (metrics.json keeps every 10th): metrics recompute
    np.savez_compressed(os.path.join(run_dir, "series.npz"), pred=pred.astype(np.float32),
                        cum=cumr.astype(np.float32), lanes=np.asarray([f"{a}|{s}" for a, s in lanes]))
    for i, (a, s) in enumerate(lanes):
        G = ev.discounted_returns(cumr[i], gamma)
        summ = ev.ett_summary(pred[i], G, cumr[i])
        summ["n_active_frac"] = float(carry.n_active[i] / T)
        summ["overshoot_frac"] = float(carry.n_overshoot[i] / T)
        rec = dict(exp="e8_ETTm2", arm=a, seed=s, git=git_sha(), config=cfg, lane=lane_cfg("td", a), summary=summ,
                   data=dict(sha256=d["sha256"], rows=d["rows"]),
                   pred_every_10=pred[i, ::10].round(5).tolist(), G_every_10=G[::10].round(5).tolist())
        write_json(os.path.join(run_dir, a, f"seed_{s}", "metrics.json"), rec)
        res.setdefault(a, {})[s] = summ
    for a in arms:
        for k in ("mse_first", "mse_last", "mse_first_official", "mse_last_official"):
            m, se = ev.mean_se([res[a][s][k] for s in seeds])
            metric("final", exp="e8_ETTm2", arm=a, stat=k, mean=m, se=se)
    return dict(exp="e8_ETTm2", seconds=secs, res=res)


# ------------------------------------------------------------------ CLI
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tag", default="dev")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--no-ckpt", action="store_true")
    p.add_argument("--max-chunks", type=int, default=None)
    a = p.parse_args(argv)
    check_device()
    cfg = load_config(a.config, a.set)
    t0 = time.time()
    if cfg["kind"] == "td":
        r = run_td(cfg, a.tag)
    else:
        r = run_rl(cfg, a.tag, ckpt=not a.no_ckpt, max_chunks=a.max_chunks)
    info("done", wall_seconds=time.time() - t0)
    if r is not None:
        write_json(os.path.join(REPO, "runs", a.tag, f"index_{cfg['exp']}_{cfg.get('env', 'ett')}.json"), r)
    return r


if __name__ == "__main__":
    main()

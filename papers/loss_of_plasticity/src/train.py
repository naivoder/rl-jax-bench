"""Entry point: vmapped online-learning runs for Slowly-Changing Regression and Online Permuted MNIST.

    uv run python -m src.train --config src/configs/scr_b10.yaml [--set n_seeds=10 n_steps=300000 tag=g2]

One process runs every (seed, arm) pair of a config as one vmapped batch: outer vmap over seeds (each seed has its
own data stream / permutation sequence / init, shared by all arms = common random numbers), inner vmap over arms
(hyper-parameter scalars; arms differ ONLY in those).  Time is a lax.scan inside one jitted chunk function
(SCR: one 10k-example flip period per outer scan step; MNIST: one task of 60k examples); the Python loop over
chunks logs, checkpoints (orbax) and resumes.

PRNG key tree: root_s = PRNGKey(seed); k_data = fold_in(root_s, 0); k_target = fold_in(root_s, 1);
k_init = fold_in(root_s, 2); k_algo = fold_in(root_s, 3) -> per step fold_in(k_algo, global_step) (split into the
S&P noise key and fold_in(., layer) CBP keys); k_diag = fold_in(root_s, 4) (reserved: MNIST diagnostics use the
first 2000 examples of the upcoming task's order, as the official code does).

NaN/Inf guard: jaxlog.nan_guard (jax.debug.callback) on the end-of-chunk state and the chunk's summed losses; a
non-finite value in any step of the chunk propagates into those sums, so the guard catches it within one chunk
(a per-step host callback would cost more than the step itself).  ``nan_guard_every_step: true`` turns on the
per-step variant for debugging.
"""
from __future__ import annotations

import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from rl_jax_bench.jaxlog import Timer, check_device, info, metric, nan_guard, warn, write_json  # noqa: E402

from . import algorithm as A  # noqa: E402
from . import data as D  # noqa: E402
from . import diagnostics as G  # noqa: E402
from .model import ACT_ID, INIT_GAIN, REINIT_GAIN, init_mlp, make_model  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------------------------------- config
def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        k, v = ov.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


NON_SEMANTIC = ("fresh",)   # keys that do not change the computation


def cfg_hash(cfg) -> str:
    c = {k: v for k, v in cfg.items() if k not in NON_SEMANTIC}
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]


def seeds_of(cfg):
    if cfg.get("seeds") is not None:          # explicit list (e.g. sweep seeds + disjoint final seeds in one process)
        return [int(s) for s in cfg["seeds"]]
    s0 = int(cfg.get("seed_offset", 0))
    return list(range(s0, s0 + int(cfg["n_seeds"])))


def arm_hps(cfg):
    """List of (name, HP, init_gain) for the config's arms; arm keys override the config-level defaults."""
    base = {k: cfg[k] for k in ("lr", "wd", "sigma", "rho", "util", "mt", "eta", "bias_transfer", "act", "adam_rule")
            if k in cfg}
    out = []
    for arm in cfg["arms"]:
        a = dict(base, **arm)
        act = a.get("act", "relu")
        rule = a.get("adam_rule", "standard")
        if rule not in A.ADAM_RULES:
            raise ValueError(f"arm {arm['name']}: adam_rule {rule!r} not in {A.ADAM_RULES}")
        if rule != "standard" and cfg.get("opt", "sgd") != "adam":
            raise ValueError(f"arm {arm['name']}: adam_rule {rule!r} needs opt: adam")
        hp = A.make_hp(lr=a["lr"], wd=a.get("wd", 0.0), sigma=a.get("sigma", 0.0), rho=a.get("rho", 0.0),
                       util=a.get("util", "overall"), mt=a.get("mt", 100), eta=a.get("eta", 0.99),
                       bias_transfer=a.get("bias_transfer", 0.0), act=ACT_ID[act], reinit_gain=REINIT_GAIN[act],
                       adam_rule=rule)
        out.append((arm["name"], hp, INIT_GAIN[act]))
    return out


def static_of(cfg, n_in, n_out, loss):
    return A.Static(n_in=n_in, widths=tuple(cfg["widths"]), n_out=n_out, loss=loss, opt=cfg.get("opt", "sgd"),
                    replace_mode=cfg.get("replace_mode", "accumulate"), k_max=int(cfg.get("k_max", 1)),
                    beta1=float(cfg.get("beta1", 0.9)), beta2=float(cfg.get("beta2", 0.999)),
                    eps=float(cfg.get("eps", 1e-8)), track_wmag=bool(cfg.get("track_wmag", True)),
                    convention=cfg.get("convention", "C"), weight_reduction=cfg.get("weight_reduction", "mean"),
                    count_base=cfg.get("count_base", "eligible"), perturb=needs_noise(cfg))


def needs_noise(cfg) -> bool:
    """S&P noise is drawn only if some arm has sigma > 0 (skipping it is exact for sigma = 0: p + 0*N == p)."""
    if "perturb" in cfg:
        return bool(cfg["perturb"])
    return any(float(a.get("sigma", cfg.get("sigma", 0.0))) > 0 for a in cfg["arms"])


def check_kmax(cfg, hps, static):
    """k_max must cover rho * n_eligible (else replacements would be silently capped)."""
    for name, hp, _ in hps:
        for w in static.widths:
            need = float(hp.rho) * w
            if static.replace_mode == "accumulate" and need >= static.k_max:
                raise ValueError(f"arm {name}: rho*n={need} >= k_max={static.k_max}")
            if static.replace_mode == "bernoulli" and np.floor(need) > static.k_max:
                raise ValueError(f"arm {name}: floor(rho*n)={need} > k_max={static.k_max}")


def init_states(static, hps, seeds, n_in, n_out):
    hp = A.stack_hp([h for _, h, _ in hps])
    gains = jnp.asarray([g for _, _, g in hps], jnp.float32)

    def one(seed_key, gain):
        p = init_mlp(jax.random.fold_in(seed_key, 2), n_in, static.widths, n_out, gain)
        return A.init_state(static, p)

    roots = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds, jnp.uint32))
    states = jax.vmap(lambda r: jax.vmap(lambda g: one(r, g))(gains))(roots)
    return states, hp, roots


def vstep(static):
    """learner_step vmapped over arms (shared x, y, key) then over seeds (shared hp)."""
    f = lambda s, hp, x, y, k: A.learner_step(static, s, hp, x, y, k)
    return jax.vmap(jax.vmap(f, in_axes=(0, 0, None, None, None)), in_axes=(0, None, 0, 0, 0))


# ------------------------------------------------------------------------------------------- checkpoint
def ckpt_mgr(path):
    import orbax.checkpoint as ocp
    return ocp.CheckpointManager(os.path.abspath(path), options=ocp.CheckpointManagerOptions(max_to_keep=2, create=True))


def ckpt_save(mgr, step, tree):
    import orbax.checkpoint as ocp
    mgr.save(int(step), args=ocp.args.StandardSave(tree))
    mgr.wait_until_finished()


def ckpt_restore(mgr, template):
    import orbax.checkpoint as ocp
    step = mgr.latest_step()
    if step is None:
        return None
    return mgr.restore(step, args=ocp.args.StandardRestore(template))


def prepare_run_dir(cfg):
    out = os.path.join(REPO, "runs", cfg["tag"])
    os.makedirs(out, exist_ok=True)
    h = cfg_hash(cfg)
    hp_path = os.path.join(out, "config.sha256")
    if os.path.exists(hp_path) and open(hp_path).read().strip() != h:
        if cfg.get("fresh", False):
            import shutil
            shutil.rmtree(os.path.join(out, "ckpt"), ignore_errors=True)
        else:
            raise RuntimeError(f"{out} holds a run of a different config; pass --set fresh=true to restart")
    with open(hp_path, "w") as f:
        f.write(h)
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


# ------------------------------------------------------------------------------------------- SCR
def run_scr(cfg):
    spec = D.scr_spec(cfg.get("scr_mode", "flip_one"), bool(cfg.get("paper_text", False)))
    seeds = seeds_of(cfg)
    hps = arm_hps(cfg)
    static = static_of(cfg, spec.n_in, 1, "mse")
    check_kmax(cfg, hps, static)
    n_periods = int(cfg["n_steps"]) // spec.T
    chunk = int(cfg.get("periods_per_chunk", 50))
    out = prepare_run_dir(cfg)
    info("scr config", tag=cfg["tag"], seeds=len(seeds), arms=len(hps), n_periods=n_periods, mode=spec.mode,
         n_in=spec.n_in, opt=static.opt, replace_mode=static.replace_mode)
    states, hp, roots = init_states(static, hps, seeds, spec.n_in, 1)
    k_data = jax.vmap(lambda r: jax.random.fold_in(r, 0))(roots)
    k_tgt = jax.vmap(lambda r: jax.random.fold_in(r, 1))(roots)
    k_algo = jax.vmap(lambda r: jax.random.fold_in(r, 3))(roots)
    tgts = jax.vmap(lambda k: D.scr_target(k, spec))(k_tgt)
    bits0 = jax.vmap(lambda k: D.scr_bits0(k, spec))(k_data)
    step_fn = vstep(static)
    S, Aa = len(seeds), len(hps)
    every_step_guard = bool(cfg.get("nan_guard_every_step", False))

    def period(carry, p):
        st, bits = carry
        bits = jnp.where(p == 0, bits, jax.vmap(lambda k, b: D.scr_next_bits(k, p, b, spec))(k_data, bits))
        X = jax.vmap(lambda k, b: D.scr_period_inputs(k, p, b, spec))(k_data, bits)       # (S, T, n_in)
        Y = jax.vmap(lambda t, x: D.scr_target_apply(t, x)[0])(tgts, X)                  # (S, T)

        def one(st, t):
            g = p * spec.T + t
            keys = jax.vmap(lambda k: jax.random.fold_in(k, g))(k_algo)
            st, inf = step_fn(st, hp, X[:, t][:, None, :], Y[:, t][:, None, None], keys)
            if every_step_guard:
                nan_guard(inf["loss"], inf["loss"], tag=g, dump_dir=os.path.join(out, "nan_dumps"))
            return st, (inf["loss"], inf["n_rep"].sum(-1), inf["overflow"], inf.get("wmag", jnp.zeros((S, Aa))))

        st, (loss, nrep, ovf, wm) = jax.lax.scan(one, st, jnp.arange(spec.T))
        return (st, bits), {"err_sum": loss.sum(0), "err_sum_2nd_half": loss[spec.T // 2:].sum(0),
                            "n_rep": nrep.sum(0), "overflow": ovf.any(0), "wmag_end": wm[-1]}

    @jax.jit
    def run_chunk(st, bits, p0):
        (st, bits), m = jax.lax.scan(period, (st, bits), p0 + jnp.arange(chunk))
        nan_guard((st["params"], m["err_sum"]), {"p0": p0, "err": m["err_sum"]}, tag=0,
                  dump_dir=os.path.join(out, "nan_dumps"))
        return st, bits, m

    hist = {k: np.zeros((n_periods, S, Aa), np.float32) for k in ("err_sum", "err_sum_2nd_half", "n_rep", "wmag_end")}
    hist["overflow"] = np.zeros((n_periods, S, Aa), bool)
    mgr = ckpt_mgr(os.path.join(out, "ckpt"))
    p_done = 0
    tree = {"states": states, "bits": bits0, "p": jnp.asarray(0, jnp.int32), "hist": hist}
    restored = ckpt_restore(mgr, tree)
    if restored is not None:
        states, bits0, p_done = restored["states"], restored["bits"], int(restored["p"])
        hist = {k: np.asarray(v) for k, v in restored["hist"].items()}
        info("resumed", p=p_done)
    bits = bits0
    if n_periods % chunk:
        raise ValueError("n_steps must be a multiple of periods_per_chunk * T")
    timer = Timer()
    t_first = None
    while p_done < n_periods:
        states, bits, m = run_chunk(states, bits, jnp.asarray(p_done, jnp.int32))
        m = jax.device_get(m)
        dt = timer.lap()
        if t_first is None:
            t_first = dt
            info("first chunk (incl. compile)", seconds=dt)
        else:
            metric("throughput", run_steps_per_s=S * Aa * chunk * spec.T / dt, steps_per_s=chunk * spec.T / dt)
        for k in hist:
            hist[k][p_done:p_done + chunk] = m[k]
        p_done += chunk
        mse = m["err_sum"].sum(0) / (chunk * spec.T)   # (S, A)
        metric("scr chunk", step=p_done * spec.T, **{n: float(mse[:, i].mean()) for i, (n, _, _) in enumerate(hps[:8])})
        if hist["overflow"][:p_done].any():
            warn("replacement count exceeded k_max (capped)")
        ckpt_save(mgr, p_done, {"states": states, "bits": bits, "p": jnp.asarray(p_done, jnp.int32), "hist": hist})
    write_scr_metrics(cfg, out, seeds, hps, hist, spec, t_first)
    return out


def write_scr_metrics(cfg, out, seeds, hps, hist, spec, t_first):
    T = spec.T
    per_period = hist["err_sum"] / T                                  # (P, S, A)
    names = [n for n, _, _ in hps]
    for si, s in enumerate(seeds):
        d = {"experiment": cfg["tag"], "seed": s, "n_steps": int(per_period.shape[0] * T), "arms": {}}
        for ai, n in enumerate(names):
            pp = per_period[:, si, ai]
            d["arms"][n] = {
                "mse_per_10k": pp.tolist(),
                "mse_whole_run": float(pp.mean()),
                "n_replacements": int(hist["n_rep"][:, si, ai].sum()),
                "wmag_per_10k_end": hist["wmag_end"][:, si, ai].tolist(),
            }
        write_json(os.path.join(out, f"seed_{s}", "metrics.json"), d)
    summary = {"tag": cfg["tag"], "seeds": seeds, "arms": names, "first_chunk_seconds": t_first,
               "mse_whole_run_mean": {n: float(per_period[:, :, i].mean()) for i, n in enumerate(names)}}
    write_json(os.path.join(out, "summary.json"), summary)
    info("wrote metrics", dir=out)


# ------------------------------------------------------------------------------------------- MNIST
def run_mnist(cfg):
    seeds = seeds_of(cfg)
    hps = arm_hps(cfg)
    static = static_of(cfg, 784, 10, "ce")
    check_kmax(cfg, hps, static)
    n_tasks = int(cfg["n_tasks"])
    n_ex = int(cfg.get("examples_per_task", 60_000))
    n_diag = int(cfg.get("diag_n", 2000))
    out = prepare_run_dir(cfg)
    x_np, y_np, sums = D.load_mnist()
    info("mnist data", **{k: v["sha256_raw"][:16] for k, v in sums.items()})
    X = jnp.asarray(x_np)
    Yl = jnp.asarray(y_np)
    states, hp, roots = init_states(static, hps, seeds, 784, 10)
    k_data = jax.vmap(lambda r: jax.random.fold_in(r, 0))(roots)
    k_algo = jax.vmap(lambda r: jax.random.fold_in(r, 3))(roots)
    S, Aa = len(seeds), len(hps)
    step_fn = vstep(static)
    model = make_model(static.widths, 10)
    info("mnist config", tag=cfg["tag"], seeds=S, arms=Aa, tasks=n_tasks, widths=str(static.widths),
         examples_per_task=n_ex)

    @jax.jit
    def task_perm(task):
        return jax.vmap(lambda k: D.mnist_task(k, task, 60_000))(k_data)

    @jax.jit
    def diagnose(st, perm, order):
        def per_seed(st_s, perm_s, order_s):
            xs = X[order_s[:n_diag]][:, perm_s].astype(jnp.float32) / 255.0
            def per_arm(st_a, act):
                _, hs = model.apply(st_a["params"], xs, act)
                return G.layer_diagnostics(hs)
            return jax.vmap(per_arm)(st_s, hp.act)
        return jax.vmap(per_seed)(st, perm, order)

    @jax.jit
    def run_task(st, perm, order, task):
        def one(st, t):
            xs = jax.vmap(lambda p, o: X[o[t]][p].astype(jnp.float32) / 255.0)(perm, order)[:, None, :]
            ys = jax.vmap(lambda o: Yl[o[t]].astype(jnp.int32))(order)[:, None]
            g = task * n_ex + t
            keys = jax.vmap(lambda k: jax.random.fold_in(k, g))(k_algo)
            st, inf = step_fn(st, hp, xs, ys, keys)
            return st, (inf["correct"], inf["loss"], inf["n_rep"].sum(-1), inf.get("wmag", jnp.zeros((S, Aa))),
                        inf["overflow"])
        st, (c, l, nr, wm, ovf) = jax.lax.scan(one, st, jnp.arange(n_ex), unroll=int(cfg.get("scan_unroll", 1)))
        m = {"acc": c.mean(0), "loss": l.mean(0), "n_rep": nr.sum(0), "wmag_mean": wm.mean(0), "overflow": ovf.any(0),
             "acc_first_10k": c[:10_000].mean(0), "acc_last_10k": c[-10_000:].mean(0)}
        nan_guard((st["params"], m["loss"]), {"task": task, "loss": m["loss"]}, tag=0,
                  dump_dir=os.path.join(out, "nan_dumps"))
        return st, m

    keys_hist = ("acc", "loss", "n_rep", "wmag_mean", "acc_first_10k", "acc_last_10k", "dead_pct", "erank_scaled",
                 "wmag_before", "wmag_after")   # wmag_mean = per-step average (track_wmag); before/after = per task
    hist = {k: np.zeros((n_tasks, S, Aa), np.float32) for k in keys_hist}
    hist["dead_per_layer"] = np.zeros((n_tasks, S, Aa, len(static.widths)), np.float32)
    hist["erank_per_layer"] = np.zeros((n_tasks, S, Aa, len(static.widths)), np.float32)
    mgr = ckpt_mgr(os.path.join(out, "ckpt"))
    tree = {"states": states, "task": jnp.asarray(0, jnp.int32), "hist": hist}
    t_done = 0
    restored = ckpt_restore(mgr, tree)
    if restored is not None:
        states, t_done = restored["states"], int(restored["task"])
        hist = {k: np.asarray(v) for k, v in restored["hist"].items()}
        info("resumed", task=t_done)
    wmag_fn = jax.jit(jax.vmap(jax.vmap(lambda p: A.mean_abs_weight(static, p))))
    timer = Timer()
    t_first = None
    ckpt_every = int(cfg.get("ckpt_every_tasks", 5))
    while t_done < n_tasks:
        tk = jnp.asarray(t_done, jnp.int32)
        perm, order = task_perm(tk)
        dg = jax.device_get(diagnose(states, perm, order))
        hist["wmag_before"][t_done] = np.asarray(wmag_fn(states["params"]))
        states, m = run_task(states, perm, order, tk)
        m = jax.device_get(m)
        hist["wmag_after"][t_done] = np.asarray(wmag_fn(states["params"]))
        dt = timer.lap()
        if t_first is None:
            t_first = dt
            info("first task (incl. compile)", seconds=dt)
        else:
            metric("throughput", run_steps_per_s=S * Aa * n_ex / dt, steps_per_s=n_ex / dt)
        for k in ("acc", "loss", "n_rep", "wmag_mean", "acc_first_10k", "acc_last_10k"):
            hist[k][t_done] = m[k]
        for k in ("dead_pct", "erank_scaled", "dead_per_layer", "erank_per_layer"):
            hist[k][t_done] = dg[k]
        if np.asarray(m["overflow"]).any():
            warn("replacement count exceeded k_max (capped)", task=t_done)
        metric("mnist task", task=t_done + 1,
               **{f"acc_{n}": float(m["acc"][:, i].mean()) for i, (n, _, _) in enumerate(hps)},
               **{f"dead_{n}": float(dg["dead_pct"][:, i].mean()) for i, (n, _, _) in enumerate(hps)})
        t_done += 1
        if t_done % ckpt_every == 0 or t_done == n_tasks:
            ckpt_save(mgr, t_done, {"states": states, "task": jnp.asarray(t_done, jnp.int32), "hist": hist})
    write_mnist_metrics(cfg, out, seeds, hps, hist, t_first, sums)
    return out


def write_mnist_metrics(cfg, out, seeds, hps, hist, t_first, sums):
    names = [n for n, _, _ in hps]
    for si, s in enumerate(seeds):
        d = {"experiment": cfg["tag"], "seed": s, "n_tasks": int(hist["acc"].shape[0]), "mnist_checksums": sums,
             "arms": {}}
        for ai, n in enumerate(names):
            d["arms"][n] = {k: hist[k][:, si, ai].tolist() for k in hist}
        write_json(os.path.join(out, f"seed_{s}", "metrics.json"), d)
    summary = {"tag": cfg["tag"], "seeds": seeds, "arms": names, "first_task_seconds": t_first,
               "acc_per_task_mean": {n: hist["acc"][:, :, i].mean(1).tolist() for i, n in enumerate(names)}}
    write_json(os.path.join(out, "summary.json"), summary)
    info("wrote metrics", dir=out)


# ------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    cfg.setdefault("tag", os.path.splitext(os.path.basename(a.config))[0])
    check_device()
    t0 = time.time()
    out = run_scr(cfg) if cfg["experiment"] == "scr" else run_mnist(cfg)
    info("done", seconds=time.time() - t0, out=out)


if __name__ == "__main__":
    main()

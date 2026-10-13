"""Train one (env, method) cell: all runs vmapped in ONE jitted program; resumable (orbax); per-run metrics.json.

    python src/train.py --config src/configs/small_shopping5.yaml --method a2c_asym_hs
    python src/train.py --config ... --method ... --runs 5 --max-env-steps 1000000   # CLI overrides (logged)

Config resolution (every value traceable): src/configs/sources.yaml `common` (paper / official / ASSUMED tags) ->
`environments.<env>` -> `table2.<env>.<method>` (alpha_pi, alpha_V, lambda0; Table 2) -> experiment yaml
(`overrides`, each with its own reason comment) -> CLI flags.

PRNG key tree: root = PRNGKey(seed); run_key = fold_in(fold_in(fold_in(root, env_id), method_id),
run_idx); (init_key, loop_key) = split(run_key); init_key -> (actor, critic) params; update u: fold_in(loop_key, u)
-> split(E) episode keys -> (reset key, step key); step t: split(fold_in(step key, t)) -> (action key, env key).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

from rl_jax_bench import gpu_bootstrap  # noqa: E402

# Capture while-loop bodies into CUDA command buffers (graphs): the update is a chain of short sequential loops
# (rollout, GRU scans) whose per-iteration launch overhead dominates; this is 3-4x faster and bit-identical
# (checked: same parameters after 30 updates with and without the flag). CONDITIONAL was
# dropped after one "illegal memory access" abort with it enabled (small-scale runs).
_CB = "--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,WHILE" + (
    ",CONDITIONAL" if os.environ.get("UA_CMDBUF_CONDITIONAL") == "1" else "")   # ~2x faster again
if "xla_gpu_enable_command_buffer" not in os.environ.get("XLA_FLAGS", "") and not os.environ.get("UA_NO_CMDBUF"):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _CB).strip()
gpu_bootstrap.setup()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_jax_bench.jaxlog import Timer, check_device, error, info, metric, nan_guard, warn, write_json  # noqa: E402

from algorithm import init_run_state, make_update  # noqa: E402
from envs import heavenhell_probes, make_env  # noqa: E402
from model import METHODS, NetSpec, init_params  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_IDS = {"heavenhell_3": 0, "heavenhell_4": 1, "shopping_5": 2, "shopping_6": 3, "car_flag": 4, "cleaner": 5}
METHOD_IDS = {m: i for i, m in enumerate(METHODS)}


def resolve_config(exp_path: str, method: str, cli: dict | None = None) -> dict:
    src = yaml.safe_load(open(os.path.join(REPO, "src", "configs", "sources.yaml")))
    exp = yaml.safe_load(open(exp_path))
    env = exp["env"]
    c = src["common"]
    a_pi, a_v, lam0 = src["table2"][env][method]
    cfg = dict(
        env=env, method=method, tag=exp["tag"], seed=int(exp.get("seed", 0)),
        runs=int(exp.get("runs", c["runs_per_config"]["value"])),
        episodes_per_update=c["episodes_per_update"]["value"], cap=c["max_episode_steps"]["value"],
        gamma=float(src["environments"][env]["gamma"] if "gamma" in src["environments"][env] else c["gamma"]["value"]),
        target_period=c["target_update_period"]["value"], n_log=c["metric_log_points"]["value"],
        lambda_final_factor=c["negentropy_final_factor"]["value"], negentropy_nsteps=c["negentropy_nsteps"]["value"],
        adam_eps=c["adam_eps"]["value"], td=c["td_estimator"]["value"], td_lambda=c["td_lambda_diagnostic"]["value"],
        bootstrap_truncation=not c["truncation_is_terminal"]["value"], actor_uses_target=False,
        max_env_steps=int(_env_steps(src, env)), alpha_pi=a_pi, alpha_v=a_v, lambda0=lam0,
        length_buckets=[16, 32, 64],          # sources.yaml common.length_buckets (implementation detail, exact)
        chunk_updates=500,                    # ASSUMED run control: updates per jitted call (no effect on results)
        ckpt_minutes=10.0,                    # ASSUMED run control: orbax checkpoint period
        probes=env.startswith("heavenhell"),  # Figure 4 critic probes (our construction)
        # f32 matmuls as in the authors' fp32 code; None = pre-fix JAX default (TF32 on GPU)
        matmul_precision=c["matmul_precision"]["value"],
        run_offset=0,                         # run control: this process trains run_idx run_offset .. run_offset+runs-1
    )
    for k, v in (exp.get("overrides") or {}).items():
        if k not in cfg:
            raise KeyError(f"unknown override {k}")
        cfg[k] = v
    for k, v in (exp.get("method_overrides", {}).get(method) or {}).items():
        if k not in cfg:                      # same check as `overrides`
            raise KeyError(f"unknown method override {k} for {method}")
        cfg[k] = v
    for k, v in (cli or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg


def _env_steps(src, env):
    e = src["environments"][env]
    return e["max_env_steps"]["value"] if isinstance(e["max_env_steps"], dict) else e["max_env_steps"]


# keys added later enter the hash only when they differ from the pre-fix behavior, so runs stored before
# they were added keep their sha (resume check) and a pre-fix reproduction (matmul_precision: null) hashes the same
_LEGACY = {"matmul_precision": None, "run_offset": 0}


def config_hash(cfg: dict) -> str:
    free = ("chunk_updates", "ckpt_minutes")
    return hashlib.sha256(json.dumps({k: v for k, v in sorted(cfg.items()) if k not in free
                                      and not (k in _LEGACY and v == _LEGACY[k])}).encode()).hexdigest()


def out_dir(cfg):
    return os.path.join(REPO, "runs", cfg["tag"], cfg["env"], cfg["method"])


def build(cfg):
    envf = make_env(cfg["env"])
    if abs(envf.gamma - cfg["gamma"]) > 1e-12:
        raise ValueError(f"gamma {cfg['gamma']} != file discount {envf.gamma}")
    spec = NetSpec(envf.kind, envf.n_actions, envf.n_obs, envf.n_states)
    probes = None
    if cfg["probes"]:
        probes = {k: jnp.asarray(v) for k, v in heavenhell_probes(cfg["env"]).items()}
    update = make_update(envf, spec, cfg["method"], cfg, probes)
    root = jax.random.PRNGKey(cfg["seed"])
    mkey = jax.random.fold_in(jax.random.fold_in(root, ENV_IDS[cfg["env"]]), METHOD_IDS[cfg["method"]])
    n_probes = 0 if probes is None else int(probes["idx"].shape[0])

    def init_one(i):
        init_key, loop_key = jax.random.split(jax.random.fold_in(mkey, i))
        pa, pc = init_params(spec, cfg["method"], init_key)
        return init_run_state(pa, pc, loop_key, cfg, n_probes)

    rs0 = jax.jit(jax.vmap(init_one))(int(cfg.get("run_offset", 0)) + jnp.arange(cfg["runs"]))
    U = int(cfg["chunk_updates"])

    @jax.jit
    def chunk(rs):
        rs = jax.lax.scan(lambda r, _: (update(r), None), rs, None, length=U)[0]
        nan_guard({"pa": rs.pa, "pc": rs.pc}, {"t_env": rs.t_env, "upd": rs.upd}, tag=0,
                  dump_dir=os.path.join(out_dir(cfg), "nan_dumps"))
        return rs

    return rs0, chunk


def _ckpt(path):
    import orbax.checkpoint as ocp
    return ocp.CheckpointManager(os.path.abspath(path), options=ocp.CheckpointManagerOptions(max_to_keep=2,
                                                                                            create=True))


def run(cfg) -> dict:
    od = out_dir(cfg)
    os.makedirs(od, exist_ok=True)
    chash = config_hash(cfg)
    info("config", **{k: v for k, v in cfg.items() if not isinstance(v, (list, dict))}, sha=chash[:16])
    plat = check_device()
    # before 2026-10-12 this was left unset, which lets XLA use TF32 for larger f32 matmuls on Ampere
    # GPUs (scripts/tf32_probe.py; we measured critic-gradient elements off by up to 5e-2 relative). Default now
    # 'highest' (full f32, as the authors' code and the oracle tests); `matmul_precision: null` reproduces the old runs.
    if cfg.get("matmul_precision") is not None:
        jax.config.update("jax_default_matmul_precision", cfg["matmul_precision"])
    info("matmul precision", jax_default_matmul_precision=str(jax.config.jax_default_matmul_precision),
         note="default: TF32 permitted on GPU" if jax.config.jax_default_matmul_precision is None else "")
    tm = Timer()
    wall0 = time.time()
    rs, chunk = build(cfg)
    sfx = "" if int(cfg.get("run_offset", 0)) == 0 else f"_off{int(cfg['run_offset'])}"   # parallel run slices
    mgr = _ckpt(os.path.join(od, "ckpt" + sfx))
    hp = os.path.join(od, f"ckpt{sfx}.config.sha256")
    if mgr.latest_step() is not None:
        old = open(hp).read().strip() if os.path.exists(hp) else None
        if old != chash:
            raise RuntimeError(f"refusing to resume {od}: config sha {old} != {chash}")
        import orbax.checkpoint as ocp
        rs = mgr.restore(mgr.latest_step(), args=ocp.args.StandardRestore(rs))
        info("resumed", step=mgr.latest_step(), t_env_min=int(np.min(rs.t_env)))
    else:
        open(hp, "w").write(chash)
    T_max = cfg["max_env_steps"]
    rs = chunk(rs)
    jax.block_until_ready(rs)
    first = tm.lap()
    info("first chunk (compile + run)", seconds=round(first, 2), chunk_updates=cfg["chunk_updates"])
    t_ckpt = time.time()
    n_chunks, steady_s, steady_steps = 1, 0.0, 0
    prev_t = np.asarray(rs.t_env).copy()
    while True:
        t_now = np.asarray(rs.t_env)
        if bool(np.any(np.asarray(rs.bad))):
            error("non-finite loss flagged", runs=np.nonzero(np.asarray(rs.bad))[0].tolist())
            sys.exit(3)
        li = np.asarray(rs.log_idx)
        last = np.asarray(rs.logs["avg100"])[np.arange(cfg["runs"]), np.maximum(li - 1, 0)]
        metric("progress", t_env_mean=int(t_now.mean()), t_env_min=int(t_now.min()), frac=float(t_now.min() / T_max),
               avg100_mean=float(last.mean()), log_points=int(li.min()), elapsed_s=round(time.time() - wall0, 1))
        if np.all(t_now >= T_max):
            break
        if time.time() - t_ckpt > 60 * cfg["ckpt_minutes"]:
            import orbax.checkpoint as ocp
            mgr.save(int(n_chunks), args=ocp.args.StandardSave(rs))
            mgr.wait_until_finished()
            t_ckpt = time.time()
            info("checkpoint", chunk=n_chunks)
        t0 = time.time()
        rs = chunk(rs)
        jax.block_until_ready(rs)
        steady_s += time.time() - t0
        steady_steps += int(np.sum(np.asarray(rs.t_env) - prev_t))
        prev_t = np.asarray(rs.t_env).copy()
        n_chunks += 1
    thr = steady_steps / max(steady_s, 1e-9)
    metric("throughput", steady_env_steps_per_s_all_runs=round(thr, 1), steady_seconds=round(steady_s, 1),
           first_chunk_seconds=round(first, 2), platform=plat)
    res = write_metrics(cfg, rs, chash, dict(wall_s=time.time() - wall0, first_chunk_s=first, steady_s=steady_s, steady_env_steps=steady_steps,
                                             env_steps_per_s=thr, platform=plat))
    return res


def write_metrics(cfg, rs, chash, timing) -> dict:
    od = out_dir(cfg)
    logs = jax.tree.map(np.asarray, rs.logs)
    li = np.asarray(rs.log_idx)
    out = {}
    off = int(cfg.get("run_offset", 0))
    for r in range(cfg["runs"]):
        n = int(li[r])
        m = {k: logs[k][r, :n].tolist() for k in logs}
        avg = np.asarray(m["avg100"])
        rec = dict(env=cfg["env"], method=cfg["method"], tag=cfg["tag"], run_idx=off + r, seed=cfg["seed"],
                   config_sha256=chash, n_log_points=n, t_env_final=int(np.asarray(rs.t_env)[r]),
                   n_updates=int(np.asarray(rs.upd)[r]),
                   final=float(avg[-10:].mean()) if n else None, auc=float(avg.mean()) if n else None,
                   curve=m, timing=timing if r == 0 else None,
                   probe_names=["(a) map0 no-priest", "(b) map1 no-priest", "(c) map0 priest", "(d) map1 priest"]
                   if cfg["probes"] else None)
        write_json(os.path.join(od, f"seed_{off + r}", "metrics.json"), rec)
        out[r] = rec
    write_json(os.path.join(od, "config.json" if off == 0 else f"config_off{off}.json"), dict(cfg=cfg, config_sha256=chash))
    fin = [out[r]["final"] for r in out if out[r]["final"] is not None]
    metric("final", env=cfg["env"], method=cfg["method"], mean=float(np.mean(fin)),
           se=float(np.std(fin, ddof=1) / np.sqrt(len(fin))) if len(fin) > 1 else 0.0, n=len(fin))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--runs", type=int)
    ap.add_argument("--max-env-steps", type=int, dest="max_env_steps")
    ap.add_argument("--chunk-updates", type=int, dest="chunk_updates")
    ap.add_argument("--tag")
    ap.add_argument("--run-offset", type=int, dest="run_offset")
    a = ap.parse_args()
    cli = dict(runs=a.runs, max_env_steps=a.max_env_steps, chunk_updates=a.chunk_updates, tag=a.tag,
               run_offset=a.run_offset)
    if any(v is not None for v in cli.values()):
        warn("CLI overrides", **{k: v for k, v in cli.items() if v is not None})
    run(resolve_config(a.config, a.method, cli))


if __name__ == "__main__":
    main()

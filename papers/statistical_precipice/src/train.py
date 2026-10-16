"""E6 own-compute pool: N seeds x arms x tasks of the ppo_implementation_details reproduction's PPO, trained as one vmapped batch per (task, arm),
then evaluated AFTER training on 100 episodes per run (paper §3 protocol).

    python -m src.train --config src/configs/e6.yaml --tag small --set seeds='[1,...,20]'

The PPO is vendored byte-identically in src/ppo8/ (sha256 checked at start against the ppo_implementation_details reproduction's files); this file only drives it: it builds ``ppo8.train.PPO`` for each config, runs the same jitted chunk
loop as the ppo_implementation_details reproduction's ``train_one`` (orbax checkpoint every chunk, resume, NaN guard inside the vendored update,
compile vs steady-state timing), and adds the post-training evaluation that the ppo_implementation_details reproduction did not have.

Arms differ ONLY in the keys of ``ppo8.train.ARMS[arm]`` (separate: architecture=separate; shared:
architecture=shared; advnorm_batch: norm_adv_level=batch).

PRNG key tree the README: the ppo_implementation_details reproduction's tree unchanged for training
    run_key = fold_in(PRNGKey(master_seed), seed); fold_in(run_key, 0..3) = init / env / act / perm
plus, for evaluation (disjoint branch, never used in training):
    k_eval = fold_in(run_key, 7) -> split(n_eval_episodes) -> per episode: reset with the episode key,
             step t uses fold_in(fold_in(ep_key, 1), t) for the env and fold_in(fold_in(ep_key, 2), t) for the action.
The statistics (src/algorithm.py, float64) are never imported here: this process stays float32 like the ppo_implementation_details reproduction.
"""
from __future__ import annotations

import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
from functools import partial  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from rl_jax_bench.jaxlog import Timer, check_device, info, metric, warn, write_json  # noqa: E402

from .ppo8 import eval as p8eval  # noqa: E402
from .ppo8 import model as p8model  # noqa: E402
from .ppo8 import train as p8  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PPO8_SRC = os.environ.get("PPO8_SRC", os.path.join(REPO, "external", "ppo_implementation_details", "src"))
PPO8_COMMIT = "19d3c41bf6756f50c88101f9bc8915c0e085446d"
VENDORED = ("__init__.py", "algorithm.py", "data.py", "model.py", "train.py", "eval.py")


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def vendored_hashes(check_against_source: bool = True) -> dict:
    """sha256 of the vendored PPO files; if the ppo_implementation_details source exists, assert byte identity."""
    out = {}
    for f in VENDORED + ("classic_control.yaml",):
        ours = os.path.join(REPO, "src", "ppo8", f)
        out[f] = _sha(ours)
        src = os.path.join(PPO8_SRC, "configs" if f.endswith(".yaml") else "", f)
        if check_against_source and os.path.exists(src) and _sha(src) != out[f]:
            raise RuntimeError(f"vendored {f} differs from the ppo_implementation_details reproduction ({src})")
    return out


def load_e6_config(path: str, overrides=()) -> dict:
    with open(path) as f:
        e6 = yaml.safe_load(f)
    base = p8.load_config(os.path.join(REPO, "src", "ppo8", "classic_control.yaml"))
    base.update(e6.get("ppo_overrides") or {})
    base = p8.load_config_overrides(base, [o for o in overrides if not o.startswith("e6.")])
    for o in overrides:
        if o.startswith("e6."):
            k, v = o[3:].split("=", 1)
            e6[k] = yaml.safe_load(v)
    return base, e6


# ------------------------------------------------------------------ post-training evaluation
def evaluate(ppo: p8.PPO, runner, seeds, n_episodes: int, sampled: bool, master_seed: int):
    """(S, n_episodes) undiscounted returns; sampled=True draws actions from the policy, else argmax."""
    env, spec, net = ppo.env, ppo.spec, ppo.net
    T = spec.max_steps

    def one_episode(params, vs, ep_key):
        o, s = env.reset_one(jax.random.fold_in(ep_key, 0))

        def body(c, t):
            s, o, ret, alive = c
            dist, _ = net.apply(params, env.normalize_frozen(vs, o[None])[0])
            a = p8model.sample(jax.random.fold_in(jax.random.fold_in(ep_key, 2), t), dist, ppo.cont) if sampled \
                else p8model.mode(dist, ppo.cont)
            o2, s2, r, term, trunc, _ = env.step_one(jax.random.fold_in(jax.random.fold_in(ep_key, 1), t), s,
                                                     env._prep_action(a))
            ret = ret + alive * r
            alive = alive * (1.0 - jnp.logical_or(term, trunc).astype(jnp.float32))
            return (s2, o2, ret, alive), None

        (_, _, ret, alive), _ = jax.lax.scan(body, (s, o, jnp.float32(0.0), jnp.float32(1.0)), jnp.arange(T))
        return ret, alive

    def per_seed(params, vs, seed):
        run_key = jax.random.fold_in(jax.random.PRNGKey(master_seed), seed)
        keys = jax.random.split(jax.random.fold_in(run_key, 7), n_episodes)
        return jax.vmap(lambda k: one_episode(params, vs, k))(keys)

    ret, alive = jax.jit(jax.vmap(per_seed))(runner.params, runner.vs, jnp.asarray(seeds, jnp.int32))
    if float(np.asarray(alive).max()) > 0:
        warn("an evaluation episode did not terminate within max_steps", n=int(np.asarray(alive).sum()))
    return np.asarray(ret)


def train_and_eval(cfg: dict, e6: dict, arm: str, out_root: str) -> dict:
    """the ppo_implementation_details reproduction's chunk loop (same compile/ckpt/resume semantics) followed by the post-training evaluation."""
    seeds = list(cfg["seeds"])
    ppo = p8.PPO(cfg)
    tag = f"{cfg['env_id']}/{arm}"
    info("build", tag=tag, seeds=len(seeds), updates=ppo.U, batch=ppo.batch)
    t = Timer()
    runner, consts = jax.jit(jax.vmap(ppo.init))(jnp.asarray(seeds, jnp.int32))
    jax.block_until_ready(runner)
    run_dir = os.path.join(out_root, cfg["env_id"], arm)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    done_u, chunks = 0, []
    if os.path.isdir(ckpt_dir):
        runner, done_u, chunks = p8._restore(ckpt_dir, runner)
        if done_u:
            info("resumed from checkpoint", tag=tag, updates_done=done_u)
    compiled, compile_s, run_s, C = {}, 0.0, 0.0, cfg["chunk_updates"]
    while done_u < ppo.U:
        n = min(C, ppo.U - done_u)
        if n not in compiled:
            t.lap()
            compiled[n] = jax.jit(jax.vmap(partial(ppo.run_chunk, n=n))).lower(runner, consts).compile()
            dt = t.lap()
            compile_s += dt
            info("compiled chunk", tag=tag, n_updates=n, compile_seconds=dt)
        t.lap()
        runner, m = compiled[n](runner, consts)
        m = jax.tree.map(np.asarray, jax.device_get(m))
        dt = t.lap()
        run_s += dt
        chunks.append(m)
        done_u += n
        cnt = m["ep_cnt"].sum(axis=(1, 2))
        rec = np.where(cnt > 0, m["ep_ret_sum"].sum(axis=(1, 2)) / np.maximum(cnt, 1), np.nan)
        metric("chunk", tag=tag, global_step=done_u * ppo.batch, ep_return_mean=float(np.nanmean(rec)),
               sps_total=n * ppo.batch * len(seeds) / dt)
        if done_u < ppo.U:
            p8._save(ckpt_dir, runner, done_u, chunks)
    allm = {k: np.concatenate([c[k] for c in chunks], axis=1) for k in chunks[0]}
    sps = ppo.U * ppo.batch * len(seeds) / max(run_s, 1e-9)
    info("timing", tag=tag, compile_seconds=compile_s, steady_seconds=run_s, sps_total_steady=sps)
    t.lap()
    ev_s = evaluate(ppo, runner, seeds, e6["eval_episodes"], True, cfg["master_seed"])
    ev_g = evaluate(ppo, runner, seeds, e6["eval_episodes"], False, cfg["master_seed"])
    eval_s = t.lap()
    info("eval done", tag=tag, seconds=eval_s, sampled_mean=float(ev_s.mean()), greedy_mean=float(ev_g.mean()))
    gsteps = p8.global_steps(cfg, ppo.U)
    recs = {}
    for i, s in enumerate(seeds):
        cnt = allm["ep_cnt"][i]
        nz = cnt > 0
        episodes = np.stack([gsteps[nz], allm["ep_ret_sum"][i][nz], cnt[nz], allm["ep_len_sum"][i][nz]], 1)
        summ = p8eval.summarize(episodes, cfg["total_timesteps"], cfg["env_id"])
        summ.pop("curve_50bins", None)
        summ.update(eval_sampled_mean=float(ev_s[i].mean()), eval_greedy_mean=float(ev_g[i].mean()),
                    eval_episodes=int(e6["eval_episodes"]))
        rec = dict(env_id=cfg["env_id"], arm=arm, seed=s, summary=summ,
                   eval_sampled_returns=ev_s[i].tolist(), eval_greedy_returns=ev_g[i].tolist(),
                   timing=dict(compile_seconds=compile_s, steady_seconds=run_s, eval_seconds=eval_s,
                               n_seeds_vmapped=len(seeds), device=str(jax.devices()[0])))
        write_json(os.path.join(run_dir, f"seed_{s}", "metrics.json"), rec)
        recs[s] = rec
    metric("final", tag=tag, eval_sampled_mean=float(ev_s.mean()),
           eval_sampled_se=float(ev_s.mean(1).std(ddof=1) / np.sqrt(len(seeds))), eval_greedy_mean=float(ev_g.mean()))
    if os.path.isdir(ckpt_dir):
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    return dict(tag=tag, records=recs, compile_seconds=compile_s, steady_seconds=run_s, eval_seconds=eval_s)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(REPO, "src", "configs", "e6.yaml"))
    p.add_argument("--tag", default="dev")
    p.add_argument("--set", action="append", default=[], help="PPO key=value, or e6.<key>=value")
    p.add_argument("--skip-done", action="store_true", help="reuse (task, arm) pairs whose every seed file exists")
    a = p.parse_args(argv)
    check_device()
    hashes = vendored_hashes()
    info("vendored ppo8 byte-identical to the ppo_implementation_details reproduction", commit=PPO8_COMMIT)
    base, e6 = load_e6_config(a.config, a.set)
    out_root = os.path.join(REPO, "runs", a.tag)
    T = Timer()
    per_seed: dict[int, dict] = {}
    timing = {}
    for env_id in e6["tasks"]:
        for arm in e6["arms"]:
            cfg = p8.load_config_overrides(dict(base, env_id=env_id, **p8.ARMS[arm]),
                                           [o for o in a.set if not o.startswith("e6.")])
            done = [os.path.join(out_root, env_id, arm, f"seed_{sd}", "metrics.json") for sd in cfg["seeds"]]
            if a.skip_done and all(os.path.exists(f) for f in done):
                import json
                recs = {sd: json.load(open(f)) for sd, f in zip(cfg["seeds"], done)}
                info("skip (already complete)", tag=f"{env_id}/{arm}")
                r = dict(tag=f"{env_id}/{arm}", records=recs, compile_seconds=float("nan"), steady_seconds=float("nan"),
                         eval_seconds=float("nan"))
            else:
                r = train_and_eval(cfg, e6, arm, out_root)
            timing[r["tag"]] = dict(compile=r["compile_seconds"], steady=r["steady_seconds"], eval=r["eval_seconds"])
            for s, rec in r["records"].items():
                per_seed.setdefault(s, {})[f"{env_id}/{arm}"] = rec["summary"]
    wall = T.lap()
    for s, d in per_seed.items():   # runs/<tag>/seed_<n>/metrics.json: every (task, arm) of one seed
        write_json(os.path.join(out_root, f"seed_{s}", "metrics.json"), dict(seed=s, tag=a.tag, runs=d))
    write_json(os.path.join(out_root, "index.json"),
               dict(tag=a.tag, e6=e6, ppo_base=base, vendored_sha256=hashes, ppo8_commit=PPO8_COMMIT,
                    wall_seconds=wall, gpu_hours_upper_bound=wall / 3600.0, timing=timing,
                    device=str(jax.devices()[0])))
    info("done", wall_seconds=wall, gpu_hours_upper_bound=wall / 3600.0)


if __name__ == "__main__":
    main()

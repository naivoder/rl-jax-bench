"""Training loop: one process, all replications (seeds x same-graph variants) vmapped on one device.

Usage:
    python -m src.train --config pendulum_v.yaml --set algo=sac --set cell=lstm --set arch=separate \
        --seeds 0-11 --out runs/e1/sac-lstm-200-oar/separate
    python -m src.train --config pendulum_v.yaml --set arch=shared --variants official,noleak,detach ...

Schedule (official learner.train; README): ``num_init_rollouts`` random-action episodes; if the update
count is a ratio (float) do int(ratio * env_steps) updates; then ``num_iters`` x [``rollouts_per_iter`` exploratory
episodes, then ceil(ratio * new_env_steps) (or the integer ``updates_per_iter``) gradient steps]; evaluate when
(env_steps // (rollouts_per_iter * H)) % log_interval == 0.  All randomness from the key tree in data.py, indexed
by global episode / update / eval counters held in the state, so a resumed run reproduces the uninterrupted one.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from functools import partial

from . import common  # noqa: F401  (GPU bootstrap before jax)
from .common import Timer, check_device, git_dirty, git_sha, info, load_config, metric, nan_guard, warn, write_json

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from . import data as D  # noqa: E402
from .algorithm import Agent  # noqa: E402
from .envs import make_env  # noqa: E402
from .model import Nets, n_params  # noqa: E402

VARIANTS = {"official": (1.0, 1.0), "noleak": (0.0, 1.0), "detach": (0.0, 0.0)}  # (leak, enc) for arch=shared


class Runner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.env = make_env(cfg)
        self.nets = Nets(cfg, self.env.obs_dim, self.env.act_dim, self.env.discrete)
        self.agent = Agent(cfg, self.nets)
        e = self.env
        self.H = e.H
        self.n_init = int(cfg["num_init_rollouts"])
        self.n_iters = int(cfg["num_iters"])
        self.rpi = int(cfg["rollouts_per_iter"])
        self.n_ep_total = self.n_init + self.n_iters * self.rpi
        self.L = int(cfg["sampled_seq_len"]) if int(cfg["sampled_seq_len"]) > 0 else self.H
        self.B = int(cfg["batch_size"])
        self.meta = e.n_train_tasks > 0
        self.n_eval_eps = int(cfg["num_eval_episodes"]) if not self.meta else e.n_eval_tasks
        self.seed_base = int(cfg.get("seed_base", 0))

    # ------------------------------------------------------------------ state
    def init_state(self, seed_idx, coef):
        rk = D.run_key(self.seed_base, seed_idx)
        e = self.env
        learner = self.agent.init(D.sub(rk, D.K_INIT))
        return {"learner": learner,
                "buf": D.init_buffer(self.n_ep_total, self.H, e.obs_dim, e.act_dim),
                "tasks": e.make_tasks(D.sub(rk, D.K_TASK)),
                "n_ep": jnp.zeros((), jnp.int32), "n_upd": jnp.zeros((), jnp.int32),
                "n_eval": jnp.zeros((), jnp.int32), "rk": rk,
                "coef": {"leak": jnp.asarray(coef[0], jnp.float32), "enc": jnp.asarray(coef[1], jnp.float32)}}

    # ------------------------------------------------------------------ episodes
    def run_episodes(self, learner, tasks, ekeys, xkey, mode: str):
        """n parallel episodes of H steps. ekeys (n,) per-episode env keys; xkey: exploration key (batch)."""
        env, ag = self.env, self.agent
        n = tasks.shape[0]
        state, obs = jax.vmap(env.reset)(jax.vmap(lambda k: jax.random.fold_in(k, 0))(ekeys), tasks)
        A = env.act_dim
        carry0 = ag.init_carry(n) if ag.arch != "markov" else jnp.zeros((n, 1))

        def step(c, t):
            state, obs, rnn, prev_a, r = c
            kx = jax.random.fold_in(xkey, t)
            if mode == "random":
                a = jax.vmap(env.warmup_action)(jax.random.split(kx, n))
            else:
                a, rnn = ag.act(learner, rnn, prev_a, r, obs, kx, deterministic=(mode == "det"))
            skeys = jax.vmap(lambda k: jax.random.fold_in(k, t + 1))(ekeys)
            state, obs2, rew, done, term = jax.vmap(env.step)(skeys, state, tasks, a)
            out = {"obs": obs, "act": a, "rew": rew[:, None], "term": term[:, None]}
            return (state, obs2, rnn, a, rew[:, None]), out

        c0 = (state, obs, carry0, jnp.zeros((n, A)), jnp.zeros((n, 1)))
        (state, obs_last, _, _, _), tr = jax.lax.scan(step, c0, jnp.arange(self.H))
        sw = lambda x: jnp.swapaxes(x, 0, 1)  # noqa: E731
        ep = {"obs": jnp.concatenate([sw(tr["obs"]), obs_last[:, None]], 1), "act": sw(tr["act"]),
              "rew": sw(tr["rew"]), "term": sw(tr["term"]), "length": jnp.full((n,), self.H, jnp.int32)}
        return ep

    def _collect(self, ts, n: int, mode: str):
        rk = ts["rk"]
        idx = ts["n_ep"] + jnp.arange(n)
        ekeys = jax.vmap(lambda i: jax.random.fold_in(D.sub(rk, D.K_ENV), i))(idx)
        if self.meta:
            tr = ts["tasks"]["train"]
            choice = jax.vmap(lambda k: jax.random.randint(jax.random.fold_in(k, 999_999), (), 0, tr.shape[0]))(ekeys)
            tasks = tr[choice]
        else:
            tasks = jnp.zeros((n, 0), jnp.float32)
        xkey = jax.random.fold_in(D.sub(rk, D.K_EXPL), ts["n_ep"])
        ep = self.run_episodes(ts["learner"], tasks, ekeys, xkey, mode)
        ts = {**ts, "buf": D.add_episodes(ts["buf"], ep), "n_ep": ts["n_ep"] + n}
        return ts, jnp.mean(jnp.sum(ep["rew"][..., 0], -1))

    def _evaluate(self, ts):
        rk = ts["rk"]
        k = jax.random.fold_in(D.sub(rk, D.K_EVAL), ts["n_eval"])
        n = self.n_eval_eps
        ekeys = jax.vmap(lambda i: jax.random.fold_in(k, i))(jnp.arange(n))
        tasks = ts["tasks"]["eval"] if self.meta else jnp.zeros((n, 0), jnp.float32)
        ep = self.run_episodes(ts["learner"], tasks, ekeys, jax.random.fold_in(k, 10 ** 6), "det")
        ret = jnp.sum(ep["rew"][..., 0], -1)                     # (n,) total (meta-)episode return
        ts = {**ts, "n_eval": ts["n_eval"] + 1}
        return ts, {"return": jnp.mean(ret), "return_std": jnp.std(ret)}

    # ------------------------------------------------------------------ updates
    def _one_update(self, ts, diag: bool):
        rk, u = ts["rk"], ts["n_upd"]
        kr = jax.random.fold_in(D.sub(rk, D.K_REPL), u)
        ku = jax.random.fold_in(D.sub(rk, D.K_UPD), u)
        if self.agent.arch == "markov":
            Bm = int(self.cfg.get("markov_batch_size", 256))
            b = D.sample_transitions(ts["buf"], kr, Bm)
            noise = self.agent.draw_noise(ku, None, Bm)
        else:
            b = D.sample_sequences(ts["buf"], kr, self.B, self.L)
            noise = self.agent.draw_noise(ku, self.L + 1, self.B)
        learner, inf = self.agent.update(ts["learner"], b, noise, ts["coef"], diag)
        return {**ts, "learner": learner, "n_upd": u + 1}, inf

    def _train_chunk(self, ts, K: int):
        def body(c, _):
            c, inf = self._one_update(c, False)
            return c, inf
        ts, infs = jax.lax.scan(body, ts, None, length=K - 1)
        ts, dinf = self._one_update(ts, True)
        out = {k: jnp.mean(jnp.concatenate([infs[k], dinf[k][None]])) for k in infs}
        out.update({k: v for k, v in dinf.items() if k not in infs})
        nan_guard({"info": out, "params": ts["learner"]["params"]}, {"info": out}, tag=1,
                  dump_dir=self.cfg.get("nan_dump_dir", "nan_dumps"))
        return ts, out

    # ------------------------------------------------------------------ jitted, vmapped entry points
    def compile(self):
        V = lambda f: jax.jit(jax.vmap(f), donate_argnums=0)  # noqa: E731
        self.j_collect_random = V(partial(self._collect, n=self.n_init, mode="random"))
        self.j_collect = V(partial(self._collect, n=self.rpi, mode="explore"))
        self.j_eval = V(self._evaluate)
        self._chunks = {}

    def j_chunk(self, K):
        if K not in self._chunks:
            self._chunks[K] = jax.jit(jax.vmap(partial(self._train_chunk, K=K)), donate_argnums=0)
        return self._chunks[K]


# ---------------------------------------------------------------------------------------------- driver
def parse_seeds(s: str):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def chunk_sizes(n: int, K: int):
    return [K] * (n // K) + ([n % K] if n % K else [])


def tree_index(tree, i):
    return jax.tree_util.tree_map(lambda x: x[i], tree)


def save_ckpt(path, ts, host):
    import orbax.checkpoint as ocp
    os.makedirs(path, exist_ok=True)
    tmp = os.path.join(path, "state_tmp")
    final = os.path.join(path, "state")
    ck = ocp.StandardCheckpointer()
    if os.path.exists(tmp):
        import shutil
        shutil.rmtree(tmp)
    ck.save(tmp, ts)
    ck.wait_until_finished()
    if os.path.exists(final):
        import shutil
        shutil.rmtree(final)
    os.replace(tmp, final)
    write_json(os.path.join(path, "host.json"), host)


def load_ckpt(path, ts_like):
    import orbax.checkpoint as ocp
    final = os.path.join(path, "state")
    if not os.path.exists(final) or not os.path.exists(os.path.join(path, "host.json")):
        return None, None
    ck = ocp.StandardCheckpointer()
    abstract = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, ts_like)
    ts = ck.restore(final, abstract)
    with open(os.path.join(path, "host.json")) as f:
        host = json.load(f)
    return ts, host


def final_metric(steps, rets, total_steps):
    """mean eval metric over evaluations with env_step >= 0.8 * T (Appendix C, 'last 20%')."""
    steps, rets = np.asarray(steps), np.asarray(rets)
    sel = steps >= 0.8 * total_steps
    return float(rets[sel].mean()) if sel.any() else float("nan")


def train(cfg: dict, seeds, variants, out: str, resume=True, max_iters=None):
    t_all = Timer()
    plat = check_device()
    R = Runner(cfg)
    names = [v for v in variants]
    reps = [(s, v) for v in names for s in seeds]
    seed_arr = jnp.array([s for s, _ in reps])
    coef = jnp.array([VARIANTS.get(v, (1.0, 1.0)) for _, v in reps], jnp.float32)
    info("runner", env=R.env.name, arch=cfg["arch"], algo=cfg["algo"], cell=cfg["cell"], inputs=cfg["inputs"],
         L=R.L, B=R.B, H=R.H, n_reps=len(reps), variants=",".join(names), seeds=",".join(map(str, seeds)),
         platform=plat)
    R.compile()
    ts = jax.jit(jax.vmap(R.init_state))(seed_arr, coef)
    p0 = tree_index(ts["learner"]["params"], 0)
    info("params", n=n_params(p0))
    total_steps = R.H * (R.n_init + R.n_iters * R.rpi)
    n_iters_run = R.n_iters if max_iters is None else min(R.n_iters, int(max_iters))
    ratio = cfg.get("update_ratio")
    K = int(cfg.get("chunk", 200))
    ckpt_dir = os.path.join(os.path.abspath(out), "ckpt")   # orbax requires absolute paths
    host = {"it": -1, "n_env": 0, "evals": [], "train": [], "timing": {}}
    if resume:
        rts, rhost = load_ckpt(ckpt_dir, ts)
        if rts is not None:
            ts, host = rts, rhost
            info("resumed", it=host["it"], n_env=host["n_env"])
    timing = host["timing"]

    def do_updates(ts, n):
        infs = []
        for k in chunk_sizes(n, K):
            ts, inf = R.j_chunk(k)(ts)
            infs.append(inf)
        return ts, infs

    def log_train(it, n_env, infs):
        if not infs:
            return
        last = jax.device_get(infs[-1])
        means = {k: np.mean([np.asarray(i[k]) for i in jax.device_get(infs)], 0).tolist()
                 for k in infs[0] if not k.startswith(("off_", "clean_"))}
        diag = {k: np.asarray(v).tolist() for k, v in last.items() if k.startswith(("off_", "clean_"))}
        host["train"].append({"it": it, "n_env": n_env, **means, **diag})

    def do_eval(ts, n_env):
        ts, ev = R.j_eval(ts)
        ev = jax.device_get(ev)
        host["evals"].append({"n_env": n_env, "return": ev["return"].tolist()})
        for v in names:
            idx = [i for i, (_, vv) in enumerate(reps) if vv == v]
            r = np.asarray(ev["return"])[idx]
            metric("eval", variant=v, n_env=n_env, mean=float(r.mean()), min=float(r.min()), max=float(r.max()))
        return ts

    if host["it"] < 0:
        t = Timer()
        ts, _ = R.j_collect_random(ts)
        jax.block_until_ready(ts["n_ep"])
        timing["compile_collect_random_s"] = t.lap()
        n_env = R.n_init * R.H
        if ratio is not None and float(ratio) > 0:
            ts, infs = do_updates(ts, int(n_env * float(ratio)))
            jax.block_until_ready(ts["n_upd"])
            timing["init_updates_incl_compile_s"] = t.lap()
            log_train(-1, n_env, infs)
        host["it"], host["n_env"] = 0, n_env
    it0 = host["it"]
    n_env = host["n_env"]
    t_steady, steady_iters, steady_updates, t_session = 0.0, 0, 0, 0.0
    ck_every = int(cfg.get("ckpt_every_iters", 0))
    for it in range(it0, n_iters_run):
        t = Timer()
        ts, _ = R.j_collect(ts)
        n_new = R.rpi * R.H
        n_env += n_new
        nu = int(cfg["updates_per_iter"]) if ratio is None else int(math.ceil(float(ratio) * n_new))
        ts, infs = do_updates(ts, nu)
        log_train(it, n_env, infs)
        cur = n_env // (R.rpi * R.H)
        if cur % int(cfg["log_interval"]) == 0:
            ts = do_eval(ts, n_env)
        jax.block_until_ready(ts["n_upd"])
        dt = t.lap()
        if it == it0:
            timing["first_iter_incl_compile_s"] = dt
        else:
            t_steady += dt
            t_session += dt
            steady_iters += 1
            steady_updates += nu
        host["it"], host["n_env"] = it + 1, n_env
        if it % 25 == 0 or it == n_iters_run - 1:
            info("iter", it=it, n_env=n_env, s_per_iter=round(dt, 3),
                 upd_per_s=round(nu / dt, 1) if dt > 0 else -1)
        if ck_every and (it + 1) % ck_every == 0 and it + 1 < n_iters_run:
            timing["steady_s"] = timing.get("steady_s", 0.0) + t_steady
            t_steady = 0.0
            host["timing"] = timing
            save_ckpt(ckpt_dir, ts, host)
    timing["steady_s"] = timing.get("steady_s", 0.0) + t_steady
    if steady_iters:
        timing["steady_s_per_iter_this_session"] = t_session / steady_iters
        timing["steady_updates_per_s_this_session"] = steady_updates / max(t_session, 1e-9)
    timing["wall_s_this_session"] = t_all.lap()
    host["timing"] = timing
    total_done = R.H * (R.n_init + n_iters_run * R.rpi)
    write_outputs(cfg, out, reps, host, total_steps, total_done, n_iters_run)
    if ck_every:
        save_ckpt(ckpt_dir, ts, host)
    return ts, host, R


def write_outputs(cfg, out, reps, host, total_steps, total_done, n_iters_run):
    steps = [e["n_env"] for e in host["evals"]]
    # The paper's metric is the mean over the last 20 % of the PAPER's horizon (App. C).  A run shorter than that
    # (config num_iters override or --max-iters) is scored over the last 20 % of what was run and the key says so.
    p_it = cfg.get("paper_num_iters")
    paper_steps = None
    if p_it is not None:
        n_init, rpi = int(cfg["num_init_rollouts"]), int(cfg["rollouts_per_iter"])
        paper_steps = total_done // (n_init + n_iters_run * rpi) * (n_init + int(p_it) * rpi)
    metric_key = "final_last20_of_run" if (paper_steps is not None and total_done < paper_steps) else "final_last20"
    rets = np.array([e["return"] for e in host["evals"]]) if host["evals"] else np.zeros((0, len(reps)))
    summary = {}
    for i, (s, v) in enumerate(reps):
        curve = rets[:, i].tolist() if len(rets) else []
        fm = final_metric(steps, curve, total_done) if curve else float("nan")
        tr = [{k: (x[k][i] if isinstance(x[k], list) else x[k]) for k in x} for x in host["train"]]
        m = {"seed": s, "variant": v, "arch": cfg["arch"], "algo": cfg["algo"], "cell": cfg["cell"],
             "inputs": cfg["inputs"], "env": cfg["env"], "sampled_seq_len": cfg["sampled_seq_len"],
             "total_env_steps_planned": total_steps, "total_env_steps_run": total_done,
             "truncated": total_done < total_steps, "iters_run": n_iters_run,
             metric_key: fm, "final_metric_key": metric_key, "paper_horizon_env_steps": paper_steps,
             "horizon_fraction": (total_done / paper_steps) if paper_steps else None, "eval_env_steps": steps, "eval_returns": curve, "train": tr,
             "timing": host["timing"], "git_sha": git_sha(), "git_dirty": git_dirty(), "config": cfg}
        write_json(os.path.join(out, v, f"seed_{s}", "metrics.json"), m)
        summary.setdefault(v, {})[s] = fm
    for v, d in summary.items():
        vals = np.array(list(d.values()))
        se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else float("nan")
        metric(metric_key, variant=v, mean=float(vals.mean()), se=float(se), n=len(vals))
    write_json(os.path.join(out, "summary.json"), {metric_key: summary, "final_metric_key": metric_key,
                                                   "timing": host["timing"], "total_env_steps_run": total_done,
                                                   "paper_horizon_env_steps": paper_steps})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--seeds", default="0-3")
    ap.add_argument("--variants", default=None, help="shared only: official,noleak,detach")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-iters", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    common.set_precision(str(cfg.get("matmul_precision", "highest")))
    if a.variants:
        variants = a.variants.split(",")
    else:
        variants = ["official"] if cfg["arch"] == "shared" else [cfg["arch"]]
    if cfg["arch"] != "shared":
        assert variants == [cfg["arch"]], "variants only apply to arch=shared"
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({"cfg": cfg, "seeds": parse_seeds(a.seeds), "variants": variants, "argv": argv}, f, indent=1)
    t0 = time.time()
    train(cfg, parse_seeds(a.seeds), variants, a.out, resume=not a.no_resume, max_iters=a.max_iters)
    info("done", wall_s=round(time.time() - t0, 1))


if __name__ == "__main__":
    main()

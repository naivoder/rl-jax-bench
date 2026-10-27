"""Entry point: one game, a set of arms x seeds, all vmapped in one process.

    python -m src.train --game Breakout --seeds 10 --steps 1e6 --out runs/main

Arms (src/algorithm.py; they differ ONLY in traced hyper-parameters): JSRL JSRL-Random PDD DQfD BCinit-FT IMIT.
Stage 1: pre-training (DQfD loss for DQfD, cross-entropy for IMIT; BCinit-FT starts from IMIT's weights) on the guide's demo set.
Stage 2: ONE vmapped interaction batch over all interacting arms x seeds; every `chunk` = steps/100 env steps one evaluation point
(pi_e ALONE at eps 0.01) and, for the curriculum arm, the beta rule.  IMIT is evaluated at the same 100 points without
any online update.
Key tree: root = PRNGKey(top_seed); learner seed s -> K = fold_in(root, 100 + s), split(K, 5) =
[init, pretrain, env_reset, interact, eval] (the earlier DQfD reproduction's layout, so PDD is differentially testable against it), switch key =
fold_in(K, 1005).  The SAME seed key is used by every arm (common random numbers).
Resumable: orbax checkpoints every --ckpt-every interaction steps; `--resume` continues from the latest one.
"""
from __future__ import annotations

import os, sys
from rl_jax_bench import gpu_bootstrap  # noqa: E402
gpu_bootstrap.setup()  # BEFORE jax is imported

import argparse, json, subprocess, time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402
import yaml  # noqa: E402

from rl_jax_bench import jaxlog  # noqa: E402
from . import algorithm as A, data as D, replay as R  # noqa: E402
from .envs import MinAtar  # noqa: E402
from .guide import make_guide_fn  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cfg(path=None):
    with open(path or os.path.join(REPO, "src/configs/default.yaml")) as f:
        return yaml.safe_load(f)


def seed_keys(top_seed: int, seeds: list[int]):
    """(S, 5, 2) keys [init, pretrain, env_reset, interact, eval] and (S, 2) switch keys per learner seed (shared across arms)."""
    root = jax.random.PRNGKey(top_seed)
    ks = [jax.random.fold_in(root, 100 + s) for s in seeds]
    return jnp.stack([jax.random.split(k, 5) for k in ks]), jnp.stack([jax.random.fold_in(k, 1005) for k in ks])


def tree_bytes(t):
    return sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(t))


class Group:
    """A vmapped batch of runs = arms x seeds (arm-major, seed-minor)."""

    def __init__(self, env, cfg, demo, guide_fn, arms, seeds, top_seed, steps, pretrain_steps, ckpt_dir, tag, h1):
        self.env, self.cfg, self.arms, self.seeds, self.tag = env, cfg, arms, seeds, tag
        self.S, self.nA = len(seeds), len(arms)
        self.Rn = self.S * self.nA
        self.steps, self.pretrain_steps = steps, pretrain_steps
        nd = int(demo["act"].shape[0])
        obs_bytes = int(demo["obs"].shape[1])
        self.L = R.make_layout(nd, cfg["per"]["agent_capacity"], obs_bytes, cfg["dqfd"]["n_step"], cfg["env"]["gamma"],
                               cfg["per"]["alpha"], cfg["per"]["block_size"])
        self.chunk = steps // cfg["run"]["n_eval_points"]
        assert self.chunk * cfg["run"]["n_eval_points"] == steps, "steps must be a multiple of n_eval_points"
        self.fns = A.Fns(env, self.L, cfg, steps, guide_fn, self.chunk)
        self.demo = D.demo_to_buffer_arrays(demo)
        self.demo_np = demo
        self.hp = A.stack_hp(arms, self.S, cfg, h1)
        sk, swk = seed_keys(top_seed, seeds)
        self.rk, self.swk = jnp.tile(sk, (self.nA, 1, 1)), jnp.tile(swk, (self.nA, 1))
        self.ckpt_dir, self.mgr = ckpt_dir, {}
        self._build()

    def _mgr(self, name):
        if name not in self.mgr:
            path = os.path.abspath(os.path.join(self.ckpt_dir, self.tag, name))
            self.mgr[name] = ocp.CheckpointManager(path, options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True))
        return self.mgr[name]

    def _build(self):
        f, L, demo = self.fns, self.L, self.demo
        dump = os.path.join(REPO, "runs", "nan_dumps")

        def init_one(hp, keys):
            params = f.init_params(keys[0])
            return dict(params=params, tparams=params, opt=f.tx.init(params), buf=R.init_buffer(L, demo, hp), cnt=jnp.zeros((), jnp.int32))

        self.init = jax.jit(jax.vmap(init_one))
        self.pre_fn = {}

        def step_one(c, hp, key, ekey, train_on):
            return f.chunk_and_eval(c, hp, key, ekey, train_on)

        def step_all(c, hp, keys, ekeys, train_on):
            c2 = jax.vmap(step_one)(c, hp, keys, ekeys, train_on)
            jaxlog.nan_guard(c2["params"], jnp.zeros(()), tag=2, dump_dir=dump)
            return c2

        self.step = jax.jit(step_all, donate_argnums=0)
        self.init_int = jax.jit(jax.vmap(lambda s, k, sw, hp: f.init_interact(s["params"], s["params"], s["opt"], s["buf"], k, sw, hp)))
        self.ev0 = jax.jit(jax.vmap(f.eval_point0))
        self.evaluate = jax.jit(jax.vmap(lambda p, h, k: f.evaluate(p, h, k)))
        self.agree = jax.jit(jax.vmap(lambda p: f.demo_agreement(p, self.demo["obs"], self.demo["act"])))
        self.preflight = jax.jit(jax.vmap(lambda p: f.preflight(p, self.demo["obs"], self.demo["act"])))

    def pretrain_fn(self, n):
        if n not in self.pre_fn:
            f = self.fns

            def all_(s, hp, ks):
                s2, dg = jax.vmap(lambda a, b, c: f.pretrain_chunk(a, b, c, n))(s, hp, ks)
                jaxlog.nan_guard(s2["params"], jnp.zeros(()), tag=1, dump_dir=os.path.join(REPO, "runs", "nan_dumps"))
                return s2, dg

            self.pre_fn[n] = jax.jit(all_, donate_argnums=0)
        return self.pre_fn[n]

    # ------------------------------------------------------------------ phases
    def run_pretrain(self, resume, pre_chunk=5000):
        state = self.init(self.hp, self.rk)
        done = 0
        m = self._mgr("pre")
        if resume and m.latest_step() is not None:
            done = m.latest_step()
            state = m.restore(done, args=ocp.args.StandardRestore(state))
            jaxlog.info("resumed pretrain", step=done)
        hist, first, t_steady = [], True, None
        while done < self.pretrain_steps:
            n = min(pre_chunk, self.pretrain_steps - done)
            keys = jax.vmap(lambda k: jax.random.fold_in(k[1], done))(self.rk)
            tc = time.time()
            state, dg = self.pretrain_fn(n)(state, self.hp, keys)
            jax.block_until_ready(state["params"])
            if first:
                jaxlog.info("pretrain first chunk (compile + run)", secs=time.time() - tc, runs=self.Rn); first = False; t_steady = time.time(); d0 = done + n
            done += n
            hist.append(dict(step=done, loss=np.asarray(dg["loss"]).tolist()))
            if done % 25000 == 0 or done == self.pretrain_steps:
                ag = np.asarray(self.agree(state["params"]))
                hist[-1]["agreement"] = ag.tolist()
                jaxlog.metric("pretrain", step=done, loss=float(np.mean(dg["loss"])), agreement=float(ag.mean()))
            if done % 50000 == 0 and done < self.pretrain_steps:
                m.save(done, args=ocp.args.StandardSave(state)); m.wait_until_finished()
        if not first and done > d0:
            jaxlog.info("pretrain steady throughput", updates_per_s=(done - d0) / max(time.time() - t_steady, 1e-9), runs=self.Rn)
        return state, hist

    def hl_eval0(self):
        """Per-run h of the evaluation-point-0 combined episodes (stage 0 for the curriculum arm)."""
        return jnp.where(self.hp["jsrl_mode"] == 1, self.hp["h_list"][:, 0], 0.0).astype(jnp.int32)

    def run_interact(self, state, resume, ckpt_every):
        sk_env = jax.vmap(lambda k: k[2])(self.rk)
        c = self.init_int(state, sk_env, self.swk, self.hp)
        c = dict(c, tparams=jax.tree_util.tree_map(jnp.copy, c["params"]))   # target <- online at the start of interaction
        done = 0
        m = self._mgr("int")
        if resume and m.latest_step() is not None:
            done = m.latest_step()
            c = m.restore(done, args=ocp.args.StandardRestore(c))
            jaxlog.info("resumed interaction", step=done)
        else:
            c = self.ev0(c, self.hp, jax.vmap(lambda k: jax.random.fold_in(k[4], 0))(self.rk))   # evaluation point 0
        train_on = jnp.ones(self.Rn, bool)
        hist, first, t_steady = [], True, None
        j = done // self.chunk
        every = max(1, (self.steps // self.chunk) // 20)
        self.pf_mid = None                   # warm-start pre-flight at the END OF EARLY (evaluation point n_eval/10)
        probe_j = max(1, int(round(0.10 * (self.steps // self.chunk))))
        while done < self.steps:
            keys = jax.vmap(lambda k: jax.random.fold_in(k[3], j))(self.rk)
            ekeys = jax.vmap(lambda k: jax.random.fold_in(k[4], j + 1))(self.rk)
            tc = time.time()
            c = self.step(c, self.hp, keys, ekeys, train_on)
            jax.block_until_ready(c["t"])
            if first:
                jaxlog.info("interact first chunk (compile + run)", secs=time.time() - tc); first = False; t_steady = time.time(); d_steady = done + self.chunk
            done += self.chunk; j += 1
            if j == probe_j:
                self.pf_mid = {k: np.asarray(v) for k, v in self.preflight(c["params"]).items()}
            dg = {k: np.asarray(v) for k, v in c["dg"].items()}
            hist.append(dict(step=done, loss=(dg["loss"] / np.maximum(dg["did"], 1)).tolist(), qmax=(dg["qmax"] / np.maximum(dg["did"], 1)).tolist(),
                             guide_share=(dg["guide"] / dg["steps"]).tolist(), eps=(dg["eps"] / dg["steps"]).tolist(),
                             stage=np.asarray(c["ev_stage"][:, j]).tolist()))
            if j % every == 0 or done >= self.steps:
                ra = np.asarray(c["ev_alone"][:, j]).reshape(self.nA, self.S)
                st = np.asarray(c["ev_stage"][:, j]).reshape(self.nA, self.S)
                sps = (done - d_steady) / max(time.time() - t_steady, 1e-9) if done > d_steady else float("nan")
                jaxlog.metric("interact", step=done, steps_per_s=sps, run_steps_per_s=sps * self.Rn,
                              **{f"R_{a}": float(v) for a, v in zip(self.arms, ra.mean(1))}, **{f"stage_{a}": float(v) for a, v in zip(self.arms, st.mean(1)) if a.startswith("JSRL")})
            if ckpt_every and done % ckpt_every == 0 and done < self.steps:
                m.save(done, args=ocp.args.StandardSave(c)); m.wait_until_finished()
        return c, hist


def git_head():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_game(game, demo_dir, sticky=0.0):
    """sticky > 0: the env repeats the previous effective action with this probability (original MinAtar default 0.1) and the
    guide statistics / stage list come from data/demos/<game>_guide_sticky<p>.json (scripts/guide_stats.py --sticky p); the demo FILE is the same
    (used only by DQfD / IMIT / BCinit-FT, which a sticky run does not fly; PDD / JSRL / JSRL-Random read it for the buffer layout only)."""
    gname = f"{game}_guide.json" if sticky == 0 else f"{game}_guide_sticky{sticky:g}.json"
    gi = json.load(open(os.path.join(demo_dir, gname)))
    info = json.load(open(os.path.join(demo_dir, f"{game}_demos.json")))
    demo = D.load_demos(os.path.join(demo_dir, gi["demo_file"]), gi["demo_sha256"])
    env = MinAtar(game, max_steps=gi["max_steps"], sticky_action_prob=sticky)
    assert env.asterix_spawn_repair == (game == "Asterix")
    return env, gi, demo


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--arms", default=",".join(A.ARM_NAMES)); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--top-seed", type=int, default=0)
    ap.add_argument("--steps", type=float, default=None); ap.add_argument("--pretrain", type=int, default=None)
    ap.add_argument("--demo-dir", default=os.path.join(REPO, "data", "demos"))
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "dev")); ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], help="override section.key=value (recorded in metrics meta)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--sticky", type=float, default=0.0, help="sticky_action_prob of the training/eval env (default 0 = the default env bit for bit)")
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    for kv in a.set:
        k, v = kv.split("=", 1); sec, key = k.split(".")
        cfg[sec][key] = yaml.safe_load(v)
    steps = int(a.steps or cfg["run"]["interaction_steps"])
    pre = int(a.pretrain if a.pretrain is not None else cfg["dqfd"]["pretrain_steps"])
    arms = a.arms.split(","); seeds = list(range(a.seed0, a.seed0 + a.seeds))
    ckpt_every = cfg["run"]["checkpoint_every"] if a.ckpt_every is None else a.ckpt_every
    jaxlog.check_device()
    env, gi, demo = load_game(a.game, a.demo_dir, a.sticky)
    guide_fn = make_guide_fn(env, cfg, cfg["jsrl"]["guide_eps"])
    h1 = gi["mean_len"]
    ckpt_dir = a.ckpt_dir or os.path.join(a.out, "_ckpt")
    tag = a.tag or a.game
    jaxlog.info("config", game=a.game, arms=",".join(arms), seeds=len(seeds), steps=steps, pretrain=pre, n_demo=int(demo["act"].shape[0]),
                demo_sha=demo["sha256"][:12], h1=h1, stages=gi["stage_list"], guide_true_mean=gi["true_mean"])
    meta = dict(game=a.game, sticky=a.sticky, steps=steps, pretrain_steps=pre, seeds=seeds, top_seed=a.top_seed, guide=gi, demo_sha256=demo["sha256"],
                n_eval_points=cfg["run"]["n_eval_points"], overrides=a.set, jax=jax.__version__, devices=str(jax.devices()), git_head=git_head(),
                cfg=cfg)
    pre_arms = [x for x in arms if A.base_arm(x) in A.PRETRAIN_ARMS]
    int_arms = [x for x in arms if A.base_arm(x) not in A.NO_INTERACTION_ARMS]
    need_imit_weights = [x for x in int_arms if A.base_arm(x) == "BCinit-FT"]
    if need_imit_weights and "IMIT" not in pre_arms:
        pre_arms.append("IMIT")
    t_all = time.time()
    history, state_pre, hist_pre, gp, imit_ev = {}, None, [], None, None
    if pre_arms:
        gp = Group(env, cfg, demo, guide_fn, pre_arms, seeds, a.top_seed, steps, pre, ckpt_dir, f"{tag}_pre", h1)
        jaxlog.info("pretrain group", arms=",".join(pre_arms), runs=gp.Rn)
        state_pre, hist_pre = gp.run_pretrain(a.resume)
        if "IMIT" in arms:   # IMIT: evaluated at the same 100 points with the same eval keys, never updated
            ii = pre_arms.index("IMIT") * gp.S
            sl = slice(ii, ii + gp.S)
            p_im = jax.tree_util.tree_map(lambda x: x[sl], state_pre["params"])
            h0 = jnp.zeros(gp.S, jnp.int32)
            rows = {k: [] for k in ("alone", "alone_sd", "thr", "greedy")}
            t0 = time.time()
            for j in range(gp.fns.n_eval + 1):
                ev = gp.evaluate(p_im, h0, jax.vmap(lambda k: jax.random.fold_in(k[4], j))(gp.rk[sl]))
                for k in rows:
                    rows[k].append(np.asarray(ev[k]))
            imit_ev = {k: np.stack(v, 1) for k, v in rows.items()}
            jaxlog.metric("imitation", mean_final=float(imit_ev["alone"][:, -3:].mean()), mean_all=float(imit_ev["alone"].mean()), secs=time.time() - t0)
    result = {}
    if int_arms:
        g = Group(env, cfg, demo, guide_fn, int_arms, seeds, a.top_seed, steps, pre, ckpt_dir, f"{tag}_int", h1)
        jaxlog.info("interaction group", arms=",".join(int_arms), runs=g.Rn, chunk=g.chunk,
                    buffer_MB_per_run=tree_bytes(jax.eval_shape(lambda: R.init_buffer(g.L, g.demo, {k: v[0] for k, v in g.hp.items()}))) / 1e6)
        state = g.init(g.hp, g.rk)                       # fresh init (arms without pre-training) ...
        for ai, arm in enumerate(int_arms):              # ... pre-trained state spliced in for the arms that have it
            sl_i = slice(ai * g.S, (ai + 1) * g.S)
            if A.base_arm(arm) == "DQfD":                # whole state (buffer with the demo priorities), as the earlier DQfD reproduction
                pj = pre_arms.index(arm); sl_p = slice(pj * g.S, (pj + 1) * g.S)
                state = jax.tree_util.tree_map(lambda full, part: full.at[sl_i].set(part[sl_p]), state, state_pre)
            elif A.base_arm(arm) == "BCinit-FT":         # weights + optimiser of the CE clone; EMPTY replay with no demos (policy only)
                pj = pre_arms.index("IMIT"); sl_p = slice(pj * g.S, (pj + 1) * g.S)
                for key in ("params", "tparams", "opt"):
                    state[key] = jax.tree_util.tree_map(lambda full, part: full.at[sl_i].set(part[sl_p]), state[key], state_pre[key])
        pf0 = {k: np.asarray(v) for k, v in g.preflight(state["params"]).items()}
        c, ihist = g.run_interact(state, a.resume, ckpt_every)
        pf1 = {k: np.asarray(v) for k, v in g.preflight(c["params"]).items()}
        for k0 in ("alive_conv", "alive_v", "alive_a", "agreement"):
            jaxlog.metric("preflight", key=k0, **{f"{ar}_start": float(pf0[k0][i * g.S:(i + 1) * g.S].mean()) for i, ar in enumerate(int_arms)},
                          **{f"{ar}_end": float(pf1[k0][i * g.S:(i + 1) * g.S].mean()) for i, ar in enumerate(int_arms)})
        host = {k: np.asarray(c[k]) for k in ("bin_sum", "bin_cnt", "ev_alone", "ev_alone_sd", "ev_thr", "ev_greedy", "ev_stage", "ev_gsteps", "ev_cap", "guide_steps", "ep_count")}
        for ai, arm in enumerate(int_arms):
            sl = slice(ai * g.S, (ai + 1) * g.S)
            result[arm] = {k: v[sl] for k, v in host.items()}
            result[arm].update({f"pf_start_{k}": v[sl] for k, v in pf0.items()}); result[arm].update({f"pf_end_{k}": v[sl] for k, v in pf1.items()})
            if g.pf_mid is not None:
                result[arm].update({f"pf_mid_{k}": v[sl] for k, v in g.pf_mid.items()})
            history[arm] = [{k: (v[sl] if isinstance(v, list) and len(v) == g.Rn else v) for k, v in row.items()} for row in ihist]
    if imit_ev is not None:
        result["IMIT"] = dict(ev_alone=imit_ev["alone"], ev_alone_sd=imit_ev["alone_sd"], ev_thr=imit_ev["thr"], ev_greedy=imit_ev["greedy"])
    write_outputs(a.out, seeds, arms, a.game, result, history, meta, hist_pre, pre_arms)
    jaxlog.info("finished", secs=time.time() - t_all)


def write_outputs(out_dir, seeds, arms, game, result, history, meta, hist_pre, pre_arms):
    """runs/<out>/seed_<n>/metrics.json: per game, per arm, the 101 evaluation points (index 0 = before training) of R_alone (GRADED),
    R_thr (combined policy at the current stage), R_greedy (eps 0), the stage log, cumulative guide steps and per-bin training returns."""
    for si, s in enumerate(seeds):
        d = os.path.join(out_dir, f"seed_{s}")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "metrics.json")
        cur = json.load(open(p)) if os.path.exists(p) else {}
        cell = cur.setdefault(game, {})
        cell["meta"] = meta
        for arm, r in result.items():
            cell[arm] = {k: (np.asarray(v[si]).tolist() if np.asarray(v).ndim >= 1 else float(v)) for k, v in r.items()}
            if arm in history:
                cell[arm]["history"] = [{k: (v[si] if isinstance(v, list) else v) for k, v in row.items()} for row in history[arm]]
            st = np.asarray(r.get("ev_stage", [[0]])[si]) if "ev_stage" in r else None
            if st is not None and A.base_arm(arm) == "JSRL":
                n = meta["cfg"]["jsrl"]["n_stages"]
                hit = np.nonzero(st >= n)[0]
                cell[arm]["stage_reached"] = int(st.max())
                cell[arm]["first_eval_at_h0"] = int(hit[0]) if len(hit) else None
        jaxlog.write_json(p, cur)


if __name__ == "__main__":
    main()

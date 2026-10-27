"""Entry point: one (game, demonstrator) cell, a set of arms x seeds vmapped in one process per group.

    python -m src.train --game Breakout --demonstrator subopt --arms DQfD,PDD,... --seeds 10 --steps 5e6 --out runs/x

Groups: arms with a pre-training phase (DQfD, NoMargin, NoNstep, ADET, IMIT) share one vmapped batch; arms without
(PDD, PDDconst, RBS, HER) another.  Key tree: root = PRNGKey(top_seed); learner seed s uses
fold_in(root, 100 + s) split into {init, pretrain, env_reset, interact, eval}; the SAME seed key is used by every arm
(common random numbers).  Resumable: orbax checkpoints every `checkpoint_every` interaction steps and at the end of
pre-training; `--resume` continues from the latest one.
"""
from __future__ import annotations

import os, sys
from rl_jax_bench import gpu_bootstrap  # noqa: E402
gpu_bootstrap.setup()  # BEFORE jax is imported

import argparse, json, time  # noqa: E402
from functools import partial  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402
import yaml  # noqa: E402

from rl_jax_bench import jaxlog  # noqa: E402
from . import algorithm as A, data as D, replay as R  # noqa: E402
from .envs import MinAtar  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_ARMS = A.ARM_NAMES


def load_cfg(path=None):
    with open(path or os.path.join(REPO, "src/configs/default.yaml")) as f:
        return yaml.safe_load(f)


def seed_keys(top_seed: int, seeds: list[int]):
    """(S, 5, 2) keys: init, pretrain, env_reset, interact, eval  per learner seed (shared across arms)."""
    root = jax.random.PRNGKey(top_seed)
    return jnp.stack([jax.random.split(jax.random.fold_in(root, 100 + s), 5) for s in seeds])


def tree_bytes(t):
    return sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(t))


class Group:
    """A vmapped batch of runs = arms x seeds (arm-major)."""

    def __init__(self, env, cfg, demo, arms, seeds, top_seed, steps, pretrain_steps, ckpt_dir, tag):
        self.env, self.cfg, self.arms, self.seeds, self.tag = env, cfg, arms, seeds, tag
        self.S, self.nA = len(seeds), len(arms)
        self.Rn = self.S * self.nA
        self.steps, self.pretrain_steps = steps, pretrain_steps
        nd = int(demo["act"].shape[0])
        obs_bytes = int(demo["obs"].shape[1])
        self.L = R.make_layout(nd, cfg["per"]["agent_capacity"], obs_bytes, cfg["dqfd"]["n_step"], cfg["env"]["gamma"],
                               cfg["per"]["alpha"], cfg["per"]["block_size"])
        self.fns = A.Fns(env, self.L, cfg, steps)
        self.demo = D.demo_to_buffer_arrays(demo)
        self.hp = A.stack_hp(arms, self.S, cfg)
        sk = seed_keys(top_seed, seeds)                                   # (S,5,2)
        self.rk = jnp.tile(sk, (self.nA, 1, 1))                           # (R,5,2): arm-major, seed-minor
        self.ckpt_dir = ckpt_dir
        self.mgr = {}
        self.chunk = cfg["run"]["chunk_steps"]
        self._build()

    def _mgr(self, name):
        if name not in self.mgr:
            path = os.path.abspath(os.path.join(self.ckpt_dir, self.tag, name))
            self.mgr[name] = ocp.CheckpointManager(path, options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True))
        return self.mgr[name]

    def _build(self):
        f, L = self.fns, self.L
        demo = self.demo

        def init_one(hp, keys):
            params = f.init_params(keys[0])
            buf = R.init_buffer(L, demo, hp)
            return dict(params=params, tparams=params, opt=f.tx.init(params), buf=buf, cnt=jnp.zeros((), jnp.int32))

        self.init = jax.jit(jax.vmap(init_one))
        self.pre_fn = {}

        def interact_one(c, hp, key, train_on):
            return f.interact_chunk(c, hp, key, self.chunk, train_on)

        def interact_all(c, hp, keys, train_on):
            c2 = jax.vmap(interact_one)(c, hp, keys, train_on)
            jaxlog.nan_guard(c2["params"], jnp.zeros(()), tag=2, dump_dir=os.path.join(REPO, "runs", "nan_dumps"))
            return c2

        self.interact = jax.jit(interact_all, donate_argnums=0)
        self.init_int = jax.jit(jax.vmap(lambda s, k: f.init_interact(s["params"], s["params"], s["opt"], s["buf"], k)))
        self.greedy = jax.jit(jax.vmap(lambda p, k: f.greedy_returns(p, k, self.cfg["run"].get("diag_eval_eps", 20), 0.0)))
        self.agree = jax.jit(jax.vmap(lambda p: f.demo_agreement(p, self.demo["obs"], self.demo["act"])))
        self.gap = jax.jit(jax.vmap(lambda p: f.demo_gap(p, self.demo["obs"], self.demo["act"])))
        n_fin = self.cfg["run"].get("final_eval_episodes", 30)
        self.greedy_fin = jax.jit(jax.vmap(lambda p, k: f.greedy_returns(p, k, n_fin, 0.0)))

    def pretrain_fn(self, n):
        if n not in self.pre_fn:
            f = self.fns

            def one(s, hp, k):
                return f.pretrain_chunk(s, hp, k, n)

            def all_(s, hp, ks):
                s2, dg = jax.vmap(one)(s, hp, ks)
                jaxlog.nan_guard(s2["params"], jnp.zeros(()), tag=1, dump_dir=os.path.join(REPO, "runs", "nan_dumps"))
                return s2, dg

            self.pre_fn[n] = jax.jit(all_, donate_argnums=0)
        return self.pre_fn[n]

    # ------------------------------------------------------------------ phases
    def run_pretrain(self, resume, log_marks):
        state = self.init(self.hp, self.rk)
        done = 0
        m = self._mgr("pre")
        if resume and m.latest_step() is not None:
            done = m.latest_step()
            state = m.restore(done, args=ocp.args.StandardRestore(state))
            jaxlog.info("resumed pretrain", step=done)
        hist = []
        ck = self.chunk
        t0, first = time.time(), True
        while done < self.pretrain_steps:
            n = min(ck, self.pretrain_steps - done)
            keys = jax.vmap(lambda k: jax.random.fold_in(k[1], done))(self.rk)
            tc = time.time()
            state, dg = self.pretrain_fn(n)(state, self.hp, keys)
            jax.block_until_ready(state["params"])
            if first:
                jaxlog.info("pretrain first chunk (compile + run)", secs=time.time() - tc); first = False; t_steady = time.time()
            done += n
            row = dict(step=done, loss=np.asarray(dg["loss"]).tolist(), demo_share=float(np.mean(dg["demo_share"])))
            if done in log_marks or done == self.pretrain_steps:
                ret = np.asarray(self.greedy(state["params"], jax.vmap(lambda k: jax.random.fold_in(k[4], done))(self.rk)))
                row["greedy_return"] = ret.mean(1).tolist()
                row["agreement"] = np.asarray(self.agree(state["params"])).tolist()
                gm, gf = self.gap(state["params"])
                row["demo_gap_mean"], row["demo_gap_frac_ge_margin"] = np.asarray(gm).tolist(), np.asarray(gf).tolist()
                jaxlog.metric("pretrain", step=done, greedy_return_mean=float(ret.mean()), agreement=float(np.mean(row["agreement"])),
                              loss=float(np.mean(dg["loss"])))
            hist.append(row)
            if done % self.cfg["run"]["checkpoint_every"] == 0 or done == self.pretrain_steps:
                m.save(done, args=ocp.args.StandardSave(state)); m.wait_until_finished()
        if not first:
            jaxlog.info("pretrain steady throughput", updates_per_s=(done - ck) / max(time.time() - t_steady, 1e-9), runs=self.Rn)
        return state, hist

    def run_interact(self, state, train_on, resume, hist_in=None):
        c = self.init_int(state, jax.vmap(lambda k: k[2])(self.rk))
        # target <- online at the start of interaction (Algorithm 1: tau divides k in the paper; explicit here).  CHOICE: at
        # k = 75 k, which tau = 10 k does not divide, this adds one sync that Algorithm 1 does not have; the update counter `cnt` restarts at 0.
        c = dict(c, tparams=jax.tree_util.tree_map(jnp.copy, c["params"]))
        done = 0
        m = self._mgr("int")
        if resume and m.latest_step() is not None:
            done = m.latest_step()
            c = m.restore(done, args=ocp.args.StandardRestore(c))
            jaxlog.info("resumed interaction", step=done)
        hist = []
        t0, first = time.time(), True
        train_on = jnp.asarray(train_on)
        while done < self.steps:
            keys = jax.vmap(lambda k: jax.random.fold_in(k[3], done // self.chunk))(self.rk)
            tc = time.time()
            c = self.interact(c, self.hp, keys, train_on)
            jax.block_until_ready(c["t"])
            if first:
                jaxlog.info("interact first chunk (compile + run)", secs=time.time() - tc); first = False; t_steady = time.time(); d_steady = done + self.chunk
            done += self.chunk
            dg = {k: np.asarray(v) for k, v in c["dg"].items()}
            hist.append(dict(step=done, demo_share=(dg["demo_share"] / np.maximum(dg["did"], 1)).tolist(),
                             td_demo=(dg["td_demo"] / np.maximum(dg["n_demo"], 1)).tolist(),
                             td_agent=(dg["td_agent"] / np.maximum(dg["n_agent"], 1)).tolist(),
                             loss=(dg["loss"] / np.maximum(dg["did"], 1)).tolist(), eps=(dg["eps"] / dg["steps"]).tolist()))
            if (done // self.chunk) % max(1, (self.steps // self.chunk) // 20) == 0 or done >= self.steps:
                bs, bc = np.asarray(c["bin_sum"]), np.asarray(c["bin_cnt"])
                nb = min(self.fns.nbins, -(-done // self.fns.bin_w))
                lo = max(0, nb - 4)
                rec = bs[:, lo:nb].sum(1) / np.maximum(bc[:, lo:nb].sum(1), 1)
                per_arm = rec.reshape(self.nA, self.S).mean(1)
                sps = (done - d_steady) / max(time.time() - t_steady, 1e-9) if done > d_steady else float("nan")
                jaxlog.metric("interact", step=done, steps_per_s=sps, run_steps_per_s=sps * self.Rn,
                              **{f"ret_{a}": float(v) for a, v in zip(self.arms, per_arm)})
            if done % self.cfg["run"]["checkpoint_every"] == 0 and done < self.steps:
                m.save(done, args=ocp.args.StandardSave(c)); m.wait_until_finished()
        # Selection-free end-of-run descriptors (greedy eps=0 return on fresh episodes; demo Q-gap)
        tf = time.time()
        gk = jax.vmap(lambda k: jax.random.fold_in(k[4], 10 ** 9 + done))(self.rk)
        self.final = dict(greedy=np.asarray(self.greedy_fin(c["params"], gk)))
        gm, gf = self.gap(c["params"])
        self.final.update(gap_mean=np.asarray(gm), gap_frac=np.asarray(gf))
        jaxlog.info("final greedy eval", secs=time.time() - tf, mean=float(self.final["greedy"].mean()))
        return c, hist


def write_metrics(out_dir, seeds, arms, game, demonstrator, result, meta):
    """runs/<out>/seed_<n>/metrics.json (merged per cell), episode logs next to it."""
    for si, s in enumerate(seeds):
        p = os.path.join(out_dir, f"seed_{s}", "metrics.json")
        cur = json.load(open(p)) if os.path.exists(p) else {}
        cell = cur.setdefault(f"{game}/{demonstrator}", {})
        cell["meta"] = meta
        for ai, a in enumerate(arms):
            r = si + ai * len(seeds)
            cell[a] = {k: (v[r].tolist() if isinstance(v, np.ndarray) and v.ndim >= 1 else v) for k, v in result[a].items() if k != "ep"} if a in result else {}
            if a in result and "ep" in result[a]:
                ep = result[a]["ep"]
                np.savez_compressed(os.path.join(out_dir, f"seed_{s}", f"episodes_{game}_{demonstrator}_{a}.npz"),
                                    ret=ep["ret"][r], step=ep["step"][r], count=ep["count"][r])
        jaxlog.write_json(p, cur)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True); ap.add_argument("--demonstrator", default="subopt", choices=["subopt", "expert", "subopt_eps"])
    ap.add_argument("--arms", default=",".join(ALL_ARMS)); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--top-seed", type=int, default=0)
    ap.add_argument("--steps", type=float, default=None); ap.add_argument("--pretrain", type=int, default=None)
    ap.add_argument("--bin-width", type=int, default=None); ap.add_argument("--demo-dir", default=os.path.join(REPO, "data", "demos"))
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "dev")); ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], help="override section.key=value (recorded in metrics meta)")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    for kv in a.set:
        k, v = kv.split("=", 1); sec, key = k.split(".")
        cfg[sec][key] = yaml.safe_load(v)
    if a.bin_width:
        cfg["run"]["bin_width"] = a.bin_width
    steps = int(a.steps or cfg["run"]["interaction_steps"]); pre = int(a.pretrain if a.pretrain is not None else cfg["dqfd"]["pretrain_steps"])
    arms = a.arms.split(","); seeds = list(range(a.seed0, a.seed0 + a.seeds))
    jaxlog.check_device()
    info = json.load(open(os.path.join(a.demo_dir, f"{a.game}_demos.json")))
    demo = D.load_demos(os.path.join(a.demo_dir, f"{a.game}_{a.demonstrator}.npz"), info[a.demonstrator]["sha256"])
    env = MinAtar(a.game, max_steps=info.get("max_steps"), sticky_action_prob=info.get("sticky_action_prob", 0.0),
                 asterix_spawn_repair=info.get("asterix_spawn_repair", False))  # env flags follow the demo file
    ckpt_dir = a.ckpt_dir or os.path.join(a.out, "_ckpt")
    tag = a.tag or f"{a.game}_{a.demonstrator}"
    jaxlog.info("config", game=a.game, demonstrator=a.demonstrator, arms=",".join(arms), seeds=len(seeds), steps=steps,
                pretrain=pre, n_demo=int(demo["act"].shape[0]), demo_sha=demo["sha256"][:12])
    meta = dict(game=a.game, demonstrator=a.demonstrator, steps=steps, pretrain_steps=pre, seeds=seeds, top_seed=a.top_seed,
                demo_sha256=demo["sha256"], demonstrator_info=info[a.demonstrator], bin_width=cfg["run"]["bin_width"],
                overrides=a.set, jax=jax.__version__, devices=str(jax.devices()))
    pre_arms = [x for x in arms if A.base_arm(x) in A.PRETRAIN_ARMS]
    int_arms = [x for x in arms if A.base_arm(x) not in A.NO_INTERACTION_ARMS]
    result, history = {}, {}
    t_all = time.time()
    log_marks = {10000, 50000, 150000, pre}
    state_pre, hist_pre, gp = None, [], None
    if pre_arms:                                   # --- phase 1: pre-training of every arm that has one (demo data only)
        gp = Group(env, cfg, demo, pre_arms, seeds, a.top_seed, steps, pre, ckpt_dir, f"{tag}_pre")
        jaxlog.info("pretrain group", arms=",".join(pre_arms), runs=gp.Rn)
        state_pre, hist_pre = gp.run_pretrain(a.resume, log_marks)
        for imit in [x for x in pre_arms if A.base_arm(x) == "IMIT"][:1]:
            ii = pre_arms.index(imit) * gp.S
            ret = np.asarray(jax.jit(jax.vmap(lambda p, k: gp.fns.greedy_returns(p, k, 100, cfg["imitation"]["eval_eps"])))(
                jax.tree_util.tree_map(lambda x: x[ii:ii + gp.S], state_pre["params"]), gp.rk[ii:ii + gp.S, 4]))
            result[imit] = dict(online_return=ret.mean(1))
            history[imit] = dict(interact=[], pretrain=_slice_hist(hist_pre, slice(ii, ii + gp.S)))
            jaxlog.metric("imitation", mean=float(ret.mean()), per_seed=ret.mean(1).tolist())
    if int_arms:                                   # --- phase 2: ONE vmapped interaction batch over all arms x seeds
        g = Group(env, cfg, demo, int_arms, seeds, a.top_seed, steps, pre, ckpt_dir, f"{tag}_int")
        jaxlog.info("interaction group", arms=",".join(int_arms), runs=g.Rn,
                    buffer_MB_per_run=tree_bytes(jax.eval_shape(lambda: R.init_buffer(g.L, g.demo, {k: v[0] for k, v in g.hp.items()}))) / 1e6)
        state = g.init(g.hp, g.rk)                 # fresh init (arms without pre-training) ...
        for ai, arm in enumerate(int_arms):        # ... pre-trained state spliced in for the arms that have it
            if arm in pre_arms:
                sl_i, pj = slice(ai * g.S, (ai + 1) * g.S), pre_arms.index(arm)
                sl_p = slice(pj * g.S, (pj + 1) * g.S)
                state = jax.tree_util.tree_map(lambda full, part: full.at[sl_i].set(part[sl_p]), state, state_pre)
        del state_pre
        c, ihist = g.run_interact(state, np.ones(g.Rn, bool), a.resume)
        host = {k: np.asarray(c[k]) for k in ("bin_sum", "bin_cnt", "ep_ret_log", "ep_step_log", "ep_count")}
        for ai, arm in enumerate(int_arms):
            sl = slice(ai * g.S, (ai + 1) * g.S)
            result[arm] = dict(bin_sum=host["bin_sum"][sl], bin_cnt=host["bin_cnt"][sl],
                               ep_ret_log=host["ep_ret_log"][sl], ep_step_log=host["ep_step_log"][sl], ep_count=host["ep_count"][sl],
                               final_greedy=g.final["greedy"][sl], final_gap_mean=g.final["gap_mean"][sl], final_gap_frac=g.final["gap_frac"][sl])
            pj = pre_arms.index(arm) if arm in pre_arms else None
            history[arm] = dict(interact=_slice_hist(ihist, sl),
                                pretrain=_slice_hist(hist_pre, slice(pj * g.S, (pj + 1) * g.S)) if pj is not None else [])
    write_outputs(a.out, seeds, arms, a.game, a.demonstrator, result, history, meta)
    jaxlog.info("finished", secs=time.time() - t_all)


def _slice_hist(hist, sl):
    """Per-run lists in a history row -> that arm's per-seed lists."""
    out = []
    for row in hist:
        out.append({k: (v[sl] if isinstance(v, list) and len(v) > sl.stop - sl.start else v) for k, v in row.items()})
    return out


def write_outputs(out_dir, seeds, arms, game, demonstrator, result, history, meta):
    """runs/<out>/seed_<n>/metrics.json (per cell game/demonstrator, per arm: bin sums/counts, final stats);
    full episode logs in episodes_<game>_<demo>_<arm>.npz; training histories in history_<game>_<demo>.json."""
    for si, s in enumerate(seeds):
        d = os.path.join(out_dir, f"seed_{s}")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "metrics.json")
        cur = json.load(open(p)) if os.path.exists(p) else {}
        cell = cur.setdefault(f"{game}/{demonstrator}", {})
        cell["meta"] = meta
        for arm, r in result.items():
            if arm == "IMIT":
                cell[arm] = dict(online_return=float(r["online_return"][si]))
                continue
            cell[arm] = dict(bin_sum=r["bin_sum"][si], bin_cnt=r["bin_cnt"][si], n_episodes=int(r["ep_count"][si]))
            if "final_greedy" in r:
                cell[arm].update(final_greedy=r["final_greedy"][si], final_gap_mean=float(r["final_gap_mean"][si]),
                                 final_gap_frac=float(r["final_gap_frac"][si]))
            np.savez_compressed(os.path.join(d, f"episodes_{game}_{demonstrator}_{arm}.npz"),
                                ret=r["ep_ret_log"][si], step=r["ep_step_log"][si], count=r["ep_count"][si])
        jaxlog.write_json(p, cur)
    jaxlog.write_json(os.path.join(out_dir, f"history_{game}_{demonstrator}.json"), history)


if __name__ == "__main__":
    main()

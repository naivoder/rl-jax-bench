"""Training entry point: PPO with time-awareness / partial-episode bootstrapping arms (E3, E4/E5, E7) and the tabular
experiments (E1, E2, E9). One process per experiment; every (arm, seed) lane is one vmapped batch.

    python -m src.train --config src/configs/e3_queue_of_cars.yaml --tag small [--set key=value ...]
(PPO experiments only. E1/E2 run through scripts/toys/run_toys.py, which reads src/configs/e1_last_moment.yaml and
src/configs/e2_two_goal.yaml; E9 through scripts/small/run_e9.py with src/configs/e9_difficult_grid.yaml.)

PRNG key tree (ARM never enters it, so paired arms share every key = common random numbers):
    root   = PRNGKey(seed)                       seed in cfg.seeds (paper §1: 0, 1000, ..., 9000; E9: 0..29)
    exp    = fold_in(root, EXP_ID)               EXP_ID = cfg.exp_id (1..9 = E1..E9)
    k_init = fold_in(exp, 0)                     network init (PPO); unused by the tabular learners
    k_env  = fold_in(exp, 1)  -> initial reset: fold_in(fold_in(k_env, 0), 0)
                              -> step g:        fold_in(fold_in(k_env, 1), g)  (split -> step key, reset key)
                              (tabular: episode-e start state: fold_in(fold_in(k_env, 0), e))
    k_act  = fold_in(exp, 2)  -> fold_in(k_act, g)                       action sampling / ε-greedy coins
    k_mb   = fold_in(exp, 3)  -> PPO: fold_in(fold_in(k_mb, u), epoch)   minibatch permutation
                              -> Buffer-Q: fold_in(k_mb, g)              replay minibatch indices
    k_eval = fold_in(exp, 4)  -> fold_in(k_eval, u) -> fold_in(., episode) evaluation episodes (E7) / probes (E5)
(g = global env step, 0-based; u = 0-based PPO update.) A resumed run reproduces the uninterrupted one.
"""
from __future__ import annotations

import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from functools import partial  # noqa: E402
from typing import Any, NamedTuple  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import yaml  # noqa: E402

# full-precision float32 matmuls (TF32 on the 3090 otherwise; the PPO-details reproduction convention)
jax.config.update("jax_default_matmul_precision", "highest")
from rl_jax_bench.jaxlog import Timer, check_device, info, metric, nan_guard, warn, write_json  # noqa: E402

from . import algorithm as A  # noqa: E402
from . import data as D  # noqa: E402
from . import model as M  # noqa: E402
from .ppo8 import algorithm as P8  # noqa: E402
from .ppo8 import data as P8D  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO, "src", "configs")


# ===================================================================== config helpers
def load_config(path: str, overrides=()) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base = load_config(os.path.join(os.path.dirname(path), cfg.pop("base")))
        arms = cfg.pop("arms", None)
        base.update(cfg)
        if arms is not None:
            base["arms"] = arms
        cfg = base
    for ov in overrides:
        k, v = ov.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


_GIT_AT_START: str | None = None


def git_sha_at_start() -> str:
    """git_sha() evaluated once, at the first call in the process (review fix: metrics.json used to record HEAD at
    write time, which can move while a long run is in flight). Entry points call it before training starts."""
    global _GIT_AT_START
    if _GIT_AT_START is None:
        _GIT_AT_START = git_sha()
    return _GIT_AT_START


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "src"], stderr=subprocess.DEVNULL)
        return sha + ("-dirty" if dirty.strip() else "")
    except Exception:
        return "unknown"


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def env_checksum(cfg: dict) -> dict:
    """Env provenance for metrics.json : SHA-256 of src/data.py (all our env code), of the gym-v4 model
    file for the MJX ports, and the brax / mujoco versions for the Brax proxies."""
    out = dict(data_py_sha256=file_sha(os.path.join(REPO, "src", "data.py")))
    env = str(cfg.get("env", ""))
    if env.startswith("gymv4-"):
        xml = {"gymv4-inverted_pendulum": "inverted_pendulum.xml", "gymv4-hopper": "hopper.xml"}[env]
        out["xml_sha256"] = file_sha(os.path.join(REPO, "src", "assets", "gym_v4", xml))
    if env.startswith("brax-") or env.startswith("gymv4-"):
        import importlib.metadata as md
        for pkg in ("brax", "mujoco", "mujoco-mjx"):
            try:
                out[pkg] = md.version(pkg)
            except Exception:
                pass
    return out


def exp_key(seed, exp_id):
    return jax.random.fold_in(jax.random.PRNGKey(seed), exp_id)


def pick_device(cfg: dict):
    """GPU is the default; cfg.device = 'cpu' places the run on CPU deliberately (logged with the reason)."""
    want = cfg.get("device", "gpu")
    if want == "cpu":
        info("placing this run on CPU deliberately", reason=cfg.get("device_reason", "unspecified"),
             visible=str(jax.devices()))
        return jax.devices("cpu")[0]
    return jax.devices()[0]


# ===================================================================== PPO
class Lane(NamedTuple):
    """Per-lane (arm x seed) traced settings: the ONLY things that differ between arms."""
    seed: jax.Array
    time_mode: jax.Array   # src.data.TIME_MODES
    peb: jax.Array         # 1.0 = partial-episode bootstrapping (Eq. 6), 0.0 = timeout is terminal (Eq. 4)
    T: jax.Array           # training time limit
    gamma: jax.Array


class Runner(NamedTuple):
    params: Any
    opt_state: Any
    st: D.TLState
    obs: jax.Array         # raw obs for the next action
    tau: jax.Array
    obs_rms: P8D.RMS
    ret_rms: P8D.RMS
    ret_acc: jax.Array     # VecNormalize discounted-return accumulator (1)
    ep_ret: jax.Array      # raw episode return so far
    ep_len: jax.Array
    u: jax.Array
    ts_done: jax.Array = jnp.float32(0)  # env steps of COMPLETED episodes (Baselines ppo1 timesteps_so_far)


class Keys(NamedTuple):
    k_act: jax.Array
    k_env_step: jax.Array
    k_mb: jax.Array
    k_eval: jax.Array


class Tr(NamedTuple):
    x: jax.Array
    a: jax.Array
    logp: jax.Array
    v: jax.Array
    r: jax.Array        # training reward (scaled if reward_scaling)
    term: jax.Array
    reset: jax.Array
    v_fin: jax.Array    # V(s'_t) (pre-reset next obs)
    ep_ret: jax.Array   # raw return of the episode ending at this step (0 otherwise)
    ep_len: jax.Array
    ep_done: jax.Array
    timeout: jax.Array
    raw: jax.Array = None  # un-normalized network input concat(obs, tau) of the acting step (ppo1_numerics)


class PPO:
    def __init__(self, cfg: dict):
        self.cfg = cfg = copy.deepcopy(cfg)
        self.env = D.make_env(cfg)
        self.d = self.env.obs_dim
        self.cont = self.env.continuous
        self.net = M.build(self.env.n_out, self.cont, cfg["hidden"])
        self.Mst = cfg["num_steps"]
        self.n_mb, self.K = cfg["num_minibatches"], cfg["update_epochs"]
        assert self.Mst % self.n_mb == 0
        self.mb = self.Mst // self.n_mb
        # stop rule. 'updates' (default) = Baselines ppo2: nupdates = total // nbatch (98 304 steps for a nominal 1e5).
        # 'ppo1_completed_steps' (recheck) = Baselines ppo1 @9fa8e1b pposgd_simple.learn: iterate while
        # timesteps_so_far (= summed lengths of COMPLETED episodes) < max_timesteps, lrmult = max(1 - ts/max, 0) for the
        # 'linear' schedule (lr AND clip). Lanes run a common upper bound of updates; a lane that has met its stop
        # condition is frozen (state unchanged, its episodes masked) and its records are trimmed to its own updates.
        self.stop_rule = cfg.get("stop_rule", "updates")
        assert self.stop_rule in ("updates", "ppo1_completed_steps"), self.stop_rule
        if self.stop_rule == "updates":
            self.U = cfg["total_timesteps"] // self.Mst
        else:
            t_max = max(int(a["T"]) for a in cfg["arms"].values())
            # completed steps >= u * Mst - (t_max - 1), so this many updates always reach the budget
            self.U = -(-(cfg["total_timesteps"] + t_max - 1) // self.Mst)
        self.tx = P8.make_optimizer(cfg["learning_rate"], cfg["max_grad_norm"], cfg["adam_eps"],
                                    cfg.get("adam_eps_mode", "torch"))
        self.eps, self.oclip, self.rclip = cfg["norm_eps"], cfg["obs_clip"], cfg["reward_clip"]

    # -------------------------------------------------------------- helpers
    def norm(self, rms, o):
        if not self.cfg["obs_norm"]:
            return o
        return P8D.rms_normalize(rms, o, self.eps, self.oclip)

    def x_of(self, rms, o, tau):
        """Network input. obs_rms tracks concat(obs, tau) per dimension (d + 1 entries; the first d are exactly the
        VecNormalize statistics of the observation). Default (paper/Baselines): tau is appended RAW after the
        observation is normalized. Ablation tau_obs_norm (Tonic MeanStd, oracle-stage caveat): tau is z-scored with the
        same running statistics (time-unaware arms: tau = 0 has mean 0 exactly, so their input stays 0)."""
        if not self.cfg["obs_norm"]:
            return M.net_input(o, tau)
        if self.cfg.get("ppo1_numerics", False):
            return self.ppo1_norm(rms, jnp.concatenate([o, jnp.asarray(tau, o.dtype)[..., None]], -1))
        if self.cfg.get("tau_obs_norm", False):
            z = P8D.rms_normalize(rms, jnp.concatenate([o, jnp.asarray(tau, o.dtype)[..., None]], -1), self.eps,
                                  self.oclip)
            return z
        sub = P8D.RMS(rms.mean[: self.d], rms.var[: self.d], rms.count)
        return M.net_input(P8D.rms_normalize(sub, o, self.eps, self.oclip), tau)

    def ppo1_norm(self, rms, z):
        """Baselines ppo1 @9fa8e1b mlp_policy: clip((ob - mean) / std, -5, 5) with mpi_running_mean_std's
        std = sqrt(max(E[x^2] - mean^2, 1e-2)) (no epsilon); every input entry, tau included (recheck C14 (b))."""
        return jnp.clip((z - rms.mean) / jnp.sqrt(jnp.maximum(rms.var, 1e-2)), -self.oclip, self.oclip)

    def normc_params(self, params, key, time_mode):
        """Baselines ppo1 init (tf_util.normc_initializer: randn, each output column scaled to norm `std`): hidden and
        value layers 1.0, policy head 0.01, zero biases (C14 (c)). ppo1 builds a time-unaware net WITHOUT a tau input,
        so for time_mode = none the first-layer columns are normalized over the d observation rows (tau row 0); the
        random draws are shared by all arms of a seed (common random numbers)."""
        p = jax.tree.map(lambda x: x, params)
        mask = jnp.where(time_mode == D.TIME_MODES["none"], jnp.r_[jnp.ones(self.d), 0.0], jnp.ones(self.d + 1))
        for i, (net, layer, std) in enumerate([("actor", "l0", 1.0), ("actor", "l1", 1.0), ("actor_head", None, 0.01),
                                               ("critic", "l0", 1.0), ("critic", "l1", 1.0), ("critic_head", None, 1.0)]):
            node = p["params"][net] if layer is None else p["params"][net][layer]
            w = jax.random.normal(jax.random.fold_in(key, i), node["kernel"].shape, node["kernel"].dtype)
            if layer == "l0":
                w = w * mask[:, None].astype(w.dtype)
            node["kernel"] = w * std / jnp.sqrt(jnp.sum(jnp.square(w), axis=0, keepdims=True))
            node["bias"] = jnp.zeros_like(node["bias"])
        return p

    def rms_upd(self, rms, o, tau):
        return P8D.rms_update(rms, jnp.concatenate([o, jnp.asarray(tau, o.dtype)[..., None]], -1)[None])

    def env_action(self, a):
        """tonic_rescale (default): clip to the env's [-1, 1] bounds (Tonic ActionRescaler clips before rescaling).
        raw (recheck): the paper-era Baselines runner passes the raw sample (ppo2/ppo1 @9fa8e1b
        `env.step(actions)`); the simulator clamps to ctrlrange, so the runner does not clip."""
        if not self.cont or self.cfg.get("action_interface", "tonic_rescale") == "raw":
            return a
        return jnp.clip(a, self.env.act_low, self.env.act_high)

    # -------------------------------------------------------------- init
    def init(self, lane: Lane):
        cfg = self.cfg
        ek = exp_key(lane.seed, cfg["exp_id"])
        k_init, k_env, k_act, k_mb, k_eval = (jax.random.fold_in(ek, i) for i in range(5))
        params = self.net.init(k_init, jnp.zeros((1, self.d + 1)))
        ppo1n = cfg.get("ppo1_numerics", False)
        if ppo1n:
            params = self.normc_params(params, jax.random.fold_in(k_init, 7), lane.time_mode)
        opt_state = self.tx.init(params)
        st, o, tau = D.tl_reset_obs(self.env, jax.random.fold_in(jax.random.fold_in(k_env, 0), 0), lane.T,
                                    lane.time_mode)
        obs_rms = P8D.rms_init((self.d + 1,))  # obs dims + the tau dim (see x_of)
        if ppo1n:  # mpi_running_mean_std init: sum 0, sumsq 1e-2, count 1e-2 => mean 0, var 1; no reset update
            obs_rms = P8D.RMS(jnp.zeros((self.d + 1,)), jnp.ones((self.d + 1,)), jnp.asarray(1e-2, jnp.float32))
        elif cfg["obs_norm"]:
            obs_rms = self.rms_upd(obs_rms, o, tau)  # VecNormalize.reset updates with the first obs
        z = jnp.zeros((1,), jnp.float32)
        r = Runner(params, opt_state, st, o, tau, obs_rms, P8D.rms_init(()), z, jnp.float32(0), jnp.float32(0),
                   jnp.int32(0), jnp.float32(0))
        return r, Keys(k_act, jax.random.fold_in(k_env, 1), k_mb, k_eval)

    # -------------------------------------------------------------- rollout
    def rollout(self, r: Runner, keys: Keys, lane: Lane):
        cfg = self.cfg

        def step(c, t):
            st, o, tau, obs_rms, ret_rms, ret_acc, ep_ret, ep_len = c
            g = r.u * self.Mst + t
            x = self.x_of(obs_rms, o, tau)
            dist, v = self.net.apply(r.params, x)
            a = M.sample(jax.random.fold_in(keys.k_act, g), dist, self.cont)
            lp = M.log_prob(dist, a, self.cont)
            s = D.tl_step(self.env, jax.random.fold_in(keys.k_env_step, g), st, self.env_action(a), lane.T,
                          lane.time_mode)
            if cfg["obs_norm"] and not cfg.get("ppo1_numerics", False):  # ppo1: statistics fixed during the rollout
                obs_rms = self.rms_upd(obs_rms, s.obs, s.tau)
            if cfg["reward_scaling"]:
                if cfg.get("reward_scale_reset_on_done", True):
                    ret_acc, ret_rms, rw = P8D.scale_reward_baselines(ret_acc, ret_rms, s.reward[None], s.reset[None],
                                                                      cfg["reward_scale_gamma"], self.eps, self.rclip)
                else:  # recheck C2: paper-era VecNormalize, the return accumulator is never reset
                    ret_acc, ret_rms, rw = D.scale_reward_baselines_2018(ret_acc, ret_rms, s.reward[None],
                                                                         cfg["reward_scale_gamma"], self.eps,
                                                                         self.rclip)
                rw = rw[0]
            else:
                rw = s.reward
            v_fin = self.net.apply(r.params, self.x_of(obs_rms, s.final_obs, s.final_tau))[1]
            ep_ret = ep_ret + s.reward
            ep_len = ep_len + 1.0
            rf = s.reset.astype(jnp.float32)
            tr = Tr(x, a, lp, v, rw, s.term.astype(jnp.float32), rf, v_fin, ep_ret * rf, ep_len * rf, rf,
                    s.timeout.astype(jnp.float32), jnp.concatenate([o, jnp.asarray(tau, o.dtype)[..., None]], -1))
            ep_ret, ep_len = ep_ret * (1 - rf), ep_len * (1 - rf)
            return (s.state, s.obs, s.tau, obs_rms, ret_rms, ret_acc, ep_ret, ep_len), tr

        c0 = (r.st, r.obs, r.tau, r.obs_rms, r.ret_rms, r.ret_acc, r.ep_ret, r.ep_len)
        c, traj = jax.lax.scan(step, c0, jnp.arange(self.Mst))
        return c, traj

    # -------------------------------------------------------------- update
    def _loss(self, params, mb, clip_coef=None):
        cfg = self.cfg
        clip_coef = cfg["clip_coef"] if clip_coef is None else clip_coef
        dist, v = self.net.apply(params, mb["x"])
        nl = M.log_prob(dist, mb["a"], self.cont)
        ent = M.entropy(dist, self.cont)
        return P8.ppo_terms(nl, mb["logp"], ent, v, mb["v"], mb["ret"], mb["adv"], clip_coef=clip_coef,
                            clip_vloss=cfg["clip_vloss"], ent_coef=cfg["ent_coef"], vf_coef=cfg["vf_coef"],
                            norm_adv_mb=cfg["norm_adv"] and cfg.get("adv_norm_scope", "minibatch") == "minibatch",
                            ddof=cfg["adv_norm_ddof"])

    def update(self, r: Runner, keys: Keys, lane: Lane):
        cfg = self.cfg
        (st, o, tau, obs_rms, ret_rms, ret_acc, ep_ret, ep_len), tj = self.rollout(r, keys, lane)
        adv, ret = A.gae_peb(tj.r, tj.v, tj.v_fin, tj.term, tj.reset, lane.gamma, cfg["gae_lambda"], lane.peb)
        if cfg.get("adv_norm_scope", "minibatch") == "batch":
            # ablation (Tonic segments.py: one normalisation over the whole segment); ppo2 default = per minibatch
            adv = P8.normalize_adv(adv, cfg["adv_norm_ddof"])
        data = dict(x=tj.x, a=tj.a, logp=tj.logp, v=tj.v, ret=ret, adv=adv)
        if cfg.get("ppo1_numerics", False):
            # pposgd_simple @9fa8e1b: pi.ob_rms.update(seg["ob"]) once per iteration, then assign_old_eq_new(): the
            # batch is re-normalized with the NEW statistics and the old log-probs are those of the unchanged params
            # on it; GAE / value targets keep the rollout's vpred (C14 (a))
            obs_rms = P8D.rms_update(obs_rms, tj.raw)
            xn = self.ppo1_norm(obs_rms, tj.raw)
            dist_n, _ = self.net.apply(r.params, xn)
            data = dict(data, x=xn, logp=M.log_prob(dist_n, tj.a, self.cont))
        opt0 = r.opt_state
        ppo1_stop = self.stop_rule == "ppo1_completed_steps"
        if ppo1_stop:  # Baselines ppo1: lrmult = max(1 - timesteps_so_far / max_timesteps, 0); stop when >= max
            lrmult = jnp.maximum(1.0 - r.ts_done / float(cfg["total_timesteps"]), 0.0)
            active = r.ts_done < float(cfg["total_timesteps"])
        if cfg.get("anneal_lr", False):
            # ablation (Baselines ppo1 schedule='linear'): lr_u = lr0 (1 - u/U); ppo2 default = constant
            if ppo1_stop:
                opt0 = P8.set_lr(opt0, cfg["learning_rate"] * lrmult)
            else:
                opt0 = P8.set_lr(opt0, P8.lr_at(r.u, self.U, cfg["learning_rate"], True))
        clip = cfg["clip_coef"]  # python float on the default path (trace identical to the full-scale trainer)
        if cfg.get("anneal_clip", False):
            # ablation (Baselines ppo1 @9fa8e1b 2018-01 pposgd_simple.py:100 "clip_param = clip_param *
            # lrmult", lrmult = 1 - t/T with schedule='linear'; removed upstream only in 2019-02 b875fb7)
            if ppo1_stop:
                clip = cfg["clip_coef"] * lrmult
            else:
                clip = cfg["clip_coef"] * (1.0 - r.u.astype(jnp.float32) / self.U)
        k_u = jax.random.fold_in(keys.k_mb, r.u)

        def epoch(c, e):
            def mb_step(c2, ids):
                p, os_ = c2
                mb = jax.tree.map(lambda z: z[ids], data)
                (loss, aux), g = jax.value_and_grad(self._loss, has_aux=True)(p, mb, clip)
                upd, os_ = self.tx.update(g, os_, p)
                aux = dict(aux, loss=loss, grad_norm=optax.tree.norm(g))
                return (optax.apply_updates(p, upd), os_), aux

            perm = jax.random.permutation(jax.random.fold_in(k_u, e), self.Mst)
            return jax.lax.scan(mb_step, c, perm.reshape(self.n_mb, self.mb))

        (params, opt_state), aux = jax.lax.scan(epoch, (r.params, opt0), jnp.arange(self.K))
        m = {k: v.mean() for k, v in aux.items() if k != "max_ratio_dev"}
        m["ratio_dev_first_mb"] = aux["max_ratio_dev"][0, 0]
        m["explained_variance"] = P8.explained_variance(tj.v, ret)
        m["ep_ret_sum"] = tj.ep_ret.sum()
        m["ep_cnt"] = tj.ep_done.sum()
        m["ep_len_sum"] = tj.ep_len.sum()
        m["n_timeouts"] = tj.timeout.sum()
        m["n_terms"] = tj.term.sum()
        m["ret_scale"] = jnp.sqrt(ret_rms.var + self.eps)
        # per-step episode-end records (for the paper's training-return curve vs env steps)
        m["ep_ret_steps"] = tj.ep_ret
        m["ep_done_steps"] = tj.ep_done
        if self.cont:
            m["log_std_mean"] = params["params"]["log_std"].mean()
        if cfg.get("probe") == "qoc":
            m["p_danger"] = self.qoc_probe(params, lane)
        nan_guard({"params": params, "loss": m["loss"]}, {"m": m}, tag=0,
                  dump_dir=os.path.join(REPO, "runs", "nan_dumps"))
        new = Runner(params, opt_state, st, o, tau, obs_rms, ret_rms, ret_acc, ep_ret, ep_len, r.u + 1,
                     r.ts_done + tj.ep_len.sum())
        if ppo1_stop:  # a lane past its budget is frozen; its records from here on are masked and trimmed later
            new = jax.tree.map(lambda a, b: jnp.where(active, a, b), new, r)
            for k in ("ep_ret_sum", "ep_cnt", "ep_len_sum", "ep_ret_steps", "ep_done_steps"):
                m[k] = m[k] * active
            m["active"] = active.astype(jnp.float32)
            m["ts_done"] = new.ts_done
        return new, m

    def run_chunk(self, r, keys, lane, n: int):
        return jax.lax.scan(lambda rr, _: self.update(rr, keys, lane), r, None, length=n)

    # -------------------------------------------------------------- probes / evaluation
    def qoc_probe(self, params, lane: Lane):
        """P(dangerous | slot, elapsed t) for t = 0..T_probe-1 (E3; Figure 2). One-hot obs, no obs norm."""
        Tp = self.cfg["probe_T"]
        tt = jnp.arange(Tp)
        taus = jax.vmap(lambda t: D.time_feature(t, lane.T, lane.time_mode))(tt)
        oh = jnp.eye(self.d, dtype=jnp.float32)
        x = jax.vmap(lambda ta: M.net_input(oh, jnp.full((self.d,), ta)))(taus)  # (Tp, d, d+1)
        logits, _ = self.net.apply(params, x)
        return jax.nn.softmax(logits, -1)[..., 1]

    def eval_discounted(self, r: Runner, keys: Keys, lane: Lane, n_ep: int, H: int, gamma_eval: float,
                        deterministic: bool = False):
        """E7 evaluation (paper Figure 6): n_ep episodes of at most H steps (no training time limit; env
        terminations end them), stochastic policy, frozen obs normalizer; returns per-episode discounted and
        undiscounted raw returns and lengths. tau follows the lane's own mode with T = H (time-unaware arms: 0)."""
        k_u = jax.random.fold_in(keys.k_eval, r.u)

        def one(ep):
            k = jax.random.fold_in(k_u, ep)
            es, o = self.env.reset(jax.random.fold_in(k, 0))

            def st(c, t):
                es, o, alive, G, Gu, L = c
                x = self.x_of(r.obs_rms, o, D.time_feature(t, H, lane.time_mode))
                dist, _ = self.net.apply(r.params, x)
                if deterministic:  # recheck: mean action (the paper does not say which policy Fig. 6 used)
                    a = M.mode(dist, self.cont)
                else:
                    a = M.sample(jax.random.fold_in(jax.random.fold_in(k, 1), t), dist, self.cont)
                es2, o2, rew, term = self.env.step(jax.random.fold_in(jax.random.fold_in(k, 2), t), es,
                                                   self.env_action(a))
                G = G + jnp.where(alive, gamma_eval ** t.astype(jnp.float32) * rew, 0.0)
                Gu = Gu + jnp.where(alive, rew, 0.0)
                L = L + alive.astype(jnp.float32)
                alive = jnp.logical_and(alive, jnp.logical_not(term))
                return (es2, o2, alive, G, Gu, L), None

            (_, _, _, G, Gu, L), _ = jax.lax.scan(st, (es, o, jnp.bool_(True), 0.0, 0.0, 0.0), jnp.arange(H))
            return G, Gu, L

        return jax.vmap(one)(jnp.arange(n_ep))

    def value_readout(self, r: Runner, keys: Keys, lane: Lane, n_ep: int):
        """E5 (Figure 4): critic value vs elapsed step on n_ep episodes of the lane's training time limit,
        un-scaled by the reward-scale std (critic * sqrt(ret_rms.var + eps)). Returns (values (n_ep, T_max),
        alive masks)."""
        Tm = self.cfg["readout_T"]
        k_u = jax.random.fold_in(jax.random.fold_in(keys.k_eval, 10_000_000), r.u)
        scale = jnp.sqrt(r.ret_rms.var + self.eps) if self.cfg["reward_scaling"] else 1.0

        def one(ep):
            k = jax.random.fold_in(k_u, ep)
            es, o = self.env.reset(jax.random.fold_in(k, 0))

            def st(c, t):
                es, o, alive = c
                x = self.x_of(r.obs_rms, o, D.time_feature(t, lane.T, lane.time_mode))
                dist, v = self.net.apply(r.params, x)
                a = M.sample(jax.random.fold_in(jax.random.fold_in(k, 1), t), dist, self.cont)
                es2, o2, _, term = self.env.step(jax.random.fold_in(jax.random.fold_in(k, 2), t), es,
                                                 self.env_action(a))
                out = (v * scale, alive)
                alive = jnp.logical_and(alive, jnp.logical_not(term))
                return (es2, o2, alive), out

            _, (vs, al) = jax.lax.scan(st, (es, o, jnp.bool_(True)), jnp.arange(Tm))
            return vs, al

        return jax.vmap(one)(jnp.arange(n_ep))


def lanes_from_cfg(cfg: dict):
    names, rows = [], []
    for arm, a in cfg["arms"].items():
        for s in cfg["seeds"]:
            names.append((arm, s))
            rows.append((s, D.TIME_MODES[a.get("time_mode", "none")], float(a.get("peb", 0)), int(a["T"]),
                         float(a.get("gamma", cfg["gamma"]))))
    cols = list(zip(*rows))
    lane = Lane(jnp.asarray(cols[0], jnp.int32), jnp.asarray(cols[1], jnp.int32), jnp.asarray(cols[2], jnp.float32),
                jnp.asarray(cols[3], jnp.int32), jnp.asarray(cols[4], jnp.float32))
    return names, lane


def _save(ckpt_dir, runner, done_u, chunks):
    import orbax.checkpoint as ocp
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(ckpt_dir, f"u{done_u:06d}"))
    with ocp.StandardCheckpointer() as cp:
        cp.save(path, runner)
    with open(os.path.join(ckpt_dir, f"u{done_u:06d}_chunks.pkl"), "wb") as f:
        import pickle
        pickle.dump(chunks, f)
    for f in sorted(os.listdir(ckpt_dir)):
        if not f.startswith(f"u{done_u:06d}"):
            p = os.path.join(ckpt_dir, f)
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)


def _restore(ckpt_dir, runner_like):
    import orbax.checkpoint as ocp
    import pickle
    dirs = sorted(d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d)))
    if not dirs:
        return runner_like, 0, []
    last = dirs[-1]
    with ocp.StandardCheckpointer() as cp:
        runner = cp.restore(os.path.abspath(os.path.join(ckpt_dir, last)), runner_like)
    with open(os.path.join(ckpt_dir, f"{last}_chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return runner, int(last[1:]), chunks


def train_ppo(cfg: dict, out_dir: str, ckpt: bool = True, max_chunks: int | None = None):
    dev = pick_device(cfg)
    with jax.default_device(dev):
        return _train_ppo(cfg, out_dir, ckpt, max_chunks, dev)


def _train_ppo(cfg, out_dir, ckpt, max_chunks, dev):
    git_sha_at_start()  # pin the launch commit before any training
    ppo = PPO(cfg)
    names, lane = lanes_from_cfg(cfg)
    L = len(names)
    info("build", exp=cfg["name"], env=ppo.env.name, lanes=L, updates=ppo.U, obs_dim=ppo.d, device=str(dev))
    t = Timer()
    runner, keys = jax.jit(jax.vmap(ppo.init))(lane)
    jax.block_until_ready(runner)
    # paired-init assertion: every arm of a seed starts from identical parameters 
    p0 = jax.tree.leaves(runner.params)
    for i, (arm, s) in enumerate(names):
        j = names.index((names[0][0], s))
        if cfg.get("ppo1_numerics", False):  # ppo1 init differs by design between TA and unaware arms (C14 (c))
            break
        assert all(np.array_equal(np.asarray(x[i]), np.asarray(x[j])) for x in p0), "paired init violated"
    ckpt_dir = os.path.join(out_dir, "ckpt")
    done_u, chunks = 0, []
    if ckpt and os.path.isdir(ckpt_dir):
        runner, done_u, chunks = _restore(ckpt_dir, runner)
        if done_u:
            info("resumed from checkpoint", updates_done=done_u)
    C = cfg["chunk_updates"]
    ev = cfg.get("eval")
    compiled: dict[int, Any] = {}
    compile_s, run_s, eval_s = 0.0, 0.0, 0.0
    eval_fn = det_eval_fn = None
    if ev:
        eval_fn = jax.jit(jax.vmap(partial(ppo.eval_discounted, n_ep=ev["episodes"], H=ev["horizon"],
                                           gamma_eval=ev["gamma"])))
        if ev.get("deterministic_too", False):  # recheck: also evaluate the mean-action policy
            det_eval_fn = jax.jit(jax.vmap(partial(ppo.eval_discounted, n_ep=ev["episodes"], H=ev["horizon"],
                                                   gamma_eval=ev["gamma"], deterministic=True)))
    evals = [c["eval"] for c in chunks if "eval" in c]
    # (Figure 4 is value vs episode step AND training step): optional critic read-outs at intermediate
    # updates. They use their own keys (fold_in(fold_in(k_eval, 1e7), u)) and never touch the training state.
    ro_at = sorted(int(u) for u in cfg.get("readout_updates", []) or [])
    if ro_at:
        bounds = set(range(C, ppo.U + 1, C)) | {ppo.U}
        bad = [u for u in ro_at if u not in bounds]
        assert not bad, f"readout_updates must fall on chunk boundaries (chunk_updates={C}): {bad}"
        ro_fn = jax.jit(jax.vmap(partial(ppo.value_readout, n_ep=cfg.get("readout_episodes", 5))))
        ro_t = np.unique(np.r_[np.arange(0, cfg["readout_T"], cfg.get("readout_stride", 10)), cfg["readout_T"] - 1])
    readouts = [c["readout"] for c in chunks if "readout" in c]
    while done_u < ppo.U:
        n = min(C, ppo.U - done_u)
        if n not in compiled:
            t.lap()
            compiled[n] = jax.jit(jax.vmap(partial(ppo.run_chunk, n=n))).lower(runner, keys, lane).compile()
            dt = t.lap()
            compile_s += dt
            info("compiled chunk", n_updates=n, compile_seconds=round(dt, 2))
        t.lap()
        runner, m = compiled[n](runner, keys, lane)
        m = jax.tree.map(np.asarray, jax.device_get(m))
        dt = t.lap()
        run_s += dt
        done_u += n
        rec = {k: v for k, v in m.items()}
        if ev and (done_u % ev["every_updates"] == 0 or done_u == ppo.U):
            t.lap()
            G, Gu, Lh = jax.device_get(eval_fn(runner, keys, lane))
            es = t.lap()
            eval_s += es
            rec["eval"] = dict(u=done_u, step=done_u * ppo.Mst, disc=np.asarray(G), undisc=np.asarray(Gu),
                               length=np.asarray(Lh))
            if det_eval_fn is not None:
                t.lap()
                Gd, Gud, Lhd = jax.device_get(det_eval_fn(runner, keys, lane))
                eval_s += t.lap()
                rec["eval"].update(det_disc=np.asarray(Gd), det_undisc=np.asarray(Gud), det_length=np.asarray(Lhd))
            evals.append(rec["eval"])
            for arm in cfg["arms"]:
                ii = [i for i, (a, _) in enumerate(names) if a == arm]
                metric("eval", exp=cfg["name"], arm=arm, step=done_u * ppo.Mst,
                       disc_mean=float(np.asarray(G)[ii].mean()), len_mean=float(np.asarray(Lh)[ii].mean()),
                       seconds=round(es, 1))
        if ro_at and done_u in ro_at:
            t.lap()
            vs, al = jax.device_get(ro_fn(runner, keys, lane))
            rs = t.lap()
            eval_s += rs
            vs, al = np.asarray(vs), np.asarray(al)
            rec["readout"] = dict(u=done_u, step=done_u * ppo.Mst, t=ro_t, v=vs[:, :, ro_t].round(4),
                                  full=al.all(-1))
            readouts.append(rec["readout"])
            info("value readout", step=done_u * ppo.Mst, seconds=round(rs, 1))
        chunks.append(rec)
        cnt = m["ep_cnt"].sum(1)
        ret = np.where(cnt > 0, m["ep_ret_sum"].sum(1) / np.maximum(cnt, 1), np.nan)
        for arm in cfg["arms"]:
            ii = [i for i, (a, _) in enumerate(names) if a == arm]
            metric("chunk", exp=cfg["name"], arm=arm, step=done_u * ppo.Mst, ep_return=float(np.nanmean(ret[ii])),
                   ep_len=float(np.nanmean((m["ep_len_sum"].sum(1) / np.maximum(cnt, 1))[ii])),
                   expl_var=float(np.nanmean(m["explained_variance"][ii])),
                   k3=float(m["k3"][ii].mean()), clipfrac=float(m["clipfrac"][ii].mean()))
        if float(np.max(m["ratio_dev_first_mb"])) > 1e-4:
            warn("ratio != 1 on the first minibatch", max=float(np.max(m["ratio_dev_first_mb"])))
        info("throughput", steps_per_s_total=n * ppo.Mst * L / dt, chunk_seconds=round(dt, 2))
        if ckpt and done_u < ppo.U:
            _save(ckpt_dir, runner, done_u, chunks)
        if max_chunks is not None and len(chunks) >= max_chunks and done_u < ppo.U:
            warn("stopping early (max_chunks)", updates_done=done_u)
            return None
    info("timing", exp=cfg["name"], compile_seconds=round(compile_s, 2), steady_seconds=round(run_s, 2),
         eval_seconds=round(eval_s, 2), sps_total_steady=ppo.U * ppo.Mst * L / max(run_s, 1e-9))
    allm = {k: np.concatenate([c[k] for c in chunks], axis=1) for k in chunks[0] if k not in ("eval", "readout")}
    extra = {}
    if cfg.get("readout_T"):
        vs, al = jax.device_get(jax.jit(jax.vmap(partial(ppo.value_readout, n_ep=cfg.get("readout_episodes", 5))))(
            runner, keys, lane))
        extra["readout_values"], extra["readout_alive"] = np.asarray(vs), np.asarray(al)
    results = []
    gs = ((np.arange(ppo.U)[:, None] * ppo.Mst + np.arange(ppo.Mst)[None, :] + 1)).reshape(-1)
    for i, (arm, s) in enumerate(names):
        done = allm["ep_done_steps"][i].reshape(-1) > 0
        episodes = np.stack([gs[done], allm["ep_ret_steps"][i].reshape(-1)[done]], 1)
        per_update = {k: allm[k][i] for k in allm if k not in ("ep_ret_steps", "ep_done_steps")}
        n_upd = ppo.U
        if "active" in per_update:  # ppo1 stop rule: keep this lane's own updates only (inactive ones are trailing)
            act = np.asarray(per_update["active"]) > 0.5
            n_upd = int(act.sum())
            assert act[:n_upd].all() and not act[n_upd:].any()
            per_update = {k: v[:n_upd] for k, v in per_update.items()}
            assert episodes.size == 0 or episodes[:, 0].max() <= n_upd * ppo.Mst
        rec = dict(exp=cfg["name"], arm=arm, seed=s, lane=dict(time_mode=cfg["arms"][arm].get("time_mode", "none"),
                                                                 peb=cfg["arms"][arm].get("peb", 0),
                                                                 T=cfg["arms"][arm]["T"],
                                                                 gamma=cfg["arms"][arm].get("gamma", cfg["gamma"])),
                   git=git_sha_at_start(), config_hash=cfg_hash(cfg), config=cfg, env=ppo.env.name,
                   env_checksum=env_checksum(cfg),
                   episodes_cols=["global_step", "return"], episodes=episodes,
                   per_update=per_update, per_update_global_step=(np.arange(n_upd) + 1) * ppo.Mst,
                   evals=[dict(u=e["u"], step=e["step"], disc=e["disc"][i], undisc=e["undisc"][i],
                               length=e["length"][i],
                               **{k: e[k][i] for k in ("det_disc", "det_undisc", "det_length") if k in e})
                          for e in evals],
                   timing=dict(compile_seconds=compile_s, steady_seconds=run_s, eval_seconds=eval_s, n_lanes=L,
                               device=str(dev)))
        if readouts:
            rec["readouts"] = [dict(u=x["u"], step=x["step"], t=x["t"], v=x["v"][i], full=x["full"][i])
                               for x in readouts]
        for k, v in extra.items():
            rec[k] = v[i]
        write_json(os.path.join(out_dir, arm, f"seed_{s}", "metrics.json"), rec)
        results.append(rec)
    if ckpt and os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    return results


# ===================================================================== tabular experiments
def _tab_arrays(tab):
    return (jnp.asarray(tab.NX, jnp.int32), jnp.asarray(tab.R, jnp.float32),
            jnp.asarray(np.asarray(tab.is_terminal)[np.asarray(tab.NX)], jnp.float32), jnp.asarray(tab.start, jnp.float32))


def train_tabular_q(cfg: dict, out_dir: str):
    """E1 / E2: tabular Q-learning, agents x seeds in one vmapped batch (shared keys per seed)."""
    dev = pick_device(cfg)
    with jax.default_device(dev):
        if cfg["env"] == "two_goal":
            tab = D.two_goal(cfg.get("bump_reward", -1.0))
        elif cfg["env"] == "last_moment":
            tab = D.last_moment()
        else:
            raise ValueError(cfg["env"])
        NX, R, TN, st = _tab_arrays(tab)
        agents = {"standard": A.AGENT_STANDARD, "time_aware": A.AGENT_TA, "peb": A.AGENT_PEB}
        names = [(a, s) for a in cfg["agents"] for s in cfg["seeds"]]
        ag = jnp.asarray([agents[a] for a, _ in names], jnp.int32)
        sd = jnp.asarray([s for _, s in names], jnp.int32)
        f = jax.jit(jax.vmap(lambda s, a: A.tabular_q(exp_key(s, cfg["exp_id"]), NX, R, TN, st, agent=a, T=cfg["T"],
                                                     gamma=cfg["gamma"], eps=cfg["eps"], n_episodes=cfg["episodes"],
                                                     lr_power=cfg["lr_power"])))
        t = Timer()
        out = f(sd, ag)
        jax.block_until_ready(out)
        dt = t.lap()
        info("tabular_q done", exp=cfg["name"], lanes=len(names), seconds_incl_compile=round(dt, 2), device=str(dev))
        return names, jax.device_get(out), tab


def train_buffer_q(cfg: dict, B: int):
    dev = pick_device(cfg)
    with jax.default_device(dev):
        tab = D.difficult_grid()
        NX = jnp.asarray(tab.NX, jnp.int32)
        names = [(peb, s) for peb in cfg["peb"] for s in cfg["seeds"]]
        pv = jnp.asarray([float(p) for p, _ in names], jnp.float32)
        sd = jnp.asarray([s for _, s in names], jnp.int32)
        f = jax.jit(jax.vmap(lambda s, p: A.buffer_q(exp_key(s, cfg["exp_id"]), NX, D.DGRID_GOAL, D.DGRID_START,
                                                    peb=p, B=B, total_steps=cfg["total_steps"],
                                                    bin_steps=cfg["bin_steps"], T=cfg["T"], eps=cfg["eps"],
                                                    lr=cfg["lr"], minibatch=cfg["minibatch"],
                                                    min_buffer=cfg["min_buffer"],
                                                    sequential=cfg["update_mode"] == "sequential")))
        t = Timer()
        lowered = f.lower(sd, pv)
        comp = lowered.compile()
        c_s = t.lap()
        out = comp(sd, pv)
        jax.block_until_ready(out)
        r_s = t.lap()
        info("buffer_q done", B=B, lanes=len(names), compile_seconds=round(c_s, 2), run_seconds=round(r_s, 2),
             steps_per_s_total=cfg["total_steps"] * len(names) / r_s, device=str(dev))
        return names, jax.device_get(out), dict(compile_seconds=c_s, run_seconds=r_s, device=str(dev))


# ===================================================================== CLI
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tag", default="dev")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--no-ckpt", action="store_true")
    a = p.parse_args(argv)
    check_device()
    cfg = load_config(a.config, a.set)
    out_dir = os.path.join(REPO, "runs", a.tag, cfg["name"])
    if cfg["kind"] == "ppo":
        return train_ppo(cfg, out_dir, ckpt=not a.no_ckpt)
    raise SystemExit("tabular experiments are driven by scripts/toys/*.py and scripts/small/*.py")


if __name__ == "__main__":
    main()

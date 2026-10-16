"""PPO training (paper C1-C13, K1-K9, L1-L5) as one jitted update step, scanned over updates and vmapped over seeds.

Usage:
    python -m src.train --config src/configs/classic_control.yaml --arms separate,shared \
        --envs CartPole-v1,Acrobot-v1 --tag small --set total_timesteps=100000 --set seeds=[1,2,3]

One process runs every (env, arm) listed; within each, all seeds are ONE vmapped batch. Arms differ only in the
config keys in ARMS. PRNG key tree the README:
    root      = PRNGKey(master_seed)
    run_key   = fold_in(root, seed)                       # seed = the integer in cfg.seeds (pairs arms by seed)
    k_init    = fold_in(run_key, 0)                       # network init
    k_env     = fold_in(run_key, 1) -> reset: fold_in(k_env, 0) -> split(N);
                                       step:  fold_in(fold_in(k_env, 1), u*M + t) -> split(N)
    k_act     = fold_in(run_key, 2) -> fold_in(., u) -> fold_in(., t)        # one key per vector step
    k_perm    = fold_in(run_key, 3) -> fold_in(., u) -> fold_in(., epoch)     # minibatch permutation
(u = 0-based update index, t = step within the rollout.) A resumed run therefore reproduces the uninterrupted one.
"""
from __future__ import annotations

import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()

import argparse  # noqa: E402
import copy  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from functools import partial  # noqa: E402
from typing import Any, NamedTuple  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import yaml  # noqa: E402

# float32 matmuls in full precision: the RTX 3090 otherwise uses TF32 (~1e-3 relative error), which breaks the
# ratio == 1 check at the first minibatch (debug Q-C) and the 1e-5 differential tests against the CleanRL oracle.
jax.config.update("jax_default_matmul_precision", "highest")
from rl_jax_bench.jaxlog import Timer, check_device, error, info, metric, nan_guard, warn, write_json  # noqa: E402

from . import algorithm as alg  # noqa: E402
from . import model as mdl  # noqa: E402
from .data import VecEnv, VecState  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO, "src", "configs")

# Arms: the ONLY config keys that differ from the experiment's base config.
ARMS: dict[str, dict] = {
    "separate": {"architecture": "separate"},                          # ppo.py (paper C13)
    "shared": {"architecture": "shared"},                              # ppo_shared.py (paper C13 comparison)
    "default": {},                                                     # the config as written
    "advnorm_batch": {"norm_adv_level": "batch"},                      # OUR EXTENSION (C7 ablation)
    "advnorm_ddof0": {"adv_norm_ddof": 0},                             # D1 flag (baselines np.std)
    "no_vclip": {"clip_vloss": False},                                 # OUR EXTENSION (C9 ablation)
    "no_rscale": {"reward_scaling": False},                            # OUR EXTENSION (K8 ablation)
    "trunc_bootstrap": {"bootstrap_truncation": True},                 # OUR EXTENSION (C5 note)
    "gymnasium029": {"reward_scaling_mode": "gymnasium029"},           # D7 flag (current CleanRL wrappers)
    "gym021": {"reward_scaling_mode": "gym021"},                       # D14 flag (the post's own gym-0.21 wrappers)
    "lstm": {},                                                        # all L1-L5
    "lstm_noreset": {"lstm_reset": False},                             # OUR EXTENSION (L3 off, rollout + training)
    "lstm_zero_train": {"lstm_train_from_zero": True},                 # deliberately broken L5 (W12 power check)
    "ff_masked": {"recurrent": False, "architecture": "separate"},     # memoryless control on the masked obs
    "ff_full": {"recurrent": False, "architecture": "separate", "mask_velocity": False},  # full-obs ceiling
    # the post's LSTM recipe on the proxy (post §'5 LSTM implementation details': "we use the Atari hyperparameters
    # again"): atari cliprange=0.1, and ClipRewardEnv (A6, sign) is the identity on CartPole's +1, so no reward
    # normalisation. lstm_eps01 keeps A-HP-1b reward scaling and changes only eps, to separate the two changes.
    "adam_tf1": {"adam_eps_mode": "tf1"},                              # D16 flag (baselines' TF1 Adam epsilon)
    "lstm_atari": {"clip_coef": 0.1, "reward_scaling": False},
    "lstm_eps01": {"clip_coef": 0.1},
}


# ------------------------------------------------------------------ config
def load_config(path: str, overrides=()) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base = load_config(os.path.join(os.path.dirname(path), cfg.pop("base")))
        base.update(cfg)
        cfg = base
    for ov in overrides:
        k, v = ov.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "src"], stderr=subprocess.DEVNULL)
        return sha + ("-dirty" if dirty.strip() else "")
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ state
class Runner(NamedTuple):
    params: Any
    opt_state: Any
    vs: VecState
    obs: jax.Array
    done: jax.Array
    carry: Any
    update: jax.Array


class Consts(NamedTuple):
    k_act: jax.Array
    k_env_step: jax.Array
    k_perm: jax.Array


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array          # stored WITH obs (paper C1)
    trunc: jax.Array         # transition ended by a time limit (not a termination)
    final_value: jax.Array   # V(final obs), only used by the truncation-bootstrap flag
    ep_ret: jax.Array
    ep_len: jax.Array
    ep_done: jax.Array


class PPO:
    """Builds the pure functions for one config (one env x one arm)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg = copy.deepcopy(cfg)
        assert cfg["target_kl"] is None, "early stopping (X3) is off by default and not implemented"
        self.env = VecEnv(cfg)
        self.spec = self.env.spec
        self.cont = self.spec.continuous
        self.recurrent = bool(cfg["recurrent"])
        if self.recurrent:
            assert cfg["architecture"] == "lstm" and not cfg["bootstrap_truncation"]
        self.net = mdl.build_network(cfg, self.spec.n_out, self.cont)
        self.N, self.M = cfg["num_envs"], cfg["num_steps"]
        self.batch = self.N * self.M
        self.n_mb, self.K = cfg["num_minibatches"], cfg["update_epochs"]
        if self.recurrent:
            assert self.N % self.n_mb == 0
            self.envs_per_mb = self.N // self.n_mb
        else:
            assert self.batch % self.n_mb == 0
            self.mb_size = self.batch // self.n_mb
        self.U = cfg["total_timesteps"] // self.batch  # official code: num_iterations = total // batch
        self.tx = alg.make_optimizer(cfg["learning_rate"], cfg["max_grad_norm"], cfg["adam_eps"],
                                     cfg.get("adam_eps_mode", "torch"))

    # -------------------------------------------------------------- init
    def init(self, seed):
        cfg = self.cfg
        run_key = jax.random.fold_in(jax.random.PRNGKey(cfg["master_seed"]), seed)
        k_init, k_env, k_act, k_perm = (jax.random.fold_in(run_key, i) for i in range(4))
        d = self.spec.obs_dim
        if self.recurrent:
            carry = self.net.initial_carry(self.N)
            params = self.net.init(k_init, jnp.zeros((1, self.N, d)), jnp.zeros((1, self.N)), carry)
        else:
            carry = (jnp.zeros((self.N, 1)), jnp.zeros((self.N, 1)))  # unused placeholder
            params = self.net.init(k_init, jnp.zeros((1, d)))
        opt_state = self.tx.init(params)
        vs, obs = self.env.reset(jax.random.fold_in(k_env, 0))
        runner = Runner(params, opt_state, vs, obs, jnp.zeros((self.N,), jnp.float32), carry, jnp.int32(0))
        return runner, Consts(k_act, jax.random.fold_in(k_env, 1), k_perm)

    # -------------------------------------------------------------- policy
    def _act(self, params, obs, done, carry, key):
        if self.recurrent:
            logits, v, carry = self.net.apply(params, obs[None], done[None], carry)
            dist, v = logits[0], v[0]
        else:
            dist, v = self.net.apply(params, obs)
        a = mdl.sample(key, dist, self.cont)
        return a, mdl.log_prob(dist, a, self.cont), v, carry

    def _value(self, params, obs, done, carry):
        if self.recurrent:
            return self.net.apply(params, obs[None], done[None], carry)[1][0]
        return self.net.apply(params, obs)[1]

    # -------------------------------------------------------------- rollout (paper C1)
    def rollout(self, runner: Runner, consts: Consts):
        u = runner.update
        k_act_u = jax.random.fold_in(consts.k_act, u)

        def step(c, t):
            vs, obs, done, carry = c
            a, lp, v, carry = self._act(runner.params, obs, done, carry, jax.random.fold_in(k_act_u, t))
            vs, so = self.env.step(jax.random.fold_in(consts.k_env_step, u * self.M + t), vs, a)
            if self.cfg["bootstrap_truncation"]:
                vfin = self._value(runner.params, so.final_obs, None, None)
            else:
                vfin = jnp.zeros_like(v)
            tr = Transition(obs, a, lp, v, so.reward, done, so.truncated.astype(jnp.float32), vfin,
                            so.ep_ret_done, so.ep_len_done, so.done.astype(jnp.float32))
            return (vs, so.obs, so.done.astype(jnp.float32), carry), tr

        init_carry = runner.carry
        (vs, obs, done, carry), traj = jax.lax.scan(step, (runner.vs, runner.obs, runner.done, runner.carry),
                                                    jnp.arange(self.M))
        return vs, obs, done, carry, init_carry, traj

    # -------------------------------------------------------------- loss
    def _loss(self, params, mb, init_carry=None):
        cfg = self.cfg
        if self.recurrent:
            if cfg["lstm_train_from_zero"]:
                init_carry = jax.tree.map(jnp.zeros_like, init_carry)
            dist, v, _ = self.net.apply(params, mb.obs, mb.done, init_carry)  # paper L5: from the saved state
        else:
            dist, v = self.net.apply(params, mb.obs)
        new_logp = mdl.log_prob(dist, mb.action, self.cont)
        ent = mdl.entropy(dist, self.cont)
        flat = lambda x: x.reshape(-1)  # noqa: E731
        return alg.ppo_terms(flat(new_logp), flat(mb.logp), flat(ent), flat(v), flat(mb.value), flat(mb.ret),
                             flat(mb.adv), clip_coef=cfg["clip_coef"], clip_vloss=cfg["clip_vloss"],
                             ent_coef=cfg["ent_coef"], vf_coef=cfg["vf_coef"],
                             norm_adv_mb=cfg["norm_adv"] and cfg["norm_adv_level"] == "minibatch",
                             ddof=cfg["adv_norm_ddof"])

    def sgd_step(self, params, opt_state, mb, init_carry=None):
        """One minibatch step: loss -> grads -> global-norm clip (C11) -> Adam (C3). mb has fields obs, action, logp,
        value, ret, adv, done (flat for feed-forward; (M, E, ...) env columns for the LSTM)."""
        (loss, aux), g = jax.value_and_grad(self._loss, has_aux=True)(params, mb, init_carry)
        aux = dict(aux, loss=loss, grad_norm=optax.tree.norm(g))
        upd, opt_state = self.tx.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, aux

    # -------------------------------------------------------------- one PPO update
    def update(self, runner: Runner, consts: Consts):
        cfg = self.cfg
        u = runner.update
        lr = alg.lr_at(u, self.U, cfg["learning_rate"], cfg["anneal_lr"])  # paper C4
        opt_state = alg.set_lr(runner.opt_state, lr)
        vs, next_obs, next_done, carry, init_carry, traj = self.rollout(runner, consts)
        next_value = self._value(runner.params, next_obs, next_done, carry)
        tb = cfg["bootstrap_truncation"]
        adv, ret = alg.gae(traj.reward, traj.value, traj.done, next_value, next_done, cfg["gamma"],
                           cfg["gae_lambda"], traj.trunc if tb else None, traj.final_value if tb else None)
        if cfg["norm_adv"] and cfg["norm_adv_level"] == "batch":  # OUR EXTENSION: per-batch normalisation
            adv = alg.normalize_adv(adv, cfg["adv_norm_ddof"]).reshape(adv.shape)
        k_perm_u = jax.random.fold_in(consts.k_perm, u)

        class MB(NamedTuple):
            obs: Any
            action: Any
            logp: Any
            value: Any
            ret: Any
            adv: Any
            done: Any

        data = MB(traj.obs, traj.action, traj.logp, traj.value, ret, adv, traj.done)  # leaves (M, N, ...)
        apply_grad = self.sgd_step

        if self.recurrent:  # paper L4: minibatches of whole env columns, all M steps in order
            def epoch(c, e):
                params, opt_state = c
                envinds = jax.random.permutation(jax.random.fold_in(k_perm_u, e), self.N)

                def mb_step(c2, ids):
                    params, opt_state = c2
                    mb = jax.tree.map(lambda x: x[:, ids], data)
                    ic = jax.tree.map(lambda x: x[ids], init_carry)
                    params, opt_state, aux = apply_grad(params, opt_state, mb, ic)
                    return (params, opt_state), aux

                return jax.lax.scan(mb_step, (params, opt_state), envinds.reshape(self.n_mb, self.envs_per_mb))
        else:  # paper C6: fresh permutation per epoch, contiguous chunks
            flat = jax.tree.map(lambda x: x.reshape((self.batch,) + x.shape[2:]), data)

            def epoch(c, e):
                params, opt_state = c
                perm = jax.random.permutation(jax.random.fold_in(k_perm_u, e), self.batch)

                def mb_step(c2, ids):
                    params, opt_state = c2
                    mb = jax.tree.map(lambda x: x[ids], flat)
                    params, opt_state, aux = apply_grad(params, opt_state, mb, None)
                    return (params, opt_state), aux

                return jax.lax.scan(mb_step, (params, opt_state), perm.reshape(self.n_mb, self.mb_size))

        (params, opt_state), aux = jax.lax.scan(epoch, (runner.params, opt_state), jnp.arange(self.K))
        m = {k: v.mean() for k, v in aux.items() if k != "max_ratio_dev"}
        m["ratio_dev_first_mb"] = aux["max_ratio_dev"][0, 0]  # debug Q-C: must be ~0 (float32 noise)
        m["k3_last"] = aux["k3"][-1, -1]                       # CleanRL logs approx_kl of the last minibatch
        # recheck M3: the post's code logs losses/{policy_loss,value_loss,entropy} from the LAST minibatch of the
        # last epoch too; pg_loss / v_loss / entropy above stay the all-minibatch means (pre-fix fields, diagnostic)
        for k in ("pg_loss", "v_loss", "entropy"):
            m[k + "_last"] = aux[k][-1, -1]
        m["lr"] = lr
        m["explained_variance"] = alg.explained_variance(traj.value.reshape(-1), ret.reshape(-1))
        m["ep_ret_sum"] = traj.ep_ret.sum(1)      # (M) raw returns of episodes finishing at step t, summed over N
        m["ep_cnt"] = traj.ep_done.sum(1)         # (M)
        m["ep_len_sum"] = traj.ep_len.sum(1)
        if self.cont:
            m["log_std_mean"] = jax.tree.leaves(params["params"]["log_std"])[0].mean()
        nan_guard({"params": params, "loss": m["loss"]}, {"m": m}, tag=0,
                  dump_dir=os.path.join(REPO, "runs", "nan_dumps"))
        new = Runner(params, opt_state, vs, next_obs, next_done, carry, u + 1)
        return new, m

    def run_chunk(self, runner, consts, n: int):
        return jax.lax.scan(lambda r, _: self.update(r, consts), runner, None, length=n)


# ------------------------------------------------------------------ host driver
def global_steps(cfg: dict, U: int) -> np.ndarray:
    """Global step at which an episode finishing at (update u, step t) is logged: (u*M + t + 1) * N (CleanRL)."""
    M, N = cfg["num_steps"], cfg["num_envs"]
    return ((np.arange(U)[:, None] * M + np.arange(M)[None, :] + 1) * N)


def train_one(cfg: dict, arm: str, out_root: str, ckpt: bool = True, max_chunks: int | None = None) -> dict | None:
    """Train all seeds of one (env, arm). max_chunks (testing only) stops early, as a kill would, leaving the
    checkpoint for a later call to resume from."""
    from . import eval as ev
    seeds = list(cfg["seeds"])
    ppo = PPO(cfg)
    tag = f"{cfg['env_id']}/{arm}"
    info("build", tag=tag, seeds=len(seeds), updates=ppo.U, batch=ppo.batch, obs_dim=ppo.spec.obs_dim,
         continuous=ppo.cont, recurrent=ppo.recurrent)
    t = Timer()
    init = jax.jit(jax.vmap(ppo.init))
    runner, consts = init(jnp.asarray(seeds, jnp.int32))
    jax.block_until_ready(runner)
    info("init done", seconds=t.lap())
    run_dir = os.path.join(out_root, cfg["env_id"], arm)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    done_u, chunks = 0, []
    if ckpt and os.path.isdir(ckpt_dir):
        runner, done_u, chunks = _restore(ckpt_dir, runner)
        if done_u:
            info("resumed from checkpoint", updates_done=done_u)
    compiled: dict[int, Any] = {}
    compile_s, run_s = 0.0, 0.0
    C = cfg["chunk_updates"]
    while done_u < ppo.U:
        n = min(C, ppo.U - done_u)
        if n not in compiled:
            t.lap()
            compiled[n] = jax.jit(jax.vmap(partial(ppo.run_chunk, n=n))).lower(runner, consts).compile()
            dt = t.lap()
            compile_s += dt
            info("compiled chunk", n_updates=n, compile_seconds=dt)
        t.lap()
        runner, m = compiled[n](runner, consts)
        m = jax.tree.map(np.asarray, jax.device_get(m))
        dt = t.lap()
        run_s += dt
        chunks.append(m)
        done_u += n
        steps = done_u * ppo.batch
        cnt = m["ep_cnt"].sum(axis=(1, 2))
        rec = np.where(cnt > 0, m["ep_ret_sum"].sum(axis=(1, 2)) / np.maximum(cnt, 1), np.nan)
        k3max = float(m["k3_last"].max())
        metric("chunk", tag=tag, global_step=steps, ep_return_mean=float(np.nanmean(rec)),
               ep_return_min_seed=float(np.nanmin(rec)), ep_return_max_seed=float(np.nanmax(rec)),
               k3=float(m["k3"].mean()), clipfrac=float(m["clipfrac"].mean()),
               expl_var=float(np.nanmean(m["explained_variance"])),
               ratio_dev_first_mb=float(m["ratio_dev_first_mb"].max()),
               sps_total=n * ppo.batch * len(seeds) / dt)
        # the post's "approx_kl stays below 0.02" does not name the estimator; k3 is inferred from the post's ppo.py,
        # which logs k3 under the name approx_kl the README)
        frac_kl = float((m["k3_last"] > 0.02).mean())
        if frac_kl > 0:
            warn("approx_kl (k3, inferred estimator) > 0.02 (post rule of thumb, Q-C)", tag=tag, frac_updates=frac_kl, max=k3max)
        if float(m["ratio_dev_first_mb"].max()) > 1e-5:
            warn("ratio != 1 on the first minibatch (debug Q-C)", tag=tag, max=float(m["ratio_dev_first_mb"].max()))
        if ckpt and done_u < ppo.U:
            _save(ckpt_dir, runner, done_u, chunks)
        if max_chunks is not None and len(chunks) >= max_chunks and done_u < ppo.U:
            warn("stopping early (max_chunks, simulated interruption)", updates_done=done_u)
            return None
    allm = {k: np.concatenate([c[k] for c in chunks], axis=1) for k in chunks[0]}  # (S, U, ...)
    sps = ppo.U * ppo.batch * len(seeds) / max(run_s, 1e-9)
    info("timing", tag=tag, compile_seconds=compile_s, steady_seconds=run_s, sps_total_steady=sps,
         sps_per_seed=sps / len(seeds))
    gsteps = global_steps(cfg, ppo.U)
    summaries = []
    greedy = ev.greedy_eval(ppo, runner, n_episodes=10)
    for i, s in enumerate(seeds):
        cnt = allm["ep_cnt"][i]
        nz = cnt > 0
        episodes = np.stack([gsteps[nz], allm["ep_ret_sum"][i][nz], cnt[nz], allm["ep_len_sum"][i][nz]], 1)
        summ = ev.summarize(episodes, cfg["total_timesteps"], cfg["env_id"])
        summ["greedy_eval_mean"] = float(greedy[i].mean())
        per_update = {k: allm[k][i] for k in allm if k not in ("ep_ret_sum", "ep_cnt", "ep_len_sum")}
        rec = dict(env_id=cfg["env_id"], arm=arm, seed=s, git=git_sha(), config=cfg, summary=summ,
                   episodes_cols=["global_step", "return_sum", "count", "length_sum"],
                   episodes=episodes.tolist(), per_update=per_update,
                   per_update_global_step=((np.arange(ppo.U) + 1) * ppo.batch).tolist(),
                   greedy_eval_returns=greedy[i].tolist(),
                   timing=dict(compile_seconds=compile_s, steady_seconds=run_s, n_seeds_vmapped=len(seeds),
                               sps_per_seed=sps / len(seeds), device=str(jax.devices()[0])))
        write_json(os.path.join(run_dir, f"seed_{s}", "metrics.json"), rec)
        summaries.append(summ)
    fw = np.array([x["final_window"] for x in summaries])
    metric("final", tag=tag, final_window_mean=float(np.nanmean(fw)),
           final_window_se=float(np.nanstd(fw, ddof=1) / np.sqrt(len(fw))) if len(fw) > 1 else float("nan"),
           auc_mean=float(np.mean([x["auc"] for x in summaries])), greedy_mean=float(greedy.mean()))
    if ckpt and os.path.isdir(ckpt_dir):
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    return dict(tag=tag, summaries=summaries, compile_seconds=compile_s, steady_seconds=run_s)


def _save(ckpt_dir, runner, done_u, chunks):
    import orbax.checkpoint as ocp
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(ckpt_dir, f"u{done_u:06d}"))
    with ocp.StandardCheckpointer() as cp:
        cp.save(path, runner)
    np.savez(os.path.join(ckpt_dir, f"u{done_u:06d}_metrics.npz"),
             **{f"{i}__{k}": v for i, c in enumerate(chunks) for k, v in c.items()})
    for f in sorted(os.listdir(ckpt_dir)):  # keep only the newest
        if not f.startswith(f"u{done_u:06d}"):
            p = os.path.join(ckpt_dir, f)
            import shutil
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)


def _restore(ckpt_dir, runner_like):
    import orbax.checkpoint as ocp
    dirs = sorted(d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d)))
    if not dirs:
        return runner_like, 0, []
    last = dirs[-1]
    done_u = int(last[1:])
    with ocp.StandardCheckpointer() as cp:
        runner = cp.restore(os.path.abspath(os.path.join(ckpt_dir, last)), runner_like)
    z = np.load(os.path.join(ckpt_dir, f"{last}_metrics.npz"))
    chunks: dict[int, dict] = {}
    for key in z.files:
        i, k = key.split("__", 1)
        chunks.setdefault(int(i), {})[k] = z[key]
    return runner, done_u, [chunks[i] for i in sorted(chunks)]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(CONFIG_DIR, "classic_control.yaml"))
    p.add_argument("--arms", default="default")
    p.add_argument("--envs", default=None, help="comma list of env_ids (default: the config's)")
    p.add_argument("--tag", default="dev")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--no-ckpt", action="store_true")
    a = p.parse_args(argv)
    check_device()
    base = load_config(a.config, a.set)
    envs = a.envs.split(",") if a.envs else [base["env_id"]]
    out_root = os.path.join(REPO, "runs", a.tag)
    results = []
    for env_id in envs:
        for arm in a.arms.split(","):
            cfg = dict(base, env_id=env_id, **ARMS[arm])
            # re-apply explicit --set overrides on top of the arm (they win)
            cfg = load_config_overrides(cfg, a.set)
            try:
                results.append(train_one(cfg, arm, out_root, ckpt=not a.no_ckpt))
            except Exception as e:  # noqa: BLE001
                error("run failed", env=env_id, arm=arm, err=repr(e))
                raise
    write_json(os.path.join(out_root, "index.json"), {"results": results})
    return results


def load_config_overrides(cfg: dict, overrides) -> dict:
    cfg = dict(cfg)
    for ov in overrides:
        k, v = ov.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


if __name__ == "__main__":
    main()

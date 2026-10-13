"""Recheck, OUR side: multi-update teacher-forced export for the FEED-FORWARD
paths (separate / shared / Gaussian), closing the coverage gap states ("the epoch x minibatch loop with its
shuffling is not diff-tested step by step"). Same design as scripts/review/lstm_multiupdate_export.py: run U whole
PPO updates of OUR trainer from a fixed init with the paper-scale configs, and record what the authors' code needs to
replay them teacher-forced: the initial weights (torch layout), per update the rollout obs / dones / actions /
training rewards, next_obs / next_done, OUR per-epoch flat-batch permutations, and our outputs.
Used by tests/test_ff_multiupdate_parity.py. Run: uv run python -m scripts.recheck.ff_multiupdate_export
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src.train import ARMS, CONFIG_DIR, PPO, load_config  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts.small.difftest_export import params_flat, torch_sd  # noqa: E402
from src import algorithm as alg  # noqa: E402

OUT = os.path.join(REPO, "runs", "recheck", "official-oracle", "ff_multi")

CASES = {
    # name: (config file, arm, env_id, U, seed)
    "separate_cartpole": ("classic_control.yaml", "separate", "CartPole-v1", 6, 3),
    "shared_cartpole": ("classic_control.yaml", "shared", "CartPole-v1", 6, 3),
    "separate_acrobot_novclip": ("classic_control.yaml", "no_vclip", "Acrobot-v1", 6, 9),
    "gauss_pendulum": ("pendulum.yaml", "default", "Pendulum-v1", 3, 2),
}


def export(name):
    cfgf, arm, env_id, U, seed = CASES[name]
    cfg = load_config(os.path.join(CONFIG_DIR, cfgf))
    cfg.update(env_id=env_id, **ARMS[arm])
    cfg["total_timesteps"] = 50 * cfg["num_envs"] * cfg["num_steps"]  # lr schedule over 50 updates; we run U of them
    ppo = PPO(cfg)
    arch, cont = cfg["architecture"], ppo.cont
    runner, consts = ppo.init(jnp.int32(seed))
    out = torch_sd(runner.params, arch, cont)
    out.update({"meta/U": U, "meta/N": ppo.N, "meta/M": ppo.M, "meta/n_mb": ppo.n_mb, "meta/K": ppo.K,
                "meta/num_iterations": ppo.U, "meta/lr0": cfg["learning_rate"], "meta/clip_coef": cfg["clip_coef"],
                "meta/ent_coef": cfg["ent_coef"], "meta/vf_coef": cfg["vf_coef"], "meta/gamma": cfg["gamma"],
                "meta/lam": cfg["gae_lambda"], "meta/max_grad_norm": cfg["max_grad_norm"],
                "meta/clip_vloss": int(cfg["clip_vloss"]), "meta/arch": arch, "meta/cont": int(cont),
                "meta/obs_dim": ppo.spec.obs_dim, "meta/n_out": ppo.spec.n_out})
    upd = jax.jit(ppo.update)
    for u in range(U):
        vs, next_obs, next_done, carry, _, traj = ppo.rollout(runner, consts)
        next_value = ppo._value(runner.params, next_obs, next_done, carry)
        adv, ret = alg.gae(traj.reward, traj.value, traj.done, next_value, next_done, cfg["gamma"], cfg["gae_lambda"])
        k_perm_u = jax.random.fold_in(consts.k_perm, u)
        perms = jnp.stack([jax.random.permutation(jax.random.fold_in(k_perm_u, e), ppo.batch) for e in range(ppo.K)])
        new, m = upd(runner, consts)
        p = f"u{u}/"
        out.update({p + "obs": traj.obs, p + "done": traj.done, p + "action": traj.action, p + "reward": traj.reward,
                    p + "next_obs": next_obs, p + "next_done": next_done, p + "perms": perms,
                    p + "logp": traj.logp, p + "value": traj.value, p + "adv": adv, p + "ret": ret, p + "lr": m["lr"],
                    p + "clipfrac": m["clipfrac"], p + "k3_last": m["k3_last"], p + "v_loss": m["v_loss"],
                    p + "pg_loss": m["pg_loss"], p + "pg_loss_last": m["pg_loss_last"],
                    p + "v_loss_last": m["v_loss_last"]})
        out.update(params_flat(new.params, arch, cont, p + "params_after"))
        runner = new
    return {k: np.asarray(v) for k, v in out.items()}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name in (sys.argv[1:] or CASES):
        o = export(name)
        np.savez(os.path.join(OUT, f"ours_{name}.npz"), **o)
        U = int(o["meta/U"])
        print(f"[INFO] {name}: U={U} episode ends={int(sum(o[f'u{u}/done'].sum() for u in range(U)))} "
              f"clipfrac={[float(o[f'u{u}/clipfrac']) for u in range(U)]}")

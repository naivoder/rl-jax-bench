"""follow-up: multi-update teacher-forced differential test of the LSTM path, our side.

Runs our PPO-LSTM (masked CartPole, dict shrunk to N=4, M=32, 2 minibatches, K=4) for U=3 whole updates from
a fixed init, and records everything the authors' code needs to replay the same updates teacher-forced: the initial
weights (torch layout), and per update the rollout obs / done flags / actions / training rewards, the final next_obs /
next_done, the per-epoch env-column permutations, and our outputs (rollout log-probs, values, the LSTM state carried
out of the rollout, advantages, returns, and the weights after the update). This covers the paths the single-minibatch
LSTM parity test does not: the per-step rollout carry, the carry ACROSS updates (initial_lstm_state of update u+1 is
the rollout's final state of update u), GAE with the LSTM bootstrap value, K epochs x all minibatches of Adam, and lr
annealing. The env and the reward wrapper are teacher-forced (both are parity-tested separately).
Run: uv run python -m scripts.review.lstm_multiupdate_export [lstm_reset 0|1]
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src.train import CONFIG_DIR, PPO, load_config  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts.small.difftest_export import params_flat, torch_sd  # noqa: E402
from src import algorithm as alg  # noqa: E402

OUT = os.path.join(REPO, "runs", "review_lstm_multi", "difftest")


def export(reset: bool, seed: int = 3, U: int = 3) -> dict:
    cfg = load_config(os.path.join(CONFIG_DIR, "lstm_masked_cartpole.yaml"))
    cfg.update(num_envs=4, num_steps=32, num_minibatches=2, update_epochs=4, lstm_reset=reset)
    cfg["total_timesteps"] = U * cfg["num_envs"] * cfg["num_steps"]
    ppo = PPO(cfg)
    runner, consts = ppo.init(jnp.int32(seed))
    out = torch_sd(runner.params, "lstm", False)
    out.update({"meta/U": U, "meta/N": ppo.N, "meta/M": ppo.M, "meta/n_mb": ppo.n_mb, "meta/K": ppo.K,
                "meta/lr0": cfg["learning_rate"], "meta/clip_coef": cfg["clip_coef"], "meta/ent_coef": cfg["ent_coef"],
                "meta/vf_coef": cfg["vf_coef"], "meta/gamma": cfg["gamma"], "meta/lam": cfg["gae_lambda"],
                "meta/max_grad_norm": cfg["max_grad_norm"], "meta/reset": int(reset)})
    upd = jax.jit(ppo.update)
    for u in range(U):
        assert int(runner.update) == u
        vs, next_obs, next_done, carry, init_carry, traj = ppo.rollout(runner, consts)
        next_value = ppo._value(runner.params, next_obs, next_done, carry)
        adv, ret = alg.gae(traj.reward, traj.value, traj.done, next_value, next_done, cfg["gamma"], cfg["gae_lambda"])
        k_perm_u = jax.random.fold_in(consts.k_perm, u)
        perms = jnp.stack([jax.random.permutation(jax.random.fold_in(k_perm_u, e), ppo.N) for e in range(ppo.K)])
        new, m = upd(runner, consts)
        assert np.allclose(np.asarray(new.carry[0]), np.asarray(carry[0]))  # the carry handed to update u+1
        p = f"u{u}/"
        out.update({p + "obs": traj.obs, p + "done": traj.done, p + "action": traj.action, p + "reward": traj.reward,
                    p + "next_obs": next_obs, p + "next_done": next_done, p + "perms": perms,
                    p + "init_h": init_carry[0], p + "init_c": init_carry[1],
                    p + "logp": traj.logp, p + "value": traj.value, p + "final_h": carry[0], p + "final_c": carry[1],
                    p + "adv": adv, p + "ret": ret, p + "lr": m["lr"]})
        out.update(params_flat(new.params, "lstm", False, p + "params_after"))
        runner = new
    return {k: np.asarray(v) for k, v in out.items()}


def main(reset: bool):
    out = export(reset)
    U = int(out["meta/U"])
    os.makedirs(OUT, exist_ok=True)
    tag = "reset" if reset else "noreset"
    np.savez(os.path.join(OUT, f"ours_lstm_multi_{tag}.npz"), **{k: np.asarray(v) for k, v in out.items()})
    n_d = int(sum(np.asarray(out[f"u{u}/done"]).sum() for u in range(U)))
    print(f"[INFO] wrote ours_lstm_multi_{tag}.npz keys={len(out)} episode starts inside rollouts={n_d}")


if __name__ == "__main__":
    main(bool(int(sys.argv[1])) if len(sys.argv) > 1 else True)

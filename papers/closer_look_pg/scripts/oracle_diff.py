"""Differential test against the official PyTorch primitives (standing rule 2).

Records a real 2048-step MJX Hopper batch with the JAX code, runs the official functions on the identical batch
in the isolated oracle venv ($OFFICIAL_ROOT/.venv, scripts/official/oracle_harness.py), and compares:
GAE/returns, both MRE formulas and MSEs, Gaussian log-lik and KL, the normalised surrogate and its gradient, CG,
one full PPO step (10 epochs x 32 minibatches, replayed permutations), one value step, one TRPO step (replayed
Fisher subset) and the ZFilter/RewardFilter normalisers. Writes runs/oracle/oracle_diff.json.

Usage: JAX_PLATFORMS=cpu python scripts/oracle_diff.py
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402  (bootstrap first)
from src.common import info, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
from src.configs.default import algo_config  # noqa: E402
from src.model import Nets, gauss_kl  # noqa: E402
from src.train import batch_from_traj  # noqa: E402

ORACLE = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official"))
HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "official")
OUT = os.path.join(common.REPO, "runs", "oracle")


def layers(nets, flat, prefix, policy=True):
    d = (nets.pi_unravel if policy else nets.v_unravel)(flat)["params"]
    sub = d["mean"] if policy else d["v"]
    out = {}
    for i in range(3):
        out[f"{prefix}W{i}"] = np.asarray(sub[f"Dense_{i}"]["kernel"], np.float32)
        out[f"{prefix}b{i}"] = np.asarray(sub[f"Dense_{i}"]["bias"], np.float32)
    if policy:
        out[f"{prefix}log_std"] = np.asarray(d["log_std"], np.float32)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = algo_config("trpo", "hopper")       # xavier init (the harness builds xavier nets, then overwrites weights)
    env, spec = D.make_env("hopper", cfg["physics_backend"])
    nets = Nets(spec.obs_dim, spec.act_dim, cfg)
    key = jax.random.PRNGKey(123)
    ks = jax.random.split(key, 12)
    pi, v = nets.init(ks[0])
    pi = pi + 0.05 * jax.random.normal(ks[1], pi.shape)          # move off init (non-zero log-std etc.)
    rollout = jax.jit(D.make_rollout(env, cfg, nets), static_argnums=(3, 4))
    norm = D.norm_init(spec.obs_dim)
    _, norm = rollout(pi, norm, ks[2], 2048, True)             # warm normalisers
    traj, _ = rollout(pi, norm, ks[3], 2048, False)
    b = batch_from_traj(nets, cfg, v, traj)
    n = b.obs.shape[0]
    v_new = b.values + 0.3 * jax.random.normal(ks[4], (n,))
    pi2 = pi + 0.02 * jax.random.normal(ks[5], pi.shape)
    logp_old = b.logp + 0.01 * jax.random.normal(ks[6], (n,))  # ratio != 1 so the clip/ratio paths are exercised
    M = jax.random.normal(ks[7], (30, 30))
    H = M @ M.T + 3 * jnp.eye(30)
    cg_b = jax.random.normal(ks[8], (30,))
    # ours: one PPO step / value step / TRPO step with known keys
    ppo_cfg = algo_config("ppo", "hopper")
    ppo_cfg["init"] = "xavier"
    ppo_lr, v_lr = 4.5e-4, 2e-4
    k_ppo, k_v, k_trpo = ks[9], ks[10], ks[11]
    pi_tx, v_tx = A.make_optimizers({**ppo_cfg, "value_lr": v_lr})
    bb = A.Batch(b.obs, b.act, logp_old, b.adv, b.ret, b.values, b.not_done)
    pi_ppo, _, _ = A.ppo_policy_update(nets, ppo_cfg, pi, pi_tx.init(pi), pi_tx, bb, ppo_lr, k_ppo)
    v_cfg = {**ppo_cfg, "value_clipping": False, "value_lr": v_lr}
    v_step, _, _ = A.value_update(nets, v_cfg, v, v_tx.init(v), v_tx, bb, v_lr, k_v)
    pi_trpo, trpo_st = A.trpo_policy_update(nets, cfg, pi, bb, k_trpo)
    ppo_perms = np.asarray(A._minibatch_perms(k_ppo, n, 10, 32)).reshape(10, n)
    v_perms = np.asarray(A._minibatch_perms(k_v, n, 10, 32)).reshape(10, n)
    subset = np.asarray(jax.random.choice(k_trpo, n, (int(n * 0.1),), replace=False))
    # normaliser stream
    stream_obs = np.asarray(traj.obs[:300] * 3.0 + 1.0)
    stream_rew = np.asarray(traj.rew_raw[:300])
    stream_reset = np.zeros(300, bool); stream_reset[[0, 37, 150, 151, 299]] = True

    def zstream():
        rs, rr = D.runstat_init((spec.obs_dim,)), D.runstat_init(())
        acc = 0.0
        zo, ro = [], []
        for x, r, reset in zip(stream_obs, stream_rew, stream_reset):
            if reset:
                acc = 0.0
            rs, y = D.obs_filter(rs, jnp.asarray(x), 10.0, True)
            acc = acc * cfg["gamma"] + r
            rr = D.runstat_push(rr, jnp.float32(acc))
            ro.append(float(jnp.clip(r / (D.runstat_std(rr) + 1e-8), -10, 10)))
            zo.append(np.asarray(y))
        return np.asarray(zo), np.asarray(ro)

    z_obs, z_rew = zstream()
    vals = A.value_errors(v_new, b.values, b.adv, b.ret, b.not_done, False)
    vals_c = A.value_errors(v_new, b.values, b.adv, b.ret, b.not_done, True)
    g_ours = A.pg_vector(nets, pi, b.obs, b.act, b.adv)       # at ratio 1
    # surrogate gradient with logp_old != logp (oracle computes this one)
    def surr(p):
        return jnp.mean(jnp.exp(nets.logp(p, b.obs, b.act) - logp_old) * A.adv_normalize(b.adv))
    s_val, g_surr = jax.value_and_grad(surr)(pi)
    m1, l1 = nets.pi_apply(pi, b.obs)
    m2, l2 = nets.pi_apply(pi2, b.obs)
    inp = dict(obs=b.obs, act=b.act, rew=traj.rew, values=b.values, not_done=b.not_done, gamma=cfg["gamma"],
               lam=cfg["gae_lambda"], v_new=v_new, ret_ours=b.ret, adv_ours=b.adv, logp_old=logp_old, H=H, cg_b=cg_b,
               ppo_epochs=10, n_mb=32, ppo_lr=ppo_lr, ppo_perms=ppo_perms, v_epochs=10, v_lr=v_lr, v_clip=False,
               v_perms=v_perms, damping=cfg["cg_damping"], max_kl=cfg["kl_delta"], fisher_subset=subset,
               stream_obs=stream_obs, stream_rew=stream_rew, stream_reset=stream_reset,
               keys=np.stack([np.asarray(k_ppo), np.asarray(k_v), np.asarray(k_trpo)]),
               **layers(nets, pi, "pi_"), **layers(nets, pi2, "pi2_"), **layers(nets, v, "v_", policy=False))
    inp = {k: np.asarray(x) for k, x in inp.items()}
    np.savez(os.path.join(OUT, "inputs.npz"), **inp)
    py = os.path.join(ORACLE, ".venv", "bin", "python")
    res = subprocess.run([py, "-I", os.path.join(HARNESS, "oracle_harness.py"), os.path.join(OUT, "inputs.npz"),
                          os.path.join(OUT, "outputs.npz")], capture_output=True, text=True)
    info("oracle", stdout=res.stdout.strip()[-200:], rc=res.returncode)
    if res.returncode != 0:
        print(res.stderr[-3000:])
        raise SystemExit(1)
    o = dict(np.load(os.path.join(OUT, "outputs.npz")))
    # commit-able fixtures for tests/test_official_parity.py (inputs recorded here, outputs from the official code)
    # (not when run from tests/test_oracle.py: a test must not rewrite the committed fixtures it is checked against)
    if os.environ.get("ORACLE_WRITE_FIXTURES", "1") == "1":
        fix = os.path.join(common.REPO, "tests", "fixtures", "official")
        os.makedirs(fix, exist_ok=True)
        np.savez_compressed(os.path.join(fix, "primitives_in.npz"), **inp)
        np.savez_compressed(os.path.join(fix, "primitives_out.npz"), **o)

    def cmp(name, ours, theirs, tol):
        ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
        err = float(np.max(np.abs(ours - theirs)))
        rel = err / max(float(np.max(np.abs(theirs))), 1e-12)
        return name, dict(max_abs=err, max_rel=rel, tol=tol, ok=bool(err <= tol or rel <= tol))

    def cmp_layers(name, flat, prefix, tol, policy=True, grads=False):
        ours = layers(nets, flat, "x_", policy)
        errs = []
        for i in range(3):
            errs.append(np.max(np.abs(ours[f"x_W{i}"] - o[f"{prefix}W{i}"])))
            errs.append(np.max(np.abs(ours[f"x_b{i}"] - o[f"{prefix}b{i}"])))
        if policy:
            errs.append(np.max(np.abs(ours["x_log_std"] - o[f"{prefix}log_std"])))
        scale = max(np.max(np.abs(o[f"{prefix}W{i}"])) for i in range(3))
        err = float(max(errs))
        return name, dict(max_abs=err, max_rel=err / scale, tol=tol, ok=bool(err <= tol or err / scale <= tol))

    rows = dict([
        cmp("gae_advantages", b.adv, o["adv"], 1e-5),
        cmp("discounted_returns", b.ret, o["ret"], 1e-5),
        cmp("gae_mre_code_noclip", vals["gae_mre_code"], o["gae_mre_code_clip0"], 1e-5),
        cmp("gae_mre_code_clip", vals_c["gae_mre_code"], o["gae_mre_code_clip1"], 1e-5),
        cmp("gae_mse_noclip", vals["gae_mse"], o["gae_mse_clip0"], 1e-5),
        cmp("returns_mre", vals["returns_mre"], o["returns_mre"], 1e-5),
        cmp("returns_mse", vals["returns_mse"], o["returns_mse"], 1e-5),
        cmp("gauss_logp", nets.logp(pi, b.obs, b.act), o["logp"], 1e-5),
        cmp("gauss_kl", gauss_kl(m1, l1, m2, l2), o["kl"], 1e-5),
        cmp("adv_normalize", A.adv_normalize(b.adv), o["adv_normalized"], 1e-5),
        cmp("surrogate_value", s_val, o["surrogate"], 1e-5),
        cmp_layers("surrogate_gradient", g_surr, "gpi_", 1e-4, grads=True),
        cmp("cg_solution", A.cg_solve(lambda x: H @ x, cg_b, 10), o["cg_x"], 1e-4),
        cmp_layers("ppo_step_params", pi_ppo, "ppo_pi_", 1e-4),
        cmp_layers("value_step_params", v_step, "vstep_v_", 1e-4, policy=False),
        cmp_layers("trpo_step_params", pi_trpo, "trpo_pi_", 1e-4),
        cmp("zfilter_stream", z_obs, o["stream_obs_norm"], 1e-4),
        cmp("rewardfilter_stream", z_rew, o["stream_rew_norm"], 1e-4),
    ])
    # sanity: the steps actually moved the parameters
    rows["_ppo_step_norm"] = float(jnp.linalg.norm(pi_ppo - pi))
    rows["_trpo_step_norm"] = float(jnp.linalg.norm(pi_trpo - pi))
    rows["_trpo_ls_scale"] = float(trpo_st["ls_scale"])
    rows["_grad_at_ratio1_vs_surrogate_grad_cos"] = float(A.cosine(g_ours, g_surr))
    ok = all(v["ok"] for k, v in rows.items() if isinstance(v, dict))
    for k, v in rows.items():
        print(f"{k:32s} {v}")
    oracle_commit = subprocess.run(["git", "-C", os.path.join(ORACLE, "implementation-matters"), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip()
    write_json(os.path.join(OUT, "oracle_diff.json"), {"pass": ok, "oracle_commit": oracle_commit, "rows": rows,
                                                       **common.git_info()})
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

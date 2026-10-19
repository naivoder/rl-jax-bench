"""Paper-budget learner run (approved 2026-10-19; declared in README).

One encoder family per process (5 lanes vmapped), the paper's protocol (§3): train with per-lane best-model selection
and early stop at no improvement in 2.5e6 env steps, cap 1e7 (SB3 EvalCallback semantics: the best model is chosen
on the DETERMINISTIC validation return, recheck). The validation curves go to a log nobody reads before the
final evaluation (review). The common step cap below 1e7 is a timing-only budget decision passed through
<out>/train_<family>/n_it_override_<enc> (README declaration). Each lane's selected best model is then evaluated
ONCE on the 100 full-scale evaluation scenarios fold_in(root, 0xE7A1), stochastic and deterministic (co-primary).

CNN liveness  is logged, never acted on: at the PPO start (pre-training metric, teacher data) and, on
512 random-policy observations from the 16 validation scenarios, for the PPO-start, final and selected parameters.

    python scripts/paper_budget.py --family mhsa --steps 10000000 --select-best-patience 2500000 --out runs/paper_budget
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common as C  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src.configs.config import Config, K_EVAL_SCEN, K_VALID  # noqa: E402
from src import train as TR, eval as E, data  # noqa: E402
from src.model import NatureCNN, raster_of  # noqa: E402
from scripts.full_learners import eval_scen_sha, eval_lanes  # noqa: E402

FAMILIES = {"mhsa": ["mhsa"], "flat": ["flat"], "bigru": ["bigru"],
            "mhsa_ablations": ["mhsa_nobc", "mhsa_noae", "mhsa_lit"],
            # planned-check verification block (README), common random numbers
            "v_mhsa": ["mhsa_lr1e-4", "mhsa", "mhsa_lr1e-3", "mhsa_lit_crn"],
            "v_flat": ["flat_lr1e-4", "flat", "flat_lr1e-3"],
            "v_unmasked": ["mhsa_unmasked"],
            "v_single": ["flat_single"],
            # lr-1e-3 lever at the paper-budget protocol (README)
            "v_mhsa_lr1e-3": ["mhsa_lr1e-3"]}
FAMILY_CFG = {"v_unmasked": dict(net=dict(masked=False)),          # descriptive reading: unmasked zero padding
              "v_single": dict(env=dict(ssv_norm="single"))}       #: T = 0 scan-value normaliser (r_SV x 43.8)


def liveness_obs(cfg, n_obs=512):
    """Random-policy observations on the 16 validation scenarios (never the evaluation set)."""
    ecfg = replace(cfg.env, gospa_metrics=False)
    sk = E.scenario_keys(jax.random.PRNGKey(cfg.seed_root), K_VALID, cfg.ppo.curve_scenarios)

    def episode(k):
        s = data.reset(k, ecfg)
        o = data.observe(s, ecfg)

        def body(c, t):
            s, o = c
            a = jax.random.randint(jax.random.fold_in(jax.random.fold_in(k, 0x11FE), t), (2,), 0, ecfg.n_a)
            s2, o2, _ = data.step(s, a.astype(jnp.int32), ecfg)
            return (s2, o2), o

        _, obs = jax.lax.scan(body, (s, o), jnp.arange(ecfg.T))
        return obs

    obs = jax.jit(jax.vmap(episode))(sk)
    flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), obs)
    n = flat.sv_az.shape[0]
    return jax.tree.map(lambda x: x[:: max(1, n // n_obs)][:n_obs], flat)


def cnn_alive(cfg, params, obs):
    """Per lane: fraction of the 128 z_SH units > 0 on at least one observation, and the median ||z_SH||."""
    cnn = NatureCNN(cfg.net.cnn_out)
    img = raster_of(obs, replace(cfg.env, gospa_metrics=False)).astype(jnp.float32)

    def one(p):
        z = cnn.apply({"params": p["params"]["cnn"]}, img)
        return jnp.mean(jnp.any(z > 0, axis=0).astype(jnp.float32)), jnp.median(jnp.linalg.norm(z, axis=-1))
    a, nz = jax.jit(jax.vmap(one))(params)
    return np.asarray(a).tolist(), np.asarray(nz).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--select-best-patience", type=int, default=2_500_000)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--curve-every", type=int, default=98_304)     # 6 PPO iterations of 16,384 steps
    ap.add_argument("--ckpt-every-it", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(C.REPO, "runs", "paper_budget"))
    ap.add_argument("--eval-final", action="store_true",
                    help="also evaluate each lane's FINAL parameters (stochastic), co-reported (verification block)")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU plumbing check (T=40, 4 eval scenarios); not a result")
    args = ap.parse_args()
    t_start = time.time()
    plat = C.check_device()
    cfg = Config()
    if args.smoke:
        cfg = replace(cfg, env=replace(cfg.env, T=40), n_eval_scenarios=4,
                      ppo=replace(cfg.ppo, n_steps=32, n_envs=2, n_epochs=1, minibatch=32, curve_scenarios=2),
                      pre=replace(cfg.pre, ae_steps=20, bc_episodes=1, bc_epochs=1))
    arms = FAMILIES[args.family]
    fc = FAMILY_CFG.get(args.family, {})
    if fc.get("net"):
        cfg = replace(cfg, net=replace(cfg.net, **fc["net"]))
    if fc.get("env"):
        cfg = replace(cfg, env=replace(cfg.env, **fc["env"]))
    seeds = [int(s) for s in args.seeds.split(",")]
    lm = [(a, s) for a in arms for s in seeds]
    tr_dir = os.path.join(args.out, "train_" + args.family)
    os.makedirs(tr_dir, exist_ok=True)
    net, lanes, meta, summary = TR.run_family(cfg, arms, seeds, tr_dir, args.steps, args.curve_every,
                                              ckpt_every_it=args.ckpt_every_it, lanes_meta=lm,
                                              select_best_patience=args.select_best_patience,
                                              curve_log=os.path.join(tr_dir, "curves_unread.jsonl"),
                                              select_deterministic=True)
    t_train = time.time() - t_start
    enc = TR.ARMS[arms[0]]["encoder"]
    cfgf = replace(cfg, net=replace(cfg.net, encoder=enc))
    best = pickle.load(open(os.path.join(tr_dir, f"params_best_{enc}.pkl"), "rb"))
    final = pickle.load(open(os.path.join(tr_dir, f"params_{enc}.pkl"), "rb"))
    start = pickle.load(open(os.path.join(tr_dir, f"pretrain_{enc}.pkl"), "rb"))
    # ---- CNN liveness (logged only)
    lo = liveness_obs(cfgf)
    live = {k: cnn_alive(cfgf, jax.tree.map(jnp.asarray, p), lo)
            for k, p in (("ppo_start", start["params"]), ("final", final["params"]), ("best", best["params"]))}
    C.metric("cnn liveness on validation random-policy obs", **{k: ";".join(f"{x:.3f}" for x in v[0])
                                                              for k, v in live.items()})
    # ---- final evaluation of each lane's SELECTED model (the only use of the 0xE7A1 scenarios)
    root = jax.random.PRNGKey(cfg.seed_root)
    sk = E.scenario_keys(root, K_EVAL_SCEN, cfg.n_eval_scenarios)
    sha = eval_scen_sha(cfg, sk)
    C.info("eval scenarios", **sha)
    keys = jnp.stack([TR.lane_key(cfg, a, s) for a, s in meta])
    tm = C.Timer()
    ev_s = eval_lanes(cfgf, net, lanes.params, keys, sk, False, args.chunk)
    ev_d = eval_lanes(cfgf, net, lanes.params, keys, sk, True, args.chunk)
    ev_f = None
    if args.eval_final:
        ev_f = eval_lanes(cfgf, net, jax.tree.map(jnp.asarray, final["params"]), keys, sk, False, args.chunk)
    t_eval = tm.lap()
    n_it = len(summary[0]["train_hist"].get("rew", []))
    per_it = cfg.ppo.n_steps * cfg.ppo.n_envs
    steps_done = n_it * per_it
    ovr = os.path.join(tr_dir, f"n_it_override_{enc}")
    n_ovr = int(open(ovr).read().strip()) if os.path.exists(ovr) else None
    for i, (a, s) in enumerate(meta):
        red_s = E.reduce_episodes({k: v[i] for k, v in ev_s.items()})
        red_d = E.reduce_episodes({k: v[i] for k, v in ev_d.items()})
        d = summary[i]
        bstep = int(best["best_step"][i])
        m = dict(arm=a, seed=s, stage="paper_budget", env_steps=steps_done, iterations=n_it, n_it_override=n_ovr,
                 best_step=bstep, best_ret_det_valid=float(best["best_ret"][i]),
                 stalled_at_end=bool(steps_done - bstep >= args.select_best_patience),
                 patience=args.select_best_patience, step_cap=args.steps,
                 cnn_alive={k: dict(alive=v[0][i], median_norm=v[1][i]) for k, v in live.items()},
                 eval_stochastic=red_s, eval_deterministic=red_d,
                 per_episode={k: v[i].tolist() for k, v in ev_s.items() if k != "life_hist"},
                 per_episode_det={k: v[i].tolist() for k, v in ev_d.items() if k != "life_hist"},
                 curve=d["curve"], pretrain=d["pretrain"], train_hist=d["train_hist"], timing=d["timing"],
                 env_version=d["env_version"], eval_scenarios=sha, n_eval=int(sk.shape[0]), device=plat,
                 init_style=cfg.net.init_style, selection="deterministic validation return",
                 masked=cfg.net.masked, ssv_norm=cfg.env.ssv_norm, lr_scale=TR.ARMS[a].get("lr_scale", 1.0),
                 rtl_sign=TR.ARMS[a]["rtl_sign"])
        if ev_f is not None:
            m["eval_final_stochastic"] = E.reduce_episodes({k: v[i] for k, v in ev_f.items()})
            m["per_episode_final"] = {k: v[i].tolist() for k, v in ev_f.items() if k != "life_hist"}
        C.write_json(os.path.join(args.out, a, f"seed_{s}", "metrics.json"), m)
        C.metric("paper-budget learner", arm=a, seed=s, best_step=bstep, m_cov=red_s["m_cov"],
                 gospa_d=red_s["gospa_d"], gospa_miss=red_s["gospa_miss"], det_m_cov=red_d["m_cov"])
    wall = time.time() - t_start
    C.write_json(os.path.join(args.out, f"timing_{args.family}.json"),
                 dict(family=args.family, lanes=len(meta), steps_per_lane=steps_done, wall_s=wall, train_s=t_train,
                      eval_s=t_eval, gpu_h_process=wall / 3600, t_start=t_start, t_end=time.time(), device=plat))
    C.info("paper-budget family done", family=args.family, wall_s=round(wall, 1), gpu_h=round(wall / 3600, 4))


if __name__ == "__main__":
    main()

"""Train a grid of runs (arm x seed x lr [x alpha] x env-parameter cell) in ONE vmapped program and write
runs/<tag>/<cell>/<arm>/lr<lr>/seed_<n>/metrics.json plus runs/<tag>/summary.json.

    python scripts/run_grid.py --env pd --tag prelim_pd --arms ppo bc dagger bc_tf1 bc_ppo bc_then_ppo advisor \
        --seeds 0 1 2 3 4 100 101 --lrs 3e-4 1e-3 3e-3
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import bootstrap, check_device, git_provenance, info, warn, write_json  # noqa: E402

bootstrap()
import jax  # noqa: E402
import numpy as np  # noqa: E402

from src import model as M  # noqa: E402
from src import train as TR  # noqa: E402
from src.data import get_spec  # noqa: E402
from src.envs import lighthouse as LH  # noqa: E402


def official_draws(task):
    """draw index -> (lr, alpha, stage-split) from the authors' random_hp_search_runs_<task>.tsv. The same 50 draws are
    used for every method (official random_hp_search.py, numpy seeds 1/2/3); asserted here across all methods."""
    from src.eval import official_tsv
    out = {}
    for r in official_tsv(task):
        c = r["config_kwargs_str"]
        d = int(r["seed"])
        key = (float(c["LR"]), float(c.get("FIXED_ALPHA", np.nan)), float(c.get("TF_RATIO", np.nan)))
        if d in out:
            o = out[d]
            assert abs(o[0] - key[0]) < 1e-15, (d, o, key)
            out[d] = tuple(o[i] if not np.isnan(o[i]) else key[i] for i in range(3))
        else:
            out[d] = key
    # methods without alpha / split leave those NaN in their row; every draw has them from some method
    return {d: (v[0], 8.0 if np.isnan(v[1]) else v[1], 0.5 if np.isnan(v[2]) else v[2]) for d, v in out.items()}


def lh_cells(cells_arg):
    out = []
    for c in cells_arg:
        i, j = (int(x) for x in c.split(","))
        out.append((f"i{i}_j{j}", {"i": i, "j": j}, LH.optimal_len(i)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["pd", "lh", "mg"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--lrs", nargs="+", type=float, default=None)
    ap.add_argument("--alphas", nargs="+", type=float, default=[8.0])
    ap.add_argument("--split", type=float, default=0.5)            # ASSUMED (centre of U[0.1, 0.9), App. A.7.1)
    ap.add_argument("--total", type=int, default=None)
    ap.add_argument("--teacher", default="privileged", choices=["privileged", "public"])
    ap.add_argument("--n_wrong", type=int, default=1, help="public teacher: number of wrong final code digits (extension)")
    ap.add_argument("--cells", nargs="+", default=None, help="2D-LH cells i,j (e.g. 1,15 1,1)")
    ap.add_argument("--n_eval", type=int, default=None)
    ap.add_argument("--n_ckpt", type=int, default=4)
    ap.add_argument("--n_eval_mid", type=int, default=None, help="episodes at non-final checkpoints (curves only)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--draws", nargs="+", type=int, default=None,
                    help="official hp-draw indices: lr, alpha, stage-split and seed taken from the authors' TSV "
                         "(App. A.7.1 random search; identical draws across methods, as in the official code)")
    ap.add_argument("--plan", default=None, help="JSON list of run groups {arms, seeds, lrs, alphas, draws, cells}; "
                    "each group is a cross product, all groups train in ONE vmapped program")
    ap.add_argument("--tsv_task", default=None, help="official TSV task name for --draws (default per env)")
    args = ap.parse_args()

    plat = check_device()
    spec = get_spec(args.env)
    if args.env == "pd":
        total = args.total or 300000                    # Table A.3
        net = M.PDNet()
        cfg = TR.TrainCfg(env="pd", total_steps=total)
        cname = "pd_" + args.teacher + (f"_w{args.n_wrong}" if args.teacher == "public" and args.n_wrong != 1 else "")
        cells = [(cname, {"public": args.teacher == "public", "n_wrong": args.n_wrong}, 0.0)]
        n_eval = args.n_eval or 1000                    # official NUM_TEST_TASKS
    elif args.env == "lh":
        total = args.total or 300000                    # Table A.3
        net = M.LHNet()
        cfg = TR.TrainCfg(env="lh", total_steps=total, early_stop=True)
        cells = lh_cells(args.cells or [])
        n_eval = args.n_eval or 200                     # paper Sec. 4.4 / official
    else:
        from src.envs import crossing as MG
        total = args.total or 1000000                   # Table A.3
        net = M.MGNet(n_actions=MG.N_ACTIONS)
        cfg = TR.TrainCfg(env="mg", total_steps=total)
        cells = [("lc_once", {}, 0.0)]
        n_eval = args.n_eval or 1000

    groups = [dict(arms=args.arms, seeds=args.seeds, lrs=args.lrs, alphas=args.alphas, draws=args.draws,
                   cells=None)]
    if args.plan:
        groups = []
        for g in json.load(open(args.plan)):
            groups.append(dict(arms=g["arms"], seeds=g.get("seeds", args.seeds), lrs=g.get("lrs", args.lrs),
                               alphas=g.get("alphas", args.alphas), draws=g.get("draws"), cells=g.get("cells")))
    specs = []
    for g in groups:
        gcells = lh_cells(g["cells"]) if (args.env == "lh" and g["cells"]) else cells
        if g["draws"] is not None:
            task = args.tsv_task or {"pd": "PoisonedDoors", "mg": "AskForHelpLavaCrossingOnce"}[args.env]
            D = official_draws(task)
            for cname, ep, opt in gcells:
                for arm in g["arms"]:
                    for d in g["draws"]:
                        lr, alpha, split = D[d]
                        specs.append(dict(arm=arm, seed=d, lr=lr, alpha=alpha, beta=0.0, split=split, ep=ep,
                                          opt_len=opt, cell=cname, draw=d))
            continue
        for cname, ep, opt in gcells:
            for arm in g["arms"]:
                for lr in g["lrs"]:
                    alphas = g["alphas"] if "advisor" in arm else [g["alphas"][0]]
                    for alpha in alphas:
                        for sd in g["seeds"]:
                            specs.append(dict(arm=arm, seed=sd, lr=lr, alpha=alpha, beta=0.0, split=args.split,
                                              ep=ep, opt_len=opt, cell=cname))
    keys = [(r["cell"], r["arm"], r["lr"], r["alpha"], r["seed"], r.get("draw")) for r in specs]
    assert len(keys) == len(set(keys)), "duplicate run specs in the plan"
    U = cfg.n_updates
    eval_at = [0] + [int(np.ceil(U * (k + 1) / args.n_ckpt)) for k in range(args.n_ckpt)]
    out_dir = os.path.join("runs", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    info("grid", env=args.env, runs=len(specs), updates=U, eval_at=str(eval_at), platform=plat, n_eval=n_eval,
         n_eval_mid=args.n_eval_mid or n_eval)
    t0 = time.perf_counter()
    met, evals, st = TR.run_experiment(cfg, spec, net, specs, eval_at=eval_at, n_eval=n_eval, n_eval_mid=args.n_eval_mid,
                                       ckpt_dir=os.path.join(out_dir, "ckpt") if args.resume else None)
    wall = time.perf_counter() - t0
    finish = np.asarray(st["finish_steps"])
    prov = git_provenance()
    summary = []
    for k, r in enumerate(specs):
        ev = {}
        for u, e in evals:
            ev[str(u * cfg.steps_per_update)] = {m: float(np.mean(np.asarray(v)[k])) for m, v in e.items()}
        final = evals[-1][1]
        rec = dict(arm=r["arm"], seed=r["seed"], draw=r.get("draw"), lr=r["lr"], alpha=r["alpha"], cell=r["cell"], ep=r["ep"],
                   split=r["split"], total_steps=total, train_steps=int(finish[k]) if finish[k] >= 0 else int(np.asarray(st["total"])[k]),
                   final={m: float(np.mean(np.asarray(v)[k])) for m, v in final.items()},
                   final_ret_per_episode_sha=None, evals=ev,
                   train_curve={m: np.asarray(met[m][k]).astype(float).round(5).tolist()
                                for m in ("ep_ret", "ep_len", "w_mean", "entropy", "loss", "mode")},
                   **prov, platform=plat, wall_s_all_runs=wall)
        if args.env == "lh":
            rec["final"]["near_opt_1.1"] = float(rec["final"]["len"] < 1.1 * r["opt_len"])
            rec["final"]["near_opt_1.05"] = float(rec["final"]["len"] < 1.05 * r["opt_len"])
            rec["opt_len"] = r["opt_len"]
        sub = (f"draw{r['draw']}" if r.get("draw") is not None else
               f"lr{r['lr']:g}" + (f"_a{r['alpha']:g}" if "advisor" in r["arm"] else ""))
        d = os.path.join(out_dir, r["cell"], r["arm"], sub, f"seed_{r['seed']}")
        write_json(os.path.join(d, "metrics.json"), rec)
        summary.append({k2: v for k2, v in rec.items() if k2 not in ("evals", "train_curve")})
    write_json(os.path.join(out_dir, "summary.json"), {"args": vars(args), "wall_s": wall, "runs": summary})
    info("wrote", out=out_dir, wall_s=round(wall, 1))


if __name__ == "__main__":
    main()

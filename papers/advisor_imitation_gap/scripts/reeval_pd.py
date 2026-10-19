"""Re-evaluate a finished PD grid from its orbax checkpoints with the fixed evaluation-task table (an earlier version's traced int32 good-door expression overflowed). Training is NOT re-run: the restored
parameters at every checkpoint are evaluated with the same eval keys (fold_in(fold_in(key, 4), ckpt_idx)) and the
same episode counts; only the good door of each evaluation task changes. --old_doors re-creates the pre-fix doors
(used to check that the pipeline reproduces the stored numbers bit for bit).

    python scripts/reeval_pd.py --src full_pd --dst full_pd_evfix [--old_doors]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import bootstrap, check_device, git_provenance, info, write_json  # noqa: E402

bootstrap()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import model as M  # noqa: E402
from src import train as TR  # noqa: E402
from src.data import PD_SPEC  # noqa: E402
from src.envs import pd as PD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--old_doors", action="store_true")
    args = ap.parse_args()
    plat = check_device()
    S = json.load(open(os.path.join("runs", args.src, "summary.json")))
    a = S["args"]
    cfg = TR.TrainCfg(env="pd", total_steps=a["total"] or 300000)
    net = M.PDNet()
    spec = PD_SPEC
    if args.old_doors:
        spec = PD_SPEC._replace(eval_reset=lambda k, key, ep: PD.reset(key, good=1 + ((2 ** 31 - 1 + k) % 3)))
    specs = [dict(arm=r["arm"], seed=r["seed"], lr=r["lr"], alpha=r["alpha"], beta=0.0, split=r["split"],
                  ep=r["ep"], opt_len=0.0) for r in S["runs"]]
    runs = TR.build_runs(cfg, specs)
    st0 = jax.jit(jax.vmap(lambda r: TR.init_run(cfg, spec, net, r)))(runs)
    U = cfg.n_updates
    eval_at = [0] + [int(np.ceil(U * (k + 1) / a["n_ckpt"])) for k in range(a["n_ckpt"])]
    n_eval, n_mid = a["n_eval"] or 1000, a["n_eval_mid"] or a["n_eval"] or 1000
    ev_final = jax.jit(jax.vmap(TR.make_eval(cfg, spec, net, n_eval), in_axes=(0, 0, None)), static_argnums=2)
    ev_mid = jax.jit(jax.vmap(TR.make_eval(cfg, spec, net, n_mid), in_axes=(0, 0, None)), static_argnums=2)
    ckdir = os.path.join(os.path.abspath("runs"), args.src, "ckpt")
    evals = []
    for ci, u in enumerate(eval_at):
        if u == 0:
            params = st0["params"]
        else:
            st, got = TR._maybe_restore_at(ckdir, st0, u)
            assert got == u, (got, u)
            params = st["params"]
        f = ev_final if u == eval_at[-1] else ev_mid
        evals.append((u, jax.device_get(f(params, runs, ci))))
        info("re-evaluated", update=u, ret_mean=float(np.mean(evals[-1][1]["ret"])))
    prov = git_provenance()
    out = []
    for k, r in enumerate(S["runs"]):
        sub = (f"draw{r['draw']}" if r.get("draw") is not None else
               f"lr{r['lr']:g}" + (f"_a{r['alpha']:g}" if "advisor" in r["arm"] else ""))
        old_p = os.path.join("runs", args.src, r["cell"], r["arm"], sub, f"seed_{r['seed']}", "metrics.json")
        old = json.load(open(old_p))
        ev = {str(u * cfg.steps_per_update): {m: float(np.mean(np.asarray(v)[k])) for m, v in e.items()}
              for u, e in evals}
        rec = dict(old)
        rec.update(final={m: float(np.mean(np.asarray(v)[k])) for m, v in evals[-1][1].items()}, evals=ev,
                   reeval_of=args.src, reeval_doors="pre-fix (overflowed)" if args.old_doors else "official table",
                   reeval_train_git=old.get("git"), reeval_platform=plat, final_before_reeval=old["final"])
        rec.update({("reeval_" + k2): v for k2, v in prov.items()})
        write_json(os.path.join("runs", args.dst, r["cell"], r["arm"], sub, f"seed_{r['seed']}", "metrics.json"), rec)
        out.append({k2: v for k2, v in rec.items() if k2 not in ("evals", "train_curve", "final_before_reeval")})
    write_json(os.path.join("runs", args.dst, "summary.json"), {"args": a, "reeval_args": vars(args), "runs": out})
    d = [abs(o["final"]["ret"] - r["final"]["ret"]) for o, r in zip(out, S["runs"])]
    info("done", max_abs_change_final_ret=float(max(d)), runs=len(out))


if __name__ == "__main__":
    main()

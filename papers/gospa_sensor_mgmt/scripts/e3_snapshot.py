"""E3 (Figure 4 analogue): frame-152 snapshot of Myopic-KLD and MCTS3-GD, run m = 0, lambda_C = 1, CFG-O (CRN: identical measurement stream for both).
Quantitative content for C5: sensor positions at the frame, distance to the origin, estimates, GOSPA of the frame.
Usage: python scripts/e3_snapshot.py --png <path> --json <path>"""
import argparse, json, os, sys
from rl_jax_bench import gpu_bootstrap
gpu_bootstrap.setup()
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from src import model as M
from src import algorithm as A
from src import data as D


def run(cfg, name, frame, lam=1.0, m=0):
    alg = M.ALGS[name]
    runner = A.make_chunk_runner(cfg, alg, 0, "off", emit_est=True)
    carry = A.init_carry(cfg, 1)
    carry, o = runner(carry, jnp.arange(0, frame + 1), jnp.asarray([m]), jnp.asarray([lam]))
    return {k: np.asarray(v)[:, 0] for k, v in o.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--frame", type=int, default=152)
    ap.add_argument("--cfg", default="CFG-O")
    a = ap.parse_args()
    cfg = M.CONFIGS[a.cfg]
    tr, al, comp, cm = D.truth_arrays()          # tr [200, 9, 4], al [200, 9]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    res = {}
    for ax, name in zip(axs, ("Myopic-KLD", "MCTS3-GD")):
        o = run(cfg, name, a.frame)
        k = a.frame
        sp = o["spos"]                               # [K, 2, 2]
        for s, col in enumerate(("tab:blue", "tab:orange")):
            ax.plot(sp[:, s, 0], sp[:, s, 1], "-", color=col, lw=1, alpha=0.7)
            ax.plot(*sp[k, s], "s", color=col, ms=7)
            ax.add_patch(Circle(sp[k, s], M.FOV, fill=False, ec=col, ls="--", lw=0.8))
        ax.add_patch(Rectangle((M.OBSTACLE[0], M.OBSTACLE[1]), M.OBSTACLE[2] - M.OBSTACLE[0], M.OBSTACLE[3] - M.OBSTACLE[1], fc="0.6", ec="k"))
        alive = al[k].astype(bool)
        ax.plot(tr[k][alive, 0], tr[k][alive, 2], "g^", ms=8, label="truth")
        est = o["est"][k][o["est_m"][k].astype(bool)]
        ax.plot(est[:, 0], est[:, 2], "rx", ms=9, mew=2, label="estimate")
        ax.plot(0, 0, "k+", ms=10)
        ax.set_xlim(-250, 250); ax.set_ylim(-250, 250); ax.set_aspect("equal")
        ax.set_title("%s, frame %d (run 0, lambda_C = 1)\nGOSPA = %.1f" % (name, k, np.sqrt(o["gospa_sq"][k])))
        ax.legend(loc="lower left")
        res[name] = dict(frame=k, sensors=sp[k].tolist(), dist_origin=np.sqrt((sp[k] ** 2).sum(-1)).tolist(), n_truth=int(alive.sum()),
                         n_est=int(o["est_m"][k].sum()), est=est[:, [0, 2]].tolist(), truth=tr[k][alive][:, [0, 2]].tolist(), gospa=float(np.sqrt(o["gospa_sq"][k])),
                         sensors_t50=sp[50].tolist(), dist_origin_t50=np.sqrt((sp[50] ** 2).sum(-1)).tolist())
        print("[METRIC] E3", name, json.dumps({kk: vv for kk, vv in res[name].items() if kk in ("sensors", "dist_origin", "n_truth", "n_est", "gospa")}), flush=True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.png), exist_ok=True)
    fig.savefig(a.png, dpi=110)
    json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

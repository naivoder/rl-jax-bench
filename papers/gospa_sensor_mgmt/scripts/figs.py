"""Figures from results/e1_report.json: Table III ours vs paper (CFG-O), and the E2 per-step decomposition at lambda_C = 1.
Palette: three CVD-safe hues in fixed order (blue, orange, grey); series are also distinguished by marker / line style."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREY = "#2a6fb0", "#d9731a", "#6b6b6b"
algs = ["Myopic-GD", "MCTS1-GD", "MCTS2-GD", "MCTS3-GD", "MCTS4-GD", "MCTS1-KLD", "MCTS2-KLD", "MCTS3-KLD", "MCTS4-KLD"]
fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, lam in zip(axs, (0.1, 1.0, 2.0)):
    for i, cfg, col, mk in ((0, "CFG-O", BLUE, "o"), (1, "CFG-P", GREY, "s")):
        rows = {(r["alg"], r["clutter"]): r for r in R[cfg]["table3"]}
        y = [rows[(a, lam)]["G"] for a in algs]
        lo = [rows[(a, lam)]["G"] - rows[(a, lam)]["ci"][0] for a in algs]
        hi = [rows[(a, lam)]["ci"][1] - rows[(a, lam)]["G"] for a in algs]
        ax.errorbar(np.arange(len(algs)) + 0.12 * (i - 0.5), y, yerr=[lo, hi], fmt=mk, color=col, ms=5, lw=1, capsize=2, label=f"ours {cfg} (50 runs, 95% CI)")
    rows = {(r["alg"], r["clutter"]): r for r in R["CFG-O"]["table3"]}
    ax.plot(np.arange(len(algs)), [rows[(a, lam)]["paper"] for a in algs], "D", color=ORANGE, ms=6, mfc="none", mew=1.5, label="paper Table III")
    ax.set_xticks(range(len(algs))); ax.set_xticklabels([a.replace("-GD", "\nGD").replace("-KLD", "\nKLD") for a in algs], fontsize=7)
    ax.set_title(f"clutter {lam}"); ax.grid(alpha=0.25)
axs[0].set_ylabel("time-averaged RMS-GOSPA"); axs[0].legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "table3.png"), dpi=120); plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
cur = R["CFG-O"]["E2_curves_lam1"]
for j, a in enumerate(("Myopic-GD", "MCTS3-GD")):
    for i, (k, lab, col, ls) in enumerate((("gospa_sq", "total", BLUE, "-"), ("loc", "localization", ORANGE, "--"), ("missed", "missed", GREY, "-."), ("false", "false", "k", ":"))):
        axs[0, j].plot(cur[a][k], color=col, ls=ls, lw=1.4, label=lab)
    axs[0, j].set_title(f"{a}, lambda_C = 1 (CFG-O, 50 runs)"); axs[0, j].grid(alpha=0.25)
    axs[0, j].set_ylabel("per-step RMS-GOSPA")
axs[0, 0].legend(fontsize=8)
for a, col in (("Myopic-GD", GREY), ("MCTS1-GD", ORANGE), ("MCTS3-GD", BLUE)):
    axs[1, 0].plot(cur[a]["gospa_sq"], color=col, lw=1.3, label=a)
axs[1, 0].legend(fontsize=8); axs[1, 0].set_xlabel("time step"); axs[1, 0].set_ylabel("per-step RMS-GOSPA"); axs[1, 0].grid(alpha=0.25)
for a, col, ls in (("MCTS3-GD", BLUE, "-"), ("MCTS3-KLD", ORANGE, "--")):
    axs[1, 1].plot(cur[a]["gospa_sq"], color=col, ls=ls, lw=1.3, label=a)
axs[1, 1].legend(fontsize=8); axs[1, 1].set_xlabel("time step"); axs[1, 1].grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_decomposition.png"), dpi=120); plt.close(fig)
print("figures written to", OUT)

"""Markdown tables of the full-scale results from runs/full/analysis.json (+ attribution jsons, demonstrator_greedy.json)."""
import json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(REPO, "runs/full/analysis.json")))
dg = json.load(open(os.path.join(REPO, "runs/full/demonstrator_greedy.json"))) if os.path.exists(os.path.join(REPO, "runs/full/demonstrator_greedy.json")) else {}
f = lambda t, nd=2: "-" if t is None else (f"{t[0]:.{nd}f}" if t[1] is None else f"{t[0]:.{nd}f} +- {t[1]:.{nd}f}")
ORDER = ["IMIT", "DQfD", "NoMargin", "NoNstep", "ADET", "PDD", "PDDconst", "RBS", "HER", "DQfD+eps0=0.1+eps1=0.1", "PDD+eps1=0.1", "NoMargin+eps0=0.1+eps1=0.1",
         "DQfD+lam_e=0.1", "DQfD+eps_d=0.1", "DQfD+eps0=0.1+eps1=0.1+lam_e=0.1"]
for g in ["Breakout", "SpaceInvaders", "Asterix", "Seaquest"]:
    b = d[g]; dm = b["demonstrator"]
    gd = dg.get(f"{g}/subopt/eps0.0", {}).get("mean")
    print(f"\n#### {g} / subopt ({b['steps']:.0f} steps; EARLY first {b['early_bins']*b['bin_width']:.0f}, BEST/FINAL {b['window_bins']*b['bin_width']:.0f}); demonstrator eps0.01 mean {dm['true_mean']:.2f} +- {dm['true_se']:.2f}"
          + (f", eps0 {gd:.2f}" if gd is not None else "") + f", best/worst demo episode {dm['best_episode']:.0f}/{dm['worst_episode']:.0f}\n")
    print("| arm | n | EARLY | BEST (biased) | FINAL | greedy eps=0 (end) | demo Q-gap |\n|---|---|---|---|---|---|---|")
    for a in ORDER:
        if a in b["arms"]:
            r = b["arms"][a]
            print(f"| {a} | {r['n']} | {f(r['early'])} | {f(r['best'])} | {f(r['final'])} | {f(r['greedy'])} | {f(r['gap'],3)} |")
    print(f"\ncatch-up (PDD curve above DQfD 3 consecutive bins): {b['catch_up_step_PDD_over_DQfD']}; up-sample ratio: {b['upsample_ratio_DQfD']}")
print("\n#### Primary family (Holm over 8)\n\n| test | n | diff (SE) | wins | raw p | Holm p | CI |\n|---|---|---|---|---|---|---|")
for k, v in d["primary_holm"].items():
    name, g = k.split("[")[0], k.split("[")[1][:-1]
    t = d[g]["tests"][name]
    print(f"| {name} [{g}] | {t['n']} | {t['diff_mean']:.3f} ({t['diff_se']:.3f}) | {t['wins']}/{t['n']} | {v['p_raw']:.2e} | {v['p_holm']:.2e} | [{t['boot95'][0]:.3f}, {t['boot95'][1]:.3f}] |")
for fam, key in (("S1", "secondary_S1"), ("S2", "secondary_S2")):
    print(f"\n#### {fam}\n\n| test | game | n | diff (SE) | wins | raw p | Holm p |\n|---|---|---|---|---|---|---|")
    for k, v in d[f"{fam}_holm"].items():
        name, g = k.split("[")[0], k.split("[")[1][:-1]
        t = d[g][key][name]
        print(f"| {name} | {g} | {t['n']} | {t['diff_mean']:.3f} ({t['diff_se']:.3f}) | {t['wins']}/{t['n']} | {v['p_raw']:.2e} | {v['p_holm']:.2e} |")
print("\n#### lr probe\n"); print(json.dumps(d["lr_probe"], indent=0, default=float)[:3000])

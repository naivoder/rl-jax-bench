"""Aggregate the SMPyBandits reference runs and compare with the paper read-offs and our Full-scale means."""
import json, glob, numpy as np, sys
OURS = {"e1": {"ucb1": (837.2, 35.0, 259.1, 5356), "exp3s": (535.5, 100.3, 305.5, 6225), "ducb": (610.9, 184.0, 279.8, 4460),
               "swucb": (338.7, 68.0, 137.4, 6328), "ducb_padB": (381.4, 111.2, 144.7, 5566)},
        "e2": {"ucb1": (944.3, None, None, 7714), "exp3s": (665.1, None, None, 4883), "ducb": (350.0, None, None, 5707),
               "swucb": (259.8, None, None, 5965), "ducb_padB": (218.2, None, None, 5756)}}
PAPER = {"e1": {"ucb1": (905, 45, 210, 4500), "exp3s": (510, 70, 315, 6800), "ducb": (377, 120, 155, 5650), "swucb": (267, 65, 82, 6400)},
         "e2": {"ucb1": (835, None, None, 8100), "exp3s": (455, None, None, 4700), "ducb": (222, None, None, 5950), "swucb": (238, None, None, 5950)}}
MAP = {"ucb1_paper(xi=1/2)": "ucb1", "ducb_printed2B(alpha=4)": "ducb", "ducb_smpy_default_pad(alpha=1)": "ducb_padB",
       "swucb_paper(alpha=1/2)": "swucb", "exp3s_paper(S=2)": "exp3s"}
out = {}
for env in ("e1", "e2"):
    files = sorted(glob.glob(f"ref/ref_{env}_s*.json"))
    if not files: continue
    ds = [json.load(open(f)) for f in files]
    out[env] = {}
    print(f"\n## {env}: {len(ds)} seeds x {[d['R'] for d in ds]} reps (SMPyBandits 012fc13a, unpatched)")
    print("| policy (SMPy) | ours-equivalent | oracle mean ± SE(seed means) [run SE] | 5-95% | ours Full-scale | paper | oracle/paper | ours−oracle (z) | R3000 | R5000 | arm1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ds[0]["res"]:
        reg = np.concatenate([d["res"][name]["regret"] for d in ds]); sm = np.array([np.mean(d["res"][name]["regret"]) for d in ds])
        r3 = np.concatenate([d["res"][name]["r3000"] for d in ds]); r5 = np.concatenate([d["res"][name]["r5000"] for d in ds])
        a1 = np.concatenate([d["res"][name]["arm1"] for d in ds])
        se_run = reg.std(ddof=1) / np.sqrt(len(reg)); se_seed = sm.std(ddof=1) / np.sqrt(len(sm)) if len(sm) > 1 else float('nan')
        key = MAP.get(name); ours = OURS[env].get(key) if key else None; pap = PAPER[env].get(key) if key else None
        p05, p95 = np.percentile(reg, [5, 95])
        z = (ours[0] - reg.mean()) / se_run if ours else None
        out[env][name] = dict(n=int(len(reg)), mean=float(reg.mean()), se_run=float(se_run), se_seed=float(se_seed), p05=float(p05), p95=float(p95),
                              r3000=float(r3.mean()), r5000=float(r5.mean()), arm1=float(a1.mean()), ours=ours, paper=pap,
                              paper_pct=float((reg < pap[0]).mean() * 100) if pap else None, z_ours_minus_oracle=z)
        print(f"| {name} | {key or '-'} | {reg.mean():.1f} ± {se_seed:.1f} [{se_run:.1f}] | {p05:.0f}–{p95:.0f} | {ours[0] if ours else '-'} | {pap[0] if pap else '-'} | "
              f"{(reg.mean()/pap[0]) if pap else float('nan'):.2f} | {('%+.1f (%.1f)' % (ours[0]-reg.mean(), z)) if ours else '-'} | {r3.mean():.0f} | {r5.mean():.0f} | {a1.mean():.0f} |"
              + (f" paper pct in oracle runs: {out[env][name]['paper_pct']:.1f}" if pap else ""))
json.dump(out, open("reference_summary.json", "w"), indent=1)

"""Summarise the third-party-JSRL reference run (scripts/official/ref_pointmaze.py) -> runs/parity/ref_pointmaze_summary.json and a markdown table on stdout.
Statistics (declared before the arms finished): per seed, EARLY = mean pi_e-alone success over evaluation points at 5k..20k steps, MID = 25k..50k, FINAL = 100-episode
pi_e-alone success at 100k, T80 = first evaluation step with pi_e-alone >= 0.8 (censored at 100k+5k).  Mean +- SE over 3 seeds; paired (same seed = same guide) t on the
arm differences with the scipy t distribution (n = 3: descriptive only, no Holm family, no GO/KILL claim)."""
import glob, json, os, sys
import numpy as np
from scipy import stats

D = sys.argv[1]; OUT = sys.argv[2]
arms = ["scratch", "curr_default", "curr_paper", "random"]
R = {a: {int(os.path.basename(f).split("_s")[-1][:-5]): json.load(open(f)) for f in sorted(glob.glob(f"{D}/{a}_s?.json"))} for a in arms}
guides = {int(os.path.basename(f)[7:-5]): json.load(open(f)) for f in sorted(glob.glob(f"{D}/guide_s?.json"))}

def per_seed(r):
    st = np.array([row["step"] for row in r["curve"]]); al = np.array([row["alone"] for row in r["curve"]]); cb = np.array([row["combined"] for row in r["curve"]])
    hz = np.array([row["horizon"] for row in r["curve"]])
    first = lambda m: int(st[np.argmax(al >= m)]) if (al >= m).any() else 105000
    return dict(early=float(al[(st >= 5000) & (st <= 20000)].mean()), mid=float(al[(st >= 25000) & (st <= 50000)].mean()), final=float(r["final_alone100"]),
                t80=first(0.8), t50=first(0.5), early_comb=float(cb[(st >= 5000) & (st <= 20000)].mean()),
                horizon_at_50k=int(hz[st == 50000][0]) if (st == 50000).any() else None, steps_to_h0=int(st[np.argmax(hz == 0)]) if (hz == 0).any() else None)

S = {a: {s: per_seed(r) for s, r in R[a].items()} for a in arms if R[a]}
out = dict(guide_final100={s: g["final100"] for s, g in guides.items()}, per_seed=S, summary={}, paired={})
for a in S:
    out["summary"][a] = {k: dict(mean=float(np.mean([v[k] for v in S[a].values() if v[k] is not None])), se=float(stats.sem([v[k] for v in S[a].values() if v[k] is not None])) if len(S[a]) > 1 else None, n=len(S[a]))
                         for k in ("early", "mid", "final", "t80", "t50", "early_comb", "steps_to_h0")}
for a in S:
    for b in S:
        if a < b or a == b or "scratch" not in (a, b):
            continue
for a in ("curr_default", "curr_paper", "random"):
    if a in S and "scratch" in S:
        for k in ("early", "mid", "final", "t80"):
            ss = sorted(set(S[a]) & set(S["scratch"]))
            d = np.array([S[a][s][k] - S["scratch"][s][k] for s in ss])
            t = stats.ttest_1samp(d, 0.0) if len(d) > 2 else None
            out["paired"][f"{a}-scratch:{k}"] = dict(diff=d.tolist(), mean=float(d.mean()), se=float(stats.sem(d)) if len(d) > 1 else None, wins=int((d > 0).sum()), n=len(d), p_two_sided=float(t.pvalue) if t is not None else None)
json.dump(out, open(OUT, "w"), indent=1)
print("guide 100-ep success:", out["guide_final100"])
print("| arm | EARLY 5-20k | MID 25-50k | FINAL 100k | T80 (steps) | steps to h=0 |")
print("|---|---|---|---|---|---|")
for a in S:
    m = out["summary"][a]; f = lambda k, p=3: f"{m[k]['mean']:.{p}f} +- {m[k]['se']:.{p}f}" if m[k]["se"] is not None else "-"
    print(f"| {a} | {f('early')} | {f('mid')} | {f('final')} | {f('t80',0)} | {f('steps_to_h0',0) if S[a] and all(v['steps_to_h0'] is not None for v in S[a].values()) else 'n/a'} |")
for k, v in out["paired"].items():
    print(k, "diff", np.round(v["diff"], 3).tolist(), f"mean {v['mean']:.3f} wins {v['wins']}/{v['n']} p={v['p_two_sided']}")

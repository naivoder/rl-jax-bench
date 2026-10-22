"""Markdown tables of the review stage: baselines v2 (mean +- SE over replicates), the E7 provenance regrade, the 10-seed group lr sweep.
    python scripts/review_tables.py"""
import glob
import json
import os
import sys

import numpy as np

from rl_jax_bench.stats import mean_se, paired_compare, t_interval  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fmt(x):
    m, se = mean_se(x)
    return f"{m:.3f} +- {se:.3f}" if se is not None else f"{m:.3f}"


def fmt1(x):
    m, se = mean_se(x)
    return f"{m:.2f} +- {se:.2f}" if se is not None else f"{m:.2f}"


print("### Baselines v2 (runs/baselines_v2/; mean +- SE over action-key replicates on the SAME worlds; plateau slots 60-99)\n")
print("| exp | arm | kw | n worlds | plateau util | plateau rate (Mbps) | collision | idle | efficiency (succ/feasible) | feasible bound (util) | note |\n|---|---|---|---|---|---|---|---|---|---|---|")
order = ["aloha", "tdma_genie", "oca", "oca_sameslot", "dp_lit_tau0.1", "dp_lit_tau0.5", "dp_lit_tau1.0", "dp_tuned", "dpv1_lit_tau0.1", "dpv1_lit_tau0.5", "dpv1_lit_tau1.0", "dpv1_tuned"]
for exp in ("E2", "E2proxy", "E3", "E3proxy", "E1K2"):
    for arm in order:
        p = os.path.join(ROOT, "runs", "baselines_v2", exp, f"{arm}.json")
        if not os.path.exists(p):
            continue
        b = json.load(open(p))
        r = b["per_rep"]
        kw = ",".join(f"{k}={v}" for k, v in b["kw"].items()) if b.get("kw") else ""
        note = {"tdma_genie": "TDMA genie REFERENCE (not a bound)", "oca": "version (prev-slot counts, PU-aware)", "oca_sameslot": "pre-review oracle",
                "dp_tuned": "version, tuned on tuning worlds", "dpv1_tuned": "pre-review oracle, tuned"}.get(arm, "literal, tau ASSUMED" if "lit" in arm else "")
        print(f"| {exp} | {arm} | {kw} | {b['n_exp']} | {fmt([x['plateau_util'] for x in r])} | {fmt1([x['plateau_rate'] for x in r])} | "
              f"{fmt([x['plateau_coll'] for x in r])} | {fmt([x['plateau_idle'] for x in r])} | {fmt([x.get('efficiency_plateau', float('nan')) for x in r])} | "
              f"{fmt([x.get('util_bound_plateau', float('nan')) for x in r])} | {note} |")
print()
print("### DP tuning (tuning worlds offset 100000; best per setting and whether it sits on the grid edge)\n")
for p in sorted(glob.glob(os.path.join(ROOT, "runs", "baselines_v2", "dp_tuning_*.json"))):
    j = json.load(open(p))
    b = j["best"]
    print(f"- {os.path.basename(p)}: best p_update={b['p_update']}, tau={b['tau']} -> {b['plateau_rate']:.2f} Mbps (util {b['plateau_util']:.3f}); on grid edge: {j['best_on_grid_edge']}; "
          f"literal (p_update = 1) best over tau: {max(g['plateau_rate'] for g in j['grid'] if g['p_update'] == 1.0):.2f} Mbps; n_exp {j['n_exp']} x {j['reps']} rep")
print()
print("### E7 lr 3e-4, provenance regrade \n")
print("| seed | original (commit, S) | original plateau util | review rerun (commit, S) | rerun plateau util | HEAD, ONE group of 10 (S=10) | note |\n|---|---|---|---|---|---|---|")
orig, rer, g10 = {}, {}, {}
for f in sorted(glob.glob(os.path.join(ROOT, "runs", "small", "E7_competitive_lr0.0003", "seed_*", "metrics.json"))):
    m = json.load(open(f)); orig[m["seed"]] = m
for f in sorted(glob.glob(os.path.join(ROOT, "runs", "review", "*", "seed_*", "metrics.json"))):
    m = json.load(open(f)); rer.setdefault(m["seed"], []).append(m)
for f in sorted(glob.glob(os.path.join(ROOT, "runs", "small_g10", "E7_competitive_lr0.0003", "seed_*", "metrics.json"))):
    m = json.load(open(f)); g10[m["seed"]] = m
for s in sorted(set(orig) | set(g10)):
    o = orig.get(s)
    rr = rer.get(s, [])
    g = g10.get(s)
    note = []
    if o and g and o["final"]["plateau_util"] == g["final"]["plateau_util"]:
        note.append("original == S=10 exactly")
    if o and g and o["final"]["plateau_util"] != g["final"]["plateau_util"]:
        note.append("original (dev-era code) != S=10")
    print(f"| {s} | " + (f"{o['git_commit']}, S={o['n_vmapped_seeds']} | {o['final']['plateau_util']:.3f}" if o else "- | -") + "| "
          + ("; ".join(f"{r['git_commit']}, S={r['n_vmapped_seeds']}" for r in rr) or "-") + "| " + ("; ".join(f"{r['final']['plateau_util']:.3f}" for r in rr) or "-") + "| "
          + (f"{g['final']['plateau_util']:.3f}" if g else "-") + "| " + "; ".join(note) + "|")
v_o = [orig[s]["final"]["plateau_util"] for s in sorted(orig)]
print(f"\nseeds 100-104: original mixed-provenance mean {fmt(v_o)}", end="")
if g10:
    v5 = [g10[s]["final"]["plateau_util"] for s in sorted(g10) if s <= 104]
    v10 = [g10[s]["final"]["plateau_util"] for s in sorted(g10)]
    print(f"; HEAD (S=10 group) seeds 100-104 {fmt(v5)}; all {len(v10)} seeds {fmt(v10)}", end="")
print("(E7 lr 1e-3, 5 seeds: 0.465 +- 0.058; Aloha 0.444)")
print()
print("### revised: ONE vmapped group of seeds 100-109 per run (runs/small_g10/)\n")
print("| run | git (dirty) | group | n seeds | plateau util | Pareto strict | pattern | per-seed util | paired vs Aloha (t95 of the difference) |\n|---|---|---|---|---|---|---|---|---|")
for d in sorted(glob.glob(os.path.join(ROOT, "runs", "small_g10", "*"))):
    ms = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "seed_*", "metrics.json")))]
    if not ms:
        continue
    fs = [m["final"] for m in ms]
    exp = ms[0]["exp"]
    pat = [k for k in fs[0] if k.startswith("pattern")]
    al = os.path.join(ROOT, "runs", "small_baselines", exp, "aloha.json")
    cmp_ = ""
    if os.path.exists(al):
        a = json.load(open(al))
        av = float(np.mean([x["plateau_util"] for x in a["per_rep"]]))
        d_ = np.array([f["plateau_util"] for f in fs]) - av
        t = t_interval(d_)
        cmp_ = f"diff {d_.mean():+.3f}, t95 [{t[0]:+.3f}, {t[1]:+.3f}], wins {(d_ > 0).sum()}/{len(d_)} (Aloha {av:.3f} constant)"
    print(f"| {os.path.basename(d)} | {ms[0]['git_commit']} ({ms[0]['git_dirty']}) | {ms[0]['vmapped_group_seeds']} | {len(ms)} | {fmt([f['plateau_util'] for f in fs])} | {fmt([f['pareto_strict'] for f in fs])} | "
          + ("; ".join(f"{k}={fmt([f[k] for f in fs])}" for k in pat) or "-") + "| " + ", ".join(f"{f['plateau_util']:.2f}" for f in fs) + f" | {cmp_} |")

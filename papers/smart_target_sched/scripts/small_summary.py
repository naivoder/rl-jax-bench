"""Small-scale summary tables (markdown): comparators vs the index policy, variants (sampled modes, skip-negative, standard F^CT), timing."""
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import eval as E
from rl_jax_bench.stats import paired_compare, mean_se

tag = sys.argv[1] if len(sys.argv) > 1 else "small"
root = os.path.join(HERE, "..", "runs", tag)
cells, meta = E.load(root)
print("### comparators (mean over replications; paired difference vs the index policy: negative = comparator cheaper)\n")
print("| cell | n | index | myopic | passive_cont | passive_cont_h10 | passive_cont_revisit | random | rollout1 | rollout1 - index (paired, t_p) |\n|---|---|---|---|---|---|---|---|---|---|")
for name in E.PAPER:
    for K in (1, 2, 3):
        c = cells.get(f"{name}|K={K}|primary")
        if not c: continue
        J = c["J"]; w = J["whittle"]
        f = lambda p: f"{J[p].mean():.1f}" if p in J else "-"
        r1 = "-"
        if "rollout1" in J:
            pc = paired_compare(J["rollout1"], w); r1 = f"{pc['diff_mean']:+.2f} ({100*pc['rel_improvement']:+.2f} %, p={pc['t_p']:.2g})"
        print(f"| {name} K={K} | {len(w)} | {w.mean():.1f} | {f('myopic')} | {f('passive_cont')} | {f('passive_cont_h10')} | {f('passive_cont_revisit')} | {f('random')} | {f('rollout1')} | {r1} |")
print("\n### passive_cont == myopic decisions? (identical J to 1e-9 in every cell where it holds)")
same = [(cid) for cid, c in cells.items() if cid.endswith("|primary") and "passive_cont" in c["J"] and np.allclose(c["J"]["passive_cont"], c["J"]["myopic"], rtol=1e-9)]
print(len(same), "of", sum(1 for cid, c in cells.items() if cid.endswith("|primary") and "passive_cont" in c["J"]), "cells:", sorted(same))
print("\n### variants vs primary (index policy mean; relative change; ordering index < min(myopic, TEV) holds?)\n")
print("| scenario K | primary | skipneg | sampled (X3) | ft_standard | pcli1_literal | index vs myopic / TEV under sampled modes | skipneg |\n|---|---|---|---|---|---|---|---|")
for name in E.PAPER:
    for K in (1, 2, 3):
        row = []
        base = cells.get(f"{name}|K={K}|primary")
        if not base: continue
        b = base["J"]["whittle"].mean()
        outs = []
        orders = []
        for v in ("skipneg", "sampled", "ft_standard", "pcli1_literal"):
            c = cells.get(f"{name}|K={K}|{v}")
            if c:
                m = c["J"]["whittle"].mean(); outs.append(f"{m:.1f} ({100*(m-b)/b:+.2f} %)")
                if v in ("sampled", "skipneg") and "myopic" in c["J"] and "tev" in c["J"]:
                    pcs = [paired_compare(c["J"]["whittle"], c["J"][o]) for o in ("myopic", "tev")]
                    orders.append(" / ".join(f"{100*p['rel_improvement']:+.2f} % (p={p['t_p']:.2g})" for p in pcs))
                else: orders.append("-")
            else: outs.append("-"); orders.append("-")
        print(f"| {name} K={K} | {b:.1f} | {outs[0]} | {outs[1]} | {outs[2]} | {outs[3]} | {orders[1]} | {orders[0]} |")
print("\n### PCLI1 violations (g <= 1e-12) counted by the closed-loop runs, and max visited state s(P)")
tot = {}
for cid, c in cells.items():
    if "n_pcli1_viol" in c and "whittle" in c["n_pcli1_viol"]:
        tot[cid] = int(c["n_pcli1_viol"]["whittle"].sum())
print("cells with any violation:", {k: v for k, v in tot.items() if v}, "| cells checked:", len(tot))
print("\n### timings (first call = compile + run, steady = second call), seconds")
logs = sorted(glob.glob(os.path.join(root, "run_log_*.json")))
rows = []
for lg in logs:
    for t in json.load(open(lg))["timings"]:
        if t.get("steady_s") is not None: rows.append((t["cell"], t["policy"], t["first_call_s"], t["steady_s"]))
for r in sorted(rows, key=lambda r: -r[3])[:8]:
    print(f"{r[0]} {r[1]}: first {r[2]:.2f}s steady {r[3]:.2f}s (compile ~ {r[2]-r[3]:.2f}s)")

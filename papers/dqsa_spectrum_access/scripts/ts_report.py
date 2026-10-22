"""Markdown tables + the order-invariance check from runs/recheck/rerun/ts_sweep_<exp>.json."""
import json, os, sys, itertools
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "recheck", "rerun") + os.sep
ARMS = ("aloha", "oca", "dp_tuned_at_Ts", "dp_literal_tau0.5", "oca_sameslot")
for exp in sys.argv[1:]:
    p = R + f"ts_sweep_{exp}.json"
    if not os.path.exists(p): print(exp, "missing"); continue
    d = json.load(open(p)); res = d["results"]
    print(f"\n#### {exp} (n_exp {d['n_exp']}, DP tuning worlds {d['n_tune']}); plateau slots 60-99, Mbps\n")
    print("| T_s (ms) | rho | aloha | OCA  | DP tuned AT T_s (p_update, tau) | DP literal tau 0.5 | OCA same-slot (genie ceiling) | strongest legitimate comparator |")
    print("|---|---|---|---|---|---|---|---|")
    for k, v in res.items():
        b = v["dp_best"]
        strong = max(v["oca"]["rate"], v["dp_tuned_at_Ts"]["rate"])
        print(f"| {float(k[3:])*1e3:g} | {v['rho']:.3f} | {v['aloha']['rate']:.1f} | {v['oca']['rate']:.1f} | {v['dp_tuned_at_Ts']['rate']:.1f} ({b['p_update']}, {b['tau']}) | {v['dp_literal_tau0.5']['rate']:.1f} | {v['oca_sameslot']['rate']:.1f} | {strong:.1f} |")
    flips = []
    for k, v in res.items():
        for t in (0, 1):
            ords = {}
            for var in v["aloha"]["lograte_t50_t100"]:
                ords[var] = tuple(sorted(("aloha", "oca", "dp_tuned_at_Ts", "dp_literal_tau0.5"), key=lambda a: -v[a]["lograte_t50_t100"][var][t]))
            if len(set(ords.values())) > 1:
                flips.append((k, [50, 100][t], {o: [x for x in ords if ords[x] == o] for o in set(ords.values())}))
    print(f"\nlog-rate order of the four non-genie baseline arms under the 7 conventions (t = 50 and 100, every T_s): "
          + ("INVARIANT in all cells" if not flips else f"{len(flips)} cell(s) change order: {flips}"))

import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs")
def ld(path):
    out = {}
    for p in glob.glob(os.path.join(path, "seed_*", "metrics.json")):
        d = json.load(open(p)); out[int(d["seed"])] = d
    return out
def fw(rec, tot=100000, frac=0.1):
    ep = np.asarray(rec["episodes"], float); m = ep[:, 0] > (1 - frac) * tot
    return ep[m, 1].mean()
PS = list(range(0, 10000, 1000))
runs = {"oracle(P2.5)": f"{R}/oracle/e4_inverted_pendulum_gymv4", "full_120": f"{R}/full/e4_inverted_pendulum_gymv4",
        "enc": f"{R}/full/e4_ip_gymv4_ablation_encoding", "taunorm": f"{R}/full/e4_ip_gymv4_ablation_taunorm",
        "advbatch": f"{R}/full/e4_ip_gymv4_ablation_advbatch", "annealLR": f"{R}/full/e4_ip_gymv4_ablation_annealLR",
        "ppo1like": f"{R}/full/e4_ip_gymv4_ablation_ppo1like", "smallproxy": f"{R}/small/e4_inverted_pendulum", "fullproxy": f"{R}/full/e4_inverted_pendulum"}
V = {}
for k, p in runs.items():
    if not os.path.isdir(p): print("missing", k); continue
    for arm in sorted(os.listdir(p)):
        if not arm.endswith("g1") and arm not in ("ppo_g1", "ta_ppo_g1"): continue
        d = ld(f"{p}/{arm}")
        if not d: continue
        V[(k, arm)] = {s: fw(r) for s, r in d.items() if s in PS}
        r0 = d[min(d)]
        print(f"{k:13s} {arm:16s} n={len(V[(k,arm)])} lanes={r0['timing']['n_lanes']} git={r0['git'][:7]} cfg={r0['config_hash']} mean={np.mean(list(V[(k,arm)].values())):.1f}")
keys = list(V)
print("\nidentical per-seed vectors (paper seeds) between processes:")
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = V[keys[i]], V[keys[j]]
        ss = sorted(set(a) & set(b))
        if len(ss) >= 5:
            eq = sum(abs(a[s] - b[s]) < 1e-9 for s in ss)
            if eq: print(keys[i], keys[j], f"{eq}/{len(ss)} seeds identical")

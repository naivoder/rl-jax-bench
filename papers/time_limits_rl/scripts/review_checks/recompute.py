"""Independent recomputation of the full-scale headline numbers from raw per-seed metrics.json (review check)."""
import glob, json, os, sys
import numpy as np
from scipy import stats as st
R = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs")
def ld(path):
    out = {}
    for p in glob.glob(os.path.join(path, "seed_*", "metrics.json")):
        d = json.load(open(p)); out[int(d["seed"])] = d
    return out
def fw(rec, tot, frac=0.1):
    ep = np.asarray(rec["episodes"], float)
    m = ep[:, 0] > (1 - frac) * tot
    return ep[m, 1].mean()
def pair(a, b, label):
    sa, sb = set(a), set(b)
    assert sa == sb, (label, sa ^ sb)
    ss = sorted(sa)
    d = np.array([a[s] - b[s] for s in ss])
    n = len(d); se = d.std(ddof=1) / np.sqrt(n) if np.ptp(d) > 0 else 0.0
    if np.ptp(d) == 0:
        p = 2 * 0.5 ** n
    else:
        p = st.ttest_rel([a[s] for s in ss], [b[s] for s in ss]).pvalue
    rng = np.random.default_rng(123)
    bs = d[rng.integers(0, n, (20000, n))].mean(1)
    print(f"{label:45s} n={n} diff={d.mean():+.4f} se={se:.4f} wins={(d>0).sum()}/{n} p={p:.3g} boot=[{np.percentile(bs,2.5):.4g},{np.percentile(bs,97.5):.4g}]")
    return p, d.mean()
fam = {}
# E2 (toys.json)
g = json.load(open(f"{R}/toys/toys.json"))["e2"]
print("toys e2 keys", list(g["standard"].keys())[:10])
seeds = g["standard"].get("seeds") or list(range(10))
for ag in ("standard", "time_aware", "peb"):
    print(ag, "seeds", g[ag].get("seeds"))
mk = lambda ag, k: {s: v for s, v in zip(g[ag].get("seeds", seeds), g[ag][k])}
fam["P1"] = pair(mk("peb", "value_inf"), mk("standard", "value_inf"), "P1 E2 PEB-std inf")
fam["P2"] = pair(mk("time_aware", "value_T3"), mk("standard", "value_T3"), "P2 E2 TA-std T3")
# E2 cross-check from per-seed metrics files
for ag in ("standard", "time_aware", "peb"):
    rr = ld(f"{R}/toys/e2_two_goal/{ag}")
    k = list(rr[min(rr)].keys())
    print(" per-seed file keys", ag, [x for x in k if 'value' in x], len(rr))
# E3 exact value
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_ENABLE_X64"] = "1"
from src import eval as E
e3 = {a: ld(f"{R}/full/e3_queue_of_cars/{a}") for a in ("ta_ppo", "ppo", "ta_remaining", "ta_raw")}
v3 = {a: {s: E.qoc_exact_value(np.asarray(r["per_update"]["p_danger"])[-1][:15], 0.99) for s, r in d.items()} for a, d in e3.items()}
for a in v3: print("E3", a, np.mean(list(v3[a].values())), np.std(list(v3[a].values()), ddof=1)/np.sqrt(10))
fam["P3"] = pair(v3["ta_ppo"], v3["ppo"], "P3 E3 TA-PPO exact")
# E4
for tag, path, seeds in (("A proxy", f"{R}/full/e4_inverted_pendulum", None), ("B port", f"{R}/full/e4_inverted_pendulum_gymv4", list(range(0, 10000, 1000))),
                         ("port fresh", f"{R}/full/e4_inverted_pendulum_gymv4", list(range(10000, 30000, 1000))), ("port all30", f"{R}/full/e4_inverted_pendulum_gymv4", None)):
    arms = {a: ld(f"{path}/{a}") for a in ("ppo_g1", "ta_ppo_g1", "ppo_g099", "ta_ppo_g099")}
    f = {a: {s: fw(r, 100000) for s, r in d.items() if seeds is None or s in seeds} for a, d in arms.items()}
    for a in f: print(tag, a, f"{np.mean(list(f[a].values())):.1f} ± {np.std(list(f[a].values()),ddof=1)/np.sqrt(len(f[a])):.1f} n={len(f[a])}")
    p4 = pair(f["ta_ppo_g1"], f["ppo_g1"], f"P4 {tag}")
    p5 = pair(f["ta_ppo_g099"], f["ppo_g099"], f"P5 {tag}")
    if tag.startswith("A"): famA = (p4, p5)
    if tag.startswith("B"): famB = (p4, p5)
    if tag == "port fresh":
        d = np.array([f["ta_ppo_g1"][s] - f["ppo_g1"][s] for s in sorted(f["ppo_g1"])])
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        print("  fresh one-sided p", st.t.sf(t, len(d) - 1), "wilcoxon", st.wilcoxon(d).pvalue)
# E7
e7 = {}
for a in ("peb_ppo_T300", "ppo_T300", "ppo_T1000"):
    e7[a] = {}
    for blk in ("p1", "p2", "p3"):
        for s, r in ld(f"{R}/full/e7_hopper_{blk}/{a}").items():
            e7[a][s] = float(np.mean(r["evals"][-1]["disc"]))
            last_step = r["evals"][-1]["step"]
    print("E7", a, f"{np.mean(list(e7[a].values())):.2f} ± {np.std(list(e7[a].values()),ddof=1)/np.sqrt(len(e7[a])):.2f} n={len(e7[a])} last step {last_step}")
fam["P6"] = pair(e7["peb_ppo_T300"], e7["ppo_T300"], "P6 E7 PEB300-PPO300")
pair(e7["peb_ppo_T300"], e7["ppo_T1000"], "E7 PEB300-PPO1000")
# E9
for B, P in (("B1000000", "P7"), ("B100", "P8"), ("B40", None), ("B60", None)):
    f = {}
    for a in ("peb", "no_peb"):
        d = ld(f"{R}/full/e9_difficult_grid/{B}/{a}")
        f[a] = {s: np.mean(np.asarray(r["bin_return_mean"])[-4:]) for s, r in d.items()}
        print("E9", B, a, f"{np.mean(list(f[a].values())):.3f} ± {np.std(list(f[a].values()),ddof=1)/np.sqrt(len(f[a])):.3f} n={len(f[a])} ")
    p = pair(f["peb"], f["no_peb"], f"E9 {B}")
    if P: fam[P] = p
def holm(ps):
    ks = sorted(ps, key=lambda k: ps[k]); m = len(ks); out = {}; run = 0
    for i, k in enumerate(ks):
        run = max(run, min(1, (m - i) * ps[k])); out[k] = run
    return out
for lab, (p4, p5) in (("A", famA), ("B", famB)):
    ps = {k: v[0] for k, v in fam.items()}; ps["P4"] = p4[0]; ps["P5"] = p5[0]
    h = holm(ps)
    print("Holm", lab, {k: f"{h[k]:.3g}" for k in sorted(h)})
# E2 pairing alignment: toys.json list order vs per-seed files
for ag in ("standard", "time_aware", "peb"):
    rr = ld(f"{R}/toys/e2_two_goal/{ag}")
    fromfiles = [rr[s]["value_inf"] for s in sorted(rr)]
    print("E2 align", ag, sorted(rr), np.allclose(fromfiles, g[ag]["value_inf"]), np.allclose([rr[s]["value_T3"] for s in sorted(rr)], g[ag]["value_T3"]))

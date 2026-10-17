"""Grades the pre-declared our-code-first audit variants (the README full-scale pre-declaration: T-A window-DRQN, T-B
hard_sigmoid) against the shipped DDRQN and DQN of runs/full, paired by seed on identical E1 streams.
Usage: python scripts/variants_report.py runs/full_variants [runs/full]"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import data as D  # noqa: E402
import eval as EV  # noqa: E402
from rl_jax_bench.stats import holm, mean_se, paired_compare, t_interval  # noqa: E402

AUD = sys.argv[1]
G3 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "runs", "full")


def one_sided(a, b):
    c = paired_compare(a, b, n_boot=10_000)
    c["p_one"] = c["t_p"] / 2 if c["t"] > 0 else 1 - c["t_p"] / 2
    return c


def main():
    la = json.load(open(os.path.join(AUD, "lanes.json")))
    ids = [tuple(x) for x in la["ids"]]
    ntr = la["config"]["train_cpis"]
    g3 = json.load(open(os.path.join(G3, "lanes.json")))
    g3ids = [tuple(x) for x in g3["ids"]]
    bz = np.load(os.path.join(G3, "baselines.npz"))
    arms = [a for a in la["config"]["arms"] if os.path.exists(os.path.join(AUD, f"arm_{a}.npz"))]
    R = dict(arms=arms, ids=ids)
    res = {}
    for arm in arms:
        z = np.load(os.path.join(AUD, f"arm_{arm}.npz"))
        o, fp = z["outs"][:, ntr:], z["fps"][:, ntr:]
        same = all(np.array_equal(fp[i], bz["pistar__fps"][g3ids.index(ids[i])]) for i in range(len(ids)))
        res[arm] = dict(stream_identity_vs_full=bool(same), metrics={ids[i]: EV.lane_metrics(o[i], [70]) for i in range(len(ids))},
                        frozen={ids[i]: EV.lane_metrics(z["frozen_outs"][i], [70]) for i in range(len(ids))} if "frozen_outs" in z.files else None,
                        probe_agree_final=float(np.nanmean(z["probe_agree"][:, -1])))
    out = {}
    for rd in ("eq3", "lin"):
        seeds = sorted(s for e, r, s in ids if r == rd)
        g = {s: json.load(open(os.path.join(G3, f"seed_{s}", "metrics.json")))["lanes"][f"mc_p04/{rd}"]["metrics"] for s in seeds}
        ship = {m: np.array([g[s][m]["J_conv"] for s in seeds]) for m in ("ddrqn", "dqn", "pistar")}
        ship_sd = {m: np.array([g[s][m]["sigma_within"] for s in seeds]) for m in ("ddrqn", "dqn")}
        tests = {}
        for arm in arms:
            v = np.array([res[arm]["metrics"][("mc_p04", rd, s)]["J_conv"] for s in seeds])
            sd = np.array([res[arm]["metrics"][("mc_p04", rd, s)]["sigma_within"] for s in seeds])
            ad = np.array([res[arm]["metrics"][("mc_p04", rd, s)]["adapt_pct"] for s in seeds])
            vs_ship = one_sided(v, ship["ddrqn"])
            vs_dqn = paired_compare(v, ship["dqn"], n_boot=10_000)
            gap = ship["pistar"] - v
            tests[arm] = dict(J_conv=dict(zip(("mean", "se"), mean_se(v))), sigma_within=dict(zip(("mean", "se"), mean_se(sd))),
                              adapt_pct=dict(zip(("mean", "se"), mean_se(ad))),
                              vs_shipped_ddrqn={k: vs_ship[k] for k in ("diff_mean", "diff_se", "t95", "boot95", "wins", "n", "p_one")},
                              vs_dqn={k: vs_dqn[k] for k in ("diff_mean", "diff_se", "t95", "boot95", "wins", "n", "t_p")},
                              sd_vs_dqn={k: v_ for k, v_ in paired_compare(ship_sd["dqn"], sd, n_boot=2000).items() if k in ("diff_mean", "t95", "wins")},
                              H_a_gap_t95=t_interval(gap), approaches=bool(t_interval(gap)[1] <= 0.02))
        adj = holm({a: tests[a]["vs_shipped_ddrqn"]["p_one"] for a in arms})
        for a in arms:
            t = tests[a]
            t["p_holm"] = adj[a]["p_holm"]
            t["SUPPORTED"] = bool(t["p_holm"] < 0.05 and t["vs_shipped_ddrqn"]["diff_mean"] > 0)
            t["EXPLAINS"] = bool(t["SUPPORTED"] and t["vs_dqn"]["t95"][1] >= 0)
        out[rd] = dict(tests=tests, shipped=dict(ddrqn=mean_se(ship["ddrqn"]), dqn=mean_se(ship["dqn"]), pistar=mean_se(ship["pistar"])))
    R["results"] = out
    R["stream_identity"] = {a: res[a]["stream_identity_vs_full"] for a in arms}
    R["probe_agree_final"] = {a: res[a]["probe_agree_final"] for a in arms}
    with open(os.path.join(AUD, "variants_tests.json"), "w") as f:
        json.dump(R, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    for rd, v in out.items():
        print(f"== {rd} shipped ddrqn {v['shipped']['ddrqn']} dqn {v['shipped']['dqn']} pi* {v['shipped']['pistar']}")
        for a, t in v["tests"].items():
            s, q = t["vs_shipped_ddrqn"], t["vs_dqn"]
            print(f"  {a}: J_conv={t['J_conv']['mean']:.4f}±{t['J_conv']['se']:.4f} sd={t['sigma_within']['mean']:.4f} "
                  f"adapt={t['adapt_pct']['mean']:.1f} | vs ship {s['diff_mean']:+.4f} t95=[{s['t95'][0]:+.4f},{s['t95'][1]:+.4f}] "
                  f"p1={s['p_one']:.2e} holm={t['p_holm']:.2e} wins={s['wins']}/{s['n']} | vs DQN {q['diff_mean']:+.4f} "
                  f"t95=[{q['t95'][0]:+.4f},{q['t95'][1]:+.4f}] | SUPPORTED={t['SUPPORTED']} EXPLAINS={t['EXPLAINS']} H-a={t['approaches']}")
    print("stream identity", R["stream_identity"], "probe", R["probe_agree_final"])


if __name__ == "__main__":
    main()

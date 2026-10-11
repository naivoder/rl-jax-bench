"""small-scale comparison tables vs the paper numbers read off Figs. B.10 / D.12 -> runs/small_tables.json."""
import json
import os
import sys

from rl_jax_bench.stats import paired_compare  # noqa: E402

from src.eval import RUNS, judge, scr_arm_table  # noqa: E402

B10 = {  # (first bin, last bin) read off Fig. B.10 (±0.03); None = off-axis (> 1.2)
    "tanh_0.01": (0.38, 0.88), "tanh_0.003": (0.50, 0.88), "tanh_0.001": (0.67, 0.83),
    "sigmoid_0.01": (0.42, 0.93), "sigmoid_0.003": (0.75, 0.80), "sigmoid_0.001": (0.88, 0.76),
    "elu_0.01": (0.48, 0.70), "elu_0.003": (0.55, 0.69), "elu_0.001": (0.75, 0.68),
    "relu_0.01": (0.60, None), "relu_0.003": (0.65, 1.0), "relu_0.001": (0.68, 0.82),
    "leaky_relu_0.01": (0.62, 0.95), "leaky_relu_0.003": (0.60, 1.07), "leaky_relu_0.001": (0.70, 0.85),
    "swish_0.01": (0.55, None), "swish_0.003": (0.60, 0.75), "swish_0.001": (0.65, 0.70)}
D12 = {"bp": {r: 0.76 for r in ("1e-05", "0.0001", "0.001", "0.01")},
       "overall": {"1e-05": 0.45, "0.0001": 0.32, "0.001": 0.36, "0.01": 0.57},
       "mean_corrected_contribution": {"1e-05": 0.47, "0.0001": 0.325, "0.001": 0.37, "0.01": 0.57},
       "contribution": {"1e-05": 0.48, "0.0001": 0.39, "0.001": 0.47, "0.01": 0.67},
       "weight_magnitude": {"1e-05": 0.48, "0.0001": 0.39, "0.001": 0.475, "0.01": 0.675},
       "adaptation": {"1e-05": 0.64, "0.0001": 0.63, "0.001": 0.67, "0.01": 0.80},
       "random": {"1e-05": 0.49, "0.0001": 0.42, "0.001": 0.78, "0.01": None}}


def main(prefix="g2"):
    out = {"b10": {}, "d12": {}, "methods": {}}
    t = scr_arm_table(f"{prefix}_scr_b10")
    for a, (pf, pl) in B10.items():
        v = t[a]
        out["b10"][a] = {"first_bin": v["first_bin"], "paper_first": pf, "judge_first": judge(v["first_bin"][0], pf, 0.03),
                         "last_bin": v["last_bin"], "paper_last": pl,
                         "judge_last": judge(v["last_bin"][0], pl, 0.03) if pl else ("MATCH" if v["last_bin"][0] > 1.2 else "DIVERGENT"),
                         "lop": v["last_250k"][0] > v["min_bin_first_1M"][0],
                         "paired_last250k_vs_min_first1M": paired_compare(v["per_seed_last_250k"], v["per_seed_min_bin_first_1M"])}
    lin = scr_arm_table(f"{prefix}_scr_linear")
    out["linear"] = {a: lin[a]["whole_run"] for a in lin}
    d = scr_arm_table(f"{prefix}_scr_d12")
    for u, row in D12.items():
        for r, p in row.items():
            a = "bp" if u == "bp" else f"{u}_{r}"
            if u != "bp" and a not in d:
                a = f"{u}_{float(r):g}"
            v = d[a]["whole_run"]
            out["d12"][f"{u}@{r}"] = {"ours": v, "paper": p, "judge": judge(v[0], p, 0.01) if p else
                                      ("MATCH" if v[0] > 0.9 else "DIVERGENT")}
    m = scr_arm_table(f"{prefix}_scr_methods")
    for a in m:
        out["methods"][a] = {"whole_run": m[a]["whole_run"], "last_bin": m[a]["last_bin"],
                             "vs_bp": paired_compare(m[a]["per_seed_whole_run"], m["bp"]["per_seed_whole_run"]) if a != "bp" else None}
    with open(os.path.join(RUNS, f"{prefix}_tables.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out["b10"].items():
        print(f"B10 {k:18s} first {v['first_bin'][0]:.3f} vs {v['paper_first']} {v['judge_first']:9s} last {v['last_bin'][0]:.3f}±{v['last_bin'][1]:.3f} vs {v['paper_last']} {v['judge_last']:9s} LoP={v['lop']} p={v['paired_last250k_vs_min_first1M']['t_p']:.2g}")
    for k, v in out["d12"].items():
        print(f"D12 {k:36s} {v['ours'][0]:.3f}±{v['ours'][1]:.3f} vs {v['paper']} {v['judge']}")
    for k, v in out["methods"].items():
        vb = v["vs_bp"]
        print(f"MET {k:12s} {v['whole_run'][0]:.3f}±{v['whole_run'][1]:.3f}" + (f" diff vs bp {vb['diff_mean']:+.3f} p={vb['t_p']:.2g}" if vb else ""))


if __name__ == "__main__":
    main(*sys.argv[1:])

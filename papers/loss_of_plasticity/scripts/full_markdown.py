"""Render the full-scale tables of runs/full_tables.json as Markdown rows (summarized in the README)."""
import json
import sys

from src.eval import RUNS

d = json.load(open(f"{RUNS}/full_tables.json"))
f3 = lambda v: f"{v[0]:.3f} ± {v[1]:.3f}"
pct = lambda x: "—" if x is None else f"{x:+.1f} %"


def b10():
    print("| arm | first 40k bin: paper | ours (n = 100) | Δ% | judgment | last 40k bin: paper | ours | Δ% | judgment | authors' code (first / last bin, n = 10) | Δ LoP (Holm p, one-sided) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for a, r in d["b10"].items():
        o = r["oracle"]
        os_ = f"{o['first_bin'][0]:.3f} / {o['last_bin'][0]:.3f}" if o else "—"
        pl = r["paper_last"] if r["paper_last"] else "> 1.2 (off-axis)"
        print(f"| {a} | {r['paper_first']} | {f3(r['first_bin'])} | {pct(r['d_first_pct'])} | {r['judge_first']} | {pl} | "
              f"{f3(r['last_bin'])} | {pct(r['d_last_pct'])} | {r['judge_last']} | {os_} | +{r['lop_diff']:.3f} ({r['p_holm']:.0e}) |")


def d12(key):
    rows = d[key]["rows"]
    print("| utility | ρ | paper | ours (n = 100) | Δ% | judgment | authors' code (our streams) |")
    print("|---|---|---|---|---|---|---|")
    for a, r in rows.items():
        u, rho = ("BP", "—") if a == "bp" else a.rsplit("_", 1)
        o = f"{r['oracle'][0]:.3f} ± {r['oracle'][1]:.3f} (n = 3)" if r.get("oracle") else "—"
        print(f"| {u} | {rho} | {r['paper'] if r['paper'] else '> 0.9'} | {f3(r['ours'])} | {pct(r['d_pct'])} | {r['judge']} | {o} |")


def tests(key):
    for k, v in d[key]["tests"].items():
        print(f"| {k} | {v['diff_mean']:+.4f} ± {v['diff_se']:.4f} | [{v['boot95'][0]:+.4f}, {v['boot95'][1]:+.4f}] | {v['t_p']:.1e} | {100 - v['wins']}/{v['n']} |")


if __name__ == "__main__":
    {"b10": b10, "d12rs": lambda: d12("d12_rs"), "d12flip": lambda: d12("d12_flip"),
     "trs": lambda: tests("d12_rs"), "tflip": lambda: tests("d12_flip")}[sys.argv[1]]()

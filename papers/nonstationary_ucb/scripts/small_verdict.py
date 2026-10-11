"""Small-scale verdict: the pre-declared criteria from runs/small/summary.json plus the labelled D-UCB diagnostic
(paired per replication under CRN: the diagnostic configs share exp_id/cfg_idx and policy keys with e1/e2).

    uv run python scripts/small_verdict.py   -> runs/small/verdict.json
"""
import json
import os
import sys

import numpy as np

from rl_jax_bench.stats import paired_compare  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SEEDS = (100, 101, 102)


def load(d, s):
    return json.load(open(os.path.join(ROOT, d, f"seed_{s}", "metrics.json")))


def main():
    summ = json.load(open(os.path.join(ROOT, "runs/small/summary.json")))
    out = {"criteria": summ["checks"], "rule_PASS": summ["PASS"], "diagnostic": {}}
    for e in ("e1", "e2"):
        base = [load(f"runs/small/{e}", s) for s in SEEDS]
        dia = [load(f"runs/small_diag/{e}", s) for s in SEEDS]
        assert all(b["env"]["uniforms_fp_sha256"] == d["env"]["uniforms_fp_sha256"] for b, d in zip(base, dia))
        ro = base[0]["config"]["readoff"]
        rec = {}
        ref = np.concatenate([b["policies"]["ducb"]["final_regret"] for b in base])
        for v in ("ducb_padB", "ducb_lit"):
            a = np.concatenate([d["policies"][v]["final_regret"] for d in dia])
            pc = paired_compare(a, ref)  # per replication, n = 1500, CRN-paired with the as-printed D-UCB
            rec[v] = {"final_regret_mean": float(a.mean()),
                      "seed_means": [d["policies"][v]["final_regret_mean"] for d in dia],
                      "regret_at_mean": {k: float(np.mean([d["policies"][v]["regret_at_mean"][k] for d in dia]))
                                         for k in dia[0]["policies"][v]["regret_at_mean"]},
                      "final_arm1_mean": float(np.mean([d["policies"][v]["final_arm1_mean"] for d in dia])),
                      "minus_ducb_as_printed": {k: pc[k] for k in ("diff_mean", "diff_se", "t95", "n")},
                      "paper_ducb_readoff": ro["final_regret"]["ducb"],
                      "rel_diff_to_paper": float((a.mean() - ro["final_regret"]["ducb"]) / ro["final_regret"]["ducb"])}
        a = np.concatenate([d["policies"]["ducb_padB"]["final_regret"] for d in dia])
        rec["padB_minus_other_arms"] = {}
        for n in ("ucb1", "exp3s", "swucb"):
            o = np.concatenate([b["policies"][n]["final_regret"] for b in base])
            pc = paired_compare(a, o)
            rec["padB_minus_other_arms"][n] = {k: pc[k] for k in ("diff_mean", "diff_se", "t95", "n")}
        out["diagnostic"][e] = rec
    out["verdict"] = ("PASS on criteria 1-2 (integrity, order of magnitude). Criterion 3 FAILS for the as-printed "
                      "D-UCB on E1 (D-UCB > EXP3.S). Debugged: no implementation bug (Toy-check exact replay); the "
                      "cause is the D-UCB padding constant (diagnostic above). See README Small-scale.")
    with open(os.path.join(ROOT, "runs/small/verdict.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({e: {v: (round(r["final_regret_mean"], 1), round(r["minus_ducb_as_printed"]["diff_mean"], 2),
                              round(r["minus_ducb_as_printed"]["diff_se"], 2)) for v, r in d.items() if v.startswith("ducb")}
                      for e, d in out["diagnostic"].items()}))


if __name__ == "__main__":
    main()

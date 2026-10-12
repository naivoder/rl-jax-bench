"""Three-way comparison: paper / authors' code on our hardware / ours at small-scale scale (official-code check, standing rule 2).

Reads (numpy only, no JAX):
  * official training runs (MadryLab/implementation-matters@5ee6ecb, unmodified Trainer, gymnasium MuJoCo v4 proxies):
      $OFFICIAL_ROOT/ref_full/<env>_<alg>_s<k>.json          (488 its, paper Table-1 settings)
      $OFFICIAL_ROOT/ref_full/hopper_ppo_official_s<k>.json  (488 its, the repo's own Hopper-PPO LRs)
      $OFFICIAL_ROOT/ref_diag/<env>_trpo_normrew_s<k>.json   (49 its, TRPO + reward normalisation)
  * official-primitive gradient quality: $OFFICIAL_ROOT/grad_ref/<env>_<alg>_s<k>.json
  * ours: runs/small/small_report.json
Paper numbers: read off the paper's figures (Fig 8 final means, Figs 3/12 held-out MRE medians, Figs 9/10 it-0 cosines).
Writes runs/oracle/official_reference.json and prints markdown tables (saved as runs/oracle/official_reference_tables.txt).
"""
import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFF = os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official"))
ARMS = ["hopper_ppo", "hopper_trpo", "walker2d_ppo", "walker2d_trpo"]

# Fig 8 : mean of 5 agents over the last ~20 iterations
PAPER_FINAL = {"hopper_ppo": 2450, "hopper_trpo": 2408, "walker2d_ppo": 2396, "walker2d_trpo": 2892}
# Figs 3 / 12 : held-out medians over 5 agents at it 150 / 450
PAPER_VAL = {
    "walker2d_trpo": {"gae": {150: 0.042, 450: 0.051}, "ret": {150: 0.30, 450: 0.41}},
    "walker2d_ppo": {"gae": {150: 0.018, 450: 0.043}, "ret": {150: 0.27, 450: 0.30}},
    "hopper_trpo": {"gae": {150: 0.012, 450: 0.030}, "ret": {150: 0.20, 450: 0.28}},
    "hopper_ppo": {"gae": {150: 0.002, 450: 0.030}, "ret": {150: 0.046, 450: 0.35}},
}
# Figs 9 / 10 it 0 : mean of 3 agents; 2e3 is the paper's own log-interpolation
PAPER_GRAD0 = {
    "hopper_ppo": {"pw": {100: 0.012, 1000: 0.098, 2000: 0.282, 10000: 0.712},
                   "tt": {100: 0.115, 1000: 0.337, 2000: 0.472, 10000: 0.787}},
    "hopper_trpo": {"pw": {100: 0.053, 1000: 0.508, 2000: 0.634, 10000: 0.925},
                    "tt": {100: 0.248, 1000: 0.711, 2000: 0.783, 10000: 0.952}},
    "walker2d_ppo": {"pw": {100: 0.091, 1000: 0.531, 2000: 0.657, 10000: 0.949},
                     "tt": {100: 0.305, 1000: 0.725, 2000: 0.794, 10000: 0.956}},
    "walker2d_trpo": {"pw": {100: 0.104, 1000: 0.532, 2000: 0.657, 10000: 0.947},
                      "tt": {100: 0.314, 1000: 0.729, 2000: 0.799, 10000: 0.960}},
}


def mse(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")]


def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        out.append(d)
    return out


def rewards(runs, lo, hi):
    return [float(np.mean([h["mean_reward"] for h in r["history"] if lo <= h["it"] <= hi
                           and np.isfinite(h["mean_reward"])])) for r in runs]


def val_at(runs, it, key, split="heldout"):
    v = [h[split][key] for r in runs for h in r["history"] if h["it"] == it and split in h]
    return v


def main():
    rep = {}
    g2 = json.load(open(os.path.join(REPO, "runs", "small", "small_report.json")))["report"]
    for arm in ARMS:
        env, alg = arm.split("_")
        R = {}
        full = [r for r in load(f"{OFF}/ref_full/{arm}_s[0-9].json") if r.get("iters_done", 0) >= 488]
        short = load(f"{OFF}/ref/{arm}_s[0-9].json")
        R["n_full"], R["n_short"] = len(full), len(short)
        R["reward"] = {
            "paper_final": PAPER_FINAL[arm], "paper_it48": g2[arm]["paper_reward_it48"],
            "official_final_last20": mse(rewards(full, 468, 487)) if full else None,
            "official_final_per_seed": rewards(full, 468, 487) if full else None,
            "official_it40_48": mse(rewards(full, 40, 48)) if full else None,
            "ours_it40_48": g2[arm]["reward_last9"],
        }
        if arm == "hopper_ppo":
            od = [r for r in load(f"{OFF}/ref_full/hopper_ppo_official_s[0-9].json") if r.get("iters_done", 0) >= 488]
            R["reward"]["official_defaults_final_last20"] = mse(rewards(od, 468, 487)) if od else None
            R["reward"]["official_defaults_per_seed"] = rewards(od, 468, 487) if od else None
            R["reward"]["official_defaults_it40_48"] = mse(rewards(od, 40, 48)) if od else None
        # held-out value MREs (official formula = gae_mre_code; dimensionless = gae_mre_true)
        ours_metrics = [json.load(open(f)) for f in
                        sorted(glob.glob(os.path.join(REPO, "runs", "small", arm, "seed_*", "metrics.json")))]
        V = {}
        for it in (10, 20, 30, 40, 150, 450):
            row = {}
            for key, pk in (("returns_mre", "ret"), ("gae_mre_code", "gae"), ("gae_mre_true", None)):
                off = val_at(full, it, key) if full else []
                row[key] = {"official_median": float(np.median(off)) if off else None,
                            "official": off}
                if it <= 40:     # per-seed values straight from our small-scale metrics.json
                    ours = [h["heldout_" + key] for m in ours_metrics for h in m["history"]
                            if h["it"] == it and ("heldout_" + key) in h]
                    row[key]["ours_median"] = float(np.median(ours)) if ours else None
                    row[key]["ours"] = ours
                    if pk:
                        row[key]["paper"] = g2[arm]["paper_value"]["heldout_" + pk].get(str(it))
                elif pk:
                    row[key]["paper"] = PAPER_VAL[arm][pk][it]
            V[it] = row
        R["value_heldout"] = V
        if alg == "trpo":
            diag = load(f"{OFF}/ref_diag/{env}_trpo_normrew_s[0-9].json")
            R["trpo_normrew_diag"] = {
                it: {k: float(np.median(val_at(diag, it, k))) for k in ("returns_mre", "gae_mre_code", "gae_mre_true")}
                for it in (10, 20, 30, 40)} if diag else None
            R["trpo_normrew_reward_it40_48"] = mse(rewards(diag, 40, 48)) if diag else None
            nf = [r for r in load(f"{OFF}/ref_full/{env}_trpo_normrew_s[0-9].json") if r.get("iters_done", 0) >= 488]
            if nf:   # H1 (TRPO trained with reward normalisation) vs H2 (dimensionless MRE on raw rewards)
                R["trpo_normrew_full"] = {
                    "n": len(nf), "final_last20": mse(rewards(nf, 468, 487)),
                    "final_per_seed": rewards(nf, 468, 487),
                    "value": {it: {k: float(np.median(val_at(nf, it, k)))
                                   for k in ("returns_mre", "gae_mre_code", "gae_mre_true")} for it in (150, 450)}}
            ours_d = [json.load(open(f)) for f in sorted(glob.glob(
                os.path.join(REPO, "runs", "oracle", "trpo_rewnorm", f"{env}_trpo-rewnormon", "seed_*", "metrics.json")))]
            if ours_d:
                R["trpo_normrew_ours"] = {it: {k: float(np.median([h["heldout_" + k] for r in ours_d for h in r["history"]
                                                                    if h["it"] == it and ("heldout_" + k) in h]))
                                               for k in ("returns_mre", "gae_mre_code", "gae_mre_true")}
                                          for it in (10, 20, 30, 40)}
                R["trpo_normrew_ours_reward_it40_48"] = mse(
                    [np.nanmean([h["mean_reward"] for h in r["history"] if 40 <= h["it"] <= 48]) for r in ours_d])
        # gradient quality
        G = load(f"{OFF}/grad_ref/{arm}_s[0-9].json")
        R["n_grad"] = len(G)
        gq = {}
        for it in ("0", "15", "45"):
            gq[it] = {}
            for n in (100, 1000, 2000, 10000):
                cell = {}
                if G:
                    cell["official_pw"] = mse([g["cells"][it][str(n)]["pairwise"] for g in G])
                    cell["official_tt"] = mse([g["cells"][it][str(n)]["to_true"] for g in G])
                    cell["official_tt_raw"] = mse([g["cells"][it][str(n)]["raw"]["to_true"] for g in G])
                ours = g2[arm]["grad"][f"it{it}"][str(n)]
                cell["ours_pw"], cell["ours_tt"] = ours["pairwise"], ours["to_true"]
                cell["ours_tt_raw"] = ours.get("to_true_rawadv")
                if it == "0":
                    cell["paper_pw"], cell["paper_tt"] = PAPER_GRAD0[arm]["pw"][n], PAPER_GRAD0[arm]["tt"][n]
                gq[it][n] = cell
        R["grad"] = gq
        rep[arm] = R
    os.makedirs(os.path.join(REPO, "runs", "oracle"), exist_ok=True)
    json.dump(rep, open(os.path.join(REPO, "runs", "oracle", "official_reference.json"), "w"), indent=1)

    f = lambda x: "–" if x is None else (f"{x[0]:.0f} ± {x[1]:.0f}" if isinstance(x, list) else f"{x:.0f}")
    print("| arm | paper Fig 8 final | official final (its 468–487) | official it 40–48 | ours it 40–48 | paper it 48 |")
    print("|---|---|---|---|---|---|")
    for arm, R in rep.items():
        r = R["reward"]
        print(f"| {arm} | {r['paper_final']} | {f(r['official_final_last20'])} | {f(r['official_it40_48'])} | "
              f"{f(r['ours_it40_48'])} | {r['paper_it48']:.0f} |")
    r = rep["hopper_ppo"]["reward"]
    print(f"| hopper_ppo, repo defaults (3e-4/2.5e-4) | – | {f(r.get('official_defaults_final_last20'))} | "
          f"{f(r.get('official_defaults_it40_48'))} | – | – |")
    g = lambda x: "–" if x is None else f"{x:.3g}"
    print("\n| arm | metric | it | paper | official | ours |")
    print("|---|---|---|---|---|---|")
    for arm, R in rep.items():
        for key in ("returns_mre", "gae_mre_code", "gae_mre_true"):
            for it in (10, 20, 30, 40, 150, 450):
                c = R["value_heldout"][it][key]
                print(f"| {arm} | {key} | {it} | {g(c.get('paper'))} | {g(c['official_median'])} | "
                      f"{g(c.get('ours_median'))} |")
    print("\n| arm | it | n | paper pw / tt | official pw / tt (mean ± SE) | ours pw / tt (mean ± SE) |")
    print("|---|---|---|---|---|---|")
    for arm, R in rep.items():
        for it, row in R["grad"].items():
            for n, c in row.items():
                p = f"{c['paper_pw']:.3f} / {c['paper_tt']:.3f}" if "paper_pw" in c else "–"
                o = (f"{c['official_pw'][0]:.3f}±{c['official_pw'][1]:.3f} / {c['official_tt'][0]:.3f}±"
                     f"{c['official_tt'][1]:.3f}") if "official_pw" in c else "–"
                u = f"{c['ours_pw'][0]:.3f}±{c['ours_pw'][1]:.3f} / {c['ours_tt'][0]:.3f}±{c['ours_tt'][1]:.3f}"
                print(f"| {arm} | {it} | {n} | {p} | {o} | {u} |")
    for arm in ("hopper_trpo", "walker2d_trpo"):
        print(f"\n{arm} TRPO+reward-norm diagnostic (official code):", json.dumps(rep[arm].get("trpo_normrew_diag")),
              "reward it40-48", rep[arm].get("trpo_normrew_reward_it40_48"))
        print(f"{arm} TRPO+reward-norm FULL (official, H1 test):", json.dumps(rep[arm].get("trpo_normrew_full")))
        print(f"{arm} TRPO+reward-norm diagnostic (ours):", json.dumps(rep[arm].get("trpo_normrew_ours")),
              "reward it40-48", rep[arm].get("trpo_normrew_ours_reward_it40_48"))


if __name__ == "__main__":
    main()

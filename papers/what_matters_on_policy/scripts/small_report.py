"""small-scale report: per env x arm mean +- SE of the per-seed score (and score_guarded), ratio to REC, the paper's
raw-data ratio (paper_raw_p95.txt, conditional 95th percentiles of the official study data) and whether the
DIRECTION agrees. 3 seeds = direction only (no tests). Usage: python scripts/small_report.py <tag>"""
import glob, json, math, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# paper ratio arm/REC-value from src/configs/paper_raw_p95.txt
# (group in brackets; env order: hopper, walker2d, halfcheetah)
PAPER = {
    "A1_obsnorm_none":   ((1665 / 1661, 904 / 1808, 1081 / 2088), "F input_normalization None/Avg"),
    "A2a_lastlayer_1":   ((1315 / 1350, 607 / 673, 1797 / 2213), "E2 last_kernel_init_policy_scaling 1.0/0.01"),
    "A2b_initstd_1":     ((1419 / 1281, 651 / 736, 1613 / 2857), "E2 initial_action_std 1.0/0.5"),
    "A2c_initstd_0p1":   ((609 / 1281, 494 / 736, 2168 / 2857), "E2 initial_action_std 0.1/0.5"),
    "A3a_shared_bc1":    ((1334 / 1360, 911 / 657, 2349 / 2331), "E shared@baseline_cost 1 / separate (marginal shared/separate: 0.878, 1.040, 0.854)"),
    "A3b_shared_bc0p1":  ((944 / 1360, 468 / 657, 1666 / 2331), "E shared@0.1 / separate"),
    "A3c_shared_bc10":   ((1418 / 1360, 815 / 657, 2350 / 2331), "E shared@10 / separate"),
    "A4_lambda_0p95":    ((1912 / 1910, 1419 / 1594, 1498 / 1867), "G gae_lambda 0.95/0.9"),
    "A4b_lambda_0p99":   ((1830 / 1910, 1274 / 1594, 1522 / 1867), "G gae_lambda 0.99/0.9"),
    "A5a_lr_1em4":       ((1579 / 1775, 622 / 954, 950 / 1191), "J adam lr 1e-4/3e-4"),
    "A5b_lr_1em3":       ((1810 / 1775, 938 / 954, 1081 / 1191), "J adam lr 1e-3/3e-4"),
    "A6_entropy_1em3":   ((1339 / 1714, 761 / 839, 1699 / 1451), "K entropy penalty 1e-3 / no regularisation"),
    "A7_valuenorm_none": ((1617 / 1691, 1776 / 1035, 1376 / 1903), "F value (reward) normalization None/PopArt"),
    "A8_advnorm_true":   ((1679 / 1633, 1516 / 1573, 1685 / 1780), "F normalize_advantages True/False"),
}
ENVI = {"hopper": 0, "walker2d": 1, "halfcheetah": 2}


def main(tag):
    lanes = []
    for f in sorted(glob.glob(os.path.join(REPO, "runs", tag, "*", "seed_*", "metrics.json"))):
        lanes += json.load(open(f))["lanes"]
    out = []
    for env in ["halfcheetah", "walker2d", "hopper"]:
        L = [l for l in lanes if l["env"] == env]
        if not L:
            continue
        arms = list(dict.fromkeys(l["arm"] for l in L))
        rec = {l["seed"]: l["score_guarded"] for l in L if l["arm"] == "REC"}
        rec_m = np.mean(list(rec.values()))
        step0 = np.mean([l["curve"][0] for l in L])
        print(f"\n## {env}  (random-policy step-0 return {step0:.0f})")
        print("| arm | n | score mean ± SE | final mean | det final | ratio to REC | paired diff ± SE | paper ratio | direction | non-finite |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for a in arms:
            la = [l for l in L if l["arm"] == a]
            sc = np.array([l["score_guarded"] for l in la])
            se = sc.std(ddof=1) / math.sqrt(len(sc)) if len(sc) > 1 else float("nan")
            nf = sum(l["nonfinite"] for l in la)
            fin = np.mean([l["final"] for l in la])
            det = np.mean([l["det_final"] for l in la if l["det_final"] is not None])
            r = sc.mean() / rec_m
            d = np.array([l["score_guarded"] - rec[l["seed"]] for l in la])
            dse = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            pr, dirn = "", ""
            if a in PAPER:
                p = PAPER[a][0][ENVI[env]]
                pr = f"{p:.3f}"
                ours_sign = 0 if abs(d.mean()) < 2 * dse else int(np.sign(d.mean()))
                pap_sign = 0 if abs(p - 1) < 0.05 else int(np.sign(p - 1))
                dirn = "agree" if ours_sign == pap_sign else ("ours≈0" if ours_sign == 0 else ("paper≈0" if pap_sign == 0 else "OPPOSITE"))
            print(f"| {a} | {len(sc)} | {sc.mean():.0f} ± {se:.0f} | {fin:.0f} | {det:.0f} | {r:.3f} | "
                  f"{'' if a == 'REC' else f'{d.mean():+.0f} ± {dse:.0f}'} | {pr} | {dirn} | {nf} |")
            out.append({"env": env, "arm": a, "n": len(sc), "mean": sc.mean(), "se": se, "final": fin, "det_final": det,
                        "ratio": r, "diff": d.mean(), "diff_se": dse, "paper_ratio": pr, "direction": dirn, "nonfinite": nf})
    json.dump(out, open(os.path.join(REPO, "runs", tag, "small_table.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1])

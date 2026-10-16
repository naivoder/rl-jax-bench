"""No-ObGD recheck report: the paper-F.4 no-ObGD arm (Adam 1e-7) and the start of the
F.4 halving protocol (Adam 3e-4) on REAL Hopper-v4 (version-matched), against the pre-declared predictions (a)-(d).
Reads runs/recheck/no_obgd_real_hopper/<arm>/seed_<n>/metrics.json, the version-matched full stream AC (same host loop, seeds 1-5)
and the tick-mark-calibrated Fig. 7. Writes runs/recheck/no_obgd_real_hopper/no_obgd_report.json."""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src import eval as ev  # noqa: E402
from rl_jax_bench.stats import mean_se, paired_compare  # noqa: E402

D = os.path.join(REPO, "runs", "recheck", "no_obgd_real_hopper")
S = 50_000
PAPER = json.load(open(os.path.join(REPO, "src", "configs", "paper_digitized.json")))["figures"]["fig7"]["Hopper"]
LSA = os.path.join(REPO, "runs", "recheck", "learner-strength", "real_hopper")


def load(path):
    m = json.load(open(path))
    return m, np.asarray(m["episodes"], float).reshape(-1, 3)


def lanes(arm):
    out = {}
    base = os.path.join(D, arm)
    for sd in sorted(os.listdir(base), key=lambda x: int(x.split("_")[1])):
        m, e = load(os.path.join(base, sd, "metrics.json"))
        out[int(sd.split("_")[1])] = (m, e)
    return out


L7, L3 = lanes("no_obgd@lr=1e-7"), lanes("no_obgd@lr=3e-4")
T = min(m["steps_done"] for m, _ in list(L7.values()) + list(L3.values()))
NB = T // S
grid = [(i + 1) * S for i in range(NB)]
paper = {g: ev.grid_value(PAPER["no_obgd"]["points"], g / 1e6) for g in grid}
paper_full = {g: ev.grid_value(PAPER["stream_ac"]["points"], g / 1e6) for g in grid}


def lane_rows(L):
    rows = {}
    for sd, (m, e) in L.items():
        b = ev.run_bins(e, NB * S, S)
        last = e[(e[:, 0] > (NB - 1) * S) & (e[:, 0] <= NB * S)]
        rows[sd] = dict(bins=np.round(b, 2).tolist(), last_minus_first=round(float(b[-1] - b[0]), 2),
                        pi_drift_mean=m["pi_drift_mean"], pi_drift_max=m["pi_drift_max"],
                        diverged_by_step=m["diverged_by_step"], last_bin_mean_len=round(float(last[:, 2].mean()), 1),
                        steps_done=m["steps_done"])
    return rows


r7, r3 = lane_rows(L7), lane_rows(L3)
old = [1, 2, 3, 4, 5]
fresh = [s for s in r7 if s not in old]


def arm_curve(rows, seeds):
    return [dict(step=g, **dict(zip(("mean", "se"), mean_se([rows[s]["bins"][i] for s in seeds]))))
            for i, g in enumerate(grid)]


# full stream AC, same version-matched host loop , seeds 1-5
full = {}
for s in old:
    _, e = load(os.path.join(REPO, "runs", "review", "ours_mujoco_vm", "stream_ac", f"seed_{s}", "metrics.json"))
    full[s] = ev.run_bins(e, NB * S, S)
paired = {}
for i, g in enumerate(grid):
    a = [full[s][i] for s in old]
    b = [r7[s]["bins"][i] for s in old]
    pc = paired_compare(a, b)
    pc["t_p_one_sided"] = pc["t_p"] / 2 if pc["diff_mean"] > 0 else 1 - pc["t_p"] / 2
    paired[g] = dict(full_mean=float(np.mean(a)), nobgd7_mean=float(np.mean(b)), **pc)

band = (10.0, 40.0)
pred_a = all(band[0] <= x <= band[1] for s in old for x in r7[s]["bins"]) and \
    all(r7[s]["last_minus_first"] < 15 for s in old) and np.mean([r7[s]["pi_drift_mean"] for s in old]) < 1e-3
pred_b = paired[grid[-1]]["wins"] == 5 and paired[grid[-1]]["t_p_one_sided"] < 0.05
pred_c = all(band[0] <= x <= band[1] for s in fresh for x in r7[s]["bins"]) and \
    all(r7[s]["last_minus_first"] < 15 for s in fresh) and np.mean([r7[s]["pi_drift_mean"] for s in fresh]) < 1e-3
pred_d = all(r3[s]["diverged_by_step"] is None for s in r3)

# determinism: seeds 1-3 at 1e-7 vs the LS-A probe (same keys, 9-lane program) -- identical episodes?
same = {}
for s in (1, 2, 3):
    p = os.path.join(LSA, "no_obgd@lr=1e-7", f"seed_{s}", "metrics.json")
    if os.path.exists(p):
        _, ea = load(p)
        _, eb = load(os.path.join(D, "no_obgd@lr=1e-7", f"seed_{s}", "metrics.json"))
        n = min(len(ea), len(eb))
        same[s] = dict(n_compared=n, max_abs_return_diff=float(np.abs(ea[:n, 1] - eb[:n, 1]).max()))

out = dict(steps_compared=T, grid=grid, paper_no_obgd=paper, paper_stream_ac=paper_full,
           no_obgd_1e7=dict(lanes=r7, seeds_1_5=arm_curve(r7, old), fresh=arm_curve(r7, fresh), fresh_seeds=fresh,
                            all=arm_curve(r7, sorted(r7))),
           no_obgd_3e4=dict(lanes=r3, seeds_1_3=arm_curve(r3, sorted(r3))),
           full_stream_ac_version_matched=dict((str(g), mean_se([full[s][i] for s in old])) for i, g in enumerate(grid)),
           paired_full_minus_nobgd7=paired,
           predictions=dict(a_no_learning_seeds_1_5=bool(pred_a), b_full_beats_it_5of5_p05=bool(pred_b),
                            c_no_learning_fresh=bool(pred_c), d_lr3e4_finite=bool(pred_d), band=band),
           determinism_vs_LS_A=same)
json.dump(out, open(os.path.join(D, "no_obgd_report.json"), "w"), indent=1)
print(json.dumps(dict(T=T, preds=out["predictions"], paper=paper, all7=out["no_obgd_1e7"]["all"],
                      fresh=out["no_obgd_1e7"]["fresh"], lr3e4=out["no_obgd_3e4"]["seeds_1_3"],
                      paired_last={k: paired[grid[-1]][k] for k in ("full_mean", "nobgd7_mean", "diff_mean", "diff_se",
                                                                    "wins", "t_p_one_sided", "boot95")},
                      det=same), indent=1))

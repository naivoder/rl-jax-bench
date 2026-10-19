"""Aggregate runs/review/scene_*.json (scripts/review_scenes.py) into the review scene table: per scene and fixed
policy, every M_cov reading with n_episodes_no_track, the ratios to Random (Static / Random averaged over action seeds
0-2), GOSPA live and library, duplicate rates, and the paper's qualitative checks:
  Q1  M_cov ordering Coverage < Random < Static (paper: 574 < 948 < 2631), per reading
  Q2  "Random has a chance of perfectly tracking a target": fraction of Random episodes with a track that reached
      >= 50 updates (operational reading, ASSUMED)
  Q3  SR ordering Coverage > Random > Static
Writes runs/review/scene_report.json and prints markdown rows.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["primary", "dt5", "frob", "dt5_frob", "gnn", "clutter", "clutter5", "n12", "na20"]
READ = ["m_cov", "m_cov_lib", "m_cov_zero", "m_cov_pen"]


def arm_mean(d, k):
    v = [d[s].get(k, np.nan) for s in d]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def main():
    rep = {}
    for name in ORDER:
        f = os.path.join(REPO, "runs", "review", f"scene_{name}.json")
        if not os.path.exists(f):
            continue
        sc = json.load(open(f))
        if len(sc["arms"]) < 5:
            continue
        A = sc["arms"]
        row = {"overrides": sc["overrides"]}
        for arm, d in A.items():
            r = {k: arm_mean(d, k) for k in READ + ["sr", "gospa_d", "gospa_lib_d", "gospa_miss", "gospa_lib_miss",
                                                     "mean_tracks", "n_init", "unassigned_tracks_per_step",
                                                     "dup_frac", "frac_ep_track_ge10_upd", "frac_ep_track_ge50_upd",
                                                     "max_upd_per_episode_mean", "life_median", "overflow"]}
            r["n_no_track"] = [d[s]["n_episodes_no_track"] for s in d]
            row[arm] = r
        rnd = row["random"]
        for arm in A:
            row[arm]["ratio"] = {k: row[arm][k] / rnd[k] if rnd[k] else float("nan")
                                 for k in READ + ["sr", "gospa_d", "gospa_lib_d", "gospa_miss", "gospa_lib_miss"]}
        q1 = {}
        for k in READ:
            c, r, s = row["coverage"][k], row["random"][k], row["static"][k]
            q1[k] = (bool(c < r < s) if np.isfinite(s) else "static undefined") if np.isfinite(c) else None
        row["Q1_cov_lt_rnd_lt_static"] = q1
        row["Q1_cov_lt_rnd"] = {k: bool(row["coverage"][k] < row["random"][k]) for k in READ}
        row["Q2_random_frac_ep_track_ge50"] = row["random"]["frac_ep_track_ge50_upd"]
        row["Q3_sr_order"] = bool(row["coverage"]["sr"] > row["random"]["sr"] > row["static"]["sr"])
        rep[name] = row
    json.dump(rep, open(os.path.join(REPO, "runs", "review", "scene_report.json"), "w"), indent=1)
    print("| scene | Cov/Rnd live | Cov/Rnd lib | Cov/Rnd zero | Cov/Rnd pen | Static/Rnd live (no-track eps per seed) | "
          "Static/Rnd lib | Static/Rnd pen | hand/Rnd live | ceiling/Rnd live | Rnd mean tracks | Rnd eps with a "
          "track ≥ 50 upd | Cov<Rnd<Static (live / lib / zero / pen) | SR order | GOSPA d Cov/Rnd live / lib | "
          "unassigned/step (Rnd, Cov, hand, ceiling) |")
    print("|" + "---|" * 16)
    for name, row in rep.items():
        R = lambda a, k: row[a]["ratio"][k]
        q = row["Q1_cov_lt_rnd_lt_static"]
        print(f"| {name} | {R('coverage','m_cov'):.3f} | {R('coverage','m_cov_lib'):.3f} | {R('coverage','m_cov_zero'):.3f} | "
              f"{R('coverage','m_cov_pen'):.3f} | {R('static','m_cov'):.3f} ({row['static']['n_no_track']}) | "
              f"{R('static','m_cov_lib'):.3f} | {R('static','m_cov_pen'):.3f} | {R('hand','m_cov'):.3f} | "
              f"{R('ceiling','m_cov'):.3f} | {row['random']['mean_tracks']:.3f} | "
              f"{row['Q2_random_frac_ep_track_ge50']:.2f} | {' / '.join(str(q[k])[:5] for k in READ)} | "
              f"{row['Q3_sr_order']} | {R('coverage','gospa_d'):.4f} / {R('coverage','gospa_lib_d'):.4f} | "
              f"{row['random']['unassigned_tracks_per_step']:.3f}, {row['coverage']['unassigned_tracks_per_step']:.3f}, "
              f"{row['hand']['unassigned_tracks_per_step']:.3f}, {row['ceiling']['unassigned_tracks_per_step']:.3f} |")


if __name__ == "__main__":
    main()

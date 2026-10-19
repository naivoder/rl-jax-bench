"""review resolution: re-read the STORED full-scale per-episode metrics (no re-run) under the readings the review
asked for, side by side with the pre-registered one.

   M_cov readings: live (pre-registered; no-track episodes dropped, n reported), m_cov_zero (no-track steps = 0),
      m_cov_pen (no-track steps = 5000, constructed)
   library M_cov (Stone Soup tutorial pipeline, deleted tracks at their deletion step), co-primary
   stochastic (pre-registered) and deterministic (SB3 evaluate_policy / EvalCallback default) learner evaluation
   duplicate / unassigned-track rate per arm (from GOSPA false and missed)
   the grade Random itself (ratio 1.000) would receive under each declared band, and the grades of the fixed policies
Ratios are per seed to Random (seed index pairing, as full-scale), graded with the declared grade_ratio and bands
(scripts/full_report.py, README). Writes runs/review/readings.json and prints a table.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from rl_jax_bench.stats import mean_se  # noqa: E402
from full_report import BANDS, grade_ratio  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G3 = os.path.join(REPO, "runs", "full")
LEARN = ["flat", "bigru", "mhsa", "mhsa_nobc", "mhsa_noae", "mhsa_lit"]
PEN = 5000.0
T, NT, H = 1000, 3, 10_000.0 ** 2 / 2


def per_ep_readings(pe):
    pe = {k: np.asarray(v, float) for k, v in pe.items()}
    frac, m = pe["frac_steps_tracked"], pe["m_cov"]
    mz = np.where(np.isfinite(m), m * frac, 0.0)
    un = pe["gospa_false"] / H / T
    asg = NT - pe["gospa_miss"] / H / T
    return dict(m_cov=m, m_cov_lib=pe["m_cov_lib"], m_cov_zero=mz, m_cov_pen=mz + (1 - frac) * PEN,
                no_track=(~np.isfinite(m)).astype(float), sr=pe["sr"], gospa_d=pe["gospa_d"],
                gospa_miss=pe["gospa_miss"], unassigned=un, assigned=asg,
                dup_frac=np.where(un + asg > 0, un / np.maximum(un + asg, 1e-300), np.nan),
                mean_tracks=pe["mean_tracks"])


def seed_means(list_of_pe):
    out = {}
    for pe in list_of_pe:
        r = per_ep_readings(pe)
        for k, v in r.items():
            fin = v[np.isfinite(v)]
            out.setdefault(k, []).append(float(fin.mean()) if fin.size else np.nan)
            if k == "no_track":
                out.setdefault("n_no_track", []).append(int(v.sum()))
    return {k: np.asarray(v, float) for k, v in out.items()}


def main():
    base = json.load(open(os.path.join(G3, "baselines.json")))["arms"]
    arms = {}
    for a, d in base.items():
        seeds = sorted(d, key=int)
        pes = [d[s]["per_episode"] for s in seeds]
        if len(pes) == 1:
            pes = pes * 5
        arms[a] = seed_means(pes)
    for a in LEARN:
        ms = sorted((json.load(open(f)) for f in glob.glob(os.path.join(G3, a, "seed_*", "metrics.json"))),
                    key=lambda m: m["seed"])
        arms[a] = seed_means([m["per_episode"] for m in ms])
        arms[a + "@det"] = seed_means([m["per_episode_det"] for m in ms])
    rnd = arms["random"]
    rows = {}
    for a, v in arms.items():
        row = {}
        for k in ("m_cov", "m_cov_lib", "m_cov_zero", "m_cov_pen", "sr", "gospa_d", "gospa_miss"):
            x = v[k]
            r = x / rnd[k]
            fin = r[np.isfinite(r)]
            mu, se = mean_se(fin) if fin.size > 1 else ((float(fin[0]) if fin.size else np.nan), None)
            base_arm = a.split("@")[0]
            band = BANDS.get(f"{k if k in ('sr', 'gospa_d', 'gospa_miss') else 'm_cov'}:{base_arm}")
            row[k] = dict(abs=float(np.nanmean(x)), ratio=mu, se=se, n=int(fin.size),
                          grade=grade_ratio(mu, band) if (band and np.isfinite(mu)) else None)
        row["n_no_track_per_seed"] = v["n_no_track"].astype(int).tolist()
        for k in ("unassigned", "assigned", "dup_frac", "mean_tracks"):
            row[k] = float(np.nanmean(v[k]))
        rows[a] = row
    random_grades = {b: grade_ratio(1.0, band) for b, band in BANDS.items()}
    json.dump(dict(rows=rows, random_itself_grade=random_grades, penalty=PEN),
              open(os.path.join(REPO, "runs", "review", "readings.json"), "w"), indent=1)
    print("grade Random itself (ratio 1.000) would get:", random_grades)
    hdr = f"{'arm':14s} " + " ".join(f"{k:>22s}" for k in ("m_cov", "m_cov_lib", "m_cov_zero", "m_cov_pen")) + \
        "   no-track/seed   unassigned/step dup_frac assigned/step"
    print(hdr)
    for a, r in rows.items():
        cells = []
        for k in ("m_cov", "m_cov_lib", "m_cov_zero", "m_cov_pen"):
            c = r[k]
            cells.append(f"{c['abs']:7.0f} {c['ratio']:.3f}{('±%.3f' % c['se']) if c['se'] else '      '} {str(c['grade'] or '')[:4]:4s}")
        print(f"{a:14s} " + " ".join(f"{x:>22s}" for x in cells) +
              f"   {str(r['n_no_track_per_seed']):15s} {r['unassigned']:.3f}  {r['dup_frac']:.3f}  {r['assigned']:.3f}")
    print("\nSR / GOSPA ratios with grades")
    for a, r in rows.items():
        print(f"{a:14s} " + "  ".join(f"{k} {r[k]['ratio']:.4f} {r[k]['grade'] or ''}" for k in ("sr", "gospa_d", "gospa_miss")))


if __name__ == "__main__":
    main()

"""MHSA at lr 1e-3 under the paper-budget protocol (README), CPU only.

Lanes runs/lr1e3/mhsa_lr1e-3/seed_<0-4> share every random stream with the paper-budget MHSA lanes (key-tree id 3).
Applies the declared GO / KILL rule and re-grades H1-H6 with MHSA@1e-3 substituted for MHSA (DESCRIPTIVE ONLY; the
frozen primary family is unchanged).

    python scripts/lr1e3_report.py      # -> runs/lr1e3/report.json, report.txt
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_jax_bench.stats import paired_compare, boot_ci  # noqa: E402,F401
import paper_budget_report as PR  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("LR1E3_DIR", os.path.join(REPO, "runs", "lr1e3"))
BAND, CLOSE_TOP = (0.351, 0.564), 0.487 * 1.2


def main():
    base, B, L, M, keys_sha, _ = PR.load()             # paper-budget lanes + full-scale fixed policies
    ms_ = [json.load(open(os.path.join(OUT, "mhsa_lr1e-3", f"seed_{s}", "metrics.json"))) for s in range(5)]
    for m in ms_:
        assert m["eval_scenarios"]["keys_sha256"] == keys_sha
    rnd = np.array(B["random"]["vals"]["m_cov"], float)
    rep = dict(steps=ms_[0]["env_steps"], modes={})
    for mode, ek in (("stoch", "eval_stochastic"), ("det", "eval_deterministic"), ("final_stoch", "eval_final_stochastic")):
        x = {k: np.array([m[ek][k] for m in ms_], float) for k in PR.KEYS}
        tb = (100 - x["n_episodes_no_track"]).astype(int).tolist()
        r = x["m_cov"] / rnd
        lmode = "det" if mode == "det" else "stoch"
        rep["modes"][mode] = dict(m_cov=PR.ms(x["m_cov"]), ratio=PR.ms(r), track_bearing=tb,
                                  grade="UNDEFINED (too few track-bearing episodes)" if min(tb) <= PR.F4_MIN else PR.grade(r.mean(), *PR.BANDS["m_cov:mhsa"]),
                                  sr_ratio=PR.ms(x["sr"] / np.array(B["random"]["vals"]["sr"])),
                                  gospa_d_ratio=PR.ms(x["gospa_d"] / np.array(B["random"]["vals"]["gospa_d"])),
                                  gospa_miss_ratio=PR.ms(x["gospa_miss"] / np.array(B["random"]["vals"]["gospa_miss"])),
                                  mean_tracks=PR.ms(x["mean_tracks"]))
        if mode == "final_stoch":
            continue
        # H1-H6 with MHSA@1e-3 substituted (descriptive)
        def get(arm, k, _x=x, _lm=lmode):
            if arm == "mhsa":
                return _x[k]
            v = L[(arm, _lm)] if (arm, _lm) in L else B.get(arm)
            return np.asarray(v["vals"][k], float)
        H = {}
        PR.h_family(get, H)
        rep["modes"][mode]["H_substituted"] = H
        # CRN-paired vs the paper-budget MHSA@3e-4 lanes (same streams)
        y = np.asarray(L[("mhsa", lmode)]["vals"]["m_cov"], float)
        pc = paired_compare(x["m_cov"], y)
        rep["modes"][mode]["vs_mhsa_3e-4"] = dict(diff=pc["diff_mean"], se=pc["diff_se"], boot95=pc["boot95"],
                                                  p_two_sided=pc["t_p"], lower=int((x["m_cov"] < y).sum()))
    s = rep["modes"]["stoch"]
    h1 = s["H_substituted"]["H1"]
    flat = np.asarray(L[("flat", "stoch")]["vals"]["m_cov"], float)
    x = np.array([m["eval_stochastic"]["m_cov"] for m in ms_], float)
    d = x - flat
    p1 = float(st.ttest_1samp(d, 0.0, alternative="less").pvalue)
    bc = boot_ci(d)
    beats_flat = bool(p1 < 0.05 and bc[1] < 0)
    in_band = bool(BAND[0] <= s["ratio"]["mean"] <= BAND[1])
    divergent = s["ratio"]["mean"] > CLOSE_TOP
    verdict = "GO" if (in_band or beats_flat) else ("KILL" if (divergent and not beats_flat) else "UNRESOLVED")
    rep["rule"] = dict(ratio=s["ratio"]["mean"], in_band=in_band, divergent=divergent,
                       vs_flat=dict(diff=float(d.mean()), p_one_sided=p1, boot95=bc, beats=beats_flat),
                       verdict=verdict, h1_substituted_p_one_sided=h1.get("p_one_sided"))
    rep["training"] = dict(best_step=[m["best_step"] for m in ms_], ent_last=[m["train_hist"]["entropy"][-1] for m in ms_],
                           ev_last=[m["train_hist"]["ev"][-1] for m in ms_],
                           cnn_alive_final=[m["cnn_alive"]["final"]["alive"] for m in ms_],
                           cnn_alive_best=[m["cnn_alive"]["best"]["alive"] for m in ms_],
                           stalled=[m["stalled_at_end"] for m in ms_],
                           cap_limited_k=int(sum(m["best_step"] >= m["env_steps"] - 1_000_000 for m in ms_)))
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1, default=float)
    out = [f"steps {rep['steps']}"]
    for mode, r in rep["modes"].items():
        out.append(f"{mode}: M_cov {r['m_cov']['mean']:.1f}, ratio {r['ratio']['mean']:.3f} ± {r['ratio']['se']:.3f} "
                   f"{[round(v, 3) for v in r['ratio'].get('per_seed', [])]} grade {r['grade']} tb {r['track_bearing']} "
                   f"SR {r['sr_ratio']['mean']:.2f} Σd {r['gospa_d_ratio']['mean']:.3f} Σmiss {r['gospa_miss_ratio']['mean']:.3f} "
                   f"tracks {r['mean_tracks']['mean']:.3f}")
        if "H_substituted" in r:
            for h, x in r["H_substituted"].items():
                if "diff" in x:
                    out.append(f"   {h}: Δ {x['diff']:.4g} ± {x['diff_se']:.3g} boot {[round(v, 1) for v in x['boot95']]} "
                               f"p1 {x['p_one_sided']:.3g} holm {x['p_holm']:.3g} p2 {x['p_two_sided']:.3g} "
                               f"wins {x['wins_lower']}/5 {x['verdict']} / {x['grade']}")
            out.append(f"   vs MHSA@3e-4 (CRN): {json.dumps(r['vs_mhsa_3e-4'], default=float)}")
    out.append("rule: " + json.dumps(rep["rule"], default=float))
    out.append("training: " + json.dumps(rep["training"], default=float))
    txt = "\n".join(out)
    open(os.path.join(OUT, "report.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()

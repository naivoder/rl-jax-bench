"""Two-way (stream x learner) comparison on the OFFICIAL generator's streams 0-4: our learner (5 init draws per stream:
ours_on_official_streams.json + init_variance_probe.json) vs the official learner (2 runs per stream: torch's own
init from the official pipeline, and our init from init_matched.py).  Also reports how long init-matched
trajectories stay together.  -> runs/oracle_pipeline/learner_on_same_streams.json"""
import json
import os

import numpy as np

R = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs", "oracle_pipeline")


def window(per, k):
    return {"whole_run": per.mean(-1), "last_250k": per[..., -25:].mean(-1), "last_bin": per[..., -4:].mean(-1),
            "first_bin": per[..., :4].mean(-1)}[k]


def main():
    P = json.load(open(f"{R}/init_variance_probe.json"))
    C = json.load(open(f"{R}/ours_on_official_streams.json"))
    C10 = json.load(open(f"{R}/ours_on_official_streams_10.json"))
    S = json.load(open(f"{R}/summary.json"))
    res = {}
    for case in ("flip_tanh_bp_0.01", "flip_relu_bp_0.01", "rs_relu_bp_0.01"):
        M = np.array([np.load(f"{R}/init_matched/{case}_seed{s}.npz")["per10k"] for s in range(5)])
        U = np.array(C10[case]["ours_per10k"])[:5]           # same init (90000+s) as M
        d = np.abs(M - U)
        diverge = [int(np.argmax(r > 1e-2)) if (r > 1e-2).any() else None for r in d]
        res[case] = {"init_matched_first_period_absdiff_gt_1e-2": diverge,
                     "init_matched_whole_run_ours": U.mean(1).tolist(), "init_matched_whole_run_official": M.mean(1).tolist()}
        for k in ("whole_run", "last_250k", "last_bin", "first_bin"):
            W = np.vstack([C[case]["ours_per_seed"][k]] + [r[k] for r in P[case]])        # (5 inits, 5 streams)
            O = np.vstack([np.asarray(S[case]["official_per_seed"][k][:5]), window(M, k)])  # (2 runs, 5 streams)
            dd = O.mean(0) - W.mean(0)
            sw = np.sqrt(((W - W.mean(0)) ** 2).sum() / (W.size - 5))
            so = np.sqrt(((O - O.mean(0)) ** 2).sum() / (O.size - 5))
            se = np.sqrt(sw ** 2 / W.shape[0] + so ** 2 / O.shape[0]) / np.sqrt(5)
            res[case][k] = {"official_minus_ours_per_stream": dd.tolist(), "mean": float(dd.mean()), "se": float(se),
                            "t": float(dd.mean() / se), "rel": float(dd.mean() / W.mean()),
                            "within_stream_sd_ours": float(sw), "within_stream_sd_official": float(so)}
            print(f"[METRIC] same_streams {case} {k} official-ours={dd.mean():+.4f}+-{se:.4f} t={dd.mean() / se:+.2f} "
                  f"rel={dd.mean() / W.mean():+.3f} sd_ours={sw:.3f} sd_off={so:.3f}", flush=True)
        print(f"[INFO] {case} init-matched divergence period (10k steps): {diverge}", flush=True)
    json.dump(res, open(f"{R}/learner_on_same_streams.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""Pack the small-scale differential-test inputs + the AUTHORS' CODE outputs into small committed fixtures.

Each tests/fixtures/official/<case>.npz holds (a) the inputs we fed both sides (weights in torch state_dict layout,
observations, actions, rollout batch, wrapper stream, env states/actions) and (b) the oracle outputs under
"oracle/<key>", produced by scripts/official/oracle_*.py running
CleanRL @fe8d8a0 / the post's ppo_shared.py @fbef824 / baselines @ea25b9e VecNormalize / gymnasium 0.29.1.
Only numbers are committed, never the authors' code. tests/test_oracle_parity.py recomputes OUR side from (a).
Run after scripts/small/run_difftests.sh: uv run python -m scripts.small.make_parity_fixtures
"""
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "runs", "small", "difftest")
DST = os.path.join(REPO, "tests", "fixtures", "official")
INPUTS = {
    "ff": ("sd/", "fwd/obs", "fwd/act", "gae/rewards", "gae/values", "gae/dones", "gae/next_obs", "gae/next_done",
           "b/", "meta/"),
    "lstm": ("sd/", "seq/obs", "seq/dones", "seq/h0", "seq/c0", "b/"),
    "wrappers": ("stream/",),
    "envs": ("/tf/state", "/tf/action"),
}
CASES = {"separate": "ff", "separate_novclip": "ff", "shared": "ff", "gaussian": "ff", "lstm": "lstm",
         "wrappers": "wrappers", "wrappers_gym021": "wrappers", "envs": "envs", "envs_gym021": "envs"}


def is_input(k, kind):
    for p in INPUTS[kind]:
        if p.startswith("/"):            # suffix pattern (env cases)
            hit = k.endswith(p)
        elif p.endswith("/"):            # prefix pattern
            hit = k.startswith(p)
        else:                            # exact key
            hit = k == p
        if hit:
            return True
    return False


def main():
    os.makedirs(DST, exist_ok=True)
    prov = {"oracle_commits": {"cleanrl": "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb",
                               "ppo-implementation-details": "fbef824effc284137943ff9c058125435ec68cd3",
                               "openai/baselines": "ea25b9e8b234e6ee1bca43083f8f3cf974143998",
                               "gymnasium": "0.29.1", "torch": "2.4.1 (CPU)",
                               "gym (the post's own stack, *_gym021 cases)": "0.21.0"},
            "harness": "scripts/official/oracle_{difftest,wrappers,envs}.py and oracle_{wrappers,envs}_gym021.py",
            "cases": {}}
    for case, kind in CASES.items():
        u = np.load(os.path.join(SRC, f"ours_{case.replace('_gym021', '')}.npz"))
        o = np.load(os.path.join(SRC, f"oracle_{case}.npz"))
        out = {k: u[k] for k in u.files if is_input(k, kind)}
        n_in = len(out)
        for k in o.files:
            if "/ol/" in k:          # open-loop env drift is descriptive only, not a parity check
                continue
            out["oracle/" + k] = o[k]
        np.savez_compressed(os.path.join(DST, f"{case}.npz"), **out)
        prov["cases"][case] = {"n_inputs": n_in, "n_oracle": len(out) - n_in}
        print(f"[INFO] {case}: {n_in} inputs, {len(out) - n_in} oracle arrays")
    # authors'-code check metric case: the same stream, RecordEpisodeStatistics under gymnasium 0.29.1 and gym 0.21.0
    u = np.load(os.path.join(SRC, "ours_metric.npz"))
    out = {k: u[k] for k in u.files if k.startswith("stream/")}
    for tag in ("gymnasium029", "gym021"):
        o = np.load(os.path.join(SRC, f"oracle_metric_{tag}.npz"))
        out.update({"oracle/" + k: o[k] for k in o.files})
    np.savez_compressed(os.path.join(DST, "metric.npz"), **out)
    prov["cases"]["metric"] = {"n_inputs": 6, "n_oracle": len(out) - 6,
                               "harness": "scripts/official/oracle_metric.py under .venv (gymnasium) and .venv_post (gym 0.21)"}
    print(f"[INFO] metric: {len(out) - 6} oracle arrays")
    prov["post_repo_note"] = (
        "fbef824 (2022-07-01) is the authors' repo three months AFTER the post (2022-03-25). The post's only pinned "
        "permalink, eb40cbe, no longer exists (GitHub 422 / not our ref / Wayback 404). The PPO files and the "
        "pre-existing scripts are byte-identical to the oldest surviving commit 339df2e (2022-01-14, before the post), "
        "and the gym 0.21.0 pin is unchanged (see the README).")
    json.dump(prov, open(os.path.join(DST, "provenance.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

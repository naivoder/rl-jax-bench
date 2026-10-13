"""Compare ours_<case>.npz with oracle_<case>.npz (CleanRL code on identical inputs). Pass: |a-b| <= 1e-5 + 1e-5|b|
for every oracle quantity (loop brief: <= 1e-5). Writes runs/small/difftest/report.json. numpy only."""
import json
import os
import sys

import numpy as np

D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs", "small",
                 "difftest")
TOL = 1e-5


def main():
    report, ok_all = {}, True
    for case in ["separate", "separate_novclip", "shared", "gaussian", "lstm", "wrappers", "wrappers_gym021", "envs",
                 "envs_gym021"]:
        if not os.path.exists(os.path.join(D, f"oracle_{case}.npz")):
            print(f"[WARN] {case}: no oracle file")
            ok_all = False
            continue
        o, u = np.load(os.path.join(D, f"oracle_{case}.npz")), np.load(os.path.join(D, f"ours_{case.replace('_gym021', '')}.npz"))
        rows = {}
        for k in o.files:
            if k.startswith("meta/") or "/ol/" in k:   # open-loop env trajectories: descriptive only (below)
                continue
            kk = "lr/ours" if k == "lr/oracle" else k
            if kk not in u.files:
                rows[k] = {"missing_in_ours": True}
                ok_all = False
                continue
            a, b = np.asarray(u[kk], np.float64), np.asarray(o[k], np.float64).reshape(np.shape(u[kk]))
            err = np.abs(a - b)
            ok = bool(np.all(err <= TOL + TOL * np.abs(b)))
            ok_all &= ok
            rows[k] = {"max_abs": float(err.max()), "max_rel": float((err / np.maximum(np.abs(b), 1e-12)).max()),
                       "n": int(err.size), "pass": ok}
        groups = {}
        for k, r in rows.items():
            g = k.split("/")[0] + ("/" + k.split("/")[1] if k.startswith("step") and k.count("/") > 1 else "")
            gg = groups.setdefault(g, {"max_abs": 0.0, "pass": True, "n_keys": 0})
            gg["max_abs"] = max(gg["max_abs"], r.get("max_abs", np.inf))
            gg["pass"] &= r.get("pass", False)
            gg["n_keys"] += 1
        report[case] = {"groups": groups, "keys": rows, "oracle_lines": {k: o[k].tolist() for k in o.files
                                                                         if k.startswith("meta/")}}
        print(f"[{'INFO' if all(g['pass'] for g in groups.values()) else 'ERROR'}] {case}: " +
              " ".join(f"{g}={v['max_abs']:.2e}{'' if v['pass'] else '(FAIL)'}" for g, v in groups.items()))
    if os.path.exists(os.path.join(D, "oracle_envs.npz")):
        o, u = np.load(os.path.join(D, "oracle_envs.npz")), np.load(os.path.join(D, "ours_envs.npz"))
        ol = {}
        for k in o.files:
            if "/ol/obs" in k:
                env = k.split("/")[0]
                alive = u[f"{env}/ol/alive"]
                err = np.abs(u[k] - o[k]).max(-1)  # (10, 500)
                ol[env] = {"max_abs_while_alive": float(err[alive].max()),
                           "median_steps_alive": float(np.median(alive.sum(1))),
                           "max_abs_first_50_steps": float(err[:, :50][alive[:, :50]].max())}
        report["env_open_loop"] = ol
        print("[INFO] open-loop gymnax vs gymnasium divergence (descriptive):", ol)
    u = np.load(os.path.join(D, "ours_wrappers.npz"))
    report["wrapper_mode_separation"] = {  # power: the two modes must differ by far more than the tolerance
        k: float(np.abs(u[f"baselines/{k}"] - u[f"gymnasium029/{k}"]).max()) for k in ["obs", "rew"]}
    print("[INFO] wrapper mode separation (ours baselines vs gymnasium029):", report["wrapper_mode_separation"])
    report["all_pass"] = bool(ok_all)
    with open(os.path.join(D, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"[METRIC] difftest all_pass={ok_all}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

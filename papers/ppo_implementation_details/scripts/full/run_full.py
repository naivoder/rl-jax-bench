"""full-scale launcher: one GPU subprocess per (group, env, arm), seeds 1-10 vmapped,
arms in the declared priority order. Hard budget line per group: an arm is not started if
(elapsed group wall + longest arm so far in the group) > line. Skipped arms are logged and reported.
Usage: python -m scripts.full.run_full [group ...]   (groups: E1 E3 E4; default all)"""
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.environ.get("PY", sys.executable)
SEEDS = "seeds=[1,2,3,4,5,6,7,8,9,10]"
GROUPS = {
    "E1": dict(line_s=540, cfg="classic_control.yaml", tag="full", jobs=[
        (env, arm) for env in ("CartPole-v1", "Acrobot-v1")
        for arm in ("separate", "shared", "advnorm_batch", "no_vclip", "trunc_bootstrap", "advnorm_ddof0")]
        + [("MountainCar-v0", "separate"), ("MountainCar-v0", "shared")]),
    "E3": dict(line_s=900, cfg="pendulum.yaml", tag="full", jobs=[
        ("Pendulum-v1", a) for a in ("default", "gym021", "gymnasium029", "trunc_bootstrap", "no_rscale", "no_vclip",
                                     "advnorm_batch")]),
    "E4": dict(line_s=540, cfg="lstm_masked_cartpole.yaml", tag="full_lstm", jobs=[
        ("CartPole-v1", a) for a in ("lstm", "gymnasium029", "lstm_noreset", "ff_masked", "ff_full", "lstm_zero_train",
                                     "no_rscale")]),
}


def main(groups):
    log_dir = os.path.join(REPO, "runs", "full", "logs")
    os.makedirs(log_dir, exist_ok=True)
    ledger_p = os.path.join(REPO, "runs", "full", "budget_ledger.json")
    ledger = json.load(open(ledger_p)) if os.path.exists(ledger_p) else {}
    for g in groups:
        G = GROUPS[g]
        t0, longest, rows = time.time(), 0.0, []
        for env, arm in G["jobs"]:
            el = time.time() - t0
            if el + longest > G["line_s"]:
                print(f"[WARN] SKIPPED (budget line) group={g} env={env} arm={arm} elapsed={el:.0f}s "
                      f"longest={longest:.0f}s line={G['line_s']}s", flush=True)
                rows.append(dict(env=env, arm=arm, status="skipped_budget"))
                continue
            cmd = [PY, "-m", "src.train", "--config", f"src/configs/{G['cfg']}", "--tag", G["tag"], "--envs", env,
                   "--arms", arm, "--set", SEEDS]
            ts = time.time()
            with open(os.path.join(log_dir, f"{g}_{env}_{arm}.log"), "w") as f:
                rc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
            dt = time.time() - ts
            longest = max(longest, dt)
            rows.append(dict(env=env, arm=arm, status="ok" if rc == 0 else f"rc={rc}", wall_s=dt))
            print(f"[INFO] group={g} env={env} arm={arm} rc={rc} wall={dt:.0f}s group_elapsed={time.time() - t0:.0f}s",
                  flush=True)
        tot = time.time() - t0
        ledger[g] = dict(line_s=G["line_s"], wall_s=tot, gpu_h=tot / 3600, rows=rows)
        json.dump(ledger, open(ledger_p, "w"), indent=1)
        print(f"[METRIC] group={g} wall_s={tot:.0f} gpu_h={tot / 3600:.4f} line_gpu_h={G['line_s'] / 3600:.3f}",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["E1", "E3", "E4"])

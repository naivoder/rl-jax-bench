"""Build tests/fixtures/tonic/*.npz: inputs + the outputs of Tonic @0e20c89's OWN code on them.

Runs the oracle in its isolated venv ($OFFICIAL_ROOT/.venv, numpy 1.23.5, gym 0.23.1; default external/official)
through scripts/official/run_tonic.py. Only NUMBERS are committed here, never
Tonic code. Usage: uv run python scripts/toys/make_tonic_fixtures.py
"""
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFF = os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official"))  # authors' clones + oracle venvs
HARNESS = os.path.join(REPO, "scripts", "official")
FIX = os.path.join(REPO, "tests", "fixtures", "tonic")


def run_oracle(inp: dict, name: str):
    os.makedirs(FIX, exist_ok=True)
    ip = os.path.join(FIX, f"{name}_in.npz")
    op = os.path.join(FIX, f"{name}_oracle.npz")
    np.savez(ip, **inp)
    subprocess.run([os.path.join(OFF, ".venv", "bin", "python"), os.path.join(HARNESS, "run_tonic.py"), ip, op],
                   check=True)
    return dict(np.load(op))


def main():
    rng = np.random.default_rng(20261014)
    cases = {}
    for name, (M, N, p_reset, p_term_given_reset) in {"lr_mixed": (64, 4, 0.15, 0.5),
                                                      "lr_dense": (40, 3, 0.5, 0.3)}.items():
        resets = (rng.random((M, N)) < p_reset).astype(np.float32)
        terms = resets * (rng.random((M, N)) < p_term_given_reset).astype(np.float32)
        values = rng.normal(size=(M, N)).astype(np.float32)
        # rollout contract: next_values[t] = V(s_{t+1}) = values[t+1] unless transition t ended an episode (then it is
        # V of the pre-reset obs, unrelated to values[t+1]); the last row closes the rollout.
        next_values = rng.normal(size=(M, N)).astype(np.float32)
        next_values[:-1] = np.where(resets[:-1] > 0, next_values[:-1], values[1:])
        inp = dict(lr_values=values, lr_next_values=next_values,
                   lr_rewards=rng.normal(size=(M, N)).astype(np.float32), lr_resets=resets, lr_terminations=terms,
                   lr_gamma=np.float32(0.99), lr_lam=np.float32(0.95))
        run_oracle(inp, name)
        cases[name] = "lambda_returns"
    # scripted runner: timeouts at T = 5 and terminations whenever action == 1
    acts = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                    np.int64)
    run_oracle(dict(rn_actions=acts, rn_T=np.int64(5)), "runner_T5")
    cases["runner_T5"] = "Sequential+TimeFeature"
    commit = subprocess.check_output(["git", "-C", os.path.join(OFF, "tonic"), "rev-parse", "HEAD"]).decode().strip()
    with open(os.path.join(FIX, "provenance.json"), "w") as f:
        json.dump(dict(oracle="Tonic", commit=commit, harness=os.path.join(HARNESS, "run_tonic.py"),
                       venv=os.path.join(OFF, ".venv"), cases=cases), f, indent=1)
    print("fixtures written", cases, commit)


if __name__ == "__main__":
    sys.exit(main())

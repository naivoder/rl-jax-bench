"""Record teacher-forced CBP + AdamGnT fixtures from the PAPER-ERA official optimiser (recheck).

    JAX_PLATFORMS=cpu uv run python -m scripts.oracle.record_fixtures_prefix

Runs the unmodified official tree exported at 63c35f3 (`git archive`, the last commit before f00ca40 rewrote
lop/utils/AdamGnT.py; its AdamGnT is byte-identical to dcc3619, the version that existed when the paper's figures were
made) through run_oracle.py with LOP_ORACLE_REPO.  Writes tests/fixtures/official_pre_f00ca40/<name>.npz and a
MANIFEST.json (tree, sha256 of AdamGnT.py, torch version, sha256 of each fixture).  Only inputs and outputs are stored.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import numpy as np

from scripts.oracle.diff import ORACLE_PY, RUNNER

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FX = os.path.join(REPO, "tests", "fixtures", "official_pre_f00ca40")
TMP = os.path.join(REPO, "runs", "oracle_diff", "fixtures_tmp_prefix")
TREE = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official")), "repo_pre_f00ca40")


def call(case, inp, name):
    os.makedirs(TMP, exist_ok=True)
    ip, op = os.path.join(TMP, f"{name}_in.npz"), os.path.join(TMP, f"{name}_out.npz")
    np.savez(ip, **inp)
    subprocess.run([ORACLE_PY, RUNNER, case, ip, op], check=True, capture_output=True,
                   env={**os.environ, "LOP_ORACLE_REPO": TREE})
    out = dict(np.load(op))
    path = os.path.join(FX, f"{name}.npz")
    np.savez_compressed(path, **{f"in__{k}": np.asarray(v) for k, v in inp.items()},
                        **{f"out__{k}": v for k, v in out.items()})
    print(f"[INFO] fixture {name}: {os.path.getsize(path) / 1024:.0f} KiB", flush=True)
    return path


def main():
    os.makedirs(FX, exist_ok=True)
    paths = []
    # teacher-forced CBP (GnT + paper-era AdamGnT), FFNN 20->5->1 tanh: 60 steps, m = 3, rho = 0.05 -> several
    # replacements, each followed by steps on freshly reset (t small) Adam state
    for ut in ("adaptable_contribution", "adaptation", "contribution"):
        cfg = {"kind": "ffnn", "n_in": 20, "width": 5, "act": "tanh", "opt": "adam", "lr": 0.01, "rho": 0.05,
               "eta": 0.9, "mt": 3, "util_type": ut, "seed": 23, "n_steps": 60}
        paths.append(call("gnt_traj", cfg, f"gnt_ffnn_{ut}_prefix"))
    adam = os.path.join(TREE, "lop", "utils", "AdamGnT.py")
    torch_v = subprocess.run([ORACLE_PY, "-c", "import torch; print(torch.__version__)"], capture_output=True,
                             text=True).stdout.strip()
    man = {"oracle_repo": "https://github.com/shibhansh/loss-of-plasticity",
           "oracle_tree": "git archive 63c35f3 (last commit before f00ca40); AdamGnT.py == dcc3619",
           "adamgnt_sha256": hashlib.sha256(open(adam, "rb").read()).hexdigest(), "oracle_torch": torch_v,
           "recorder": "scripts/oracle/record_fixtures_prefix.py",
           "files": {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest() for p in paths}}
    json.dump(man, open(os.path.join(FX, "MANIFEST.json"), "w"), indent=1)
    print("[INFO] fixtures written:", len(paths), flush=True)


if __name__ == "__main__":
    main()

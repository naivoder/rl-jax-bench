"""Pack the FF multi-update oracle outputs (runs/recheck/official-oracle/ff_multi/oracle_*.npz, produced by
oracle_ff_multiupdate.py in the official venv from ours_*.npz) plus the teacher-forced input stream that the oracle
replayed (in/...) into tests/fixtures/official/ff_multi_<case>.npz."""
import os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(REPO, "runs", "recheck", "official-oracle", "ff_multi")
for case in ("separate_cartpole", "shared_cartpole", "separate_acrobot_novclip", "gauss_pendulum"):
    a, b = np.load(os.path.join(D, f"ours_{case}.npz")), np.load(os.path.join(D, f"oracle_{case}.npz"))
    out = {"in/" + k: a[k] for k in a.files if k.split("/")[-1] in ("obs", "action", "done", "reward", "perms")
           or k.startswith("sd/")}
    out.update({"oracle/" + k: b[k] for k in b.files if not k.startswith("meta/")})
    np.savez_compressed(os.path.join(REPO, "tests", "fixtures", "official", f"ff_multi_{case}.npz"), **out)
    print(case, len(out))

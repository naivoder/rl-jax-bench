"""Cut the official replay dumps (scripts/official/official_dump.py output, runs/replay/dump_<cfg>.npz) to their
first two iterations (REC) / first iteration (REC + value-norm None) as the committed regression fixture for
tests/test_replay_parity.py."""
import os
import numpy as np

PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(PAPER, "runs", "replay")
OUT = os.path.join(PAPER, "tests", "fixtures")
for name, K in (("rec_pre", 2), ("rec_pre_vnone", 1)):
    z = np.load(f"{SRC}/dump_{name}.npz")
    keep = {k: z[k] for k in z.files if not k.startswith("it") or int(k.split("/")[0][2:]) < K}
    np.savez_compressed(f"{OUT}/replay_slim_{name}.npz", **keep)
    print(name, len(keep), "arrays", os.path.getsize(f"{OUT}/replay_slim_{name}.npz") // 1024, "KiB")

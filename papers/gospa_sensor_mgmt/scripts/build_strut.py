"""Build data/truths_strut.npz for the S-TRUTH sensitivity row from the authors' generator output (scripts/official/gen_truths.py, seeds 1..60).
Selection rule (fixed before any result): the first 10 seeds >= 1 whose truth has <= 9 targets (fixed array shape N_TRUTH) and <= 5 simultaneously alive (K_TRUTH).
usage: build_strut.py <raw.npz> <out.npz>"""
import hashlib, sys
import numpy as np
raw = np.load(sys.argv[1])
out, used = {}, []
for seed in range(1, 61):
    T = raw[f"seed{seed}"]
    alive = ~np.isnan(T[:, 0, :])
    if T.shape[0] <= 9 and alive.sum(0).max() <= 5:
        pad = np.full((9, 4, 200), np.nan)
        pad[: T.shape[0]] = T
        out[f"truth{len(used)}"] = pad
        used.append(seed)
        if len(used) == 10:
            break
np.savez(sys.argv[2], **out)
print("seeds used", used, "sha256", hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest())

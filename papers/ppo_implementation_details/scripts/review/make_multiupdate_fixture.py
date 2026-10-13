"""Pack the authors'-code side of the multi-update LSTM replay into a committed test fixture:
tests/fixtures/official/lstm_multi.npz = the teacher-forced inputs (to check our rerun reproduces the same stream) and
the oracle outputs for L3 on (every update) and L3 off (last update, for the power check).
Run: uv run python -m scripts.review.make_multiupdate_fixture"""
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(REPO, "runs", "review_lstm_multi", "difftest")
ours = np.load(os.path.join(D, "ours_lstm_multi_reset.npz"))
on = np.load(os.path.join(D, "oracle_lstm_multi_reset.npz"))
off = np.load(os.path.join(D, "oracle_lstm_multi_noreset.npz"))
out = {}
for u in range(int(ours["meta/U"])):
    p = f"u{u}/"
    for k in ("obs", "done", "action", "reward", "next_obs", "next_done", "perms"):
        out["in/" + p + k] = ours[p + k]
    for k in ("logp", "value", "init_h", "init_c", "final_h", "final_c", "adv", "ret", "lr"):
        out["oracle/" + p + k] = on[p + k]
last = f"u{int(ours['meta/U']) - 1}/"
out.update({"oracle/" + k: on[k] for k in on.files if k.startswith(last + "params_after/")})
out.update({"oracle_noreset/" + last + k: off[last + k] for k in ("logp", "value", "final_h")})
out["meta/oracle_file"] = on["meta/file"]
out["meta/gae_lines"], out["meta/update_lines"] = on["meta/gae_lines"], on["meta/update_lines"]
np.savez_compressed(os.path.join(REPO, "tests", "fixtures", "official", "lstm_multi.npz"), **out)
print("keys", len(out))

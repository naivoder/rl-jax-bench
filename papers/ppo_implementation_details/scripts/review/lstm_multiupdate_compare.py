"""Compare ours vs the authors' code on the multi-update LSTM replay. Writes a json summary.
Run: uv run python -m scripts.review.lstm_multiupdate_compare"""
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(REPO, "runs", "review_lstm_multi")
res = {}
for tag in ("reset", "noreset"):
    a = np.load(os.path.join(D, "difftest", f"ours_lstm_multi_{tag}.npz"))
    b = np.load(os.path.join(D, "difftest", f"oracle_lstm_multi_{tag}.npz"))
    U = int(a["meta/U"])
    rows = {}
    for u in range(U):
        p = f"u{u}/"
        for k in ("logp", "value", "init_h", "init_c", "final_h", "final_c", "adv", "ret", "lr"):
            rows[p + k] = float(np.abs(np.asarray(a[p + k], np.float64) - np.asarray(b[p + k], np.float64)).max())
        pk = [k for k in b.files if k.startswith(p + "params_after/")]
        assert len(pk) == 10 and all(k in a.files for k in pk), pk
        rows[p + "params_after"] = max(float(np.abs(a[k].astype(np.float64) - b[k]).max()) for k in pk)
        before = "sd/" if u == 0 else f"u{u - 1}/params_after/"
        rows[p + "param_change_size"] = max(float(np.abs(a[k].astype(np.float64) - a[before + k.split("/", 2)[2]]).max())
                                            for k in pk)
    n_start = int(sum(np.asarray(a[f"u{u}/done"]).sum() for u in range(U)))
    carry_norm = float(np.abs(a[f"u{U - 1}/init_h"]).max())
    res[tag] = dict(max_abs=rows, worst=max(v for k, v in rows.items() if not k.endswith("param_change_size")),
                    episode_starts_in_rollouts=n_start, max_abs_carry_into_last_update=carry_norm)
    print(f"[METRIC] {tag}: worst |ours-oracle| = {res[tag]['worst']:.2e} over {U} updates; episode starts "
          f"{n_start}; |carry into update {U - 1}| max {carry_norm:.3f}")
    for k, v in rows.items():
        print(f"   {k:28s} {v:.2e}")
json.dump(res, open(os.path.join(D, "multiupdate_report.json"), "w"), indent=1)

"""Summary statistics of measurement-set samples (numpy only) shared by scripts/record_official_fixtures.py (run on the authors' generator) and
tests/test_official_parity.py (run on src/data.py): value and standard error of each statistic, so the two generators can be compared by z-scores."""
import numpy as np


def stats(sets, origin, hist_max=6):
    """sets: list of [m, 2] arrays (one per trial); origin: [2] the point the offsets are measured from."""
    n = len(sets)
    cnt = np.array([len(s) for s in sets], float)
    out = {"count_mean": (cnt.mean(), cnt.std(ddof=1) / np.sqrt(n))}
    for b in range(hist_max + 1):
        p = float((cnt == b).mean()) if b < hist_max else float((cnt >= b).mean())
        out[f"count_hist{b}"] = (p, np.sqrt(max(p * (1 - p), 1e-12) / n))
    pts = np.concatenate([s for s in sets if len(s)], axis=0) - np.asarray(origin)[None, :] if cnt.sum() else np.zeros((0, 2))
    k = len(pts)
    if k > 10:
        r2 = (pts ** 2).sum(1)
        out["mean_dx"] = (pts[:, 0].mean(), pts[:, 0].std(ddof=1) / np.sqrt(k))
        out["mean_dy"] = (pts[:, 1].mean(), pts[:, 1].std(ddof=1) / np.sqrt(k))
        out["mean_r2"] = (r2.mean(), r2.std(ddof=1) / np.sqrt(k))
        for j, nm in enumerate(("var_dx", "var_dy")):
            v = pts[:, j].var(ddof=1)
            out[nm] = (v, v * np.sqrt(2.0 / (k - 1)))
    return out

"""Summarise shkrwnd reference runs (runs/official/shk_N*K*_s*/trace.npz) with OUR metric definitions (src/eval.py): utilization = successes / K,
collision = fraction of channels with >= 2 transmitters, idle = fraction with 0, strict Pareto = every slot of the last W=50 has successes == K,
pattern E7/E6 as in eval.patterns.  Their traces are the ONLINE TRAINING STREAM (epsilon 0.02 -> 0.01 joint random slots included), not a frozen-network test.
Usage (repo environment): uv run python scripts/official/analyze_shk.py [--json out.json]"""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
W = 50   # eval.PARETO_WINDOW


def one(path):
    d = np.load(path); a = d["actions"].astype(int); r = d["rewards"].astype(int)
    T, N = a.shape
    K = int(re.search(r"K(\d+)_", path).group(1))
    succ = r.sum(1)
    counts = np.stack([(a == k).sum(1) for k in range(1, K + 1)], 1)
    util = succ / K; coll = (counts >= 2).mean(1); idle = (counts == 0).mean(1)
    out = dict(N=N, K=K, T=T,
               util_blocks=[float(util[i:i + 10000].mean()) for i in range(0, T, 10000)],
               util_last5k=float(util[-5000:].mean()), coll_last5k=float(coll[-5000:].mean()), idle_last5k=float(idle[-5000:].mean()),
               util_first5k=float(util[:5000].mean()))
    # strict Pareto: fraction of W-slot windows (non-overlapping, last 5000 slots) with succ == K in every slot
    ws = succ[-5000:].reshape(-1, W)
    out["pareto_strict_windows"] = float((ws == K).all(1).mean())
    sf = r[-5000:].reshape(-1, W, N).mean(1); txf = (a[-5000:] > 0).reshape(-1, W, N).mean(1)
    srt = np.sort(sf, 1)[:, ::-1]
    strict = (ws == K).all(1)
    if K == 2 and N == 3:
        ok = (srt[:, 0] >= .99) & (srt[:, 1] >= .4) & (srt[:, 1] <= .6) & (srt[:, 2] >= .4) & (srt[:, 2] <= .6)
        out["pattern_E7_windows"] = float((strict & ok).mean())
    if K == 2 and N == 4:
        always = (txf >= .99).sum(1); silent = (txf <= .01).sum(1)
        out["pattern_E6_windows"] = float((strict & (always == K) & (silent == N - K)).mean())
    return out


if __name__ == "__main__":
    res = {}
    for p in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "..", "runs", "official", "shk_*", "trace.npz"))):
        res[os.path.basename(os.path.dirname(p))] = one(p)
    for k, v in res.items():
        print(k, json.dumps({kk: (np.round(vv, 3).tolist() if isinstance(vv, list) else (round(vv, 3) if isinstance(vv, float) else vv)) for kk, vv in v.items()}))
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)

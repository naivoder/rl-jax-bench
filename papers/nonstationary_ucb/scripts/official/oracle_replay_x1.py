"""SMPyBandits (aligned conventions, lowest ties) replay of our long-horizon X1 trajectories. Throwaway probe."""
import sys, math, time
sys.argv_saved = list(sys.argv)
import os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oracle_replay.py')).read().split("d = np.load(sys.argv[1])")[0]
exec(compile(src, "oracle_replay_header", "exec"))
d = np.load(sys.argv_saved[1]); means = d["means"].astype(np.float64)
T, K = means.shape
for c in ("ducb", "swucb"):
    kw = eval(str(d[f"{c}_kw"]))
    for r in range(d[f"{c}_X"].shape[0]):
        t0 = time.time()
        for mode in ("aligned", "asis"):
            acts, _, reg = run(make(c, kw, K, T, mode), c, d[f"{c}_X"][r], None, means, K, T)
            oa = d[f"{c}_acts"][r]
            mism = np.nonzero(acts != oa)[0]
            print(c, kw, "rep", r, mode, "identical" if mism.size == 0 else f"first mismatch round {mism[0]+1}",
                  "regret oracle %.2f ours %.2f" % (reg, d[f'{c}_regret'][r]), "%.0fs" % (time.time() - t0), flush=True)

"""Drive SMPyBandits' own PieceWiseStationaryMAB (newRandomArms(t) per step, as Evaluator does) and diff its
mean schedule with OUR data.abrupt E1 schedule (changePoints given 0-indexed = our round - 1). Throwaway probe."""
import sys, numpy as np
import os
OFFICIAL_ROOT = os.environ.get('OFFICIAL_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'external', 'official'))
sys.path.insert(0, os.path.join(OFFICIAL_ROOT, 'smpybandits', 'SMPyBandits'))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from Environment.MAB import PieceWiseStationaryMAB
    from Arms import Bernoulli
    env = PieceWiseStationaryMAB({"arm_type": Bernoulli, "params": {"listOfMeans": [[.5, .3, .4], [.5, .9, .4], [.5, .3, .4]], "changePoints": [0, 2999, 4999]}})
    T = 10_000; sched = np.zeros((T, 3))
    sched[0] = env.newRandomArms(0)
    for t in range(1, T):
        sched[t] = env.newRandomArms(t)
t1 = np.arange(1, T + 1)
segs = np.array([[.5, .3, .4], [.5, .9, .4], [.5, .3, .4]])
ours = segs[np.searchsorted(np.array([3000, 5000]), t1, side="right")]
print("SMPy PieceWiseStationaryMAB schedule == ours (E1 figure means):", np.array_equal(sched, ours), "rows differing:", int((sched != ours).any(1).sum()))

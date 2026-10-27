"""Stone Soup 1.9.1 reference values (run with the radar_search_track_seq venv python):
python ss_ref.py in.npz out.npy ; in.npz: E, T (object arrays of [m,2] / [n,2] position sets), c.
out[i, ip, :] = (OSPA, GOSPA distance, loc, missed, false) for p in (1, 2); NaN rows for empty/empty."""
import datetime
import sys

import numpy as np
from stonesoup.metricgenerator.ospametric import GOSPAMetric, OSPAMetric
from stonesoup.types.array import StateVector
from stonesoup.types.state import State

inp, outp = sys.argv[1], sys.argv[2]
d = np.load(inp, allow_pickle=True)
E, T, c = d["E"], d["T"], float(d["c"])
t0 = datetime.datetime(2020, 1, 1)


def mk(a):
    return [State(StateVector(r), timestamp=t0) for r in a]


res = np.full((len(E), 2, 5), np.nan)
for ip, p in enumerate((1.0, 2.0)):
    om = OSPAMetric(c=c, p=p)
    gm = GOSPAMetric(c=c, p=p)
    for i, (e, t) in enumerate(zip(E, T)):
        if len(e) == 0 and len(t) == 0:
            continue
        res[i, ip, 0] = om.compute_OSPA_distance(mk(e), mk(t)).value
        m, _ = gm.compute_gospa_metric(mk(e), mk(t))
        v = m.value
        res[i, ip, 1:] = [v["distance"], v["localisation"], v["missed"], v["false"]]
np.save(outp, res)

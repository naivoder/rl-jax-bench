"""ORACLE (official venv): the official bookkeeping of main_a2c.py on a synthetic sequence of batches -- the avg-100
window, the log Dispenser (fired on the PRE-batch step count), the target Dispenser and the linear negentropy
schedule (both on the POST-batch count). tests/test_schedules.py replays the same batches through our `learn`."""
import os
import sys

PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(PAPER, "external", "official"))
sys.path.insert(0, os.path.join(OFFICIAL_ROOT, "asym-rlpo-aamas2022"))
import numpy as np

from asym_rlpo.utils.running_average import WindowRunningAverage
from asym_rlpo.utils.scheduling import linear_schedule
from asym_rlpo.utils.timer import Dispenser

N_UPD, T_MAX, LAM0 = 400, 30_000, 0.3
rng = np.random.default_rng(3)
lens = rng.integers(1, 101, (N_UPD, 2))
rets = rng.normal(0, 5, (N_UPD, 2)).round(3)
avg = WindowRunningAverage(100)
logd, tgtd = Dispenser(T_MAX // 200), Dispenser(10_000)
t = 0
log_t, log_avg, lam, next_tgt, next_log = [], [], [], [], []
for u in range(N_UPD):
    if t >= T_MAX:
        break
    avg.extend(rets[u].tolist())
    if logd.dispense(t):
        log_t.append(t)
        log_avg.append(avg.value())
    t += int(lens[u].sum())
    lam.append(linear_schedule(t, value_from=LAM0, value_to=LAM0 * 0.1, nsteps=20_000))
    tgtd.dispense(t)
    next_tgt.append(tgtd.next_i)
    next_log.append(logd.next_i)
np.savez(os.path.join(OFFICIAL_ROOT, "out", "schedules.npz"), lens=lens, rets=rets,
         log_t=log_t, log_avg=log_avg, lam=lam, next_tgt=next_tgt, next_log=next_log, T_MAX=T_MAX, LAM0=LAM0)
print("updates", len(lam), "logs", len(log_t))

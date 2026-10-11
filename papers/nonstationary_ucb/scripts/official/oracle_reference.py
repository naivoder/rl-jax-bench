"""REFERENCE RUN with the SMPyBandits policy code (third-party oracle, commit 012fc13a), UNPATCHED
(random ties via np.random.choice, SMPy's own initialisation), on the paper's E1 (figure means) and E2.
Rewards: Bernoulli 1{U < p_t(i)} from numpy's global RNG (SMPy's Arms.Bernoulli draws the same way),
time-indexed; pseudo-regret sum_t max_i p_t(i) - p_t(I_t). Changepoints are SMPy-0-indexed steps 2999,4999
(= our 1-indexed rounds 3000, 5000). One worker = one (env, seed); reps independent.
Usage: python oracle_reference.py ENV SEED NREPS OUT.json
"""
import sys, json, math, time
import os
OFFICIAL_ROOT = os.environ.get('OFFICIAL_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'external', 'official'))
sys.path.insert(0, os.path.join(OFFICIAL_ROOT, 'smpybandits', 'SMPyBandits', 'Policies'))
import numpy as np
from UCBalpha import UCBalpha
from DiscountedUCB import DiscountedUCB, DiscountedUCBPlus
from SlidingWindowUCB import SWUCB, SWUCBPlus
from Exp3S import Exp3S

ename, seed, R, outp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
T = 10_000
t1 = np.arange(1, T + 1)
if ename == "e1":
    segs = np.array([[.5, .3, .4], [.5, .9, .4], [.5, .3, .4]])
    means = segs[np.searchsorted(np.array([3000, 5000]), t1, side="right")]
else:
    means = np.stack([0.5 + 0.4 * np.cos(2 * math.pi * 3 * t1 / T), np.full(T, 0.5)], 1)
K = means.shape[1]
gamma_paper = 1 - 1 / (4 * math.sqrt(T))
tau_paper = int(round(4 * math.sqrt(T * math.log(T))))
g_paper = min(1.0, math.sqrt(K * (2 * math.log(K * T) + math.e) / ((math.e - 1) * T)))
import io, contextlib
def quiet(f):
    with contextlib.redirect_stdout(io.StringIO()):
        return f()
POL = {
    # paper settings expressed in SMPy's parametrisation
    "ucb1_paper(xi=1/2)":        lambda: UCBalpha(K, alpha=1),
    "ducb_printed2B(alpha=4)":   lambda: DiscountedUCB(K, alpha=4, gamma=gamma_paper),
    "ducb_smpy_default_pad(alpha=1)": lambda: DiscountedUCB(K, alpha=1, gamma=gamma_paper),
    "swucb_paper(alpha=1/2)":    lambda: SWUCB(K, tau=tau_paper, alpha=0.5),
    "exp3s_paper(S=2)":          lambda: Exp3S(K, gamma=g_paper, alpha=1 / T, gamma0=1.0, alpha0=1.0),
    # SMPy's own defaults / heuristics (horizon-aware constructors)
    "ucb1_smpy_default(alpha=4)": lambda: UCBalpha(K),
    "ducbPlus_smpy_default(Ups=2)": lambda: DiscountedUCBPlus(K, horizon=T, max_nb_random_events=2),
    "swucbPlus_smpy_default(alpha=1)": lambda: SWUCBPlus(K, horizon=T),
    "exp3s_smpy_default(Ups=2)":  lambda: Exp3S(K, horizon=T, max_nb_random_events=2),
}
np.random.seed(10_000 * seed + (1 if ename == "e1" else 2))
res = {k: {"regret": [], "arm1": [], "r3000": [], "r5000": []} for k in POL}
info = {}
t0 = time.time()
best = means.max(1)
for r in range(R):
    U = np.random.random((T, K))
    X = (U < means).astype(float)
    for name, mk in POL.items():
        pol = quiet(mk)
        if r == 0:
            info[name] = {k: float(getattr(pol, k)) for k in ("alpha", "gamma", "tau") if hasattr(pol, k) and np.isscalar(getattr(pol, k))}
        pol.startGame()
        acts = np.empty(T, np.int64)
        for t in range(T):
            a = pol.choice()
            acts[t] = a
            pol.getReward(a, X[t, a])
        inc = best - means[np.arange(T), acts]
        c = np.cumsum(inc)
        res[name]["regret"].append(float(c[-1])); res[name]["r3000"].append(float(c[2999])); res[name]["r5000"].append(float(c[4999]))
        res[name]["arm1"].append(int((acts == 0).sum()))
json.dump({"env": ename, "seed": seed, "R": R, "wall_s": time.time() - t0, "params": info, "res": res,
           "smpy_commit": "012fc13a9d82e361930fdf701dfff9a91439d60c"}, open(outp, "w"))
print(ename, seed, "done", time.time() - t0)

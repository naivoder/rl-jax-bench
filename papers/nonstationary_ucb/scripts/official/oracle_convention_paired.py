"""(copied harness header from oracle_replay.py)
Replay OUR reward tables / EXP3.S uniforms through the SMPyBandits policy code (third-party oracle,
commit 012fc13a) and diff actions, indices/probabilities and regret step by step. Throwaway probe.
Harness changes to the oracle (labelled): (1) ties -> lowest index (IndexPolicy.choice patched) so that
action sequences are comparable with our tie='lowest' trajectories; (2) EXP3.S draw uses OUR uniform
u_t by inverse CDF (count(cdf < u*cdf[-1])) and skips SMPy's random-permutation first-K exploration.
Mode 'asis' keeps every SMPy formula; mode 'aligned' patches only the log-argument convention
(log(t) with t = current round) and D-UCB's '+inf if N<1' -> '+inf if N==0', to isolate those two.
Usage: oracle venv python oracle_replay.py ours_traj.npz out.json
"""
import sys, json, math
import os
OFFICIAL_ROOT = os.environ.get('OFFICIAL_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'external', 'official'))
sys.path.insert(0, os.path.join(OFFICIAL_ROOT, 'smpybandits', 'SMPyBandits', 'Policies'))
import numpy as np
from IndexPolicy import IndexPolicy
from UCBalpha import UCBalpha
from DiscountedUCB import DiscountedUCB
from SlidingWindowUCB import SWUCB
from Exp3S import Exp3S

def lowest_choice(self):
    self.computeAllIndex()
    return int(np.nonzero(self.index == np.max(self.index))[0][0])
IndexPolicy.choice = lowest_choice

class UCBaligned(UCBalpha):
    def computeAllIndex(self):
        t = self.t + 1
        idx = self.rewards / self.pulls + np.sqrt(self.alpha * np.log(t) / (2 * self.pulls))
        idx[self.pulls < 1] = np.inf
        self.index[:] = idx

class SWaligned(SWUCB):
    def computeIndex(self, arm):
        n = np.count_nonzero(self.last_choices == arm)
        if n < 1:
            return float('inf')
        return np.sum(self.last_rewards[self.last_choices == arm]) / n + math.sqrt(self.alpha * math.log(min(self.t + 1, self.tau)) / n)

class DUCBaligned(DiscountedUCB):
    def computeAllIndex(self):
        n = np.sum(self.discounted_pulls)
        idx = self.discounted_rewards / self.discounted_pulls + np.sqrt(self.alpha * np.log(n) / (2 * self.discounted_pulls))
        idx[self.discounted_pulls == 0] = np.inf
        self.index[:] = idx

def make(case, kw, K, T, mode):
    if case == "ucb1":
        return (UCBaligned if mode == "aligned" else UCBalpha)(K, alpha=2 * kw["xi"])
    if case in ("ducb", "ducb_padB"):
        # SMPy pad = sqrt(alpha log n / (2N)); ours = 2B sqrt(xi log n / N)  =>  alpha = 8 xi (B = 1)
        return (DUCBaligned if mode == "aligned" else DiscountedUCB)(K, alpha=8 * kw["xi"], gamma=kw["gamma"])
    if case == "swucb":
        # SMPy pad = sqrt(alpha log min(t,tau) / N); ours = B sqrt(xi log min(t,tau) / N) => alpha = xi
        return (SWaligned if mode == "aligned" else SWUCB)(K, tau=kw["tau"], alpha=kw["xi"])
    if case == "exp3s":
        return Exp3S(K, gamma=kw["g"], alpha=kw["alpha"], gamma0=1.0, alpha0=1.0)

def run(pol, case, X, Up, means, K, T):
    pol.startGame()
    acts = np.empty(T, np.int64); aux = np.full((T, K), np.nan)
    for t in range(T):
        if case == "exp3s":
            p = pol.trusts
            c = np.cumsum(p)
            a = min(int(np.sum(c < Up[t] * c[-1])), K - 1)
            aux[t] = p
        else:
            a = pol.choice()
            aux[t] = pol.index
        acts[t] = a
        pol.getReward(a, float(X[t, a]))
    reg = float((means.max(1) - means[np.arange(T), acts]).sum())
    return acts, aux, reg


ename, R, seed, outp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
T = 10_000
t1 = np.arange(1, T + 1)
if ename == "e1":
    segs = np.array([[.5, .3, .4], [.5, .9, .4], [.5, .3, .4]])
    means = segs[np.searchsorted(np.array([3000, 5000]), t1, side="right")]
else:
    means = np.stack([0.5 + 0.4 * np.cos(2 * math.pi * 3 * t1 / T), np.full(T, 0.5)], 1)
K = means.shape[1]
gam = 1 - 1 / (4 * math.sqrt(T)); tau = int(round(4 * math.sqrt(T * math.log(T))))
cases = {"ucb1": dict(xi=0.5), "ducb": dict(xi=0.5, gamma=gam), "ducb_padB": dict(xi=0.125, gamma=gam),
         "swucb": dict(xi=0.5, tau=tau)}
rng = np.random.default_rng(seed)
res = {c: {"asis": [], "aligned": []} for c in cases}
for r in range(R):
    X = (rng.random((T, K)) < means).astype(float)
    for c, kw in cases.items():
        for mode in ("asis", "aligned"):
            _, _, reg = run(make(c, kw, K, T, mode), c, X, None, means, K, T)
            res[c][mode].append(reg)
summ = {}
for c in cases:
    a, b = np.array(res[c]["asis"]), np.array(res[c]["aligned"])
    dd = a - b
    summ[c] = dict(asis=float(a.mean()), aligned=float(b.mean()), diff=float(dd.mean()), se=float(dd.std(ddof=1) / np.sqrt(R)))
    print(ename, c, summ[c], flush=True)
json.dump({"env": ename, "R": R, "seed": seed, "summary": summ, "res": res}, open(outp, "w"))

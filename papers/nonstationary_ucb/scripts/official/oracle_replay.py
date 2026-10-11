"""Replay OUR reward tables / EXP3.S uniforms through the SMPyBandits policy code (third-party oracle,
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

d = np.load(sys.argv[1])
T = 10_000
out = {}
for ename, K in (("e1", 3), ("e2", 2)):
    means = d[f"{ename}_means"]
    for case in ("ucb1", "ducb", "ducb_padB", "swucb", "exp3s"):
        kw = eval(str(d[f"{ename}_{case}_kw"]))
        X, Up, oa, oaux, oreg = (d[f"{ename}_{case}_{k}"] for k in ("X", "Up", "acts", "aux", "regret"))
        modes = ("asis",) if case == "exp3s" else ("asis", "aligned")
        for mode in modes:
            rows = []
            for r in range(X.shape[0]):
                pol = make(case, kw, K, T, mode)
                acts, aux, reg = run(pol, case, X[r], Up[r], means, K, T)
                mism = np.nonzero(acts != oa[r])[0]
                first = int(mism[0]) if mism.size else -1
                pre = slice(K, first if first >= 0 else T)  # skip init rounds
                a1, a2 = aux[pre], oaux[r][pre]
                fin = np.isfinite(a1) & np.isfinite(a2)
                maxdiff = float(np.max(np.abs(a1[fin] - a2[fin]))) if fin.any() else 0.0
                rows.append(dict(rep=r, identical=bool(first < 0), first_mismatch_round=(first + 1 if first >= 0 else -1),
                                 frac_agree=float(np.mean(acts == oa[r])), regret_oracle=reg, regret_ours=float(oreg[r]),
                                 max_abs_aux_diff_before_mismatch=maxdiff))
            key = f"{ename}/{case}/{mode}"
            n_id = sum(x["identical"] for x in rows)
            dreg = np.array([x["regret_oracle"] - x["regret_ours"] for x in rows])
            out[key] = dict(kw=kw, n_reps=len(rows), n_identical=n_id,
                            mean_regret_oracle=float(np.mean([x["regret_oracle"] for x in rows])),
                            mean_regret_ours=float(np.mean([x["regret_ours"] for x in rows])),
                            mean_paired_regret_diff=float(dreg.mean()), se_paired=float(dreg.std(ddof=1) / np.sqrt(len(dreg))),
                            max_aux_diff=float(max(x["max_abs_aux_diff_before_mismatch"] for x in rows)),
                            first_mismatch_rounds=[x["first_mismatch_round"] for x in rows][:10],
                            mean_frac_agree=float(np.mean([x["frac_agree"] for x in rows])))
            print(key, {k: v for k, v in out[key].items() if k != "kw"}, flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)

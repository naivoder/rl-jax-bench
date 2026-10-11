"""Replay OUR UCB-1 trajectories through the AUTHORS' own pyBandits UCB (Cappé & Garivier, pymaBandits 2012,
pyBandits/policy/UCB.py = klUCB with klucbGauss(sig2=.25): x + sqrt(0.5 log t / N), t = current round).
Harness patch (labelled): ties -> lowest index (their IndexPolicy uses random.choice). Throwaway probe.
"""
import os, sys
OFFICIAL_ROOT = os.environ.get('OFFICIAL_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'external', 'official'))
P = os.path.join(OFFICIAL_ROOT, 'pymabandits_dl', 'py3port')
sys.path.insert(0, P + '/policy'); sys.path.insert(0, P)
import numpy as np
import IndexPolicy as IP
from UCB import UCB
def lowest(self):
    idx = [self.computeIndex(a) for a in range(self.nbArms)]
    m = max(idx); return idx.index(m)
IP.IndexPolicy.choice = lowest
d = np.load(sys.argv[1]); T = 10_000
for env, K in (("e1", 3), ("e2", 2)):
    X, oa, means = d[f"{env}_ucb1_X"], d[f"{env}_ucb1_acts"], d[f"{env}_means"]
    n_id, firsts = 0, []
    for r in range(X.shape[0]):
        pol = UCB(K); pol.startGame(); acts = np.empty(T, int)
        for t in range(T):
            a = pol.choice(); acts[t] = a; pol.getReward(a, float(X[r, t, a]))
        same = np.array_equal(acts, oa[r]); n_id += same
        firsts.append(-1 if same else int(np.argmax(acts != oa[r])) + 1)
    print(env, "authors' pyBandits UCB vs our ucb1: identical", n_id, "/", X.shape[0], "first mismatch rounds", firsts[:10])

"""Analysis of the official-code oracle runs (rules declared before launch; see the README).  Project venv (scipy); cwd = the work directory."""
import sys, json, glob, os
import numpy as np
from scipy import stats
MU_G = 6.784
H = os.getcwd()   # the work directory holding ref/ and probe_ours_warmup.json
def load(arm, n):
    out = {}
    for f in glob.glob(f'{H}/ref/{arm}_n{n}_s*.json'):
        d = json.load(open(f)); out[d['seed']] = d
    return out
def early(d): return float(np.mean(d['alone'][1:11]))
def final(d): return float(np.mean(d['alone'][98:101])) if len(d['alone']) >= 101 else None
def pair(a, b, f, name):
    s = sorted(set(a) & set(b)); x = np.array([f(a[k]) for k in s]); y = np.array([f(b[k]) for k in s])
    if len(s) < 2 or any(v is None for v in x) or any(v is None for v in y): return dict(name=name, n=len(s))
    D = x - y; t = stats.ttest_rel(x, y)
    return dict(name=name, n=len(s), seeds=s, A=float(x.mean()), B=float(y.mean()), D=float(D.mean()), se=float(D.std(ddof=1)/np.sqrt(len(D))),
                wins=int((D > 0).sum()), p2=float(t.pvalue), per_seed=D.round(3).tolist())
res = {}
A = {k: load(k, 10) for k in ('scratch', 'jsrl_paper', 'jsrl_default', 'jsrl_paper_g0')}
A1 = {k: load(k, 1) for k in ('scratch', 'jsrl_paper')}
res['OR-1 EARLY paper-scratch'] = pair(A['jsrl_paper'], A['scratch'], early, 'OR-1')
res['OR-1d EARLY default-scratch'] = pair(A['jsrl_default'], A['scratch'], early, 'OR-1d')
res['P-A EARLY g0-paper'] = pair(A['jsrl_paper_g0'], A['jsrl_paper'], early, 'P-A')
res['P-A EARLY g0-scratch'] = pair(A['jsrl_paper_g0'], A['scratch'], early, 'P-A2')
res['FINAL paper-scratch'] = pair(A['jsrl_paper'], A['scratch'], final, 'C3f')
res['FINAL default-scratch'] = pair(A['jsrl_default'], A['scratch'], final, 'C3fd')
res['NSTEP1 EARLY paper-scratch'] = pair(A1['jsrl_paper'], A1['scratch'], early, 'N1')
for k, v in list(A.items()) + [(k + '_n1', v) for k, v in A1.items()]:
    if not v: continue
    L = min(len(d['alone']) for d in v.values())
    M = np.array([d['alone'][:L] for d in v.values()])
    curve = M.mean(0)
    hz = [next((d['step'][i] for i, h in enumerate(d['horizon']) if h == 0 and i > 0), None) for d in v.values()] if 'jsrl' in k else None
    t50 = next((v0['step'][i] for i, c in enumerate(curve) if c >= 0.5 * MU_G), None) if (v0 := next(iter(v.values()))) else None
    res[f'curve {k}'] = dict(n=len(v), early_mean=float(M[:, 1:11].mean()), final_mean=float(M[:, 98:101].mean()) if L >= 101 else None,
                            final_minus_mu_g=float(M[:, 98:101].mean() - MU_G) if L >= 101 else None,
                            at_pts=[round(float(curve[i]), 2) for i in (0, 5, 10, 15, 20, 30, 50, 75, 100) if i < L], t50=t50, first_h0_step=hz,
                            horizon_at_pt10=[d['horizon'][10] for d in v.values()] if 'jsrl' in k and L > 10 else None)
pb = f'{H}/probe_ours_warmup.json'
if os.path.exists(pb):
    d = json.load(open(pb)); S = len(d['seeds']); E = np.array(d['ev_alone'])
    ea = [E[i*S:(i+1)*S, 1:11].mean(1) for i in range(3)]
    def pr(x, y, nm):
        D = x - y; t = stats.ttest_rel(x, y); return dict(name=nm, A=float(x.mean()), B=float(y.mean()), D=float(D.mean()), se=float(D.std(ddof=1)/np.sqrt(len(D))), wins=int((D > 0).sum()), p2=float(t.pvalue), per_seed=D.round(3).tolist())
    res['P-B EARLY warm-JSRL'] = pr(ea[2], ea[1], 'P-B')
    res['P-B EARLY warm-PDD'] = pr(ea[2], ea[0], 'P-B2')
    res['P-B EARLY JSRL-PDD (ours, these seeds)'] = pr(ea[1], ea[0], 'P-B3')
    res['P-B curves pts 0..20'] = {a: E[i*S:(i+1)*S, :21].mean(0).round(2).tolist() for i, a in enumerate(d['arms'])}
    res['P-B stage at pt 10/20'] = {a: [np.array(d['ev_stage'])[i*S:(i+1)*S, 10].tolist(), np.array(d['ev_stage'])[i*S:(i+1)*S, 20].tolist()] for i, a in enumerate(d['arms'])}
    res['P-B diag (agree with guide labels, Q gap guide-best-other)'] = {pt: {a: [float(np.mean(v['agree'][i*S:(i+1)*S])), float(np.mean(v['qgap'][i*S:(i+1)*S]))] for i, a in enumerate(d['arms'])} for pt, v in d['diag'].items()}
json.dump(res, open(f'{H}/analysis.json', 'w'), indent=1)
for k, v in res.items(): print(k, json.dumps(v))

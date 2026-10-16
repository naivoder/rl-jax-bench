import json,glob,math,numpy as np
from scipy import stats as st
R='runs/'
def load(tag,env):
    L=[]
    for f in sorted(glob.glob(R+f'{tag}/{env}/seed_*/metrics.json')): L+=json.load(open(f))['lanes']
    return L
L=load('full','halfcheetah')
sc={};c0={};
for l in L:
    s=float(np.mean(l['curve'][1:])); assert abs(s-l['score'])<1e-3*abs(s), (l['arm'],l['seed'])
    sc.setdefault(l['arm'],{})[l['seed']]=s; c0.setdefault(l['arm'],{})[l['seed']]=l['curve'][0]
    assert len(l['curve'])==11
print('scores recompute from curves OK; nonfinite:',[ (l['arm'],l['seed']) for l in L if l['nonfinite']])
seeds=sorted(sc['REC']); rec=np.array([sc['REC'][s] for s in seeds])
print('REC',rec.round(0),rec.mean().round(1),(rec.std(ddof=1)/math.sqrt(5)).round(1))
rep=json.load(open(R+'full/full_report.json'))
ps={}
for r in rep['primary']:
    a=r['arm']; x=np.array([sc[a][s] for s in seeds]); d=x-rec
    t=st.ttest_rel(x,rec); ps[a]=t.pvalue
    print(f"{a:20s} rho={x.mean()/rec.mean():.3f} (rep {r['rho']:.3f}) diff={d.mean():+.0f}±{d.std(ddof=1)/math.sqrt(5):.0f} p={t.pvalue:.4f} (rep {r['t_p']:.4f}) holm_rep={r['p_holm']:.3f} judg={r['judgment']} delta={r['delta_pct']:+.1f} tost={r.get('tost_p')}")
# holm
names=sorted(ps,key=ps.get);run=0
for i,n in enumerate(names):
    run=max(run,min(1,(9-i)*ps[n])); print('holm',n,round(run,3))
# CRN check: step-0 eval returns per seed across arms
print('step-0 curve by seed:')
for s in seeds: print(s,{a[:6]:round(c0[a][s],1) for a in c0})

import json, os, sys, numpy as np
from scipy import stats
R=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'runs')
def load(tag,n=100):
    out={}
    for r in range(n):
        d=json.load(open(f'{R}/{tag}/seed_{r}/metrics.json'))
        for cid,c in d['cells'].items():
            for pol,v in c.get('J',{}).items():
                out.setdefault(cid,{}).setdefault(pol,[None]*n)[r]=v
    return out
paper={'t1_q2':[(823.19,400.53,284.65),(868.71,405.85,293.80),(871.19,406.17,293.72)],
't1_q29':[(961.25,458.49,319.84),(993.93,464.43,326.04),(1009.15,465.12,334.06)],
't2_q2':[(750.91,377.36,268.30),(790.61,381.92,275.81),(790.40,384.06,275.75)],
't2_q29':[(817.23,406.26,285.67),(849.91,409.88,296.35),(861.88,410.56,296.19)],
't3_q2':[(1554.35,772.19,547.38),(1614.41,807.35,567.14),(1622.97,808.89,567.94)],
't3_q2345':[(1664.83,821.05,581.47),(1731.69,859.02,605.75),(1733.95,860.42,605.73)],
't4_reckless':[(4364.18,1142.87,610.10),(4480.28,1153.19,613.15),(4436.27,1153.43,633.80)],
't4_cautious':[(3468.00,902.06,492.69),(3584.63,931.63,500.40),(3534.40,917.27,504.08)],
't4_mixed':[(6777.79,1824.24,990.12),(7014.92,1895.25,1022.45),(6879.65,1860.21,1040.56)]}
def holm(p):
    p=np.asarray(p); m=len(p); o=np.argsort(p); adj=np.empty(m); run=0
    for i,j in enumerate(o):
        run=max(run,(m-i)*p[j]); adj[j]=min(1,run)
    return adj
for tag in ['full','full_s1']:
    d=load(tag)
    devs=[];ps=[];diffs=[];none=0
    for sc,rows in paper.items():
        for ki,K in enumerate([1,2,3]):
            c=d[f'{sc}|K={K}|primary']
            for pi,pol in enumerate(['whittle','myopic','tev']):
                x=np.array(c[pol]); assert None not in c[pol]
                devs.append(abs(x.mean()/rows[pi][ki]-1))
            w=np.array(c['whittle'])
            for pol in ['myopic','tev']:
                dd=w-np.array(c[pol]); t,p=stats.ttest_rel(w,np.array(c[pol])); ps.append(p); diffs.append(dd.mean())
    adj=holm(ps)
    print(tag,'max|dev|%.4f med %.4f'%(max(devs),np.median(devs)),'n MATCH',sum(np.array(devs)<=.05),'/',len(devs),
          'primary repro', sum((np.array(diffs)<0)&(adj<.05)),'/54 maxHolm %.3g'%adj.max())
# SE check
d=load('full'); x=np.array(d['t1_q2|K=1|primary']['whittle']); print('t1_q2K1 whittle mean %.2f se %.2f sd %.2f'%(x.mean(),x.std(ddof=1)/10,x.std(ddof=1)))
x=np.array(d['t4_reckless|K=3|primary']['whittle'])-np.array(d['t4_reckless|K=3|primary']['myopic']); print('t4rK3 diff %.3f se %.3f'%(x.mean(),x.std(ddof=1)/10), stats.ttest_rel(np.array(d['t4_reckless|K=3|primary']['whittle']),np.array(d['t4_reckless|K=3|primary']['myopic'])))
# rollout1 vs whittle
for sc in paper:
  for K in [1,2,3]:
    c=d[f'{sc}|K={K}|primary']
    if 'rollout1' in c:
      w=np.array(c['whittle']);r=np.array(c['rollout1']); print(sc,K,'idx-r1 rel %.2f%%'%(100*(w-r).mean()/r.mean()), end='; ')
print()
# tuning parity: selection on full_s1
s1=load('full_s1')
print('cells in full_s1 with tau variants:', sum(1 for k in s1 if 'tau' in k))

import numpy as np, itertools
target=np.array([1.07,1.24,1.45,1.72,2.04,2.46,2.96]); ps=np.linspace(0.1,0.9,7)
def solve(p,n,eps,step_s,step_l,fin,absv,mode):
    V=fin; Qs=Ql=None
    out=[]
    for k in range(n-1,-1,-1):
        nxt=V
        e_s = eps if mode in('all','short') else 0
        e_l = eps if mode in('all','long') else 0
        qs=step_s+(1-e_s)*fin+e_s*absv
        ql=step_l+(1-e_l)*nxt+e_l*absv
        V=(1-p)*qs+p*ql; out.append((qs,ql,V))
    return out[-1]
res=[]
for n,eps,ss,sl,fin,absv,mode in itertools.product(range(2,12),[0.1],[-1,0],[-1],[0,1,2,-1],[0,1,-1,-2],['all','long','short']):
    for qty in ['gap','A','Apl']:
        vals=[]
        for p in ps:
            qs,ql,v=solve(p,n,eps,ss,sl,fin,absv,mode)
            vals.append({'gap':qs-ql,'A':qs-v,'Apl':(qs-v)/p}[qty])
        e=np.abs(np.array(vals)-target).max()
        qs,ql,v0=solve(0.5,n,eps,ss,sl,fin,absv,mode)
        res.append((round(e,3),n,ss,fin,absv,mode,qty,round(v0,3),round(qs,3),np.round(vals,2)))
res.sort(key=lambda r:r[0])
for r in res[:15]: print(r)
print('--- with V0 near -1.8')
for r in res:
    if abs(r[7]+1.8)<0.12 and r[0]<0.3: print(r)

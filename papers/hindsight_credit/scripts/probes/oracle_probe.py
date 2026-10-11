import numpy as np, itertools
target=np.array([1.07,1.24,1.45,1.72,2.04,2.46,2.96])
ps=np.linspace(0.1,0.9,7)
def solve(p,n,eps,short_r,final_r,step,absorb_val,absorb_mode):
    # states 0..n-1; returns Q_s, Q_l, V at state 0
    # last chain state: long action goes to goal
    V_next=None
    Vs=[0]*(n+1)
    Qs=[0]*n;Ql=[0]*n
    Vgoal=final_r
    for k in range(n-1,-1,-1):
        nxt = Vgoal if k==n-1 else Vs[k+1]
        if absorb_mode=='all':
            qs = step + (1-eps)*(short_r) + eps*absorb_val
            ql = step + (1-eps)*nxt + eps*absorb_val
        else: # absorb only on long
            qs = step + short_r
            ql = step + (1-eps)*nxt + eps*absorb_val
        Qs[k]=qs;Ql[k]=ql;Vs[k]=(1-p)*qs+p*ql
    return Qs[0],Ql[0],Vs[0],Qs,Ql,Vs
best=[]
for n in range(2,12):
  for eps in [0.0,0.1]:
    for final_r in [0,1]:
      for absorb_val in [0,1,-1]:
        for mode in ['all','long']:
          for qty in ['gap','A','A_over_pi','sumA']:
            vals=[]
            for p in ps:
              qs,ql,v,Qs,Ql,Vs=solve(p,n,eps,final_r,final_r,-1,absorb_val,mode)
              if qty=='gap': vals.append(qs-ql)
              elif qty=='A': vals.append(qs-v)
              elif qty=='A_over_pi': vals.append((qs-v)/(1-p))
              else:
                # expected sum of A(short) over visited chain states
                tot=0;reach=1
                for k in range(n):
                  tot+=reach*(Qs[k]-Vs[k]); reach*=p*(1-eps)
                vals.append(tot)
            err=np.abs(np.array(vals)-target).max()
            best.append((err,n,eps,final_r,absorb_val,mode,qty,np.round(vals,2)))
best.sort(key=lambda t:t[0])
for b in best[:12]: print(b)

import numpy as np
N=9
fig={0:[0],3:[1,1,.95,0],6:[1,1,1,.98,.5,.1,0],9:[1,1,1,1,1,1,.9,.1,0],14:[1,1,1,1,1,1,1,1,.97]}
def dp(T,g,pa,pc,ps=0.5):
    V=np.zeros((T+1,N+1)); P=np.full((T,N),0.5)
    for t in range(T-1,-1,-1):
        for x in range(N):
            nxt=lambda y: 1.0 if y==N else g*V[t+1,y]
            qs=ps*nxt(x+1)+(1-ps)*g*V[t+1,x]
            qd=pa*nxt(x+1)+(1-pa-pc)*g*V[t+1,x]
            V[t,x]=max(qs,qd); P[t,x]=1.0 if qd>qs+1e-12 else (0.0 if qs>qd+1e-12 else 0.5)
    return P
res=[]
for T in range(10,25):
  for pa in (0.7,0.72,0.8,0.9):
    for pc in (0.1,):
      for g in (0.99,1.0):
        P=dp(T,g,pa,pc)
        if P.shape[0]<15: continue
        err=0
        for t,row in fig.items():
            for x,v in enumerate(row):
                if P[t,x]==0.5: continue
                err+=abs(P[t,x]-v)
        res.append((err,T,pa,pc,g))
for r in sorted(res)[:12]: print(r)

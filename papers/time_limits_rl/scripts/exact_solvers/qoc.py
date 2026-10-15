import numpy as np, itertools
N=9
def trans(sem):
    # returns dict action-> list of (prob, delta or 'crash')
    safe=[(0.5,1),(0.5,0)]
    if sem=='a': dang=[(0.8,1),(0.1,'c'),(0.1,0)]
    elif sem=='b': dang=[(0.8*0.9,1),(0.1,'c'),(0.2*0.9,0)]  # crash first then 80% advance
    elif sem=='c': dang=[(0.8,1),(0.2*0.1/0.2*0.5,'c'),(0.1,0)]  # crash only when not advancing? (0.2 non-advance, half crash)
    return safe,dang
def dp(T,g,sem):
    safe,dang=trans(sem)
    V=np.zeros((T+1,N+1)); P=np.zeros((T,N))
    for t in range(T-1,-1,-1):
        for x in range(N):
            q=[]
            for A in (safe,dang):
                v=0
                for p,d in A:
                    if d=='c': continue
                    y=x+d
                    if y==N: v+=p*1.0
                    else: v+=p*g*V[t+1,y]
                q.append(v)
            V[t,x]=max(q); P[t,x]= 1.0 if q[1]>q[0]+1e-12 else (0.5 if abs(q[1]-q[0])<1e-12 else 0.0)
    return V,P
target={3:[1,1,1,0],6:[1,1,1,1,.5,0,0],9:[1,1,1,1,1,1,.9,0,0]}
for sem in 'ab':
  for g in (0.99,1.0):
    for T in range(12,31):
        V,P=dp(T,g,sem)
        if P.shape[0]<15: continue
        rows={t:P[t,:t+1 if t<9 else 9] for t in (0,3,6,9,14)}
        ok = P[0,0]==0 and all(P[14,:]==1) if T==15 else P[0,0]==0
        s3=''.join(str(int(v)) for v in P[3,:4]); s6=''.join(str(int(v)) for v in P[6,:7]); s9=''.join(str(int(v)) for v in P[9,:9]); s14=''.join(str(int(v)) for v in P[14,:])
        print(sem,g,T,'t0:',int(P[0,0]),'t3',s3,'t6',s6,'t9',s9,'t14',s14,'V0=%.3f'%V[0,0])
print('---- with ties as "-"')
def s(r): return ''.join('-' if v==0.5 else str(int(v)) for v in r)
for sem in 'ab':
  for g in (0.99,1.0):
    for T in range(15,21):
        V,P=dp(T,g,sem)
        print(sem,g,T,' | '.join(f't{t}:'+s(P[t]) for t in (0,3,6,9,14)))

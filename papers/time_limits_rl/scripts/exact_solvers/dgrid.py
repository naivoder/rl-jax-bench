import numpy as np
N=20
wall=np.zeros((N,N),bool)
wall[1,1]=True; wall[1,3:19]=True; wall[1:19,18]=True; wall[18,1:19]=True; wall[3:19,1]=True
start=(10,10); goal=(19,19)
for r in range(N): print(''.join('#' if wall[r,c] else ('S' if (r,c)==start else ('G' if (r,c)==goal else '.')) for c in range(N)))
A=[(-1,0),(1,0),(0,-1),(0,1)]
cells=[(r,c) for r in range(N) for c in range(N) if not wall[r,c]]
idx={p:i for i,p in enumerate(cells)}; nS=len(cells)
NX=np.zeros((nS,4),int)
for i,(r,c) in enumerate(cells):
    for a,(dr,dc) in enumerate(A):
        nr,nc=r+dr,c+dc
        NX[i,a]=idx[(nr,nc)] if 0<=nr<N and 0<=nc<N and not wall[nr,nc] else i
G=idx[goal]; S=idx[start]
V=np.zeros(nS)
for _ in range(1000):
    Q=-1+V[NX]; Q[G]=0; Vn=Q.max(1); Vn[G]=0
    if np.allclose(Vn,V): break
    V=Vn
print('optimal undiscounted V*(start) =',V[S], ' (=-shortest path length)')
# epsilon-greedy (eps=0.1, uniform over 4 incl greedy) return with T=200 truncation, exact via time-indexed DP
eps=0.1
Q=-1+V[NX]
for tie in ('first','uniform'):
    pi=np.full((nS,4),eps/4)
    for i in range(nS):
        best=np.flatnonzero(Q[i]>=Q[i].max()-1e-9)
        if tie=='first': pi[i,best[0]]+=1-eps
        else: pi[i,best]+=(1-eps)/len(best)
    # expected return over horizon T=200 from start: E[-steps until goal or 200]
    Vt=np.zeros(nS)
    for k in range(200):
        Vt=np.array([0.0 if i==G else (pi[i]*(-1+Vt[NX[i]])).sum() for i in range(nS)])
    # infinite horizon
    Vi=np.zeros(nS)
    for k in range(5000):
        Vi=np.array([0.0 if i==G else (pi[i]*(-1+Vi[NX[i]])).sum() for i in range(nS)])
    print(f'eps-greedy(0.1, ties {tie}) return from start: T=200: {Vt[S]:.4f}; untruncated {Vi[S]:.4f}')

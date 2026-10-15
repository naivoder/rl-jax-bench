import numpy as np, sys
M=["..#..A",
   "....#.",
   ".##...",
   "..###.",
   "#...#.",
   "B....."]
H,W=6,6
cells=[(r,c) for r in range(H) for c in range(W) if M[r][c]!='#']
idx={p:i for i,p in enumerate(cells)}
goal={idx[(0,5)]:50.,idx[(5,0)]:20.}
nong=[i for i in range(len(cells)) if i not in goal]
ACTS=[(-1,0),(1,0),(0,-1),(0,1),(0,0)]  # up down left right stay
NAMES='^v<>.'
BUMP=float(sys.argv[1]) if len(sys.argv)>1 else -1.0
def step(i,a):
    r,c=cells[i]; dr,dc=ACTS[a]
    if a==4: return i,0.0
    nr,nc=r+dr,c+dc
    if 0<=nr<H and 0<=nc<W and M[nr][nc]!='#':
        j=idx[(nr,nc)]; rew=-1.0+goal.get(j,0.0); return j,rew
    return i,BUMP
nS,nA=len(cells),5
NX=np.zeros((nS,nA),int); R=np.zeros((nS,nA))
for i in nong:
    for a in range(nA): NX[i,a],R[i,a]=step(i,a)
g=0.99; T=3
isg=np.array([i in goal for i in range(nS)])
# time-aware finite-horizon: Q_k with k remaining steps
Vk=np.zeros(nS); Qta={}
for k in range(1,T+1):
    Q=R+g*np.where(isg[NX],0,1)*Vk[NX]; Q[isg]=0; Qta[k]=Q; Vk=Q.max(1); Vk[isg]=0
# PEB infinite horizon
V=np.zeros(nS)
for _ in range(5000):
    Q=R+g*np.where(isg[NX],0,1)*V[NX]; Q[isg]=0; V=Q.max(1); V[isg]=0
Qpeb=Q
# visitation under uniform random behavior, uniform start over non-goal
d=np.zeros((T,nS)); d[0,nong]=1/len(nong)
for t in range(1,T):
    for i in nong:
        for a in range(nA):
            j=NX[i,a]
            if not isg[j]: d[t,j]+=d[t-1,i]/nA
pboot=np.where(d.sum(0)>0, d[:T-1].sum(0)/np.maximum(d.sum(0),1e-12),0)
Q=np.zeros((nS,nA))
for _ in range(20000):
    V=Q.max(1); V[isg]=0
    Qn=R+g*np.where(isg[NX],0,1)*pboot[:,None]*V[NX]; Qn[isg]=0
    if np.abs(Qn-Q).max()<1e-13: break
    Q=Qn
Qstd=Q
def show(Q,name,vals=True):
    print('==',name)
    for r in range(H):
        row=[]
        for c in range(W):
            if M[r][c]=='#': row.append('   ##   '); continue
            i=idx[(r,c)]
            if isg[i]: row.append(f'  G{int(goal[i]):3d}  '); continue
            q=Q[i]; best=''.join(NAMES[a] for a in range(nA) if q[a]>=q.max()-1e-9)
            row.append(f'{q.max():6.2f}{best:<3s}')
        print(' '.join(row))
print('pboot (P not last step | s):'); 
for r in range(H): print(' '.join('  ## ' if M[r][c]=='#' else f'{pboot[idx[(r,c)]]:.3f}' for c in range(W)))
show(Qta[3],'time-aware, 3 steps remaining (t=0)')
show(Qta[2],'time-aware, 2 remaining'); show(Qta[1],'time-aware, 1 remaining')
show(Qstd,'standard time-unaware Q-learning fixed point (uniform-random behavior)')
show(Qpeb,'PEB = infinite-horizon gamma=0.99')
np.savez('grid_fp.npz',Qstd=Qstd,Qpeb=Qpeb,Qta3=Qta[3],pboot=pboot)

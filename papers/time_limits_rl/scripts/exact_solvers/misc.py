import numpy as np
from scipy.optimize import minimize_scalar
print('== Last Moment: time-unaware memoryless jump prob p, return from A at t=0')
def E(p,T,g):
    tot=0.0
    for k in range(T):
        tail=sum(g**j for j in range(1,T-k))   # penalties in B after jump at step k
        tot+=(1-p)**k*p*g**k*(1-tail)
    return tot
for g in (1.0,0.99):
  for T in (1,2,3,4,5,10):
    r=minimize_scalar(lambda p:-E(p,T,g),bounds=(0,1),method='bounded',options={'xatol':1e-12})
    pg=np.linspace(0,1,100001); ev=np.array([E(p,T,g) for p in pg]) if T<=5 else None
    pbest = pg[ev.argmax()] if ev is not None else r.x
    ta = g**(T-1)  # time-aware: stay T-1 steps then jump
    print(f'g={g} T={T}: p*={pbest:.5f} E*={E(pbest,T,g):.6f} | time-aware optimum={ta:.6f}')
print('closed form T=2: p*=1/(2g):',1/(2*0.99))
print('== constant-reward aliased chain (InvertedPendulum idealisation, r=1, episode always reaches T)')
def alias(T,g,lam):
    a=np.zeros(T); b=np.zeros(T); a[0]=1; b[0]=0
    for k in range(1,T): a[k]=1+g*lam*a[k-1]; b[k]=g*(1-lam)+g*lam*b[k-1]
    return a.mean()/(1-b.mean())
for g in (0.99,1.0):
  for T in (1000,300,50):
    ta=[(1-g**k)/(1-g) if g<1 else k for k in (1,T)]
    mc=np.mean([(1-g**k)/(1-g) if g<1 else k for k in range(1,T+1)])
    print(f'g={g} T={T}: TD0={alias(T,g,0):.4f} GAE.95={alias(T,g,0.95):.4f} MC={alias(T,g,1):.4f} (mean of time-aware values={mc:.4f}); time-aware v(k=1)={ta[0]:.3f} v(k=T)={ta[1]:.3f}; PEB/infinite={1/(1-g) if g<1 else float("inf")}')

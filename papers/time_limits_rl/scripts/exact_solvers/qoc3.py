import numpy as np
from scipy.optimize import minimize
N=9;T=15;g=0.99
def ret(p,pa=0.8,pc=0.1):
    # p: P(dangerous) per slot (memoryless). return expected discounted reward from slot 0, t=0
    dist=np.zeros(N); dist[0]=1; tot=0
    for t in range(T):
        adv=0.5*(1-p)+pa*p; stay=0.5*(1-p)+(1-pa-pc)*p
        tot+=g**t*dist[N-1]*adv[N-1]   # reward at t+1 discounted by g^t (reward R_{t+1} weight g^t)
        nd=np.zeros(N); nd[1:]+=dist[:-1]*adv[:-1]; nd+=dist*stay
        dist=nd
    return tot
best=None
for s in range(200):
    x0=np.random.default_rng(s).uniform(-3,3,N)
    r=minimize(lambda x:-ret(1/(1+np.exp(-x))),x0,method='L-BFGS-B')
    if best is None or r.fun<best.fun: best=r
pb=1/(1+np.exp(-best.x))
print('best memoryless (time-unaware) policy P(dangerous):',np.round(pb,3),' value=%.6f'%(-best.fun))
# deterministic enumeration
import itertools
vals=[(ret(np.array(b,float)),b) for b in itertools.product([0,1],repeat=N)]
vals.sort(reverse=True); print('best deterministic memoryless:',vals[0][1],'%.6f'%vals[0][0])
print('all-safe %.6f all-dangerous %.6f'%(ret(np.zeros(N)),ret(np.ones(N))))
best=None
rng=np.random.default_rng(0)
inits=[np.array(b,float)*8-4 for _,b in vals[:50]]+[rng.uniform(-6,6,N) for _ in range(300)]
for x0 in inits:
    r=minimize(lambda x:-ret(1/(1+np.exp(-x))),x0,method='L-BFGS-B',options={'maxiter':2000})
    if best is None or r.fun<best.fun: best=r
pb=1/(1+np.exp(-best.x)); print('refined best memoryless:',np.round(pb,4),'value=%.6f'%(-best.fun))
for gg in (1.0,):
    g=gg; print('gamma=1: TA-free check all-safe %.4f'%ret(np.zeros(N)))

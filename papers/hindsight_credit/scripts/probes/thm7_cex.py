# Counterexample to Theorem 7 (T-step bootstrapped state-HCA with a single mixture h_{beta,T}).
# x --a--> y --> z(terminal)   ;   x --b--> w --> y --> z
# state rewards r(x)=0, r(w)=c, r(y)=0, r(z)=1 ; pi uniform ; T=2 ; beta=gamma
import numpy as np
for g in (0.9,0.99,1.0):
    c=0.5; pi=0.5; T=2
    # trajectories: (prob, [X0,X1,X2,...], rewards per step)
    rew={'x':0,'w':c,'y':0,'z':1}
    def V(s):  # exact discounted value (state-reward convention, rewards collected on arrival incl. current)
        path={'z':['z'],'y':['y','z'],'w':['w','y','z']}[s]
        return sum(g**i*rew[p] for i,p in enumerate(path))
    trajA=['x','y','z']; trajB=['x','w','y','z']
    Q={'a':sum(g**k*rew[s] for k,s in enumerate(trajA)), 'b':sum(g**k*rew[s] for k,s in enumerate(trajB))}
    # rho_T(k): beta^(k-1)(1-beta) for k<T, beta^(T-1) for k=T  (beta=gamma); at beta=1 -> delta_T
    b=g
    rho={1:(1-b), 2:b} if b<1 else {1:0.0,2:1.0}
    # h_{beta,T}(a|x,y) = P(A0=a | X_K=y, K~rho_T)
    num={};den={}
    for act,tr in (('a',trajA),('b',trajB)):
        for k in (1,2):
            s=tr[k]; den[s]=den.get(s,0)+pi*rho[k]; num[(s,act)]=num.get((s,act),0)+pi*rho[k]
    h=lambda act,s: num.get((s,act),0)/den[s] if den.get(s,0)>0 else np.nan
    for act in ('a','b'):
        est=0
        for tr in (trajA,trajB):
            e=rew['x']  # r(x,a)=0
            e+=g*h(act,tr[1])/pi*rew[tr[1]] if not np.isnan(h(act,tr[1])) else 0
            e+=g**2*h(act,tr[2])/pi*V(tr[2]) if not np.isnan(h(act,tr[2])) else 0
            est+=pi*e
        print(f'gamma={g} action {act}: Thm7 estimator mean {est:.4f}  true Q {Q[act]:.4f}')

# Exact enumeration of the Shortcut chain (n=5, eps_absorb=0.1, step -1, goal +1 as state reward of G).
import itertools, numpy as np
from collections import defaultdict
n=5; eps=0.1; pl=0.5  # prob of long action at every chain state
S,L=0,1
def trajs():
    # yields (prob, states list, actions list, rewards list); states: ints 0..n-1, 'G','D'
    out=[]
    def rec(x,prob,xs,as_,rs):
        if x=='G': out.append((prob,xs+['G'],as_+[None],rs+[1.0])); return
        if x=='D': out.append((prob,xs+['D'],as_+[None],rs+[0.0])); return
        for a,pa in ((S,1-pl),(L,pl)):
            # absorb
            rec('D',prob*pa*eps,xs+[x],as_+[a],rs+[-1.0])
            nxt='G' if (a==S or x==n-1) else x+1
            rec(nxt,prob*pa*(1-eps),xs+[x],as_+[a],rs+[-1.0])
    rec(0,1.0,[],[],[]); return out
T=trajs(); print('n traj',len(T),'sumP',sum(t[0] for t in T))
pi={S:1-pl,L:pl}
Z=lambda t: sum(t[3])
Q={a: sum(t[0]*Z(t) for t in T if t[2][0]==a)/pi[a] for a in (S,L)}
V=sum(t[0]*Z(t) for t in T)
print('Q',Q,'V',V,'gap',Q[S]-Q[L])
# per-k hindsight hk(a|x0,y)
Pk=defaultdict(float); Pka=defaultdict(float)
for p,xs,as_,rs in T:
    for k in range(1,len(xs)):
        Pk[(k,xs[k])]+=p; Pka[(k,xs[k],as_[0])]+=p
hk=lambda a,k,y: Pka[(k,y,a)]/Pk[(k,y)]
# Theorem 1 check
for a in (S,L):
    est=-1+sum(p*sum(hk(a,k,xs[k])/pi[a]*rs[k] for k in range(1,len(xs))) for p,xs,as_,rs in T)
    print('Thm1 Q',a,est)
# occupancy (gamma=1) h: weights = visits at k>=1
Oc=defaultdict(float); Oca=defaultdict(float)
for p,xs,as_,rs in T:
    for k in range(1,len(xs)):
        Oc[xs[k]]+=p; Oca[(xs[k],as_[0])]+=p
ho=lambda a,y: Oca[(y,a)]/Oc[y]
print('h_occ(short|x0,G)=',ho(S,'G'),' h_occ(short|x0,D)=',ho(S,'D'),' h_occ(short|x0,s1)=',ho(S,1))
print('h_1(short|x0,G)=',hk(S,1,'G'),' h_2(short|x0,G)=',hk(S,2,'G'))
for a in (S,L):
    est=-1+sum(p*sum(ho(a,xs[k])/pi[a]*rs[k] for k in range(1,len(xs))) for p,xs,as_,rs in T)
    print('Thm6(gamma=1 occupancy) Q',a,est)
# Return-conditional exact
Pz=defaultdict(float); Pza=defaultdict(float)
for t in T: Pz[Z(t)]+=t[0]; Pza[(Z(t),t[2][0])]+=t[0]
print('Z support', sorted((round(z,3),round(Pz[z],5)) for z in Pz))
hz=lambda a,z: Pza[(z,a)]/Pz[z]
for a in (S,L):
    thm5=sum(t[0]*Z(t)*hz(a,Z(t))/pi[a] for t in T)
    # thm2 needs hz>0 for all z reachable under a
    ok=all(hz(a,Z(t))>0 for t in T if t[2][0]==a)
    thm2=sum(t[0]/pi[a]*Z(t)*pi[a]/hz(a,Z(t)) for t in T if t[2][0]==a) if ok else float('nan')
    print('Thm5 Q',a,thm5,' Thm2 V via a',a,thm2)
# T-step bootstrapped (Thm 7 analog at gamma=1): exact V at X_T; h = uniform mixture over k in 1..T
def Vx(x):
    # exact value of state x under pi via enumeration of the chain recursion
    if x in ('G',): return 1.0
    if x=='D': return 0.0
    vs=[]
    v=1.0
    for k in range(n-1,-1,-1):
        qs=-1+(1-eps)*1.0; ql=-1+(1-eps)*v
        v=(1-pl)*qs+pl*ql; vs.append(v)
    return vs[::-1][x]
print('V(x0) via DP',Vx(0))
for Tn in (2,3):
    # mixture h over k=1..Tn (states at k beyond termination don't count)
    M=defaultdict(float); Ma=defaultdict(float)
    for p,xs,as_,rs in T:
        for k in range(1,min(Tn,len(xs)-1)+1):
            M[xs[k]]+=p; Ma[(xs[k],as_[0])]+=p
    hm=lambda a,y: Ma[(y,a)]/M[y]
    for a in (S,L):
        est=0.0; est_k=0.0
        for p,xs,as_,rs in T:
            e=-1; ek=-1
            for k in range(1,len(xs)):
                if k<Tn: e+=hm(a,xs[k])/pi[a]*rs[k]; ek+=hk(a,k,xs[k])/pi[a]*rs[k]
                elif k==Tn:
                    # bootstrap; terminal states G/D: value of the remaining (reward at G counted as state reward)
                    vb = rs[k] if xs[k] in ('G','D') else Vx(xs[k])
                    e+=hm(a,xs[k])/pi[a]*vb; ek+=hk(a,k,xs[k])/pi[a]*vb
            est+=p*e; est_k+=p*ek
        print(f'T={Tn} a={a}: mixture-h Thm7 estimate {est:.5f}  per-k h estimate {est_k:.5f}  true {Q[a]:.5f}')

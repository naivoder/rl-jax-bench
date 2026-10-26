import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import jax, jax.numpy as jnp, numpy as np
import model as M, algorithm as A, data as D
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
EXACT_CASES = {"n2_k1_T10": (("reckless","cautious"),(4.0,4.0),(5.0,1.0),1,10),
 "n3_k1_T8": (("reckless","cautious","reckless"),(2.0,4.0,6.0),(5.0,1.0,1.0),1,8),
 "n4_k2_T6": (("reckless","reckless","cautious","cautious"),(2.0,5.0,2.0,5.0),(5.0,5.0,1.0,1.0),2,6)}
for name,(types,qct,d,K,T) in EXACT_CASES.items():
    bank=M.scalar_bank(list(types),list(qct),list(d)); N=len(types); nrep=40
    rk=D.rep_keys(0,"X2",50+N,K,nrep); p0k,tie,mode=D.split_rep_keys(rk)
    P0=jnp.maximum(jax.vmap(lambda k: jax.random.uniform(k,(N,),minval=0.0,maxval=2.0))(p0k),1e-12)[...,None,None]
    Jw=np.asarray(A.simulate(bank,P0,tie,mode,"whittle",K,A.Cfg(T=T,guard=False))["J"])
    a=[];b=[];nidle=0
    for r in range(nrep):
        j1,seq,Atab=A.exact_enumerate(bank,P0[r],T,K,allow_idle=True)
        j2,_,_=A.exact_enumerate(bank,P0[r],T,K,allow_idle=False)
        a.append(j1);b.append(j2); nidle+=int((seq==0).any())
    a=np.array(a);b=np.array(b)
    print(name,'J*idle %.4f J*fill %.4f maxrel %.2e  opt-uses-idle %d/40  regret idx vs idle %.3f%% vs fill %.3f%% n_opt_fill %d'%(a.mean(),b.mean(),np.max((b-a)/a),nidle,100*np.mean((Jw-a)/a),100*np.mean((Jw-b)/b),int((np.abs(Jw-b)<=1e-9*b).sum())))

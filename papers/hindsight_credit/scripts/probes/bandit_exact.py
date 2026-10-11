import numpy as np
from scipy.stats import norm
mu=np.array([1.0,2.0]); 
for sig,eps in ((1.5,0.1),(0.5,0.0),(0.5,0.2)):
  pi=np.array([0.5,0.5])
  P=np.array([[1-eps,eps],[eps,1-eps]])  # P[a,s]
  Q=P@mu; V=pi@Q; A=Q-V
  ps=pi@P
  h=(pi[:,None]*P)/ps[None,:]          # h[a,s]
  print(f'sig={sig} eps={eps}: Q={Q} V={V} A={A}  h(a|s)=\n{h}')
  # exact state-HCA all-actions Qx = E_s[h(a|s)/pi(a) mu_s]
  print('  thm1 Q', [(ps*h[a]/pi[a]*mu).sum() for a in range(2)])
  # return-conditional hz on continuous Z: density mixture
  zs=np.linspace(-12,15,200001); dz=zs[1]-zs[0]
  f=np.array([[norm.pdf(zs,mu[s],sig) for s in range(2)]])[0]  # f[s,z]
  fa=P@f   # density of Z given a
  fz=pi@fa
  hz=pi[:,None]*fa/fz[None,:]
  thm5=[(fz*zs*hz[a]/pi[a]).sum()*dz for a in range(2)]
  thm2=[(fa[a]*zs*pi[a]/hz[a]).sum()*dz for a in range(2)]
  print('  thm5 Q',np.round(thm5,5),' thm2 V',np.round(thm2,5))
  # binned hz, nb bins on [lo,hi] (outer bins open-ended)
  for nb in (3,10):
    lo,hi=mu.min()-3*sig,mu.max()+3*sig
    edges=np.linspace(lo,hi,nb+1); idx=np.clip(np.digitize(zs,edges)-1,0,nb-1)
    hb=np.zeros((2,nb))
    for b in range(nb):
      m=idx==b; hb[:,b]=(pi[:,None]*fa[:,m]).sum(1)/ (fz[m].sum())
    hzb=hb[:,idx]
    thm5b=[(fz*zs*hzb[a]/pi[a]).sum()*dz for a in range(2)]
    thm2b=[(fa[a]*zs*pi[a]/hzb[a]).sum()*dz for a in range(2)]
    # expected return-HCA PG at uniform logits: grad_theta_a = sum_b pi_b (1{a=b}-pi_a) E_b[(1-pi_b/h)Z]
    Eb=[(fa[b]*(1-pi[b]/hzb[b])*zs).sum()*dz for b in range(2)]
    g=[sum(pi[b]*((a==b)-pi[a])*Eb[b] for b in range(2)) for a in range(2)]
    gtrue=pi*A
    print(f'  nb={nb}: binned thm5 Q {np.round(thm5b,4)} thm2 V {np.round(thm2b,4)}  E[retHCA grad] {np.round(g,4)} true grad {np.round(gtrue,4)}')

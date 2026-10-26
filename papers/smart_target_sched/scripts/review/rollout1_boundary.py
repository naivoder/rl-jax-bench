import json, os, sys, numpy as np
from rl_jax_bench.stats import paired_compare, holm
R=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'runs'); S=os.path.join(R, 'review')
def load(root,tag):
    out={}
    for r in range(100):
        p=f'{root}/{tag}/seed_{r}/metrics.json'
        if not os.path.exists(p): return out
        for cid,c in json.load(open(p))['cells'].items():
            for pol,v in c.get('J',{}).items(): out.setdefault(cid,{}).setdefault(pol,[]).append(v)
    return out
g0,g1=load(R,'full'),load(R,'full_s1'); n0,n1=load(S,'r1_s0'),load(S,'r1_s1')
def J(d,cid,pol):
    v=d.get(cid,{}).get(pol); return np.array(v) if v is not None and len(v)==100 else None
CELLS=[(n,K) for n in ['t1_q2','t1_q29','t2_q2','t2_q29','t3_q2','t3_q2345','t4_reckless','t4_cautious','t4_mixed'] for K in (1,2,3)]
TAUS=[2,3,5,10,20,50,100,200]
fam={};rows=[]
for n,K in CELLS:
    var=lambda t:'primary' if t==100 else f'tau{t}'
    tc={t:(J(g1,f'{n}|K={K}|{var(t)}','whittle'),J(g0,f'{n}|K={K}|{var(t)}','whittle')) for t in TAUS}
    ts=min(tc,key=lambda t:tc[t][0].mean()); Jw=tc[ts][1]
    bc={}
    for b in (1,2,3,5,10,30,100):
        v='primary' if b==30 else f'r1b{b}'
        cid=f'{n}|K={K}|{v}'
        a=J(g1,cid,'rollout1'); c=J(g0,cid,'rollout1')
        if a is None: a=J(n1,cid,'rollout1')
        if c is None: c=J(n0,cid,'rollout1')
        if a is not None and c is not None: bc[b]=(a,c)
    bs=min(bc,key=lambda b:bc[b][0].mean()); old=[b for b in bc if b in (5,10,30,100)]; bso=min(old,key=lambda b:bc[b][0].mean())
    p=paired_compare(Jw,bc[bs][1]); fam[f'{n}|{K}']=p
    rows.append((n,K,ts,Jw.mean(),bso,bc[bso][1].mean(),bs,bc[bs][1].mean(),sorted(bc), {b:round(bc[b][1].mean(),1) for b in sorted(bc)}))
h=holm({k:v['t_p'] for k,v in fam.items()})
ch=de=0
for (n,K,ts,w,bso,ro,bs,rn,bl,allb) in rows:
    k=f'{n}|{K}'; p=fam[k]; s='index cheaper' if p['diff_mean']<0 and h[k]['p_holm']<.05 else ('index DEARER' if p['diff_mean']>0 and h[k]['p_holm']<.05 else 'ns')
    ch+=s=='index cheaper'; de+=s=='index DEARER'
    print(f"{n} K={K} tau*={ts} idx={w:.1f} | old b*={bso} R1={ro:.1f} | new b*={bs} R1={rn:.1f} rel(idx-R1)={100*p['rel_improvement']:+.2f}% holm(27)={h[k]['p_holm']:.2g} {s} | grid {allb}")
print(f'tuned index vs re-tuned ROLLOUT-1 (b grid incl 1,2,3 [+100 for 4-D where run]): cheaper {ch}/27, dearer {de}/27 (Holm over these 27 only)')

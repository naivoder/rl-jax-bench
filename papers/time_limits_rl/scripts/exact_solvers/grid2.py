import numpy as np
exec(open('grid.py').read().split("def show")[0])
def greedy(Q):
    return Q.argmax(1)
def eval_T(pol_fn,Tl):   # finite horizon value from each start with policy pol_fn(i,k)->a, k remaining
    Vk=np.zeros(nS)
    for k in range(1,Tl+1):
        Vn=np.zeros(nS)
        for i in nong:
            a=pol_fn(i,k); j=NX[i,a]; Vn[i]=R[i,a]+(0 if isg[j] else g*Vk[j])
        Vk=Vn
    return Vk
def eval_inf(pol):
    V=np.zeros(nS)
    for _ in range(20000):
        Vn=np.zeros(nS)
        for i in nong:
            a=pol[i]; j=NX[i,a]; Vn[i]=R[i,a]+(0 if isg[j] else g*V[j])
        V=Vn
    return V
# deterministic tie-break: first argmax
pstd=greedy(Qstd); ppeb=greedy(Qpeb)
ta=lambda i,k: Qta[k][i].argmax()
m=lambda V: V[nong].mean()
print('Objective T=3 (time-limited), mean over uniform start:')
print('  TA optimal          %.4f'%m(eval_T(ta,3)))
print('  standard greedy     %.4f'%m(eval_T(lambda i,k:pstd[i],3)))
print('  PEB greedy          %.4f'%m(eval_T(lambda i,k:ppeb[i],3)))
print('Objective infinite horizon gamma=0.99:')
print('  PEB greedy (optimal) %.4f'%m(eval_inf(ppeb)))
print('  standard greedy      %.4f'%m(eval_inf(pstd)))
print('  per-cell standard vs PEB greedy action differ at', [cells[i] for i in nong if Qpeb[i,pstd[i]]<Qpeb[i].max()-1e-9])
print('standard Q fixed point, max_a per cell (row-major non-goal):'); print(np.round(Qstd.max(1)[nong],6))
print('pboot non-goal:',np.round(pboot[nong],6))
print('d(t|s) for (0,4):',[d[t,idx[(0,4)]] for t in range(3)])

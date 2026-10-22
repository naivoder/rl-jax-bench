"""probe: train the SAME seed in vmapped groups of size 1, 2, 3, 4 for a few iterations (paper-size network, --lstm <n>) and report whether the parameters are bit-identical.
    python scripts/probe_group_size.py <lstm> <exp> <iters>      e.g. 100 E7 60  (output: runs/review/group_size_probe_<exp>.log)"""
import os, sys, tempfile, json
os.environ["JAX_PLATFORMS"]="cpu"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, train
lstm=sys.argv[1]; exp=sys.argv[2]; iters=sys.argv[3]
out={}
with tempfile.TemporaryDirectory() as d:
    for name,seeds in (("S2",["100","101"]),("S3",["100","101","102"]),("S1",["100"]),("S4",["99","100","101","102"])):
        train.main(["--exp",exp,"--seeds",*seeds,"--chunk",iters,"--ckpt-every",iters,"--monitor-every","100000","--n-exp","4","--lstm",lstm,"--iters",iters,"--sched-iters",iters,"--out",os.path.join(d,name),"--no-eval"] )
        # no-eval skips params.npz; read the checkpoint instead
        import orbax.checkpoint as ocp, jax
        tag=os.listdir(os.path.join(d,name))[0]
        mgr=ocp.CheckpointManager(os.path.abspath(os.path.join(d,name,tag,"ckpt")))
        r=mgr.restore(mgr.latest_step(), args=ocp.args.StandardRestore(None))
        out[name]=r
    def leaves(r,i): 
        import jax
        return [np.asarray(x)[i] for x in jax.tree.leaves(r["qs"]["params"])]
    for a,b,ia,ib in (("S2","S3",0,0),("S2","S3",1,1),("S3","S4",0,1),("S2","S1",0,0),("S3","S1",0,0)):
        A=leaves(out[a],ia); B=leaves(out[b],ib)
        print(a,b,"seed idx",ia,ib,"bit-identical:",all(np.array_equal(x,y) for x,y in zip(A,B)),"max|d|=%.3e"%max(float(np.abs(x-y).max()) for x,y in zip(A,B)))

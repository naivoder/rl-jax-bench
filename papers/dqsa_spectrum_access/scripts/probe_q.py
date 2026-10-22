"""Print Q(.|history) of a trained network on hand-made histories (diagnostic).  usage: probe_q.py <seed_dir> <k_local>"""
import sys, os
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np, jax, jax.numpy as jnp
import model as M
sd, K = sys.argv[1], int(sys.argv[2])
z = np.load(os.path.join(sd, "params.npz"))
def tree(z):
    out = {}
    for k in z.files:
        cur = out
        parts = k.split("/")
        for p in parts[:-1]: cur = cur.setdefault(p, {})
        cur[parts[-1]] = jnp.asarray(z[k])
    return out
params = tree(z)
net = M.make_net(K + 1)
def x(a, ack): return jnp.concatenate([jax.nn.one_hot(a, K + 1), jnp.ones(K), jnp.array([ack])])
def run(hist):
    seq = jnp.stack([x(a, k) for a, k in hist])[None]
    q, _ = M.unroll(net, params, seq, M.zero_carry(net, (1,)))
    return np.asarray(q[0, -1]).round(3)
print("t=1 (silent,ack0):", run([(0, 0.)]))
for t in (5, 20):
    print(f"after {t} slots on ch1 with ack=1:", run([(0, 0.)] + [(1, 1.)] * t))
    print(f"after {t} slots on ch1 with ack=0:", run([(0, 0.)] + [(1, 0.)] * t))
    if K > 1:
        print(f"after {t} slots on ch2 with ack=1:", run([(0, 0.)] + [(2, 1.)] * t))
    print(f"after {t} slots silent:", run([(0, 0.)] * (t + 1)))

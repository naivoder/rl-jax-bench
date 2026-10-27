"""Differential test 2, OUR side: n-step return / bootstrap discount / bootstrap observation computed by OUR replay (src/replay.py,
gather_batch) on a synthetic transition stream with terminations AND truncations (main venv).  Output: nstep_stream.npz."""
import sys, os
from rl_jax_bench import gpu_bootstrap; os.environ.setdefault('JAX_PLATFORMS', 'cpu'); gpu_bootstrap.setup()
PAPER = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, PAPER)
import numpy as np, jax, jax.numpy as jnp
from src import replay as R
rng = np.random.default_rng(20261027)
T, N, G = 3000, 10, 0.99
rew = rng.choice([0.0, 1.0, -1.0, 0.5, np.log(2.0)], T).astype(np.float32)
term = np.zeros(T, bool); trunc = np.zeros(T, bool)
t = 0
while t < T:
    L = int(rng.integers(1, 40)); t += L
    if t - 1 < T:
        if rng.random() < 0.6: term[t - 1] = True
        else: trunc[t - 1] = True
end = term | trunc
L = R.make_layout(0, 4096, 4, n_step=N, gamma=G)
hp = dict(demo_on=0.0, eps_d=1.0, eps_a=0.001, rbs=0.0)
buf = R.init_buffer(L, None, hp)
add = jax.jit(lambda b, o, no, a, r, te, en: R.add_transition(L, b, o, no, a, r, te, en))
ids = lambda i: jnp.asarray(np.frombuffer(np.int32(i).tobytes(), np.uint8))
for i in range(T):
    buf = add(buf, ids(i), ids(100000 + i), jnp.int32(0), jnp.float32(rew[i]), jnp.bool_(term[i]), jnp.bool_(end[i]))
idx = jnp.arange(T)
out = jax.jit(lambda b, i: R.gather_batch(L, b, i))(buf, idx)
nobs_id = np.frombuffer(np.asarray(out['nobsn']).tobytes(), np.int32) - 100000
active = np.asarray(buf['pa'][:T]) > 0
np.savez('nstep_stream.npz', rew=rew, term=term, trunc=trunc, G=np.asarray(out['Gn']), disc=np.asarray(out['disc_n']), last=nobs_id, m=np.asarray(out['m']), active=active)
print('ours done; active', active.sum(), 'of', T, 'episodes', end.sum())

"""Train/eval PRNG streams are disjoint: a block (iterations + evaluation) leaves exactly the same training state as
the same iterations without the evaluation."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np
import jax

from src import train as T


def test_eval_does_not_perturb_training():
    cfg = T.Config(**dict(T.ARMS['a2d_compact'], env='TD', iters=2, batch=64, buffer=128, val_epochs=1, n_mb=4,
                          il_mb=16, eval_every=2))
    tr = T.Trainer(cfg)
    rk = jax.random.PRNGKey(3)
    ts0 = tr.init(rk)
    ts_blk, _ = jax.jit(lambda t: tr.block(t, rk, 2))(ts0)

    def two(t):
        for _ in range(2):
            t, _ = tr.iteration(t, jax.random.fold_in(jax.random.fold_in(rk, 1), t.it))
        return t
    ts_it = jax.jit(two)(ts0)
    for a, b in zip(jax.tree_util.tree_leaves(ts_blk.params), jax.tree_util.tree_leaves(ts_it.params)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)

"""MinAtar parity check: gymnax <Game>-MinAtar + our sticky wrapper + lifted time limit vs MinAtar
1.0.15 (official venv rollouts, scripts/official/minatar_rollouts.py), uniform-random and always-0 policies, N episodes
each. Compares mean return (95 % CI overlap via Welch t), KS on returns (p > 0.01), mean length, reward per step.
Usage: python scripts/minatar_parity.py <official.npz[,official2.npz]> [N] [cap]   -> runs/small/minatar_parity.json"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
sys.path.insert(0, REPO)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from rl_jax_bench.jaxlog import check_device, metric, write_json  # noqa: E402

from src import data as dt  # noqa: E402

ref = {}
for f in sys.argv[1].split(','):
    ref.update(dict(np.load(f)))
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
MAXS = int(sys.argv[3]) if len(sys.argv) > 3 else 5000   # same cap as the reference rollouts
check_device()
out = {}
for gi, game in enumerate(dt.MINATAR_GAMES):
    env = dt.minatar_env(game)
    for pol in ("random", "fixed0"):
        def episode(key):
            k0, k1 = jax.random.split(key)
            st, _ = env.reset(k0)

            def cond(c):
                return jnp.logical_and(~c[3], c[5] < MAXS)

            def body(c):
                st, last, ret, done, k, n = c
                k, ka, ks, ke = jax.random.split(k, 4)
                a = jax.random.randint(ka, (), 0, env.n_actions) if pol == "random" else jnp.int32(0)
                ae = dt.sticky(ks, last, a)
                st, _, r, d = env.step(ke, st, ae)
                return st, ae, ret + r, d, k, n + 1

            c = jax.lax.while_loop(cond, body, (st, jnp.int32(0), jnp.float32(0), jnp.asarray(False), k1, jnp.int32(0)))
            return c[2], c[5]

        R, L = jax.jit(jax.vmap(episode))(jax.random.split(jax.random.PRNGKey(10 + gi), N))
        R, L = np.asarray(R, float), np.asarray(L, float)
        r0, l0 = ref[f"{game}/{pol}/ret"], ref[f"{game}/{pol}/len"]
        ks = stats.ks_2samp(R, r0)
        w = stats.ttest_ind(R, r0, equal_var=False)
        wl = stats.ttest_ind(L, l0, equal_var=False)
        rec = dict(ours_ret=[R.mean(), R.std(ddof=1) / np.sqrt(N)], ref_ret=[r0.mean(), r0.std(ddof=1) / np.sqrt(len(r0))],
                   ours_len=L.mean(), ref_len=l0.mean(), ours_rps=R.sum() / L.sum(), ref_rps=r0.sum() / l0.sum(),
                   ks_p=float(ks.pvalue), welch_p_ret=float(w.pvalue), welch_p_len=float(wl.pvalue),
                   ok=bool(ks.pvalue > 0.01 and w.pvalue > 0.01))
        out[f"{game}/{pol}"] = rec
        metric("parity", game=game, policy=pol, ours=round(R.mean(), 3), ref=round(r0.mean(), 3),
               ours_len=round(L.mean(), 1), ref_len=round(l0.mean(), 1), ks_p=round(ks.pvalue, 4),
               welch_p=round(w.pvalue, 4), ok=rec["ok"])
write_json(os.path.join(REPO, "runs", "small", "minatar_parity.json"), out)

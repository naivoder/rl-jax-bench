"""Trained-regime whole-iteration parity.

Fixture tests/data/official_trained_td2_lam05.npz: the OFFICIAL code's complete state of a TD2 main-args (lambda 0.5,
official seed 1) run after iteration 20 (nets, Adam moments, FIFO buffer, beta, damping), plus iterations 21-24 with
every official draw recorded (rollout batch, value-epoch permutations, replay-buffer permutations, Fisher subsample)
and the official parameters after each iteration (recorded by scripts/official/record_trained_state.py, trimmed
losslessly to uint8 / int16). Each iteration is replayed through OUR Trainer.iteration from the official inputs.

At iteration 24 (index 3) 157 / 2000 taken actions have expert probability < 2.22e-16. The official
Categorical.log_prob clamps those (zero gradient), so with Config.loglik_clamp=True every net must match to <= 1e-6;
with the pre-fix exact log_softmax (loglik_clamp=False) the expert step diverges (~1e-2). Iterations 21-23 have no
clamped sample and match either way. The earlier fixtures (fresh nets) contained no p < eps sample at all."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np        # noqa: E402
import jax                # noqa: E402
import jax.numpy as jnp   # noqa: E402
import optax              # noqa: E402
import pytest             # noqa: E402
jax.config.update('jax_enable_x64', True)
from src import train as TR      # noqa: E402
from src import data as D        # noqa: E402
from src import algorithm as AL  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), 'data', 'official_trained_td2_lam05.npz')
NETS = ('expert_pol', 'expert_val', 'learner_pol', 'learner_val')


def _replay(monkeypatch, clamp: bool):
    Z = np.load(FIX)

    def mlp(prefix):
        return {'params': {'head': {f'Dense_{i}': {'kernel': jnp.asarray(Z[f'{prefix}/head.{li}.weight'].T),
                                                     'bias': jnp.asarray(Z[f'{prefix}/head.{li}.bias'])}
                                    for i, li in enumerate([0, 2, 4])}}}

    def adam_state(name):
        def tr_(sfx):
            return {'params': {'head': {f'Dense_{i}': {
                'kernel': jnp.asarray(Z[f'pre/opt/{name}/head.{li}.weight/{sfx}'].T),
                'bias': jnp.asarray(Z[f'pre/opt/{name}/head.{li}.bias/{sfx}'])} for i, li in enumerate([0, 2, 4])}}}
        return (optax.ScaleByAdamState(count=jnp.asarray(int(Z[f'pre/opt/{name}/step']), jnp.int32), mu=tr_('m'),
                                       nu=tr_('v')), optax.EmptyState())

    cfg = TR.Config(env='TD2', method='a2d', obs='partial', lam_a2d=0.5, lam_rl=0.5, dtype='float64',
                    kl_fallback=True, loglik_clamp=clamp)
    tr = TR.Trainer(cfg)
    spec = tr.spec
    ts0 = tr.init(jax.random.PRNGKey(0))

    def decode(S, O):
        S = np.asarray(S, np.float64)
        pos = S[:, :spec.n_pos].argmax(1)
        h = S[:, spec.n_pos:spec.n_pos + 2].argmax(1)
        il = S[:, -1] > 0.5
        lat = D.Latent(jnp.asarray(pos, jnp.int32), jnp.asarray(h, jnp.int32), jnp.asarray(il))
        np.testing.assert_array_equal(np.asarray(jax.vmap(lambda l: D.state_vec(spec, l, jnp.float64))(lat)), S)
        np.testing.assert_array_equal(np.asarray(jax.vmap(lambda l: D.partial_vec(spec, l, jnp.float64))(lat)),
                                      np.asarray(O, np.float64).reshape(len(S), -1))
        return lat

    L = int(Z['pre/buf/active_length'])
    blat = decode(Z['pre/buf/states'][:L], Z['pre/buf/obs'][:L])
    buf = TR.Buffer(blat.pos, blat.h, blat.illum, jnp.zeros((L, 2), jnp.uint32),
                    jnp.int32(int(Z['pre/buf/counter']) % cfg.buffer), jnp.int32(L))
    ts = ts0._replace(params={k: mlp(f'pre/{k}') for k in NETS},
                      opt={k: adam_state(k) for k in ('learner_pol', 'learner_val', 'expert_val')},
                      damping=jnp.float64(float(Z['pre/damping'])), buf=buf)
    st = {}

    def load(j):
        k = f'it{j}'
        st['vq'] = [np.asarray(x, np.int32) for x in Z[k + '/vq']]
        st['sel'] = [np.asarray(x, np.int32) for x in Z[k + '/sel']]
        st['rb'] = [np.asarray(Z[f'{k}/rb{q}'], np.int32) for q in range(int(Z[k + '/rb_n']))]
        lat = decode(Z[k + '/batch/S'], Z[k + '/batch/O'])
        n = len(Z[k + '/batch/S'])
        st['batch'] = (lat, jnp.zeros((n, 2), jnp.uint32), jnp.asarray(Z[k + '/batch/A'], jnp.int32),
                       jnp.asarray(Z[k + '/batch/R'], jnp.float64), jnp.asarray(Z[k + '/batch/ND'] == 1))
        st['n_vq'] = st['n_rb'] = st['n_sel'] = 0

    def perms(key, n_ep, N):
        out = np.stack(st['vq'][st['n_vq']:st['n_vq'] + n_ep]); st['n_vq'] += n_ep
        return jnp.asarray(out)

    def masked_perm(key, Ln, cap):
        p = st['rb'][st['n_rb']]; st['n_rb'] += 1
        return jnp.asarray(np.concatenate([p, np.arange(len(p), cap)]).astype(np.int32))

    def choice(key, a, shape=(), replace=True, p=None, axis=0):
        s = st['sel'][st['n_sel']]; st['n_sel'] += 1
        return jnp.asarray(s)

    tr.rollout = lambda params, beta, key, T, actor: st['batch']
    tr._perms = perms
    monkeypatch.setattr(AL, 'masked_perm', masked_perm)
    monkeypatch.setattr(jax.random, 'choice', choice)
    diffs, n_clamped = [], []
    n = sum(1 for k in Z.files if k.endswith('/proj_loss'))
    for j in range(n):
        ts = ts._replace(beta=jnp.float32(float(Z[f'it{j}/beta'])))
        load(j)
        ts, info = tr.iteration(ts, jax.random.PRNGKey(j))
        assert st['n_vq'] == len(st['vq']) and st['n_sel'] == len(st['sel']) and st['n_rb'] <= len(st['rb'])
        d = {}
        for nm in NETS:
            o = mlp(f'it{j}/{nm}')
            d[nm] = max(float(np.abs(np.asarray(a) - np.asarray(b)).max())
                        for a, b in zip(jax.tree_util.tree_leaves(ts.params[nm]), jax.tree_util.tree_leaves(o)))
        diffs.append(d)
        n_clamped.append(int(info['n_clamped']))
        ts = ts._replace(params={k: mlp(f'it{j}/{k}') for k in NETS})     # re-sync: each iteration from official inputs
        print(j, n_clamped[-1], {k: f'{v:.1e}' for k, v in d.items()})
    return diffs, n_clamped


@pytest.mark.skipif(not os.path.exists(FIX), reason='fixture missing')
def test_trained_state_parity_with_official_clamp(monkeypatch):
    diffs, n_clamped = _replay(monkeypatch, clamp=True)
    assert n_clamped[3] == 157 and n_clamped[:3] == [0, 0, 0]
    for d in diffs:
        assert max(d.values()) <= 1e-6, diffs


@pytest.mark.skipif(not os.path.exists(FIX), reason='fixture missing')
def test_trained_state_diverges_without_clamp(monkeypatch):
    diffs, _ = _replay(monkeypatch, clamp=False)
    for d in diffs[:3]:
        assert max(d.values()) <= 1e-6, diffs
    assert diffs[3]['expert_pol'] > 1e-3, diffs          # the pre-fix estimator is NOT the official one here

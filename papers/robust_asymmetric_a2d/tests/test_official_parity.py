"""Differential tests against the OFFICIAL code (plai-group/a2d @368aaf5), float64.

tests/data/oracle.npz is produced by scripts/official/dump.py running the
official PyTorch code in its own venv on fixed inputs (randomness replayed via monkeypatching, draws saved).
Weights are transferred torch -> flax; every comparison is at <= 1e-5 (most at ~1e-10).
"""
import os
import numpy as np
import pytest

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from src import data as D                      # noqa: E402
from src import model as M                     # noqa: E402
from src import algorithm as AL                # noqa: E402

ORC = np.load(os.path.join(os.path.dirname(__file__), 'data', 'oracle.npz'))


def _mlp_params(prefix, n_in):
    """torch head.{0,2,4}.{weight,bias} -> flax MLP params."""
    p = {}
    for i, li in enumerate([0, 2, 4]):
        p[f'Dense_{i}'] = {'kernel': jnp.asarray(ORC[f'{prefix}/head.{li}.weight'].T),
                           'bias': jnp.asarray(ORC[f'{prefix}/head.{li}.bias'])}
    return {'params': {'head': p}}


# ---------------------------------------------------------------------------------------------- environment
@pytest.mark.parametrize('tag', ['FL', 'TD', 'TD1', 'TD2', 'TD3'])
def test_env_golden_rollout(tag):
    spec = D.get_spec(tag)
    k = f'env/{tag}/'
    ep, hid, x, y = ORC[k + 'ep'], ORC[k + 'hid'], ORC[k + 'x'], ORC[k + 'y']
    A, R, DN, S, PV = ORC[k + 'a'], ORC[k + 'r'], ORC[k + 'done'], ORC[k + 'state'], ORC[k + 'partial']
    tab = jnp.asarray(D.image_table(tag)) if spec.has_image else None
    step = jax.jit(lambda st, a: D.step(spec, st, a))
    img_err, st = 0.0, None
    for i in range(len(ep)):
        if i == 0 or ep[i] != ep[i - 1]:
            st = D.reset(spec, jax.random.PRNGKey(0))._replace(h=jnp.int32(hid[i]))
        assert (int(st.r), int(st.c)) == (y[i] - 1, x[i] - 1)
        lat = D.latent(spec, st)
        np.testing.assert_array_equal(np.asarray(D.state_vec(spec, lat, jnp.float64)), S[i])
        np.testing.assert_array_equal(np.asarray(D.partial_vec(spec, lat, jnp.float64)), PV[i])
        if spec.has_image and i < len(ORC[k + 'img']):
            im = np.asarray(D.image(spec, tab, lat, None, noise=False, dtype=jnp.float64))
            img_err = max(img_err, float(np.abs(im - ORC[k + 'img'][i] / 255.0).max()))
        st, r, dn = step(st, int(A[i]))
        assert float(r) == R[i] and int(dn) == DN[i]
    # noiseless images equal up to the float16 storage quantization (both sides store half precision)
    assert img_err < 5e-4


# ---------------------------------------------------------------------------------------------- algorithm pieces
def _batch():
    return (jnp.asarray(ORC['batch/S']), jnp.asarray(ORC['batch/O']), jnp.asarray(ORC['batch/A']),
            jnp.asarray(ORC['batch/R']), jnp.asarray(ORC['batch/ND']).astype(bool))


def test_gae_matches_official():
    S, O, A, R, ND = _batch()
    vnet = M.ValueNet(kind='flat', out_gain=0.01)
    vp = _mlp_params('vnet', 34)
    V = vnet.apply(vp, S)
    np.testing.assert_allclose(V, ORC['gae/V'], atol=1e-10)
    adv, ret = AL.gae(R, V, ND, 0.995, 0.9)
    np.testing.assert_allclose(adv, ORC['gae/adv'], atol=1e-8)
    np.testing.assert_allclose(ret, ORC['gae/ret'], atol=1e-8)


def test_value_loss_gae_signed_max():
    S, O, A, R, ND = _batch()
    V = jnp.asarray(ORC['gae/V'])
    adv = jnp.asarray(ORC['gae/adv'])
    loss = AL.value_loss_gae(jnp.asarray(ORC['vloss/vs']), V, adv, ND, jnp.ones_like(V), 0.2)
    np.testing.assert_allclose(loss, ORC['vloss/loss'], rtol=1e-10)


def test_importance_weights():
    S, O, A, R, ND = _batch()
    pol = M.PolicyNet(kind='flat', out_gain=0.01)
    e_lp = pol.apply(_mlp_params('expert', 34), S)
    l_lp = pol.apply(_mlp_params('learner', 25), O)
    w = AL.importance_weight(jnp.take_along_axis(e_lp, A[:, None], 1)[:, 0],
                             jnp.take_along_axis(l_lp, A[:, None], 1)[:, 0], float(ORC['iw/beta']))
    np.testing.assert_allclose(w, ORC['iw/w'], rtol=1e-10)


@pytest.mark.parametrize('tag,ent,reg', [('main', 10.0, True), ('qargs', 0.02, False)])
def test_trpo_surrogate_and_grad(tag, ent, reg):
    S, O, A, R, ND = _batch()
    pol = M.PolicyNet(kind='flat', out_gain=0.01)
    pe = _mlp_params('expert', 34)
    old = jnp.asarray(ORC['surr/old_lp']) + jnp.asarray(ORC['surr2/pert'])
    adv_in = jnp.asarray(ORC['surr/adv_in'])
    f = lambda p: AL.surrogate(pol.apply(p, S), A, old, adv_in, ent, reg)
    val, g = jax.value_and_grad(f)(pe)
    np.testing.assert_allclose(val, ORC[f'surr2/{tag}/value'], rtol=1e-9)
    for i, li in enumerate([0, 2, 4]):
        np.testing.assert_allclose(g['params']['head'][f'Dense_{i}']['kernel'],
                                   ORC[f'surr2/{tag}/grad/head.{li}.weight'].T, atol=1e-9)
        np.testing.assert_allclose(g['params']['head'][f'Dense_{i}']['bias'],
                                   ORC[f'surr2/{tag}/grad/head.{li}.bias'], atol=1e-9)


def test_ail_kl_loss_and_grad():
    S, O, A, R, ND = _batch()
    pol = M.PolicyNet(kind='flat', out_gain=0.01)
    e_lp = pol.apply(_mlp_params('expert', 34), S)
    f = lambda p: AL.kl_loss(e_lp, pol.apply(p, O))
    val, g = jax.value_and_grad(f)(_mlp_params('learner', 25))
    np.testing.assert_allclose(val, ORC['kl/value'], rtol=1e-10)
    for i, li in enumerate([0, 2, 4]):
        np.testing.assert_allclose(g['params']['head'][f'Dense_{i}']['kernel'], ORC[f'kl/grad/head.{li}.weight'].T,
                                   atol=1e-10)


def _cmp_params(p, prefix, atol):
    for i, li in enumerate([0, 2, 4]):
        np.testing.assert_allclose(p['params']['head'][f'Dense_{i}']['kernel'], ORC[f'{prefix}/head.{li}.weight'].T,
                                   atol=atol)
        np.testing.assert_allclose(p['params']['head'][f'Dense_{i}']['bias'], ORC[f'{prefix}/head.{li}.bias'],
                                   atol=atol)


def test_full_trpo_step():
    S, O, A, R, ND = _batch()
    pol = M.PolicyNet(kind='flat', out_gain=0.01)
    pe = _mlp_params('expert', 34)
    sel = jnp.asarray(ORC['trpo/sel'])
    cfg = AL.TRPOCfg(max_kl=0.01, ent=10.0, reg_adv=True)
    new, info = AL.trpo_step(lambda p, x: pol.apply(p, x), pe, S, A, jnp.asarray(ORC['surr/old_lp']),
                             jnp.asarray(ORC['surr/adv_in']), cfg, sel, jnp.float64(0.1))
    _cmp_params(new, 'trpo/new', 1e-7)


def test_full_value_step():
    S, O, A, R, ND = _batch()
    vnet = M.ValueNet(kind='flat', out_gain=0.01)
    vp = _mlp_params('vnet', 34)
    perms = jnp.asarray(ORC['vstep/perms'])
    adv, ret = jnp.asarray(ORC['gae/adv']), jnp.asarray(ORC['gae/ret'])
    opt = AL.adam(7e-4)
    new, _, _ = AL.value_update(lambda p, x: vnet.apply(p, x), vp, opt.init(vp), opt, S, adv, ret, ND,
                                'gae', perms, 32, 0.2)
    _cmp_params(new, 'vstep/new', 1e-7)


def test_full_q_step():
    S, O, A, R, ND = _batch()
    qnet = M.ValueNet(kind='flat', out_gain=1.0)
    qp = _mlp_params('qnet', 29)
    perms = jnp.asarray(ORC['qstep/perms'])
    ret = jnp.asarray(ORC['gae/ret'])
    opt = AL.adam(3e-4)
    X = jnp.concatenate([O, jax.nn.one_hot(A, 4, dtype=O.dtype)], -1)
    new, _, _ = AL.q_update(lambda p, x: qnet.apply(p, x), qp, opt.init(qp), opt, X, ret, perms, 32)
    _cmp_params(new, 'qstep/new', 1e-7)


def test_projection_epochs_official_sampler():
    """Two AIL epochs through the official epoch sampler (discarded wrap-around batch) on 300 buffered pairs."""
    S, O, A, R, ND = _batch()
    NB = int(ORC['proj/NB'])
    pol = M.PolicyNet(kind='flat', out_gain=0.01)
    e_lp = pol.apply(_mlp_params('expert', 34), S[:NB])
    perms = jnp.asarray(ORC['proj/perms'])
    opt = AL.adam(3e-4)
    lp = _mlp_params('learner', 25)
    idx, valid = AL.epoch_indices(perms[0], perms[1], jnp.int32(NB), 64, max_batches=AL.n_proj_batches(NB, 64))
    losses = []
    st = opt.init(lp)
    for e in range(2):
        def loss_fn(p, ii, k):
            return AL.kl_loss(e_lp[ii], pol.apply(p, O[:NB][ii]))
        lp, st, l = AL.projection_epoch(loss_fn, lp, st, opt, idx[e], valid[e])
        losses.append(float(l))
    np.testing.assert_allclose(losses, ORC['proj/losses'], rtol=1e-8)
    _cmp_params(lp, 'proj/new', 1e-7)


def _img_params(prefix):
    convs = {}
    for i in range(4):
        w = ORC[f'{prefix}/encoder.convs.{i}.weight'].astype(np.float64)        # (O, I, kh, kw)
        convs[f'Conv_{i}'] = {'kernel': jnp.asarray(w.transpose(2, 3, 1, 0)),
                              'bias': jnp.asarray(ORC[f'{prefix}/encoder.convs.{i}.bias'].astype(np.float64))}
    fw = ORC[f'{prefix}/encoder.fc.weight'].astype(np.float64)                  # (64, C*X*Y)
    fw = fw.reshape(64, 32, 14, 14).transpose(2, 3, 1, 0).reshape(-1, 64)
    enc = dict(convs)
    enc['Dense_0'] = {'kernel': jnp.asarray(fw), 'bias': jnp.asarray(ORC[f'{prefix}/encoder.fc.bias'].astype(np.float64))}
    enc['LayerNorm_0'] = {'scale': jnp.asarray(ORC[f'{prefix}/encoder.ln.weight'].astype(np.float64)),
                          'bias': jnp.asarray(ORC[f'{prefix}/encoder.ln.bias'].astype(np.float64))}
    head = {}
    for i, li in enumerate([0, 2, 4]):
        head[f'Dense_{i}'] = {'kernel': jnp.asarray(ORC[f'{prefix}/head.{li}.weight'].astype(np.float64).T),
                              'bias': jnp.asarray(ORC[f'{prefix}/head.{li}.bias'].astype(np.float64))}
    return {'params': {'encoder': enc, 'head': head}}


def test_image_networks_forward():
    imgs = jnp.asarray(ORC['inet/imgs'].astype(np.float64))
    x = jnp.transpose(imgs, (0, 2, 1, 3))            # the official conv sees render.T: (x, y, C) order
    pol = M.PolicyNet(kind='image', out_gain=1.0)
    val = M.ValueNet(kind='image', out_gain=1.0)
    np.testing.assert_allclose(pol.apply(_img_params('ipol'), x), ORC['inet/logp'], atol=1e-5)
    np.testing.assert_allclose(val.apply(_img_params('ival'), x), ORC['inet/v'], atol=1e-5)


def test_augmentation_crop_resize():
    """Official RandCrop(pad 5 %, scale) + Speckle + bilinear Resize with the noise switched off and fixed draws."""
    x = jnp.asarray(ORC['aug/in'][:, 0].astype(np.float64))      # (B, W, H, C) in the official buffer layout
    cf = jnp.asarray(ORC['aug/cf'].astype(np.float64))
    orig = jnp.asarray(ORC['aug/orig'])
    out = AL.crop_resize(x, cf, orig, jnp.zeros((x.shape[0], 46, 46, 3)))
    np.testing.assert_allclose(out, ORC['aug/out'][:, 0], atol=1e-5)


# ================================================================================================================
# WHOLE official training iterations replayed through OUR Trainer.iteration.
# tests/data/official_iter.npz is produced by scripts/official/dump_iter.py
# (+ compact_fixture.py): the official AdpAsymDagger.step() runs unmodified; its rollouts and every random draw
# (value/Q minibatch shuffles, TRPO Fisher subsample, replay-buffer epoch orders) are RECORDED and replayed here.
# This checks the wiring the piecewise tests above cannot: which critic feeds which advantage, pre/post-update
# ordering, the IS weight / Q-advantage placement, beta handling, buffer FIFO wrap, AIL pre-training, RL arms.
# ================================================================================================================
from src import train as TR                    # noqa: E402

IT = np.load(os.path.join(os.path.dirname(__file__), 'data', 'official_iter.npz'))
NETS = ('expert_pol', 'expert_val', 'learner_pol', 'learner_val', 'expert_q', 'learner_q')

CASES = {
    'a2d_fl': ('FL', dict(method='a2d', obs='partial', lam_a2d=0.9), 3),
    'a2d_td_b0': ('TD', dict(method='a2d', obs='partial', beta_decay=0.0), 2),
    'a2dq_td1': ('TD1', TR.ARMS['a2d_q'], 2),
    'a2dq_td1_mix': ('TD1', dict(TR.ARMS['a2d_q'], beta_decay=0.8), 2),
    'rlmdp_td': ('TD', TR.ARMS['rl_mdp'], 2),
    'rlpo_fl': ('FL', TR.ARMS['rl_compact'], 2),
    'arl_td': ('TD', dict(method='rl', obs='partial', asym_value=True), 2),
    'ail_fl': ('FL', TR.ARMS['ail_compact'], 2),
}


def _torch_mlp(prefix):
    p = {}
    for i, li in enumerate([0, 2, 4]):
        p[f'Dense_{i}'] = {'kernel': jnp.asarray(IT[f'{prefix}/head.{li}.weight'].T),
                           'bias': jnp.asarray(IT[f'{prefix}/head.{li}.bias'])}
    return {'params': {'head': p}}


def _decode_lat(spec, S, O, obs_is_state=False):
    """Official state vector -> our Latent (pos, hidden, illuminated); asserts both encoders agree exactly.
    obs_is_state: RL(MDP) runs with render_type 'state', so the official stored observation IS the state vector."""
    S = np.asarray(S, np.float64)
    pos = S[:, :spec.n_pos].argmax(1)
    h = S[:, spec.n_pos:spec.n_pos + (9 if spec.kind == 'fl' else 2)].argmax(1)
    illum = S[:, -1] > 0.5 if spec.kind == 'td' else np.zeros(len(S), bool)
    lat = D.Latent(jnp.asarray(pos, jnp.int32), jnp.asarray(h, jnp.int32), jnp.asarray(illum))
    np.testing.assert_array_equal(np.asarray(jax.vmap(lambda l: D.state_vec(spec, l))(lat)), S)
    obs_f = D.state_vec if obs_is_state else D.partial_vec
    np.testing.assert_array_equal(np.asarray(jax.vmap(lambda l: obs_f(spec, l))(lat)), O)
    return lat


def _batch_of(spec, key, obs_is_state=False):
    S, O = IT[key + '/S'], IT[key + '/O']
    lat = _decode_lat(spec, S, O, obs_is_state)
    n = len(S)
    return (lat, jnp.zeros((n, 2), jnp.uint32), jnp.asarray(IT[key + '/A'], jnp.int32),
            jnp.asarray(IT[key + '/R'], jnp.float64), jnp.asarray(IT[key + '/ND'] == 1))


def _cmp_net(ours, prefix, atol):
    worst = 0.0
    for i, li in enumerate([0, 2, 4]):
        for ours_x, theirs in [(ours['params']['head'][f'Dense_{i}']['kernel'], IT[f'{prefix}/head.{li}.weight'].T),
                               (ours['params']['head'][f'Dense_{i}']['bias'], IT[f'{prefix}/head.{li}.bias'])]:
            worst = max(worst, float(np.abs(np.asarray(ours_x) - theirs).max()))
    assert worst <= atol, f'{prefix}: max |ours - official| = {worst:.3e} > {atol:.0e}'
    return worst


class _Replay:
    """Feeds the recorded official draws to our code, in the order our code consumes them, and checks that order."""

    def __init__(self, case, spec, obs_is_state=False):
        self.case, self.spec, self.obs_is_state = case, spec, obs_is_state

    def load(self, tag):
        k = f'{self.case}/{tag}'
        self.vq = [np.asarray(x, np.int32) for x in IT[k + '/vq']] if (k + '/vq') in IT.files else []
        self.vq_tag = list(IT[k + '/vq_tag']) if (k + '/vq_tag') in IT.files else []
        self.sel = [np.asarray(x, np.int32) for x in IT[k + '/sel']] if (k + '/sel') in IT.files else []
        self.rb = [np.asarray(IT[f'{k}/rb{j}'], np.int32) for j in range(int(IT[k + '/rb_n']))] \
            if (k + '/rb_n') in IT.files else []
        self.batch = _batch_of(self.spec, k + '/batch', self.obs_is_state)
        self.n_vq = self.n_rb = self.n_sel = 0

    def rollout(self, params, beta, key, T, actor):
        assert T == len(self.batch[2])
        return self.batch

    def perms(self, key, n_ep, N):
        out = np.stack(self.vq[self.n_vq:self.n_vq + n_ep])
        assert len(set(self.vq_tag[self.n_vq:self.n_vq + n_ep])) == 1          # one whole value_step OR q_step
        self.n_vq += n_ep
        assert out.shape == (n_ep, N)
        return jnp.asarray(out)

    def masked_perm(self, key, L, cap):
        p = self.rb[self.n_rb]
        self.n_rb += 1
        assert len(p) == int(L)
        return jnp.asarray(np.concatenate([p, np.arange(len(p), cap)]).astype(np.int32))

    def choice(self, key, a, shape=(), replace=True, p=None, axis=0):
        s = self.sel[self.n_sel]
        self.n_sel += 1
        assert s.shape == tuple(shape) and not replace
        return jnp.asarray(s)


def _trainer_and_state(case, env, arm):
    cfg = TR.Config(env=env, **arm)
    tr = TR.Trainer(cfg)
    ts = tr.init(jax.random.PRNGKey(0))
    params = dict(ts.params)
    for nm in NETS:
        if f'{case}/init/{nm}/head.0.weight' in IT.files:
            params[nm] = _torch_mlp(f'{case}/init/{nm}')
    opt = {'learner_pol': tr.opt_il.init(params['learner_pol']), 'learner_val': tr.opt_v.init(params['learner_val']),
           'expert_val': tr.opt_v.init(params['expert_val'])}
    if cfg.use_q:
        opt['expert_q'] = tr.opt_q.init(params['expert_q'])
        opt['learner_q'] = tr.opt_q.init(params['learner_q'])
    ts = ts._replace(params=params, opt=opt, damping=jnp.float64(0.1),
                     beta=jnp.float32(float(IT[f'{case}/it0/beta'])))
    return cfg, tr, ts


ITER_ATOL = 1e-6


@pytest.mark.parametrize('case', list(CASES))
def test_whole_iterations_match_official(case, monkeypatch):
    env, arm, n_it = CASES[case]
    spec = D.get_spec(env)
    cfg, tr, ts = _trainer_and_state(case, env, arm)
    rp = _Replay(case, spec, obs_is_state=cfg.method == 'rl' and cfg.rl_actor == 'expert')
    monkeypatch.setattr(tr, 'rollout', rp.rollout)
    monkeypatch.setattr(tr, '_perms', rp.perms)
    monkeypatch.setattr(AL, 'masked_perm', rp.masked_perm)
    monkeypatch.setattr(jax.random, 'choice', rp.choice)
    if cfg.method == 'ail':
        rp.load('pre')
        ts = tr.ail_pretrain(ts, jax.random.PRNGKey(0))
        assert rp.n_rb == 2, 'official pre-training runs max_proj_iter = 2 epochs'
        _cmp_net(ts.params['learner_pol'], f'{case}/pre/learner_pol', ITER_ATOL)
    for i in range(n_it):
        assert abs(float(ts.beta) - float(IT[f'{case}/it{i}/beta'])) < 1e-6, 'beta schedule'
        rp.load(f'it{i}')
        ts, info = tr.iteration(ts, jax.random.PRNGKey(i))
        assert rp.n_vq == len(rp.vq) and rp.n_sel == len(rp.sel), 'every official draw consumed, in order'
        if cfg.method != 'rl':
            assert rp.n_rb == 2                                                # push order + epoch-2 order
            np.testing.assert_allclose(float(ts.proj_loss), float(IT[f'{case}/it{i}/proj_loss']), rtol=1e-6)
        for nm in NETS:
            if f'{case}/it{i}/{nm}/head.0.weight' in IT.files:
                _cmp_net(ts.params[nm], f'{case}/it{i}/{nm}', ITER_ATOL)


# ---------------------------------------------------------------------------------------------- evaluation metric
@pytest.mark.parametrize('case,env', [('eval_fl', 'FL'), ('eval_td', 'TD')])
def test_eval_metric_on_identical_rollouts(case, env, monkeypatch):
    """Official logging() argmax evaluation (complete episodes, every episode started before step 2000 counted) vs our
    make_evaluate on the SAME rollouts: deterministic policies, and our resets forced to the official per-episode
    hidden draws (decoded from the recorded official trajectory)."""
    from src import eval as EV
    spec = D.get_spec(env)
    lanes = (('expert', True), ('learner', True))
    pol_e, pol_l = M.make_policy('flat'), M.make_policy('flat')
    params = {'expert_pol': _torch_mlp(f'{case}/init/expert_pol'), 'learner_pol': _torch_mlp(f'{case}/init/learner_pol')}
    T = EV.EVAL_STEPS + D.MAX_STEPS
    key = jax.random.PRNGKey(5)
    tab_k, tab_h = [], []
    for li, (lane, _) in enumerate(lanes):
        S, ND = IT[f'{case}/{lane}/S'], IT[f'{case}/{lane}/ND']
        lat = _decode_lat(spec, S, S[:, :spec.n_pos] if spec.kind == 'fl' else
                          np.asarray(jax.vmap(lambda l: D.partial_vec(spec, l))(
                              D.Latent(jnp.asarray(S[:, :spec.n_pos].argmax(1), jnp.int32),
                                       jnp.asarray(S[:, spec.n_pos:spec.n_pos + 2].argmax(1), jnp.int32),
                                       jnp.asarray(S[:, -1] > 0.5)))))
        starts = np.concatenate([[0], np.nonzero(ND == 0)[0] + 1])
        starts = starts[starts < len(ND)]
        k_reset = jax.random.fold_in(jax.random.fold_in(key, li), 0)
        hs = np.zeros(T + 1, np.int32)
        hs[starts] = np.asarray(lat.h)[starts]
        tab_k.append(jax.vmap(lambda t: jax.random.fold_in(k_reset, t))(jnp.arange(T + 1)))
        tab_h.append(hs)
    tab_k, tab_h = jnp.concatenate(tab_k), jnp.asarray(np.concatenate(tab_h))
    real_reset = D.reset

    def forced_reset(sp, k):
        st = real_reset(sp, k)
        hit = jnp.all(tab_k == k[None], -1)
        return st._replace(h=tab_h[jnp.argmax(hit)])
    monkeypatch.setattr(D, 'reset', forced_reset)
    evaluate = EV.make_evaluate(spec, D.obs_fn(spec, 'partial'), lambda l: D.state_vec(spec, l), pol_e, pol_l, lanes,
                                need_kl=True)
    ev = evaluate(params, key)
    for li, (lane, _) in enumerate(lanes):
        ND = IT[f'{case}/{lane}/ND']
        n_ep_official = int((ND == 0).sum())
        assert int(ev['n_ep'][li]) == n_ep_official, (lane, int(ev['n_ep'][li]), n_ep_official)
        np.testing.assert_allclose(float(ev['ret'][li]), float(IT[f'{case}/{lane}/ret']), rtol=1e-12, atol=1e-9)
        assert abs(float(ev['ep_len'][li]) * n_ep_official - len(ND)) < 1e-6        # same total step count


@pytest.mark.parametrize('case,env', [('eval_fl', 'FL'), ('eval_td', 'TD')])
def test_stochastic_eval_metric_same_policy(case, env):
    """The plotted (stochastic) column: 30 official evaluations vs 30 of ours for the SAME trained official expert;
    means must agree within 4 combined standard errors (different random streams, so a statistical check)."""
    from src import eval as EV
    spec = D.get_spec(env)
    pol = M.make_policy('flat')
    params = {'expert_pol': _torch_mlp(f'{case}/init/expert_pol')}
    evaluate = jax.jit(EV.make_evaluate(spec, D.obs_fn(spec, 'partial'), lambda l: D.state_vec(spec, l), pol, None,
                                        (('expert', False),), need_kl=False))
    ours = np.array([float(evaluate(params, jax.random.PRNGKey(100 + i))['ret'][0]) for i in range(30)])
    theirs = IT[f'{case}/expert_sto/rets']
    se = np.sqrt(ours.var(ddof=1) / len(ours) + theirs.var(ddof=1) / len(theirs))
    assert abs(ours.mean() - theirs.mean()) <= 4 * se + 1e-9, (ours.mean(), theirs.mean(), se)

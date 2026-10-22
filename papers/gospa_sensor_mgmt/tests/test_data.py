"""Truth array (Figure 3 / E4) and measurement-generator statistics (paper SS V-A; official sensor.py behavior for CFG-O)."""
import numpy as np
import jax, jax.numpy as jnp
import pytest
from src import data as D
from src import model as M


def test_truth_sha_and_fig3_counts():
    T = M.load_truth()
    assert T.shape == (9, 4, 200)
    tr, al, comp, cm = D.truth_arrays()
    ch = D.count_changes(al)
    births = [k for k, a, b in ch if b > a]
    deaths = [k for k, a, b in ch if b < a]
    assert births == [6, 40, 48, 75, 114, 122, 140, 166]
    assert deaths == [76, 85, 128, 138, 150, 160, 192, 199]
    cnt = al.sum(1)
    assert cnt.max() == 4 and set(np.nonzero(cnt == 4)[0]) == {75, 122, 123, 124, 125, 126, 127}
    assert cm.sum(1).tolist() == cnt.tolist()


def test_truth_sha_mismatch_raises(tmp_path, monkeypatch):
    p = tmp_path / "t.npy"
    np.save(p, np.zeros((9, 4, 200)))
    monkeypatch.setattr(M, "TRUTH_PATH", str(p))
    with pytest.raises(RuntimeError):
        M.load_truth()


def _sample(cfg, lam, n=4000, spos=(0.0, 0.0), truth_pos=None):
    tr, al, comp, cm = D.truth_arrays()
    k = 75                                   # 4 alive targets
    t = jnp.asarray(tr[k]); a = jnp.asarray(al[k])
    if truth_pos is not None:
        t = t.at[:, 0].set(truth_pos[0]).at[:, 2].set(truth_pos[1])
    f = jax.jit(jax.vmap(lambda key: D.gen_measurements(key, t, a, jnp.asarray(spos), lam, cfg)))
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    return f(keys), a


def test_cfgO_statistics():
    cfg = M.CFG_O
    # targets placed at the sensor: detection prob 0.999 (hard model), noise variance R = 3
    (z, v, over), a = _sample(cfg, 0.0, spos=(0.0, 0.0), truth_pos=(0.0, 0.0))
    z, v = np.asarray(z), np.asarray(v)
    nalive = int(np.asarray(a).sum())
    det = v[:, :9].sum(1)
    # noisy measurement is deleted if > 40 m from the sensor -> with sigma = sqrt(3) essentially never at the origin
    assert abs(det.mean() / nalive - 0.999) < 2e-3
    err = z[:, :9][v[:, :9]] - 0.0
    assert abs(err.var() - cfg.R_var) < 0.1
    # clutter: Poisson(lam) in the 80x80 square cropped to the 40 m disc -> mean count lam*pi/4, all within the disc
    (z, v, over), _ = _sample(cfg, 2.0, n=20000, truth_pos=(1e3, 1e3))
    z, v = np.asarray(z), np.asarray(v)
    nc = v[:, 9:].sum(1)
    assert abs(nc.mean() - 2.0 * np.pi / 4) < 0.03
    d = np.sqrt((z[:, 9:] ** 2).sum(-1))[v[:, 9:]]
    assert d.max() <= 40.0 + 1e-9
    assert not np.asarray(over).any()


def test_cfgP_statistics():
    cfg = M.CFG_P
    (z, v, over), a = _sample(cfg, 0.0, n=20000, spos=(0.0, 0.0), truth_pos=(40.0, 0.0))
    v = np.asarray(v)
    pd = 0.999 * np.exp(-0.5)
    nalive = int(np.asarray(a).sum())
    assert abs(v[:, :9].sum(1).mean() / nalive - pd) < 0.01                 # eq. 48 on the TRUE position, no truncation
    (z, v, over), _ = _sample(cfg, 1.0, n=20000, truth_pos=(1e3, 1e3))
    z, v = np.asarray(z), np.asarray(v)
    assert abs(v[:, 9:].sum(1).mean() - 1.0) < 0.03                          # rate lam in the disc
    d = np.sqrt((z[:, 9:] ** 2).sum(-1))[v[:, 9:]]
    assert d.max() <= 40.0 + 1e-9 and abs((d ** 2).mean() - 40.0 ** 2 / 2) < 15.0   # uniform in the disc: E r^2 = R^2/2


def test_common_random_numbers_across_policies_and_clutter():
    cfg = M.CFG_O
    tr, al, comp, cm = D.truth_arrays()
    key = jax.random.PRNGKey(3)
    t, a = jnp.asarray(tr[75]), jnp.asarray(al[75])
    s = jnp.asarray([10.0, 20.0])
    z1, v1, _ = D.gen_measurements(key, t, a, s, 1.0, cfg)
    z2, v2, _ = D.gen_measurements(key, t, a, s, 1.0, cfg)
    assert np.array_equal(np.asarray(z1), np.asarray(z2)) and np.array_equal(np.asarray(v1), np.asarray(v2))
    # target draws do not depend on the clutter rate
    z3, v3, _ = D.gen_measurements(key, t, a, s, 2.0, cfg)
    assert np.array_equal(np.asarray(z1)[:9], np.asarray(z3)[:9]) and np.array_equal(np.asarray(v1)[:9], np.asarray(v3)[:9])

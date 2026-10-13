"""Replay sampler vs the official RAMEfficient_SeqReplayBuffer semantics (sample_weight_baseline 0):
start index uniform over all valid starts (an episode of n transitions has max(1, n-L+1)), masks zero from the
episode end onward, layout obs[0..L], act[0] = rew[0] = term[0] = 0.  Chi-square test on 1e5 draws."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from src import common  # noqa: F401
from src import data as D


def _buf(lengths, H=12, L=7):
    buf = D.init_buffer(len(lengths) + 2, H, 1, 1)
    n = len(lengths)
    obs = np.zeros((n, H + 1, 1), np.float32)
    for e, ln in enumerate(lengths):
        obs[e, : ln + 1, 0] = 100 * e + np.arange(ln + 1)   # encodes (episode, t)
    ep = {"obs": jnp.asarray(obs), "act": jnp.ones((n, H, 1)), "rew": jnp.ones((n, H, 1)), "term": jnp.zeros((n, H, 1)),
          "length": jnp.asarray(lengths, jnp.int32)}
    return D.add_episodes(buf, ep)


def test_start_distribution_chi2():
    lengths, L = [3, 9, 7, 12], 7
    buf = _buf(lengths, L=L)
    b = jax.jit(lambda k: D.sample_sequences(buf, k, 100_000, L))(jax.random.PRNGKey(0))
    first = np.asarray(b["obs"][0, :, 0]).astype(int)
    cells = [(e, s) for e, n in enumerate(lengths) for s in range(max(1, n - L + 1))]
    counts = np.array([np.sum(first == 100 * e + s) for e, s in cells])
    assert counts.sum() == 100_000                                    # nothing outside the valid starts
    chi = stats.chisquare(counts)                                     # uniform over the 11 valid (episode, start)
    assert len(cells) == 11 and chi.pvalue > 1e-3, (counts, chi)


def test_mask_and_layout():
    lengths, L = [3, 9, 7, 12], 7
    buf = _buf(lengths, L=L)
    b = jax.jit(lambda k: D.sample_sequences(buf, k, 512, L))(jax.random.PRNGKey(1))
    obs, m = np.asarray(b["obs"][..., 0]), np.asarray(b["mask"][..., 0])
    assert obs.shape == (L + 1, 512) and m.shape == (L, 512)
    assert np.all(np.asarray(b["act"][0]) == 0) and np.all(np.asarray(b["rew"][0]) == 0)
    for j in range(512):
        e, s = int(obs[0, j]) // 100, int(obs[0, j]) % 100
        n_valid = min(L, lengths[e] - s)
        assert np.all(m[:n_valid, j] == 1) and np.all(m[n_valid:, j] == 0)
        assert np.all(obs[: n_valid + 1, j] == 100 * e + s + np.arange(n_valid + 1))   # incl. the final next-obs
        assert np.all(np.asarray(b["act"][1: n_valid + 1, j]) == 1)


def test_transition_sampling_uniform():
    lengths = [3, 9, 7, 12]
    buf = _buf(lengths)
    b = jax.jit(lambda k: D.sample_transitions(buf, k, 62_000))(jax.random.PRNGKey(2))
    o = np.asarray(b["obs"][:, 0]).astype(int)
    assert np.all(np.asarray(b["obs2"][:, 0]) == o + 1)
    cells = [100 * e + t for e, n in enumerate(lengths) for t in range(n)]
    counts = np.array([np.sum(o == c) for c in cells])
    assert counts.sum() == 62_000 and stats.chisquare(counts).pvalue > 1e-3

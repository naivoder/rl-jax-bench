"""Full-scale additions (CPU): the traced learning rate reproduces optax.adam exactly; analysis helpers."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
import os

from src import algorithm as A

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = yaml.safe_load(open(os.path.join(REPO, "src/configs/default.yaml")))


def test_traced_lr_equals_optax_adam():
    """chain(clip, scale_by_adam) * (-hp_lr) must equal chain(clip, adam(lr, eps)) step for step."""
    o = CFG["optim"]
    ref = optax.chain(optax.clip_by_global_norm(o["grad_clip_global_norm"]), optax.adam(3e-4, eps=o["adam_eps"]))
    new = optax.chain(optax.clip_by_global_norm(o["grad_clip_global_norm"]), optax.scale_by_adam(eps=o["adam_eps"]))
    k = jax.random.PRNGKey(0)
    p_ref = p_new = {"w": jax.random.normal(k, (7, 5)), "b": jnp.zeros(5)}
    s_ref, s_new = ref.init(p_ref), new.init(p_new)
    for i in range(30):
        g = jax.tree_util.tree_map(lambda x: jax.random.normal(jax.random.fold_in(k, i), x.shape) * (1 + i % 3), p_ref)
        u, s_ref = ref.update(g, s_ref, p_ref); p_ref = optax.apply_updates(p_ref, u)
        u, s_new = new.update(g, s_new, p_new)
        u = jax.tree_util.tree_map(lambda x: -jnp.float32(3e-4) * x, u); p_new = optax.apply_updates(p_new, u)
    for a, b in zip(jax.tree_util.tree_leaves(p_ref), jax.tree_util.tree_leaves(p_new)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)


def test_lr_is_an_hp_key_and_overridable():
    assert A.arm_hp("DQfD", CFG)["lr"] == CFG["optim"]["lr"]
    assert A.arm_hp("PDD+lr=0.0003", CFG)["lr"] == 3e-4
    assert A.base_arm("PDD+lr=0.0003+eps1=0.1") == "PDD"


def test_huber_flag_matches_optax_and_default_is_squared():
    hp0 = {k: jnp.float32(v) for k, v in A.arm_hp("DQfD", CFG).items()}
    hp1 = {k: jnp.float32(v) for k, v in A.arm_hp("DQfD+huber=1", CFG).items()}
    B, nA = 16, 4
    k = jax.random.PRNGKey(1)
    q = jax.random.normal(k, (B, nA)) * 3
    q1 = jax.random.normal(jax.random.fold_in(k, 1), (B, nA)); qn = jax.random.normal(jax.random.fold_in(k, 2), (B, nA))
    batch = dict(act=jnp.arange(B) % nA, r1=jax.random.normal(jax.random.fold_in(k, 3), (B,)) * 2, term1=jnp.zeros(B, bool),
                 Gn=jax.random.normal(jax.random.fold_in(k, 4), (B,)) * 4, disc_n=jnp.full((B,), 0.9), is_demo=jnp.arange(B) % 2 == 0)
    t0 = A.dqfd_losses(q, q1, q1, qn, qn, batch, hp0, 0.99)
    t1 = A.dqfd_losses(q, q1, q1, qn, qn, batch, hp1, 0.99)
    np.testing.assert_allclose(np.asarray(t0["j1"]), np.asarray(t0["td1"]) ** 2, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(t1["j1"]), np.asarray(optax.huber_loss(t1["td1"], delta=1.0)), rtol=1e-5, atol=1e-6)
    assert float(jnp.abs(t1["td1"]).max()) > 1.0   # the batch exercises the linear branch


def test_update_every_gate_and_prio_flag_exist():
    h = A.arm_hp("DQfD+upd_every=4+prio_total=1", CFG)
    assert h["upd_every"] == 4.0 and h["prio_total"] == 1.0 and h["huber"] == 0.0
    d = A.arm_hp("DQfD", CFG)
    assert d["upd_every"] == 1.0 and d["prio_total"] == 0.0

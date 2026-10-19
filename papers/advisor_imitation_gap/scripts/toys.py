"""Exactly computable toys (toy_a-toy_g: exact-flow ADVISOR / imitation fixed points, Prop. 1 gaps) + env/teacher/eval anchors.
Writes runs/toys/toys.json. CPU is enough (JAX_PLATFORMS=cpu)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import bootstrap, info, metric, write_json  # noqa: E402

bootstrap()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)
from src import algorithm as A  # noqa: E402
from src import eval as E  # noqa: E402
from src.envs import lighthouse as LH  # noqa: E402
from src.envs import pd as PD  # noqa: E402

OUT = {}


def rec(k, **v):
    OUT[k] = v
    metric(k, **{a: b for a, b in v.items() if not isinstance(b, (list, dict))})


# ---------------- toy_a: KL(onehot || q) = -log q(a*) and w = q(a*)^alpha
rng = np.random.RandomState(0)
z = jnp.asarray(rng.randn(1000, 7))
lab = jnp.asarray(rng.randint(7, size=1000))
q = jax.nn.softmax(z)
onehot = jax.nn.one_hot(lab, 7)
kl = jnp.sum(jnp.where(onehot > 0, onehot * (jnp.log(jnp.where(onehot > 0, onehot, 1.0)) - jnp.log(q)), 0.0), -1)
lp = A.take(A.log_softmax(z), lab)
err_kl = float(jnp.max(jnp.abs(kl + lp)))
w = A.advisor_weight(lp, jnp.ones(1000), 8.0, 0.0)
err_w = float(jnp.max(jnp.abs(w - A.take(q, lab) ** 8)))
rec("toy_a", kl_vs_neglogp_maxabs=err_kl, w_vs_q_pow_alpha_maxabs=err_w, pass_=bool(err_kl < 1e-6 and err_w < 1e-12))


# ---------------- toy_b/toy_c: exact expected-gradient flow at the PD door state, through per_step_terms/advisor_weight
def door_flow(mode, alpha, q1, steps=200000, eta=0.5, ent=0.0):
    """Door actions (d1..d4); labels uniform over d2..d4; aux at the ideal pi_IL = (0,1/3,1/3,1/3); exact A = Q - V."""
    Q = jnp.array([q1, -2 / 3, -2 / 3, -2 / 3])
    acts = jnp.repeat(jnp.arange(4), 3)                 # 12 rows: (action a, label l)
    labs = jnp.tile(jnp.arange(1, 4), 4)
    aux_logits = jnp.log(jnp.array([1e-300, 1 / 3, 1 / 3, 1 / 3]))

    def loss(zz):
        pi = jax.nn.softmax(zz)
        V = jax.lax.stop_gradient(jnp.dot(pi, Q))
        Adv = Q - V
        p_row = jax.lax.stop_gradient(pi[acts] / 3.0)
        lp = jax.lax.stop_gradient(A.log_softmax(zz))
        batch = dict(action=acts, logp=lp[acts], adv_norm=Adv[acts], value=jnp.zeros(12), ret=jnp.zeros(12),
                     label=labs, label_valid=jnp.ones(12, bool))
        t = A.per_step_terms(jnp.tile(zz, (12, 1)), jnp.tile(aux_logits, (12, 1)), jnp.zeros(12), batch, 0.1)
        wv = A.advisor_weight(t["lp_aux_label"], t["valid"], alpha, 0.0)
        ce = -(t["lp_label"]) / 4.0                    # label term: each label prob 1/3, rows repeated 4x -> /12*3
        pg = t["action_loss"]                           # sum_rows p_row * pg = E_{a~pi}[-ratio * A]
        if mode == "advisor":
            return jnp.sum(p_row * 0 + (wv * ce) / 3.0) + jnp.sum(p_row * (1 - wv) * pg) \
                + ent * jnp.sum((1 - wv[0]) * jnp.sum(pi * jnp.log(pi)))
        if mode == "static":
            return jnp.sum(ce / 3.0) + jnp.sum(p_row * pg)
        if mode == "bc":
            return jnp.sum(ce / 3.0)
        raise ValueError(mode)

    g = jax.jit(jax.grad(loss))

    def body(zz, _):
        return zz - eta * g(zz), None
    zz, _ = jax.lax.scan(body, jnp.zeros(4), None, length=steps)
    pi = jax.nn.softmax(zz)
    return float(jnp.dot(pi, Q)), np.asarray(pi)


res = {}
for name, mode, alpha, q1, target in [
        ("adv_a4", "advisor", 4.0, 1.0, 0.987500), ("adv_a8", "advisor", 8.0, 1.0, 0.999848),
        ("static_code", "static", 0.0, 1.0, 0.0), ("static_nocode", "static", 0.0, 0.0, -2 / 3),
        ("bc", "bc", 0.0, 1.0, -2 / 3)]:
    V, pi = door_flow(mode, alpha, q1)
    res[name] = dict(V=V, target=target, abs_err=abs(V - target), pi=pi.round(6).tolist())
    info("toy_b flow", case=name, V=V, target=target)
g1b_pass = all(r["abs_err"] < 1e-3 for r in res.values())
pi_bc = np.array(res["bc"]["pi"])
g1c_err = float(np.max(np.abs(pi_bc - np.array([0, 1 / 3, 1 / 3, 1 / 3]))))
rec("toy_b", cases=res, pass_=g1b_pass)
rec("toy_c", bc_minimiser=pi_bc.tolist(), maxabs_vs_pibar=g1c_err, pass_=bool(g1c_err < 1e-3))
# with the official entropy bonus 0.01 inside (1-w) (descriptive: the fixed point moves by O(0.01))
V_ent, _ = door_flow("advisor", 8.0, 1.0, ent=0.01)
OUT["toy_b_with_entropy_descriptive"] = {"adv_a8_V": V_ent}


# ---------------- PD env anchors: teacher returns 2 (privileged) / 0 (public); best f-restricted (locked + code) = 1
def pd_policy_return(policy, good):
    s = PD.reset(jax.random.PRNGKey(0), good=good)
    total, done = 0.0, False
    while not done:
        s, r, d = PD.step(s, policy(s))
        total += float(r)
        done = bool(d)
    return total


teach = {g: pd_policy_return(PD.teacher_privileged, g) for g in (1, 2, 3)}
pub = {g: pd_policy_return(PD.teacher_public, g) for g in (1, 2, 3)}
best = {g: pd_policy_return(lambda s: jnp.where(s.phase == 0, 3, jnp.asarray(PD.CODE)[jnp.clip(s.idx, 0, 9)]), g)
        for g in (1, 2, 3)}
pi_il = np.mean([np.mean([pd_policy_return(lambda s, d=d: jnp.int32(3 + d), g) for d in (1, 2, 3)]) for g in (1, 2, 3)])
rec("PD_anchors", teacher=float(np.mean(list(teach.values()))), public_teacher=float(np.mean(list(pub.values()))),
    locked_plus_code=float(np.mean(list(best.values()))), il_average_policy=float(pi_il),
    pass_=bool(np.mean(list(teach.values())) == 2 and np.mean(list(pub.values())) == 0
               and np.mean(list(best.values())) == 1 and abs(pi_il + 2 / 3) < 1e-12))


# ---------------- toy_d: 1D-LH closed forms, Monte Carlo of the IL policy
def lh1d_il(N, i, n=20000, seed=0):
    r = np.random.RandomState(seed)
    L = []
    for _ in range(n):
        pos, t = 0, 0
        while abs(pos) < N - i:
            pos += 1 if r.rand() < 0.5 else -1
            t += 1
        goal = N if r.rand() < 0.5 else -N
        t += abs(goal - pos)
        L.append(t)
    return float(np.mean(L)), float(np.std(L) / np.sqrt(n))


g1d = {}
for N, i in [(4, 1), (10, 1)]:
    m, se = lh1d_il(N, i)
    g1d[f"N{N}_i{i}"] = dict(teacher=N, f_opt=2 * N - i, il_closed=(N - i) ** 2 + N, il_mc=m, il_mc_se=se)
rec("toy_d", cases=g1d, pass_=all(abs(v["il_mc"] - v["il_closed"]) < 4 * v["il_mc_se"] for v in g1d.values()))


# ---------------- toy_e: our ported 2D-LH teacher at j = i reaches 5N - 3i exactly (average over the 4 goals)
@jax.jit
def lh_teacher_len(goal, j):
    s = LH.reset(jax.random.PRNGKey(0), goal=goal)

    def body(c, _):
        s, done, length = c
        a = LH.teacher(s, j)
        s2, _, d = LH.step(s, a)
        s = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y), s, s2)
        length = jnp.where(done, length, length + 1)
        return (s, done | d, length), None
    (s, done, length), _ = jax.lax.scan(body, (s, jnp.zeros((), bool), jnp.zeros((), jnp.int32)), None, length=1000)
    return length, done


g1e = {}
for i in range(1, 16):
    lens = [int(lh_teacher_len(g, i)[0]) for g in range(4)]
    g1e[i] = dict(mean_len=float(np.mean(lens)), per_goal=lens, closed=5 * LH.N - 3 * i, official=LH.optimal_len(i))
rec("toy_e", cases=g1e, pass_=all(abs(v["mean_len"] - v["closed"]) <= 0.01 * v["closed"] for v in g1e.values()),
    exact=all(v["mean_len"] == v["closed"] for v in g1e.values()))

# ---------------- toy_f: RobustReward@K
rr = E.robust_reward_at_k([0, 1, 2, 3], 2)
rr1 = E.robust_reward_at_k([0, 1, 2, 3], 1)
rr4 = E.robust_reward_at_k([0, 1, 2, 3], 4)
# against the authors' per-run TSV (final checkpoint): Table A.5 PD row
tsv = E.official_tsv("PoisonedDoors")
tab = {"advisor": 1.0, "ppo": 0.0, "bc": -0.60, "dagger": -0.59, "bc_teacher_forcing": -0.62, "bc_with_ppo": -0.59,
       "bc_then_ppo": -0.17, "gail": -0.09}
rr_tab = {}
for m, target in tab.items():
    v = [r["reward"][-1] for r in tsv if r["exp_type"] == m]
    rr_tab[m] = dict(n=len(v), rr10=round(E.robust_reward_at_k(v, 10), 3), table=target)
rec("toy_f", rr2=rr, rr1=rr1, rr4=rr4, target=7 / 3, table_A5_pd=rr_tab,
    pass_=bool(abs(rr - 7 / 3) < 1e-12 and rr1 == 1.5 and rr4 == 3.0
               and all(abs(v["rr10"] - v["table"]) <= 0.005 + 1e-9 for v in rr_tab.values())))

# ---------------- toy_g: sampled one-hot labels, aux = p: E[w] = sum p^(alpha+1)
key = jax.random.PRNGKey(1)
g1g = {}
for p, alpha in [((1 / 3, 1 / 3, 1 / 3), 4.0), ((0.7, 0.2, 0.1), 4.0), ((0.5, 0.5), 8.0)]:
    p = jnp.asarray(p)
    labs = jax.random.categorical(key, jnp.log(p), shape=(2_000_000,))
    wv = A.advisor_weight(jnp.log(p)[labs], jnp.ones(labs.shape), alpha, 0.0)
    exact = float(jnp.sum(p ** (alpha + 1)))
    mc, se = float(wv.mean()), float(wv.std() / np.sqrt(wv.size))
    g1g[str(np.asarray(p).round(3).tolist()) + f"_a{alpha}"] = dict(mc=mc, se=se, exact=exact, z=(mc - exact) / max(se, 1e-30))
rec("toy_g", cases=g1g, pass_=all(abs(v["mc"] - v["exact"]) <= max(4 * v["se"], 1e-12) for v in g1g.values()),
    uniform3_a4=float(3.0 ** -4))

OUT["all_pass"] = all(v.get("pass_", True) for k, v in OUT.items() if isinstance(v, dict) and "pass_" in v)
os.makedirs("runs/toys", exist_ok=True)
write_json("runs/toys/toys.json", OUT)
info("GATE 1", all_pass=OUT["all_pass"])

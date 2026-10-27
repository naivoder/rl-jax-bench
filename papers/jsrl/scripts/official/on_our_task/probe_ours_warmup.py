"""PROBE P-B (OUR code, CPU): does a random warm-up before the guide rolls in (what THEIR SB3 wrapper does by accident, issue 18) remove OUR
early pi_e-alone deficit?  Arms in ONE vmapped program, same seeds, same keys (common random numbers):
  PDD | JSRL | JSRL-warm = the JSRL arm run with jsrl_mode = 0 (no guide; pi_e eps-greedy at eps ~ 1) for the first WARM env steps,
  then jsrl_mode = 1 with the curriculum state reset to cur_init (stage 0, empty window) at the switch.
Everything else = the production code (src/train.Group, src/algorithm.Fns), budget/eps/chunk exactly those of a 200 k run (chunk 2,000),
stopped after N_CHUNKS chunks (EARLY = points 1..10).  Also logs, at points 5 and 10, the greedy agreement with the guide labels on
2,048 demo states and the mean Q-gap between the guide's action and the best other action on those states.
usage: probe_ours_warmup.py SEEDS SEED0 N_CHUNKS WARM OUT.json"""
import sys, os, json, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
PAPER = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, PAPER)
from src import train as T          # train imports gpu_bootstrap and calls setup() before jax
import jax, jax.numpy as jnp, numpy as np
from src import algorithm as A
from src.curriculum import cur_init
from src.guide import make_guide_fn

nseeds, seed0, n_chunks, warm, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
cfg = T.load_cfg()
env, gi, demo = T.load_game("Breakout", os.path.join(T.REPO, "data", "demos"))
guide_fn = make_guide_fn(env, cfg, cfg["jsrl"]["guide_eps"])
arms = ["PDD", "JSRL", "JSRL"]                       # third block = JSRL-warm (hp switched below)
seeds = list(range(seed0, seed0 + nseeds))
g = T.Group(env, cfg, demo, guide_fn, arms, seeds, 0, 200_000, 0, "/tmp/unused_ckpt", "pb", gi["mean_len"])
S = g.S
hp_run = dict(g.hp)
hp_warm = {k: v for k, v in g.hp.items()}
hp_warm["jsrl_mode"] = g.hp["jsrl_mode"].at[2 * S:].set(0.0)
state = g.init(g.hp, g.rk)
sk_env = jax.vmap(lambda k: k[2])(g.rk)
c = g.init_int(state, sk_env, g.swk, g.hp)
c = dict(c, tparams=jax.tree_util.tree_map(jnp.copy, c["params"]))
c = g.ev0(c, g.hp, jax.vmap(lambda k: jax.random.fold_in(k[4], 0))(g.rk))
train_on = jnp.ones(g.Rn, bool)
assert warm % g.chunk == 0
warm_chunks = warm // g.chunk
obs_d = g.demo["obs"][:2048]; act_d = jnp.asarray(g.demo["act"][:2048])
qfn = jax.jit(jax.vmap(lambda p: g.fns.net.apply(p, g.fns.unpack(obs_d))))
diag = {}
t0 = time.time()
for j in range(n_chunks):
    if j == warm_chunks and warm_chunks > 0:      # switch JSRL-warm to the curriculum with a fresh curriculum state
        fresh = cur_init(g.fns.ma_m)
        cur = {k: v.at[2 * S:].set(jnp.broadcast_to(fresh[k], v[2 * S:].shape)) for k, v in c["cur"].items()}
        # its next episode keeps the h drawn at its last reset (= 0); from the next reset on h = h_list[stage 0]
        c = dict(c, cur=cur)
    hp = hp_warm if j < warm_chunks else hp_run
    keys = jax.vmap(lambda k: jax.random.fold_in(k[3], j))(g.rk)
    ekeys = jax.vmap(lambda k: jax.random.fold_in(k[4], j + 1))(g.rk)
    c = g.step(c, hp, keys, ekeys, train_on)
    jax.block_until_ready(c["t"])
    if (j + 1) in (5, 10, 15, 20):
        q = np.asarray(qfn(c["params"]))                                  # (R, 2048, 3)
        a = np.asarray(act_d)
        qa = np.take_along_axis(q, a[None, :, None].repeat(q.shape[0], 0), 2)[..., 0]
        qo = np.where(np.arange(3)[None, None, :] == a[None, :, None], -np.inf, q).max(-1)
        diag[j + 1] = dict(agree=(q.argmax(-1) == a[None]).mean(1).tolist(), qgap=(qa - qo).mean(1).tolist())
    print(f"chunk {j+1} {time.time()-t0:.0f}s alone " + " ".join(f"{np.asarray(c['ev_alone'][i*S:(i+1)*S, j+1]).mean():.2f}" for i in range(3)), flush=True)
res = dict(arms=["PDD", "JSRL", f"JSRL-warm{warm}"], seeds=seeds, warm=warm, n_chunks=n_chunks, chunk=g.chunk,
           ev_alone=np.asarray(c["ev_alone"]).tolist(), ev_thr=np.asarray(c["ev_thr"]).tolist(), ev_stage=np.asarray(c["ev_stage"]).tolist(),
           ev_gsteps=np.asarray(c["ev_gsteps"]).tolist(), diag=diag, secs=time.time() - t0)
json.dump(res, open(out, "w"))
print("done", res["secs"])

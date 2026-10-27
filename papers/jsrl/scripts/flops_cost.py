"""XLA cost analysis (FLOPs, bytes) of the pure pieces of the six arms on CPU (one run, Breakout shapes): one gradient update, one
guide forward, one pi_e forward, and the totals per method for a given env-step budget.  All arms share ONE vmapped program, so per-arm wall
is not separable; this script gives the per-method FLOPs on the analytic axis.  Writes runs/full/flops.json."""
import os, sys, json
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import jax, jax.numpy as jnp
from src import algorithm as A, replay as R, train as T
from src.guide import make_guide_fn

cfg = T.load_cfg(); game = sys.argv[1] if len(sys.argv) > 1 else "Breakout"
env, gi, demo = T.load_game(game, os.path.join(T.REPO, "data", "demos"))
from src import data as D
guide_fn = make_guide_fn(env, cfg, cfg["jsrl"]["guide_eps"])
L = R.make_layout(int(demo["act"].shape[0]), cfg["per"]["agent_capacity"], int(demo["obs"].shape[1]), cfg["dqfd"]["n_step"], cfg["env"]["gamma"], cfg["per"]["alpha"], cfg["per"]["block_size"])
f = A.Fns(env, L, cfg, 100000, guide_fn, 1000)
hp = {k: jnp.asarray(v, jnp.float32) for k, v in A.arm_hp("PDD", cfg, gi["mean_len"]).items()}
params = f.init_params(jax.random.PRNGKey(0)); opt = f.tx.init(params)
buf = R.init_buffer(L, D.demo_to_buffer_arrays(demo), hp)
def cost(fn, *args):
    c = jax.jit(fn).lower(*args).compile().cost_analysis()
    c = c[0] if isinstance(c, (list, tuple)) else c
    return dict(flops=float(c.get("flops", 0)), bytes=float(c.get("bytes accessed", 0)))
upd = cost(lambda p, tp, o, b, k: f.update(p, tp, o, b, k, jnp.float32(0.6), hp, jnp.bool_(True), jnp.int32(0))[0], params, params, opt, buf, jax.random.PRNGKey(1))
obs = jnp.zeros((1,) + env.obs_shape, jnp.float32)
fwd = cost(lambda p, x: f.net.apply(p, x), params, obs)
gd = cost(lambda o, k: guide_fn(o, k), jnp.zeros(env.obs_shape, bool), jax.random.PRNGKey(2))
steps, pre, nev = 1_000_000, 75_000, 100
upd_f, fwd_f, g_f = upd["flops"], fwd["flops"], gd["flops"]
n_alone, n_thr, n_gr = f.n_alone, f.n_thr, f.n_greedy
ev_ep_steps = (n_alone + n_thr + n_gr) * (gi["mean_len"] if game == "Breakout" else gi["mean_len"])  # rough: one pi_e forward + one guide forward per episode step
learn = steps - cfg["pdd_dqn"]["learning_starts"]
per = dict(
    PDD=learn * upd_f + steps * fwd_f,
    JSRL=learn * upd_f + steps * (fwd_f + g_f) + nev * (n_thr * gi["mean_len"] * (fwd_f + g_f)),
    JSRL_Random=learn * upd_f + steps * (fwd_f + g_f),
    DQfD=pre * upd_f + steps * upd_f + steps * fwd_f,
    BCinit_FT=pre * upd_f + learn * upd_f + steps * fwd_f,
    IMIT=pre * upd_f,
)
out = dict(game=game, per_call=dict(update=upd, pi_e_forward=fwd, guide_forward=gd), total_flops_at_1M_steps=per,
           note="evaluation episodes (64 per point x 100 points, identical across arms) excluded from the per-method totals except JSRL's threshold evaluation which is the arm-specific part; the update program is the same for every arm (arms differ in traced hyper-parameters), so the update FLOPs per call are equal")
os.makedirs(os.path.join(T.REPO, "runs", "full"), exist_ok=True)
json.dump(out, open(os.path.join(T.REPO, "runs", "full", f"flops_{game}.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

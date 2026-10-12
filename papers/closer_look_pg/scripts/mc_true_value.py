"""Review C7: decompose held-out Returns-MSE of V_theta into (V_theta - V_true)^2 and Var(G|s) using
K Monte-Carlo continuations from CLONED full simulator states (frozen normalisers, training reward units).
usage: python truev.py <run_dir> <it> <seed_idx...>"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # bootstrap first
import jax, jax.numpy as jnp, numpy as np
from src.eval import Ctx, advantages
from src.train import load_ckpt
from src import data as D
from src import algorithm as A

run, it = sys.argv[1], int(sys.argv[2])
sidx = [int(x) for x in sys.argv[3:]]
env_name, alg = os.path.basename(run.rstrip("/")).split("_")
ctx = Ctx(env_name, alg)
cfg, env, nets = ctx.cfg, ctx.env, ctx.nets
seeds = [0, 1, 2, 3, 4]
st0 = ctx.init_state(jax.random.PRNGKey(0))
like = {"ts": jax.tree_util.tree_map(lambda x: jnp.stack([x] * 5), st0),
        "norm": jax.tree_util.tree_map(lambda x: jnp.stack([x] * 5), D.norm_init(ctx.spec.obs_dim)), "it": jnp.int32(0)}
ck = load_ckpt(os.path.join(run, "ckpt", f"it{it:04d}"), like)
EPL, gamma = int(cfg["episode_length"]), float(cfg["gamma"])
obs_clip, rew_norm, rew_clip = cfg.get("obs_clip"), bool(cfg.get("reward_norm", False)), cfg.get("reward_clip")
T, NS, K, H = 2048, 48, 32, 1000


def rew_units(r_raw, norm):
    if not rew_norm:
        return r_raw
    r = r_raw / (D.runstat_std(norm.rew) + 1e-8)
    return jnp.clip(r, -rew_clip, rew_clip) if rew_clip is not None else r


def obsn(norm, o):
    return D.obs_filter(norm.obs, o, obs_clip, False)[1]


def heldout(pi, norm, key):
    """single-actor T-step rollout from reset (as data.rollout, frozen) that also returns env states and t."""
    k0, ks = jax.random.split(key)
    s0 = env.reset(k0)

    def step(c, k):
        st, t = c
        ka, kr = jax.random.split(k)
        m, ls = nets.pi_apply(pi, obsn(norm, st.obs))
        a = m + jnp.exp(ls) * jax.random.normal(ka, m.shape)
        st1 = env.step(st, a)
        t1 = t + 1
        done = jnp.logical_or(st1.done > 0.5, t1 >= EPL)
        r = rew_units(st1.reward, norm)
        stn = jax.lax.cond(done, lambda: env.reset(kr), lambda: st1)
        return (stn, jnp.where(done, 0, t1)), (st, t, obsn(norm, st.obs), r, 1.0 - done.astype(jnp.float32))

    _, (sts, ts, o, r, nd) = jax.lax.scan(step, (s0, jnp.int32(0)), jax.random.split(ks, T))
    return sts, ts, o, r, nd


def mc(pi, norm, st, t, key):
    """discounted return from a cloned state to episode end (termination or time limit), training reward units."""
    def step(c, k):
        s, tt, alive, g, disc = c
        m, ls = nets.pi_apply(pi, obsn(norm, s.obs))
        a = m + jnp.exp(ls) * jax.random.normal(k, m.shape)
        s1 = env.step(s, a)
        r = jnp.where(alive, rew_units(s1.reward, norm), 0.0)
        r = jnp.where(jnp.isfinite(r), r, 0.0)
        g = g + disc * r
        tt1 = tt + 1
        done = jnp.logical_or(s1.done > 0.5, tt1 >= EPL)
        return (s1, tt1, alive & ~done, g, disc * gamma), None
    (_, _, alive, g, _), _ = jax.lax.scan(step, (st, t, jnp.bool_(True), jnp.float32(0), jnp.float32(1)),
                                         jax.random.split(key, H))
    return g, alive


out = {}
for si in sidx:
    pi = ck["ts"].pi[si]; v = ck["ts"].v[si]
    norm = jax.tree_util.tree_map(lambda x: x[si], ck["norm"])
    key = jax.random.fold_in(jax.random.PRNGKey(12345), 1000 * it + si)       # review key, disjoint from the repo tree
    sts, ts, o, r, nd = jax.jit(heldout)(pi, norm, jax.random.fold_in(key, 0))
    vals = nets.v_apply(v, o)
    _, ret = advantages(ctx, type("Tr", (), {"rew": r, "not_done": nd})(), vals, "zero")
    comp = A.complete_episode_mask(nd)
    TMAX = int(os.environ.get("TMAX", "300"))
    tok = (ts <= TMAX) if TMAX >= 0 else (ts >= 0)
    elig = np.where(np.asarray(comp & (nd > 0.5) & tok))[0]
    rng = np.random.default_rng(si)
    pick = np.sort(rng.choice(elig, min(NS, len(elig)), replace=False))
    sel = jax.tree_util.tree_map(lambda x: x[pick], sts)
    keys = jax.random.split(jax.random.fold_in(key, 1), len(pick) * K).reshape(len(pick), K, -1)
    nP = len(pick)
    rep = lambda x: jnp.repeat(x, K, axis=0)
    flat_s = jax.tree_util.tree_map(rep, sel)
    flat_t = rep(ts[pick])
    flat_k = keys.reshape(nP * K, -1)
    f = jax.jit(lambda s, t, k: jax.lax.map(lambda a: mc(pi, norm, *a), (s, t, k), batch_size=64))
    G, alive = f(flat_s, flat_t, flat_k)
    G, alive = G.reshape(nP, K), alive.reshape(nP, K)
    G, alive = np.asarray(G, np.float64), np.asarray(alive)
    Vt = G.mean(1); varG = G.var(1, ddof=1)
    vth = np.asarray(vals)[pick].astype(np.float64); g_obs = np.asarray(ret)[pick].astype(np.float64)
    bias2 = np.mean((vth - Vt) ** 2) - np.mean(varG / K)            # unbiased (V_theta - V_true)^2
    noise = np.mean(varG)
    mse_obs = np.mean((vth - g_obs) ** 2)
    rec = dict(n_states=len(pick), K=K, still_alive_frac=float(alive.mean()),
               mse_vtheta_vs_G_observed=float(mse_obs), predicted_mse=float(bias2 + noise),
               vtheta_err2=float(bias2), var_G_given_s=float(noise), frac_of_mse_from_vtheta=float(bias2 / (bias2 + noise)),
               mre_vtheta_vs_G=float(np.mean(np.abs(vth - g_obs) / np.abs(g_obs))),
               mre_vtheta_vs_Vtrue=float(np.mean(np.abs(vth - Vt) / np.abs(Vt))),
               mre_Vtrue_vs_G=float(np.mean(np.abs(Vt - g_obs) / np.abs(g_obs))),
               median_Vtrue=float(np.median(Vt)), median_vtheta=float(np.median(vth)),
               mean_signed_vtheta_minus_Vtrue=float(np.mean(vth - Vt)), TMAX=TMAX,
               per_state=dict(t=np.asarray(ts)[pick].tolist(), vth=vth.tolist(), Vt=Vt.tolist(), varG=varG.tolist(), g_obs=g_obs.tolist()))
    out[f"seed{seeds[si]}"] = rec
    print("[METRIC] truev", env_name, alg, it, seeds[si], json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items() if k != "per_state"}), flush=True)
json.dump(out, open(os.path.join("runs/mc_true_value", f"truev_T{os.environ.get('TMAX', '300')}_{env_name}_{alg}_{it}_s{'_'.join(map(str, sidx))}.json"), "w"), indent=1)

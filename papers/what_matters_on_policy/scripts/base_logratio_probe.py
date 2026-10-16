"""Debug: why does BASE (clip transform, official log-prob at the clipped action) start with
approx-KL ~0.5 and clip fraction ~0.5 when X2 (log-prob at x) has 0.014 / 0.09? One iteration of hopper, per-epoch
statistics of log rho split by saturated / interior samples. CPU or GPU."""
import sys
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import jax, jax.numpy as jnp, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import algorithm as A, configs as C, data as D, model as M
from src.train import lane_cfg, seed_keys

env, spec = D.make_env(sys.argv[1] if len(sys.argv) > 1 else 'hopper')
cfg = C.load(); static = dict(cfg['static'])
st = A.Static()
root = jax.random.PRNGKey(20210101)
for arm in ['BASE', 'X2_base_rawlogp']:
    r = C.resolve(cfg, arm); c = lane_cfg(r, static); k = seed_keys(root, 1000)
    p = M.init_params(k['init'], spec.obs_dim, spec.act_dim, v_width=r['v_width'], pi_last_gain=r['pi_last_gain'])
    ls = A.init_lane(env, spec, p, st, k['env'])
    for it in range(3):
        pool = jax.vmap(env.reset)(jax.random.split(jax.random.fold_in(k['env'], it + 1), st.num_envs))
        ls, batch, fin, done = jax.jit(lambda l, kk, pl: A.collect(env, l, c, st, kk, pl))(ls, jax.random.fold_in(k['act'], it), pool)
        obs_ms = D.meanstd_update(ls.obs_ms, batch.obs[:-1])
        obs_n = D.normalize_obs(obs_ms, batch.obs[:-1], c.obs_norm, c.obs_clip)
        mu, xr, _ = M.forward(ls.params, obs_n, c.shared)
        sd = M.std_of(xr, M.std_shift(c.init_std, c.min_std, c.softplus), c.min_std, c.softplus)
        lp = M.log_prob(batch.act, mu, sd, c.tanh, c.logp_presquash); lpb = M.log_prob(batch.act, batch.mu_b, batch.std_b, c.tanh, c.logp_presquash)
        lr = np.asarray(lp - lpb).ravel()
        sat = np.asarray(jnp.any(jnp.abs(batch.act) >= 1.0, -1)).ravel()
        print(f"[DEBUG] {arm} it={it} before-update: mean|log rho| all={np.abs(lr).mean():.4f} sat={np.abs(lr[sat]).mean() if sat.any() else 0:.4f} "
              f"int={np.abs(lr[~sat]).mean():.4f} frac_sat={sat.mean():.3f} |mu-mu_b| max={float(jnp.abs(mu-batch.mu_b).max()):.4f}")
        ls2, aux = jax.jit(lambda l, b, kk: A.update(l, b, c, st, kk))(ls, batch, jax.random.fold_in(k['mb'], it))
        mu2, xr2, _ = M.forward(ls2.params, obs_n, c.shared)
        sd2 = M.std_of(xr2, M.std_shift(c.init_std, c.min_std, c.softplus), c.min_std, c.softplus)
        lp2 = M.log_prob(batch.act, mu2, sd2, c.tanh, c.logp_presquash)
        lr2 = np.asarray(lp2 - lpb).ravel()
        print(f"[DEBUG] {arm} it={it} after-update: mean log rho={lr2.mean():.4f} sat={lr2[sat].mean() if sat.any() else 0:.4f} "
              f"int={lr2[~sat].mean():.4f} |dmu| mean={float(jnp.abs(mu2-mu).mean()):.4f} std {float(sd.mean()):.3f}->{float(sd2.mean()):.3f} "
              f"approx_kl={float(aux['approx_kl']):.4f} clipfrac={float(aux['clipfrac']):.3f}")
        ls = ls2._replace(it=ls2.it + 1)

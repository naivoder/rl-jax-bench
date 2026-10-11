"""Per-method compute cost (see README): XLA-counted FLOPs per online step and CPU wall-clock.

    JAX_PLATFORMS=cpu uv run python -m scripts.method_cost

In the training runs all arms share one vmapped step that contains every mechanism (so arms cost the same there).
Here each method is compiled ALONE from the same library pieces (src/algorithm.py) to measure its intrinsic cost:
BP = forward/backward + SGD; L2 = BP with weight decay; S&P = L2 + Gaussian perturbation; CBP = BP + utility update,
selection and replacement (Algorithm 1).  Single run, batch 1, float32, one CPU thread pool.
"""
import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src import algorithm as A  # noqa: E402
from src.model import ACT_ID, INIT_GAIN, REINIT_GAIN, init_mlp  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_step(static, method):
    def step(state, hp, x, y, key):
        params = state["params"]
        (loss, (out, hs)), grads = jax.value_and_grad(A.loss_and_out, argnums=1, has_aux=True)(static, params, x, y, hp.act)
        opt = state["opt"]
        if static.opt == "sgd":
            params = A.sgd_update(params, grads, hp)
        else:
            params, opt = A.adam_update(static, params, grads, opt, hp)
        k_noise, k_cbp = jax.random.split(key)
        cbp = state["cbp"]
        if method == "snp":
            params = A.perturb(params, hp.sigma, k_noise)
        if method == "cbp":
            params, opt, cbp, _, _ = A.cbp_update(static, params, opt, cbp, hs, hp, k_cbp)
        return {"params": params, "opt": opt, "cbp": cbp}, loss
    return step


def bench(name, static, act, hp_kw, n_steps, x_shape, y_fn):
    res = {}
    params = init_mlp(jax.random.PRNGKey(0), static.n_in, static.widths, static.n_out, INIT_GAIN[act])
    st0 = A.init_state(static, params)
    kx = jax.random.PRNGKey(1)
    X = jax.random.uniform(kx, (n_steps,) + x_shape)
    Y = y_fn(jax.random.PRNGKey(2), n_steps)
    for method, kw in hp_kw.items():
        hp = A.make_hp(act=ACT_ID[act], reinit_gain=REINIT_GAIN[act], **kw)
        step = make_step(static, method)

        @jax.jit
        def run(st):
            def one(st, i):
                return step(st, hp, X[i][None], Y[i][None], jax.random.fold_in(jax.random.PRNGKey(3), i))
            return jax.lax.scan(one, st, jnp.arange(n_steps))

        flops = jax.jit(step).lower(st0, hp, X[0][None], Y[0][None], jax.random.PRNGKey(0)).compile().cost_analysis()
        flops = flops[0] if isinstance(flops, list) else flops
        jax.block_until_ready(run(st0))           # compile + warm
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            jax.block_until_ready(run(st0))
            ts.append(time.perf_counter() - t0)
        res[method] = {"flops_per_step_xla": float(flops.get("flops", float("nan"))),
                       "us_per_step_cpu": 1e6 * min(ts) / n_steps}
        print(f"[METRIC] {name} {method:4s} flops/step={res[method]['flops_per_step_xla']:.4g} "
              f"us/step={res[method]['us_per_step_cpu']:.2f}")
    return res


def main():
    out = {"device": str(jax.devices()[0])}
    scr = A.Static(n_in=20, widths=(5,), n_out=1, loss="mse", opt="sgd", replace_mode="bernoulli", track_wmag=False)
    out["scr_relu_sgd"] = bench("SCR", scr, "relu",
                                {"bp": dict(lr=0.01), "l2": dict(lr=0.01, wd=1e-2), "snp": dict(lr=0.01, wd=1e-2, sigma=1e-3),
                                 "cbp": dict(lr=0.01, rho=1e-4, mt=10)},
                                20000, (20,), lambda k, n: jax.random.normal(k, (n, 1)))
    mn = A.Static(n_in=784, widths=(100, 100, 100), n_out=10, loss="ce", opt="sgd", replace_mode="accumulate",
                  track_wmag=False)
    out["mnist_w100_sgd"] = bench("MNIST", mn, "relu",
                                  {"bp": dict(lr=0.003), "l2": dict(lr=0.003, wd=1e-4),
                                   "snp": dict(lr=0.003, wd=1e-4, sigma=1e-5), "cbp": dict(lr=0.003, rho=1e-5, mt=100)},
                                  3000, (784,), lambda k, n: jax.random.randint(k, (n,), 0, 10))
    os.makedirs(os.path.join(REPO, "runs", "method_cost"), exist_ok=True)
    with open(os.path.join(REPO, "runs", "method_cost", "method_cost.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

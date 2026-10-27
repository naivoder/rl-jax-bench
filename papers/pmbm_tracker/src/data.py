"""Paper section VII scenario: truth, measurements, deterministic .npz I/O (JAX, float64).

Layout conventions (shared with every filter and the metrics):
  truth : dict(X float64[nT, K, 4]  state [px, vx, py, vy], zero where the target is not alive;
               alive bool[nT, K])
          step index i = 0..K-1 is the paper's time step k = i + 1 (target 1 alive for k = 1..40).
  one run of measurements : z float64[K, M, 2], zvalid bool[K, M], valid entries first.
  within a scan the order is: target detections (in target-index order), then clutter. The filters are
  order-invariant (the Murty/LAP paths and the PHD recursions sum over measurements), so the order is only a
  convention that makes the output reproducible and the detection/clutter split recoverable by eye.

Key tree: root = PRNGKey(seed_top); truth = fold_in(root, 0); meas = fold_in(root, 1);
cfg = fold_in(meas, config_id); run = fold_in(cfg, mc_run); step = fold_in(run, k) with k = 1..K.
"""
from __future__ import annotations

import hashlib
import io
import os
import warnings
import zipfile
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)

DATA_SOURCES_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "sources.yaml")
DATA_TOP_SEED = 18  # sources.yaml seeds.top_seed (ASSUMED: iteration number)
# Table II row order: p_d outer, lambda_c inner (config_id = index into this list)
DATA_TABLE2_PD = (0.95, 0.9, 0.8, 0.7, 0.6)  # Table II
DATA_TABLE2_LAMBDA = (10, 15, 20)  # Table II


def table2_config(config_id: int):
    """(p_d, lambda_c) of Table II row `config_id` (0..14, paper row order)."""
    return DATA_TABLE2_PD[config_id // 3], DATA_TABLE2_LAMBDA[config_id % 3]


def _dat_val(sec, key):
    return sec[key]["value"]


def load_scenario_cfg(path: str | None = None, **overrides) -> dict:
    """Scenario + baseline-filter config from sources.yaml (every value carries its tag in the yaml).

    Overrides select sensitivity rows: q=0.02, lambda0=3.0 (PHD/CPHD official initial mass, no predict at k=1),
    pd=..., lambda_c=..., n_steps=..., etc. (any key of the returned dict).
    """
    with open(path or DATA_SOURCES_YAML) as f:
        y = yaml.safe_load(f)
    s = y["scenario"]
    T = float(_dat_val(s, "T"))  # paper §VII
    q = float(_dat_val(s, "q"))  # paper §VII (official code: 0.02)
    cv = np.array([[1.0, T], [0.0, 1.0]])
    F = np.kron(np.eye(2), cv)  # paper §VII
    Qb = np.array([[T ** 3 / 3, T ** 2 / 2], [T ** 2 / 2, T]])
    region = np.asarray(_dat_val(s, "region"), dtype=float)  # paper §VII: [[0,300],[0,300]]
    cfg = dict(
        T=T, q=q, F=F, Q=q * np.kron(np.eye(2), Qb), H=np.kron(np.eye(2), np.array([[1.0, 0.0]])), R=np.eye(2),
        p_s=float(_dat_val(s, "p_s")), pd=float(_dat_val(s, "p_d_fig4")), lambda_c=float(_dat_val(s, "lambda_c_fig4")),
        region=region, area=float(np.prod(region[:, 1] - region[:, 0])),
        birth_rate=float(_dat_val(s, "birth_rate")), birth_mean=np.asarray(_dat_val(s, "birth_mean"), dtype=float),
        birth_cov=np.diag(np.array([150.0, 1.0, 150.0, 1.0]) ** 2),  # paper §VII diag([150^2,1,150^2,1])
        lambda0=None,  # None = paper (no targets at time 0); 3.0 = official code
        n_steps=int(_dat_val(s, "n_steps")), n_targets=int(_dat_val(s, "n_targets")),
        midpoint=int(_dat_val(s, "midpoint_step")), mid_mean=np.asarray(_dat_val(s, "midpoint_mean"), dtype=float),
        mid_cov_scale=0.1,  # 0.1 * I4, paper §VII
        t_birth=np.asarray(_dat_val(s, "birth_steps"), dtype=int), t_death=np.asarray(_dat_val(s, "death_steps"), dtype=int),
        n_mc=int(_dat_val(s, "n_mc")),
        phd=dict(prune=float(_dat_val(y["phd"], "prune")), merge=float(_dat_val(y["phd"], "merge_maha")),
                 max_comp=int(_dat_val(y["phd"], "max_components"))),
        cphd=dict(prune=float(_dat_val(y["cphd"], "prune")), merge=float(_dat_val(y["cphd"], "merge_maha")),
                  max_comp=int(_dat_val(y["cphd"], "max_components")), n_card_max=int(_dat_val(y["cphd"], "N_card_max"))),
        seed_top=int(_dat_val(y["seeds"], "top_seed")),
    )
    if "q" in overrides:  # Q is derived from q
        cfg["q"] = float(overrides["q"])
        cfg["Q"] = cfg["q"] * np.kron(np.eye(2), Qb)
    for k_, v in overrides.items():
        if k_ != "q":
            cfg[k_] = v
    return cfg


# ----------------------------------------------------------------------------------------------- truth
DATA_FIG3_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "fig3_truth.npz")


def load_fig3_truth(path: str | None = None) -> dict:
    """The paper's own Fig. 3 scenario (Sec. VII 'the scenario in Figure 3'): positions digitised from the PDF vector paths by
    scripts/extract_paper_figures.py.  Velocities are finite differences (unused by z and OSPA); one red vertex in
    k = 77..80 is interpolated (error << R).  Same dict layout as make_truth."""
    z = np.load(path or DATA_FIG3_NPZ)
    return dict(X=jnp.asarray(z["X"], jnp.float64), alive=jnp.asarray(z["alive"]))


@partial(jax.jit, static_argnames=("n_targets", "n_steps", "midpoint"))
def _dat_truth_core(key, F, Q, mid_mean, mid_cov_scale, t_birth, t_death, *, n_targets, n_steps, midpoint):
    nfb = midpoint - 1
    L = jnp.linalg.cholesky(Q)  # chol(Q)' of the Octave code
    Finv = jnp.linalg.inv(F)
    k0, kf, kb = jax.random.split(key, 3)
    x0 = mid_mean[None, :] + jnp.sqrt(mid_cov_scale) * jax.random.normal(k0, (n_targets, 4), dtype=jnp.float64)
    wf = jax.random.normal(kf, (nfb, n_targets, 4), dtype=jnp.float64) @ L.T
    wb = jax.random.normal(kb, (nfb, n_targets, 4), dtype=jnp.float64) @ L.T

    def fwd(x, w):
        x = x @ F.T + w  # x_{k+1} = F x_k + w
        return x, x

    def bwd(x, w):
        x = (x + w) @ Finv.T  # x_{k-1} = F^{-1}(x_k + w)
        return x, x

    _, xf = jax.lax.scan(fwd, x0, wf)  # [nfb, nT, 4]  steps midpoint+1 ...
    _, xb = jax.lax.scan(bwd, x0, wb)  # steps midpoint-1, midpoint-2, ...
    X = jnp.concatenate([xb[::-1], x0[None], xf], axis=0)  # [K, nT, 4] in time order
    X = jnp.transpose(X, (1, 0, 2))
    k = jnp.arange(1, n_steps + 1)
    alive = (k[None, :] >= t_birth[:, None]) & (k[None, :] < t_death[:, None])
    return jnp.where(alive[..., None], X, 0.0), alive


def make_truth(key, cfg) -> dict:
    """Section VII truth: midpoint k=41 ~ N([150,0,150,0], 0.1 I4) per target, forward x_{k+1}=Fx_k+w and
    backward x_{k-1}=F^{-1}(x_k+w), w ~ N(0,Q) (independent draws forward/backward); target 1 alive k=1..40, others k=1..81.
    Mirrors TrajectoryWilliams15.m (law only; the stream differs from MATLAB's legacy generator).
    Returns dict(X float64[nT,K,4] (zero outside the lifetime, as the official X_truth), alive bool[nT,K]).
    """
    assert cfg["n_steps"] == 2 * cfg["midpoint"] - 1, "n_steps must be odd with the midpoint in the middle"
    X, alive = _dat_truth_core(
        key, jnp.asarray(cfg["F"]), jnp.asarray(cfg["Q"]), jnp.asarray(cfg["mid_mean"]), cfg["mid_cov_scale"],
        jnp.asarray(cfg["t_birth"]), jnp.asarray(cfg["t_death"]),
        n_targets=int(cfg["n_targets"]), n_steps=int(cfg["n_steps"]), midpoint=int(cfg["midpoint"]))
    return dict(X=X, alive=alive)


# ----------------------------------------------------------------------------------------- measurements
def _dat_meas_core(key, X, alive, pd, lam, H, cholR, region_lo, region_size, M_max):
    nT, K = alive.shape

    def one_step(k0):
        sk = jax.random.fold_in(key, k0 + 1)  # step key: k = 1..K
        kd, kn, kc, kp = jax.random.split(sk, 4)
        det = (jax.random.uniform(kd, (nT,), dtype=jnp.float64) < pd) & alive[:, k0]
        zt = X[:, k0, :] @ H.T + jax.random.normal(kn, (nT, 2), dtype=jnp.float64) @ cholR.T
        nc = jax.random.poisson(kc, lam)
        zc = region_lo[None, :] + jax.random.uniform(kp, (M_max, 2), dtype=jnp.float64) * region_size[None, :]   # honours the lower bound
        zall = jnp.concatenate([zt, zc], axis=0)
        vall = jnp.concatenate([det, jnp.arange(M_max) < nc], axis=0)
        order = jnp.argsort(~vall, stable=True)[:M_max]  # valid first, target detections before clutter
        ndrop = jnp.maximum(jnp.sum(det) + nc - M_max, 0)
        return zall[order], vall[order], ndrop

    z, zv, nd = jax.vmap(one_step)(jnp.arange(K))
    return jnp.where(zv[..., None], z, 0.0), zv, jnp.sum(nd).astype(jnp.int32)


_dat_meas_one = jax.jit(_dat_meas_core, static_argnames=("M_max",))
_dat_meas_batch = jax.jit(jax.vmap(_dat_meas_core, in_axes=(0, None, None, None, None, None, None, None, None, None)),
                          static_argnames=("M_max",))


def _dat_meas_args(truth, cfg):
    return (truth["X"], truth["alive"], jnp.asarray(cfg["H"]), jnp.asarray(np.linalg.cholesky(cfg["R"])),
            jnp.asarray(cfg["region"][:, 0]), jnp.asarray(cfg["region"][:, 1] - cfg["region"][:, 0]))


def make_measurements(key, truth, pd, lambda_c, cfg, M_max):
    """One MC run: z float64[K,M_max,2], zvalid bool[K,M_max], n_dropped int32 (clutter that did not fit M_max, total over steps).

    Per-target Bernoulli(pd) detection, z = Hx + v (v ~ N(0,R)), Poisson(lambda_c) clutter uniform on the region, no clipping
    (official code CreateMeasurement.m). Invalid entries are zero. `key` is the RUN key (fold_in(cfg_key, mc_run)).
    """
    X, alive, H, cholR, lo, rs = _dat_meas_args(truth, cfg)
    return _dat_meas_one(key, X, alive, jnp.asarray(pd, jnp.float64), jnp.asarray(lambda_c, jnp.float64), H, cholR, lo, rs, int(M_max))


def make_measurements_runs(root_key, truth, config_id, pd, lambda_c, n_runs, cfg, M_max, on_drop="warn"):
    """All MC runs of one config: z[R,K,M,2], zvalid[R,K,M], n_dropped int32[R], via the key tree
    run = fold_in(fold_in(fold_in(root,1), config_id), mc_run). on_drop in {'raise','warn','ignore'} for n_dropped > 0."""
    cfg_key = jax.random.fold_in(jax.random.fold_in(root_key, 1), config_id)
    keys = jax.vmap(lambda r: jax.random.fold_in(cfg_key, r))(jnp.arange(n_runs))
    X, alive, H, cholR, lo, rs = _dat_meas_args(truth, cfg)
    z, zv, nd = _dat_meas_batch(keys, X, alive, jnp.asarray(pd, jnp.float64), jnp.asarray(lambda_c, jnp.float64), H, cholR, lo, rs, int(M_max))
    tot = int(np.asarray(nd).sum())
    if tot > 0 and on_drop != "ignore":
        msg = f"make_measurements_runs: {tot} clutter measurements dropped at M_max={M_max} (config {config_id}, lambda_c={lambda_c})"
        if on_drop == "raise":
            raise ValueError(msg)
        warnings.warn(msg)
    return z, zv, nd


# ------------------------------------------------------------------------------------------------- I/O
def sha256_arrays(arrs: dict) -> str:
    """Content hash (sorted keys; name, dtype, shape, bytes) independent of the container."""
    h = hashlib.sha256()
    for k in sorted(arrs):
        a = np.ascontiguousarray(np.asarray(arrs[k]))
        h.update(k.encode() + b"|" + str(a.dtype).encode() + b"|" + str(a.shape).encode() + b"|")
        h.update(a.tobytes())
    return h.hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dat_write_npz(path, arrs):
    """np.savez-compatible zip with a fixed timestamp so that the FILE sha256 is reproducible."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for k in sorted(arrs):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(np.asarray(arrs[k])), allow_pickle=False)
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zf.writestr(zi, buf.getvalue())


def save_truth(path, truth, meta=None):
    """truth.npz with keys X[nT,K,4], alive[nT,K] (+ scalar meta). Returns file sha256."""
    arrs = dict(X=np.asarray(truth["X"]), alive=np.asarray(truth["alive"]))
    for k, v in (meta or {}).items():
        arrs["meta_" + k] = np.asarray(v)
    _dat_write_npz(path, arrs)
    return sha256_file(path)


def load_truth(path) -> dict:
    with np.load(path) as d:
        return dict(X=jnp.asarray(d["X"]), alive=jnp.asarray(d["alive"]))


def save_measurements(path, truth, z, zvalid, meta=None):
    """Measurement set + truth in ONE npz (oracle/algorithm input): z[R,K,M,2], zvalid[R,K,M], X[4,K,4], alive[4,K]
    plus scalar meta (pd, lambda_c, config_id, seed_top, M_max ...). Returns file sha256."""
    arrs = dict(z=np.asarray(z), zvalid=np.asarray(zvalid), X=np.asarray(truth["X"]), alive=np.asarray(truth["alive"]))
    for k, v in (meta or {}).items():
        arrs["meta_" + k] = np.asarray(v)
    _dat_write_npz(path, arrs)
    return sha256_file(path)


def load_measurements(path):
    """-> (truth dict, z[R,K,M,2], zvalid[R,K,M], meta dict) as numpy/jnp arrays."""
    with np.load(path) as d:
        meta = {k[5:]: d[k][()] for k in d.files if k.startswith("meta_")}
        return (dict(X=jnp.asarray(d["X"]), alive=jnp.asarray(d["alive"])), jnp.asarray(d["z"]), jnp.asarray(d["zvalid"]), meta)


def export_oracle_mat(npz_path, mat_path, cfg):
    """Octave/MATLAB driver input: z{r,k} (2 x m_k cells, valid entries only, MATLAB column order), X_truth (4*nT x K, official
    layout, zero outside the lifetime), t_birth, t_death. Reads the measurement npz written by save_measurements."""
    from scipy.io import savemat
    with np.load(npz_path) as d:
        z, zv, X, alive = d["z"], d["zvalid"], d["X"], d["alive"]
    R, K = z.shape[:2]
    cell = np.empty((R, K), dtype=object)
    for r in range(R):
        for k in range(K):
            cell[r, k] = np.ascontiguousarray(z[r, k][zv[r, k]].T).reshape(2, -1)
    nT = X.shape[0]
    Xo = np.concatenate([X[i].T for i in range(nT)], axis=0)  # [4nT, K]
    savemat(mat_path, dict(z=cell, X_truth=Xo, t_birth=np.asarray(cfg["t_birth"], float)[None], t_death=np.asarray(cfg["t_death"], float)[None]),
            do_compression=False)

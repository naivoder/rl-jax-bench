"""Shared setup: GPU bootstrap (BEFORE jax), x64, logging helpers and the frozen config objects.

Every entry point does `from src import common` (or `import common`) first; this module calls
gpu_bootstrap.setup() before jax is imported, enables float64 (A-42: environment, EKF, OA in float64; networks
explicitly float32) and re-exports the jaxlog helpers.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, fields

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)   # A-42

import yaml  # noqa: E402
from rl_jax_bench.jaxlog import Timer, check_device, error, info, metric, nan_guard, warn, write_json  # noqa: E402,F401

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_DIR = os.path.join(REPO, "src", "configs")


def _floatify(x):
    """PyYAML (YAML 1.1) reads '1.0e4' as a string; convert such scalars to float."""
    if isinstance(x, dict):
        return {k: _floatify(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_floatify(v) for v in x]
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return x
    return x


def load_yaml(name: str = "default.yaml") -> dict:
    with open(os.path.join(CFG_DIR, name)) as f:
        return _floatify(yaml.safe_load(f))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_calibration() -> dict:
    p = os.path.join(CFG_DIR, "calibration.json")
    if not os.path.exists(p):
        raise FileNotFoundError("calibration.json missing: run scripts/calibrate.py first (A-02/A-04)")
    with open(p) as f:
        return json.load(f)


@dataclass(frozen=True)
class EnvCfg:
    """Static (hashable) environment configuration. Values from default.yaml (tags there and in sources.yaml)."""
    T0: float = 2.5
    K: int = 5
    M: int = 12
    sigma_r0_sq: float = 16.0
    sigma_th0_sq: float = 1.0e-6
    sigma_w_sq: float = 16.0
    r0: float = 3000.0
    tau0: float = 1.0
    P0_diag: tuple = (1.0e4, 1.0e4, 100.0, 100.0)
    pfa: float = 1.0e-3
    gate_Td: float = 500.0
    area_radius: float = 20000.0
    lifetime: int = 3000
    spawn_every: int = 100
    spawn_prob: float = 0.03
    spawn_range: tuple = (5000.0, 15000.0)
    spawn_speed: tuple = (0.5, 2.0)
    r_min_snr: float = 100.0
    snr_floor_noise: float = 1.0e-3
    beta: float = 3.0e4
    theta_max: float = 0.9
    lambda0: float = 5000.0
    alpha: float = 5000.0
    asset_xy: tuple = (2500.0, -10000.0)
    lbm_i: int = 2
    g_on: str = "executed"        # A-13 (declared): budget term g_t on executed dwells; "requested" = A-13a amendment
    # case switches (static)
    weights: str = "equal"        # "equal" (Case I/III/IV) | "asset" (Case II, eq. 35)
    lbm: bool = False             # Case III beam-misalignment loss (eqs. 36-37)
    interf: bool = False          # Case IV high-interference regions (A-38)
    interf_scan: str = "none"     # A-38b sensitivity: "none" (A-38, tracking only) | "noise" (factor also
                                  # on scan measurement variance) | "snr" (scan SNR / factor: noise AND P_d)
    interf_regions: tuple = ((0.0, 10000.0, -10000.0, 0.0, 7.0), (-5000.0, 0.0, 0.0, 10000.0, 10.0))  # §VII-B4

    @staticmethod
    def from_yaml(y: dict | None = None, **over) -> "EnvCfg":
        y = y or load_yaml()
        e, o = y["env"], y["objective"]
        kw = dict(T0=e["T0"], K=e["K"], M=e["M"], sigma_r0_sq=e["sigma_r0_sq"], sigma_th0_sq=e["sigma_th0_sq"],
                  sigma_w_sq=e["sigma_w_sq"], r0=e["r0"], tau0=e["tau0"], P0_diag=tuple(e["P0_diag"]), pfa=e["pfa"],
                  gate_Td=e["gate_Td"], area_radius=e["area_radius"], lifetime=e["lifetime"],
                  spawn_every=e["spawn_every"], spawn_prob=e["spawn_prob"], spawn_range=tuple(e["spawn_range"]),
                  spawn_speed=tuple(e["spawn_speed"]), r_min_snr=e["r_min_snr"], snr_floor_noise=e["snr_floor_noise"],
                  beta=o["beta"], theta_max=o["theta_max"], lambda0=o["lambda0"], alpha=o["alpha"],
                  asset_xy=tuple(o["asset_xy"]), lbm_i=o["lbm_i"], g_on=o.get("g_on", "executed"))
        kw.update(over)
        return EnvCfg(**kw)

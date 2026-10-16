"""Config loading: agent.yaml (REC + arms) -> per-arm resolved dicts -> LaneCfg / init kwargs."""
from __future__ import annotations

import math
import os

import yaml

HERE = os.path.dirname(__file__)
INIT_KEYS = ("v_width", "pi_last_gain")          # choices that act at initialisation only


def load(path: str | None = None) -> dict:
    with open(path or os.path.join(HERE, "agent.yaml")) as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, arm: str) -> dict:
    """REC with the arm's overrides applied (BASE included)."""
    out = dict(cfg["rec"])
    for k, v in cfg["arms"][arm].items():
        if k not in out:
            raise KeyError(f"arm {arm} changes unknown key {k}")
        out[k] = v
    return out


def arm_list(cfg: dict, spec: str) -> list[str]:
    if spec == "primary":
        return ["REC"] + cfg["primary"]
    if spec == "all":
        return ["REC"] + cfg["primary"] + cfg["secondary"]
    if spec == "review":
        return ["REC", "A3a_shared_bc1"] + cfg["review"] + ["A5a_lr_1em4"]
    if spec == "small":
        return ["REC"] + cfg["primary"] + cfg["secondary"] + cfg["diagnostic"]
    return spec.split(",")


def lr_horizon_steps(static: dict, frames=None) -> float:
    """Optimiser steps over which lr decays (official PolynomialDecay(final_iteration*epochs*batches)).
    ``frames`` overrides static.lr_horizon_frames (per-arm  sensitivity arms; None = static)."""
    final_iteration = math.ceil((frames or static["lr_horizon_frames"]) / static["iteration_size"])
    return float(final_iteration * static["epochs"] * (static["iteration_size"] // static["minibatch"]))

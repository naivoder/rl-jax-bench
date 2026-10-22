"""model.py -- scenario constants, the two pre-declared configurations (CFG-O, CFG-P) and the algorithm table.

The paper has no neural network and no dataset: "model" here is the state-space / sensing model (paper SS V-A, eqs. 47-49)
and the frozen configuration objects that select between the paper-as-written and the official-code behavior.
Every value carries a tag (paper SS / Table / Figure / official code D# / ASSUMED); the machine-readable index is
src/configs/sources.yaml (the two files must agree; tests/test_smoke.py::test_config_matches_sources checks the key values).

CFG-O  = official code behavior at commit b5959f6 (used for the numeric grading of Table III).
CFG-P  = paper as written (used to grade the paper's claims); the code is used only where the paper is silent.

No jax import here on purpose: the module is importable by report scripts without the GPU bootstrap.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

OFFICIAL_COMMIT = "b5959f6a696a0e566e0925ec2376be361982e773"
TRUTH_SHA256 = "4e59bc8c9403341f091402e8b43d457316b41a0061fbdb2347f7f7e68b6dae36"
TRUTH_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "truth_seed123546.npy")

# --------------------------------------------------------------------------------------------------------------------
# Scenario constants shared by both configurations
# --------------------------------------------------------------------------------------------------------------------
N_STEPS = 200                       # paper SS V-A; official Parameters.number_of_timesteps
REGION = 250.0                      # [-250, 250]^2: paper SS V-A (500x500) + Figure 4 + official Parameters.surveillance_area
OBSTACLE = (-30.0, 60.0, 30.0, 90.0)  # (x0, y0, x1, y1) open rectangle; Figure 4 pixel reading + official coordD3/coordB3
SENSOR_INIT = ((5.0, 150.0), (-5.0, 150.0))  # official main.py D26; consistent with Figure 4
N_SENS = 2                          # paper SS V-A
STEP_LEN = 15.0                     # paper SS V-A (circle of radius 15)
N_MOVES = 6                         # paper SS V-A + Figure 2 (stay + 6 moves = 7 actions)
FOV = 40.0                          # paper SS V-A (sensor radius r = 40)
PD_MAX = 0.999                      # paper eq. 48
PSI = 120.0                         # paper SS V-B (3 r)
TAU = 1.0                           # paper eq. 47
Q_INT = 0.8                         # paper eq. 47 (q); official Parameters.sigma_v
P_SURV = 0.99                       # paper SS V-A
BIRTH_MEAN = (0.0, 0.1, 0.0, 0.1)   # paper SS V-A
BIRTH_VAR = 6.0                     # paper SS V-A diag(6,6,6,6); official 0.2*diag(30,..)
GOSPA_C = 80.0                      # paper SS V-C (c = 2r)
GOSPA_P = 2                         # paper SS II-C
LAMBDA_DISC = 0.9                   # paper SS V-C (discount)
UCT_EPS = 0.05                      # official Parameters.best_child_c_param (D13)
R_PRUNE = 1e-4                      # official Parameters.pruning_threshold / tomb.py r_threshold (D20, D22)
MERGE_MAHA = 1.0                    # paper SS V-C; official utils.merge_gaussians threshold
EST_THRESH = 0.5                    # official Parameters.existence_threshold (D23)
CLUTTER_RATES = (0.1, 1.0, 2.0)     # Table III
N_RUNS_PAPER = 50                   # paper SS V-C

# fixed array capacities (jit shapes) -- ASSUMED, each with an overflow counter that must read 0
NB_MAX = 64                         # filter Bernoulli slots (observed <= ~12)
N_TRUTH = 9                         # truth slots in the authors' array
K_TRUTH = 5                         # compacted alive-truth slots for GOSPA (max alive in the array is 4)
E_MAX = 16                          # extracted estimates fed to GOSPA (counter on overflow)
N_CLUT = 16                         # clutter points per sensor per step (P(Poisson(2) > 16) ~ 1e-10)
M_MEAS = N_TRUTH + N_CLUT           # measurement slots per sensor
R_CAP = 16                          # planner root Bernoulli capacity (filter count + 1 birth)

# --------------------------------------------------------------------------------------------------------------------
# The two configurations
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cfg:
    name: str
    # --- model / truth / sensing (README differences table, D1-D7) ---
    r_b: float                       # birth existence; D1
    R_var: float                     # measurement noise variance (R = R_var * I2); D2
    truth_pd: str                    # 'hard' = 0.999 + delete measurement > 40 m from sensor; 'smooth' = eq. 48 on truth; D3
    clutter_gen: str                 # 'square' = Poisson(lam) in the 80x80 square cropped to the disc; 'disc' = in the disc; D6
    filt_pd: str                     # 'hard' = 0.999 if |mean-sensor|<40 else 0; 'smooth' = eq. 48 at the mean; D4
    dummy_init: bool                 # one dummy Bernoulli r=0 at k=0 (official D27)
    # --- filter (D20, D21) ---
    lbp_mode: str                    # 'one_sweep' (official alias bug, D30) | 'converged'
    lbp_tol: float
    lbp_max_iter: int
    merge_cap: str                   # 'myopic_only' (official D21) | 'always' | 'never'
    # --- planner (D8-D19) ---
    plan_nx: int                     # planner state dimension: 2 = position random walk (official), 4 = NCV (paper)
    root_double_predict: bool        # D10
    ph_mode: str                     # 'sequential' (exact joint, official D11) | 'product' (paper eq. 38)
    plan_trace: str                  # 'pos' | 'full' (full only meaningful for plan_nx=4)
    kld_mode: str                    # 'official' (Gaussian part, joint scaled by sum r; D12) | 'official_nosum' (same without the sum-r factor) | 'bernoulli' (KL(pred || node-merged Bernoulli), ASSUMED) | 'info_gain' (E_h[KL(post_h || pred)], ASSUMED, F1 sensitivity)
    expansion: str                   # 'det' (official pop-last, D16) | 'random' (paper SS IV-C.2)
    delta_selected: str              # 'running' (official D14: selected node contributes its running C-bar) | 'first' (its stored cost)
    gospa_space: str                 # 'full' (4-D state incl. velocity, official D24) | 'pos'
    uct_eps: float = UCT_EPS
    lam: float = LAMBDA_DISC
    r_cap: int = R_CAP
    # --- follow-up options (every default reproduces the archived behavior bit for bit) ---
    budget_mode: str = "iter"        # 'iter' = budget counts MCTS iterations (official D15) | 'nodes' = budget counts nodes ADDED to the tree (paper SS IV-C.2 p.8)
    est_rule: str = "half"           # 'half' = report r > 0.5 (official D23) | 'gamma' = report r > Gamma*_d(tr P), eq. 30 (the optimal MSGOSPA estimator, eq. 20)
    merge_each: bool = False         # False: one merge after both sensor updates (official) | True: merge after EACH sensor update (paper SS V-C 'after each update', one reading)

    def replace(self, **kw) -> "Cfg":
        return dataclasses.replace(self, **kw)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()[:16]


# CFG-O: replicates the authors' HEAD behavior (tags: official code D#)
CFG_O = Cfg(
    name="CFG-O", r_b=0.05, R_var=3.0, truth_pd="hard", clutter_gen="square", filt_pd="hard", dummy_init=True,
    lbp_mode="one_sweep", lbp_tol=1e-4, lbp_max_iter=1, merge_cap="myopic_only",
    plan_nx=2, root_double_predict=True, ph_mode="sequential", plan_trace="pos", kld_mode="official",
    expansion="det", delta_selected="running", gospa_space="full",
)

# CFG-P: the paper's stated values / equations; code used only where the paper is silent (tags in sources.yaml)
CFG_P = Cfg(
    name="CFG-P", r_b=0.03, R_var=2.0, truth_pd="smooth", clutter_gen="disc", filt_pd="smooth", dummy_init=False,
    lbp_mode="converged", lbp_tol=1e-9, lbp_max_iter=1000, merge_cap="always",
    plan_nx=4, root_double_predict=False, ph_mode="product", plan_trace="pos", kld_mode="bernoulli",
    expansion="random", delta_selected="first", gospa_space="full",
)

# CFG-P2: CFG-P made internally consistent and literal where the paper is explicit.
#  - plan_trace='full': eqs. 29/39 print tr(P) of the full Bernoulli covariance, and CFG-P scores GOSPA on the full 4-D state (gospa_space='full'), so the
#    planner bound and the metric use the same coordinates (CFG-P v1 mixed a position trace with a 4-D metric).  The KLD arms use the same (full) block.
#  - budget_mode='nodes': paper SS IV-C.2 p.8 'the budget of the tree is the maximum number of nodes that can be added to the tree'.
# CFG-P (v1) is kept unchanged as the archived configuration of the first full E1 run; its numbers are superseded by CFG-P2.
CFG_P2 = CFG_P.replace(name="CFG-P2", plan_trace="full", budget_mode="nodes")

CONFIGS = {"CFG-O": CFG_O, "CFG-P": CFG_P, "CFG-P2": CFG_P2}

# --------------------------------------------------------------------------------------------------------------------
# Algorithm table (Table II: budgets and lookahead; official main.py)
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Alg:
    name: str
    cost: str          # 'GD' | 'KLD' | 'RANDOM' | 'HOLD'
    T: int             # lookahead (tree depth)
    b_joint: int       # MCTS iterations when the sensors are planned jointly
    b_ind: int         # MCTS iterations per sensor when planned individually

    @property
    def myopic(self) -> bool:
        return self.T == 1 and self.cost in ("GD", "KLD")

    @property
    def planner(self) -> bool:
        return self.cost in ("GD", "KLD")


_VAR = {"Myopic": (1, 49, 7), "MCTS1": (5, 49, 7), "MCTS2": (10, 49, 7), "MCTS3": (5, 200, 40), "MCTS4": (10, 200, 40)}
ALGS = {}
for _c in ("GD", "KLD"):
    for _v, (_T, _bj, _bi) in _VAR.items():
        ALGS[f"{_v}-{_c}"] = Alg(f"{_v}-{_c}", _c, _T, _bj, _bi)
ALGS["RANDOM"] = Alg("RANDOM", "RANDOM", 1, 0, 0)   # descriptive control, not in the paper (ASSUMED)
ALGS["HOLD"] = Alg("HOLD", "HOLD", 1, 0, 0)         # descriptive control, not in the paper (ASSUMED)
ALG_ORDER = [f"{v}-{c}" for c in ("GD", "KLD") for v in _VAR] + ["RANDOM", "HOLD"]
ALG_ID = {n: i for i, n in enumerate(ALG_ORDER)}

# Paper Table III (time-averaged RMS-GOSPA, 50 runs); columns are clutter 0.1 / 1 / 2.  The KLD columns for clutter 1 and 2
# are identical in the paper (looks like a table error).
PAPER_TABLE3 = {
    "Myopic-GD": (72.80, 72.80, 72.80), "MCTS1-GD": (49.05, 50.46, 51.78), "MCTS2-GD": (54.59, 54.00, 56.18),
    "MCTS3-GD": (47.17, 47.88, 50.53), "MCTS4-GD": (48.24, 49.96, 51.75),
    "Myopic-KLD": (72.80, 72.80, 72.80), "MCTS1-KLD": (51.94, 53.16, 53.16), "MCTS2-KLD": (54.74, 54.71, 54.71),
    "MCTS3-KLD": (51.38, 51.51, 51.51), "MCTS4-KLD": (51.76, 52.36, 52.36),
}

# --------------------------------------------------------------------------------------------------------------------
# State-space matrices (numpy float64; converted to jnp inside the jitted functions)
# --------------------------------------------------------------------------------------------------------------------


def F_ncv(tau: float = TAU) -> np.ndarray:
    """eq. 47: F = I2 kron [[1, tau],[0, 1]] with state [px, vx, py, vy]."""
    blk = np.array([[1.0, tau], [0.0, 1.0]])
    return np.kron(np.eye(2), blk)


def Q_ncv(q: float = Q_INT, tau: float = TAU) -> np.ndarray:
    """eq. 47: Q = q I2 kron [[tau^3/3, tau^2/2],[tau^2/2, tau]]."""
    blk = np.array([[tau ** 3 / 3, tau ** 2 / 2], [tau ** 2 / 2, tau]])
    return q * np.kron(np.eye(2), blk)


H_FULL = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])   # eq. 49
POS_IDX = np.array([0, 2])


def birth_cov(cfg_nx: int) -> np.ndarray:
    return BIRTH_VAR * np.eye(cfg_nx)


def planner_model(cfg: Cfg):
    """(F, Q, H, birth_mean, birth_cov, pos_idx) of the PLANNER state (D8, D9).

    CFG-O: position-only random walk, F = F_ncv[::2, ::2] = I2, Q = Q_ncv[::2, ::2] = (q/3) I2 (official MCTS*.predict).
    CFG-P: full 4-D NCV (paper eqs. 4-5), cost on the position block.
    """
    if cfg.plan_nx == 2:
        F = F_ncv()[::2, ::2]
        Q = Q_ncv()[::2, ::2]
        H = np.eye(2)
        bm = np.array(BIRTH_MEAN)[::2]
        bc = birth_cov(2)
        pidx = np.array([0, 1])
    else:
        F, Q, H = F_ncv(), Q_ncv(), H_FULL
        bm = np.array(BIRTH_MEAN)
        bc = birth_cov(4)
        pidx = POS_IDX
    return F, Q, H, bm, bc, pidx


def action_offsets() -> np.ndarray:
    """[7, 2] sensor displacement per action in the official order: angles 0, 60, ..., 300 deg then stay (D18)."""
    th = np.arange(0, 2 * np.pi, 2 * np.pi / N_MOVES)
    mv = STEP_LEN * np.stack([np.cos(th), np.sin(th)], axis=1)
    return np.concatenate([mv, np.zeros((1, 2))], axis=0)


def load_truth() -> np.ndarray:
    """The authors' ground truth [9, 4, 200] (NaN where the target is not alive); sha256 asserted at load."""
    path = os.path.abspath(TRUTH_PATH)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    if sha != TRUTH_SHA256:
        raise RuntimeError(f"truth array sha256 mismatch: {sha} != {TRUTH_SHA256}")
    return np.load(path)

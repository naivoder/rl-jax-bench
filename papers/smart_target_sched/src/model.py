"""Target model of Hao et al. (arXiv:2312.07858v1): smart targets whose tracking-error covariance (TEC) follows the
deterministic mode-weighted Kalman recursion (paper eqs 3-7).  There is NO neural network in this paper: this module is the
"environment model" (constants, target banks, the TEC map phi, the cost), kept pure and jit/vmap-friendly.

Conventions
  * float64 throughout; x64 is switched on at import and asserted.
  * A *bank* holds N independent targets with M = 2 modes (CV, CT) and state dimension L (L = 1 scalar "TEV" case, L = 4 matrix case).
    P has shape (N, L, L) (the scalar case is carried as 1x1 matrices so one API serves both).
  * act is a float 0/1 vector (N,): 1 = the target is illuminated/tracked in this slot.
  * phi(bank, P, act) is eq (3) (act = 1) / eq (7) (act = 0): the u^{act}-weighted mixture of the per-mode covariance updates.
Every constant carries a source tag (paper section/equation/table, or ASSUMED); the same values are in
src/configs/sources.yaml and tests/test_smoke.py::test_config_matches_sources keeps the two in sync.
"""
from __future__ import annotations

from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)
assert jax.config.jax_enable_x64, "float64 is mandatory"
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

# ----------------------------------------------------------------------------------------------- global constants
BETA = 0.9            # paper §IV-A/B/C; §IV-D ASSUMED
TAU = 100             # paper §IV-A/B/C, §III-G; §IV-D ASSUMED: rollout truncation of F, G
T_HORIZON = 100       # paper §IV-B, §IV-D ("T = 100 s"); slots ASSUMED
N_MC = 100            # paper §IV-B/C/D (Nmc = 100)
H_COST = 0.0          # paper §IV-A ("no measurement costs"), §IV-D
TOP_SEED = 20231213   # ASSUMED (arbitrary; arXiv date)
G_EPS = 1e-12         # PCLI1 guard: g <= G_EPS counts as a violation (not in the paper; numerics)

# scalar model (paper §IV-A)
F_CV = 1.1
F_CT = 1.3
Q_CV = 1.0
H_SCALAR = 1.0
R_SCALAR = 2.0

# smartness: U = [u^0, u^1] = prob. vector over (CV, CT) after action 0 / 1 (paper eq (35))
U_RECKLESS = ((0.90, 0.10), (0.20, 0.80))
U_CAUTIOUS = ((0.95, 0.05), (0.60, 0.40))
TYPE_U = {"reckless": U_RECKLESS, "cautious": U_CAUTIOUS}

# matrix model (paper §IV-D, eqs 36-39)
TS = 1.0                          # paper §IV-D
OMEGA_RAD = 3.0 * np.pi / 180.0   # paper §IV-D (3 deg); radians ASSUMED
Q_CV_MATRIX = 1.0                 # paper §IV-D (q^CV = 1)
R_MATRIX_DIAG = 2.0               # paper §IV-D (R = 2 I_2)


class Bank(NamedTuple):
    """N independent targets.  F, Q: (N, M, L, L); U: (N, 2, M); H: (Lz, L); R: (Lz, Lz); d, h: (N,)."""
    F: jnp.ndarray
    Q: jnp.ndarray
    U: jnp.ndarray
    H: jnp.ndarray
    R: jnp.ndarray
    d: jnp.ndarray
    h: jnp.ndarray


def n_state(bank: Bank) -> int:
    return bank.F.shape[-1]


def n_targets(bank: Bank) -> int:
    return bank.F.shape[0]


# ----------------------------------------------------------------------------------------------- bank builders
def scalar_bank(types, qct, d, h=None, f_cv=F_CV, f_ct=F_CT, q_cv=Q_CV, hh=H_SCALAR, r=R_SCALAR) -> Bank:
    """Scalar (L = 1) bank, paper §IV-A.  types: list of 'reckless'/'cautious'; qct: Q^CT per target; d: weights."""
    n = len(types)
    qct = np.broadcast_to(np.asarray(qct, float), (n,))
    d = np.broadcast_to(np.asarray(d, float), (n,))
    h = np.zeros(n) if h is None else np.broadcast_to(np.asarray(h, float), (n,))
    F = np.zeros((n, 2, 1, 1)); Q = np.zeros((n, 2, 1, 1))
    F[:, 0, 0, 0] = f_cv; F[:, 1, 0, 0] = f_ct
    Q[:, 0, 0, 0] = q_cv; Q[:, 1, 0, 0] = qct
    U = np.array([TYPE_U[t] for t in types], float)
    return Bank(jnp.asarray(F), jnp.asarray(Q), jnp.asarray(U), jnp.array([[hh]], float), jnp.array([[r]], float),
                jnp.asarray(d), jnp.asarray(h))


def f_cv_matrix(ts: float = TS) -> np.ndarray:
    """Eq (36): I_2 kron [[1, Ts], [0, 1]] for the state [x, xdot, y, ydot]."""
    return np.kron(np.eye(2), np.array([[1.0, ts], [0.0, 1.0]]))


def f_ct_matrix(omega: float = OMEGA_RAD, ts: float = TS, variant: str = "printed") -> np.ndarray:
    """Eq (37) coordinated turn.  variant 'printed' = the paper's matrix as typeset (velocity rows [0,c,0,s], [0,-s,0,c]);
    'standard' = the self-consistent CT model (velocity rows [0,c,0,-s], [0,s,0,c])."""
    s, c = np.sin(omega * ts), np.cos(omega * ts)
    if variant == "printed":
        return np.array([[1, s / omega, 0, (c - 1) / omega],
                         [0, c, 0, s],
                         [0, (1 - c) / omega, 1, s / omega],
                         [0, -s, 0, c]], float)
    if variant == "standard":
        return np.array([[1, s / omega, 0, (c - 1) / omega],
                         [0, c, 0, -s],
                         [0, (1 - c) / omega, 1, s / omega],
                         [0, s, 0, c]], float)
    raise ValueError(f"unknown F^CT variant {variant!r}")


def q_shape_matrix(ts: float = TS) -> np.ndarray:
    """Eq (38) shape: I_2 kron [[Ts^3/3, Ts^2/2], [Ts^2/2, Ts]]."""
    return np.kron(np.eye(2), np.array([[ts ** 3 / 3, ts ** 2 / 2], [ts ** 2 / 2, ts]]))


H_MATRIX = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])  # eq (39)


def matrix_bank(types, qct, d, h=None, ft_variant: str = "printed", omega: float = OMEGA_RAD, ts: float = TS) -> Bank:
    """4-D (L = 4) bank, paper §IV-D.  qct: q^CT per target (q^CV = 1)."""
    n = len(types)
    qct = np.broadcast_to(np.asarray(qct, float), (n,))
    d = np.broadcast_to(np.asarray(d, float), (n,))
    h = np.zeros(n) if h is None else np.broadcast_to(np.asarray(h, float), (n,))
    F = np.zeros((n, 2, 4, 4)); Q = np.zeros((n, 2, 4, 4))
    F[:, 0] = f_cv_matrix(ts); F[:, 1] = f_ct_matrix(omega, ts, ft_variant)
    qs = q_shape_matrix(ts)
    Q[:, 0] = Q_CV_MATRIX * qs
    Q[:, 1] = qct[:, None, None] * qs
    U = np.array([TYPE_U[t] for t in types], float)
    return Bank(jnp.asarray(F), jnp.asarray(Q), jnp.asarray(U), jnp.asarray(H_MATRIX),
                jnp.asarray(R_MATRIX_DIAG * np.eye(2)), jnp.asarray(d), jnp.asarray(h))


def to_general(bank: Bank) -> Bank:
    """Identity; kept for symmetry with phi(..., force_general=True) tests."""
    return bank


# ----------------------------------------------------------------------------------------------- the TEC map
def _inv_small(S):
    """Inverse of a (..., k, k) SPD matrix; closed form for k = 1, 2 (speed), jnp.linalg.inv otherwise."""
    k = S.shape[-1]
    if k == 1:
        return 1.0 / S
    if k == 2:
        a, b, c, d = S[..., 0, 0], S[..., 0, 1], S[..., 1, 0], S[..., 1, 1]
        det = a * d - b * c
        return jnp.stack([jnp.stack([d, -b], -1), jnp.stack([-c, a], -1)], -2) / det[..., None, None]
    return jnp.linalg.inv(S)


def kf_modes(bank: Bank, P, act, force_general: bool = False):
    """Per-mode covariance after the slot's transition: P-bar^m (act = 0, eq 5) or P-hat^m (act = 1, eqs 4-6).

    Returns (N, M, L, L).  The update is written in the symmetric form P-hat = P-bar - P-bar H' S^-1 H P-bar, which equals
    (I - K H) P-bar of eq (4) exactly in real arithmetic.  act (N,) scales the correction (0 or 1)."""
    L = bank.F.shape[-1]
    a = act[:, None, None, None]
    if L == 1 and bank.H.shape[0] == 1 and not force_general:
        f = bank.F[..., 0, 0]; q = bank.Q[..., 0, 0]; h = bank.H[0, 0]; r = bank.R[0, 0]
        pbar = f * f * P[:, 0, 0][:, None] + q                       # (N, M)
        corr = (pbar * h) ** 2 / (h * h * pbar + r)
        return (pbar - act[:, None] * corr)[..., None, None]
    Fm, Qm = bank.F, bank.Q
    Pb = Fm @ P[:, None] @ jnp.swapaxes(Fm, -1, -2) + Qm            # (N, M, L, L)   eq (5)
    PH = Pb @ bank.H.T                                              # (N, M, L, Lz)
    S = bank.H @ PH + bank.R                                        # (N, M, Lz, Lz) innovation covariance
    corr = PH @ _inv_small(S) @ jnp.swapaxes(PH, -1, -2)
    Ph = Pb - a * corr
    return 0.5 * (Ph + jnp.swapaxes(Ph, -1, -2))


def mode_probs(bank: Bank, act):
    """u^{act} per target: (N, M)."""
    return bank.U[:, 0] * (1.0 - act)[:, None] + bank.U[:, 1] * act[:, None]


def phi(bank: Bank, P, act, force_general: bool = False):
    """Eqs (3)/(7): P_next = sum_m u^{act,m} P^m (deterministic mode-mixture covariance)."""
    Pm = kf_modes(bank, P, act, force_general)
    return jnp.einsum("nm,nmij->nij", mode_probs(bank, act), Pm)


def phi_sampled(bank: Bank, P, act, key):
    """Extension X3: draw the mode m ~ u^{act} per target and apply that mode's covariance update only."""
    Pm = kf_modes(bank, P, act)
    m = jax.random.categorical(key, jnp.log(mode_probs(bank, act)), axis=-1)  # (N,)
    return jnp.take_along_axis(Pm, m[:, None, None, None], axis=1)[:, 0]


def stat(P):
    """s(P) = tr(P)/L (eq 29-32; the scalar case is P itself)."""
    return jnp.trace(P, axis1=-2, axis2=-1) / P.shape[-1]


def step_cost(bank: Bank, P, act):
    """Eq (8) summed over targets: sum_n d_n tr(P_n)/L + h_n a_n, charged on the START-of-slot state."""
    return jnp.sum(bank.d * stat(P) + bank.h * act)


def cost_vec(bank: Bank, P, act):
    return bank.d * stat(P) + bank.h * act


# ----------------------------------------------------------------------------------------------- environment checksum
REF_PATTERN = [1, 0, 0, 1] * 8  # fixed action pattern of the frozen reference trajectories
REF_FIXTURE = __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "tests", "fixtures", "reference_traj.npz")
REF_TOL = 1e-12                 # relative tolerance


def reference_trajectory_error(path: str = REF_FIXTURE) -> float:
    """Max deviation (relative to the per-step matrix scale) between the JAX model and the frozen float64 reference trajectories (generated by the
    independent numpy oracle, scripts/make_reference_traj.py).  Eval entry points abort if this exceeds REF_TOL."""
    ref = np.load(path)
    worst = 0.0
    for key in ref.files:
        if key == "matrix_P0":
            continue
        kind, rest = key.split("_", 1)
        if kind == "scalar":
            bank = scalar_bank([rest], [4.0], 1.0); P = jnp.array([[[1.3]]])
        else:
            var, typ = rest.split("_")
            bank = matrix_bank([typ], [4.0], 1.0, ft_variant=var); P = jnp.asarray(ref["matrix_P0"])[None]
        # matrix_bank uses q^CV = 1 and q^CT = 4 exactly as the fixture
        tr = [np.asarray(P[0])]
        for a in REF_PATTERN:
            P = phi(bank, P, jnp.array([float(a)])); tr.append(np.asarray(P[0]))
        tr = np.array(tr)
        scale = np.max(np.abs(ref[key]), axis=(-2, -1), keepdims=True)    # per-step matrix scale (zero entries carry rounding noise)
        worst = max(worst, float(np.max(np.abs(tr - ref[key]) / scale)))
    return worst


def assert_env_checksum(path: str = REF_FIXTURE) -> float:
    err = reference_trajectory_error(path)
    if not err <= REF_TOL:
        raise RuntimeError(f"[ERROR] environment checksum failed: max rel dev {err:.3e} > {REF_TOL:.0e}")
    return err

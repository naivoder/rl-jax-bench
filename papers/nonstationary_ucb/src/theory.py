"""Closed-form constants and bounds of the paper (numpy only), unit-tested against hand derivation numbers.

Tags: Theorem 1 / Remark 2 (D-UCB), Theorem 7 / Remark 8 (SW-UCB), Theorem 18 / Remark 20 (deviation
inequality), Theorem 13 / Corollary 14 (lower bound; D-13 constant discrepancy).
"""
from __future__ import annotations

import math

import numpy as np


def eta_of_xi(xi: float) -> float:
    """hand derivation: eta = 4 sqrt(1 - 1/(2 xi)) makes 2 xi (1 - eta^2/16) = 1 (xi > 1/2)."""
    return 4.0 * math.sqrt(1.0 - 1.0 / (2.0 * xi))


def ineq16_gap(eta):
    """4/((1+eta)^{1/4} + (1+eta)^{-1/4})^2 - (1 - eta^2/16) >= 0 (paper Eq. 16)."""
    eta = np.asarray(eta, dtype=np.float64)
    q = (1.0 + eta) ** 0.25
    return 4.0 / (q + 1.0 / q) ** 2 - (1.0 - eta ** 2 / 16.0)


def remark20_constant(eta: float = 0.3) -> float:
    """8/((1+eta)^{1/4} + (1+eta)^{-1/4})^2 at eta = 0.3 -> 1.99142 (paper Remark 20)."""
    q = (1.0 + eta) ** 0.25
    return 8.0 / (q + 1.0 / q) ** 2


def n_disc(gamma: float, t: float) -> float:
    """D-4: n_t(gamma) = (1 - gamma^t)/(1 - gamma)."""
    return (1.0 - gamma ** t) / (1.0 - gamma)


def thm18_bound(delta, n_t: float, B: float = 1.0, eta_grid=None):
    """Theorem 18: P((S - M)/sqrt(N(gamma^2)) > delta) <= min_eta ceil(log n_t / log(1+eta)) exp(-(2 delta^2/B^2)(1 - eta^2/16))."""
    if eta_grid is None:
        eta_grid = np.linspace(1e-3, 4.0 - 1e-6, 40000)
    eta_grid = np.asarray(eta_grid, dtype=np.float64)
    d = np.atleast_1d(np.asarray(delta, dtype=np.float64))
    slices = np.ceil(math.log(n_t) / np.log1p(eta_grid))
    b = slices[None, :] * np.exp(-(2.0 * d[:, None] ** 2 / B ** 2) * (1.0 - eta_grid[None, :] ** 2 / 16.0))
    return b.min(1)


def D_gamma(gamma: float, xi: float, K: int) -> float:
    """Theorem 1 burn-in D(gamma) = log((1 - gamma) xi log n_K(gamma)) / log gamma (hand derivation)."""
    return math.log((1.0 - gamma) * xi * math.log(n_disc(gamma, K))) / math.log(gamma)


def B_gamma(gamma: float, T: int, xi: float, Delta: float, B: float = 1.0) -> float:
    eta = eta_of_xi(xi)
    gp = gamma ** (1.0 / (1.0 - gamma))
    first = 16.0 * B ** 2 * xi / (gp * Delta ** 2) * math.ceil(T * (1.0 - gamma)) / (T * (1.0 - gamma))
    L = -math.log(1.0 - gamma)
    second = 2.0 * math.ceil(L / math.log1p(eta)) / (L * (1.0 - gp))
    return first + second


def B_gamma_limit(xi: float, Delta: float, B: float = 1.0) -> float:
    """Remark 2: B(gamma) -> 16 e B^2 xi / Delta^2 + 2/((1 - e^{-1}) log(1 + eta)) as gamma -> 1."""
    return 16.0 * math.e * B ** 2 * xi / Delta ** 2 + 2.0 / ((1.0 - math.exp(-1.0)) * math.log1p(eta_of_xi(xi)))


def C_gamma(gamma: float, xi: float, K: int) -> float:
    return (gamma - 1.0) / (math.log(1.0 - gamma) * math.log(gamma)) * math.log((1.0 - gamma) * xi * math.log(n_disc(gamma, K)))


def thm1_bound(gamma: float, T: int, upsilon: int, xi: float, Delta: float, K: int, B: float = 1.0) -> float:
    """Theorem 1: E N~_T(i) <= B(gamma) T (1-gamma) log(1/(1-gamma)) + C(gamma) Upsilon log(1/(1-gamma))/(1-gamma)."""
    L = math.log(1.0 / (1.0 - gamma))
    return B_gamma(gamma, T, xi, Delta, B) * T * (1.0 - gamma) * L + C_gamma(gamma, xi, K) * upsilon * L / (1.0 - gamma)


def C_tau(tau: int, T: int, xi: float, Delta: float, B: float = 1.0) -> float:
    eta = eta_of_xi(xi)
    return (4.0 * B ** 2 * xi / Delta ** 2 * math.ceil(T / tau) / (T / tau)
            + (2.0 / math.log(tau)) * math.ceil(math.log(tau) / math.log1p(eta)))


def C_tau_limit(xi: float, Delta: float, B: float = 1.0) -> float:
    """Remark 8: C(tau) -> 4 B^2 xi / Delta^2 + 2/log(1 + eta)."""
    return 4.0 * B ** 2 * xi / Delta ** 2 + 2.0 / math.log1p(eta_of_xi(xi))


def thm7_bound(tau: int, T: int, upsilon: int, xi: float, Delta: float, B: float = 1.0) -> float:
    """Theorem 7: E N~_T(i) <= C(tau) T log tau / tau + tau Upsilon + log^2 tau."""
    return C_tau(tau, T, xi, Delta, B) * T * math.log(tau) / tau + tau * upsilon + math.log(tau) ** 2


def kl_bern(p: float, q: float) -> float:
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def lower_bound_constants(mu1: float, muK: float, nu: float) -> dict:
    """Theorem 13 C(mu) = 32 delta (mu1 - muK)/(27 alpha) and the re-derived C' = 8 delta (mu1 - muK)/(27 alpha) (D-13)."""
    alpha = kl_bern(muK, nu)
    delta = nu - mu1
    return {"alpha": alpha, "delta": delta, "C_paper": 32 * delta * (mu1 - muK) / (27 * alpha),
            "C_rederived_no_sqrt2": 16 * delta * (mu1 - muK) / (27 * alpha),
            "C_rederived": 8 * delta * (mu1 - muK) / (27 * alpha)}

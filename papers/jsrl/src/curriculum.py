"""JSRL curriculum: stage list and the beta threshold state machine (paper Section 4.2 and Appendix A.4).

No official code exists.  Choices the paper leaves open are ASSUMED and tagged:
  * stage values: H1 is first rounded to an integer h_0, then h_i = floor(round(H1)(1 - i/n) + 0.5), h_n = 0   (App. A.4 'Curriculum Stages'; ASSUMED rounding:
    the step is round(H1)/n, not H1/n; n + 1 values starting at H1 per Sec. 4.2 / Alg. 1, App. A.4's n-value reading is NOT flown)
  * evaluations of the COMBINED policy every E env steps; the moving average MA is over the last M evaluations of the CURRENT
    stage (window reset at each advance; `reset_on_adv=0` = steventango's rolling window)
  * the first full MA of the run only sets the best, never advances (as in steventango/jumpstart-rl)
  * advance iff MA >= B - tol |B|, B = best MA so far (global, never reset), B updated after the test; one advance per evaluation
Pure functions of arrays (jit/vmap friendly), plus a numpy reference used only by tests (`CurRef`).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def stage_list(h1: float, n: int) -> list[int]:
    """h_0 = round(H1), h_i = floor(h_0 (1 - i/n) + 0.5) (H1 rounded FIRST, then round half up), h_n = 0 exactly.  Always n + 1 values, non-increasing."""
    h1 = int(np.floor(float(h1) + 0.5))
    out = [int(np.floor(h1 * (1.0 - i / n) + 0.5)) for i in range(n + 1)]
    out[0], out[-1] = h1, 0
    return out


def cur_init(M: int):
    return dict(stage=jnp.zeros((), jnp.int32), n=jnp.zeros((), jnp.int32), win=jnp.zeros((M,), jnp.float32),
                best=jnp.zeros((), jnp.float32), has_best=jnp.zeros((), bool))


def cur_step(cs: dict, R, tol, n_stages: int, reset_on_adv, active):
    """One evaluation R (the combined-policy mean return at the current stage).  Returns (new_state, advanced: bool).
    `active` False (non-curriculum arms) freezes `stage` but keeps the bookkeeping."""
    M = cs["win"].shape[0]
    idx = cs["n"] % M
    win = cs["win"].at[idx].set(jnp.asarray(R, jnp.float32))
    n = cs["n"] + 1
    full = n >= M                                   # all M slots were written in this stage (n counts since the last reset)
    ma = win.mean()
    thr = cs["best"] - tol * jnp.abs(cs["best"])
    adv = active & full & cs["has_best"] & (ma >= thr) & (cs["stage"] < n_stages)
    best = jnp.where(full, jnp.where(cs["has_best"], jnp.maximum(cs["best"], ma), ma), cs["best"])
    n = jnp.where(adv & (reset_on_adv > 0), 0, n)
    return dict(stage=cs["stage"] + adv.astype(jnp.int32), n=n.astype(jnp.int32), win=win, best=best,
                has_best=cs["has_best"] | full), adv


class CurRef:
    """Plain-python reference of the same state machine (tests only)."""

    def __init__(self, M, tol, n_stages, reset_on_adv=True):
        self.M, self.tol, self.ns, self.reset = M, tol, n_stages, reset_on_adv
        self.stage, self.win, self.best, self.has_best = 0, [], None, False
        self.since_start = []

    def step(self, R):
        self.win.append(float(np.float32(R)))
        full = len(self.win) >= self.M
        adv = False
        if full:
            ma = float(np.mean(np.asarray(self.win[-self.M:], np.float32)))
            if self.has_best:
                thr = self.best - self.tol * abs(self.best)
                if ma >= thr and self.stage < self.ns:
                    adv = True
                self.best = max(self.best, ma)
            else:
                self.best, self.has_best = ma, True
        if adv:
            self.stage += 1
            if self.reset:
                self.win = []
        return adv

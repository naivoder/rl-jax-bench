"""Environment definitions ("data") for the categorical POMDPs of Baisero & Amato (2022), App. C.1.

This paper has no dataset: the "data" are the environment definition files. This module is our own parser for the
subset of Cassandra's ``.pomdp`` format used by the four gym-pomdps files in scope (heavenhell_3/4,
shopping_5/6.v1), plus pinned SHA256 hashes. It refuses to load a file whose hash differs.

Semantics follow the format (and gym-pomdps@b52e520, checked by tests/test_official_env.py): later lines override
earlier ones element-wise, ``*`` is a wildcard, ``T: a identity`` sets T[a]=I, ``T: a: s reset`` marks (a, s) as
terminal (gym-pomdps ``episodic=True``: ``done = reset[a, s]``) and sets that T row to the start distribution
(rl_parsers@b2197e3; only affects the never-used observation emitted on the terminal step), ``O: a: s': o p`` and ``R: a: s: s': o r``.
Reward lines in these four files never depend on (s', o); we assert that and store R as [A, S].
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures")

# pinned hashes (gym-pomdps b52e520; byte-identical at 81ffa37)
POMDP_FILES = {
    "heavenhell_3": ("heavenhell_3.pomdp", "f02db7acc77cd1d011d75d19a2864892603ea10c783c4ccc9a75efe8ef47f4c0"),
    "heavenhell_4": ("heavenhell_4.pomdp", "46afdba42aeb11a652ffae71c47e6b1f5aadc6f539b709e1b5d338b4b01da5c9"),
    "shopping_5": ("shopping_5.v1.pomdp", "65eb7c958eb75f88a13b2dc44757ce7601ed54586e2c4a179834956f7202523d"),
    "shopping_6": ("shopping_6.v1.pomdp", "7d089ed8081647007c23566167659f4ca269ce7a01807ba3a7a4555971dfa0a9"),
}


@dataclass(frozen=True)
class CategoricalPOMDP:
    name: str
    states: tuple
    actions: tuple
    observations: tuple
    discount: float
    start: np.ndarray      # [S]
    T: np.ndarray          # [S, A, S']  (gym-pomdps layout)
    O: np.ndarray          # [A, S', O]  (standard Cassandra: depends on (a, s'))
    R: np.ndarray          # [S, A]
    D: np.ndarray          # [S, A] bool, terminal after taking a in s
    sha256: str

    @property
    def n_states(self):
        return len(self.states)

    @property
    def n_actions(self):
        return len(self.actions)

    @property
    def n_obs(self):  # real observations; the reset observation is index n_obs (extra, zero embedding)
        return len(self.observations)

    def arrays_sha256(self) -> str:
        h = hashlib.sha256()
        for a in (self.start, self.T, self.O, self.R, self.D.astype(np.uint8)):
            h.update(np.ascontiguousarray(a, dtype=np.float64 if a.dtype != np.uint8 else np.uint8).tobytes())
        return h.hexdigest()


def _names(spec: str):
    toks = spec.split()
    if len(toks) == 1 and toks[0].isdigit():
        return tuple(str(i) for i in range(int(toks[0])))
    return tuple(toks)


def _idx(tok: str, names: tuple):
    if tok == "*":
        return slice(None)
    if tok in names:
        return names.index(tok)
    if tok.isdigit() and int(tok) < len(names):
        return int(tok)
    raise ValueError(f"unknown identifier {tok!r}")


def parse_pomdp(text: str, name: str = "", sha: str = "") -> CategoricalPOMDP:
    lines = [ln.split("#", 1)[0].strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    hdr = {}
    for ln in lines:
        key = ln.split(":", 1)[0].strip()
        if key in ("discount", "values", "states", "actions", "observations"):
            hdr[key] = ln.split(":", 1)[1].strip()
    if hdr.get("values") != "reward":
        raise ValueError("only `values: reward` supported")
    S, A, Ob = _names(hdr["states"]), _names(hdr["actions"]), _names(hdr["observations"])
    nS, nA, nO = len(S), len(A), len(Ob)
    T = np.zeros((nA, nS, nS))
    O = np.zeros((nA, nS, nO))
    R = np.zeros((nA, nS))
    D = np.zeros((nA, nS), bool)
    start = None
    for ln in lines:
        if ln.startswith("start include:"):
            inc = [_idx(t, S) for t in ln.split(":", 1)[1].split()]
            start = np.zeros(nS)
            start[inc] = 1.0 / len(inc)
        elif ln.startswith("start:"):
            vals = ln.split(":", 1)[1].split()
            if len(vals) == nS:
                start = np.array([float(v) for v in vals])
            else:
                start = np.zeros(nS)
                start[_idx(vals[0], S)] = 1.0
        elif ln.startswith("T:"):
            parts = [p.strip() for p in ln[2:].split(":")]
            if len(parts) == 1:  # "T: a identity"
                a, kw = parts[0].split()
                if kw != "identity":
                    raise ValueError(ln)
                T[_idx(a, A)] = np.eye(nS)
            elif len(parts) == 2:  # "T: a: s reset" -> terminal; T row = start distribution (rl_parsers semantics)
                s, kw = parts[1].split()
                if kw != "reset":
                    raise ValueError(ln)
                if start is None:
                    raise ValueError("`reset` before `start`")
                T[_idx(parts[0], A), _idx(s, S)] = start
                D[_idx(parts[0], A), _idx(s, S)] = True
            elif len(parts) == 3:  # "T: a: s: s' p"
                s2, p = parts[2].split()
                T[_idx(parts[0], A), _idx(parts[1], S), _idx(s2, S)] = float(p)
            else:
                raise ValueError(ln)
        elif ln.startswith("O:"):
            parts = [p.strip() for p in ln[2:].split(":")]
            if len(parts) != 3:
                raise ValueError(f"unsupported O line {ln!r}")
            o, p = parts[2].split()
            O[_idx(parts[0], A), _idx(parts[1], S), _idx(o, Ob)] = float(p)
        elif ln.startswith("R:"):
            parts = [p.strip() for p in ln[2:].split(":")]
            if len(parts) != 4:
                raise ValueError(f"unsupported R line {ln!r}")
            o, r = parts[3].split()
            if parts[2] != "*" or o != "*":
                raise ValueError(f"reward depends on (s', o): {ln!r} (not supported by our [A,S] layout)")
            R[_idx(parts[0], A), _idx(parts[1], S)] = float(r)
    if start is None:
        start = np.full(nS, 1.0 / nS)
    # consistency: transition / observation rows are distributions
    if not np.allclose(T.sum(-1), 1.0):
        raise ValueError("T rows do not sum to 1")
    if not np.allclose(O.sum(-1), 1.0):
        raise ValueError("O rows do not sum to 1")
    return CategoricalPOMDP(name=name, states=S, actions=A, observations=Ob, discount=float(hdr["discount"]),
                            start=start, T=T.transpose(1, 0, 2).copy(), O=O, R=R.T.copy(), D=D.T.copy(), sha256=sha)


def load_pomdp(name: str, fixtures_dir: str = FIXTURES) -> CategoricalPOMDP:
    fname, sha = POMDP_FILES[name]
    with open(os.path.join(fixtures_dir, fname), "rb") as f:
        raw = f.read()
    got = hashlib.sha256(raw).hexdigest()
    if got != sha:
        raise RuntimeError(f"{fname}: SHA256 {got} != pinned {sha}; refusing to run")
    return parse_pomdp(raw.decode(), name=name, sha=sha)

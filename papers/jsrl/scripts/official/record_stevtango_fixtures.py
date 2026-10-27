"""Record fixtures from the third-party SB3 JSRL (steventango/jumpstart-rl e5f690fe; NO official JSRL code exists).
Run in the reference venv built by scripts/official/setup.sh (NOT the project venv), importing their package read-only:

    PYTHONPATH=$OFFICIAL_ROOT/jumpstart-rl/src $OFFICIAL_ROOT/.venv/bin/python scripts/official/record_stevtango_fixtures.py tests/fixtures/stevtango_parity.npz

Their code is only IMPORTED and executed here; nothing of it is copied into src/ or tests/.  Recorded (all produced by running their classes):
  cb_*   the REAL `JSRLAfterEvalCallback._on_step` + REAL `JSRLPolicy.update_horizon` driven by synthetic evaluation sequences: the horizon_step after each evaluation
  sw_*   the REAL `JSRLPolicy.predict` mask (which steps the guide acts on) for a grid of (horizon, within-episode step, training flag, jsrl_evaluation flag)
  sl_*   their stage list `np.arange(H1, -1, -H1 // n)` for a grid of (H1, n)
  rnd_*  their random-strategy draws (`np.random.choice(max_horizon)`), seeded
"""
import sys
from types import SimpleNamespace

import numpy as np
import gymnasium as gym
from stable_baselines3.common.base_class import Logger  # noqa: F401
from jsrl.jsrl import JSRLAfterEvalCallback, get_jsrl_policy

OUT = sys.argv[1]


class NullLogger:
    def record(self, *a, **k): pass
    def dump(self, *a, **k): pass


class StubExploration:
    """Minimal stand-in for an SB3 policy class: predict() returns 2, the guide returns 1, so the output shows who acted."""
    def __init__(self, *args, **kwargs):
        self.action_space = gym.spaces.Box(-5.0, 5.0, shape=(1,), dtype=np.float32)
        self.training = True

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        return np.full((len(observation), 1), 2.0, np.float32), None


class StubGuide:
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        return np.full((len(observation), 1), 1.0, np.float32), None


JSRLPolicy = get_jsrl_policy(StubExploration)


def make_policy(horizons, tol, M, strategy="curriculum", max_horizon=0):
    return JSRLPolicy(guide_policy=StubGuide(), max_horizon=max_horizon, horizons=list(horizons), tolerance=tol,
                      strategy=strategy, window_size=M)


def sequences():
    rng = np.random.default_rng(2026)
    T = 160
    t = np.arange(T)
    seqs = {
        "stall_down": 10.0 - 0.05 * t, "monotone_up": 1.0 + 0.1 * t, "constant": np.full(T, 5.0),
        "noisy_plateau": 6.0 + rng.normal(0, 0.4, T), "up_then_collapse": np.where(t < 60, 1 + 0.1 * t, 2.0 + 0 * t),
        "random_walk": np.cumsum(rng.normal(0, 0.3, T)) + 8.0, "zeros": np.zeros(T),
        "negative_walk": np.cumsum(rng.normal(0, 0.3, T)) - 8.0,           # exercises abs() in the tolerance
        "integer_ties": rng.integers(0, 4, T).astype(np.float64),          # exact ties and equality at the boundary (tol 0)
        "dyadic_ties": rng.integers(0, 16, T) / 8.0,                        # dyadic values: tol 0.25/0.5 boundaries are exact in float32
    }
    for i in range(30):
        seqs[f"rand{i}"] = np.cumsum(rng.normal(0, 1, T) * rng.uniform(0.05, 1)) + rng.normal(5, 3)
    names = list(seqs)
    return names, np.stack([seqs[n] for n in names]).astype(np.float32)


def run_callback(seq, horizons, tol, M):
    pol = make_policy(horizons, tol, M)
    cb = JSRLAfterEvalCallback(pol, NullLogger())
    out = np.zeros(len(seq), np.int16)
    amb = np.zeros(len(seq), bool)
    for i, R in enumerate(seq):
        cb.parent = SimpleNamespace(last_mean_reward=np.float32(R))
        pol.jsrl_evaluation = True
        # ambiguity flag: their threshold `tolerated` is a float64 built from a float32 MA (numpy 1.x promotion: python float * np.float32 -> float64),
        # ours is float32 end to end; the two can only disagree when MA sits within ~2 float32 ulp of the threshold (exact ties)
        w = np.roll(cb.mean_rewards, 1); w[0] = np.float32(R)
        ma = np.mean(w)
        thr = cb.tolerated_moving_mean_reward
        if np.isfinite(thr) and np.isfinite(ma):
            amb[i] = bool((float(ma) >= float(thr)) != (float(ma) >= float(np.float32(thr))))   # the f64 and the f32-rounded threshold disagree on this evaluation
        cb._on_step()
        out[i] = pol.horizon_step
    return out, amb


def main():
    names, seqs = sequences()
    cfgs = [(M, tol, nh) for M in (1, 2, 3, 5) for tol in (0.0, 0.05, 0.15, 0.25, 0.5) for nh in (11, 5)]  # nh = number of stages + 1
    traces = np.zeros((len(cfgs), len(seqs), seqs.shape[1]), np.int16)
    ambig = np.zeros(traces.shape, bool)
    for ci, (M, tol, nh) in enumerate(cfgs):
        horizons = [10 * (nh - 1 - i) for i in range(nh)]   # last stage is h = 0 (exercises their `horizon <= 0` early return)
        for si in range(len(seqs)):
            traces[ci, si], ambig[ci, si] = run_callback(seqs[si], horizons, tol, M)
    # switching rule
    hs = [0, 1, 2, 5, 60]; ts = np.arange(0, 9)
    sw = np.zeros((len(hs), 2, 2, len(ts)), np.int8)        # [horizon, training, jsrl_evaluation, t] -> 1 guide acted, 2 explorer acted
    for hi, h in enumerate(hs):
        for tr in (0, 1):
            for je in (0, 1):
                pol = make_policy([h], 0.0, 1)
                pol.training = bool(tr); pol.jsrl_evaluation = bool(je)
                obs = np.zeros((len(ts), 3), np.float32)
                a, _ = pol.predict(obs, ts.astype(np.int32), None, None, True)
                sw[hi, tr, je] = a[:, 0].astype(np.int8)
    # stage lists
    sl_h, sl_n, sl_list, sl_len = [], [], [], []
    for H1 in range(1, 201):
        for n in range(1, 21):
            sl_h.append(H1); sl_n.append(n)
            lst = np.arange(H1, -1, -H1 // n)         # their expression verbatim; `-H1 // n` is (-H1) // n = -ceil(H1 / n), so the step is ceil(H1/n)
            sl_len.append(len(lst)); sl_list.append(np.pad(lst, (0, 25 - len(lst)), constant_values=-9).astype(np.int32)[:25])
    # random strategy
    np.random.seed(0)
    pol = make_policy([0], 0.0, 1, strategy="random", max_horizon=60)
    draws = []
    for _ in range(3000):
        pol.update_horizon(); draws.append(int(pol.horizons[0]))
    np.savez_compressed(OUT, names=np.array(names), seqs=seqs, cfgs=np.array(cfgs, dtype=np.float64), traces=traces, ambig=ambig,
                        sw_h=np.array(hs), sw_t=ts, sw=sw, sl_h=np.array(sl_h), sl_n=np.array(sl_n), sl_len=np.array(sl_len),
                        sl_list=np.stack(sl_list), rnd=np.array(draws))
    print("[INFO] ambiguous (threshold-tie) evaluations:", int(ambig.sum()), "in", int(ambig.any(axis=2).sum()), "of", ambig.shape[0] * ambig.shape[1], "(cfg,seq) pairs")
    print("[INFO] wrote", OUT, "cfgs", len(cfgs), "seqs", len(seqs), "advances recorded", int((np.diff(traces, axis=2) > 0).sum()))
    print("[INFO] switch grid horizon 60 -> guide mask at t=0..8 (training=1, jsrl_eval=1):", sw[4, 1, 1])
    print("[INFO] switch grid horizon 0, training=0, jsrl_eval=0:", sw[0, 0, 0])


if __name__ == "__main__":
    main()

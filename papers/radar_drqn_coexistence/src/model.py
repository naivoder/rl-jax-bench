"""Networks (thin): the Q-networks live in the shared recurrent double-DQN core (algorithm.py re-exports
rl_jax_bench.drqn). This module re-exports them and builds the per-arm configs.

Architecture (Table I, §III-C/D): 7 -> FC256-ReLU -> FC128-ReLU -> FC84-ReLU -> [LSTM 84] -> linear 15.
"""
from __future__ import annotations

from algorithm import LSTMCell, QConfig, QNetwork, initial_carry, make_network  # noqa: F401
from data import N_ACT, OBS_DIM

# Arms. They differ only in (lstm, trace_len, double); every other value is Table I and shared.
ARMS = {
    "dqn":        dict(lstm=0,  trace_len=1,  double=False),   # §III-C, Eq. 11
    "ddqn":       dict(lstm=0,  trace_len=1,  double=True),    # double-Q, no LSTM
    "ddqn_tr10":  dict(lstm=0,  trace_len=10, double=True),    # double-Q + 32x10 traces, no LSTM
    "ddrqn":      dict(lstm=84, trace_len=10, double=True),    # §III-D, Eq. 14 + LSTM 84 (Table I)
    # DDRQN variants T-A / T-B, declared before the full-scale run (see the README; never the headline arm):
    "ddrqn_win":  dict(lstm=84, trace_len=10, double=True, window=True),                     # T-A Keras sliding window
    "ddrqn_hs":   dict(lstm=84, trace_len=10, double=True, recurrent_activation="hard_sigmoid"),  # T-B pre-TF2 Keras gates
    # review arms (declared before the review runs, see the README; never the headline arm):
    "dqn_lr1e4":  dict(lstm=0,  trace_len=1,  double=False, lr=1e-4),   # lr sweep (toy-check control step size)
    "ddrqn_lr1e4": dict(lstm=84, trace_len=10, double=True, lr=1e-4),   # lr sweep (toy-check control step size)
    "ddrqn_noreset": dict(lstm=84, trace_len=10, double=True, act_reset_period=1000),  # actor carry reset per CPI only
    # Recheck (2026-10-16) sensitivity arms (the README pre-declaration; never the headline arm):
    "dqn_sgd":    dict(lstm=0,  trace_len=1,  double=False, optimizer="sgd"),                 # plain SGD (Eq. 9)
    "ddrqn_sgd":  dict(lstm=84, trace_len=10, double=True, optimizer="sgd"),
    "dqn_sgd_k15":   dict(lstm=0,  trace_len=1,  double=False, optimizer="sgd", loss_scale=1 / 15),  # Keras full-vector MSE
    "ddrqn_sgd_k15": dict(lstm=84, trace_len=10, double=True, optimizer="sgd", loss_scale=1 / 15),
    "ddrqn_upd10": dict(lstm=84, trace_len=10, double=True, update_every=10),                 # one step per stored trace
    "dqn_orelu":  dict(lstm=0,  trace_len=1,  double=False, output_relu=True),                # §III-C literal output ReLU
}


def arm_config(arm: str, cfg: dict) -> QConfig:
    """QConfig of an arm from the run config (src/configs/*.yaml, every value traced in sources.yaml)."""
    a = ARMS[arm]
    L = a["trace_len"]
    # window arm: one stored 10-observation window per transition, so the buffer holds 2000 windows (Table I, transitions)
    n_buf = cfg["buffer_transitions"] if a.get("window") else cfg["buffer_transitions"] // L
    return QConfig(n_actions=N_ACT, obs_dim=OBS_DIM, hidden=tuple(cfg["hidden_sizes"]), lstm=cfg["lstm_size"] if a["lstm"] else 0,
                   double=a["double"], gamma=cfg["gamma"], lr=a.get("lr", cfg["lr"]), adam_b1=cfg["adam_b1"],
                   adam_b2=cfg["adam_b2"], adam_eps=cfg["adam_eps"], trace_len=L, burn_in=cfg["burn_in"] if L > 1 else 0,
                   batch_size=cfg["batch_size"], buffer_traces=n_buf,
                   target_period=cfg["target_update_steps"], store_carry=False,
                   recurrent_activation=a.get("recurrent_activation", "sigmoid"),
                   optimizer=a.get("optimizer", "adam"), loss_scale=float(a.get("loss_scale", 1.0)),
                   update_every=int(a.get("update_every", 1)), output_relu=bool(a.get("output_relu", False)))


def is_window(arm: str) -> bool:
    return bool(ARMS[arm].get("window", False))


def act_reset_period(arm: str) -> int:
    """Pulses between act-time carry resets: trace_len (shipped) unless the arm overrides it (variant arms)."""
    return int(ARMS[arm].get("act_reset_period", ARMS[arm]["trace_len"]))

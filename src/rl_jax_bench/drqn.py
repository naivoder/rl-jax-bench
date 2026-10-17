"""Reusable (recurrent) double-DQN core in JAX: networks, sequence replay, double-Q targets, burn-in, update.

Self-contained (jax / flax.linen / optax only). Everything is a pure function of arrays; jit/vmap at the call site.

One code path serves every arm, so arms differ ONLY in the mechanism under test:
    DQN       : QConfig(lstm=0,  trace_len=1,  double=False)   (Eq. 11, Thornton et al. §III-C)
    DDQN-MLP  : QConfig(lstm=0,  trace_len=1,  double=True)    (ablation D-ABL-1)
    DDRQN     : QConfig(lstm=84, trace_len=10, double=True)    (Eq. 14 + LSTM, §III-D)
A transition is a trace of length 1, so the MLP learner runs through the same sequence loss.

Conventions
  * batch-major arrays: inputs [B, L+1, ...] (pytree allowed), actions/rewards/discounts/mask [B, L].
  * discounts[t] = gamma * (1 - terminal_t); a continuing task passes gamma everywhere (Pardo et al. 2018).
  * resets [B, L+1] (optional): the carry is zeroed BEFORE consuming inputs[:, t] where resets[:, t] = 1
    (episode starts, or act-time resets the trace must mirror). carry0 is the state before inputs[:, 0].
  * burn-in b (R2D2): the first b steps only advance the state (no gradient); the loss uses steps b..L-1.
    b = 0 with a zero carry0 is Hausknecht & Stone's "bootstrapped random updates".
  * target: y_t = r_t + d_t * Q_tgt(h_{t+1}, argmax_a Q_on(h_{t+1}, a))  (double)  or  r_t + d_t * max_a Q_tgt(h_{t+1}, a);
    both unrolls run over the same trace from the same carry0, so h_{t+1} comes from the next observation within the trace.
  * loss = sum(mask * l(q - y)) / max(sum(mask), 1), l = squared error (or Huber); y is stop-gradient.
  * target sync is the caller's choice: `sync_due(state, period)` (every `period` gradient steps) or any external rule.
  * legal actions (optional): batch["legal"] [B, L+1, A] bool masks BOTH the double-Q argmax and the max target at the
    next state; acting uses masked_greedy(q, legal). The buffer/writer do not store masks: the caller adds them.
  * n-step (cfg.n_step = n > 1): y_t = sum_{k<m} (prod_{j<k} d_{t+j}) r_{t+k} + (prod_{j<m} d_{t+j}) V(h_{t+m}),
    m = min(n, steps left in the trace, steps before the first mask = 0); V is the double / max bootstrap value.
  * step size (see papers/radar_drqn_coexistence): the recurrent learner is step-size sensitive where the
    MLP is not. Adam lr 1e-3 at one 32 x 10-trace update per decision cost the LSTM -0.10 against its feed-forward twin;
    lr 1e-4, plain SGD (optimizer="sgd") or one update per stored trace (update_every = L) each removed most of it.
    Declare optimizer, lr and update cadence per architecture; never hand the MLP's settings to the recurrent arm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

Array = jax.Array


@dataclass(frozen=True)
class QConfig:
    n_actions: int
    obs_dim: int
    hidden: tuple = (256, 128, 84)
    lstm: int = 0                    # 0 = feed-forward (DQN); >0 = LSTM size after the last hidden layer
    double: bool = True              # double-Q target (van Hasselt et al. 2016) vs max target (Mnih et al. 2015)
    gamma: float = 0.9
    lr: float = 1e-3
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-7
    trace_len: int = 1               # L: transitions per stored trace
    burn_in: int = 0                 # b < L
    batch_size: int = 32             # traces per gradient step
    buffer_traces: int = 2000        # FIFO capacity in traces
    target_period: int = 250         # gradient steps between hard target copies (used by sync_due)
    store_carry: bool = False        # store the actor's carry at each trace start (R2D2 stored state)
    huber_delta: Optional[float] = None
    max_grad_norm: Optional[float] = None
    output_relu: bool = False        # never used for the radar headline (A-03); kept to test the literal reading
    kernel_init: str = "glorot_uniform"  # Dense/LSTM-input kernels: Keras/TF default (paper's frameworks). "lecun_normal" =
                                         # flax/CleanRL default; Phase-2.5 CartPole reference run: it matters (RESULTS.md)
    fuse_target_unroll: bool = False  # True: online+target unrolls as one vmapped pass (measured SLOWER on a 3090)
    recurrent_activation: str = "sigmoid"  # LSTM i/f/o gates: "sigmoid" (TF2 Keras / torch) or "hard_sigmoid"
                                           # (pre-TF2 Keras default, clip(0.2 x + 0.5, 0, 1)); no parameters change
    optimizer: str = "adam"          # "adam" (A-04, shipped) or "sgd" = plain SGD at lr, no momentum (Eq. 9 literal;
                                     # the recheck, sensitivity reading only)
    loss_scale: float = 1.0          # multiplies the Eq. 12 loss. 1.0 = mean over TD terms (shipped). 1/|A| = the Keras
                                     # full-target-vector MSE (mean over the 15 outputs); immaterial under Adam, a 15x
                                     # step change under SGD (the recheck)
    update_every: int = 1            # gradient step every k pulses (1 = A-07 shipped; 10 = once per completed DDRQN
                                     # trace, the sample-matched control of the recheck). Read by train.lane_step.
    n_step: int = 1                  # n-step double-Q / max target inside a trace (1 = Eq. 11 / 14, bit-identical to
                                     # the earlier core). Truncated at the trace end and at the first mask = 0 step;
                                     # discounts carry gamma^(dt/tau) for SMDP decisions (smdp_discount).
    layer_norm: bool = False         # LayerNorm after each torso Dense, before the ReLU (PQN, Gallici et al. 2024; RLPD,
                                     # Ball et al. 2023). False: parameter tree and outputs unchanged


# ----------------------------------------------------------------------------- networks
def unit_forget_bias(key, shape, dtype=jnp.float32):
    """Keras LSTM bias: zeros except 1 on the forget-gate slice (gate order i, f, c, o)."""
    h = shape[0] // 4
    return jnp.zeros(shape, dtype).at[h:2 * h].set(1.0)


def kernel_initializer(name: str):
    """glorot_uniform (Keras/TF Dense default) or lecun_normal (flax Dense default)."""
    return {"glorot_uniform": nn.initializers.glorot_uniform(), "lecun_normal": nn.initializers.lecun_normal()}[name]


def gate_activation(name: str):
    """LSTM gate nonlinearity: sigmoid, or Keras 2.x hard_sigmoid = clip(0.2 x + 0.5, 0, 1)."""
    if name == "sigmoid":
        return jax.nn.sigmoid
    if name == "hard_sigmoid":
        return lambda x: jnp.clip(0.2 * x + 0.5, 0.0, 1.0)
    raise ValueError(name)


class LSTMCell(nn.Module):
    """Keras-convention LSTM: z = x W + h U + b, gates (i, f, c, o), glorot_uniform W, orthogonal U, forget bias 1.

    Split into input_proj (x W + b, applied to a whole sequence at once) and recur (h U + gates), so only the
    recurrent half runs inside the scan. __call__ = recur(carry, input_proj(x))."""
    features: int
    kernel_init: str = "glorot_uniform"
    recurrent_activation: str = "sigmoid"

    def setup(self):
        self.kernel = nn.Dense(4 * self.features, kernel_init=kernel_initializer(self.kernel_init),
                               bias_init=unit_forget_bias)
        self.recurrent = nn.Dense(4 * self.features, use_bias=False, kernel_init=nn.initializers.orthogonal())

    def input_proj(self, x):
        return self.kernel(x)

    def recur(self, carry, xp):
        c, h = carry
        z = xp + self.recurrent(h)
        i, f, g, o = jnp.split(z, 4, axis=-1)
        sg = gate_activation(self.recurrent_activation)
        c2 = sg(f) * c + sg(i) * jnp.tanh(g)
        h2 = sg(o) * jnp.tanh(c2)
        return (c2, h2), h2

    def __call__(self, carry, x):
        return self.recur(carry, self.input_proj(x))


class QNetwork(nn.Module):
    """obs -> Dense(h1)-ReLU -> ... -> Dense(hk)-ReLU -> [LSTM] -> linear Q (glorot_uniform kernels, zero biases)."""
    hidden: tuple
    lstm: int
    n_actions: int
    output_relu: bool = False
    kernel_init: str = "glorot_uniform"
    recurrent_activation: str = "sigmoid"
    layer_norm: bool = False

    def setup(self):
        ki = kernel_initializer(self.kernel_init)
        self.torso = [nn.Dense(h, kernel_init=ki, name=f"fc{i}") for i, h in enumerate(self.hidden)]
        self.norms = [nn.LayerNorm(name=f"ln{i}") for i in range(len(self.hidden))] if self.layer_norm else None
        self.cell = LSTMCell(self.lstm, self.kernel_init, self.recurrent_activation, name="lstm") if self.lstm else None
        self.head = nn.Dense(self.n_actions, kernel_init=ki, name="q")

    def features(self, x):
        if self.norms is None:
            for d in self.torso:
                x = nn.relu(d(x))
        else:
            for d, ln in zip(self.torso, self.norms):
                x = nn.relu(ln(d(x)))
        return x

    def step(self, carry, f):
        return self.cell(carry, f)

    def features_proj(self, x):
        """Torso + the LSTM input projection (whole sequence at once)."""
        return self.cell.input_proj(self.features(x))

    def step_proj(self, carry, xp):
        return self.cell.recur(carry, xp)

    def q(self, h):
        q = self.head(h)
        return nn.relu(q) if self.output_relu else q

    def __call__(self, x, carry):
        f = self.features(x)
        if self.lstm:
            carry, f = self.step(carry, f)
        return self.q(f), carry


def make_network(cfg: QConfig) -> QNetwork:
    return QNetwork(tuple(cfg.hidden), cfg.lstm, cfg.n_actions, cfg.output_relu, cfg.kernel_init, cfg.recurrent_activation,
                      cfg.layer_norm)


def initial_carry(cfg: QConfig, batch_shape: tuple = ()):
    """Zero LSTM state (c, h), or () for the feed-forward net."""
    if not cfg.lstm:
        return ()
    z = jnp.zeros(batch_shape + (cfg.lstm,), jnp.float32)
    return (z, z)


def _zero_where(carry, flag):
    """carry with every leaf zeroed where flag (shape = carry leading dims) is true."""
    return jax.tree.map(lambda c: jnp.where(flag[..., None], jnp.zeros_like(c), c), carry)


def unroll(net: QNetwork, cfg: QConfig, params, inputs: Array, carry0, resets: Optional[Array] = None):
    """Q over a sequence: inputs [B, T, D] -> (q [B, T, A], carry_T). The torso runs on all steps at once;
    only the LSTM cell is sequential (lax.scan, fully unrolled for short traces)."""
    if cfg.lstm:
        fs = jnp.swapaxes(net.apply(params, inputs, method=QNetwork.features_proj), 0, 1)
        rs = jnp.zeros(fs.shape[:2], bool) if resets is None else jnp.swapaxes(resets, 0, 1).astype(bool)

        def body(carry, xs):
            ft, rt = xs
            carry = _zero_where(carry, rt)
            carry, h = net.apply(params, carry, ft, method=QNetwork.step_proj)
            return carry, h

        carry_T, hs = jax.lax.scan(body, carry0, (fs, rs), unroll=fs.shape[0] <= 16)
        f = jnp.swapaxes(hs, 0, 1)
    else:
        f = net.apply(params, inputs, method=QNetwork.features)
        carry_T = carry0
    return net.apply(params, f, method=QNetwork.q), carry_T


def act_q(net: QNetwork, cfg: QConfig, params, obs: Array, carry, reset: Optional[Array] = None):
    """One acting step for any leading batch dims: (q [..., A], carry'). reset zeroes the carry first."""
    if cfg.lstm and reset is not None:
        carry = _zero_where(carry, jnp.asarray(reset, bool))
    f = net.apply(params, obs, method=QNetwork.features)
    if cfg.lstm:
        carry, f = net.apply(params, carry, f, method=QNetwork.step)
    return net.apply(params, f, method=QNetwork.q), carry


def greedy(q: Array) -> Array:
    """argmax with ties to the lowest index."""
    return jnp.argmax(q, axis=-1).astype(jnp.int32)


def masked_greedy(q: Array, legal: Optional[Array] = None) -> Array:
    """argmax over legal actions (legal [..., A] bool), ties to the lowest legal index. A row with no legal action
    falls back to the unmasked argmax (the caller's mask is then wrong; never silently pick an illegal action
    when a legal one exists)."""
    if legal is None:
        return greedy(q)
    legal = jnp.asarray(legal, bool)
    qm = jnp.where(legal, q, -jnp.inf)
    return jnp.where(legal.any(-1), jnp.argmax(qm, axis=-1), jnp.argmax(q, axis=-1)).astype(jnp.int32)


def smdp_discount(dt: Array, gamma: float, tau: float = 1.0, terminal: Optional[Array] = None) -> Array:
    """Per-decision discount for a variable-duration (SMDP) step: gamma ** (dt / tau) * (1 - terminal).
    gamma is the discount per tau seconds (or per tau base steps); dt is the decision's duration in the same unit."""
    dt = jnp.asarray(dt)
    dt = dt if jnp.issubdtype(dt.dtype, jnp.floating) else dt.astype(jnp.float32)
    d = jnp.power(jnp.asarray(gamma, dt.dtype), dt / tau)
    if terminal is not None:
        d = d * (1.0 - jnp.asarray(terminal, d.dtype))
    return d


def linear_epsilon(t, start_e: float, end_e: float, duration: float):
    """CleanRL's linear_schedule: max(slope * t + start_e, end_e)."""
    slope = (end_e - start_e) / duration
    return jnp.maximum(slope * t + start_e, end_e)


# ----------------------------------------------------------------------------- prioritized replay (default off)
def sequence_priority(abs_td: Array, mask: Array, eta: float = 0.9) -> Array:
    """R2D2 sequence priority: eta * max + (1 - eta) * mean of |TD| over the valid steps. abs_td, mask [B, L] -> [B].
    A row with no valid step gets 0."""
    m = jnp.asarray(mask, abs_td.dtype)
    mx = jnp.max(jnp.where(m > 0, abs_td, 0.0), axis=-1)
    mean = (abs_td * m).sum(-1) / jnp.maximum(m.sum(-1), 1.0)
    return eta * mx + (1.0 - eta) * mean


def per_sample(key: Array, priorities: Array, count, batch_size: int, alpha: float, beta: float):
    """Proportional prioritized sampling (Schaul et al. 2016), with replacement, no sum-tree: P(i) = p_i^alpha /
    sum_{j<count} p_j^alpha over the first `count` entries. Returns (idx [B], is_weights [B]) with
    w_i = (count * P(i))^-beta / max_batch w. alpha = 0 is uniform. All-zero priorities fall back to uniform."""
    p = jnp.maximum(jnp.asarray(priorities), 0.0)
    valid = jnp.arange(p.shape[0]) < count
    pa = jnp.where(valid, jnp.power(p, alpha), 0.0)
    pa = jnp.where(pa.sum() > 0, pa, valid.astype(pa.dtype))
    probs = pa / pa.sum()
    idx = jax.random.categorical(key, jnp.where(valid & (pa > 0), jnp.log(jnp.where(pa > 0, pa, 1.0)), -jnp.inf),
                                 shape=(batch_size,))
    w = jnp.power(jnp.asarray(count, probs.dtype) * probs[idx], -beta)
    return idx, w / jnp.max(w)


def per_new_priority(priorities: Array, count) -> Array:
    """Priority for a newly stored item: the running max over the first `count` entries (1.0 when empty)."""
    p = jnp.asarray(priorities)
    mx = jnp.max(jnp.where(jnp.arange(p.shape[0]) < count, p, -jnp.inf))
    return jnp.where(count > 0, mx, jnp.ones((), p.dtype))


def apex_epsilons(n_actors: int, base: float = 0.4, alpha: float = 7.0) -> Array:
    """Ape-X (Horgan et al. 2018) per-actor epsilons: eps_i = base ** (1 + alpha * i / (n - 1)); n = 1 gives base."""
    i = jnp.arange(n_actors, dtype=jnp.float32)
    return jnp.power(base, 1.0 + alpha * i / max(n_actors - 1, 1))


# ----------------------------------------------------------------------------- demonstrations (default off)
def qfilter_mask(q: Array, a_teacher: Array, legal: Optional[Array] = None, q_eval: Optional[Array] = None) -> Array:
    """Q-filter (Nair et al. 2018): True where Q_eval(s, a_teacher) > Q_eval(s, argmax_{a legal} q), i.e. the
    second estimate still prefers the teacher's action to the student's greedy one. Stop-gradient.
    q [..., A] picks the student's action; q_eval [..., A] (default q) scores both actions. With q_eval = q (the
    one-estimate form) the mask is False on every row whose teacher action is legal, so the filter is degenerate
    for a greedy-Q student; pass a second estimate such as the target network's Q to make it live."""
    q = jax.lax.stop_gradient(q)
    q_eval = q if q_eval is None else jax.lax.stop_gradient(q_eval)
    q_t = jnp.take_along_axis(q_eval, jnp.asarray(a_teacher, jnp.int32)[..., None], axis=-1)[..., 0]
    q_best = jnp.take_along_axis(q_eval, masked_greedy(q, legal)[..., None], axis=-1)[..., 0]
    return q_t > q_best


def margin_loss_terms(q: Array, a_teacher: Array, margin: float, legal: Optional[Array] = None,
                      qfilter: bool = False, q_eval: Optional[Array] = None) -> Array:
    """DQfD large-margin loss per row: J_E = max_{a legal}[Q(s, a) + l(a_E, a)] - Q(s, a_E), l = margin for
    a != a_E and 0 for a = a_E. The teacher action always joins the max, so J_E >= 0. qfilter=True zeroes the rows
    where qfilter_mask(q, a_E, legal, q_eval) is False (see qfilter_mask; without q_eval it is degenerate)."""
    a_t = jnp.asarray(a_teacher, jnp.int32)
    A = q.shape[-1]
    onehot = jnp.arange(A) == a_t[..., None]
    aug = q + jnp.where(onehot, 0.0, margin)
    if legal is not None:
        aug = jnp.where(jnp.asarray(legal, bool) | onehot, aug, -jnp.inf)
    q_t = jnp.take_along_axis(q, a_t[..., None], axis=-1)[..., 0]
    je = jnp.max(aug, axis=-1) - q_t
    if qfilter:
        je = je * qfilter_mask(q, a_t, legal, q_eval).astype(je.dtype)
    return je


def margin_loss(q: Array, a_teacher: Array, margin: float, legal: Optional[Array] = None, qfilter: bool = False,
                mask: Optional[Array] = None, q_eval: Optional[Array] = None) -> Array:
    """Mean DQfD margin loss over the rows with mask > 0 (all rows if None). With qfilter the filtered rows
    contribute 0 but still count in the denominator."""
    je = margin_loss_terms(q, a_teacher, margin, legal, qfilter, q_eval)
    if mask is None:
        return je.mean()
    m = jnp.asarray(mask, je.dtype)
    return (je * m).sum() / jnp.maximum(m.sum(), 1.0)


# ----------------------------------------------------------------------------- targets and loss
def bootstrap_value(q_next_online: Array, q_next_target: Array, double: bool, legal: Optional[Array] = None) -> Array:
    """V(s') = Q_tgt(s', argmax_{a legal} Q_on(s', a)) (double) or max_{a legal} Q_tgt(s', a) (max)."""
    if double:
        a = masked_greedy(q_next_online, legal)
        return jnp.take_along_axis(q_next_target, a[..., None], axis=-1)[..., 0]
    if legal is None:
        return jnp.max(q_next_target, axis=-1)
    legal = jnp.asarray(legal, bool)
    return jnp.where(legal.any(-1), jnp.max(jnp.where(legal, q_next_target, -jnp.inf), axis=-1),
                     jnp.max(q_next_target, axis=-1))


def td_target(q_next_online: Array, q_next_target: Array, rewards: Array, discounts: Array, double: bool,
              legal: Optional[Array] = None) -> Array:
    """Eq. 11 (double=False): r + d max_a Q_tgt(s', a);  Eq. 14 (double=True): r + d Q_tgt(s', argmax_a Q_on(s', a)).
    legal [..., A] (optional) restricts both the argmax and the max to legal next actions."""
    return rewards + discounts * bootstrap_value(q_next_online, q_next_target, double, legal)


def n_step_target(rewards: Array, discounts: Array, v_next: Array, n: int, mask: Optional[Array] = None) -> Array:
    """n-step targets inside a trace. rewards/discounts/v_next/mask are [B, L]; v_next[:, t] = V(h_{t+1}).
    y_t = sum_{k<m} G_{t,k} r_{t+k} + G_{t,m} v_next[:, t+m-1], G_{t,k} = prod_{j<k} d_{t+j},
    m = min(n, L - t, steps before the first mask = 0 at or after t+1). n = 1 gives r + d v_next exactly."""
    B, L = rewards.shape
    if mask is None:
        mask = jnp.ones((B, L), rewards.dtype)
    pad = lambda x: jnp.concatenate([x, jnp.zeros((B, n), x.dtype)], axis=1)
    r_p, d_p, v_p, m_p = pad(rewards), pad(discounts), pad(v_next), pad(mask.astype(rewards.dtype))
    y = jnp.zeros((B, L), rewards.dtype)
    G = jnp.ones((B, L), rewards.dtype)
    alive = jnp.ones((B, L), bool)          # step t+k is inside the trace and every step t+1..t+k is unmasked
    for k in range(n):
        rk, dk, vk = r_p[:, k:k + L], d_p[:, k:k + L], v_p[:, k:k + L]
        y = y + jnp.where(alive, G * rk, 0.0)
        nxt = (jnp.arange(L)[None, :] + k + 1 < L) & (m_p[:, k + 1:k + 1 + L] > 0) & (k + 1 < n)
        stop = alive & ~nxt                  # bootstrap from h_{t+k+1} here
        y = y + jnp.where(stop, G * dk * vk, 0.0)
        G = G * dk
        alive = alive & nxt
    return y


UnrollFn = Callable[..., Any]   # (params, inputs [B,T,...], carry0, resets) -> (q [B,T,A], carry_T)


def default_unroll_fn(cfg: QConfig) -> UnrollFn:
    net = make_network(cfg)
    return lambda params, inputs, carry0, resets=None: unroll(net, cfg, params, inputs, carry0, resets)


def _slice_t(tree, sl):
    return jax.tree.map(lambda x: x[:, sl], tree)


def sequence_loss(params, target_params, batch: dict, cfg: QConfig, unroll_fn: Optional[UnrollFn] = None):
    """Masked TD loss over a batch of traces (module docstring). Returns (loss, aux)."""
    custom = unroll_fn is not None
    unroll_fn = unroll_fn or default_unroll_fn(cfg)
    inputs, actions = batch["inputs"], batch["actions"]
    B, L = actions.shape
    b = cfg.burn_in
    carry0 = batch.get("carry0")
    if carry0 is None or carry0 == ():
        carry0 = initial_carry(cfg, (B,))
    resets = batch.get("resets")
    mask = batch.get("mask")
    mask = jnp.ones((B, L), jnp.float32) if mask is None else mask
    if b > 0:
        burn_in = _slice_t(inputs, slice(0, b))
        burn_r = None if resets is None else resets[:, :b]
        _, c_on = unroll_fn(params, burn_in, carry0, burn_r)
        _, c_tg = unroll_fn(target_params, burn_in, carry0, burn_r)
        c_on = jax.lax.stop_gradient(c_on)
        inputs = _slice_t(inputs, slice(b, None))
        resets = None if resets is None else resets[:, b:]
    else:
        c_on = c_tg = carry0
    if cfg.fuse_target_unroll and not custom:
        stacked = jax.tree.map(lambda p, t: jnp.stack([p, jax.lax.stop_gradient(t)]), params, target_params)
        carries = jax.tree.map(lambda a, b: jnp.stack([a, b]), c_on, c_tg)
        q2, _ = jax.vmap(unroll_fn, in_axes=(0, None, 0, None))(stacked, inputs, carries, resets)
        q_on, q_tg = q2[0], q2[1]                                     # [B, L+1-b, A]
    else:
        q_on, _ = unroll_fn(params, inputs, c_on, resets)
        q_tg, _ = unroll_fn(target_params, inputs, c_tg, resets)
    q_tg = jax.lax.stop_gradient(q_tg)
    act, rew, disc, m = actions[:, b:], batch["rewards"][:, b:], batch["discounts"][:, b:], mask[:, b:]
    legal = batch.get("legal")
    legal_next = None if legal is None else legal[:, b + 1:]
    q_sa = jnp.take_along_axis(q_on[:, :-1], act[..., None].astype(jnp.int32), axis=-1)[..., 0]
    if cfg.n_step == 1:
        y = td_target(q_on[:, 1:], q_tg[:, 1:], rew, disc, cfg.double, legal_next)
    else:
        v_next = bootstrap_value(q_on[:, 1:], q_tg[:, 1:], cfg.double, legal_next)
        y = n_step_target(rew, disc, v_next, cfg.n_step, m)
    y = jax.lax.stop_gradient(y)
    err = q_sa - y
    if cfg.huber_delta is None:
        el = err ** 2
    else:
        el = optax.huber_loss(err, delta=cfg.huber_delta)
    denom = jnp.maximum(m.sum(), 1.0)
    weights = batch.get("weights")          # optional [B] importance-sampling weights (prioritized replay)
    if weights is None:
        loss = cfg.loss_scale * (el * m).sum() / denom
    else:
        loss = cfg.loss_scale * (el * m * jnp.asarray(weights, el.dtype)[:, None]).sum() / denom
    abs_td = jax.lax.stop_gradient(jnp.abs(err))
    aux = dict(q_sa=q_sa, y=y, q_mean=(q_sa * m).sum() / denom,
               maxq_mean=(q_on[:, :-1].max(-1) * m).sum() / denom,
               abs_td=abs_td, priorities=sequence_priority(abs_td, m))
    return loss, aux


# ----------------------------------------------------------------------------- learner state and update
class QState(NamedTuple):
    params: Any
    target_params: Any
    opt_state: Any
    n_updates: Array


def make_optimizer(cfg: QConfig) -> optax.GradientTransformation:
    if cfg.optimizer == "adam":
        opt = optax.adam(cfg.lr, b1=cfg.adam_b1, b2=cfg.adam_b2, eps=cfg.adam_eps)
    elif cfg.optimizer == "sgd":
        opt = optax.sgd(cfg.lr)          # w <- w - lr * grad (Eq. 9); Keras SGD default momentum 0
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    if cfg.max_grad_norm is None:
        return opt
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), opt)


def init(key: Array, cfg: QConfig) -> QState:
    """Online params from `key`; target = hard copy (A-28)."""
    net = make_network(cfg)
    params = net.init(key, jnp.zeros((1, cfg.obs_dim), jnp.float32), initial_carry(cfg, (1,)))
    return QState(params, jax.tree.map(jnp.copy, params), make_optimizer(cfg).init(params), jnp.zeros((), jnp.int32))


def update(state: QState, batch: dict, cfg: QConfig, unroll_fn: Optional[UnrollFn] = None):
    """One gradient step (cfg.optimizer: Adam or plain SGD) on sequence_loss. Target sync is NOT done here."""
    (loss, aux), grads = jax.value_and_grad(sequence_loss, has_aux=True)(
        state.params, state.target_params, batch, cfg, unroll_fn)
    upd, opt = make_optimizer(cfg).update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, upd)
    info = dict(loss=loss, q_mean=aux["q_mean"], maxq_mean=aux["maxq_mean"], grad_norm=optax.tree.norm(grads),
                priorities=aux["priorities"])
    return QState(params, state.target_params, opt, state.n_updates + 1), info


def sync(state: QState, do_sync) -> QState:
    """Hard target copy where do_sync (works under vmap)."""
    tgt = jax.tree.map(lambda t, p: jnp.where(do_sync, p, t), state.target_params, state.params)
    return state._replace(target_params=tgt)


def sync_due(state: QState, period: int) -> Array:
    """True right after every `period`-th gradient step."""
    return (state.n_updates > 0) & (state.n_updates % period == 0)


# ----------------------------------------------------------------------------- sequence replay
class TraceBuffer(NamedTuple):
    obs: Array          # [N, L+1, D]
    act: Array          # [N, L] int32
    rew: Array          # [N, L]
    disc: Array         # [N, L]
    resets: Array       # [N, L+1] bool
    carry0: Any         # pytree [N, ...] or ()
    ptr: Array          # next write slot
    count: Array        # number of valid traces (<= N)


def buffer_init(cfg: QConfig) -> TraceBuffer:
    N, L, D = cfg.buffer_traces, cfg.trace_len, cfg.obs_dim
    carry = initial_carry(cfg, (N,)) if cfg.store_carry else ()
    return TraceBuffer(jnp.zeros((N, L + 1, D), jnp.float32), jnp.zeros((N, L), jnp.int32),
                       jnp.zeros((N, L), jnp.float32), jnp.zeros((N, L), jnp.float32),
                       jnp.zeros((N, L + 1), bool), carry, jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32))


def buffer_add(buf: TraceBuffer, trace: dict, do_add=True) -> TraceBuffer:
    """FIFO write of one trace into slot ptr (masked by do_add, so it is vmap/scan friendly)."""
    i = buf.ptr

    def put(arr, val):
        return arr.at[i].set(jnp.where(do_add, val, arr[i]))

    carry0 = buf.carry0 if buf.carry0 == () else jax.tree.map(put, buf.carry0, trace["carry0"])
    n = buf.obs.shape[0]
    return TraceBuffer(put(buf.obs, trace["obs"]), put(buf.act, trace["act"]), put(buf.rew, trace["rew"]),
                       put(buf.disc, trace["disc"]), put(buf.resets, trace["resets"]), carry0,
                       jnp.where(do_add, (i + 1) % n, i), jnp.where(do_add, jnp.minimum(buf.count + 1, n), buf.count))


def buffer_sample(buf: TraceBuffer, key: Array, batch_size: int) -> dict:
    """Uniform with replacement over the valid traces (as CleanRL / SB3)."""
    idx = jax.random.randint(key, (batch_size,), 0, jnp.maximum(buf.count, 1))
    return buffer_gather(buf, idx)


def buffer_gather(buf: TraceBuffer, idx: Array) -> dict:
    out = dict(inputs=buf.obs[idx], actions=buf.act[idx], rewards=buf.rew[idx], discounts=buf.disc[idx],
               resets=buf.resets[idx])
    if buf.carry0 != ():
        out["carry0"] = jax.tree.map(lambda c: c[idx], buf.carry0)
    return out


class TraceWriter(NamedTuple):
    """Accumulates a stream into aligned traces of length L (non-overlapping; the last obs of a trace is the
    first obs of the next)."""
    obs: Array
    act: Array
    rew: Array
    disc: Array
    resets: Array
    carry0: Any
    k: Array


def writer_init(cfg: QConfig) -> TraceWriter:
    L, D = cfg.trace_len, cfg.obs_dim
    carry = initial_carry(cfg) if cfg.store_carry else ()
    return TraceWriter(jnp.zeros((L + 1, D), jnp.float32), jnp.zeros((L,), jnp.int32), jnp.zeros((L,), jnp.float32),
                       jnp.zeros((L,), jnp.float32), jnp.zeros((L + 1,), bool), carry, jnp.zeros((), jnp.int32))


def writer_step(w: TraceWriter, buf: TraceBuffer, obs, a, r, disc, next_obs, reset, next_reset, carry_before=()):
    """Record (obs, a, r, disc) at position k. `reset` = the actor zeroed its carry before consuming obs;
    `next_reset` = it will zero it before consuming next_obs. `carry_before` = the actor's carry before obs
    (stored at k = 0 when cfg.store_carry). Pushes the trace when k = L-1. Returns (writer, buffer, pushed)."""
    L = w.act.shape[0]
    k = w.k
    obs_arr = w.obs.at[k].set(obs)
    res_arr = w.resets.at[k].set(reset)
    carry0 = w.carry0 if w.carry0 == () else jax.tree.map(lambda c0, cb: jnp.where(k == 0, cb, c0), w.carry0,
                                                           carry_before)
    full = k == L - 1
    obs_arr = jnp.where(full, obs_arr.at[L].set(next_obs), obs_arr)
    res_arr = jnp.where(full, res_arr.at[L].set(next_reset), res_arr)
    w2 = TraceWriter(obs_arr, w.act.at[k].set(a), w.rew.at[k].set(r), w.disc.at[k].set(disc), res_arr, carry0,
                     jnp.where(full, 0, k + 1))
    buf2 = buffer_add(buf, dict(obs=w2.obs, act=w2.act, rew=w2.rew, disc=w2.disc, resets=w2.resets, carry0=carry0),
                      full)
    return w2, buf2, full

"""Environments (there is no dataset in this paper: the "data" are online interaction).

* Tabular toy MDPs as explicit tables (next state NX[s, a], reward R[s, a], environmental termination TERM[s, a]):
  Last Moment (paper §2.1), Two-Goal Gridworld (§2.2/§3.1, Figure 1a), Difficult Gridworld (§3.4, Figure 7).
  The same tables feed the exact solvers (src/exact.py) and the tabular learners (src/algorithm.py), so toy stage
  checks the environment and the solver together.
* Queue of Cars (§2.3) as a stochastic JAX step function (+ its exact transition tensor for DP).
* Brax `inverted_pendulum` / `hopper` (labelled PROXIES for MuJoCo-v1), raw env without Brax's
  EpisodeWrapper, so the env's `done` is environmental termination only (paper §3.1 "remove TimeLimit").
* `TimeLimit`: OUR runner-side time limit T (a traced scalar, so arms with different T vmap together), the time
  feature tau(t) (paper §1; Tonic TimeFeature), and the reset/termination split of Tonic's Sequential runner
  reset = term OR t == T, the pre-reset next observation is returned as `final_obs`.

Every constant carries its source tag (paper §, Figure, official code, ASSUMED + AS-n of ).
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# ===================================================================== time feature (paper §1; Tonic TimeFeature)
TIME_MODES = {"none": 0, "elapsed_pm1": 1, "remaining_pm1": 2, "remaining_raw": 3}
# none          : no time input (the input column is held at 0, which is exactly equivalent to no column: its
#                 first-layer weights get zero gradient and contribute nothing -- tests/test_smoke.py)
# elapsed_pm1   : -1 + 2 t/T  (official code: Tonic wrappers.py TimeFeature, low=-1 high=1) -- PRIMARY
# remaining_pm1 : 1 - 2 t/T = 2 k/T - 1, k = T - t  (paper §1 text "remaining time ... normalized from -1 to 1")
# remaining_raw : T - t     (owner ablation: unnormalized remaining time)


def time_feature(t, T, mode):
    """tau(t) for elapsed step t in [0, T] (t = 0 at reset). `mode` is an int (traced OK)."""
    t = jnp.asarray(t, jnp.float32)
    T = jnp.asarray(T, jnp.float32)
    return jnp.select(
        [mode == 1, mode == 2, mode == 3],
        [-1.0 + 2.0 * t / T, 1.0 - 2.0 * t / T, T - t],
        0.0,
    ).astype(jnp.float32)


# ===================================================================== tabular MDPs
class Tabular(NamedTuple):
    """Deterministic tabular MDP. Terminal (absorbing, zero-value) states have is_terminal = True."""
    NX: np.ndarray          # (S, A) int  next state
    R: np.ndarray           # (S, A) float reward
    is_terminal: np.ndarray  # (S) bool  environmental termination on ENTERING the state
    start: np.ndarray       # (S) float  start distribution
    names: Any = None       # optional per-state labels (e.g. (row, col))


def last_moment() -> Tabular:
    """Paper §2.1. States A=0, B=1; actions 0 = stay, 1 = jump. A,stay -> A r 0; A,jump -> B r +1; B,* -> B r -1.
    No environmental termination: every episode lasts exactly T steps (paper §2.1)."""
    NX = np.array([[0, 1], [1, 1]])
    R = np.array([[0.0, 1.0], [-1.0, -1.0]])        # paper §2.1 (+1 jump, -1 per step in B, 0 stay)
    return Tabular(NX, R, np.array([False, False]), np.array([1.0, 0.0]), ["A", "B"])


TWO_GOAL_MAP = ["..#..A", "....#.", ".##...", "..###.", "#...#.", "B....."]  # paper Figure 1a
TWO_GOAL_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]               # up down left right stay (paper §2.2)


def two_goal(bump_reward: float = -1.0) -> Tabular:
    """Paper §2.2 Two-Goal Gridworld. Move -1, stay 0 (paper §2.2); entering A/B = -1 + 50 / -1 + 20 (the
    text's '49 and 19'); bumping into a wall/outside = stay in place with `bump_reward` (ASSUMED, ablated 0).
    State order = row-major non-wall cells (identical to scripts/exact_solvers/grid.py)."""
    H = W = 6
    cells = [(r, c) for r in range(H) for c in range(W) if TWO_GOAL_MAP[r][c] != "#"]
    idx = {p: i for i, p in enumerate(cells)}
    goal = {idx[(0, 5)]: 50.0, idx[(5, 0)]: 20.0}
    S, A = len(cells), 5
    NX = np.zeros((S, A), int)
    R = np.zeros((S, A))
    for i, (r, c) in enumerate(cells):
        for a, (dr, dc) in enumerate(TWO_GOAL_ACTIONS):
            if i in goal:            # absorbing (never acted in: terminal on entry)
                NX[i, a], R[i, a] = i, 0.0
                continue
            if a == 4:
                NX[i, a], R[i, a] = i, 0.0
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and TWO_GOAL_MAP[nr][nc] != "#":
                j = idx[(nr, nc)]
                NX[i, a], R[i, a] = j, -1.0 + goal.get(j, 0.0)
            else:
                NX[i, a], R[i, a] = i, bump_reward
    term = np.array([i in goal for i in range(S)])
    start = np.where(term, 0.0, 1.0)
    return Tabular(NX, R, term, start / start.sum(), cells)


def difficult_grid() -> Tabular:
    """Paper §3.4 / Figure 7 (map read at 400 dpi): 20x20, start (10,10), goal (19,19); 4 actions
    (Zhang & Sutton 2017: up, down, left, right), bump = stay; r = -1 every step (paper §3.4). State = r*20 + c over
    ALL 400 cells (walls are unreachable rows of the table)."""
    N = 20
    wall = np.zeros((N, N), bool)
    wall[1, 1] = True
    wall[1, 3:19] = True
    wall[1:19, 18] = True
    wall[18, 1:19] = True
    wall[3:19, 1] = True
    acts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    S = N * N
    NX = np.zeros((S, 4), int)
    for r in range(N):
        for c in range(N):
            for a, (dr, dc) in enumerate(acts):
                nr, nc = r + dr, c + dc
                ok = 0 <= nr < N and 0 <= nc < N and not wall[nr, nc]
                NX[r * N + c, a] = nr * N + nc if ok else r * N + c
    goal = 19 * N + 19
    NX[goal] = goal
    R = -np.ones((S, 4))
    R[goal] = 0.0
    term = np.zeros(S, bool)
    term[goal] = True
    start = np.zeros(S)
    start[10 * N + 10] = 1.0
    return Tabular(NX, R, term, start, wall)


DGRID_START = 10 * 20 + 10
DGRID_GOAL = 19 * 20 + 19


# ===================================================================== Queue of Cars (paper §2.3)
class QoCParams(NamedTuple):
    n_slots: int = 9           # paper §2.3 ("exit located 9 slots away"); Figure 2 (9 cells)
    safe_adv: float = 0.5      # paper §2.3
    danger_adv: float = 0.8    # paper §2.3
    danger_crash: float = 0.1  # paper §2.3
    # remaining 0.1 of the dangerous action = stay (ASSUMED, literal reading); crash-first ablation:
    # danger_adv = 0.72, danger_crash = 0.1 (stay 0.18).
    exit_reward: float = 1.0   # paper §2.3


QOC_CRASH_FIRST = QoCParams(danger_adv=0.72, danger_crash=0.1)  # ablation


def qoc_outcomes(p: QoCParams):
    """(A, 3) probabilities of (advance, crash, stay) for actions 0 = safe, 1 = dangerous."""
    return np.array([[p.safe_adv, 0.0, 1.0 - p.safe_adv],
                     [p.danger_adv, p.danger_crash, 1.0 - p.danger_adv - p.danger_crash]])


class Env(NamedTuple):
    """Stateless env interface used by the PPO runner. reset(key) -> (state, obs); step(key, state, action) ->
    (state, obs, reward, term) with term = ENVIRONMENTAL termination only (never a time limit)."""
    reset: Callable
    step: Callable
    obs_dim: int
    continuous: bool
    n_out: int
    act_low: Any
    act_high: Any
    name: str


def queue_of_cars(p: QoCParams = QoCParams()) -> Env:
    """Observation = one-hot slot (ASSUMED). Start at slot 0 (paper §2.3)."""
    n = p.n_slots
    probs = jnp.asarray(qoc_outcomes(p), jnp.float32)

    def reset(key):
        x = jnp.int32(0)
        return x, jax.nn.one_hot(x, n, dtype=jnp.float32)

    def step(key, x, a):
        u = jax.random.uniform(key)
        pa = probs[a]
        adv = u < pa[0]
        crash = jnp.logical_and(u >= pa[0], u < pa[0] + pa[1])
        x2 = x + adv.astype(jnp.int32)
        exit_ = x2 >= n
        term = jnp.logical_or(exit_, crash)
        r = jnp.where(exit_, p.exit_reward, 0.0).astype(jnp.float32)
        x2 = jnp.minimum(x2, n - 1)  # the observation at a termination is never bootstrapped from
        return x2, jax.nn.one_hot(x2, n, dtype=jnp.float32), r, term

    return Env(reset, step, n, False, 2, None, None, "queue_of_cars")


# ===================================================================== Brax proxies
ACTION_INTERFACES = ("tonic_rescale", "raw")


def brax_env(name: str, backend: str = "generalized", action_interface: str = "tonic_rescale") -> Env:
    """Raw Brax env (no EpisodeWrapper / AutoResetWrapper): `done` is the environmental termination only.

    action_interface (recheck): 'tonic_rescale' (default; original behavior) = the policy acts in
    [-1, 1] (the runner clips there) and Brax maps it to the actuator range itself (IP: x3). 'raw' = the paper-era
    Baselines path (ppo1/ppo2 @9fa8e1b pass the raw Gaussian sample to gym; MuJoCo clamps to ctrlrange): the action
    bounds are the actuator ctrlrange, nothing is clipped runner-side, Brax's actuator clamps (brax/actuator.py) and the
    hopper control cost is charged on the raw action (gym hopper)."""
    from brax import envs as benvs
    assert action_interface in ACTION_INTERFACES, action_interface
    env = benvs.get_environment(name, backend=backend)
    raw = action_interface == "raw"
    lo = hi = None
    if raw:
        import numpy as _np
        cr = _np.asarray(env.sys.actuator.ctrl_range, _np.float32)
        lo, hi = jnp.asarray(cr[:, 0]), jnp.asarray(cr[:, 1])
        if name == "inverted_pendulum":
            # brax IP step maps x in [-1, 1] to (x + 1) * (hi - lo) / 2 + lo WITHOUT clipping; feed the inverse map so
            # the actuator receives the raw action a, which brax/actuator.py then clamps to [lo, hi]
            to_brax = lambda a: (a - lo) * 2.0 / (hi - lo) - 1.0  # noqa: E731
        elif name == "hopper":
            to_brax = lambda a: a  # noqa: E731  # brax hopper passes the action to the pipeline unchanged
        else:
            raise NotImplementedError(f"raw action interface not ported for brax {name}")

    def canon(s):  # some Brax envs return python floats for reward/done; make the state a fixed-type pytree
        return s.replace(reward=jnp.asarray(s.reward, jnp.float32), done=jnp.asarray(s.done, jnp.float32))

    def reset(key):
        s = canon(env.reset(key))
        return s, s.obs

    def step(key, s, a):
        s2 = canon(env.step(s, to_brax(a) if raw else a))
        return s2, s2.obs, s2.reward, s2.done > 0.5

    if raw:
        return Env(reset, step, env.observation_size, True, env.action_size, lo, hi, f"brax-{name}-{backend}-rawact")
    return Env(reset, step, env.observation_size, True, env.action_size, -1.0, 1.0, f"brax-{name}-{backend}")


# ===================================================================== gym MuJoCo-v4 ports (MJX, exact physics)
# oracle stage: the authors' library builds its envs with gym.make; the paper used -v1 (mujoco-py 0.5,
# not installable). These are ports of gym 0.24.1 InvertedPendulum-v4 / Hopper-v4 onto MJX with gym's OWN model
# files (src/assets/gym_v4; RK4, same timestep/frame_skip), so physics, observation, reward, termination and reset
# distribution are gym-v4's -- unlike the Brax proxies above (Euler integrator, re-placed hopper joints, brax solver
# params). Parity vs gym (driven through Tonic's env builder): tests/test_official_parity.py. Default actions arrive in
# [-1, 1] and are mapped to the actuator range exactly like Tonic's ActionRescaler (wrappers.py:7-23); the recheck
# 'raw' interface is the paper-era Baselines + gym path (see gymv4_env).
GYMV4 = {  # official code: gym 0.24.1 envs/mujoco/{inverted_pendulum_v4,hopper_v4}.py
    "inverted_pendulum": dict(xml="inverted_pendulum.xml", frame_skip=2, noise=0.01),   # reset U(-0.01, 0.01)
    "hopper": dict(xml="hopper.xml", frame_skip=4, noise=5e-3),                          # reset_noise_scale 5e-3
}


def gymv4_env(name: str, action_interface: str = "tonic_rescale") -> Env:
    """action_interface (recheck): 'tonic_rescale' (default; original behavior) = Tonic's
    ActionRescaler, ctrl = bias + scale * clip(a, -1, 1) (IP: 3 * clip(a)), cost on that ctrl. 'raw' = the paper-era
    Baselines + gym path (official code: baselines ppo2/ppo1 @9fa8e1b `env.step(actions)` with no wrapper; gym
    mujoco_env.do_simulation writes data.ctrl = a; MuJoCo/MJX clamp ctrllimited actuators to ctrlrange in
    fwd_actuation): ctrl = a, action bounds = ctrlrange (IP [-3, 3], hopper [-1, 1]) and the hopper control cost is
    1e-3 * sum(a^2) on the RAW action (gym hopper_v4.control_cost(action))."""
    import os

    import mujoco
    from mujoco import mjx

    spec = GYMV4[name]
    m = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(__file__), "assets", "gym_v4", spec["xml"]))
    mx = mjx.put_model(m)
    fs, noise = spec["frame_skip"], spec["noise"]
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    bias, scale = jnp.asarray((hi + lo) / 2, jnp.float32), jnp.asarray((hi - lo) / 2, jnp.float32)
    qpos0 = jnp.asarray(m.qpos0, jnp.float32)  # gym init_qpos = data.qpos at construction = qpos0
    dt = float(m.opt.timestep) * fs
    hop = name == "hopper"

    def obs_of(d):
        if hop:  # hopper_v4._get_obs: qpos[1:], clip(qvel, -10, 10)
            return jnp.concatenate([d.qpos[1:], jnp.clip(d.qvel, -10.0, 10.0)]).astype(jnp.float32)
        return jnp.concatenate([d.qpos, d.qvel]).astype(jnp.float32)  # inverted_pendulum_v4._get_obs

    def reset(key):
        k1, k2 = jax.random.split(key)
        qpos = qpos0 + jax.random.uniform(k1, (m.nq,), jnp.float32, -noise, noise)
        qvel = jax.random.uniform(k2, (m.nv,), jnp.float32, -noise, noise)
        # gym set_state -> mj_forward. MJX's dtype follows jax_enable_x64; keep the carry dtype-stable under x64
        d0 = mjx.make_data(mx)
        d = mjx.forward(mx, d0.replace(qpos=qpos.astype(d0.qpos.dtype), qvel=qvel.astype(d0.qvel.dtype)))
        return d, obs_of(d)

    assert action_interface in ACTION_INTERFACES, action_interface
    raw = action_interface == "raw"
    assert bool(np.all(m.actuator_ctrllimited)), "raw interface relies on MuJoCo's ctrlrange clamp"

    def step(key, d, a):
        if raw:  # Baselines + gym: data.ctrl = a; MJX clamps to ctrlrange (ctrllimited); cost on the raw a
            ctrl = a.astype(d.ctrl.dtype)
            cost_a = a
        else:
            ctrl = (bias + scale * jnp.clip(a, -1.0, 1.0)).astype(d.ctrl.dtype)  # Tonic ActionRescaler
            cost_a = ctrl
        d2 = jax.lax.fori_loop(0, fs, lambda _, x: mjx.step(mx, x), d.replace(ctrl=ctrl))  # do_simulation
        o = obs_of(d2)
        if hop:  # hopper_v4.step / is_healthy (terminate_when_unhealthy=True -> healthy_reward always 1)
            r = (d2.qpos[0] - d.qpos[0]) / dt + 1.0 - 1e-3 * jnp.sum(jnp.square(cost_a))
            sv = jnp.concatenate([d2.qpos, d2.qvel])[2:]
            healthy = (jnp.all((sv > -100.0) & (sv < 100.0)) & (d2.qpos[1] > 0.7)
                       & (d2.qpos[2] > -0.2) & (d2.qpos[2] < 0.2))
            term = jnp.logical_not(healthy)
        else:  # inverted_pendulum_v4.step: r = 1; done = not (isfinite(ob).all() and |ob[1]| <= 0.2)
            r = jnp.float32(1.0)
            term = jnp.logical_not(jnp.all(jnp.isfinite(o)) & (jnp.abs(o[1]) <= 0.2))
        return d2, o, jnp.asarray(r, jnp.float32), term

    od = (m.nq - 1 + m.nv) if hop else (m.nq + m.nv)
    if raw:
        return Env(reset, step, od, True, m.nu, jnp.asarray(lo, jnp.float32), jnp.asarray(hi, jnp.float32),
                   f"gymv4-{name}-rawact")
    return Env(reset, step, od, True, m.nu, -1.0, 1.0, f"gymv4-{name}")


def make_env(cfg: dict) -> Env:
    if cfg["env"] == "queue_of_cars":
        sem = cfg.get("qoc_semantics", "literal")
        return queue_of_cars(QOC_CRASH_FIRST if sem == "crash_first" else QoCParams())
    ai = cfg.get("action_interface", "tonic_rescale")
    if cfg["env"].startswith("brax-"):
        return brax_env(cfg["env"][5:], cfg.get("brax_backend", "generalized"), ai)
    if cfg["env"].startswith("gymv4-"):
        return gymv4_env(cfg["env"][6:], ai)
    raise ValueError(cfg["env"])


# ===================================================================== paper-era reward scaler (recheck C2)
def scale_reward_baselines_2018(ret, ret_rms, r, gamma, eps, clip):
    """Baselines VecNormalize.step_wait as of the paper's date (b05be68 2017-11, 9fa8e1b 2018-01): ret <- gamma*ret + r;
    ret_rms.update(ret); r~ = clip(r / sqrt(var + eps)); the accumulator is NEVER reset (`self.ret[news] = 0` arrived
    in ad219e2, 2018-09-06). Same arithmetic as src/ppo8/data.py scale_reward_baselines without its final reset; kept
    here because src/ppo8/ is the vendored PPO-details code (tests/test_vendor.py)."""
    from .ppo8 import data as P8D
    ret = ret * gamma + r
    ret_rms = P8D.rms_update(ret_rms, ret)
    rew = jnp.clip(r / jnp.sqrt(ret_rms.var + eps), -clip, clip)
    return ret, ret_rms, rew


# ===================================================================== runner-side time limit
class TLState(NamedTuple):
    env_state: Any
    t: jax.Array  # elapsed steps in the current episode


class TLStep(NamedTuple):
    state: TLState
    obs: jax.Array        # raw obs for the next action (the reset obs after a reset)
    tau: jax.Array        # time feature of `obs`
    reward: jax.Array
    term: jax.Array       # environmental termination
    timeout: jax.Array    # time limit reached without termination
    reset: jax.Array      # term OR timeout (episode boundary for any reason)
    final_obs: jax.Array  # true next obs s'_t (pre-reset) -- PEB bootstrap source (paper §3.1; Tonic)
    final_tau: jax.Array  # time feature of s'_t (Tonic: +1 for elapsed at t = T)


def tl_reset_obs(env: Env, key, T, mode):
    """Initial reset: (state, obs, tau(0))."""
    es, o = env.reset(key)
    return TLState(es, jnp.int32(0)), o, time_feature(0, T, mode)


def tl_step(env: Env, key, st: TLState, action, T, mode) -> TLStep:
    """One step with the runner-side limit T (Tonic Sequential.step semantics): reset = term or length == T; the
    env is reset (key split) only when reset; infos 'observations' = the pre-reset next obs."""
    k_step, k_reset = jax.random.split(key)
    es2, o2, r, term = env.step(k_step, st.env_state, action)
    t2 = st.t + 1
    timeout = jnp.logical_and(t2 >= T, jnp.logical_not(term))
    reset = jnp.logical_or(term, timeout)
    es_r, o_r = env.reset(k_reset)
    es_n = jax.tree.map(lambda a, b: jnp.where(reset, a, b), es_r, es2)
    o_n = jnp.where(reset, o_r, o2)
    t_n = jnp.where(reset, 0, t2)
    return TLStep(TLState(es_n, t_n), o_n, time_feature(t_n, T, mode), r, term, timeout, reset, o2,
                  time_feature(t2, T, mode))

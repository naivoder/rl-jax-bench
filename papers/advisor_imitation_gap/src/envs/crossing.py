"""MiniGrid LC Once Switch (paper App. A.5.5; official AskForHelpLavaCrossingOnce) as a pure-JAX environment.

Port of gym-minigrid 1.0.x CrossingEnv + MiniGridEnv.step/gen_obs and AllenAct v0.5.0 AskForHelpSimpleCrossing,
AskForHelpSimpleCrossingTask, MiniGridTask.query_expert and MiniGridTaskSampler:
  * S = 15, 7 lava rivers (AskForHelpLavaCrossingOnce)                                    [App. A.5.5 + official]
  * agent starts at (1, 1) facing right (dir 0); goal at (S-2, S-2)                         [gym-minigrid CrossingEnv]
  * actions (left, right, forward, toggle) -> minigrid (0, 1, 2, 5)                          [official task]
  * forward onto lava ends the episode with 0; onto the goal ends it with 1 - 0.9 * steps / max_steps;
    max_steps = 4 S^2 = 900                                                                   [gym-minigrid]
  * the 7x7x3 egocentric view (occlusion: see_through_walls = False) is all zeros (= MiniGrid's 'unseen' token)
    until the agent toggles; with toggle_is_permanent the view then stays lit                [official; differs from App. A.8]
  * teacher: shortest path on the (x, y, dir) graph of empty/goal cells, never toggles. Tie-breaking among
    equal-length paths: lowest action index (left, right, forward) -- ASSUMED (networkx order differs,
    the path LENGTH is identical and is the parity statistic)
  * a failed layout (episode reward 0) is repeated (same_seed_reset) until >= 1000 steps were spent on it [official]
  * layouts come from the JAX key (distributionally equivalent to gym-minigrid's numpy draws, not bit-identical)
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

S = 15                  # grid size (AskForHelpLavaCrossingOnce)
N_RIVERS = 7            # num_crossings
MAX_STEPS = 4 * S * S   # gym-minigrid CrossingEnv max_steps
VIEW = 7                # App. A.5.10 / official MG_AGENT_VIEW_SIZE
N_ACTIONS = 4           # left, right, forward, toggle
REPEAT_MIN_STEPS = 1000  # official repeat_failed_task_for_min_steps
TOGGLE_PERMANENT = True
# gym-minigrid OBJECT_TO_IDX / COLOR_TO_IDX
UNSEEN, EMPTY, WALL, GOAL, LAVA = 0, 1, 2, 8, 9
COLOR = {EMPTY: 0, WALL: 5, GOAL: 1, LAVA: 0}
DIR_TO_VEC = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], np.int32)
BIG = 10_000


class State(NamedTuple):
    grid: jax.Array        # int32 [S, S] object type indexed [x, y]
    dist: jax.Array        # int32 [S, S, 4] teacher distance-to-goal table
    pos: jax.Array         # int32 [2]
    dir: jax.Array         # int32
    t: jax.Array           # int32 steps in this episode (minigrid step_count)
    reveal: jax.Array      # bool
    ret: jax.Array         # float32
    success: jax.Array     # bool
    toggled: jax.Array     # bool (ever toggled this episode)
    layout_steps: jax.Array  # int32 steps spent on this layout (repeat rule)


# ------------------------------------------------------------------ layout generation
def gen_grid(key):
    """gym-minigrid CrossingEnv._gen_grid with a JAX key."""
    k_riv, k_path, k_open = jax.random.split(key, 3)
    cand_pos = jnp.concatenate([jnp.arange(2, S - 2, 2), jnp.arange(2, S - 2, 2)])
    cand_isv = jnp.concatenate([jnp.ones(len(range(2, S - 2, 2)), bool), jnp.zeros(len(range(2, S - 2, 2)), bool)])
    perm = jax.random.permutation(k_riv, cand_pos.shape[0])
    chosen = jnp.zeros(cand_pos.shape[0], bool).at[perm[:N_RIVERS]].set(True)
    vmask = chosen & cand_isv
    hmask = chosen & ~cand_isv
    nv, nh = vmask.sum(), hmask.sum()
    vcols = jnp.zeros(S, bool).at[cand_pos].max(vmask)
    hrows = jnp.zeros(S, bool).at[cand_pos].max(hmask)
    xs = jnp.arange(S)
    inner = (xs[:, None] >= 1) & (xs[:, None] <= S - 2) & (xs[None, :] >= 1) & (xs[None, :] <= S - 2)
    border = ~inner
    lava = inner & (vcols[:, None] | hrows[None, :])
    grid = jnp.where(border, WALL, jnp.where(lava, LAVA, EMPTY)).astype(jnp.int32)

    sorted_v = jnp.sort(jnp.where(vmask, cand_pos, BIG))
    sorted_h = jnp.sort(jnp.where(hmask, cand_pos, BIG))
    K = cand_pos.shape[0] + 2
    ks = jnp.arange(K)
    lim_v = jnp.where(ks == 0, 0, jnp.where(ks <= nv, jnp.concatenate([jnp.zeros(1, sorted_v.dtype), sorted_v,
                                                                         jnp.zeros(1, sorted_v.dtype)])[ks], S - 1))
    lim_h = jnp.where(ks == 0, 0, jnp.where(ks <= nh, jnp.concatenate([jnp.zeros(1, sorted_h.dtype), sorted_h,
                                                                         jnp.zeros(1, sorted_h.dtype)])[ks], S - 1))
    # path: nv 'h' moves (cross a vertical river) and nh 'v' moves, shuffled
    path_is_h = jax.random.permutation(k_path, jnp.arange(N_RIVERS) < nv)

    def body(c, x):
        grid, ri, rj = c
        is_h, k = x
        # 'h': opening on vertical river x = lim_v[ri + 1], y uniform in (lim_h[rj], lim_h[rj + 1])
        y_h = jax.random.randint(k, (), lim_h[rj] + 1, lim_h[rj + 1])
        x_v = jax.random.randint(k, (), lim_v[ri] + 1, lim_v[ri + 1])
        ox = jnp.where(is_h, lim_v[ri + 1], x_v)
        oy = jnp.where(is_h, y_h, lim_h[rj + 1])
        grid = grid.at[ox, oy].set(EMPTY)
        return (grid, ri + is_h.astype(jnp.int32), rj + (~is_h).astype(jnp.int32)), None

    (grid, _, _), _ = jax.lax.scan(body, (grid, jnp.int32(0), jnp.int32(0)),
                                   (path_is_h, jax.random.split(k_open, N_RIVERS)))
    grid = grid.at[S - 2, S - 2].set(GOAL)
    return grid


def teacher_distance(grid):
    """Shortest number of actions from (x, y, dir) to reach the goal cell (BIG if unreachable / invalid node)."""
    valid = (grid == EMPTY) | (grid == GOAL)
    d0 = jnp.where((grid == GOAL)[..., None], 0, BIG) * jnp.ones((1, 1, 4), jnp.int32)
    d0 = jnp.where(valid[..., None], d0, BIG)

    def relax(d):
        left = jnp.roll(d, 1, axis=2)     # d(x, y, dir - 1)
        right = jnp.roll(d, -1, axis=2)   # d(x, y, dir + 1)
        fwd = []
        for k in range(4):
            dx, dy = DIR_TO_VEC[k]
            nb = jnp.roll(d[:, :, k], shift=(-dx, -dy), axis=(0, 1))   # d(x + dx, y + dy, k)
            fwd.append(nb)
        fwd = jnp.stack(fwd, -1)
        best = jnp.minimum(jnp.minimum(left, right), fwd) + 1
        new = jnp.where(valid[..., None], jnp.minimum(d, best), BIG)
        return jnp.where((grid == GOAL)[..., None], 0, new)

    def cond(c):
        d, prev = c
        return jnp.any(d != prev)

    def step(c):
        d, _ = c
        return relax(d), d
    d, _ = jax.lax.while_loop(cond, step, (relax(d0), d0))
    return jnp.minimum(d, BIG).astype(jnp.int32)


def _fresh(grid, layout_steps):
    z = jnp.zeros((), jnp.int32)
    return State(grid, teacher_distance(grid), jnp.array([1, 1], jnp.int32), z, z, jnp.zeros((), bool),
                 jnp.zeros((), jnp.float32), jnp.zeros((), bool), jnp.zeros((), bool), layout_steps)


def reset(key):
    return _fresh(gen_grid(key), jnp.zeros((), jnp.int32))


def reset_from_grid(grid):
    """Start an episode on an imported layout (official parity tests)."""
    return _fresh(jnp.asarray(grid, jnp.int32), jnp.zeros((), jnp.int32))


def next_episode(s: State, fresh: State):
    """official MiniGridTaskSampler.next_task with repeat_failed_task_for_min_steps = 1000 (after an episode ends).
    Returns (next state, whether the fresh layout was consumed)."""
    spent = s.layout_steps + s.t
    repeat = (spent < REPEAT_MIN_STEPS) & (s.ret == 0)
    same = s._replace(pos=jnp.array([1, 1], jnp.int32), dir=jnp.zeros((), jnp.int32), t=jnp.zeros((), jnp.int32),
                      reveal=jnp.zeros((), bool), ret=jnp.zeros((), jnp.float32), success=jnp.zeros((), bool),
                      toggled=jnp.zeros((), bool), layout_steps=spent)
    return jax.tree_util.tree_map(lambda x, y: jnp.where(repeat, x, y), same, fresh), ~repeat


# ------------------------------------------------------------------ dynamics
def step(s: State, a):
    a = jnp.asarray(a, jnp.int32)
    reveal = (a == 3) | (s.reveal & TOGGLE_PERMANENT)
    t = s.t + 1
    ndir = jnp.where(a == 0, (s.dir - 1) % 4, jnp.where(a == 1, (s.dir + 1) % 4, s.dir))
    vec = jnp.asarray(DIR_TO_VEC)[s.dir]
    fwd = s.pos + vec
    cell = s.grid[fwd[0], fwd[1]]
    can_move = (a == 2) & (cell != WALL)
    pos = jnp.where(can_move, fwd, s.pos)
    hit_goal = can_move & (cell == GOAL)
    hit_lava = can_move & (cell == LAVA)
    r = jnp.where(hit_goal, 1.0 - 0.9 * (t / MAX_STEPS), 0.0).astype(jnp.float32)
    done = hit_goal | hit_lava | (t >= MAX_STEPS)
    ns = s._replace(pos=pos, dir=ndir, t=t, reveal=reveal, ret=s.ret + r, success=s.success | hit_goal,
                    toggled=s.toggled | (a == 3))
    return ns, r, done


def _view_world_coords(pos, d):
    vx, vy = jnp.meshgrid(jnp.arange(VIEW), jnp.arange(VIEW), indexing="ij")
    f = jnp.asarray(DIR_TO_VEC)[d]
    rgt = jnp.asarray(DIR_TO_VEC)[(d + 1) % 4]
    wx = pos[0] + f[0] * (VIEW - 1 - vy) + rgt[0] * (vx - VIEW // 2)
    wy = pos[1] + f[1] * (VIEW - 1 - vy) + rgt[1] * (vx - VIEW // 2)
    return wx, wy


def _process_vis_scalar(opaque):
    """Literal gym-minigrid Grid.process_vis (sequential sweep on scalars). Reference for _process_vis (tests)."""
    tr = [[~opaque[i, j] for j in range(VIEW)] for i in range(VIEW)]
    F = jnp.zeros((), bool)
    m = [[F for _ in range(VIEW)] for _ in range(VIEW)]
    m[VIEW // 2][VIEW - 1] = jnp.ones((), bool)
    for j in reversed(range(VIEW)):
        for i in range(VIEW - 1):
            go = m[i][j] & tr[i][j]
            m[i + 1][j] = m[i + 1][j] | go
            if j > 0:
                m[i + 1][j - 1] = m[i + 1][j - 1] | go
                m[i][j - 1] = m[i][j - 1] | go
        for i in reversed(range(1, VIEW)):
            go = m[i][j] & tr[i][j]
            m[i - 1][j] = m[i - 1][j] | go
            if j > 0:
                m[i - 1][j - 1] = m[i - 1][j - 1] | go
                m[i][j - 1] = m[i][j - 1] | go
    return jnp.stack([jnp.stack(row) for row in m])


_IDX = np.arange(VIEW)


def _process_vis(opaque):
    """gym-minigrid Grid.process_vis, vectorised per row. Within row j the rightward sweep gives
    m_r[i] = OR_{k<=i} m[k] & (no opaque cell in k..i-1); the leftward sweep m_l[i] = OR_{k>=i} m_r[k] & (no opaque
    in i+1..k); 'go' cells (visible and transparent) light the cell beside them and the cells below in row j-1.
    Equal to _process_vis_scalar on all 2^49-style random masks tested (tests/test_grads.py)."""
    tr = ~opaque                                         # [i, j]
    rows = [None] * VIEW
    nxt = jnp.zeros(VIEW, bool).at[VIEW // 2].set(True)  # initial mask of the bottom row
    le = _IDX[:, None] <= _IDX[None, :]                  # [k, i]: k <= i
    for j in reversed(range(VIEW)):
        t = tr[:, j]
        m0 = nxt
        c = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum((~t).astype(jnp.int32))])   # c[i] = #opaque < i
        # right sweep: all tr[k..i-1]  <=>  c[i] - c[k] == 0
        reach_r = le & (c[None, :VIEW] == c[:VIEW, None])
        m_r = jnp.any(m0[:, None] & reach_r, axis=0)
        go_r = m_r & t & (_IDX < VIEW - 1)
        # left sweep: all tr[i+1..k]  <=>  c[k+1] - c[i+1] == 0, k >= i
        reach_l = le & (c[None, 1:] == c[1:, None])          # [i, k]
        m_l = jnp.any(reach_l & m_r[None, :], axis=1)
        go_l = m_l & t & (_IDX > 0)
        rows[j] = m_l
        if j > 0:
            below = jnp.zeros(VIEW, bool)
            below = below | go_r | jnp.concatenate([jnp.zeros(1, bool), go_r[:-1]])      # i and i+1 from go_r[i]
            below = below | go_l | jnp.concatenate([go_l[1:], jnp.zeros(1, bool)])       # i and i-1 from go_l[i]
            nxt = below
    return jnp.stack(rows, axis=1)                       # [i, j]


def lit_obs(s: State):
    """Egocentric 7x7x3 (object, colour, state) image as gym-minigrid gen_obs()['image'] (indexed [x, y, c])."""
    wx, wy = _view_world_coords(s.pos, s.dir)
    inb = (wx >= 0) & (wx < S) & (wy >= 0) & (wy < S)
    typ = jnp.where(inb, s.grid[jnp.clip(wx, 0, S - 1), jnp.clip(wy, 0, S - 1)], WALL)
    vis = _process_vis(typ == WALL)
    typ = typ.at[VIEW // 2, VIEW - 1].set(EMPTY)    # the agent's own cell is shown as empty
    col = jnp.select([typ == WALL, typ == GOAL], [COLOR[WALL], COLOR[GOAL]], 0)
    img = jnp.stack([jnp.where(vis, typ, UNSEEN), jnp.where(vis, col, 0), jnp.zeros_like(typ)], -1)
    return img.astype(jnp.int32)


def obs(s: State):
    return jnp.where(s.reveal, lit_obs(s), 0).astype(jnp.uint8)


def teacher(s: State):
    """Next action on a shortest path (left, right, forward); never toggles. Ties between equally short moves are
    broken as the official networkx expert does (exhaustive check over 17 432 states of 100
    official layouts, tests/test_official_parity.py): lowest action index, EXCEPT a left/right tie while facing
    direction 0, where the official graph's adjacency order (edge to rotation 1 inserted before rotation 3) yields
    'right'. An earlier version returned 'left' on that tie (2.9 % of states differ; the first full-scale MG runs used it)."""
    d = s.dist
    x, y, k = s.pos[0], s.pos[1], s.dir
    vec = jnp.asarray(DIR_TO_VEC)[k]
    fx, fy = x + vec[0], y + vec[1]
    cand = jnp.stack([d[x, y, (k - 1) % 4], d[x, y, (k + 1) % 4], d[fx, fy, k]])
    a = jnp.argmin(cand)
    if OFFICIAL_TIEBREAK:
        a = jnp.where((k == 0) & (cand[0] == cand[1]) & (cand[1] < cand[2]), 1, a)
    return a.astype(jnp.int32)


OFFICIAL_TIEBREAK = True    # False reproduces the earlier teacher (used by the first preliminary and full-scale MG runs)


def episode_info(s: State):
    return {"ret": s.ret, "len": s.t.astype(jnp.float32), "success": s.success.astype(jnp.float32),
            "toggled": s.toggled.astype(jnp.float32)}


def _make_spec():
    from ..data import EnvSpec
    return EnvSpec(
        name="mg", n_actions=N_ACTIONS, max_steps=MAX_STEPS,
        reset=lambda key, ep: reset(key),
        step=lambda s, a, ep: step(s, a),
        obs=lambda s, ep: obs(s),
        teacher=lambda s, ep: teacher(s),
        info=episode_info,
        eval_reset=lambda k, key, ep: reset(jax.random.fold_in(jax.random.PRNGKey(2 ** 31 - 1), k)),
        next_episode=lambda s, fresh, ep: next_episode(s, fresh),
        pool_size=4,   # <= 3 successes fit in one 100-step rollout (shortest path >= 28 actions); overflow is impossible (analytic bound: <= 4 fresh layouts per rollout, a success needs >= 25 steps and a layout expiry happens at most once; REVIEW A13). pool_overflow is computed per update but not persisted
    )


_SPEC = []


def __getattr__(name):
    if name == "SPEC":
        if not _SPEC:
            _SPEC.append(_make_spec())
        return _SPEC[0]
    raise AttributeError(name)

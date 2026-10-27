"""Policy-driven differential test of our env against the ORIGINAL numpy MinAtar.

The random-action parity fixtures (scripts/record_official_env.py) give short episodes (Breakout ~13 steps, SpaceInvaders
~48, Asterix ~83), so late-game mechanics are barely exercised.  Here actions come from the trained PPO demonstrators
(eps 0.01; `seeker` / `tracker` are scripted policies), so episodes are as long as demonstrations and learners see.  The
original's random draws are scripted from gymnax's keys exactly as in record_official_env.py; for the repaired Asterix the
script follows `src.envs.asterix_spawn_repaired`.

usage: policy_parity.py GAME WHO T SEED [--repair] [--stop]
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import yaml  # noqa: E402
from gymnax.environments.minatar import asterix as gx_asterix  # noqa: E402

from record_official_env import GAMES, Script, keys_for, load_official  # noqa: E402
from src.envs import MinAtar, asterix_spawn_repaired  # noqa: E402

DEMO_DIR = os.path.join(REPO, "data", "demos")


def ppo_logits_fn(game, who, m, demo_dir=DEMO_DIR):
    from flax import serialization
    from src.model import ActorCritic
    cfg = yaml.safe_load(open(os.path.join(REPO, "src", "configs", "default.yaml")))
    net = ActorCritic(m.n_actions, channels=cfg["network"]["conv_channels"], hidden=cfg["ppo"]["hidden"])
    tmpl = net.init(jax.random.PRNGKey(0), jnp.zeros((1,) + m.obs_shape, jnp.float32))
    fn = f"ppo_{game}.msgpack" if who in ("expert", "seeker", "tracker") else f"ppo_{game}_subopt.msgpack"
    pp = serialization.from_bytes(tmpl, open(os.path.join(demo_dir, fn), "rb").read())
    return jax.jit(lambda o: net.apply(pp, o[None].astype(jnp.float32))[0][0])


def run(game, who="expert", T=4000, seed=1, repair=False, stop_at_bad=False, demo_dir=DEMO_DIR):
    """Drive our env and the original side by side; return mismatch counts and coverage statistics."""
    mod = load_official(GAMES[game])
    m = MinAtar(game, max_steps=10 ** 6, asterix_spawn_repair=repair)
    params = m.params
    logits_fn = ppo_logits_fn(game, who, m, demo_dir)
    step_key, reset_key = keys_for(game, seed + 77)
    rng = np.random.default_rng(1000 + seed)
    aset = np.asarray(m.env.action_set)
    step = jax.jit(m.step)
    reset = jax.jit(lambda k: m.env.reset_env(k, params))
    spawn_gx = jax.jit(gx_asterix.spawn_entity)
    spawn_fix = jax.jit(asterix_spawn_repaired)
    scr = Script()
    off = mod.Env(ramping=True) if game != "Breakout" else mod.Env()
    off.random = scr
    nres = [0]

    def do_reset():
        obs, st = reset(reset_key(nres[0])); nres[0] += 1
        if game == "Breakout":
            scr.push("randint", 0 if int(st.ball_dir) == 2 else 1)
        off.reset()
        return np.asarray(obs) > 0.5, st

    gobs, st = do_reset()
    first_bad = None
    n_bad_obs = n_bad_r = n_bad_t = n_gauge = 0
    ep_len, ep_ret, lens, rets = 0, 0.0, [], []
    extra = dict(max_ramp_index=0, brick_respawns=0, asterix_full_spawns=0)
    for t in range(T):
        if who == "tracker":     # Breakout: paddle toward the ball's next column (clears walls -> brick respawn)
            dx = 1 if off.ball_dir in (1, 2) else -1
            tgt = min(9, max(0, off.ball_x + dx))
            a = 0 if off.pos == tgt else (1 if tgt < off.pos else 2)
            if rng.random() < 0.05:
                a = int(rng.integers(3))
        elif rng.random() < 0.01:
            a = int(rng.integers(m.n_actions))
        else:
            a = int(np.argmax(np.asarray(logits_fn(gobs))))
        full = int(aset[a])
        if game == "Asterix" and int(st.spawn_timer) == 0:
            free = [i for i in range(len(off.entities)) if off.entities[i] is None]
            if repair:
                ent, slot, anyf = spawn_fix(step_key(t), st); anyf = bool(anyf)
            else:
                ent, slot = spawn_gx(step_key(t), st); anyf = int(ent[4]) == 1
            ent, slot = np.asarray(ent), int(slot)
            scr.push("rand", 0.25 if int(ent[2]) == 1 else 0.75); scr.push("rand", 0.1 if int(ent[3]) == 1 else 0.9)
            if not free:
                extra["asterix_full_spawns"] += 1
            if free:   # the original draws a slot among its free rows; an unrepaired gymnax no-find (perm position 0) diverges here
                scr.push("randint", free.index(slot) if (anyf and slot in free) else 0)
        if game == "Seaquest":
            ke, kd = jax.random.split(step_key(t)); kc, ky = jax.random.split(ke)
            vals = np.asarray(jax.random.uniform(kc, (2,))); ey = int(jax.random.randint(ky, (), 1, 9))
            kdd, kdy = jax.random.split(kd); ddir = bool(jax.random.bernoulli(kdd)); dy = int(jax.random.randint(kdy, (), 1, 9))
            if int(st.e_spawn_timer) == 0:
                scr.push("rand", 0.25 if vals[0] < 0.5 else 0.75); scr.push("rand", 0.1 if vals[1] < 1 / 3 else 0.9); scr.push("randint", ey)
            if int(st.d_spawn_timer) == 0:
                scr.push("rand", 0.25 if ddir else 0.75); scr.push("randint", dy)
        pre_bricks = int(off.brick_map.sum()) if game == "Breakout" else 0
        gobs_j, st, gr, gterm, _ = step(step_key(t), st, a)
        gobs = np.asarray(gobs_j)
        r, term = off.act(full)
        if game == "Breakout" and int(off.brick_map.sum()) > pre_bricks:
            extra["brick_respawns"] += 1
        if hasattr(off, "ramp_index"):
            extra["max_ramp_index"] = max(extra["max_ramp_index"], int(off.ramp_index))
        oobs = off.state()
        bad = not np.array_equal(gobs, oobs)
        if bad and (bool(gterm) or term) and game == "Seaquest" and all(
                np.array_equal(gobs[..., c], oobs[..., c]) for c in range(oobs.shape[-1]) if c not in (7, 8)):
            n_gauge += 1; bad = False     # declared: gymnax refills the gauges on the terminating step
        br, bt = float(gr) != float(r), bool(gterm) != bool(term)
        n_bad_obs += bad; n_bad_r += br; n_bad_t += bt
        if (bad or br or bt) and first_bad is None:
            first_bad = dict(t=t, ep=len(lens), ep_step=ep_len, obs=bool(bad), r=(float(gr), float(r)), term=(bool(gterm), bool(term)))
            if stop_at_bad:
                break
        ep_len += 1; ep_ret += r
        if term or bool(gterm):
            lens.append(ep_len); rets.append(ep_ret); ep_len, ep_ret = 0, 0.0
            gobs, st = do_reset()
    return dict(game=game, who=who, seed=seed, T=T, repair=repair, episodes=len(lens), ep_len_max=int(max(lens + [ep_len])),
                ret_max=float(max(rets + [ep_ret])), bad_obs=int(n_bad_obs), bad_r=int(n_bad_r), bad_t=int(n_bad_t),
                terminal_gauge_only=n_gauge, first_bad=first_bad, **extra)


if __name__ == "__main__":
    import json
    g, w, T, s = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    print(json.dumps(run(g, w, T, s, repair="--repair" in sys.argv, stop_at_bad="--stop" in sys.argv)))

"""[JSRL: copied from the DQfD reproduction (`papers/dqfd`) unchanged apart from this note and the source path; imports OUR src/envs.py; needs scripts/official/setup.sh]
Differential test of our env (gymnax MinAtar port) against the ORIGINAL numpy MinAtar (kenjyoung/MinAtar
4241f246, the code the gymnax port derives from).  Runs under the JAX venv, CPU:

    python scripts/official/record_official_env.py tests/fixtures/env_parity.npz

For every game, the same action sequence is fed to the original numpy env (sticky actions OFF: they live in the
`Environment` wrapper, not in the per-game `Env` classes) and to gymnax; observations, rewards and terminals are
compared at every step.  The random draws of the original (reset ball side; Asterix spawn side / kind / slot) are
SCRIPTED from the gymnax side (gymnax's own `spawn_entity(key, state)` is called with the key of that step), so
"same random outcome in, same state out" is tested exactly.  Space Invaders has no randomness after reset.  Seaquest
is scripted the same way from gymnax's per-step key splits (enemy: direction, kind, row; diver: direction, row).
The recorded ORIGINAL observations/rewards/terminals are saved as fixtures (tests/test_official_parity.py replays
gymnax and compares against them, without needing the original source).
"""
import importlib.util
import os
import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402
gpu_bootstrap.setup()
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import gymnax  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.envs import MinAtar  # noqa: E402
from gymnax.environments.minatar import asterix as gx_asterix  # noqa: E402

OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external", "official"))
SRC = os.path.join(OFFICIAL_ROOT, "MinAtar", "minatar", "environments")   # kenjyoung/MinAtar 4241f246 (setup.sh)
GAMES = {"Breakout": "breakout", "SpaceInvaders": "space_invaders", "Asterix": "asterix", "Seaquest": "seaquest"}
T = 3000
T_GAME = {"Seaquest": 8000}   # rescue / surfacing paths need many steps


def load_official(mod):
    spec = importlib.util.spec_from_file_location(f"official_{mod}", f"{SRC}/{mod}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class Script:
    """Stand-in for np.random.RandomState that returns pre-chosen outcomes."""
    def __init__(self): self.q = []
    def push(self, kind, v): self.q.append((kind, v))
    def rand(self, *a):
        k, v = self.q.pop(0); assert k == "rand", k; return v
    def randint(self, *a, **k):
        kind, v = self.q.pop(0); assert kind == "randint", kind; return v


def keys_for(game, seed=0):
    """Deterministic key stream shared by recorder and test: step t uses fold_in(root, t), resets fold_in(root, 10**6 + n)."""
    root = jax.random.PRNGKey(1000 + seed)
    return (lambda t: jax.random.fold_in(root, t)), (lambda n: jax.random.fold_in(root, 10 ** 6 + n))


def action_stream(game, seed=0):
    rng = np.random.default_rng(seed)
    n = T_GAME.get(game, T)
    return rng.integers(0, 6, n), rng.random(n)


def run(game, seed=0, record=True, upstream_fixes=True):
    mod = load_official(GAMES[game])
    m = MinAtar(game, max_steps=10 ** 6, upstream_fixes=upstream_fixes)   # no time cap: the original has none
    env, params = m.env, m.params
    step_key, reset_key = keys_for(game, seed)
    a_raw, u = action_stream(game, seed)
    n_act = env.num_actions
    aset = np.asarray(env.action_set)
    step = jax.jit(env.step_env)
    reset = jax.jit(env.reset_env)
    spawn = jax.jit(gx_asterix.spawn_entity) if game == "Asterix" else None

    scr = Script()
    off = mod.Env(ramping=True) if game != "Breakout" else mod.Env()
    off.random = scr
    nres = 0

    def do_reset():
        nonlocal nres
        obs, st = reset(reset_key(nres), params); nres += 1
        if game == "Breakout":
            scr.push("randint", 0 if int(st.ball_dir) == 2 else 1)
        off.reset()
        return obs, st

    gobs, st = do_reset()
    out = dict(act=[], g_obs=[], o_obs=[], g_r=[], o_r=[], g_term=[], o_term=[], reset=[])
    n_mismatch_obs = n_mismatch_r = n_mismatch_t = n_mm_obs_terminal = 0
    gauge_only = True
    n_ep, tot_r = 0, 0.0
    off_obs0 = off.state()
    if not np.array_equal(np.asarray(gobs) > 0.5, off_obs0):
        n_mismatch_obs += 1
    for t in range(T_GAME.get(game, T)):
        a = int(a_raw[t] % n_act)
        if game == "Breakout" and u[t] < 0.6:          # ball-tracking paddle on 60 % of steps (clears bricks, longer episodes)
            a = 0 if off.pos == off.ball_x else (1 if off.ball_x < off.pos else 2)
        if game == "Seaquest" and u[t] < 0.7:           # diver-seeker: collect 6 divers, then surface (exercises rescue / refill)
            tgt = None
            if off.diver_count >= 6 or not off.divers:
                tgt = (off.sub_x, 0) if off.diver_count >= 6 else None
            else:
                d = min(off.divers, key=lambda z: abs(z[0] - off.sub_x) + abs(z[1] - off.sub_y)); tgt = (d[0], d[1])
            if tgt is not None:
                a = 2 if off.sub_y > tgt[1] and off.sub_x == tgt[0] else (4 if off.sub_y < tgt[1] and off.sub_x == tgt[0] else
                    (1 if off.sub_x > tgt[0] else 3 if off.sub_x < tgt[0] else 0))
                if off.sub_y != tgt[1] and off.sub_x != tgt[0]:
                    a = 4 if off.sub_y < tgt[1] else 2
            elif u[t] < 0.35:
                a = 5
        full = int(aset[a])
        if game == "Asterix" and int(st.spawn_timer) == 0:
            ent, slot = spawn(step_key(t), st)
            ent = np.asarray(ent); slot = int(slot)
            scr.push("rand", 0.25 if int(ent[2]) == 1 else 0.75)    # lr: gymnax lr=1 <=> official lr True
            scr.push("rand", 0.1 if int(ent[3]) == 1 else 0.9)      # is_gold
            free = [i for i in range(len(off.entities)) if off.entities[i] is None]
            if free:
                assert int(ent[4]) == 1, "gymnax says no free slot but the original has one"
                scr.push("randint", free.index(slot))
            else:
                assert int(ent[4]) == 0
        if game == "Seaquest":                           # replicate gymnax's draws (seaquest.py _spawn_entities) for the original
            ke, kd = jax.random.split(step_key(t))
            kc, ky = jax.random.split(ke)
            vals = np.asarray(jax.random.uniform(kc, (2,))); ey = int(jax.random.randint(ky, (), 1, 9))
            kdd, kdy = jax.random.split(kd)
            ddir = bool(jax.random.bernoulli(kdd)); dy = int(jax.random.randint(kdy, (), 1, 9))
            if int(st.e_spawn_timer) == 0:
                scr.push("rand", 0.25 if vals[0] < 0.5 else 0.75); scr.push("rand", 0.1 if vals[1] < 1 / 3 else 0.9); scr.push("randint", ey)
            if int(st.d_spawn_timer) == 0:
                scr.push("rand", 0.25 if ddir else 0.75); scr.push("randint", dy)
        gobs, st, gr, gdone, _ = step(step_key(t), st, a, params)
        r, term = off.act(full)
        oobs = off.state()
        gobs_b = np.asarray(gobs) > 0.5
        bad = not np.array_equal(gobs_b, oobs)
        n_mismatch_obs += int(bad)
        if bad and (bool(gdone) or term):                 # a differing TERMINAL observation (never bootstrapped from)
            n_mm_obs_terminal += 1
            gauge_only &= all(np.array_equal(gobs_b[..., c], oobs[..., c]) for c in range(oobs.shape[-1]) if c not in (7, 8))
        n_mismatch_r += int(float(gr) != float(r))
        n_mismatch_t += int(bool(gdone) != bool(term))
        tot_r += r
        if record:
            out["act"].append(a); out["g_r"].append(float(gr)); out["o_r"].append(float(r))
            out["g_term"].append(bool(gdone)); out["o_term"].append(bool(term)); out["o_obs"].append(oobs)
        if term or bool(gdone):
            n_ep += 1
            out["reset"].append(t)
            gobs, st = do_reset()
    res = dict(game=game, steps=T_GAME.get(game, T), episodes=n_ep, total_reward=float(tot_r), mismatch_obs=n_mismatch_obs,
               mismatch_reward=n_mismatch_r, mismatch_term=n_mismatch_t,
               mismatch_obs_at_terminal_step=n_mm_obs_terminal, terminal_mismatch_gauge_channels_only=bool(gauge_only))
    return res, out


if __name__ == "__main__":
    path = sys.argv[1]
    save, summary = {}, []
    for g in GAMES:
        if g == "SpaceInvaders":                       # the same sequence on the UNFIXED gymnax port, for the record
            res0, _ = run(g, record=False, upstream_fixes=False)
            res0["game"] = "SpaceInvaders (gymnax 1.0.0 as shipped, no repair)"
            print("[INFO]", res0); summary.append(res0)
        res, out = run(g)
        print("[INFO]", res)
        summary.append(res)
        save[f"{g}_act"] = np.asarray(out["act"], np.int8)
        save[f"{g}_obs"] = np.packbits(np.asarray(out["o_obs"]).reshape(len(out["o_obs"]), -1), axis=1)
        save[f"{g}_r"] = np.asarray(out["o_r"], np.float32)
        save[f"{g}_term"] = np.asarray(out["o_term"], bool)
    import json
    np.savez_compressed(path, **save, summary=json.dumps(summary))

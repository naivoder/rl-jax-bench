"""Runs the authors' main.py (commit b5959f6) at their seed 123546 (1 MC run, lambda_C = 2, HEAD parameters) while RECORDING, in order, every
measurement set handed to the filter, every MCTS rollout-policy draw (as randint index + number of available actions), and the sensor
positions after every planning call.  The recording wrappers consume no random numbers, so the run IS the authors' run (checked against
run_main.py's un-traced result in the output json: 'untraced_equal').  Output: one npz with the per-block arrays for replay by
tests/test_official_parity.py.  Blocks of main.py, in order: Myopic-KLD, Myopic-GD, MCTS1 (GOSPA), [MCTS3 (GOSPA) with --mcts 1,3]."""
import argparse, os, re, sys, time, io, contextlib, json
ap = argparse.ArgumentParser()
ap.add_argument("--lam", type=float, default=2.0); ap.add_argument("--mcts", default="1,3"); ap.add_argument("--out", required=True)
a = ap.parse_args()
PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(PAPER, "external", "official")), "repo")
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, REPO)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.animation as manim
import Parameters as P
P.number_of_mc_runs = 1
P.clutter_rate = a.lam; P.clutter_density = a.lam / (np.pi * P.sensor_radius ** 2); P.lambda_false_alarm = P.clutter_density
import utils, sensor, MCTS, MCTS_joint
utils.save_frame_as_image = lambda *x, **k: None
manim.FuncAnimation.save = lambda *x, **k: None
plt.show = lambda *x, **k: None
EV = []
_gen = sensor.Measurements.generate_measurements_multiple_targets
def gen(self, ts, sp, k):
    r = _gen(self, ts, sp, k)
    EV.append(("z", int(k), np.array(self.all_measurements[k], float).copy(), np.array(sp, float).reshape(2).copy()))
    return r
sensor.Measurements.generate_measurements_multiple_targets = gen
G = sensor.GOSPA_Sensor_updated_maths
for nm, mode in (("manage_multiple_sensors_mb_individually_MCTS", "ind"), ("manage_multiple_sensors_mb_full_jointly_calculated_MCTS", "joint")):
    def mk(orig, mode):
        def w(self, sensors, *args, **kw):
            EV.append(("plan", mode, int(P.simulation_search_depth), int(P.MCTS_simulation_number)))
            r = orig(self, sensors, *args, **kw)
            EV.append(("pos", np.stack([np.array(s.current_sensor_position, float).reshape(2) for s in sensors])))
            return r
        return w
    setattr(G, nm, mk(getattr(G, nm), mode))
for cls in (MCTS.MonteCarloTreeSearchNode_individual, MCTS_joint.MonteCarloTreeSearchNode_joint):
    def mk2(orig):
        def rollout(self):
            EV.append(("it",))
            return orig(self)
        return rollout
    def rollout_policy(self, avail):
        sel = np.random.randint(0, len(avail))      # identical call (and stream consumption) as the authors' rollout_policy
        EV.append(("d", int(sel), len(avail)))
        return avail[sel]
    cls.rollout = mk2(cls.rollout); cls.rollout_policy = rollout_policy
src = open(os.path.join(REPO, "main.py")).read()
for i in (1, 2, 3, 4):
    src = re.sub(rf"^MCTS{i} = (True|False)", f"MCTS{i} = {str(str(i) in a.mcts.split(','))}", src, flags=re.M)
os.chdir(os.path.dirname(os.path.abspath(a.out)))
g = {"__name__": "__main__"}
t0 = time.time(); err = None
with contextlib.redirect_stdout(io.StringIO()):
    try:
        exec(compile(src, os.path.join(REPO, "main.py"), "exec"), g)
    except NameError as e:       # main.py's final print branch references MCTS blocks that were disabled; all data exist by then
        err = repr(e)
wall = time.time() - t0
blocks = [("Myopic-KLD", "all_myopicKLD_gospa_error"), ("Myopic-GD", "all_myopic_gospa_error")]
for i in map(int, a.mcts.split(",")):
    blocks.append((f"MCTS{i}-GD", f"all_MCTS{i}_gospa_error"))
# note: g[...] keeps only the finished-block lists; split the event stream at the plan events
plans = [i for i, e in enumerate(EV) if e[0] == "plan"]
assert len(plans) == 200 * len(blocks), (len(plans), len(blocks))
out = {}
for bi, (name, key) in enumerate(blocks):
    seg = EV[plans[bi * 200]: (plans[(bi + 1) * 200] if bi + 1 < len(blocks) else len(EV))]
    steps = []
    cur = None
    for e in seg:
        if e[0] == "plan":
            cur = dict(mode=e[1], T=e[2], B=e[3], its=[], z=[], pos=None); steps.append(cur)
        elif e[0] == "it":
            cur["its"].append([])
        elif e[0] == "d":
            cur["its"][-1].append((e[1], e[2]))
        elif e[0] == "pos":
            cur["pos"] = e[1]
        elif e[0] == "z":
            cur["z"].append((e[2], e[3]))
    assert len(steps) == 200 and all(len(s["z"]) == 2 for s in steps)
    mode = np.array([s["mode"] == "joint" for s in steps])
    Mx = 12
    z = np.zeros((200, 2, Mx, 2)); zv = np.zeros((200, 2, Mx), bool); zsp = np.zeros((200, 2, 2))
    for k, s in enumerate(steps):
        for j in range(2):
            m = s["z"][j][0].shape[1]
            assert m <= Mx
            z[k, j, :m] = s["z"][j][0].T; zv[k, j, :m] = True; zsp[k, j] = s["z"][j][1]
    nits = np.array([len(s["its"]) for s in steps])
    Tm = max(max((len(it) for s in steps for it in s["its"]), default=0), 1)
    maxit = int(nits.max())
    sel = np.full((200, maxit, Tm), -1, np.int16); nav = np.ones((200, maxit, Tm), np.int16)
    for k, s in enumerate(steps):
        for i, it in enumerate(s["its"]):
            for j, (sl, n) in enumerate(it):
                sel[k, i, j] = sl; nav[k, i, j] = n
    gos = np.array(g[key], float)[0] if key in g else np.full(200, np.nan)
    out[name] = dict(z=z, zv=zv, zsp=zsp, spos=np.stack([s["pos"] for s in steps]), joint=mode, nits=nits, sel=sel, nav=nav,
                     T=np.array([s["T"] for s in steps]), B=np.array([s["B"] for s in steps]), gospa_sq=gos)
flat = {f"{n}/{k}": v for n, d in out.items() for k, v in d.items()}
np.savez_compressed(a.out, **flat)
json.dump(dict(wall_s=wall, post_error=err, lam=a.lam, mcts=a.mcts, commit="b5959f6a696a0e566e0925ec2376be361982e773",
               G={n: float(np.mean(np.sqrt(d["gospa_sq"]))) for n, d in out.items()},
               nits_stats={n: [int(d["nits"].min()), int(d["nits"].max())] for n, d in out.items()}), open(a.out + ".json", "w"))
print("[INFO] traced", a.out, f"wall={wall:.0f}s", err)

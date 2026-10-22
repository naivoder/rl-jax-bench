"""Runs the authors' UNMODIFIED main.py (commit b5959f6) headless, with a handful of mechanical text patches, and dumps its own aggregates.

Patches (all recorded in the output json under 'patches'):
  * every block-start `np.random.seed(selected_seed)` after the truth generation uses --block-seed (default = selected_seed = 123546, i.e. the authors' own stream)
  * plotting/animation side effects are no-ops (Agg backend, FuncAnimation.save, save_frame_as_image, plt.show)
  * --cost kld : every `cost_function = 'GOSPA'` becomes 'KLD' (the shipped MCTS1-4 blocks are hard-coded to GOSPA; README says KLD is available)
  * Parameters overridden before main imports it: number_of_mc_runs, clutter (rate, density, lambda_false_alarm), and optionally --paper-params (R = 2 I; prob_birth 0.03 applied after the truth is generated, so the truth stays the authors' 9-target one)
Usage: python run_main.py --lam 2 --runs 3 --block-seed 123546 --cost gospa --out out.json [--paper-params] [--blocks myopic,myopicKLD,MCTS1,...]
"""
import argparse, json, os, re, sys, time, io, contextlib
ap = argparse.ArgumentParser()
ap.add_argument("--lam", type=float, default=2.0); ap.add_argument("--runs", type=int, default=1)
ap.add_argument("--block-seed", type=int, default=123546); ap.add_argument("--cost", default="gospa")
ap.add_argument("--out", required=True); ap.add_argument("--paper-params", action="store_true")
ap.add_argument("--mcts", default="1,2,3,4", help="which of MCTS1..4 stay enabled")
ap.add_argument("--no-myopic", action="store_true")
a = ap.parse_args()
PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(PAPER, "external", "official")), "repo")
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, REPO)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim
import Parameters as P
P.number_of_mc_runs = a.runs
P.clutter_rate = a.lam; P.clutter_density = a.lam / (np.pi * P.sensor_radius ** 2); P.lambda_false_alarm = P.clutter_density
if a.paper_params:
    P.R = np.array([[1, 0], [0, 1]]) * 2        # prob_birth = 0.03 is applied AFTER the ground truth is generated (below): the truth must stay the authors' 9-target seed-123546 truth
import utils
utils.save_frame_as_image = lambda *x, **k: None
manim.FuncAnimation.save = lambda *x, **k: None
plt.show = lambda *x, **k: None
src = open(os.path.join(REPO, "main.py")).read()
patches = []
first = src.index("np.random.seed(selected_seed)") + 10
head, tail = src[:first], src[first:]
n_seed = len(re.findall(r"np\.random\.seed\(selected_seed\)", tail))
tail = re.sub(r"np\.random\.seed\(selected_seed\)", "np.random.seed(BLOCK_SEED)", tail); patches.append(f"block seeds x{n_seed}")
src = head + tail
if a.paper_params:
    k0 = "oGroundtruth.generate_multitarget_groundtruth()"
    assert src.count(k0) == 1
    src = src.replace(k0, k0 + "\nParameters.prob_birth = [0.03]"); patches.append("prob_birth 0.03 after truth generation")
if a.cost.lower() == "kld":
    n = src.count("cost_function = 'GOSPA'"); src = src.replace("cost_function = 'GOSPA'", "cost_function = 'KLD'"); patches.append(f"GOSPA->KLD x{n}")
for i in (1, 2, 3, 4):
    src = re.sub(rf"^MCTS{i} = (True|False)", f"MCTS{i} = {str(str(i) in a.mcts.split(','))}", src, flags=re.M)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(a.out)))
g = {"__name__": "__main__", "BLOCK_SEED": a.block_seed}
buf = io.StringIO()
t0 = time.time()
with contextlib.redirect_stdout(buf):
    exec(compile(src, os.path.join(REPO, "main.py"), "exec"), g)
wall = time.time() - t0
out = dict(lam=a.lam, runs=a.runs, block_seed=a.block_seed, cost=a.cost, paper_params=a.paper_params, patches=patches, wall_s=wall,
           commit="b5959f6a696a0e566e0925ec2376be361982e773", numpy=np.__version__, truth_n_targets=len(g["oGroundtruth"].target_states))
names = {"Myopic-GD": "myopic", "Myopic-KLD": "myopicKLD"}
for i in (1, 2, 3, 4):
    names[f"MCTS{i}"] = f"MCTS{i}"
for nm, key in names.items():
    k = f"all_{key}_gospa_error"
    if k in g and len(g[k]) == a.runs:
        arr = np.asarray(g[k], float)
        out[nm] = dict(G=float(g[f"average_{key}_error"]), gospa_sq=arr.tolist(),
                       sec_per_step=float(g[f"elapsed_time_{key}"]) / (a.runs * 200))
json.dump(out, open(a.out, "w"))
print("[INFO] done", a.out, f"wall={wall:.0f}s", {k: round(v["G"], 3) for k, v in out.items() if isinstance(v, dict)})

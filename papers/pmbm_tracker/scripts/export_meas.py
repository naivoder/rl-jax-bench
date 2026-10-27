"""Export the confirmatory measurement sets (Fig.-3 truth, measurement root 4242, Table II cell (0.9, 10), config_id 3) for the
Octave oracles: <out>/meas.npz (z, zvalid, X, alive, meta) and <out>/in_paper_T{0,1e-4}.mat (authors' PMBM code, T_pruning = 0 / 1e-4) and
<out>/in_williams.mat (TOMB/MOMB).  Same call as src/train.py --truth fig3 --meas-seed 4242, so run r is byte-identical to the confirmatory runs.
  python scripts/export_meas.py OUT [--runs 100] [--meas-seed 4242]"""
import argparse, hashlib, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "official")]
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import numpy as np, jax
from src import data as D
import npz_to_mat as NM

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--runs", type=int, default=100); ap.add_argument("--meas-seed", type=int, default=4242)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
cfg = D.load_scenario_cfg(seed_top=a.meas_seed)
truth = D.load_fig3_truth()
z, zv, nd = D.make_measurements_runs(jax.random.PRNGKey(cfg["seed_top"]), truth, 3, 0.9, 10.0, a.runs, cfg, 48, on_drop="raise")
z, zv = np.asarray(z), np.asarray(zv)
np.savez(os.path.join(a.out, "meas.npz"), X=np.asarray(truth["X"]), alive=np.asarray(truth["alive"]), z=z, zvalid=zv, meta_M_max=np.array([48]),
         meta_config_id=np.array([3]), meta_lambda_c=np.array([10.0]), meta_pd=np.array([0.9]), meta_seed_top=np.array([a.meas_seed]))
for tp, name in ((0.0, "in_paper_T0.mat"), (1e-4, "in_paper_T1e-4.mat")):
    prm = NM.oracle_params("paper", p_d=0.9, lambda_c=10.0, T_pruning=tp, dump_runs=np.array([-1.0]), murty_log=0.0, dump_pred=0.0)
    NM.write_input_mat(os.path.join(a.out, name), z, zv, prm)
NM.write_input_mat(os.path.join(a.out, "in_williams.mat"), z, zv, NM.oracle_params("paper", p_d=0.9, lambda_c=10.0, dump_runs=np.array([-1.0])))
print("wrote", a.out, "z", z.shape, "n_meas_dropped", int(np.sum(nd)), "sha256 z", hashlib.sha256(z.tobytes()).hexdigest()[:16])

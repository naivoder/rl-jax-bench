"""Reference runs: train + test ONE configuration with the authors' own code (allenai/advisor @ 6849755 on
AllenAct v0.5.0), CPU, in the authors'-code venv. Mirrors `iteratively_run_experiments` (minigrid_and_pd_scripts/
random_hp_search.py) and `iteratively_run_lighthouse_experiments` (lighthouse_scripts/save_pairwise_imitation_data.py)
without their multi-process queue. No official code is copied; it is imported.

    . ${OFFICIAL_ROOT:-external/official}/env.sh      # written by scripts/official/setup.sh
    OMP_NUM_THREADS=1 ${OFFICIAL_ROOT:-external/official}/.venv/bin/python \
        scripts/official/run_reference.py --task PoisonedDoors --exp advisor --draw 7 --out <json>

Deviations from the authors' scripts (also listed in the README):
  * `seed` is passed to OnPolicyRunner (= the hp draw index); their hp search passed `args.seed` = None (unseeded).
  * `--total_steps` may shorten MiniGrid to our preliminary 3e5 proxy (MG_TOTAL_TRAIN_STEPS is a class constant there,
    patched here; the lr/clip decays then end at 3e5 exactly as in our proxy).
  * LH: NUM_TRAIN_SAMPLERS is forced to 20 (their default is `20 if torch.cuda.is_available() else 2`; the paper's
    runs were on GPU, so 20 is the setting that produced the TSVs).
"""
import argparse
import importlib
import inspect
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

torch.set_num_threads(1)
mp = mp.get_context("forkserver")

REPO = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external", "official")), "advisor")

# MG/PD total steps are CLASS constants in the authors' base config. The trainer runs in a forkserver child, which
# re-imports this file as __mp_main__, so the override must be applied at import time from the environment.
if os.environ.get("REF_TOTAL_STEPS"):
    sys.path.insert(0, REPO)
    from projects.advisor.minigrid_and_pd_experiments import base as _B
    _B.MiniGridAndPDExperimentParams.PD_TOTAL_TRAIN_STEPS = int(os.environ["REF_TOTAL_STEPS"])
    _B.MiniGridAndPDExperimentParams.MG_TOTAL_TRAIN_STEPS = int(os.environ["REF_TOTAL_STEPS"])


def hp_draw(i):
    """The authors' fixed random hp draws (random_hp_search.generate_random_lrs_tf_ratio_and_alphas, nsamples=50)."""
    from allenact.utils.misc_utils import rand_float
    np.random.seed(1)
    lr = np.exp(rand_float(math.log(1e-4), math.log(0.5), 50))
    np.random.seed(2)
    tf = rand_float(0.1, 0.9, 50)
    np.random.seed(3)
    al = [np.random.choice([4.0, 8.0, 16.0, 32.0]) for _ in range(50)]
    return float(lr[i]), float(tf[i]), float(al[i])


def config_class(pkg, exp):
    from allenact.base_abstractions.experiment_config import ExperimentConfig
    mod = importlib.import_module(f"{pkg}.{exp}")
    cls = [c for _, c in inspect.getmembers(mod, inspect.isclass)
           if issubclass(c, ExperimentConfig) and c.__module__ == mod.__name__]
    assert len(cls) == 1, cls
    return cls[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)  # PoisonedDoors | AskForHelpLavaCrossingOnce | ... | LightHouse
    ap.add_argument("--exp", required=True)   # advisor | ppo | bc | bc_then_ppo | (LH) advisor_ppo | bc | ppo
    ap.add_argument("--draw", type=int, default=None, help="official hp draw index (MG/PD); also the seed")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--tf", type=float, default=None)
    ap.add_argument("--view", type=int, default=1)
    ap.add_argument("--expert_view", type=int, default=15)
    ap.add_argument("--total_steps", type=int, default=None)
    ap.add_argument("--n_test", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_weights", action="store_true", help="also save the final checkpoint's weights (.npz)")
    a = ap.parse_args()
    sys.path.insert(0, REPO)
    seed = a.seed if a.seed is not None else (a.draw if a.draw is not None else 0)
    work = os.path.splitext(a.out)[0] + "_work"
    os.makedirs(work, exist_ok=True)
    from allenact.algorithms.onpolicy_sync.runner import OnPolicyRunner
    from allenact.main import init_logging
    init_logging("warning")
    t0 = time.time()
    if a.task == "LightHouse":
        from allenact_plugins.lighthouse_plugin.lighthouse_environment import LightHouseEnvironment
        cls = config_class("lighthouse_experiments", a.exp)
        kw = dict(GPU_ID=None, TEST_SEED_OFFSET=0, LR=a.lr if a.lr is not None else 0.0242, VIEW_RADIUS=a.view,
                  EXPERT_VIEW_RADIUS=a.expert_view, NUM_TRAIN_SAMPLERS=20)
        if a.total_steps:
            kw["TOTAL_TRAIN_STEPS"] = a.total_steps
        if a.n_test:
            kw["NUM_TEST_TASKS"] = a.n_test
        cfg = cls(**kw)
        hp = dict(kw)
        optimal = LightHouseEnvironment.optimal_ave_ep_length(world_dim=2, world_radius=15, view_radius=a.view)
    else:
        if a.total_steps:
            assert int(os.environ.get("REF_TOTAL_STEPS", -1)) == a.total_steps, "set REF_TOTAL_STEPS=--total_steps"
        cls = config_class("projects.advisor.minigrid_and_pd_experiments", a.exp)
        hp = {}
        if a.draw is not None:
            lr, tf, al = hp_draw(a.draw)
            hp = {"LR": lr}
            if "advisor" in a.exp:
                hp["FIXED_ALPHA"] = al
            if any(s in a.exp for s in ("then", "dagger")):
                hp["TF_RATIO"] = tf
        if a.lr is not None:
            hp["LR"] = a.lr
        if a.alpha is not None:
            hp["FIXED_ALPHA"] = a.alpha
        if a.tf is not None:
            hp["TF_RATIO"] = a.tf
        kw = dict(task_name=a.task, GPU_ID=None, TEST_SEED_OFFSET=0, **hp)
        if a.n_test:
            kw["NUM_TEST_TASKS"] = a.n_test
        cfg = cls(**kw)
        optimal = cfg.task_info().get("optimal_ave_ep_length")
    runner = OnPolicyRunner(config=cfg, output_dir=work, loaded_config_src_files=None, seed=seed, mode="train",
                            mp_ctx=mp, disable_tensorboard=True, disable_config_saving=True)
    start = runner.start_train(max_sampler_processes_per_worker=1, save_ckpt_after_every_pipeline_stage=False)
    t_train = time.time() - t0
    ckpt_dir = runner.checkpoint_dir(start_time_str=start, create_if_none=False)
    runner.mode = "test"
    res = runner.start_test(checkpoint_path_dir_or_pattern=ckpt_dir, max_sampler_processes_per_worker=1)
    runner.close()
    out = dict(task=a.task, exp=a.exp, draw=a.draw, seed=seed, hp=hp, optimal=optimal, total_steps=a.total_steps,
               wall_train_s=t_train, wall_total_s=time.time() - t0, torch=torch.__version__,
               train_steps=[float(r["training_steps"]) for r in res],
               reward=[float(r["reward"]) for r in res], ep_length=[float(r["ep_length"]) for r in res])
    for k in res[0]:
        if any(s in k for s in ("success", "found_goal", "max_comb_correct", "chose_", "opened_")):
            out[k] = [float(r.get(k, np.nan)) for r in res]
    if a.save_weights:
        import glob as _g
        ck = sorted(_g.glob(os.path.join(ckpt_dir, "*.pt")), key=os.path.getmtime)[-1]
        sd = torch.load(ck, map_location="cpu")["model_state_dict"]
        np.savez(os.path.splitext(a.out)[0] + "_weights.npz", **{k: v.numpy() for k, v in sd.items()})
        out["weights_ckpt"] = os.path.basename(ck)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    shutil.rmtree(work, ignore_errors=True)
    print("[METRIC]", json.dumps({k: out[k] for k in ("task", "exp", "seed", "train_steps", "reward", "ep_length",
                                                      "wall_total_s")}))


if __name__ == "__main__":
    main()

"""ORACLE (official venv): COMPACT parity fixture for tests/test_official_parity.py (committed: tests/fixtures/official/).
Same computation as export_losses.py (official asym-rlpo@1a065b0 critic_loss / actor_losses, then ONE full update:
critic Adam step, then actor Adam step with the UPDATED critic; Adam eps=1e-4), but the weights are NOT the torch init:
online (`pre`) and target (`tgt`) weights are regenerated from names by scripts/official/parity_common.py, so the fixture only has
to carry episodes, losses and gradient/update fingerprints (a few hundred kB instead of 300 MB).

    $OFFICIAL_ROOT/.venv/bin/python -I scripts/official/export_losses_compact.py [--f64]
"""
import json
import os
import sys
import types
import warnings

warnings.filterwarnings("ignore")
PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(PAPER, "external", "official"))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(OFFICIAL_ROOT, "asym-rlpo-aamas2022"))
sys.modules["gym.envs.classic_control.rendering"] = types.ModuleType("rendering")   # headless shim (render only)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import parity_common as pcm  # noqa: E402
from asym_rlpo.algorithms import make_a2c_algorithm  # noqa: E402
from asym_rlpo.envs import make_env  # noqa: E402
from asym_rlpo.sampling import sample_episodes  # noqa: E402
from asym_rlpo.utils.aggregate import average  # noqa: E402
from asym_rlpo.utils.config import get_config  # noqa: E402

get_config()._update(dict(hs_features_dim=0, normalize_hs_features=False))
CASES = {"heavenhell_3": "POMDP-heavenhell_3-episodic-v0", "shopping_5": "POMDP-shopping_5-episodic-v1",
         "car_flag": "extra-car-flag-v0", "cleaner": "extra-cleaner-v0"}
METHODS = {"a2c_asym_hs": ("asym-a2c", False, -1), "a2c_asym_s": ("asym-a2c-state", False, -1),
           "a2c": ("a2c", False, -1), "a2c_react_2": ("a2c", True, 2), "a2c_react_4": ("a2c", True, 4)}
LR_PI, LR_V, EPS, GAMMA, LAM = 1e-3, 3e-4, 1e-4, 0.99, 0.37
F64 = "--f64" in sys.argv
DT = np.float64 if F64 else np.float32
if F64:
    torch.set_default_dtype(torch.float64)


def load(module, tag):
    shapes = {k: tuple(v.shape) for k, v in module.state_dict().items()}
    w = {k: torch.from_numpy(v.astype(DT)) for k, v in pcm.gen_weights(tag, shapes).items()}
    module.load_state_dict(w)
    for k, v in module.state_dict().items():           # aliases resolved consistently by the canonical-name rule
        assert np.array_equal(v.numpy(), w[k].numpy()), k
    return shapes


def official_grad_view(k, g):
    g = g.detach().numpy().astype(np.float64)
    return g[:-1] if "observation_model.embeddings.weight" in k else g   # padding row: frozen, absent on our side


def main():
    out, meta = {}, {}
    for env_name, env_id in CASES.items():
        if F64 and env_name in ("car_flag", "cleaner"):
            continue   # the official OneHotRepresentation hard-codes .float(): no float64 replica possible
        for mname, (algo_name, trunc, n) in METHODS.items():
            torch.manual_seed(7)
            env = make_env(env_id, max_episode_timesteps=100)
            env.seed(7)
            algo = make_a2c_algorithm(algo_name, env, truncated_histories=trunc, truncated_histories_n=n)
            if F64:
                algo.models.double()
                algo.target_models.double()
            shapes = load(algo.models, "pre")
            load(algo.target_models, "tgt")
            pre = {k: v.detach().numpy().astype(np.float64).copy() for k, v in algo.models.state_dict().items()}
            episodes = [e.torch() for e in sample_episodes(env, algo.behavior_policy(), num_episodes=2)]
            if F64:
                for e in episodes:
                    e.rewards = e.rewards.double()
                    if e.states.is_floating_point():
                        e.states, e.observations = e.states.double(), e.observations.double()
            pfx = f"{env_name}/{mname}/"
            meta[f"{env_name}/{mname}"] = {"shapes": {k: list(s) for k, s in shapes.items()}}
            for i, e in enumerate(episodes):
                for key, t in (("states", e.states), ("obs", e.observations), ("actions", e.actions),
                               ("rewards", e.rewards)):
                    out[pfx + f"ep{i}/{key}"] = t.numpy()
            algo.models.zero_grad()
            closs = average([algo.critic_loss(e, discount=GAMMA) for e in episodes])
            closs.backward()
            gc = {k: official_grad_view(k, p.grad) for k, p in algo.models.named_parameters() if p.grad is not None}
            algo.models.zero_grad()
            a_l, n_l = zip(*[algo.actor_losses(e, discount=GAMMA) for e in episodes])
            aloss, nloss = average(a_l), average(n_l)
            (aloss + LAM * nloss).backward()
            ga = {k: official_grad_view(k, p.grad) for k, p in algo.models.named_parameters() if p.grad is not None}
            out[pfx + "critic_loss"] = np.float64(closs.item())
            out[pfx + "actor_loss"] = np.float64(aloss.item())
            out[pfx + "negentropy_loss"] = np.float64(nloss.item())
            algo.models.zero_grad()
            oa = torch.optim.Adam(algo.models.parameters(), lr=LR_PI, eps=EPS)
            oc = torch.optim.Adam(algo.models.parameters(), lr=LR_V, eps=EPS)
            oc.zero_grad()
            average([algo.critic_loss(e, discount=GAMMA) for e in episodes]).backward()
            oc.step()
            oa.zero_grad()
            a_l, n_l = zip(*[algo.actor_losses(e, discount=GAMMA) for e in episodes])
            (average(a_l) + LAM * average(n_l)).backward()
            oa.step()
            post = {k: v.detach().numpy().astype(np.float64) for k, v in algo.models.state_dict().items()}
            dpost = {}
            for k in post:   # the update of every parameter, in units of its learning rate
                if pcm.canonical(k) != k:
                    continue  # alias: same tensor as its canonical name
                lr = LR_PI if k.startswith("agent.") else LR_V
                d = (post[k] - pre[k]) / lr
                dpost[k] = d[:-1] if "observation_model.embeddings.weight" in k else d
            for tag, dd in (("gc", gc), ("ga", ga), ("dpost", dpost)):
                for k, v in dd.items():
                    for fk, fv in pcm.fingerprint(k, v).items():
                        out[pfx + f"{tag}/{k}/{fk}"] = fv
            print(env_name, mname, "lens", [len(e) for e in episodes], "closs", closs.item(), "aloss", aloss.item(),
                  "negent", nloss.item(), flush=True)
    out["meta"] = np.array(json.dumps(dict(cases=meta, torch=torch.__version__, commit="1a065b0", lr_pi=LR_PI, lr_v=LR_V,
                                           eps=EPS, gamma=GAMMA, lam=LAM, f64=F64)))
    np.savez_compressed(os.path.join(PAPER, "tests", "fixtures", "official",
                                     "losses_compact_f64.npz" if F64 else "losses_compact.npz"), **out)


if __name__ == "__main__":
    main()

"""Run shkrwnd/Deep-Reinforcement-Learning-for-Dynamic-Spectrum-Access train.py (THIRD-PARTY second-best reference, NOT the authors' code)
unchanged, under TensorFlow 2.15 through a tf.compat.v1 shim, and record per-slot actions / rewards.

Usage (TF venv, built by scripts/official/setup.sh; NOT the repo JAX venv):
  $OFFICIAL_ROOT/.venv/bin/python scripts/official/run_shk.py --seed 0 --slots 100000 --out runs/official/shk_N3K2_s0 [--users 3 --channels 2]

What is and is not modified (all labelled; their repository files are never edited):
  * `tensorflow` is replaced by tf.compat.v1 + a `contrib` namespace implementing the three contrib symbols they use
    (rnn.BasicLSTMCell -> tf.compat.v1.nn.rnn_cell.BasicLSTMCell, same cell; layers.layer_norm -> a re-implementation of tf.contrib.layers.layer_norm:
    normalise over axes 1.., epsilon 1e-12, learned beta=0 / gamma=1; tf.contrib.rnn is otherwise unused).
  * the module constants TIME_SLOTS / NUM_USERS / NUM_CHANNELS are overridden by CLI flags; their defaults are 100000 / 3 / 2.
  * `print` is silenced (their loop prints several lines per slot), matplotlib uses Agg (their plotting block runs, writes nothing we keep).
  * env_network.step is wrapped to LOG (action, reward) per slot; the wrapped call is their code, return value untouched.
  * seeds: python random, numpy, and the TF graph seed (set right after their tf.reset_default_graph).
"""
import argparse, json, os, random, sys, time, types
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--slots", type=int, default=100000)
ap.add_argument("--users", type=int, default=3)
ap.add_argument("--channels", type=int, default=2)
ap.add_argument("--out", required=True)
ap.add_argument("--no_leak", action="store_true", help="ABLATION: replace obs[-1] (their global per-channel outcome fed as the capacity input) by constant 1 = the paper v1/v2 capacity")
ap.add_argument("--own_reward", action="store_true", help="ABLATION: drop their cooperative reward-sum : each user gets its own 0/1 success reward (competitive Eq. 13)")
ap.add_argument("--sample_beta", type=float, default=None, help="ABLATION : act by SAMPLING softmax(beta Q) (Eq. 11, alpha 0; max-shifted for numerical stability: their raw exp overflows at beta 20) at this fixed beta instead of argmax, and stop their beta decrement")
ap.add_argument("--paper_scalars", action="store_true", help="also set gamma=0.95 (paper SVI) and hidden_size=100 (paper LSTM 100); structure stays theirs")
ap.add_argument("--repo", default=os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external", "official")), "shkrwnd"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
os.chdir(args.out)
os.makedirs("checkpoints", exist_ok=True)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")

import tensorflow as _tf
v1 = _tf.compat.v1
_tf.config.threading.set_intra_op_parallelism_threads(1); _tf.config.threading.set_inter_op_parallelism_threads(1)
v1.disable_v2_behavior()
import builtins


def layer_norm(x, center=True, scale=True, epsilon=1e-12):      # tf.contrib.layers.layer_norm defaults (begin_norm_axis=1)
    with v1.variable_scope("LayerNorm"):
        d = x.shape[-1]
        beta = v1.get_variable("beta", [d], initializer=v1.zeros_initializer())
        gamma = v1.get_variable("gamma", [d], initializer=v1.ones_initializer())
        mean, var = v1.nn.moments(x, [1], keep_dims=True)
        return (x - mean) * v1.math.rsqrt(var + epsilon) * gamma + beta


class _Shim(types.ModuleType):
    def __getattr__(self, name):
        if name == "contrib":
            return types.SimpleNamespace(rnn=types.SimpleNamespace(BasicLSTMCell=v1.nn.rnn_cell.BasicLSTMCell),
                                         layers=types.SimpleNamespace(layer_norm=layer_norm))
        if name == "reset_default_graph":
            def reset():
                v1.reset_default_graph(); v1.set_random_seed(args.seed)
            return reset
        return getattr(v1, name)


sys.modules["tensorflow"] = _Shim("tensorflow")
sys.path.insert(0, args.repo)
import multi_user_network_env as menv

LOG = {"a": [], "r": []}
_orig_step = menv.env_network.step


def _logged_step(self, action):
    a = np.array(action, copy=True)
    obs = _orig_step(self, action)
    if args.no_leak:
        obs = list(obs[:-1]) + [np.ones(self.NUM_CHANNELS)]
    LOG["a"].append(a.astype(np.int8)); LOG["r"].append(np.array([o[1] for o in obs[:self.NUM_USERS]], np.int8))
    return obs


menv.env_network.step = _logged_step
random.seed(args.seed); np.random.seed(args.seed)

src = open(os.path.join(args.repo, "train.py")).read()
for old, new in (("TIME_SLOTS = 100000", f"TIME_SLOTS = {args.slots}"), ("NUM_CHANNELS = 2 ", f"NUM_CHANNELS = {args.channels} "),
                 ("NUM_USERS = 3 ", f"NUM_USERS = {args.users} ")):
    assert old in src, old
    src = src.replace(old, new, 1)
if args.paper_scalars:
    for old, new in (("gamma = 0.9 ", "gamma = 0.95 "), ("hidden_size = 128 ", "hidden_size = 100 ")):
        assert old in src, old
        src = src.replace(old, new, 1)
if args.sample_beta is not None:
    for old, new in (("action[each_user] = np.argmax(prob,axis=1)", "action[each_user] = np.random.choice(NUM_CHANNELS + 1, p=(lambda z: z / z.sum)(np.exp(beta * (Qs[0] - Qs[0].max))))"),
                     ("beta = 1                                #Annealing", f"beta = {args.sample_beta}                                #Annealing"),
                     ("beta -=0.001", "beta -=0.0")):
        assert old in src, old
        src = src.replace(old, new, 1)
if args.own_reward:
    old = "if reward[i] > 0:\n            reward[i] = sum_r"
    assert old in src
    src = src.replace(old, "if reward[i] > 0:\n            reward[i] = reward[i]", 1)
g = {"__name__": "shk_train", "print": lambda *a, **k: None}
t0 = time.time()
try:
    exec(compile(src, "train.py", "exec"), g)
finally:
    a = np.array(LOG["a"]); r = np.array(LOG["r"])
    np.savez_compressed("trace.npz", actions=a, rewards=r)
    K = args.channels
    n_pre = len(a) - args.slots            # slots logged before TIME_SLOTS (their warm-up: 1 initial + pretrain_length*step_size*5 random slots)
    a, r = a[n_pre:], r[n_pre:]
    np.savez_compressed("trace.npz", actions=a, rewards=r, n_pre=n_pre)
    s = r.sum(1)
    json.dump(dict(seed=args.seed, users=args.users, channels=K, n_pre=n_pre, slots_logged=len(a), wall_s=time.time() - t0,
                   util_first1k=float(s[:1000].mean() / K), util_last1k=float(s[-1000:].mean() / K),
                   util_by_10k=[float(s[i:i + 10000].mean() / K) for i in range(0, len(s), 10000)]), open("summary.json", "w"), indent=1)

"""Record the parity fixtures from the THIRD-PARTY shkrwnd implementation (the authors released no code; see README).
Run in the TF venv:  $OFFICIAL_ROOT/.venv/bin/python scripts/official/make_fixtures.py
Their code is executed, not copied: the env and QNetwork modules are imported from $OFFICIAL_ROOT/shkrwnd (default external/official), the
helper functions of train.py (a script) are pulled out by AST (one_hot, state_generator) and exec'd; the three inline training expressions
(Boltzmann probabilities train.py:278-281, TD target train.py:382, MSE loss drqn.py:57) are re-evaluated in numpy with the same formulas
and the line numbers are recorded below.  Output: tests/fixtures/official/*.npz (small; committed).  Their code is NOT committed."""
import ast, os, random, sys, types
import numpy as np

REPO = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external", "official")), "shkrwnd")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "fixtures", "official")
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as _tf
v1 = _tf.compat.v1
v1.disable_v2_behavior()


def layer_norm(x, epsilon=1e-12):      # tf.contrib.layers.layer_norm defaults (center, scale, begin_norm_axis=1), re-implemented (MY code, not theirs)
    with v1.variable_scope("LayerNorm"):
        d = x.shape[-1]
        beta = v1.get_variable("beta", [d], initializer=v1.zeros_initializer())
        gamma = v1.get_variable("gamma", [d], initializer=v1.ones_initializer())
        mean, var = v1.nn.moments(x, [1], keep_dims=True)
        return (x - mean) * v1.math.rsqrt(var + epsilon) * gamma + beta


class _Shim(types.ModuleType):
    def __getattr__(self, n):
        if n == "contrib":
            return types.SimpleNamespace(rnn=types.SimpleNamespace(BasicLSTMCell=v1.nn.rnn_cell.BasicLSTMCell),
                                         layers=types.SimpleNamespace(layer_norm=layer_norm))
        return getattr(v1, n)


sys.modules["tensorflow"] = _Shim("tensorflow")
sys.path.insert(0, REPO)
import multi_user_network_env as menv
import drqn as their_drqn

# ---------------------------------------------------------------- env semantics at ATTEMPT_PROB = 1
rng = np.random.default_rng(7)
for (N, K) in ((2, 2), (3, 2), (4, 2), (5, 3), (10, 4)):
    env = menv.env_network(N, K, 1.0)
    A, R, CAP = [], [], []
    for _ in range(2000):
        a = rng.integers(0, K + 1, N).astype(np.int32)
        obs = env.step(a.copy())
        A.append(a); R.append([o[1] for o in obs[:N]]); CAP.append(np.asarray(obs[-1]))
    np.savez_compressed(f"{OUT}/env_N{N}K{K}.npz", actions=np.array(A, np.int8), rewards=np.array(R, np.int8), cap=np.array(CAP, np.int8))

# ---------------------------------------------------------------- ATTEMPT_PROB < 1 (their stale-action behaviour)
N, K, p = 4, 2, 0.6
random.seed(11); env = menv.env_network(N, K, p)
A, R, UA = [], [], []
rng = np.random.default_rng(3)
for _ in range(400):
    a = rng.integers(0, K + 1, N).astype(np.int32)
    obs = env.step(a.copy())
    A.append(a); R.append([o[1] for o in obs[:N]]); UA.append(env.users_action.copy())
np.savez_compressed(f"{OUT}/env_attempt06.npz", actions=np.array(A, np.int8), rewards=np.array(R, np.int8), users_action=np.array(UA, np.int8), seed=11, N=N, K=K, p=p)

# ---------------------------------------------------------------- their QNetwork: forward, loss, gradients (hidden 8)
H, T, S, A_ = 8, 5, 6, 3
v1.reset_default_graph(); v1.set_random_seed(0)
net = their_drqn.QNetwork(name="main", hidden_size=H, learning_rate=1e-4, step_size=T, state_size=S, action_size=A_)
rs = np.random.RandomState(5)
B = 4
x = rs.randint(0, 2, (B, T, S)).astype(np.float32)
acts = rs.randint(0, A_, B).astype(np.int32)
targets = rs.randn(B).astype(np.float32) * 2
vars_ = v1.trainable_variables()
grads = v1.gradients(net.loss, vars_)
with v1.Session() as sess:
    sess.run(v1.global_variables_initializer())
    fd = {net.inputs_: x, net.actions_: acts, net.targetQs_: targets}
    lstm_out, q, loss, vals, gv = sess.run([net.lstm_out, net.output, net.loss, vars_, grads], feed_dict=fd)
d = dict(x=x, acts=acts, targets=targets, lstm_out=lstm_out, q=q, loss=np.float32(loss), names=np.array([v.name for v in vars_]))
for v, val, g in zip(vars_, vals, gv):
    key = v.name.replace("/", "__").replace(":0", "")
    d["w__" + key] = val; d["g__" + key] = g
np.savez_compressed(f"{OUT}/their_qnetwork_H8.npz", **d)

# ---------------------------------------------------------------- state_generator (AST-extracted from train.py)
src = open(os.path.join(REPO, "train.py")).read()
tree = ast.parse(src)
keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("one_hot", "state_generator")]
ns = {"np": np, "sys": sys, "NUM_CHANNELS": 2}
exec(compile(ast.Module(keep, []), "train.py", "exec"), ns)
env = menv.env_network(3, 2, 1.0)
SG_A, SG_X = [], []
rng = np.random.default_rng(1)
for _ in range(300):
    a = rng.integers(0, 3, 3).astype(np.int32)
    obs = env.step(a.copy())
    SG_A.append(a); SG_X.append(np.array(ns["state_generator"](a, obs)))
np.savez_compressed(f"{OUT}/state_generator_N3K2.npz", actions=np.array(SG_A, np.int8), x=np.array(SG_X, np.float32))

# ---------------------------------------------------------------- Boltzmann probabilities (train.py:278-281) and TD target (train.py:382), numpy
rs = np.random.RandomState(9)
Q = rs.randn(64, 3).astype(np.float64) * 3
rows = []
for alpha, beta in ((0.0, 1.0), (0.0, 20.0), (0.05, 1.0), (0.05, 7.0), (0.02, 20.0), (0.5, 3.0)):
    prob1 = (1 - alpha) * np.exp(beta * Q)                                   # train.py:215
    prob = prob1 / np.sum(np.exp(beta * Q), axis=1, keepdims=True) + alpha / 3   # train.py:216 (NUM_CHANNELS+1 = 3; their code sums over the whole (1,A) array)
    rows.append((alpha, beta, prob))
np.savez_compressed(f"{OUT}/their_policy_probs.npz", Q=Q, alpha=np.array([r[0] for r in rows]), beta=np.array([r[1] for r in rows]), prob=np.array([r[2] for r in rows]))
qn = rs.randn(64, 3) * 2; r = rs.randint(0, 2, 64).astype(np.float64); gamma = 0.9
np.savez_compressed(f"{OUT}/their_td_target.npz", q_next=qn, r=r, gamma=gamma, target=r + gamma * np.max(qn, axis=1))   # train.py:382
print("fixtures written to", os.path.abspath(OUT)); print(sorted(os.listdir(OUT)))

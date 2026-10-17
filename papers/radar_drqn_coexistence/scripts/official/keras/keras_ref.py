"""Keras (TF 2.2) reference learner for Thornton et al. 2020 DQN / DDRQN, written independently of src/.
Consumes a stream npz from dump_streams.py. Table I: FC 256-128-84 ReLU, [LSTM 84], linear 15, Adam lr 1e-3 (Keras
eps 1e-7), batch 32 (transitions | 10-step traces), buffer 2000 transitions (= 200 traces), target hard copy every
250 updates, gamma 0.9, uniform-random actions in training, greedy + online updates in evaluation.
DDRQN reading zero-start aligned 10-step traces, act-time LSTM reset every 10 pulses, the last next-Q of a trace
computed from a zero state (the actor resets there).  --init ours|keras."""
import os, sys, argparse, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--stream", required=True)
ap.add_argument("--arm", choices=["dqn", "ddrqn"], required=True)
ap.add_argument("--init", choices=["ours", "keras"], default="ours")
ap.add_argument("--tf-seed", type=int, default=0)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--adam", choices=["keras", "optax"], default="keras", help="optax = eps added to sqrt(v_hat) (Kingma-Ba / optax form), a labelled harness patch")
ap.add_argument("--sgd", action="store_true", help="recheck, plain SGD: tf.keras.optimizers.SGD(lr), momentum 0 (Eq. 9); overrides --adam")
a = ap.parse_args()
tf.config.threading.set_intra_op_parallelism_threads(a.threads)
tf.config.threading.set_inter_op_parallelism_threads(a.threads)
tf.random.set_seed(a.tf_seed)
if os.environ.get('KREF_FLOAT64'):
    tf.keras.backend.set_floatx('float64')

S = np.load(a.stream)
OBS, OBS2, TH2, ARAND, R = S["obs"], S["obs2"], S["th2"], S["a_rand"], S["R"]
NTR, NTOT = int(S["ntr"]), int(S["ntot"])
IDX = S[f"idx_{a.arm}"]
REC = a.arm == "ddrqn"
L = 10 if REC else 1
NBUF = 2000 // L
B, GAMMA, TGT, NA, D = 32, 0.9, 250, 15, 7

def build():
    x = tf.keras.Input(shape=(None, D))
    h = x
    for i, n in enumerate((256, 128, 84)):
        h = tf.keras.layers.Dense(n, activation="relu", name=f"fc{i}")(h)
    if REC:
        h0 = tf.keras.Input(shape=(84,)); c0 = tf.keras.Input(shape=(84,))
        h, hT, cT = tf.keras.layers.LSTM(84, return_sequences=True, return_state=True, name="lstm")(h, initial_state=[h0, c0])
        q = tf.keras.layers.Dense(NA, name="q")(h)
        return tf.keras.Model([x, h0, c0], [q, hT, cT])
    q = tf.keras.layers.Dense(NA, name="q")(h)
    return tf.keras.Model(x, q)

online, target = build(), build()
if a.init == "ours":
    g = lambda k: S[f"init_{a.arm}__params/{k}"]
    for i in range(3):
        online.get_layer(f"fc{i}").set_weights([g(f"fc{i}/kernel"), g(f"fc{i}/bias")])
    online.get_layer("q").set_weights([g("q/kernel"), g("q/bias")])
    if REC:  # Keras LSTM: kernel [in,4u], recurrent [u,4u], bias [4u], gates (i,f,c,o) == ours
        online.get_layer("lstm").set_weights([g("lstm/kernel/kernel"), g("lstm/recurrent/kernel"), g("lstm/kernel/bias")])
target.set_weights(online.get_weights())
opt = tf.keras.optimizers.SGD(learning_rate=a.lr) if a.sgd else tf.keras.optimizers.Adam(learning_rate=a.lr)
step = tf.Variable(0.0, dtype=tf.keras.backend.floatx())
M = [tf.Variable(tf.zeros_like(w)) for w in online.trainable_variables]
V = [tf.Variable(tf.zeros_like(w)) for w in online.trainable_variables]   # TF 2.2 defaults: beta 0.9/0.999, epsilon 1e-7

def qfun(model, x, z=None):
    if REC:
        q, _, _ = model([x, z, z])
        return q
    return model(x)

@tf.function
def train_step(obs, act, rew):
    """obs [B, L+1, D], act [B, L] int, rew [B, L]."""
    z = tf.zeros((B, 84), tf.keras.backend.floatx())
    x_cur, x_last = obs[:, :L], obs[:, L:]
    # next-Q: positions 1..L-1 continue the trace; position L is consumed after an act-time reset (zero state)
    if REC:
        qn_on = tf.concat([qfun(online, x_cur, z)[:, 1:], qfun(online, x_last, z)], 1)
        qn_tg = tf.concat([qfun(target, x_cur, z)[:, 1:], qfun(target, x_last, z)], 1)
    else:
        qn_on, qn_tg = qfun(online, x_last), qfun(target, x_last)
    a_star = tf.argmax(qn_on, -1, output_type=tf.int32)
    v = tf.reduce_sum(qn_tg * tf.one_hot(a_star, NA, dtype=qn_tg.dtype), -1) if REC else tf.reduce_max(qn_tg, -1)
    y = tf.stop_gradient(rew + GAMMA * v)
    with tf.GradientTape() as tape:
        q = qfun(online, x_cur, z)
        qsa = tf.reduce_sum(q * tf.one_hot(act, NA, dtype=q.dtype), -1)
        loss = tf.reduce_mean(tf.square(qsa - y))
    grads = tape.gradient(loss, online.trainable_variables)
    if a.sgd or a.adam == "keras":
        opt.apply_gradients(zip(grads, online.trainable_variables))
    else:
        step.assign_add(1.0)
        for var, gr, m, v in zip(online.trainable_variables, grads, M, V):
            m.assign(0.9 * m + 0.1 * gr); v.assign(0.999 * v + 0.001 * gr * gr)
            mh = m / (1 - 0.9 ** step); vh = v / (1 - 0.999 ** step)
            var.assign_sub(a.lr * mh / (tf.sqrt(vh) + 1e-7))
    return loss

@tf.function
def act_step(o, h, c):
    if REC:
        q, h2, c2 = online([o[None, None], h, c])
        return q[0, 0], h2, c2
    return online(o[None, None])[0, 0], h, c

# own FIFO trace buffer + writer
bo = np.zeros((NBUF, L + 1, D), np.float32); ba = np.zeros((NBUF, L), np.int32); br = np.zeros((NBUF, L), np.float32)
wo = np.zeros((L + 1, D), np.float32); wa = np.zeros(L, np.int32); wr = np.zeros(L, np.float32)
k, pushed, n_upd = 0, 0, 0
h = tf.zeros((1, 84), tf.keras.backend.floatx()); c = tf.zeros((1, 84), tf.keras.backend.floatx())
acts = np.zeros(NTOT, np.int32); rews = np.zeros(NTOT, np.float32); losses = np.full(NTOT, np.nan, np.float32)
t0 = time.time()
for t in range(NTOT):
    o = OBS[t]
    if REC and t % L == 0:
        h = tf.zeros((1, 84), tf.keras.backend.floatx()); c = tf.zeros((1, 84), tf.keras.backend.floatx())
    q, h, c = act_step(tf.constant(o, tf.keras.backend.floatx()), h, c)
    act = int(ARAND[t]) if t < NTR else int(np.argmax(q.numpy()))   # np.argmax: first max, as ours
    r = float(R[act, TH2[t]])
    acts[t], rews[t] = act, r
    wo[k], wa[k], wr[k] = o, act, r
    if k == L - 1:
        wo[L] = OBS2[t]
        s = pushed % NBUF
        bo[s], ba[s], br[s] = wo, wa, wr
        pushed += 1
        k = 0
    else:
        k += 1
    count = min(pushed, NBUF)
    if count >= B:
        ix = IDX[t]
        assert ix.max() < count
        fx = tf.keras.backend.floatx()
        losses[t] = float(train_step(tf.constant(bo[ix], fx), tf.constant(ba[ix]), tf.constant(br[ix], fx)))
        n_upd += 1
        if n_upd % TGT == 0:
            target.set_weights(online.get_weights())
    if t % 20000 == 0:
        print(f"t={t} {time.time()-t0:.0f}s", flush=True)

fin = {f"final__{w.name}": w.numpy() for w in online.weights}
np.savez_compressed(a.out, acts=acts, rews=rews, losses=losses, **fin)
nev = (NTOT - NTR) // 1000
rc = rews[NTR:].reshape(nev, 1000).mean(1)
res = dict(arm=a.arm, adam=a.adam, init=a.init, lr=a.lr, J=float(rc.mean()), J_conv=float(rc[10:].mean()) if nev >= 70 else float(rc.mean()),
           sigma_within=float(rc[10:].std(ddof=1)) if nev >= 70 else None,
           adapt_pct=float(100 * np.mean(acts[NTR + 1:] != acts[NTR:-1])), wall_s=time.time() - t0, tf=tf.__version__)
json.dump(res, open(a.out.replace(".npz", ".json"), "w"), indent=1)
print(res)

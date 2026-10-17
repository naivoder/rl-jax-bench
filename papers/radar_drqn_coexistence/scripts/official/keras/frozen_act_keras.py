import os, sys, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
S = np.load("diffA_seed0.npz"); O = np.load("frozen_ours.npz"); ntr, ntot = int(S["ntr"]), int(S["ntot"])
res = {}
for arm in ("dqn", "ddrqn"):
    REC = arm == "ddrqn"
    g = lambda k: S[f"final_{arm}__params/{k}"]
    x = tf.keras.Input((None, 7)); h = x
    for i, n in enumerate((256, 128, 84)):
        h = tf.keras.layers.Dense(n, activation="relu", name=f"fc{i}")(h)
    if REC:
        h0 = tf.keras.Input((84,)); c0 = tf.keras.Input((84,))
        h, hT, cT = tf.keras.layers.LSTM(84, return_sequences=True, return_state=True, name="lstm")(h, initial_state=[h0, c0])
        m = tf.keras.Model([x, h0, c0], [tf.keras.layers.Dense(15, name="q")(h), hT, cT])
        m.get_layer("lstm").set_weights([g("lstm/kernel/kernel"), g("lstm/recurrent/kernel"), g("lstm/kernel/bias")])
    else:
        m = tf.keras.Model(x, tf.keras.layers.Dense(15, name="q")(h))
    for i in range(3):
        m.get_layer(f"fc{i}").set_weights([g(f"fc{i}/kernel"), g(f"fc{i}/bias")])
    m.get_layer("q").set_weights([g("q/kernel"), g("q/bias")])
    hs = cs = np.zeros((1, 84), np.float32); acts = []; gaps = []
    for t in range(ntr, ntot):
        o = S["obs"][t][None, None].astype(np.float32)
        if REC:
            if t % 10 == 0: hs = cs = np.zeros((1, 84), np.float32)
            q, hs, cs = m.predict_on_batch([o, hs, cs]); q = np.asarray(q)[0, 0]; hs, cs = np.asarray(hs), np.asarray(cs)
        else:
            q = np.asarray(m.predict_on_batch(o))[0, 0]
        acts.append(int(np.argmax(q))); s = np.sort(q); gaps.append(float(s[-1] - s[-2]))
    acts = np.array(acts); ag = acts == O[f"{arm}_act"]
    res[arm] = dict(agreement=float(ag.mean()), n=int(len(acts)), min_top2_gap_at_disagreement=float(np.min(np.array(gaps)[~ag])) if (~ag).any() else None,
                    max_top2_gap_at_disagreement=float(np.max(np.array(gaps)[~ag])) if (~ag).any() else None,
                    rewards_match=bool(np.allclose(S["R"][acts, S["th2"][ntr:]], O[f"{arm}_rew"]) if ag.all() else None), ours_learned_steps=float(O[f"{arm}_learned"]))
print(json.dumps(res, indent=1)); json.dump(res, open("frozen_act_compare.json", "w"), indent=1)

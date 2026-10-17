import os, json, inspect, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, tensorflow as tf
S = np.load(sys.argv[1])
lstm = tf.keras.layers.LSTM(84); d = tf.keras.layers.Dense(15); adam = tf.keras.optimizers.Adam()
out = dict(tf=tf.__version__,
  lstm=dict(activation=lstm.activation.__name__, recurrent_activation=lstm.recurrent_activation.__name__,
            unit_forget_bias=lstm.cell.unit_forget_bias, kernel_initializer=type(lstm.cell.kernel_initializer).__name__,
            recurrent_initializer=type(lstm.cell.recurrent_initializer).__name__, bias_initializer=type(lstm.cell.bias_initializer).__name__,
            implementation=lstm.cell.implementation),
  dense=dict(kernel_initializer=type(d.kernel_initializer).__name__, bias_initializer=type(d.bias_initializer).__name__),
  adam=dict(lr=float(adam.learning_rate.numpy()) if hasattr(adam.learning_rate,'numpy') else adam.learning_rate, beta_1=float(adam.beta_1.numpy()) if hasattr(adam.beta_1,'numpy') else adam.beta_1,
            beta_2=float(adam.beta_2.numpy()) if hasattr(adam.beta_2,'numpy') else adam.beta_2, epsilon=adam.epsilon, amsgrad=adam.amsgrad),
  hard_sigmoid=[float(v) for v in tf.keras.activations.hard_sigmoid(tf.constant([-3., -1., 0., 1., 3.])).numpy()])
src = inspect.getsource(type(adam)._resource_apply_dense) + inspect.getsource(type(adam)._prepare_local)
out["adam_src_lr_t"] = [l.strip() for l in src.splitlines() if "lr_t" in l or "epsilon" in l][:8]
# init statistics: Keras own init vs ours on the same shapes
x = tf.keras.Input((None, 84)); m = tf.keras.Model(x, tf.keras.layers.LSTM(84)(x))
K, U, b = m.get_weights()
ours = {k.split("__")[1]: S[k] for k in S.files if k.startswith("init_ddrqn__")}
oK, oU, ob = ours["params/lstm/kernel/kernel"], ours["params/lstm/recurrent/kernel"], ours["params/lstm/kernel/bias"]
def st(W): return dict(shape=list(W.shape), std=float(W.std()), maxabs=float(np.abs(W).max()))
out["init_compare"] = dict(keras_kernel=st(K), ours_kernel=st(oK), keras_rec=st(U), ours_rec=st(oU),
    keras_rec_rows_orthonormal=float(np.abs(U @ U.T - np.eye(84)).max()), ours_rec_rows_orthonormal=float(np.abs(oU @ oU.T - np.eye(84)).max()),
    keras_bias_forget_slice=[float(b[84:168].min()), float(b[84:168].max())], ours_bias_forget_slice=[float(ob[84:168].min()), float(ob[84:168].max())],
    keras_bias_other_maxabs=float(np.abs(np.r_[b[:84], b[168:]]).max()), ours_bias_other_maxabs=float(np.abs(np.r_[ob[:84], ob[168:]]).max()),
    glorot_limit_expected=float(np.sqrt(6 / (84 + 336))))
fc0 = tf.keras.layers.Dense(256); fc0.build((None, 7))
out["init_compare"]["keras_fc0"] = st(fc0.kernel.numpy()); out["init_compare"]["ours_fc0"] = st(ours["params/fc0/kernel"])
print(json.dumps(out, indent=1)); json.dump(out, open("keras_defaults.json", "w"), indent=1)

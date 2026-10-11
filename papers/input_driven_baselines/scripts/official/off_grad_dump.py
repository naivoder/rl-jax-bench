"""OFFICIAL side: build the authors' ActorAgent / CriticAgent (unmodified classes, TF compat.v1 shim), fix params,
feed a fixed batch (obs, actions, adv, returns, entropy weight) and dump losses + gradients.
Usage: off_grad_dump.py <batch.npz> <out.npz> <hid1> <hid2>"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
inp, out, h1, h2 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
import tensorflow.python
sys.modules['tensorflow'] = tf1
OFF = os.path.join(os.environ.get('OFFICIAL_ROOT', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'official')), 'input_driven_rl_example')  # the authors' clone
sys.path.insert(0, OFF)
sys.argv = ['x', '--num_workers', '10', '--hid_dims', h1, h2]
import numpy as np
from load_balance_actor_agent import ActorAgent
from critic_agent import CriticAgent
b = np.load(inp)
sess = tf1.Session()
actor = ActorAgent(sess); critic = CriticAgent(sess, input_dim=12)
sess.run(tf1.global_variables_initializer())
actor.set_params([b['a_W0'], b['a_b0'], b['a_W1'], b['a_b1'], b['a_W2'], b['a_b2']])
critic.set_params([b['c_W0'], b['c_b0'], b['c_W1'], b['c_b1'], b['c_W2'], b['c_b2']])
obs = b['obs']; act_vec = np.eye(10)[b['actions']]
ga, loss = actor.compute_gradients([obs], [act_vec], b['adv'][:, None], float(b['ent_w']))
gc, closs = critic.compute_gradients([b['cobs']], [b['ret'][:, None]])
probs = sess.run(actor.act_probs, {actor.inputs: obs}); vals = critic.predict(b['cobs'])
np.savez(out, **{f'ga{i}': g for i, g in enumerate(ga)}, **{f'gc{i}': g for i, g in enumerate(gc)},
         adv_loss=loss[0], ent_loss=loss[1], closs=closs, probs=probs, vals=vals)
print('ok', loss[:2], closs)

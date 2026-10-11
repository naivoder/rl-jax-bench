"""Multi-critic update differential, AUTHORS' side.  Unmodified CriticAgent class (TF compat.v1 shim), 10 critics in scopes
critic_agent_i exactly as load_balance_actor_multi_critic_train.py:172-176, set to OUR pre-update stack, then one
iteration of the trainer's critic logic: every agent's gradient at the PRE-iteration params of its net
(worker:43,:131-132), then critic_agents[model_indices[i]].apply_gradients(g_i, lr) in agent order (:298-308).
    $OFFICIAL_ROOT/.venv/bin/python scripts/official/critic_update_official.py <batch.npz>"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
inp = sys.argv[1]
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
import tensorflow.python  # noqa
sys.modules['tensorflow'] = tf1
OFF = os.path.join(os.environ.get('OFFICIAL_ROOT', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'official')), 'input_driven_rl_example')  # the authors' clone
sys.path.insert(0, OFF)
sys.argv = ['x', '--num_workers', '10', '--service_rates'] + [str(r) for r in
            (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05)]
import numpy as np
from param import args
from critic_agent import CriticAgent
b = np.load(inp)
N = 10
sess = tf1.Session()
critics = [CriticAgent(sess, input_dim=args.num_workers + 2, scope='critic_agent_' + str(i)) for i in range(N)]
sess.run(tf1.global_variables_initializer())
for n in range(N):
    critics[n].set_params([b[f'pre_{n}_{j}'] for j in range(6)])
idx = b['idx']
out = {}
grads = []
for i, m in enumerate(idx):  # worker side: gradient at the pre-iteration params of net m
    g, loss = critics[m].compute_gradients([b['x'][i]], [b['ret'][i][:, None]])
    grads.append(g)
    out[f'loss_{i}'] = loss
    for j, a in enumerate(g):
        out[f'grad_{i}_{j}'] = a
for i, m in enumerate(idx):  # master side: one Adam step per agent on its net, agent order
    critics[m].apply_gradients(grads[i], args.lr_rate)
for n in range(N):
    for j, a in enumerate(critics[n].get_params()):
        out[f'post_{n}_{j}'] = a
np.savez(inp.replace('.npz', '_official.npz'), **out)
print('ok lr', args.lr_rate, 'losses', [float(out[f'loss_{i}']) for i in range(3)])

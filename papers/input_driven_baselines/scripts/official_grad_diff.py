"""OUR side of the learner differential test.  mode 'make': build a batch from a real rollout of OUR env with a random
actor/critic (leaky_relu 0.2, hidden h1/h2 so the official class can hold the same params), save params+batch.
mode 'diff': evaluate OUR algorithm.actor_loss / value_loss on that batch and compare with the official dump.
Ours uses MEAN reductions, the official SUMs: expected exact ratio = n_steps.  Tolerance: rel 1e-4 (float32)."""
import sys, os, json
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
import numpy as np, jax, jax.numpy as jnp
from src import algorithm as alg
from src.model import init_actor, init_critic
from src.envs import make_lb_spec, lb_rollout
from src.data import sample_lb_inputs, lb_input_kwargs
from src.train import load_config
mode, path, h1, h2 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
cfg = load_config(os.path.join(REPO, 'src/configs/load_balance.yaml'))
spec = make_lb_spec(cfg)
am, ap = init_actor(jax.random.PRNGKey(1), (h1, h2), 10, 11, 'leaky_relu')
cm, cp = init_critic(jax.random.PRNGKey(2), (h1, h2), 12, 'leaky_relu')
def flat(p):
    d = p['params']['MLP_0']; return [np.asarray(d[f'Dense_{i}'][k]) for i in range(3) for k in ('kernel', 'bias')]
if mode == 'make':
    # perturb biases so they are non-zero (zero-init would hide bias bugs)
    ap = jax.tree_util.tree_map(lambda x: x + 0.05 * jax.random.normal(jax.random.PRNGKey(3), x.shape), ap)
    cp = jax.tree_util.tree_map(lambda x: x + 0.05 * jax.random.normal(jax.random.PRNGKey(4), x.shape), cp)
    inp = sample_lb_inputs(jax.random.PRNGKey(5), **lb_input_kwargs(cfg))
    tr = lb_rollout(spec, lambda o: am.apply(ap, o), inp, jax.random.PRNGKey(6))
    ret = alg.discounted_returns(tr.rewards * 1e-4, 0.995)
    v = cm.apply(cp, tr.cobs); adv = ret - v
    A = flat(ap); C = flat(cp)
    np.savez(path, obs=np.asarray(tr.obs, np.float64), cobs=np.asarray(tr.cobs, np.float64), actions=np.asarray(tr.actions),
             adv=np.asarray(adv, np.float64), ret=np.asarray(ret, np.float64), ent_w=0.37,
             **{f'a_{n}{i}': A[2 * i + j] for i in range(3) for j, n in enumerate('Wb')},
             **{f'c_{n}{i}': C[2 * i + j] for i in range(3) for j, n in enumerate('Wb')})
    # store our params too
    import pickle; pickle.dump((jax.device_get(ap), jax.device_get(cp)), open(path + '.params.pkl', 'wb')); print('made')
else:
    import pickle
    b = np.load(path); ap, cp = pickle.load(open(path + '.params.pkl', 'rb')); o = np.load(path.replace('.npz', '_official.npz'))
    T = b['obs'].shape[0]
    obs, act, adv, ret, cobs = (jnp.asarray(b[k], jnp.float32) for k in ('obs', 'actions', 'adv', 'ret', 'cobs'))
    act = act.astype(jnp.int32); ent_w = float(b['ent_w'])
    (loss, aux), ga = jax.value_and_grad(lambda p: alg.actor_loss(am.apply, p, obs, act, adv, ent_w, 10), has_aux=True)(ap)
    vl, gc = jax.value_and_grad(lambda p: alg.value_loss(cm.apply, p, cobs, ret, 'mean'))(cp)
    rep = {'T': T}
    def cmp(ours, off, scale, tag):
        r = []
        for i, (x, y) in enumerate(zip(ours, off)):
            x = np.asarray(x, np.float64) * scale
            rel = np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30); r.append(float(rel))
        rep[tag] = r
    cmp(flat(ga), [o[f'ga{i}'] for i in range(6)], T, 'actor_grad_rel_err(ours*T vs official)')
    cmp(flat(gc), [o[f'gc{i}'] for i in range(6)], T, 'critic_grad_rel_err(ours*T vs official)')
    probs = np.asarray(jax.nn.softmax(am.apply(ap, obs))); rep['probs_max_abs'] = float(np.abs(probs - o['probs']).max())
    rep['vals_max_abs'] = float(np.abs(np.asarray(cm.apply(cp, cobs)) - o['vals'][:, 0]).max())
    rep['pg_loss ours*T, official adv_loss'] = [float(aux['pg_loss']) * T, float(o['adv_loss'])]
    rep['entropy ours (normalised, mean), official -ent_loss/T'] = [float(aux['entropy']), float(-o['ent_loss'] / T)]
    rep['critic loss ours*T, official'] = [float(vl) * T, float(o['closs'])]
    print(json.dumps(rep, indent=1)); json.dump(rep, open(path.replace('.npz', '_diff.json'), 'w'), indent=1)

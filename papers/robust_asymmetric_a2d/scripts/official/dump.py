"""Oracle dump: run pieces of the OFFICIAL a2d code (plai-group/a2d @368aaf5, PyTorch, float64) on fixed inputs and
save every input/output to an .npz that the JAX re-implementation's tests compare against.

Run from repo/a2d_gridworld with the isolated venv:
    $OFFICIAL_ROOT/venv/bin/python <paper>/scripts/official/dump.py <out.npz>   (fixture: tests/data/oracle.npz)
Randomness inside the official functions (np.random.choice / shuffle, crop draws, torch.normal) is monkeypatched to
replay sequences that are saved alongside, so our side can consume the identical draws.
"""
import sys, types, copy
import numpy as np
import torch
import gym

torch.set_default_dtype(torch.float64)
import a2d.environments.minigrid.gym_minigrid  # noqa: registers envs
from a2d.models.models import DiscPolicy, ValueDenseNet
from a2d.util import rl_steps as RS
from a2d.util import rl_update as RU
from a2d.util.helpers import advantage_and_return, discrete_kl_loss
from a2d.util.replay_buffer import ReplayBuffer
from a2d.A2Dagger import AdpAsymDagger

OUT = {}
rng = np.random.RandomState(1234)
torch.manual_seed(1234)


def put(k, v):
    OUT[k] = np.asarray(v.detach().numpy() if torch.is_tensor(v) else v)


def net_params(net, prefix):
    for name, p in net.named_parameters():
        put(prefix + '/' + name, p)


# ------------------------------------------------------------------ 1. env golden rollouts (random actions)
ENVS = {'FL': 'MiniGrid-LavaGapS7-v0', 'TD': 'MiniGrid-TigerDoorEnv-v0', 'TD1': 'MiniGrid-TigerDoorEnvOSC-v1',
        'TD2': 'MiniGrid-TigerDoorEnvOSC-v2', 'TD3': 'MiniGrid-TigerDoorEnvOSC-v3'}
for tag, gid in ENVS.items():
    env = gym.make(gid).unwrapped
    env.full_rendering_obs_noise = 0.0
    rec = {k: [] for k in ['ep', 'hid', 'x', 'y', 'illum', 'a', 'r', 'done', 'state', 'partial', 'img']}
    np.random.seed(7)
    for ep in range(60 if tag in ('FL', 'TD') else 30):
        env.reset()
        hid = int(env.lava_idx[0]) if tag == 'FL' else int(env.goal_idx[0])
        policy_p = rng.dirichlet(np.ones(4))                  # a random but episode-specific action bias
        for t in range(200):
            rec['ep'].append(ep); rec['hid'].append(hid)
            rec['x'].append(env.agent_pos[0]); rec['y'].append(env.agent_pos[1])
            rec['illum'].append(int(bool(getattr(env, 'illuminated', False))))
            rec['state'].append(env.gen_obs().astype(np.float64))
            rec['partial'].append(env.render(_key='partial_state').astype(np.float64))
            if tag in ('FL', 'TD'):
                rec['img'].append(env.render(_key='observe').astype(np.float64))
            a = int(rng.choice(4, p=policy_p))
            _, r, d, _ = env.step(a)
            rec['a'].append(a); rec['r'].append(r); rec['done'].append(int(d))
            if d:
                break
    for k, v in rec.items():
        if len(v):
            put(f'env/{tag}/{k}', np.stack(v) if k in ('state', 'partial', 'img') else np.array(v))

# ------------------------------------------------------------------ 2. algorithm pieces on a compact FL-like batch
N, DS, DO = 400, 34, 25
P = types.SimpleNamespace(gamma=0.995, lambda_=0.9, entropy_reg=10.0, regularize_adv=True, fisher_frac_samples=0.1,
                          cg_steps=10, damping=0.1, max_kl=0.01, max_backtrack=10, device=torch.device('cpu'),
                          value_calc='gae', val_epochs=25, num_minibatches=32, clip_eps=0.2, value_clipping=1,
                          q_epochs=25, mode='A2D')
expert = DiscPolicy((DS,), 4); learner = DiscPolicy((DO,), 4)
vnet = ValueDenseNet((DS,)); qnet = ValueDenseNet(DO + 4, slim_init=False)
# perturb output layers so the test is not dominated by the 0.01-gain init
with torch.no_grad():
    expert.head[-1].weight.mul_(100.0); learner.head[-1].weight.mul_(50.0); vnet.head[-1].weight.mul_(300.0)
net_params(expert, 'expert'); net_params(learner, 'learner'); net_params(vnet, 'vnet'); net_params(qnet, 'qnet')

pos = rng.randint(0, 25, N); lava = rng.randint(0, 9, N)
S = np.zeros((N, DS)); S[np.arange(N), pos] = 1; S[np.arange(N), 25 + lava] = 1
O = S[:, :25].copy()
A = rng.randint(0, 4, N)
R = rng.choice([-2.0, 18.0, -102.0], N, p=[0.9, 0.05, 0.05])
ND = (rng.rand(N) > 0.06).astype(np.int64); ND[-1] = 1                      # cut last episode (bootstrap V_t)
put('batch/S', S); put('batch/O', O); put('batch/A', A); put('batch/R', R); put('batch/ND', ND)
St, Ot, At = torch.tensor(S), torch.tensor(O), torch.tensor(A)
Rt, NDt = torch.tensor(R), torch.tensor(ND) == 1

with torch.no_grad():
    V = vnet.get_value(St)
adv, ret = advantage_and_return(P, Rt.unsqueeze(0), V.unsqueeze(0), NDt.unsqueeze(0))
put('gae/V', V); put('gae/adv', adv.squeeze()); put('gae/ret', ret.squeeze())

# value_loss_gae on a perturbed prediction
vs = (V + torch.tensor(rng.randn(N)) * 0.5).requires_grad_(True)
vl = RS.value_loss_gae(vs, None, adv.squeeze(), NDt, P, V)
put('vloss/vs', vs); put('vloss/loss', vl)

# importance weights
fake = types.SimpleNamespace(params=P, expert_pol=expert, learner_pol=learner, beta=0.3,
                             to_device=lambda: None, to_cpu=lambda: None)
w = RU.importance_weight(fake, 'expert', St, Ot, At)
put('iw/beta', 0.3); put('iw/w', w)

# surrogate + grad (entropy-in-advantage, normalized), and the Q-args variant
old_lp = RU.batched_evaluation(P, expert, St, At)
put('surr/old_lp', old_lp)
advw = adv.squeeze() * w
put('surr/adv_in', advw)
# store the perturbation actually used (regenerate deterministically)
rng_p = np.random.RandomState(99)
pert = 0.05 * rng_p.randn(N)
put('surr2/pert', pert)
for tag, ent, reg in [('main', 10.0, True), ('qargs', 0.02, False)]:
    P2 = copy.copy(P); P2.entropy_reg = ent; P2.regularize_adv = reg
    pds, lp = RS._apply(expert, P2, St, At)
    s = RS.surrogate_reward(P2, advw, lp, old_lp + torch.tensor(pert), pds).mean()
    g = torch.autograd.grad(s, list(expert.parameters()))
    put(f'surr2/{tag}/value', s)
    for (name, _), gi in zip(expert.named_parameters(), g):
        put(f'surr2/{tag}/grad/{name}', gi)

# AIL KL loss + grad
kl = discrete_kl_loss(learner(Ot), expert(St).detach())
g = torch.autograd.grad(kl, list(learner.parameters()))
put('kl/value', kl)
for (name, _), gi in zip(learner.named_parameters(), g):
    put(f'kl/grad/{name}', gi)

# full TRPO step with a fixed Fisher subsample
sel = rng.choice(N, int(N * P.fisher_frac_samples), replace=False)
put('trpo/sel', sel)
_orig_choice = np.random.choice
np.random.choice = lambda *a, **k: sel
exp2 = copy.deepcopy(expert)
P3 = copy.copy(P)
RS.trpo_step(St, At, old_lp, None, None, None, advw.clone(), exp2, P3, None)
np.random.choice = _orig_choice
net_params(exp2, 'trpo/new')

# full value step (Adam) with fixed shuffles
perms = [rng.permutation(N) for _ in range(P.val_epochs)]
put('vstep/perms', np.stack(perms))
it = iter(perms)
_orig_shuffle = np.random.shuffle


def _shuf(x):
    x[:] = next(it)


np.random.shuffle = _shuf
v2 = copy.deepcopy(vnet)
opt = torch.optim.Adam(v2.parameters(), lr=7e-4, eps=1e-5)
RS.value_step(None, St, ret.squeeze(), adv.squeeze(), NDt, v2, opt, params_in=P)
np.random.shuffle = _orig_shuffle
net_params(v2, 'vstep/new')

# Q step (MC targets) with fixed shuffles
qperms = [rng.permutation(N) for _ in range(P.q_epochs)]
put('qstep/perms', np.stack(qperms))
it = iter(qperms)
np.random.shuffle = _shuf
q2 = copy.deepcopy(qnet)
qopt = torch.optim.Adam(q2.parameters(), lr=3e-4, eps=1e-5)
RS.q_step(fake, Ot, ret.squeeze(), At, None, qopt, q2, 4, P)
np.random.shuffle = _orig_shuffle
net_params(q2, 'qstep/new')

# projection (AIL) epochs through the official ReplayBuffer epoch sampler, fixed shuffles, no augmentation
NB = 300
bs = torch.tensor(S[:NB]); bo = torch.tensor(O[:NB]).unsqueeze(1)          # obs stored with a frame-stack dim
bperms = [rng.permutation(NB) for _ in range(4)]
put('proj/perms', np.stack(bperms)); put('proj/NB', NB)
it = iter(bperms)
np.random.shuffle = _shuf
rb = ReplayBuffer(64, 5000)
rb.push((bs, bo, torch.zeros(NB)))
l2 = copy.deepcopy(learner)
fakeA = types.SimpleNamespace(replay_buffer=rb, learner_pol=l2, expert_pol=expert,
                              discrete_kl_loss=discrete_kl_loss, params=P,
                              optims={'learner_pol': torch.optim.Adam(l2.parameters(), lr=3e-4, eps=1e-5)})
fakeA.compute_loss = types.MethodType(AdpAsymDagger.compute_loss, fakeA)
losses = []
for ep in range(2):
    losses.append(float(AdpAsymDagger.mini_batch_update(fakeA, update=True, augmentate=False)))
np.random.shuffle = _orig_shuffle
put('proj/losses', np.array(losses))
net_params(l2, 'proj/new')

# ------------------------------------------------------------------ 3. image networks forward
ipol = DiscPolicy((3, 42, 42), 4); ival = ValueDenseNet((3, 42, 42))
net_params(ipol, 'ipol'); net_params(ival, 'ival')
imgs = rng.rand(5, 42, 42, 3)                     # HWC as rendered
x = torch.tensor(imgs).permute(0, 3, 2, 1).unsqueeze(1).permute(0, 1, 3, 4, 2)   # official (B,S,W,H,C) of render.T
with torch.no_grad():
    put('inet/imgs', imgs); put('inet/logp', ipol(x)); put('inet/v', ival(x))

# ------------------------------------------------------------------ 4. image augmentation (crop + speckle + resize)
rb2 = ReplayBuffer(64, 5000)
rb2.push((torch.zeros(8, DS), torch.tensor(rng.rand(8, 1, 42, 42, 3)), torch.zeros(8)))
batch_obs = rb2.obs[:4].clone()
cf = np.array([0.85, 0.97]); orig = [5, 1]   # valid: origin < 46 - int(46 cf)
_u, _ri, _tn = np.random.uniform, np.random.randint, torch.normal
np.random.uniform = lambda *a, **k: cf
_oi = iter(orig)
np.random.randint = lambda *a, **k: next(_oi)
torch.normal = lambda mean, std, size: torch.zeros(size)
aug = rb2.augmentator(batch_obs)
np.random.uniform, np.random.randint, torch.normal = _u, _ri, _tn
put('aug/in', batch_obs); put('aug/out', aug); put('aug/cf', cf); put('aug/orig', np.array(orig))

np.savez_compressed(sys.argv[1], **OUT)
print('wrote', len(OUT), 'arrays to', sys.argv[1])

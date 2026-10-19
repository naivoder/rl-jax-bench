"""Parity fixtures from the authors' code. Run with the authors'-code venv, never ours:

    . ${OFFICIAL_ROOT:-external/official}/env.sh      # written by scripts/official/setup.sh
    ${OFFICIAL_ROOT:-external/official}/.venv/bin/python scripts/official/make_fixtures.py

Writes tests/fixtures/official_*.npz:
  * official_pd_trace.npz   : 10 000 random actions through PoisonedDoorsEnvironment/Task (obs, reward, done, expert)
  * official_lh_trace.npz   : random-action trajectories through LightHouseEnvironment; per step the corner tuple
                              for view radii 1..15 and the expert action for expert radii 1..15
  * official_loss.npz       : AdvisorWeightedStage(PPO) / PPO / Imitation losses and float64 gradients w.r.t. main
                              logits, aux logits and values on random inputs
  * official_lh_design.npz  : sha256 of the FactorialDesignCornerSensor matrix (DEGREE=-1)
"""
import hashlib
import os
import random
import sys

import numpy as np
import torch

OUT = os.environ.get("FIXTURE_OUT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)


def pd_trace(n_steps=10000, seed=0):
    from projects.advisor.poisoneddoors_plugin.poisoneddoors_tasks import PoisonedDoorsTaskSampler
    from projects.advisor.poisoneddoors_plugin.poisoneddoors_sensors import PoisonedDoorCurrentStateSensor
    sampler = PoisonedDoorsTaskSampler(num_doors=4, combination_length=10, sensors=[PoisonedDoorCurrentStateSensor()],
                                       max_steps=100, seed=seed)
    rng = np.random.RandomState(seed)
    rows = []
    task = sampler.next_task()
    good = task.env.good_door_ind
    obs = task.get_observations()
    for t in range(n_steps):
        o = int(np.asarray(list(obs.values())[0]).reshape(-1)[0])
        expert, ok = task.query_expert()
        # bias the random policy towards the locked door + correct digits so code states are well covered
        if rng.rand() < 0.5:
            a = int(expert) if (o != 0 or rng.rand() < 0.5) else 3
        else:
            a = int(rng.randint(7))
        res = task.step(a)
        rows.append((good, o, a, res.reward, int(res.done), int(expert)))
        obs = res.observation
        if res.done:
            task = sampler.next_task()
            good = task.env.good_door_ind
            obs = task.get_observations()
    arr = np.array(rows, np.float64)
    np.savez(os.path.join(OUT, "official_pd_trace.npz"), rows=arr,
             combination=np.array(task.env.combination))
    print("pd trace", arr.shape, "returns", arr[:, 3].sum(), "combination", task.env.combination)


def lh_trace(n_traj=200, steps=300, seed=0):
    from allenact_plugins.lighthouse_plugin.lighthouse_environment import LightHouseEnvironment
    from allenact_plugins.lighthouse_plugin.lighthouse_sensors import get_corner_observation
    from allenact_plugins.lighthouse_plugin.lighthouse_tasks import FindGoalLightHouseTask
    rng = np.random.RandomState(seed)
    goals, actions, tuples, experts = [], [], [], []
    for k in range(n_traj):
        env = LightHouseEnvironment(world_dim=2, world_radius=15, seed=k)
        env.random_reset()
        goal_idx = [tuple(c) for c in env.world_corners.tolist()].index(tuple(env.goal_position.tolist()))
        task = FindGoalLightHouseTask(env=env, sensors=[], task_info={}, max_steps=1000)
        # random walk with momentum so far corners are reached
        a_seq, tup_seq, exp_seq = [], [], []
        a = rng.randint(4)
        for t in range(steps):
            tup_seq.append([get_corner_observation(env, r, None).astype(int).tolist() for r in range(1, 16)])
            exp_seq.append([task.query_expert(expert_view_radius=r, deterministic=True)[0] for r in range(1, 16)])
            if rng.rand() < 0.3:
                a = rng.randint(4)
            a_seq.append(a)
            env.step(a)
        goals.append(goal_idx); actions.append(a_seq); tuples.append(tup_seq); experts.append(exp_seq)
    np.savez_compressed(os.path.join(OUT, "official_lh_trace.npz"), goals=np.array(goals, np.int8),
                        actions=np.array(actions, np.int8), tuples=np.array(tuples, np.int8),
                        experts=np.array(experts, np.int8))
    print("lh trace", np.array(tuples).shape)


def lh_design():
    from allenact_plugins.lighthouse_plugin.lighthouse_sensors import FactorialDesignCornerSensor
    s = FactorialDesignCornerSensor(view_radius=1, world_dim=2, degree=-1)
    M = np.asarray(s.design_matrix, dtype=bool)
    h = hashlib.sha256(np.packbits(M).tobytes()).hexdigest()
    np.savez(os.path.join(OUT, "official_lh_design.npz"), sha256=np.array(h), shape=np.array(M.shape),
             rows_sample=M[::97].astype(np.uint8))
    print("lh design", M.shape, h)


def losses(seed=0):
    """Official losses on random inputs; gradients w.r.t. logits / aux logits / values in float64."""
    torch.set_default_dtype(torch.float64)
    from allenact.algorithms.onpolicy_sync.losses.ppo import PPO, PPOConfig
    from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
    from allenact.base_abstractions.distributions import CategoricalDistr
    from allenact.base_abstractions.misc import ActorCriticOutput
    from allenact.utils.experiment_utils import LinearDecay
    from projects.advisor.advisor_losses import AdvisorWeightedStage
    rng = np.random.RandomState(seed)
    T, B, A = 7, 5, 7
    out = {}
    for case, (alpha, bound, step_count, steps) in enumerate([(8.0, 0.0, 3000, 300000), (4.0, 0.0, 150000, 300000),
                                                             (20.0, 0.2, 290000, 300000)]):
        logits = rng.randn(T, B, A) * 1.5
        aux = rng.randn(T, B, A) * 1.5
        vals = rng.randn(T, B, 1)
        actions = rng.randint(A, size=(T, B, 1))
        labels = rng.randint(A, size=(T, B, 1))
        mask = (rng.rand(T, B, 1) < 0.9).astype(np.int64)
        old_lp = np.log(np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)) + rng.randn(T, B, A) * 0.1
        old_lp_a = np.take_along_axis(old_lp, actions, -1)
        adv = rng.randn(T, B, 1)
        old_v = vals + rng.randn(T, B, 1) * 0.2
        rets = rng.randn(T, B, 1)
        res = {}
        for name in ("advisor", "ppo", "imitation"):
            lg = torch.tensor(logits, requires_grad=True)
            ax = torch.tensor(aux, requires_grad=True)
            v = torch.tensor(vals, requires_grad=True)
            aco = ActorCriticOutput(distributions=CategoricalDistr(logits=lg), values=v,
                                    extras={"auxiliary_distributions": CategoricalDistr(logits=ax)})
            batch = {"observations": {"expert_action": torch.tensor(np.concatenate([labels, mask], -1))},
                     "actions": torch.tensor(actions), "old_action_log_probs": torch.tensor(old_lp_a),
                     "norm_adv_targ": torch.tensor(adv), "adv_targ": torch.tensor(adv),
                     "values": torch.tensor(old_v), "returns": torch.tensor(rets), "masks": torch.ones(T, B, 1)}
            ppo = PPO(clip_decay=LinearDecay(steps), **PPOConfig)
            if name == "advisor":
                loss, _ = AdvisorWeightedStage(rl_loss=ppo, fixed_alpha=alpha, fixed_bound=bound).loss(
                    step_count=step_count, batch=batch, actor_critic_output=aco)
            elif name == "ppo":
                loss, _ = ppo.loss(step_count=step_count, batch=batch, actor_critic_output=aco)
            else:
                loss, _ = Imitation().loss(step_count=step_count, batch=batch, actor_critic_output=aco)
            loss.backward()
            g = lambda x: np.zeros(x.shape) if x.grad is None else x.grad.numpy().copy()
            res[name] = (float(loss), g(lg), g(ax), g(v))
        out[case] = dict(alpha=alpha, bound=bound, clip=0.1 * LinearDecay(steps)(step_count), logits=logits, aux=aux,
                         vals=vals[..., 0], actions=actions[..., 0], labels=labels[..., 0], mask=mask[..., 0],
                         old_lp_a=old_lp_a[..., 0], adv=adv[..., 0], old_v=old_v[..., 0], rets=rets[..., 0], res=res)
    flat = {}
    for c, d in out.items():
        for k, v in d.items():
            if k == "res":
                for name, (l, g1, g2, g3) in v.items():
                    flat[f"c{c}_{name}_loss"] = np.array(l)
                    flat[f"c{c}_{name}_glogits"] = g1
                    flat[f"c{c}_{name}_gaux"] = g2
                    flat[f"c{c}_{name}_gvals"] = g3[..., 0]
            else:
                flat[f"c{c}_{k}"] = np.asarray(v)
    np.savez(os.path.join(OUT, "official_loss.npz"), **flat)
    print("losses", {k: float(v) for k, v in flat.items() if k.endswith("_loss")})


def mg(n_layouts=50, steps=200, n_stats=1000, seed=0):
    """LC Once Switch: official AskForHelpSimpleCrossing(S15, N7, lava, toggle permanent) traces on official layouts,
    plus generator statistics (lava fraction, start shortest-path length) over n_stats seeds."""
    import networkx as nx
    from gym_minigrid.minigrid import Lava
    from gym_minigrid.envs.crossing import CrossingEnv
    from allenact_plugins.minigrid_plugin.minigrid_environments import AskForHelpSimpleCrossing
    from allenact_plugins.minigrid_plugin.minigrid_tasks import AskForHelpSimpleCrossingTask
    rng = np.random.RandomState(seed)
    grids, acts, imgs, lit, rews, dones, poss, dirs, plen, expert = [], [], [], [], [], [], [], [], [], []
    for k in range(n_layouts):
        env = AskForHelpSimpleCrossing(size=15, num_crossings=7, obstacle_type=Lava, toggle_is_permenant=True)
        env.seed(1000 + k)
        env.reset()
        task = AskForHelpSimpleCrossingTask(env=env, sensors=[], task_info={}, max_steps=env.max_steps)
        grids.append(env.grid.encode()[:, :, 0].copy())
        A_, I_, L_, R_, D_, P_, Dr_, PL_, E_ = [], [], [], [], [], [], [], [], []
        done = False
        for t in range(steps):
            src = (int(env.agent_pos[0]), int(env.agent_pos[1]), int(env.agent_dir))
            P_.append(src[:2]); Dr_.append(src[2])
            PL_.append(len(nx.shortest_path(task.graph, src, "unified_goal")) - 2)
            E_.append(task.query_expert()[0])
            L_.append(CrossingEnv.gen_obs(env)["image"].copy())
            u = rng.rand()
            a = 2 if u < 0.55 else (0 if u < 0.7 else (1 if u < 0.85 else 3))
            if rng.rand() < 0.3:
                a = int(E_[-1]) if E_[-1] >= 0 else a
            res = task.step(a)
            A_.append(a); I_.append(res.observation if isinstance(res.observation, np.ndarray) else env.gen_obs()["image"].copy())
            R_.append(res.reward); D_.append(int(res.done))
            if res.done:
                break
        n = len(A_)
        pad = lambda x, shape, dt: np.concatenate([np.asarray(x, dt).reshape((n,) + shape), np.zeros((steps - n,) + shape, dt)])
        acts.append(pad(A_, (), np.int8)); imgs.append(pad(I_, (7, 7, 3), np.uint8)); lit.append(pad(L_, (7, 7, 3), np.uint8))
        rews.append(pad(R_, (), np.float64)); dones.append(pad(D_, (), np.int8)); poss.append(pad(P_, (2,), np.int16))
        dirs.append(pad(Dr_, (), np.int8)); plen.append(pad(PL_, (), np.int16)); expert.append(pad(E_, (), np.int8))
    stats_lava, stats_len = [], []
    for k in range(n_stats):
        env = AskForHelpSimpleCrossing(size=15, num_crossings=7, obstacle_type=Lava, toggle_is_permenant=True)
        env.seed(50000 + k)
        env.reset()
        task = AskForHelpSimpleCrossingTask(env=env, sensors=[], task_info={}, max_steps=env.max_steps)
        g = env.grid.encode()[:, :, 0]
        stats_lava.append(float((g == 9).mean()))
        stats_len.append(len(nx.shortest_path(task.graph, (1, 1, 0), "unified_goal")) - 2)
    np.savez_compressed(os.path.join(OUT, "official_mg_trace.npz"), grids=np.array(grids, np.int8),
                        actions=np.array(acts), obs=np.array(imgs), lit=np.array(lit), rewards=np.array(rews),
                        dones=np.array(dones), pos=np.array(poss), dirs=np.array(dirs), path_len=np.array(plen),
                        expert=np.array(expert), stats_lava=np.array(stats_lava), stats_len=np.array(stats_len))
    print("mg", np.array(acts).shape, "mean start path len", np.mean(stats_len), "lava frac", np.mean(stats_lava))


def mg_expert_all(n_layouts=100):
    """The official networkx expert's action at EVERY non-terminal agent state (x, y, dir)
    of n_layouts official layouts (seeds 2000+k), so the tie-breaking among equally short moves can be compared with
    ours exhaustively, not only along traces."""
    import networkx as nx
    from gym_minigrid.minigrid import Lava
    from allenact_plugins.minigrid_plugin.minigrid_environments import AskForHelpSimpleCrossing
    from allenact_plugins.minigrid_plugin.minigrid_tasks import AskForHelpSimpleCrossingTask
    grids, states, experts = [], [], []
    for k in range(n_layouts):
        env = AskForHelpSimpleCrossing(size=15, num_crossings=7, obstacle_type=Lava, toggle_is_permenant=True)
        env.seed(2000 + k)
        env.reset()
        task = AskForHelpSimpleCrossingTask(env=env, sensors=[], task_info={}, max_steps=env.max_steps)
        g = env.grid.encode()[:, :, 0].copy()
        grids.append(g)
        for x in range(g.shape[0]):
            for y in range(g.shape[1]):
                if g[x, y] != 1:            # 'empty' only (walls, lava, goal excluded)
                    continue
                for d in range(4):
                    env.agent_pos = np.array([x, y]); env.agent_dir = d
                    try:
                        e = task.query_expert()[0]
                    except nx.NetworkXNoPath:      # empty cells walled off from the goal
                        continue
                    states.append((k, x, y, d)); experts.append(int(e))
    np.savez_compressed(os.path.join(OUT, "official_mg_expert_all.npz"), grids=np.array(grids, np.int8),
                        states=np.array(states, np.int16), experts=np.array(experts, np.int8))
    print("mg_expert_all", len(experts), "states over", n_layouts, "layouts")


def robust_reward(seed=0):
    """allenact.utils.misc_utils.expected_max_of_subset_statistic (the RobustReward@K the authors' plots use) on
    random value sets, including heavy ties (rewards in PD are often exactly 0 / 1)."""
    from allenact.utils.misc_utils import expected_max_of_subset_statistic
    rng = np.random.RandomState(seed)
    vals, ks, outs = [], [], []
    for case in range(30):
        n = int(rng.randint(5, 51))
        v = rng.randn(n) if case % 3 == 0 else (rng.randint(0, 4, n) / 3.0 if case % 3 == 1 else rng.rand(n))
        for k in sorted(set([1, 2, min(10, n), n, int(rng.randint(1, n + 1))])):
            vals.append(np.pad(v, (0, 50 - n), constant_values=np.nan)); ks.append(k)
            outs.append(expected_max_of_subset_statistic(list(map(float, v)), k))
    np.savez(os.path.join(OUT, "official_rr.npz"), vals=np.array(vals), ks=np.array(ks), outs=np.array(outs))
    print("rr", len(outs))


def early_stop(seed=0):
    """allenact_plugins.lighthouse_plugin.lighthouse_util.StopIfNearOptimal (min_memory_size 50, deviation 5 % of
    optimal) driven by streams of per-interval (episode count, mean episode length) as the engine feeds it."""
    from allenact.utils.experiment_utils import ScalarMeanTracker
    from allenact_plugins.lighthouse_plugin.lighthouse_util import StopIfNearOptimal
    rng = np.random.RandomState(seed)
    S, L = 60, 40
    counts = np.zeros((S, L), np.int64); aves = np.zeros((S, L)); opts = np.zeros(S)
    decisions = np.zeros((S, L), np.int8); means = np.zeros((S, L))
    for s in range(S):
        opt = float([72, 60, 42, 30][s % 4]); opts[s] = opt
        crit = StopIfNearOptimal(optimal=opt, deviation=opt * 0.05, min_memory_size=50)
        # a noisy learning curve from long episodes (few per interval) to near-optimal (many per interval)
        start, end = rng.uniform(150, 1000), opt * rng.uniform(0.95, 1.25)
        for t in range(L):
            ave = max(10.0, end + (start - end) * np.exp(-t / rng.uniform(2, 15)) + rng.randn() * opt * 0.05)
            cnt = int(max(0, rng.poisson(10000.0 / ave))) if rng.rand() > 0.05 else 0
            tr = ScalarMeanTracker()
            if cnt > 0:
                tr.add_scalars({"ep_length": ave}, n={"ep_length": cnt})
            decisions[s, t] = int(crit(stage_steps=0, total_steps=0, training_metrics=tr))
            counts[s, t], aves[s, t], means[s, t] = cnt, ave, crit.memory.mean()
    np.savez(os.path.join(OUT, "official_es.npz"), counts=counts, aves=aves, opts=opts, decisions=decisions,
             means=means)
    print("es", decisions.sum(1)[:10])


def model_update(seed=0):
    """Official model forward + ONE full AllenAct update (4 epochs x 2 minibatches of whole sampler sequences:
    AdvisorWeightedStage(PPO) loss, backward, clip_grad_norm_(0.5), Adam with the LambdaLR factor) in float64 for the
    PD model (RNNActorCriticWithEmbed + LinearAdvisorActorCritic), the MiniGrid model (MiniGridSimpleConvRNN) and the
    2D-LH linear model. Returns use the authors' RolloutStorage.compute_returns. Minibatch order follows
    recurrent_generator (contiguous sampler halves, random.shuffle) and is recorded."""
    import random
    import types
    torch.set_default_dtype(torch.float64)
    from allenact.algorithms.onpolicy_sync.losses.ppo import PPO, PPOConfig
    from allenact.algorithms.onpolicy_sync.storage import RolloutStorage
    from allenact.base_abstractions.misc import Memory
    from allenact.utils.experiment_utils import LinearDecay
    from projects.advisor.advisor_losses import AdvisorWeightedStage
    from projects.advisor.minigrid_and_pd_experiments.advisor import MgPdAdvisor
    flat = {}
    T, B, n_mb, epochs, lr0, alpha = 12, 4, 2, 4, 3e-3, 8.0
    total_before, total_steps = 60000, 300000          # lr factor = LinearDecay(total_steps)(total_before)
    stage_steps = total_before + T * B                  # clip decay uses steps in stage incl. this rollout
    for name in ("pd", "mg", "lh"):
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed + 1)
        random.seed(seed + 2)
        if name == "lh":
            from allenact_plugins.lighthouse_plugin.lighthouse_models import LinearAdvisorActorCritic
            from allenact_plugins.lighthouse_plugin.lighthouse_sensors import FactorialDesignCornerSensor
            import gym
            from gym.spaces.dict import Dict as SpaceDict
            sens = FactorialDesignCornerSensor(view_radius=1, world_dim=2, degree=-1)
            M = np.asarray(sens.design_matrix, dtype=np.float64)
            model = LinearAdvisorActorCritic(input_uuid="x", action_space=gym.spaces.Discrete(4),
                                             observation_space=SpaceDict({"x": gym.spaces.Box(0, 1, (M.shape[1],))}),
                                             ensure_same_init_aux_weights=False).double()
            rows = rng.randint(M.shape[0], size=(T + 1, B))
            obs_t = torch.tensor(M[rows]); uuid = "x"; flat[f"{name}_rows"] = rows; nA = 4
        elif name == "pd":
            cfg = MgPdAdvisor("PoisonedDoors", LR=lr0, FIXED_ALPHA=alpha)
            model = cfg.create_model().double(); uuid = cfg.get_sensors()[0].uuid; nA = 7
            obs = rng.randint(4, size=(T + 1, B, 1)); obs_t = torch.tensor(obs); flat[f"{name}_obs"] = obs[..., 0]
        else:
            cfg = MgPdAdvisor("AskForHelpLavaCrossingOnce", LR=lr0, FIXED_ALPHA=alpha)
            model = cfg.create_model().double(); sen = cfg.get_sensors()[0]; uuid = sen.uuid; nA = 4
            flat["mg_ncodes"] = np.array([sen.num_objects, sen.num_colors, sen.num_states])
            obs = np.stack([rng.randint(sen.num_objects, size=(T + 1, B, 7, 7)), rng.randint(sen.num_colors, size=(T + 1, B, 7, 7)),
                            rng.randint(sen.num_states, size=(T + 1, B, 7, 7))], -1).astype(np.uint8)
            obs_t = torch.tensor(obs); flat[f"{name}_obs"] = obs
        recurrent = name != "lh"
        H = 128
        masks = (rng.rand(T + 1, B, 1) > 0.2).astype(np.float64); masks[0, 0] = 0.0
        h0 = rng.randn(2, B, H) * 0.5 if recurrent else None
        sd0 = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}

        def fwd(sl, t0, t1):
            mem = Memory({"rnn": (torch.tensor(h0[:, sl]), 1)}) if recurrent else None
            out, _ = model(observations={uuid: obs_t[t0:t1, sl]}, memory=mem,
                           prev_actions=torch.zeros(t1 - t0, len(range(*sl.indices(B))), dtype=torch.long),
                           masks=torch.tensor(masks[t0:t1, sl]))
            return out
        with torch.no_grad():
            out = fwd(slice(0, B), 0, T + 1)       # all T+1 steps from h0 (step T only for the bootstrap value)
            logits = out.distributions.logits.numpy(); aux = out.extras["auxiliary_distributions"].logits.numpy()
            vals = out.values.numpy()[..., 0]
        actions = np.stack([[rng.choice(nA, p=np.exp(l - l.max()) / np.exp(l - l.max()).sum()) for l in row] for row in logits[:T]])
        lsm = logits[:T] - np.log(np.exp(logits[:T]).sum(-1, keepdims=True))
        old_lp = np.take_along_axis(lsm, actions[..., None], -1)[..., 0]
        rewards = rng.randn(T, B) * 0.5
        labels = rng.randint(nA, size=(T, B)); lmask = (rng.rand(T, B) < 0.9).astype(np.int64)
        st = types.SimpleNamespace(masks=torch.tensor(masks), rewards=torch.tensor(rewards[..., None]),
                                   value_preds=torch.tensor(vals[..., None]).clone(), returns=torch.zeros(T + 1, B, 1))
        st._extend_tensor = lambda x: RolloutStorage._extend_tensor(st, x)
        RolloutStorage.compute_returns(st, next_value=torch.tensor(vals[T][:, None]), use_gae=True, gamma=0.99, tau=1.0)
        rets = st.returns[:-1, :, 0].numpy()
        adv = rets - vals[:T]
        adv_t = torch.tensor(adv)
        nadv = ((adv_t - adv_t.mean()) / (adv_t.std() + 1e-5)).numpy()
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=LinearDecay(steps=total_steps))
        sched.step(epoch=total_before)
        loss_obj = AdvisorWeightedStage(rl_loss=PPO(clip_decay=LinearDecay(total_steps), **PPOConfig),
                                        fixed_alpha=alpha, fixed_bound=0)
        inds = np.round(np.linspace(0, B, n_mb + 1)).astype(np.int32)
        order, losses = [], []
        for e in range(epochs):
            pairs = list(zip(inds[:-1], inds[1:]))
            random.shuffle(pairs)
            for a_, b_ in pairs:
                order.append(int(a_ // (B // n_mb)))
                sl = slice(int(a_), int(b_))
                o = fwd(sl, 0, T)
                batch = {"observations": {"expert_action": torch.tensor(np.stack([labels[:, sl], lmask[:, sl]], -1))},
                         "actions": torch.tensor(actions[:, sl][..., None]),
                         "old_action_log_probs": torch.tensor(old_lp[:, sl][..., None]),
                         "norm_adv_targ": torch.tensor(nadv[:, sl][..., None]), "adv_targ": torch.tensor(adv[:, sl][..., None]),
                         "values": torch.tensor(vals[:T, sl][..., None]), "returns": torch.tensor(rets[:, sl][..., None]),
                         "masks": torch.tensor(masks[:T, sl])}
                loss, _ = loss_obj.loss(step_count=stage_steps, batch=batch, actor_critic_output=o)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step(); losses.append(float(loss))
        sd1 = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
        for k in sd0:
            flat[f"{name}_p0::{k}"] = sd0[k]; flat[f"{name}_p1::{k}"] = sd1[k]
        flat.update({f"{name}_{k}": v for k, v in dict(
            logits=logits, aux=aux, vals=vals, masks=masks[..., 0], actions=actions, old_lp=old_lp, rewards=rewards,
            labels=labels, lmask=lmask, rets=rets, nadv=nadv, order=np.array(order), losses=np.array(losses),
            lr_factor=np.array(LinearDecay(steps=total_steps)(total_before)),
            clip=np.array(0.1 * LinearDecay(total_steps)(stage_steps))).items()})
        if recurrent:
            flat[f"{name}_h0"] = h0
        print(name, "params", list(sd0)[:8], "losses", np.round(losses, 4))
    flat.update(T=np.array(T), B=np.array(B), lr0=np.array(lr0), alpha=np.array(alpha))
    np.savez_compressed(os.path.join(OUT, "official_update.npz"), **flat)


if __name__ == "__main__":
    what = sys.argv[1:] or ["pd", "lh", "design", "loss"]
    if "pd" in what:
        pd_trace()
    if "lh" in what:
        lh_trace()
    if "design" in what:
        lh_design()
    if "loss" in what:
        losses()
    if "mg" in what:
        mg()
    if "mg_expert_all" in what:
        mg_expert_all()
    if "rr" in what:
        robust_reward()
    if "es" in what:
        early_stop()
    if "update" in what:
        model_update()

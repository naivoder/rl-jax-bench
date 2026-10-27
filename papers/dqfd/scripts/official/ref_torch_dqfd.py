"""INDEPENDENT reference run of DQfD (labelled SECOND-BEST ORACLE; no official code exists).

Everything here is a separate stack from the JAX implementation in src/:
  * environment : the ORIGINAL numpy MinAtar (kenjyoung/MinAtar 4241f246, `minatar/environments/<game>.py` loaded by path),
  * loss        : opendilab/DI-engine d0b21d06 `dqfd_nstep_td_error` (loaded by scripts/official/oracle_di_engine.py:load_td; PyTorch),
  * network     : PyTorch re-implementation of src/model.py DuelingQNet (parity-tested in tests/test_official_parity.py),
  * replay      : numpy, written from the PAPER text (ONE buffer, permanent demo region, p = |delta| + eps_d/eps_a,
                  alpha 0.4, beta 0.6 -> 1, n-step windows at sample time) -- NOT DI-engine's separate-buffer scheme,
  * optimiser   : torch Adam(lr 1e-4, eps 1.5e-4) + clip_grad_norm 10 (same values as src/configs/default.yaml).
Runs under the torch venv (NOT the JAX venv):

  nice -n 19 $OFFICIAL_ROOT/.venv/bin/python -I scripts/official/ref_torch_dqfd.py \
      --demo data/demos/Breakout_subopt.npz --arm DQfD --seed 0 --steps 200000 --pretrain 75000 --out runs/ref/DQfD_s0.json

Declared deviations from the JAX code: a time-capped episode (1000 steps) is treated as TERMINATED (DI-engine's
`dqfd_nstep_td_error` has no truncation input), the per-run RNG streams differ (numpy/torch vs jax.random), the initial
weights are a truncated normal with lecun scale written in torch.
"""
import argparse, importlib.util, json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(os.path.dirname(HERE))
MINATAR = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(PAPER, "external", "official")), "MinAtar", "minatar", "environments")
GAMES = {"Breakout": "breakout", "SpaceInvaders": "space_invaders", "Asterix": "asterix", "Seaquest": "seaquest"}
ACTION_SETS = {"Breakout": [0, 1, 3], "SpaceInvaders": [0, 1, 3, 5], "Asterix": [0, 1, 2, 3, 4], "Seaquest": [0, 1, 2, 3, 4, 5]}


def load_td():
    spec = importlib.util.spec_from_file_location("oracle_di_engine", os.path.join(HERE, "oracle_di_engine.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.load_td()


class Net(nn.Module):
    """Conv(16,3x3,SAME)+ReLU -> flatten -> 2 streams Dense(64)+ReLU -> V, A;  Q = V + A - mean(A)  (src/model.py)."""
    def __init__(self, n_actions, in_ch, channels=16, hidden=64):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, channels, 3, padding=1)
        self.v_hidden, self.v_out = nn.Linear(channels * 100, hidden), nn.Linear(hidden, 1)
        self.a_hidden, self.a_out = nn.Linear(channels * 100, hidden), nn.Linear(hidden, n_actions)
        for m in (self.conv, self.v_hidden, self.v_out, self.a_hidden, self.a_out):
            fan_in = m.weight[0].numel()
            nn.init.trunc_normal_(m.weight, std=(1.0 / fan_in) ** 0.5 / 0.87962566103423978, a=-2 * (1.0 / fan_in) ** 0.5 / 0.8796, b=2 * (1.0 / fan_in) ** 0.5 / 0.8796)
            nn.init.zeros_(m.bias)

    def forward(self, x):  # x (B, C, 10, 10) NCHW
        x = F.relu(self.conv(x)).flatten(1)   # NOTE: CHW flatten order (the JAX net flattens HWC); a fixed permutation of the dense input
        v = self.v_out(F.relu(self.v_hidden(x)))
        a = self.a_out(F.relu(self.a_hidden(x)))
        return v + a - a.mean(1, keepdim=True)


class StickyEnv:
    """Original per-game Env + the original `Environment` wrapper's sticky-action rule (prob p repeats the last action)."""
    def __init__(self, game, seed, sticky=0.0, max_steps=1000):
        spec = importlib.util.spec_from_file_location(f"mi_{game}", f"{MINATAR}/{GAMES[game]}.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.env = mod.Env() if game == "Breakout" else mod.Env(ramping=True)
        self.rs = np.random.RandomState(seed)
        self.env.random = self.rs
        self.aset, self.sticky, self.max_steps = ACTION_SETS[game], sticky, max_steps
        self.n_actions = len(self.aset)
        self.last = 0

    def reset(self):
        self.env.reset(); self.t = 0; self.last = 0
        return self.obs()

    def obs(self):
        return np.ascontiguousarray(self.env.state().transpose(2, 0, 1)).astype(np.uint8)

    def step(self, a):
        full = self.aset[a]
        if self.sticky > 0 and self.rs.rand() < self.sticky:
            full = self.last
        self.last = full
        r, term = self.env.act(full)
        self.t += 1
        capped = (not term) and self.t >= self.max_steps
        return self.obs(), float(r), bool(term or capped), bool(capped)


class Replay:
    """One buffer, demo region [0, nd) permanent + agent ring [nd, nd+cap).  Paper: p_i = |delta_i| + eps (eps_d / eps_a)."""
    def __init__(self, demo, cap, n, gamma, alpha, eps_d, eps_a, use_demo, obs_dim):
        self.nd, self.cap, self.n, self.g, self.alpha = len(demo["act"]), cap, n, gamma, alpha
        self.eps_d, self.eps_a = eps_d, eps_a
        N = self.nd + cap
        self.obs = np.zeros((N, obs_dim), np.uint8); self.nobs = np.zeros((N, obs_dim), np.uint8)
        self.act = np.zeros(N, np.int64); self.rew = np.zeros(N, np.float32)
        self.term = np.zeros(N, bool); self.end = np.zeros(N, bool); self.is_demo = np.zeros(N, bool)
        self.pa = np.zeros(N, np.float64)
        self.maxp = 1.0
        if use_demo:
            s = slice(0, self.nd)
            self.obs[s], self.nobs[s], self.act[s] = demo["obs"], demo["nobs"], demo["act"]
            self.rew[s], self.term[s], self.end[s], self.is_demo[s] = demo["rew"], demo["term"], demo["end"], True
            self.maxp = 1.0 + eps_d                               # demo initial priority |delta| := 1
            self.pa[s] = self.maxp ** alpha
        self.ptr, self.nw = 0, 0

    def add(self, o, no, a, r, term, end):
        w = self.nd + self.ptr
        self.obs[w], self.nobs[w], self.act[w], self.rew[w], self.term[w], self.end[w] = o, no, a, r, term, end
        self.is_demo[w] = False; self.pa[w] = 0.0                  # pending until its n-step window is complete
        self.nw += 1
        if self.nw >= self.n:
            self.pa[self.nd + (self.ptr - (self.n - 1)) % self.cap] = self.maxp ** self.alpha
        self.ptr = (self.ptr + 1) % self.cap

    def sample(self, B, rng, beta):
        c = np.cumsum(self.pa); tot = c[-1]
        idx = np.minimum(np.searchsorted(c, rng.random(B) * tot, side="right"), len(self.pa) - 1)
        P = self.pa[idx] / tot
        n_act = float((self.pa > 0).sum())
        raw = (n_act * P) ** (-beta)
        return idx, (raw / raw.max()).astype(np.float32)

    def windows(self, idx):
        """n-step window per sampled index: rewards (n,B) zero-padded after the episode end, done (window terminated), s_n."""
        B, n = len(idx), self.n
        k = np.arange(n)[None, :]
        W = np.where(idx[:, None] >= self.nd, self.nd + (idx[:, None] - self.nd + k) % self.cap, np.minimum(idx[:, None] + k, len(self.pa) - 1))
        ek = self.end[W]
        alive = np.concatenate([np.ones((B, 1), bool), np.cumsum(ek[:, :-1], axis=1) == 0], axis=1)
        rew = np.where(alive, self.rew[W], 0.0).T.astype(np.float32)          # (n, B)
        m = alive.sum(1)
        last = W[np.arange(B), m - 1]
        done_n = (self.end[last]).astype(np.float32)                           # window reached an episode end (time cap treated as end)
        # if the window is a full n steps without an end, bootstrap from s_n = nobs[last] with gamma^n; if it ended, done=1.
        return rew, done_n, self.nobs[last], m

    def update_priorities(self, idx, td_abs):
        eps = np.where(self.is_demo[idx], self.eps_d, self.eps_a)
        p = td_abs + eps
        self.pa[idx] = p ** self.alpha
        self.maxp = max(self.maxp, float(p.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True); ap.add_argument("--game", default="Breakout")
    ap.add_argument("--arm", default="DQfD", choices=["DQfD", "PDD", "NoMargin"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--pretrain", type=int, default=75000); ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--sticky", type=float, default=0.0); ap.add_argument("--out", required=True)
    ap.add_argument("--lam-e", type=float, default=1.0); ap.add_argument("--bin", type=int, default=5000)
    ap.add_argument("--eval-eps", type=int, default=50)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    td = load_td()
    z = np.load(a.demo, allow_pickle=False)
    unp = lambda p: np.unpackbits(p, axis=-1)[..., :400].astype(np.uint8)   # (N, 400) HWC order as stored; reshaped below
    # the stored order is (H, W, C) -> convert to (C, H, W) flat
    to_chw = lambda x: x.reshape(-1, 10, 10, 4).transpose(0, 3, 1, 2).reshape(-1, 400)
    demo = dict(obs=to_chw(unp(z["obs"])), nobs=to_chw(unp(z["nobs"])), act=z["act"].astype(np.int64), rew=z["rew"].astype(np.float32),
                term=z["term"].astype(bool), end=z["end"].astype(bool))
    env = StickyEnv(a.game, seed=1000 + a.seed, sticky=a.sticky)
    nA = env.n_actions
    use_demo = a.arm != "PDD"
    lam_e = 0.0 if a.arm == "PDD" else (0.0 if a.arm == "NoMargin" else a.lam_e)
    net, tnet = Net(nA, 4), Net(nA, 4)
    tnet.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, eps=1.5e-4)
    G, NSTEP, B, TAU, L2 = 0.99, 10, 32, 10000, 1e-5
    R = Replay(demo, 100000, NSTEP, G, 0.4, 1.0 if use_demo else 0.0, 0.001, use_demo, 400)
    if not use_demo:
        R.maxp = 1.0
    t_f = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float()
    cnt = [0]

    def update(beta):
        idx, w = R.sample(B, rng, beta)
        rew, done_n, sn, m = R.windows(idx)
        s = t_f(R.obs[idx]).view(B, 4, 10, 10); s1 = t_f(R.nobs[idx]).view(B, 4, 10, 10); snt = t_f(sn).view(B, 4, 10, 10)
        act = torch.from_numpy(R.act[idx]); r1 = R.rew[idx]; done1 = R.term[idx].astype(np.float32) + (R.end[idx] & ~R.term[idx]).astype(np.float32)
        x_next = torch.cat([s1, snt], 0)
        with torch.no_grad():
            qt = tnet(x_next); qo = net(x_next)
        q = net(s)
        data = td.dqfd_nstep_td_data(q, qt[B:], act, qo[B:].argmax(1), torch.from_numpy(rew), torch.from_numpy(done_n),
                                     torch.from_numpy(done1), torch.from_numpy(w), qt[:B], qo[:B].argmax(1),
                                     torch.from_numpy(R.is_demo[idx].astype(np.float32)))
        loss, _, _ = td.dqfd_nstep_td_error(data, G, lambda_n_step_td=1.0, lambda_supervised_loss=lam_e, margin_function=0.8,
                                            lambda_one_step_td=1.0, nstep=NSTEP)
        l2 = 0.5 * sum((p ** 2).sum() for p in net.parameters())
        opt.zero_grad(); (loss + L2 * l2).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        with torch.no_grad():
            y1 = torch.from_numpy(r1) + G * (1 - torch.from_numpy(done1)) * qt[:B].gather(1, qo[:B].argmax(1, keepdim=True))[:, 0]
            td1 = (y1 - q.detach().gather(1, act[:, None])[:, 0]).abs().numpy()
        R.update_priorities(idx, td1)
        cnt[0] += 1
        if cnt[0] % TAU == 0:
            tnet.load_state_dict(net.state_dict())
        return float(loss), float(R.is_demo[idx].mean())

    log = dict(args=vars(a), torch=torch.__version__, pretrain=[], bins=[], notes={})
    t0 = time.time()
    if use_demo:
        for u in range(1, a.pretrain + 1):
            l, ds = update(0.6)
            if u % 5000 == 0:
                print(f"[METRIC] pretrain u={u} loss={l:.4f} {u / (time.time() - t0):.0f} upd/s", flush=True)
        log["pretrain_secs"] = time.time() - t0
    # ---- interaction
    def greedy_eval(n, eps=0.0):
        e2 = StickyEnv(a.game, seed=555 + a.seed, sticky=a.sticky)
        rets = []
        for _ in range(n):
            o = e2.reset(); tot = 0.0
            while True:
                with torch.no_grad():
                    q = net(t_f(o).view(1, 4, 10, 10))
                act_ = int(q.argmax()) if rng.random() >= eps else int(rng.integers(nA))
                o, r, d, _ = e2.step(act_); tot += r
                if d: break
            rets.append(tot)
        return rets
    if use_demo:
        log["greedy_after_pretrain"] = float(np.mean(greedy_eval(20)))
        print(f"[METRIC] greedy after pretrain {log['greedy_after_pretrain']:.3f}", flush=True)
    o = env.reset(); ep_ret, eps_log, ncap = 0.0, [], 0
    t1 = time.time()
    for t in range(a.steps):
        if a.arm == "PDD":
            eps = 1.0 + (a.eps - 1.0) * min(1.0, t / 100000)
        else:
            eps = a.eps
        if rng.random() < eps:
            act_ = int(rng.integers(nA))
        else:
            with torch.no_grad():
                act_ = int(net(t_f(o).view(1, 4, 10, 10)).argmax())
        o2, r, done, capped = env.step(act_)
        ncap += int(capped)
        # reward transform rho(r) = sign(r) log(1+|r|); `end`=done (cap treated as termination); term=done
        R.add(o.reshape(-1), o2.reshape(-1), act_, np.sign(r) * np.log1p(abs(r)), done, done)
        ep_ret += r
        if done:
            eps_log.append((t, ep_ret)); ep_ret = 0.0; o = env.reset()
        else:
            o = o2
        if a.arm != "PDD" or t >= 5000:
            update(0.6 + 0.4 * t / a.steps)
        if (t + 1) % a.bin == 0:
            lo = t + 1 - a.bin
            xs = [r_ for (tt, r_) in eps_log if tt >= lo]
            log["bins"].append(dict(step=t + 1, mean=float(np.mean(xs)) if xs else None, n=len(xs)))
            if (t + 1) % (a.bin * 4) == 0:
                print(f"[METRIC] t={t + 1} train_return_last_bin={log['bins'][-1]['mean']} eps={eps:.3f} {(t + 1) / (time.time() - t1):.0f} step/s", flush=True)
    log["episodes"] = eps_log; log["n_capped_episodes"] = ncap
    log["interact_secs"] = time.time() - t1
    log["greedy_final_eps0"] = greedy_eval(a.eval_eps)
    log["greedy_final_mean"] = float(np.mean(log["greedy_final_eps0"]))
    print(f"[METRIC] final greedy {log['greedy_final_mean']:.3f} capped_eps={ncap}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(log, open(a.out, "w"))


if __name__ == "__main__":
    main()

"""No-ObGD construction in the authors' loop: the AUTHORS' stream_ac_continuous.main (407dca7, unmodified loop, wrappers, networks, SparseInit, LN,
normalizers) with ONLY the optimizer swapped from ObGD to our B3 no-ObGD construction written in torch:
the eligibility trace exactly as ObGD keeps it (e <- g*lam*e + grad of their -v / -log pi - tau H sign(delta)), then
torch.optim.Adam(lr, eps=1e-4) on the pseudo-gradient delta*e (their ObGD step is p -= step*delta*e), trace reset
after the update. Question: does Adam at 1e-7 learn in THEIR framework as it does in our JAX one?
Usage: run_official_trace_adam.py <seed> <total_steps> <lr>"""
import sys, os
import torch
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' clone + venv (scripts/official/setup.sh)
sys.path.insert(0, os.path.join(OFFICIAL_ROOT, "repo"))
import stream_ac_continuous as sac


class TraceAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-7, gamma=0.99, lamda=0.8, kappa=None, eps=1e-4):
        params = list(params)
        super().__init__(params, dict(lr=lr, gamma=gamma, lamda=lamda))
        self.adam = torch.optim.Adam(params, lr=lr, eps=eps)

    def step(self, delta, reset=False):
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state[p]
                if len(st) == 0:
                    st["eligibility_trace"] = torch.zeros_like(p.data)
                e = st["eligibility_trace"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                p.grad.copy_(delta * e)
        self.adam.step()
        if reset:
            for group in self.param_groups:
                for p in group["params"]:
                    self.state[p]["eligibility_trace"].zero_()


sac.Optimizer = TraceAdam
seed, T, lr = int(sys.argv[1]), int(float(sys.argv[2])), float(sys.argv[3])
torch.set_num_threads(1)
sac.main("Hopper-v4", seed, lr, 0.99, 0.8, T, 0.01, 3.0, 2.0, False, False)

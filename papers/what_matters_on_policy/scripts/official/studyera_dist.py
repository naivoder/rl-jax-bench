# Study-era action distribution: an adapter that gives the HEAD seed_rl modules (which call
# dist(params).sample /.log_prob(a) /.entropy) the semantics of seed_rl@68c15aa's NormalSquashedDistribution,
# i.e. the "postprocessing" design that was in force when the study ran (its rows' gin keys are
# NormalTanhDistribution.*, a class renamed on 2020-09-25 and removed with postprocessing on 2020-12-01):
#   * the stored action is the PRE-squash Gaussian sample x;
#   * log_prob(x) = log N(x; mu, sigma) - log|d post(x)/dx|  (post = SafeTanh: the tanh log-det; ClippedIdentity: 0);
#   * entropy = H(N) + log-det at ONE sample;
#   * the env receives post(x) = tanh(x) or clip(x, -1, 1).
# All numerics are the 68c15aa module's own code (scripts/official/pd_68c15aa.py, verbatim); this file only adapts the call
# signature.
import os
import sys

# pd_68c15aa.py is the authors' file, extracted into $OFFICIAL_ROOT (see README, Checked against)
sys.path.insert(0, os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official")))
import pd_68c15aa as OLD  # noqa: E402


class _Dist:
    def __init__(self, old, params):
        self.old, self.p = old, params

    def sample(self, seed=None):
        return self.old.sample(self.p)

    def log_prob(self, a):
        return self.old.log_prob(self.p, a)

    def entropy(self, seed=None):
        return self.old.entropy(self.p)


class StudyEraDist:
    def __init__(self, num_actions, tanh, softplus, init_std, min_std=1e-3):
        std_fn = (OLD.softplus_std_fn if softplus else OLD.safe_exp_std_fn)(init_std, min_std)
        post = OLD.SafeTanh() if tanh else OLD.ClippedIdentity()
        self.old = OLD.NormalSquashedDistribution(num_actions, OLD.ContinuousDistributionConfig(std_fn, post))
        self.std_fn = std_fn

    @property
    def param_size(self):
        return self.old.param_size

    @property
    def reparametrizable(self):
        return True

    def __call__(self, params):
        return _Dist(self.old, params)

    def postprocess(self, x):
        return self.old.postprocess(x)


def _kl(self, other):
    return self.old.kl_divergence(self.p, other.p)


_Dist.kl_divergence = _kl

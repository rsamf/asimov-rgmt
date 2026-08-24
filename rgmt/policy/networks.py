import torch
from torch import nn, Tensor

class RunningMeanStd(nn.Module):
    def __init__(self, shape, epsilon: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon))

    @torch.no_grad()
    def update(self, x: Tensor):
        x = x.reshape(-1, *self.mean.shape)
        b_mean, b_var, b_count = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean += delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        self.var = (m_a + m_b + delta**2 * self.count * b_count / tot) / tot
        self.count = tot

    def normalize(self, x: Tensor) -> Tensor:
        # Variance floor: constant/near-dead channels (e.g. the pelvis
        # keypoint's root-relative position in critic_obs, which is zero by
        # construction) otherwise collapse to var ~1e-12 and amplify float
        # noise x10^4 into the network, the root cause of episodic
        # advantage/KL shocks seen repeatedly in large-batch runs. A 1e-2 std
        # floor caps amplification at x100 of the raw signal.
        return (x - self.mean) / torch.sqrt(self.var.clamp(min=1e-4) + 1e-8)


from dataclasses import dataclass
from rgmt.policy.encoders import HistoryEncoder, CommandEncoder

@dataclass
class PolicyDims:
    priv_dim: int
    obs_dim: int = 98
    cmd_dim: int = 55
    act_dim: int = 23
    n_embd: int = 128
    hist_len: int = 10
    cmd_len: int = 21
    n_heads: int = 4

def _mlp(sizes, act=nn.ELU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)

class RGMTActorCritic(nn.Module):
    def __init__(self, dims: PolicyDims, actor_hidden=(512, 256), critic_hidden=(512, 256)):
        super().__init__()
        self.dims = dims
        self.history_enc = HistoryEncoder(dims.obs_dim, dims.n_embd, n_heads=dims.n_heads)
        self.command_enc = CommandEncoder(dims.cmd_dim, dims.n_embd, n_heads=dims.n_heads)
        actor_in = dims.obs_dim + 2 * dims.n_embd          # [o_t, h, u]
        self.actor = _mlp((actor_in, *actor_hidden, dims.act_dim))
        self.critic = _mlp((dims.priv_dim, *critic_hidden, 1))
        self.log_std = nn.Parameter(torch.zeros(dims.act_dim) - 0.5)
        self.obs_rms = RunningMeanStd((dims.obs_dim,))
        self.cmd_rms = RunningMeanStd((dims.cmd_dim,))
        self.critic_rms = RunningMeanStd((dims.priv_dim,))

    def _features(self, bundle, obs_noise_std: float = 0.0):
        """Actor features. ``obs_noise_std > 0`` perturbs the NORMALIZED current
        observation for the spatial-smoothness (Lipschitz) penalty.

        The perturbation must hit BOTH copies of o_t: ``step`` rolls the fresh
        post-step obs into ``history`` (track_env.py) and ``_bundle`` rebuilds the
        identical vector as ``obs``, so history[:, -1] IS obs. Noising only
        ``obs`` would let the actor route its current-state dependence through the
        history encoder and make the penalty measure nothing. One noise draw is
        shared by both copies — a single consistently-perturbed sensor reading.
        The other 9 history rows and cmd_window stay clean (see the plan: we bound
        sensitivity to proprioception, never to the reference command).

        Noise lives only in this graph: RunningMeanStd.normalize is pure and
        update_rms() runs separately on the raw rollout tensors, so the normalizer
        statistics, the rollout, and the critic can never see it.
        """
        hist = self.obs_rms.normalize(bundle["history"])
        cmd = self.cmd_rms.normalize(bundle["cmd_window"])
        o = self.obs_rms.normalize(bundle["obs"])
        if obs_noise_std > 0.0:
            eps = torch.randn_like(o) * obs_noise_std
            o = o + eps
            hist = torch.cat([hist[:, :-1], (hist[:, -1] + eps).unsqueeze(1)], dim=1)
        h = self.history_enc(hist)
        u = self.command_enc(h, cmd)
        return torch.cat([o, h, u], dim=-1)

    def _dist(self, bundle):
        mean = self.actor(self._features(bundle))
        # Clamp log_std to prevent entropy runaway (std in [~0.018, 1.0]). The
        # first full run's entropy grew unbounded (21 -> 127) as the policy
        # collected the entropy bonus once returns plateaued.
        std = self.log_std.clamp(-4.0, 0.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, bundle):
        return self.critic(self.critic_rms.normalize(bundle["critic_obs"])).squeeze(-1)

    def act(self, bundle, return_mean: bool = False):
        """Sample an action. ``return_mean`` additionally returns the distribution
        mean (mu) so the rollout can store tanh(mu) for the temporal-smoothness
        penalty. dist.sample() is called identically either way, so the RNG stream
        is unchanged when the flag is off."""
        dist = self._dist(bundle)
        a = dist.sample()
        out = (a, dist.log_prob(a).sum(-1), self.value(bundle))
        return (*out, dist.mean) if return_mean else out

    def act_inference(self, bundle):
        return self.actor(self._features(bundle))

    def actor_mean(self, bundle, obs_noise_std: float = 0.0):
        """Action mean, optionally under a perturbed observation. Used for the
        second (noisy) forward pass of the spatial-smoothness penalty."""
        return self.actor(self._features(bundle, obs_noise_std))

    def evaluate(self, bundle, action, return_mean: bool = False):
        """``return_mean`` also returns mu — free, the dist is already built."""
        dist = self._dist(bundle)
        out = (dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.value(bundle))
        return (*out, dist.mean) if return_mean else out

    @torch.no_grad()
    def update_rms(self, bundle):
        self.obs_rms.update(bundle["obs"])
        self.obs_rms.update(bundle["history"])
        self.cmd_rms.update(bundle["cmd_window"])
        self.critic_rms.update(bundle["critic_obs"])

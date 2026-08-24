"""The actor's log_std must be clamped so entropy can't run away."""
import torch

from rgmt.policy.networks import RGMTActorCritic, PolicyDims


def _bundle(B, d):
    return dict(
        obs=torch.randn(B, d.obs_dim),
        history=torch.randn(B, d.hist_len, d.obs_dim),
        cmd_window=torch.randn(B, d.cmd_len, d.cmd_dim),
        critic_obs=torch.randn(B, d.priv_dim),
    )


def test_log_std_clamped_high_and_low():
    d = PolicyDims(priv_dim=64)
    ac = RGMTActorCritic(d).eval()
    b = _bundle(4, d)
    with torch.no_grad():
        ac.log_std.data.fill_(10.0)           # try to blow the variance up
        assert float(ac._dist(b).stddev.max()) <= 1.0 + 1e-5     # capped at exp(0)=1
        ac.log_std.data.fill_(-20.0)          # try to collapse it
        assert float(ac._dist(b).stddev.min()) >= torch.tensor(-4.0).exp() - 1e-6  # exp(-4)

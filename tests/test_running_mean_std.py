import torch
from rgmt.policy.networks import RunningMeanStd

def test_normalize_converges():
    rms = RunningMeanStd((4,))
    data = torch.randn(100000, 4) * 3 + 7
    rms.update(data)
    out = rms.normalize(data)
    assert torch.allclose(out.mean(0), torch.zeros(4), atol=0.05)
    assert torch.allclose(out.std(0), torch.ones(4), atol=0.05)


def test_variance_floor_caps_dead_channel_amplification():
    import torch
    from rgmt.policy.networks import RunningMeanStd
    rms = RunningMeanStd((4,))
    # channel 3 is constant (like the pelvis root-relative keypoint)
    x = torch.randn(10000, 4)
    x[:, 3] = 0.0
    rms.update(x)
    y = rms.normalize(x + 1e-6)  # microscopic perturbation
    # without the floor, channel 3 would blow up to ~1e2-1e4; with it, tiny
    assert float(y[:, 3].abs().max()) < 1e-2
    # live channels still normalize to ~unit scale
    assert 0.5 < float(y[:, 0].std()) < 2.0

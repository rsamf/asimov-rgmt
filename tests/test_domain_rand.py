"""Domain randomization: sampling math, config surface, URDF effort limits.

Pure CPU — the sim-facing half lives in tests/test_dr_sim.py (CUDA-gated).
"""
import pytest
import torch

from rgmt.data.joint_map import ASIMOV_ACTUATED_JOINT_NAMES
from rgmt.env.domain_rand import DomainRand, DRConfig, NOMINAL_FOOT_MU
from rgmt.env.gains import urdf_effort_limits


class _MockSim:
    def __init__(self):
        self.calls = []

    def set_foot_friction_per_env(self, mu):
        self.calls.append(("friction", mu.clone()))

    def set_body_mass_scale(self, s):
        self.calls.append(("mass", s.clone()))

    def set_joint_gain_scale(self, kp_s, kd_s):
        self.calls.append(("gains", kp_s.clone(), kd_s.clone()))

    def set_effort_limit_scale(self, s):
        self.calls.append(("effort", s.clone()))


def test_defaults_are_fully_off():
    dr = DomainRand(DRConfig(), 16, "cpu")
    dr.sample(torch.arange(16))
    assert torch.equal(dr.obs(), torch.zeros(16, DomainRand.DR_OBS_DIM))
    sim = _MockSim()
    dr.write_to_sim(sim)
    assert sim.calls == [], "disabled params must never touch the sim"


def test_urdf_effort_limits_order_and_values():
    eff = urdf_effort_limits()
    assert len(eff) == 23
    by_name = dict(zip(ASIMOV_ACTUATED_JOINT_NAMES, eff))
    assert by_name["left_ankle_roll_joint"] == 57.0
    assert by_name["left_knee_joint"] == 75.0
    assert by_name["left_ankle_pitch_joint"] == 145.0
    assert by_name["right_hip_roll_joint"] == 90.0
    # left/right symmetric
    for n, e in by_name.items():
        if n.startswith("left_"):
            assert by_name["right_" + n[len("left_"):]] == e


def test_ranges_statistical_band():
    torch.manual_seed(0)
    cfg = DRConfig(friction_range=[0.4, 1.0], mass_scale_range=[0.9, 1.1],
                   kp_scale_range=[0.9, 1.1], kd_scale_range=[0.9, 1.1],
                   effort_scale_range=[0.8, 1.2])
    dr = DomainRand(cfg, 1000, "cpu")
    dr.sample(torch.arange(1000))
    for attr, lo, hi in (("mu", 0.4, 1.0), ("mass_scale", 0.9, 1.1),
                         ("kp_scale", 0.9, 1.1), ("kd_scale", 0.9, 1.1),
                         ("effort_scale", 0.8, 1.2)):
        v = getattr(dr, attr)
        assert float(v.min()) >= lo and float(v.max()) <= hi
        mid = 0.5 * (lo + hi)
        band = 0.05 * (hi - lo)          # N=1000 -> mean well inside +-5% of range
        assert abs(float(v.mean()) - mid) < band, attr


def test_sample_touches_only_requested_envs():
    torch.manual_seed(0)
    dr = DomainRand(DRConfig(friction_range=[0.4, 1.0]), 8, "cpu")
    dr.sample(torch.arange(8))
    before = dr.mu.clone()
    dr.sample(torch.tensor([2, 5]))
    changed = dr.mu != before
    assert bool(changed[2]) or bool(changed[5])   # astronomically unlikely to tie
    untouched = torch.tensor([0, 1, 3, 4, 6, 7])
    assert torch.equal(dr.mu[untouched], before[untouched])


def test_obs_normalization_and_layout():
    cfg = DRConfig(friction_range=[0.4, 1.0], effort_scale_range=[0.8, 1.2])
    dr = DomainRand(cfg, 4, "cpu")
    # force known draws
    dr.mu[:] = torch.tensor([0.4, 0.7, 1.0, 0.7])
    dr.effort_scale[:] = torch.tensor([0.8, 1.0, 1.2, 1.0])
    o = dr.obs()
    assert o.shape == (4, DomainRand.DR_OBS_DIM)
    assert torch.allclose(o[:, 0], torch.tensor([-1.0, 0.0, 1.0, 0.0]))   # mu col
    assert torch.allclose(o[:, 5], torch.tensor([-1.0, 0.0, 1.0, 0.0]))   # effort col
    # disabled columns exactly zero
    assert torch.equal(o[:, 1:5], torch.zeros(4, 4))
    eps = 1e-5   # float32 rounding at exact range edges
    assert float(o.min()) >= -1.0 - eps and float(o.max()) <= 1.0 + eps


def test_config_validation():
    with pytest.raises(ValueError):
        DRConfig(friction_range=[1.0, 0.4])          # lo > hi
    with pytest.raises(ValueError):
        DRConfig(mass_scale_range=[-0.1, 1.1])       # non-positive scale
    with pytest.raises(NotImplementedError):
        DRConfig(payload_mass_range=[0.0, 2.0])      # reserved knob

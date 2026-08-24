"""Unit tests for the instability/fallen criteria (paper §II-D, lax thresholds)."""
import torch

from rgmt.env.track_env import compute_fallen

# New lax thresholds: pelvis sits ~0.08 on the ground, so <0.12 = truly collapsed;
# up_z<0 = tilted past horizontal; head <0.3 = head abnormally low.
Z_FALL, UP_MIN, HEAD_MIN = 0.12, 0.0, 0.3


def test_upright_not_fallen():
    f = compute_fallen(torch.tensor([0.6]), torch.tensor([1.0]), torch.tensor([1.1]),
                       z_fall=Z_FALL, up_dot_min=UP_MIN, head_z_min=HEAD_MIN)
    assert not bool(f[0])


def test_each_criterion_triggers():
    base_z = torch.tensor([0.08, 0.60, 0.60])   # pelvis on ground / ok / ok
    up_z = torch.tensor([1.00, -0.10, 1.00])    # ok / past horizontal / ok
    head_z = torch.tensor([1.10, 1.10, 0.20])   # ok / ok / head on ground
    f = compute_fallen(base_z, up_z, head_z, z_fall=Z_FALL, up_dot_min=UP_MIN,
                       head_z_min=HEAD_MIN)
    assert f.tolist() == [True, True, True]


def test_lax_allows_deep_crouch_and_tilt():
    # Deep crouch (z=0.20) with a 60deg tilt (up_z=0.30): FALLEN under the old
    # (0.30 / 0.5) thresholds, NOT fallen under the new lax ones.
    f = compute_fallen(torch.tensor([0.20]), torch.tensor([0.30]), torch.tensor([0.6]),
                       z_fall=Z_FALL, up_dot_min=UP_MIN, head_z_min=HEAD_MIN)
    assert not bool(f[0])


def test_head_none_skips_head_term():
    f = compute_fallen(torch.tensor([0.6]), torch.tensor([1.0]), None,
                       z_fall=Z_FALL, up_dot_min=UP_MIN, head_z_min=HEAD_MIN)
    assert not bool(f[0])

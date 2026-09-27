import math
import torch
from locomotion.geom import (wrap_angle, rot2, rotate2, rot3z, quat_from_yaw, quat_to_rotvec,
                             rotvec_to_quat, quat_mul, yaw_from_quat)
from locomotion.trajectory import (halflife_to_damping, decay_spring_damper_exact,
                                   spring_velocity_update)


def test_wrap_angle():
    assert abs(wrap_angle(math.pi + 0.1) - (-math.pi + 0.1)) < 1e-6
    t = wrap_angle(torch.tensor([3 * math.pi, -3 * math.pi, 0.5]))
    assert abs(abs(float(t[0])) - math.pi) < 1e-5 and abs(abs(float(t[1])) - math.pi) < 1e-5
    assert abs(float(t[2]) - 0.5) < 1e-6


def test_rot2_and_rotate2():
    v = torch.tensor([1.0, 0.0])
    out = rotate2(v, math.pi / 2)
    assert torch.allclose(out, torch.tensor([0.0, 1.0]), atol=1e-6)      # ccw positive
    R = rot2(torch.tensor([0.3, -0.3]))
    assert R.shape == (2, 2, 2)
    assert torch.allclose(R[0] @ R[1], torch.eye(2), atol=1e-6)
    # batched vectors with one yaw, and per-row yaws
    vs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert torch.allclose(rotate2(vs, math.pi / 2), torch.tensor([[0.0, 1.0], [-1.0, 0.0]]), atol=1e-6)
    assert torch.allclose(rotate2(vs, torch.tensor([0.0, math.pi])), torch.tensor([[1.0, 0.0], [0.0, -1.0]]), atol=1e-6)


def test_rot3z_matches_rot2():
    R3 = rot3z(0.7); R2 = rot2(0.7)
    assert torch.allclose(R3[:2, :2], R2, atol=1e-6) and R3[2, 2] == 1.0


def test_quat_from_yaw_roundtrip():
    yaw = torch.tensor([0.0, 1.0, -2.5])
    q = quat_from_yaw(yaw)
    assert q.shape == (3, 4)
    assert torch.allclose(yaw_from_quat(q), yaw, atol=1e-6)
    q2 = quat_mul(quat_from_yaw(torch.tensor(0.4)), quat_from_yaw(torch.tensor(0.5)))
    assert abs(float(yaw_from_quat(q2)) - 0.9) < 1e-6


def test_rotvec_roundtrip_and_small_angle():
    r = torch.tensor([[0.3, -0.2, 0.1], [1e-9, 0.0, 0.0], [0.0, 0.0, 2.0]])
    q = rotvec_to_quat(r)
    assert torch.allclose(q.norm(dim=-1), torch.ones(3), atol=1e-6)
    assert torch.allclose(quat_to_rotvec(q), r, atol=1e-5)
    assert torch.allclose(quat_to_rotvec(-q), r, atol=1e-5)   # -q is the same rotation


def test_decay_spring_reaches_zero():
    x, v = torch.tensor([1.0, -2.0]), torch.tensor([0.0, 3.0])
    for _ in range(60 * 2):
        x, v = decay_spring_damper_exact(x, v, 0.1, 1 / 60)
    assert x.abs().max() < 1e-4 and v.abs().max() < 1e-3


def test_decay_spring_halflife_meaning():
    x, v = torch.tensor([1.0]), torch.tensor([0.0])
    x, v = decay_spring_damper_exact(x, v, 0.2, 0.2)
    assert 0.3 < float(x) < 0.7


def test_spring_velocity_converges_and_is_time_consistent():
    hl, dt = 0.2, 1 / 60
    x = torch.zeros(2); v = torch.zeros(2); a = torch.zeros(2)
    v_des = torch.tensor([1.0, 0.5])
    for _ in range(int(10 * hl / dt)):
        x, v, a = spring_velocity_update(x, v, a, v_des, hl, dt)
    assert torch.allclose(v, v_des, atol=1e-3)
    xs, vs, _ = spring_velocity_update(torch.zeros(2), torch.zeros(2), torch.zeros(2), v_des, hl, 1.0)
    x2 = torch.zeros(2); v2 = torch.zeros(2); a2 = torch.zeros(2)
    for _ in range(60):
        x2, v2, a2 = spring_velocity_update(x2, v2, a2, v_des, hl, dt)
    assert torch.allclose(xs, x2, atol=1e-4) and torch.allclose(vs, v2, atol=1e-4)


def test_halflife_to_damping():
    assert abs(halflife_to_damping(1.0) - 4 * math.log(2) / (1.0 + 1e-5)) < 1e-9

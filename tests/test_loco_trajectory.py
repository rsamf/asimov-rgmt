import math
import torch
from locomotion.config import SpringConfig
from locomotion.trajectory import TrajectoryModel


def _model():
    return TrajectoryModel(SpringConfig(velocity_halflife=0.2, rotation_halflife=0.1), fps=60,
                           horizons=[20, 40, 60])


def test_zero_command_from_rest_predicts_here():
    m = _model()
    pos, d = m.predict(0.7, torch.zeros(2), 0.0)
    assert pos.shape == (3, 2) and d.shape == (3, 2)
    assert torch.allclose(pos, torch.zeros(3, 2), atol=1e-7)
    assert torch.allclose(d, torch.tensor([[1.0, 0.0]] * 3), atol=1e-7)


def test_forward_command_predicts_increasing_heading_x():
    m = _model()
    yaw0 = math.pi / 2                                   # facing +y in the world
    v_des = torch.tensor([0.0, 1.0])                     # world-frame forward for that facing
    pos, d = m.predict(yaw0, v_des, 0.0)
    assert pos[0, 0] > 0 and pos[1, 0] > pos[0, 0] and pos[2, 0] > pos[1, 0]
    assert pos[:, 1].abs().max() < 1e-5
    assert torch.allclose(d, torch.tensor([[1.0, 0.0]] * 3), atol=1e-6)


def test_yaw_command_rotates_facings():
    m = _model()
    _, d = m.predict(0.0, torch.zeros(2), 1.0)
    ang = torch.atan2(d[:, 1], d[:, 0])
    assert ang[0] > 0 and ang[1] > ang[0] and ang[2] > ang[1]
    _, d2 = m.predict(0.0, torch.zeros(2), -1.0)
    assert torch.atan2(d2[0, 1], d2[0, 0]) < 0


def test_step_converges_to_desired():
    m = _model()
    v_des = torch.tensor([1.0, 0.0])
    for _ in range(120):
        m.step(v_des, 0.5)
    assert torch.allclose(m.vel, v_des, atol=1e-3)
    assert abs(m.yaw_rate - 0.5) < 1e-3
    m.reset()
    assert m.vel.abs().max() == 0 and m.yaw_rate == 0.0


def test_prediction_matches_stepping():
    m = _model()
    v_des = torch.tensor([0.8, -0.3])
    pos, _ = m.predict(0.0, v_des, 0.0)                  # horizon 60 = 1 s
    xy = torch.zeros(2)
    for _ in range(60):
        m.step(v_des, 0.0)
        xy = xy + m.vel / 60
    assert torch.allclose(pos[2], xy, atol=3e-2)


def test_predict_world_is_predict_rotated():
    m = _model()
    m.step(torch.tensor([1.0, 0.0]), 0.0)
    yaw0 = 0.5
    xy0 = torch.tensor([2.0, -1.0])
    ph, dh = m.predict(yaw0, torch.tensor([1.0, 0.0]), 0.3)
    pw, yw = m.predict_world(xy0, yaw0, torch.tensor([1.0, 0.0]), 0.3)
    c, s = math.cos(yaw0), math.sin(yaw0)
    R = torch.tensor([[c, -s], [s, c]])
    assert torch.allclose(pw, xy0 + ph @ R.T, atol=1e-6)
    assert torch.allclose(yw, yaw0 + torch.atan2(dh[:, 1], dh[:, 0]), atol=1e-6)

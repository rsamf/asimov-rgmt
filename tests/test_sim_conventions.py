"""Empirical regression tests for sim state conventions.

These exist because the freejoint qd slot order was assumed (ang, lin) for
the project's entire history while the installed Newton uses (lin, ang) —
and because the views and reset writes were swapped SELF-CONSISTENTLY, no
shape check or round-trip test could catch it. Only physics can: command a
linear velocity and demand translation, not rotation.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"
)

from rgmt.env.sim import NewtonSim
from rgmt.data.joint_map import KEYPOINT_LINKS

DEV = "cuda:0"


@pytest.fixture(scope="module")
def sim():
    return NewtonSim(1, kp=100.0, kd=5.0, control_decimation=1, dt=1.0 / 60.0,
                     foot_friction=0.75, keypoint_links=KEYPOINT_LINKS, device=DEV)


def _airborne_reset(sim, lin, ang):
    dev = torch.device(DEV)
    ids = torch.zeros(1, dtype=torch.long, device=dev)
    pos = torch.tensor([[0.0, 0.0, 1.5]], device=dev)   # airborne: no contacts
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=dev)
    hq = torch.zeros(1, 25, device=dev)
    hqd = torch.zeros(1, 25, device=dev)
    sim.reset_idx(ids, pos, quat,
                  torch.tensor([lin], device=dev),
                  torch.tensor([ang], device=dev), hq, hqd)


def test_linear_velocity_translates_not_rotates(sim):
    _airborne_reset(sim, lin=[1.0, 0.0, 0.0], ang=[0.0, 0.0, 0.0])
    p0, q0 = sim.base_pos.clone(), sim.base_quat.clone()
    for _ in range(6):
        sim.step(torch.zeros(1, 23, device=DEV))
    dx = float(sim.base_pos[0, 0] - p0[0, 0])
    dq = float((sim.base_quat - q0).abs().max())
    assert dx > 0.08, f"lin_vel=+1 x should translate ~0.1 m in 0.1 s, got {dx:.4f}"
    assert dq < 0.02, f"lin_vel must not rotate the base, quat drift {dq:.4f}"


def test_angular_velocity_rotates_not_translates(sim):
    _airborne_reset(sim, lin=[0.0, 0.0, 0.0], ang=[0.0, 0.0, 3.0])
    p0, q0 = sim.base_pos.clone(), sim.base_quat.clone()
    for _ in range(6):
        sim.step(torch.zeros(1, 23, device=DEV))
    dxy = float((sim.base_pos[0, :2] - p0[0, :2]).abs().max())
    dqz = float(sim.base_quat[0, 2] - q0[0, 2])
    assert dqz > 0.10, f"ang_vel=3 rad/s z should rotate (quat z ~+0.15), got {dqz:+.4f}"
    assert dxy < 0.02, f"ang_vel must not translate the base, xy drift {dxy:.4f}"


def test_velocity_views_read_back(sim):
    _airborne_reset(sim, lin=[2.0, 0.0, 0.0], ang=[0.0, 0.0, 5.0])
    assert torch.allclose(sim.base_lin_vel[0].cpu(), torch.tensor([2.0, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(sim.base_ang_vel[0].cpu(), torch.tensor([0.0, 0.0, 5.0]), atol=1e-5)


def test_keypoint_lin_vel_is_linear(sim):
    _airborne_reset(sim, lin=[1.0, 0.0, 0.0], ang=[0.0, 0.0, 0.0])
    sim.step(torch.zeros(1, 23, device=DEV))
    kv = sim.keypoint_lin_vel[0]
    # every keypoint on a purely translating rigid body moves at ~base velocity
    assert float(kv[:, 0].mean()) > 0.8, (
        f"keypoints of a robot translating +x at 1 m/s must report ~1 m/s x, "
        f"got mean {float(kv[:, 0].mean()):.4f}"
    )


def test_actuator_ref_compensation(sim):
    """Commanding target q settles at q for ALL joints, elbows included.

    The MJCF elbows carry ref=±0.785398; uncompensated, the servo settles at
    (target − ref) — a 45° error beyond the policy's ±0.5 rad authority.
    """
    _airborne_reset(sim, lin=[0.0, 0.0, 0.0], ang=[0.0, 0.0, 0.0])
    for _ in range(60):
        sim.step(torch.zeros(1, 23, device=DEV))
    q = sim.joint_q[0, sim.actuated_idx]
    worst = float(q.abs().max())
    assert worst < 0.15, f"joint settled {worst:.3f} rad away from commanded 0 (ref offset?)"

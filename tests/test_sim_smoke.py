import pytest, torch
torch = pytest.importorskip("torch")
try:
    import warp  # noqa
    _HAS = torch.cuda.is_available()
except Exception:
    _HAS = False
pytestmark = pytest.mark.skipif(not _HAS, reason="needs CUDA + warp")

from rgmt.env.sim import NewtonSim
from rgmt.data.joint_map import KEYPOINT_LINKS

def test_build_and_step():
    sim = NewtonSim(num_envs=2, kp=100.0, kd=5.0, control_decimation=1,
                    dt=1/60, foot_friction=0.75, keypoint_links=KEYPOINT_LINKS)
    assert sim.joint_q.shape == (2, 25)
    assert sim.actuated_idx.shape == (23,)
    assert sim.keypoint_pos.shape == (2, len(KEYPOINT_LINKS), 3)
    tgt = torch.zeros(2, 23, device="cuda:0")
    for _ in range(5):
        sim.step(tgt)
    assert torch.isfinite(sim.base_pos).all()

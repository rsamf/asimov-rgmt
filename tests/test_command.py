import torch
from rgmt.data.motion import MotionRef
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from tests.fixtures.make_synthetic_motion import make_synthetic_motion

def _load(tmp_path):
    p = make_synthetic_motion(tmp_path / "m.npz")
    return MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device="cpu")

def test_command_dim(tmp_path):
    m = _load(tmp_path)
    g = m.command_at(torch.tensor([0, 5]))
    assert g.shape == (2, 55)

def test_window_shape_and_clamp(tmp_path):
    m = _load(tmp_path)
    w = m.command_window(torch.tensor([0]), L=10)
    assert w.shape == (1, 21, 55)
    # idx 0 with L=10: past clamps to frame 0 -> first 11 rows identical command
    assert torch.allclose(w[0, 0], w[0, 10])  # both map to frame 0 command? (past clamp)

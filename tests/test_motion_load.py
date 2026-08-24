"""Tests for MotionRef loader with synthetic motion fixture."""

from rgmt.data.motion import MotionRef
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from tests.fixtures.make_synthetic_motion import make_synthetic_motion


def test_load_and_at(tmp_path):
    p = make_synthetic_motion(tmp_path / "m.npz")
    m = MotionRef.load(p, ROBOT_XML, ROBOT_URDF, physics_fps=60, src_fps=30, device="cpu")
    assert m.n_frames >= 40  # upsampled 30->60
    f = m.at([0, 1])
    assert f["joint_q"].shape == (2, 23) and f["base_quat"].shape == (2, 4)

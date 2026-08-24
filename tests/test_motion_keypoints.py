# tests/test_motion_keypoints.py
import torch
from rgmt.data.motion import MotionRef
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from tests.fixtures.make_synthetic_motion import make_synthetic_motion

def test_keypoints_shape(tmp_path):
    p = make_synthetic_motion(tmp_path / "m.npz")
    m = MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device="cpu", keypoint_links=KEYPOINT_LINKS)
    pos, vel = m.keypoints_at(torch.tensor([0, 3]))
    assert pos.shape == (2, len(KEYPOINT_LINKS), 3)
    assert vel.shape == (2, len(KEYPOINT_LINKS), 3)

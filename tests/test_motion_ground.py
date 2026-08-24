"""Ground-normalization: shift each clip so the lowest foot sole rests at z=0."""
import pytest
import torch

from rgmt.data.motion import MotionRef, _FOOT_SOLE_OFFSET
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from tests.fixtures.make_synthetic_motion import make_synthetic_motion

_FOOT = [i for i, n in enumerate(KEYPOINT_LINKS) if "ankle_roll" in n]


def test_ground_rests_lowest_sole_at_floor(tmp_path):
    p = make_synthetic_motion(tmp_path / "m.npz", n_frames=40)
    ung = MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device="cpu",
                         keypoint_links=KEYPOINT_LINKS, ground=False)
    grd = MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device="cpu",
                         keypoint_links=KEYPOINT_LINKS, ground=True)
    # After grounding, the lowest ankle_roll link sits exactly one sole-offset
    # above the floor -> the sole rests at z=0.
    min_foot_grd = float(grd._kp_pos_world[:, _FOOT, 2].min())
    assert abs(min_foot_grd - _FOOT_SOLE_OFFSET) < 1e-4
    # base and keypoints are shifted by the same constant world-z translation.
    shift = float(ung._kp_pos_world[:, _FOOT, 2].min()) - min_foot_grd
    assert torch.allclose(ung.base_pos[:, 2] - shift, grd.base_pos[:, 2], atol=1e-4)


def test_ground_requires_keypoints(tmp_path):
    p = make_synthetic_motion(tmp_path / "m.npz", n_frames=20)
    with pytest.raises(ValueError):
        MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device="cpu",
                       keypoint_links=None, ground=True)

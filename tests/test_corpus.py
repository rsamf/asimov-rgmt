# tests/test_corpus.py
import pytest
import torch
from rgmt.data.motion import MotionRef
from rgmt.data.corpus import MotionCorpus
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from tests.fixtures.make_synthetic_motion import make_synthetic_clips

def _corpus(tmp_path):
    paths = make_synthetic_clips(tmp_path, [("a", 20), ("b", 35), ("c", 12)])
    clips = [MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device="cpu", keypoint_links=KEYPOINT_LINKS)
             for p in paths]
    return MotionCorpus.from_clips(clips, ["a", "b", "c"], KEYPOINT_LINKS, output_device="cpu")

def test_concat_and_boundaries(tmp_path):
    c = _corpus(tmp_path)
    # each 30fps clip upsamples to (n-1)*2+1: 39, 69, 23 -> total 131
    assert c.n_clips == 3
    assert c.n_frames == 39 + 69 + 23
    assert c.clip_start.tolist() == [0, 39, 39 + 69]
    assert c.clip_end.tolist() == [38, 38 + 69, c.n_frames - 1]

def test_command_window_does_not_cross_clip(tmp_path):
    c = _corpus(tmp_path)
    end0 = int(c.clip_end[0])             # last frame of clip a
    idx = torch.tensor([end0 - 2])
    w = c.command_window(idx, L=5)        # offsets -5..+5 -> would reach end0+3 (into clip b)
    assert w.shape == (1, 11, 55)
    # frames beyond end0 must clamp to end0 (clip a), never equal clip b's start (end0+1)
    last = c.command_at(torch.tensor([end0]))[0]
    nxt = c.command_at(torch.tensor([end0 + 1]))[0]   # clip b start
    assert torch.allclose(w[0, -1], last, atol=1e-6)
    assert not torch.allclose(w[0, -1], nxt, atol=1e-6)

def test_sample_index_in_clip_with_room(tmp_path):
    c = _corpus(tmp_path)
    idx = c.sample_index(500, max_lookahead=4)
    ce = c.clip_end_of(idx)
    assert torch.all(ce - idx >= 4)      # room within owning clip
    assert torch.all(idx >= 0) and torch.all(idx < c.n_frames)

def test_from_clips_raises_without_keypoints(tmp_path):
    p = make_synthetic_clips(tmp_path, [("a", 20)])[0]
    ref = MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device="cpu")  # NO keypoint_links
    with pytest.raises(ValueError):
        MotionCorpus.from_clips([ref], ["a"], KEYPOINT_LINKS, output_device="cpu")

def test_from_clips_raises_on_fps_mismatch(tmp_path):
    p = make_synthetic_clips(tmp_path, [("a", 20), ("b", 20)])
    c0 = MotionRef.load(p[0], ROBOT_XML, ROBOT_URDF, device="cpu", physics_fps=60, src_fps=30, keypoint_links=KEYPOINT_LINKS)
    c1 = MotionRef.load(p[1], ROBOT_XML, ROBOT_URDF, device="cpu", physics_fps=120, src_fps=30, keypoint_links=KEYPOINT_LINKS)
    with pytest.raises(ValueError):
        MotionCorpus.from_clips([c0, c1], ["a", "b"], KEYPOINT_LINKS, output_device="cpu")

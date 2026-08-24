"""Tests for clip-weighted RSI sampling (N2: failure-weighted hard-example mining).

Pure corpus-level tests — no sim, CPU-safe.
"""

import pytest
import torch

from rgmt.data.corpus import MotionCorpus
from rgmt.data.motion import MotionRef
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from tests.fixtures.make_synthetic_motion import make_synthetic_motion


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    clips, names = [], []
    for name, n_frames in [("clip_a", 120), ("clip_b", 120)]:
        p = make_synthetic_motion(d / f"{name}.npz", n_frames=n_frames)
        clips.append(MotionRef.load(
            p, ROBOT_XML, ROBOT_URDF, device="cpu", keypoint_links=KEYPOINT_LINKS))
        names.append(name)
    return MotionCorpus.from_clips(clips, names, KEYPOINT_LINKS, output_device="cpu")


def _clip_share(corpus, idx, clip_id):
    return float((corpus.frame_clip_id[idx] == clip_id).float().mean())


def test_uniform_default_unchanged(corpus):
    torch.manual_seed(0)
    idx = corpus.sample_index(20000, max_lookahead=11)
    share_b = _clip_share(corpus, idx, 1)
    assert 0.45 < share_b < 0.55, f"uniform sampling skewed: clip_b share {share_b:.3f}"


def test_weighted_sampling_shifts_distribution(corpus):
    stats = corpus.set_clip_sampling_weights({"clip_b": 9.0})
    assert stats["matched"] == 1 and stats["total_clips"] == 2
    assert abs(stats["boosted_mass_share"] - 0.9) < 1e-5
    torch.manual_seed(0)
    idx = corpus.sample_index(20000, max_lookahead=11)
    share_b = _clip_share(corpus, idx, 1)
    assert 0.85 < share_b < 0.95, f"expected ~0.9 clip_b share, got {share_b:.3f}"
    # lookahead validity must hold for weighted draws too
    room = corpus.clip_end[corpus.frame_clip_id[idx]] - idx
    assert (room >= 11).all()
    corpus._frame_sample_weights = None  # restore for other tests


def test_unknown_clip_name_raises(corpus):
    with pytest.raises(ValueError, match="not in corpus"):
        corpus.set_clip_sampling_weights({"no_such_clip": 4.0})
    assert corpus._frame_sample_weights is None

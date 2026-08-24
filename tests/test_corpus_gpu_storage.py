"""GPU-resident corpus storage must be value-identical to CPU storage."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from rgmt.data.motion import MotionRef
from rgmt.data.corpus import MotionCorpus
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from rgmt.env.track_env import _clip_aware_grid
from tests.fixtures.make_synthetic_motion import make_synthetic_clips


def _corpus(tmp_path, storage):
    paths = make_synthetic_clips(tmp_path, [("a", 20), ("b", 35)])
    clips = [MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device="cpu",
                            keypoint_links=KEYPOINT_LINKS) for p in paths]
    c = MotionCorpus.from_clips(clips, ["a", "b"], KEYPOINT_LINKS, output_device="cuda:0")
    return c.to_storage(storage)


def test_gpu_storage_matches_cpu(tmp_path):
    cpu = _corpus(tmp_path / "c", "cpu")
    gpu = _corpus(tmp_path / "g", "cuda:0")
    assert gpu.storage_device.type == "cuda"
    idx = torch.tensor([0, 5, 38, 40, cpu.n_frames - 1], device="cuda:0")
    # accessors agree in value and land on the output device
    for k, v in cpu.at(idx).items():
        assert torch.allclose(v, gpu.at(idx)[k], atol=1e-6), k
        assert gpu.at(idx)[k].device.type == "cuda"
    assert torch.allclose(cpu.command_window(idx, 10), gpu.command_window(idx, 10), atol=1e-6)
    assert torch.equal(cpu.clip_end_of(idx), gpu.clip_end_of(idx))
    p_c, v_c = cpu.keypoints_at(idx)
    p_g, v_g = gpu.keypoints_at(idx)
    assert torch.allclose(p_c, p_g, atol=1e-6) and torch.allclose(v_c, v_g, atol=1e-6)
    # clip-aware grid stays within owning clip on GPU storage
    end0 = int(gpu.clip_end[0])
    grid = _clip_aware_grid(torch.tensor([end0 - 2], device="cuda:0"), 10, gpu)
    assert grid.device.type == "cuda"
    assert int(grid.max()) <= end0
    # sample_index respects lookahead on GPU storage
    s = gpu.sample_index(200, 4)
    assert torch.all(gpu.clip_end_of(s) - s >= 4)

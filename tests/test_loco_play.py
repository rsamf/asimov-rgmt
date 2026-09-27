import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from rgmt.preprocess import run_preprocess
from locomotion.corpus import ClipsMeta
from locomotion.play import main

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton sim needs CUDA")


def _npz(path, vx, vy, n=150):
    t = np.arange(n) / 30.0
    np.savez(path, base_frame_pos=np.stack([vx * t, vy * t, 0.7 + 0 * t], 1).astype(np.float32),
             base_frame_wxyz=np.tile(np.array([1.0, 0, 0, 0], np.float32), (n, 1)),
             joint_angles=(0.2 * np.sin(2 * np.pi * 1.4 * t)[:, None] * np.ones((1, 23))).astype(np.float32))


def test_headless_scripted_run(tmp_path):
    npz = tmp_path / "npz"; npz.mkdir()
    _npz(npz / "fwd.npz", 1.0, 0.0); _npz(npz / "left.npz", 0.0, 0.5); _npz(npz / "idle.npz", 0.0, 0.0)
    run_preprocess(OmegaConf.create({"motion_dir": str(npz), "motion_path": None, "cache_dir": str(tmp_path / "cache"),
                                     "physics_fps": 60, "src_fps": 30, "keypoint_links": None,
                                     "ground_clips": True, "force": False}))
    ClipsMeta(classes={"fwd": "walk_fwd", "left": "strafe", "idle": "idle"},
              speeds={"forward": 1.0, "backward": 0.5, "lateral": 0.5, "turn_rate": 1.0}).save(tmp_path / "clips.json")
    steps = main(["--corpus", str(tmp_path), "--viewer", "null", "--script", "w:1,a:0.5,q:0.5",
                  "--seconds", "1.5", "--set", "corpus.mirror=false"])
    assert steps == 90

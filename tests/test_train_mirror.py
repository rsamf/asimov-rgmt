"""mirror_augment wiring: corpus build, split expansion, checkpoint flag, resume guard."""
import argparse
import json

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from omegaconf import OmegaConf

from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.train import _build_corpus, run_training
from tests.fixtures.make_synthetic_motion import make_synthetic_clips, make_synthetic_motion

DEV = "cuda:0"


def test_build_corpus_mirror_flag_doubles_clips(tmp_path):
    make_synthetic_clips(tmp_path / "src", [("a", 30), ("b", 20)], asymmetric=True)
    base = dict(motion_dir=str(tmp_path / "src"), motion_cache=None, motion_path=None,
                corpus_device="cpu")
    plain = _build_corpus(OmegaConf.create(base), DEV, KEYPOINT_LINKS)
    mirrored = _build_corpus(OmegaConf.create({**base, "mirror_augment": True}), DEV, KEYPOINT_LINKS)
    assert plain.clip_names == ["a", "b"]
    assert mirrored.clip_names == ["a", "b", "a__mirror", "b__mirror"]
    assert mirrored.n_frames == 2 * plain.n_frames
    assert mirrored.storage_device.type == "cpu" and mirrored.output_device.type == "cuda"
    gpu = _build_corpus(OmegaConf.create({**base, "mirror_augment": True, "corpus_device": None}),
                        DEV, KEYPOINT_LINKS)
    assert gpu.storage_device.type == "cuda" and gpu.n_clips == 4


def _cfg(tmp_path, split_path, **over):
    cfg = dict(
        experiment_name="smoke_mirror", seed=0, device=DEV, log_dir=str(tmp_path / "runs"),
        motion_dir=str(tmp_path / "src"), motion_cache=None, motion_path=None,
        mirror_augment=True,
        env=dict(num_envs=64, control_decimation=1, dt=0.016666, kp=100.0, kd=5.0,
                 action_scale=0.5, foot_friction=0.75, K=9, L=10, episode_len=200,
                 z_fall=0.12, up_dot_min=0.0, head_z_min=0.3, joint_err_done=100.0,
                 root_err_done=5.0, noise_level=1.0,
                 noise=dict(v=[0.5, 0.5, 0.2], w=0.52, g=0.05, q=0.1),
                 keypoint_links=None, recovery_fraction=0.15),
        algo=dict(iterations=2, rollout_len=8, lr=3e-4, n_epochs=2, mb_size=256, clip=0.2,
                  value_coef=1.0, entropy_coef=0.005, max_grad_norm=1.0, dual_clip=3.0,
                  target_kl=0.02, gamma=0.99, lam=0.95, assist_anneal_iters=2),
        network=dict(n_embd=128, n_heads=4, actor_hidden=[512, 256], critic_hidden=[512, 256]),
        reward=dict(),
        eval=dict(every=50, greedy=True, robust_every=0, split_json=str(split_path), mining=True),
    )
    cfg.update(over)
    return OmegaConf.create(cfg)


def test_training_with_mirror_expands_train_split_and_records_flag(tmp_path, capsys):
    make_synthetic_clips(tmp_path / "src", [("a", 120), ("b", 90)], asymmetric=True)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["a"], "test": ["b"]}))
    stats = run_training(_cfg(tmp_path, split))
    assert torch.isfinite(torch.tensor(stats["avg_return"]))
    out = capsys.readouterr().out
    assert "mirror_augment: 2 -> 4 clips" in out
    # RSI weights: b and b__mirror are zero-weighted (2 names set of 4 clips)
    assert "RSI weights: 2/4 clips set" in out
    ck = torch.load(tmp_path / "runs" / "smoke_mirror" / "latest.pt",
                    map_location="cpu", weights_only=False)
    assert ck["config"]["mirror_augment"] is True
    # resuming with the flag flipped is refused loudly
    with pytest.raises(ValueError, match="mirror_augment"):
        run_training(_cfg(tmp_path, split, mirror_augment=False,
                          resume_from=str(tmp_path / "runs" / "smoke_mirror" / "latest.pt")))


def test_view_load_corpus_mirror_flag(tmp_path):
    from rgmt.view import _load_corpus
    p = make_synthetic_motion(tmp_path / "m.npz", n_frames=40, asymmetric=True)
    ns = argparse.Namespace(cache=None, motion_path=str(p), mirror=True)
    c = _load_corpus(ns, torch.device(DEV))
    assert c.clip_names == ["clip0", "clip0__mirror"]
    ns.mirror = False
    assert _load_corpus(ns, torch.device(DEV)).clip_names == ["clip0"]


def test_train_medium_recipe_passes_mirror_flag():
    from scripts.train_medium import build_cfg
    assert build_cfg("cache", "runs", True, DEV, 0).mirror_augment is False
    assert build_cfg("cache", "runs", True, DEV, 0, mirror=True).mirror_augment is True

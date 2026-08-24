import json
from omegaconf import OmegaConf
from rgmt.preprocess import run_preprocess
from rgmt.data.corpus import MotionCorpus
from rgmt.data.cache_key import file_sha256
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_URDF
from tests.fixtures.make_synthetic_motion import make_synthetic_clips

def _cfg(src, cache, force=False):
    return OmegaConf.create(dict(motion_dir=str(src), motion_path=None, cache_dir=str(cache),
                                 physics_fps=60, src_fps=30, keypoint_links=None, force=force))

def test_preprocess_then_load(tmp_path):
    src = tmp_path / "src"; make_synthetic_clips(src, [("a", 18), ("b", 22)])
    stats = run_preprocess(_cfg(src, tmp_path / "cache"))
    assert stats["processed"] == 2 and stats["n_clips"] == 2
    assert (tmp_path / "cache" / "manifest.json").exists()
    corpus = MotionCorpus.load_cache(tmp_path / "cache", output_device="cpu",
                                     urdf_hash=file_sha256(ROBOT_URDF), physics_fps=60,
                                     src_fps=30, keypoint_links=KEYPOINT_LINKS)
    assert corpus.n_clips == 2

def test_rerun_skips_unchanged(tmp_path):
    src = tmp_path / "src"; make_synthetic_clips(src, [("a", 18), ("b", 22)])
    run_preprocess(_cfg(src, tmp_path / "cache"))
    stats2 = run_preprocess(_cfg(src, tmp_path / "cache"))
    assert stats2["skipped"] == 2 and stats2["processed"] == 0

def test_force_rebuilds(tmp_path):
    src = tmp_path / "src"; make_synthetic_clips(src, [("a", 18)])
    run_preprocess(_cfg(src, tmp_path / "cache"))
    stats = run_preprocess(_cfg(src, tmp_path / "cache", force=True))
    assert stats["processed"] == 1 and stats["skipped"] == 0

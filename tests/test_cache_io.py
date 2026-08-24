# tests/test_cache_io.py
import json
import pytest
import torch
from rgmt.data.motion import MotionRef
from rgmt.data.corpus import MotionCorpus
from rgmt.data.cache_key import file_sha256
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from tests.fixtures.make_synthetic_motion import make_synthetic_clips

def _build(tmp_path):
    paths = make_synthetic_clips(tmp_path / "src", [("a", 20), ("b", 15)])
    clips = [MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device="cpu", keypoint_links=KEYPOINT_LINKS)
             for p in paths]
    corpus = MotionCorpus.from_clips(clips, ["a", "b"], KEYPOINT_LINKS, output_device="cpu")
    src_hashes = {"a": file_sha256(paths[0]), "b": file_sha256(paths[1])}
    return corpus, src_hashes, file_sha256(ROBOT_URDF)

def _load(tmp_path, urdf_hash):
    return MotionCorpus.load_cache(
        tmp_path / "cache", output_device="cpu", urdf_hash=urdf_hash,
        physics_fps=60, src_fps=30, keypoint_links=KEYPOINT_LINKS)

def test_round_trip_identical(tmp_path):
    corpus, src_hashes, urdf_hash = _build(tmp_path)
    corpus.save_cache(tmp_path / "cache", source_hashes=src_hashes, urdf_hash=urdf_hash)
    loaded = _load(tmp_path, urdf_hash)
    assert loaded.n_frames == corpus.n_frames and loaded.n_clips == 2
    assert torch.allclose(loaded.base_pos, corpus.base_pos)
    assert torch.allclose(loaded.kp_pos_world, corpus.kp_pos_world)
    assert loaded.clip_end.tolist() == corpus.clip_end.tolist()

def test_stale_source_hash_raises(tmp_path):
    corpus, src_hashes, urdf_hash = _build(tmp_path)
    corpus.save_cache(tmp_path / "cache", source_hashes=src_hashes, urdf_hash=urdf_hash)
    man_path = tmp_path / "cache" / "manifest.json"
    man = json.loads(man_path.read_text())
    man["clips"][0]["source_hash"] = "deadbeef"      # pretend source changed
    man_path.write_text(json.dumps(man))
    with pytest.raises(RuntimeError, match="a"):       # message names clip 'a'
        _load(tmp_path, urdf_hash)

def test_wrong_urdf_hash_raises(tmp_path):
    corpus, src_hashes, urdf_hash = _build(tmp_path)
    corpus.save_cache(tmp_path / "cache", source_hashes=src_hashes, urdf_hash=urdf_hash)
    with pytest.raises(RuntimeError):
        MotionCorpus.load_cache(tmp_path / "cache", output_device="cpu", urdf_hash="other",
                                physics_fps=60, src_fps=30, keypoint_links=KEYPOINT_LINKS)

def test_missing_clip_file_raises(tmp_path):
    import os
    corpus, src_hashes, urdf_hash = _build(tmp_path)
    corpus.save_cache(tmp_path / "cache", source_hashes=src_hashes, urdf_hash=urdf_hash)
    # delete clip 'a' safetensors file but leave the manifest
    os.remove(tmp_path / "cache" / "clip_a.safetensors")
    with pytest.raises(RuntimeError):
        _load(tmp_path, urdf_hash)

def test_truncated_frame_count_raises(tmp_path):
    import json
    corpus, src_hashes, urdf_hash = _build(tmp_path)
    corpus.save_cache(tmp_path / "cache", source_hashes=src_hashes, urdf_hash=urdf_hash)
    man_path = tmp_path / "cache" / "manifest.json"
    man = json.loads(man_path.read_text())
    man["clips"][0]["n_frames"] = man["clips"][0]["n_frames"] + 5  # lie about frame count
    man_path.write_text(json.dumps(man))
    with pytest.raises(RuntimeError, match="frame count"):
        _load(tmp_path, urdf_hash)

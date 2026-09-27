import json
import numpy as np
from rgmt.assets.paths import ROBOT_URDF
from rgmt.data.cache_key import file_sha256
from rgmt.data.corpus import MotionCorpus
from rgmt.data.joint_map import KEYPOINT_LINKS
from locomotion.build_corpus import build, main
from locomotion.config import load_config
from locomotion.corpus import ClipsMeta
from tests.fixtures.make_synthetic_motion import make_synthetic_motion


def _walk_npz(path, vx=1.0, n=90):
    t = np.arange(n) / 30.0
    np.savez(path, base_frame_pos=np.stack([vx * t, 0 * t, 0.7 + 0 * t], 1).astype(np.float32),
             base_frame_wxyz=np.tile(np.array([1.0, 0, 0, 0], np.float32), (n, 1)),
             joint_angles=(0.1 * np.sin(t)[:, None] * np.ones((1, 23))).astype(np.float32))


def test_build_end_to_end(tmp_path):
    d1, d2 = tmp_path / "loco", tmp_path / "v8"
    d1.mkdir(); d2.mkdir()
    _walk_npz(d1 / "fwd.npz", 1.0)
    _walk_npz(d2 / "back.npz", -0.5)
    _walk_npz(d2 / "excluded.npz", 0.3)
    make_synthetic_motion(d1 / "still - one.npz")
    sub = tmp_path / "subset.json"
    sub.write_text(json.dumps({"rules_version": "t", "clips": {
        "walk_fwd": ["fwd", "excluded"], "walk_back": ["back"], "idle": ["still - one"]}}))
    cfg = load_config(overrides=["corpus.exclude=[{name: excluded, reason: test}]"])
    out = tmp_path / "out"
    build(out, sub, [d1, d2], cfg)
    assert (out / "npz" / "fwd.npz").is_symlink()
    assert (out / "npz" / "back.npz").resolve() == (d2 / "back.npz").resolve()
    assert not (out / "npz" / "excluded.npz").exists()
    man = json.loads((out / "cache" / "manifest.json").read_text())
    assert sorted(c["name"] for c in man["clips"]) == ["back", "fwd", "still - one"]
    m2 = ClipsMeta.load(out / "clips.json")
    assert m2.classes == {"fwd": "walk_fwd", "back": "walk_back", "still - one": "idle"}
    assert abs(m2.speeds["forward"] - 1.0) < 0.1 and abs(m2.speeds["backward"] - 0.5) < 0.1
    assert m2.speeds["lateral"] is None and m2.extra["rules_version"] == "t"
    corpus = MotionCorpus.load_cache(out / "cache", output_device="cpu", urdf_hash=file_sha256(ROBOT_URDF),
                                     physics_fps=60, src_fps=30, keypoint_links=KEYPOINT_LINKS)
    assert corpus.n_clips == 3
    # rerun is incremental and removes stale links
    (out / "npz" / "stale.npz").symlink_to(d1 / "fwd.npz")
    main(["--out", str(out), "--subset", str(sub), "--dirs", str(d1), str(d2),
          "--set", "corpus.exclude=[{name: excluded, reason: test}]"])
    assert not (out / "npz" / "stale.npz").exists()

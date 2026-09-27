import json
import pytest
import torch
from locomotion.corpus import resolve_clips, ClipsMeta, measure_speeds
from tests.fixtures.loco_corpus import make_loco_corpus
from tests.fixtures.make_synthetic_motion import make_synthetic_motion


def _subset(tmp_path, clips):
    p = tmp_path / "subset.json"
    p.write_text(json.dumps({"rules_version": "t", "clips": clips}))
    return p


def test_resolve_prefers_first_dir_and_handles_spaces(tmp_path):
    d1, d2 = tmp_path / "loco", tmp_path / "v8"
    d1.mkdir(); d2.mkdir()
    spaced = "ACCAD__Female1Walking_c3d__B22 - side step left"
    make_synthetic_motion(d1 / "a.npz"); make_synthetic_motion(d2 / "a.npz")   # both: d1 wins
    make_synthetic_motion(d2 / f"{spaced}.npz")                                 # only in d2
    sub = _subset(tmp_path, {"walk_fwd": ["a"], "strafe": [spaced]})
    entries = resolve_clips(sub, [d1, d2], exclude={})
    by = {e.name: e for e in entries}
    assert by["a"].path == d1 / "a.npz" and by["a"].cls == "walk_fwd"
    assert by[spaced].path == d2 / f"{spaced}.npz" and by[spaced].cls == "strafe"


def test_resolve_raises_listing_missing(tmp_path):
    d1 = tmp_path / "loco"; d1.mkdir()
    make_synthetic_motion(d1 / "a.npz")
    sub = _subset(tmp_path, {"walk_fwd": ["a", "missing_one", "missing_two"]})
    with pytest.raises(FileNotFoundError) as ei:
        resolve_clips(sub, [d1], exclude={})
    assert "missing_one" in str(ei.value) and "missing_two" in str(ei.value) and "2" in str(ei.value)


def test_resolve_honours_excludes(tmp_path):
    d1 = tmp_path / "loco"; d1.mkdir()
    make_synthetic_motion(d1 / "a.npz"); make_synthetic_motion(d1 / "b.npz")
    sub = _subset(tmp_path, {"walk_fwd": ["a", "b"]})
    entries = resolve_clips(sub, [d1], exclude={"b": "test reason"})
    assert [e.name for e in entries] == ["a"]


def test_clips_meta_roundtrip_and_mirror_names(tmp_path):
    m = ClipsMeta(classes={"a": "walk_fwd", "x y - z": "strafe"}, sources={"a": "/p/a.npz"},
                  speeds={"forward": 1.1, "lateral": None}, extra={"rules_version": "t"})
    m.save(tmp_path / "clips.json")
    m2 = ClipsMeta.load(tmp_path / "clips.json")
    assert m2.classes == m.classes and m2.speeds == m.speeds and m2.extra["rules_version"] == "t"
    assert m2.class_of("a__mirror") == "walk_fwd"
    assert m2.class_of("nope") == "unknown"


def test_measure_speeds_on_synthetic_corpus():
    corpus, meta = make_loco_corpus()
    s = measure_speeds(corpus, meta.classes)
    assert abs(s["forward"] - 1.0) < 0.15          # walk_fwd clips: 1.0, 0.5, ~1.0, 1.0 -> median ~1
    assert abs(s["backward"] - 0.6) < 0.05
    assert abs(s["lateral"] - 0.5) < 0.05
    assert abs(s["turn_rate"] - 1.0) < 0.05
    s2 = measure_speeds(corpus, {"idle": "idle"})
    assert s2 == {"forward": None, "backward": None, "lateral": None, "turn_rate": None}


def test_measure_speeds_ignores_standstill_frames():
    corpus, meta = make_loco_corpus()
    classes = dict(meta.classes); classes["idle"] = "walk_fwd"
    s = measure_speeds(corpus, classes)
    assert s["forward"] > 0.4


def test_synthetic_corpus_shapes():
    corpus, meta = make_loco_corpus(seconds=1.0)
    assert corpus.n_clips == 10 and corpus.physics_fps == 60
    assert corpus.kp_pos_world.shape[1] == len(corpus.keypoint_links)
    assert set(meta.classes.values()) == {"walk_fwd", "walk_back", "strafe", "turn", "idle"}
    f = corpus.at(torch.tensor([5]))
    assert f["joint_q"].shape == (1, 23) and f["base_quat"].shape == (1, 4)

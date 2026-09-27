import pytest
import torch
from locomotion.command import Command, from_keys
from locomotion.config import load_config
from locomotion.controller import MotionMatchingController, run_script
from locomotion.corpus import ClipsMeta
from locomotion.geom import yaw_from_quat, wrap_angle
from tests.fixtures.loco_corpus import make_loco_corpus

KEYS = ("base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "joint_q", "joint_qd")


@pytest.fixture(scope="module")
def world():
    corpus, meta = make_loco_corpus(seconds=4.0)
    cfg = load_config()
    ctl = MotionMatchingController(corpus, meta, cfg, device="cpu")
    return corpus, meta, cfg, ctl


def _pose(frames):
    last = frames[-1]
    return last["base_pos"][0, :2], float(yaw_from_quat(last["base_quat"][0]))


def test_speeds_resolve_from_meta_and_config(world):
    corpus, meta, cfg, ctl = world
    assert abs(ctl.speeds["forward"] - meta.speeds["forward"]) < 1e-6
    cfg2 = load_config(overrides=["speeds.forward=0.42"])
    ctl2 = MotionMatchingController(corpus, meta, cfg2, device="cpu", feature_db=ctl.db)
    assert ctl2.speeds["forward"] == 0.42


def test_missing_speed_raises_with_config_key(world):
    corpus, meta, cfg, ctl = world
    meta2 = ClipsMeta(classes=meta.classes, speeds={**meta.speeds, "lateral": None})
    with pytest.raises(ValueError) as ei:
        MotionMatchingController(corpus, meta2, cfg, device="cpu", feature_db=ctl.db)
    assert "speeds.lateral" in str(ei.value)


def test_reset_starts_on_idle_and_contract(world):
    corpus, meta, cfg, ctl = world
    ctl.reset()
    assert ctl.debug.clip_class == "idle"
    out = ctl.step(Command())
    assert set(out) == set(KEYS)
    assert out["base_pos"].shape == (1, 3) and out["base_quat"].shape == (1, 4)
    assert out["joint_q"].shape == (1, 23) and out["joint_qd"].shape == (1, 23)
    assert out["base_lin_vel"].shape == (1, 3) and out["base_ang_vel"].shape == (1, 3)


def test_w_walks_forward(world):
    _, _, _, ctl = world
    frames = run_script(ctl, "w:2")
    xy, yaw = _pose(frames)
    assert xy[0] > 0.6 and abs(xy[1]) < 0.25 and abs(yaw) < 0.2
    assert ctl.debug.clip_class == "walk_fwd"


def test_w_plus_s_stays(world):
    _, _, _, ctl = world
    frames = run_script(ctl, "ws:2")
    xy, yaw = _pose(frames)
    assert xy.norm() < 0.1 and abs(yaw) < 0.1
    assert ctl.debug.clip_class == "idle"


def test_w_plus_d_moves_diagonally_without_turning(world):
    _, _, _, ctl = world
    frames = run_script(ctl, "wd:2")
    xy, yaw = _pose(frames)
    assert xy[0] > 0.3 and xy[1] < -0.3 and abs(yaw) < 0.2


def test_q_turns_in_place(world):
    _, _, _, ctl = world
    frames = run_script(ctl, "q:2")
    xy, yaw = _pose(frames)
    assert yaw > 0.5 and xy.norm() < 0.3
    assert ctl.debug.clip_class == "turn"
    frames = run_script(ctl, "e:2")
    _, yaw = _pose(frames)
    assert yaw < -0.5


def test_stream_has_no_pops(world):
    _, _, _, ctl = world
    frames = run_script(ctl, "w:1,q:1,wd:1,:1")
    q = torch.cat([f["joint_q"] for f in frames])
    assert (q[1:] - q[:-1]).abs().max() < 0.2          # raw clip steps are ~0.04; a pop would be ~0.6
    p = torch.cat([f["base_pos"] for f in frames])
    assert (p[1:, :2] - p[:-1, :2]).norm(dim=-1).max() < 0.05


def test_search_happens_on_command_change_and_never_enters_clip_tail(world):
    corpus, meta, cfg, ctl = world
    ctl.reset()
    for _ in range(3):
        ctl.step(Command())
    assert not ctl.debug.searched                    # interval 6 not reached, command unchanged
    ctl.step(from_keys({"w"}))
    assert ctl.debug.searched
    c = corpus.clip_names.index("walk_fwd")
    end = int(corpus.clip_end[c])
    ctl.reset(frame=end - 62)                        # two frames before the last searchable one
    seen = []
    for _ in range(20):
        ctl.step(from_keys({"w"}))
        seen.append(ctl.debug.frame)
    in_clip = [f for f in seen if int(corpus.frame_clip_id[f]) == c]
    assert max(in_clip) <= end - 50                  # forced search before the horizon-less tail


def test_observed_root_re_anchors(world):
    _, _, _, ctl = world
    ctl.reset()
    ctl.step(from_keys({"w"}))
    out = ctl.step(from_keys({"w"}), observed_root=((3.0, 4.0), 1.2))
    assert torch.allclose(out["base_pos"][0, :2], torch.tensor([3.0, 4.0]), atol=3e-2)
    assert abs(wrap_angle(float(yaw_from_quat(out["base_quat"][0])) - 1.2)) < 5e-2


def test_preview_shape(world):
    _, _, _, ctl = world
    ctl.reset()
    ctl.step(Command())
    pv = ctl.preview(10)
    assert pv["joint_q"].shape == (10, 23) and pv["base_pos"].shape == (10, 3)


def _frames_of(ctl, script, **reset_kw):
    ctl.reset(**reset_kw)
    from locomotion.command import ScriptedSource
    seen = []
    for c in ScriptedSource(script, fps=ctl.cfg.fps):
        ctl.step(c)
        seen.append(ctl.debug.frame)
    return seen


def _repeats(seen):
    return sum(1 for a, b in zip(seen[1:], seen[2:]) if a == b)   # skip the reset re-emit tick


def test_playback_never_repeats_a_frame(world):
    _, _, _, ctl = world
    assert _repeats(_frames_of(ctl, "w:3")) == 0
    assert _repeats(_frames_of(ctl, "wd:3")) == 0
    assert _repeats(_frames_of(ctl, "q:1,:1,s:1")) == 0


def test_clip_tail_is_left_without_freezing(world):
    corpus, _, _, ctl = world
    c = corpus.clip_names.index("walk_fwd")
    end = int(corpus.clip_end[c])
    seen = _frames_of(ctl, "w:0.5", frame=end - 62)
    assert _repeats(seen) == 0
    in_clip = [f for f in seen if int(corpus.frame_clip_id[f]) == c]
    assert max(in_clip) <= end - 50                  # never plays into the horizon-less tail
    assert any(int(corpus.frame_clip_id[f]) != c or f < end - 62 for f in seen)   # it did jump away


def test_w_plus_d_covers_nominal_distance(world):
    _, _, _, ctl = world
    frames = run_script(ctl, "wd:2")
    xy, _ = _pose(frames)
    assert xy[0] > 0.5 and xy[1] < -0.5          # ~0.99 m/s diagonal minus the 0.5 s ramp

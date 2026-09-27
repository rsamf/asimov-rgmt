import math
import torch
from locomotion.config import BlendConfig
from locomotion.geom import yaw_from_quat, wrap_angle
from locomotion.player import Player
from tests.fixtures.loco_corpus import make_loco_corpus

KEYS = ("base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "joint_q", "joint_qd")


def _player(halflife=0.1):
    corpus, _ = make_loco_corpus(seconds=2.0)
    return corpus, Player(corpus, BlendConfig(halflife=halflife), fps=60, device="cpu")


def _start(corpus, name):
    return int(corpus.clip_start[corpus.clip_names.index(name)])


def _qclose(a, b, atol):
    """Quaternions equal up to sign (q and -q are the same rotation)."""
    return torch.allclose(a, b, atol=atol) or torch.allclose(a, -b, atol=atol)


def test_reset_places_root_and_emits_current_frame_first():
    corpus, p = _player()
    f0 = _start(corpus, "walk_fwd") + 5
    p.reset(f0, xy=(2.0, -1.0), yaw=0.8)
    assert p.frame == f0 and p.next_frame == f0
    assert torch.allclose(p.world_xy, torch.tensor([2.0, -1.0]), atol=1e-6)
    assert abs(p.world_yaw - 0.8) < 1e-6
    out = p.advance()
    assert set(out) == set(KEYS) and out["joint_q"].shape == (1, 23) and out["base_pos"].shape == (1, 3)
    assert torch.allclose(out["base_pos"][0, :2], torch.tensor([2.0, -1.0]), atol=1e-6)
    assert abs(float(yaw_from_quat(out["base_quat"][0])) - 0.8) < 1e-6
    assert p.next_frame == f0 + 1
    p.advance()
    assert p.frame == f0 + 1


def test_anchor_rotates_velocities_and_root_path():
    corpus, p = _player()
    f0 = _start(corpus, "walk_fwd") + 5              # walks along clip +x at 1 m/s
    p.reset(f0, xy=(0.0, 0.0), yaw=math.pi / 2)        # but the character faces +y
    a = p.advance(); b = p.advance()
    v = a["base_lin_vel"][0]
    assert v[1] > 0.9 and abs(v[0]) < 0.05
    assert b["base_pos"][0, 1] > a["base_pos"][0, 1]
    assert torch.allclose(a["joint_q"], corpus.at(torch.tensor([f0]))["joint_q"], atol=1e-6)


def test_switch_is_continuous_and_offset_decays():
    corpus, p = _player(halflife=0.1)
    p.reset(_start(corpus, "walk_fwd") + 10, xy=(0.0, 0.0), yaw=0.0)
    last = None
    for _ in range(5):
        last = p.advance()
    j = _start(corpus, "turn_left") + 40                # very different pose and root yaw
    p.switch(j)
    first = p.advance()
    for k in KEYS:
        if k == "base_quat":
            assert _qclose(first[k], last[k], 1e-5)
        elif k == "base_lin_vel":
            # planar root velocity is not inertialized by design (the matcher owns it); height rate is
            assert abs(float(first[k][0, 2] - last[k][0, 2])) < 1e-5
        else:
            assert torch.allclose(first[k], last[k], atol=1e-5), k
    for _ in range(int(5 * 0.1 * 60)):                  # five half-lives
        out = p.advance()
    raw = p.map_frames(torch.tensor([p.frame]))
    assert torch.allclose(out["joint_q"], raw["joint_q"], atol=2e-2)
    assert _qclose(out["base_quat"], raw["base_quat"], 2e-2)


def test_anchor_continuous_across_pi_wrap():
    corpus, p = _player()
    j = _start(corpus, "turn_right") + 100             # clip yaw = -100/60 rad
    p.reset(_start(corpus, "idle") + 3, xy=(0.0, 0.0), yaw=math.pi - 0.05)
    last = p.advance()
    p.switch(j)
    first = p.advance()
    d = wrap_angle(float(yaw_from_quat(first["base_quat"][0])) - float(yaw_from_quat(last["base_quat"][0])))
    assert abs(d) < 1e-5
    assert abs(wrap_angle(p.world_yaw - (math.pi - 0.05))) < 1e-4


def test_re_anchor_moves_output_to_observed_root():
    corpus, p = _player()
    p.reset(_start(corpus, "walk_fwd") + 5, xy=(0.0, 0.0), yaw=0.0)
    p.advance()
    p.re_anchor(xy=(5.0, 5.0), yaw=1.0)
    out = p.advance()
    assert torch.allclose(out["base_pos"][0, :2], torch.tensor([5.0, 5.0]), atol=2e-2)
    assert abs(float(yaw_from_quat(out["base_quat"][0])) - 1.0) < 1e-3


def test_preview_and_clip_end_clamp():
    corpus, p = _player()
    end = int(corpus.clip_end[0])
    p.reset(end - 2, xy=(0.0, 0.0), yaw=0.0)
    p.advance()                                         # emits end-2
    pv = p.preview(5)
    assert pv["joint_q"].shape == (5, 23)
    assert torch.allclose(pv["joint_q"][2], pv["joint_q"][4])       # clamped at the clip end
    p.advance(); p.advance(); p.advance()
    assert p.frame == end and p.next_frame == end

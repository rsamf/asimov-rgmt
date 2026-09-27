"""Procedural MotionCorpus for locomotion tests: no URDF, no FK, no data files.

Ten 60 fps clips with consistent finite-difference velocities and keypoints
from a toy leg model (feet swing in antiphase in the heading frame while the
root moves). Classes match the asimov-gmr taxonomy.
"""
import math
import torch

from rgmt.data.corpus import MotionCorpus, clip_bookkeeping
from rgmt.data.joint_map import ASIMOV_ACTUATED_JOINT_NAMES, KEYPOINT_LINKS
from locomotion.corpus import ClipsMeta, measure_speeds
from locomotion.geom import quat_from_yaw, rotate2

_LI = KEYPOINT_LINKS.index("left_ankle_roll_link")
_RI = KEYPOINT_LINKS.index("right_ankle_roll_link")
_PELVIS = KEYPOINT_LINKS.index("pelvis_link")
_STEP_HZ = 1.4


def _grad(x: torch.Tensor, fps: int) -> torch.Tensor:
    return torch.gradient(x, spacing=1.0 / fps, dim=0)[0]


def _clip(n: int, fps: int, vel_xy, yaw0: float, yaw_rate: float, moving: bool):
    t = torch.arange(n, dtype=torch.float32) / fps
    xy = torch.stack([vel_xy[0] * t, vel_xy[1] * t], -1)
    yaw = yaw0 + yaw_rate * t
    base_pos = torch.cat([xy, torch.full((n, 1), 0.70)], -1)
    base_quat = quat_from_yaw(yaw)
    amp = 0.25 if moving else 0.0
    phase = torch.sin(2 * math.pi * _STEP_HZ * t)
    swing = amp * phase
    lift = (0.05 if moving else 0.0) * torch.clamp(phase, min=0.0)
    # feet in the heading frame: left at +y, right at -y, antiphase swing
    lf_h = torch.stack([swing, torch.full((n,), 0.10), lift], -1)
    rf_h = torch.stack([-swing, torch.full((n,), -0.10), (0.05 if moving else 0.0) - lift], -1)
    kp = torch.zeros(n, len(KEYPOINT_LINKS), 3)
    kp[:, _PELVIS] = base_pos
    for i, f_h in ((_LI, lf_h), (_RI, rf_h)):
        kp[:, i, :2] = xy + rotate2(f_h[:, :2], yaw)
        kp[:, i, 2] = f_h[:, 2]
    for i in range(len(KEYPOINT_LINKS)):                      # remaining links: fixed body offsets
        if i in (_LI, _RI, _PELVIS):
            continue
        off = torch.tensor([0.05 * i, 0.02 * (-1) ** i])
        kp[:, i, :2] = xy + rotate2(off.expand(n, 2), yaw)
        kp[:, i, 2] = 0.3 + 0.1 * i
    j = torch.arange(23, dtype=torch.float32)
    joint_q = (0.3 if moving else 0.02) * torch.sin(2 * math.pi * _STEP_HZ * t[:, None] + 0.4 * j[None])
    return {
        "base_pos": base_pos, "base_quat": base_quat,
        "base_lin_vel": _grad(base_pos, fps),
        "base_ang_vel": torch.stack([torch.zeros(n), torch.zeros(n), torch.full((n,), yaw_rate)], -1),
        "joint_q": joint_q, "joint_qd": _grad(joint_q, fps),
        "kp_pos_world": kp, "kp_vel_world": _grad(kp, fps),
    }


_SPECS = [  # name, class, planar velocity (world), initial yaw, yaw rate, moving feet
    ("walk_fwd", "walk_fwd", (1.0, 0.0), 0.0, 0.0, True),
    ("walk_fwd_slow", "walk_fwd", (0.5, 0.0), 0.0, 0.0, True),
    ("walk_diag", "walk_fwd", (0.7, -0.7), 0.0, 0.0, True),        # forward-right while facing +x
    ("walk_y", "walk_fwd", (0.0, 1.0), math.pi / 2, 0.0, True),    # same gait as walk_fwd, facing +y
    ("walk_back", "walk_back", (-0.6, 0.0), 0.0, 0.0, True),
    ("strafe_left", "strafe", (0.0, 0.5), 0.0, 0.0, True),
    ("strafe_right", "strafe", (0.0, -0.5), 0.0, 0.0, True),
    ("turn_left", "turn", (0.0, 0.0), 0.0, 1.0, True),
    ("turn_right", "turn", (0.0, 0.0), 0.0, -1.0, True),
    ("idle", "idle", (0.0, 0.0), 0.0, 0.0, False),
]


def make_loco_corpus(fps: int = 60, seconds: float = 4.0, device="cpu"):
    n = int(seconds * fps)
    clips, names, classes = [], [], {}
    for name, cls, v, yaw0, w, moving in _SPECS:
        clips.append(_clip(n, fps, v, yaw0, w, moving)); names.append(name); classes[name] = cls
    cat = {k: torch.cat([c[k] for c in clips], 0).to(device) for k in clips[0]}
    lens = torch.tensor([n] * len(clips))
    starts, ends, fcid = clip_bookkeeping(lens)
    corpus = MotionCorpus(
        physics_fps=fps, src_fps=30, actuated_idx=torch.arange(23),
        actuated_names=list(ASIMOV_ACTUATED_JOINT_NAMES), keypoint_links=list(KEYPOINT_LINKS),
        output_device=torch.device(device), clip_start=starts, clip_len=lens, clip_end=ends,
        frame_clip_id=fcid, clip_names=names, **cat)
    corpus = corpus.to_storage(device)
    meta = ClipsMeta(classes=classes)
    meta.speeds = measure_speeds(corpus, classes)
    return corpus, meta

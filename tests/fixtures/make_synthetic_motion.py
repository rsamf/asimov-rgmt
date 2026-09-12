"""Synthetic motion fixture for testing MotionRef loader."""

import numpy as np
from pathlib import Path
from typing import Union


def make_synthetic_motion(
    path: Union[str, Path],
    n_frames: int = 40,
    fps: int = 30,
    *,
    asymmetric: bool = False,
) -> Path:
    """Write a minimal synthetic motion .npz for testing.

    Produces:
        base_frame_pos  (n_frames, 3)  float32
        base_frame_wxyz (n_frames, 4)  float32  (identity quaternion)
        joint_angles    (n_frames, 23) float32  (sinusoidal)

    ``asymmetric=True`` instead writes a clip with distinct per-joint
    amplitude and phase, a lateral (y) root drift with z bob, and a root
    rotation about a tilted axis: a motion whose left-right mirror is
    distinguishable from itself (the default is bilaterally symmetric and
    cannot discriminate a correct mirror from a buggy one).
    """
    path = Path(path)
    t = np.linspace(0, 2 * np.pi, n_frames)
    if not asymmetric:
        base_pos = np.stack([0.01 * t, np.zeros_like(t), 0.63 + 0.0 * t], axis=1)
        wxyz = np.tile(np.array([1.0, 0, 0, 0]), (n_frames, 1))
        ja = 0.2 * np.sin(t)[:, None] * np.ones((1, 23))
    else:
        j = np.arange(23)
        amp = 0.05 + 0.01 * j                      # 0.05 .. 0.27 rad
        phase = 0.3 * j
        ja = amp[None, :] * np.sin(t[:, None] + phase[None, :])
        base_pos = np.stack([0.02 * t, 0.05 * np.sin(t), 0.70 + 0.03 * np.cos(2 * t)], axis=1)
        axis = np.array([0.3, 0.5, 0.8]); axis /= np.linalg.norm(axis)
        ang = 0.4 * np.sin(t) + 0.2 * t            # net yaw-ish drift + wobble
        wxyz = np.concatenate(
            [np.cos(ang / 2)[:, None], np.sin(ang / 2)[:, None] * axis[None, :]], axis=1)
    np.savez(
        path,
        base_frame_pos=base_pos.astype(np.float32),
        base_frame_wxyz=wxyz.astype(np.float32),
        joint_angles=ja.astype(np.float32),
    )
    return path


def mirror_synthetic_npz(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """Write the left-right mirror of an .npz clip straight from the math
    (independent of rgmt.data.mirror): pos (x,-y,z), wxyz (w,-x,y,-z),
    joints swapped left<->right and negated."""
    from rgmt.data.joint_map import ASIMOV_ACTUATED_JOINT_NAMES as NAMES
    d = np.load(src)
    pos = d["base_frame_pos"].copy(); pos[:, 1] *= -1
    q = d["base_frame_wxyz"].copy(); q[:, 1] *= -1; q[:, 3] *= -1
    idx = {n: i for i, n in enumerate(NAMES)}

    def partner(n):
        return n.replace("left_", "right_") if n.startswith("left_") else \
            n.replace("right_", "left_") if n.startswith("right_") else n
    perm = [idx[partner(n)] for n in NAMES]
    ja = -d["joint_angles"][:, perm]
    np.savez(dst, base_frame_pos=pos.astype(np.float32),
             base_frame_wxyz=q.astype(np.float32), joint_angles=ja.astype(np.float32))
    return Path(dst)


def make_synthetic_clips(directory, specs, **kwargs):
    """Write several tiny .npz clips. specs = list of (name, n_frames). Returns list of paths."""
    import os
    os.makedirs(directory, exist_ok=True)
    paths = []
    for name, n in specs:
        p = os.path.join(str(directory), f"{name}.npz")
        make_synthetic_motion(p, n_frames=n, **kwargs)
        paths.append(p)
    return paths

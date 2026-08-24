"""Synthetic motion fixture for testing MotionRef loader."""

import numpy as np
from pathlib import Path
from typing import Union


def make_synthetic_motion(
    path: Union[str, Path],
    n_frames: int = 40,
    fps: int = 30,
) -> Path:
    """Write a minimal synthetic motion .npz for testing.

    Produces:
        base_frame_pos  (n_frames, 3)  float32
        base_frame_wxyz (n_frames, 4)  float32  (identity quaternion)
        joint_angles    (n_frames, 23) float32  (sinusoidal)
    """
    path = Path(path)
    t = np.linspace(0, 2 * np.pi, n_frames)
    base_pos = np.stack([0.01 * t, np.zeros_like(t), 0.63 + 0.0 * t], axis=1)
    wxyz = np.tile(np.array([1.0, 0, 0, 0]), (n_frames, 1))
    ja = 0.2 * np.sin(t)[:, None] * np.ones((1, 23))
    np.savez(
        path,
        base_frame_pos=base_pos.astype(np.float32),
        base_frame_wxyz=wxyz.astype(np.float32),
        joint_angles=ja.astype(np.float32),
    )
    return path


def make_synthetic_clips(directory, specs):
    """Write several tiny .npz clips. specs = list of (name, n_frames). Returns list of paths."""
    import os
    os.makedirs(directory, exist_ok=True)
    paths = []
    for name, n in specs:
        p = os.path.join(str(directory), f"{name}.npz")
        make_synthetic_motion(p, n_frames=n)
        paths.append(p)
    return paths

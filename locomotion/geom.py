"""Planar rotation and quaternion helpers (xyzw) for the locomotion controller.

Rotation matrices are built on the device/dtype of the vector they act on
(when one is given), so the same code runs on CPU tests and the CUDA scene.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from rgmt.data.motion import quat_mul, quat_conj          # noqa: F401  (re-export)
from rgmt.utils.rotation import yaw_from_quat             # noqa: F401  (re-export)

__all__ = ["wrap_angle", "rot2", "rotate2", "rot3z", "quat_from_yaw", "quat_to_rotvec",
           "rotvec_to_quat", "quat_mul", "quat_conj", "yaw_from_quat"]


def wrap_angle(a):
    """Wrap an angle (float or tensor) into [-pi, pi)."""
    if isinstance(a, Tensor):
        return torch.remainder(a + math.pi, 2 * math.pi) - math.pi
    return (a + math.pi) % (2 * math.pi) - math.pi


def _t(yaw, like: Tensor | None = None) -> Tensor:
    if isinstance(yaw, Tensor):
        return yaw if like is None else yaw.to(device=like.device, dtype=like.dtype)
    if like is None:
        return torch.tensor(float(yaw))
    return torch.tensor(float(yaw), device=like.device, dtype=like.dtype)


def rot2(yaw, like: Tensor | None = None) -> Tensor:
    """Counter-clockwise planar rotation matrix R(yaw), shape (..., 2, 2)."""
    y = _t(yaw, like)
    c, s = torch.cos(y), torch.sin(y)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def rotate2(v: Tensor, yaw) -> Tensor:
    """R(yaw) @ v for v of shape (..., 2); yaw is a scalar or broadcastable batch."""
    return torch.einsum("...ij,...j->...i", rot2(yaw, like=v), v)


def rot3z(yaw, like: Tensor | None = None) -> Tensor:
    """Rotation about +z by yaw, shape (..., 3, 3)."""
    y = _t(yaw, like)
    c, s = torch.cos(y), torch.sin(y)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, -s, z], -1),
                        torch.stack([s, c, z], -1),
                        torch.stack([z, z, o], -1)], -2)


def quat_from_yaw(yaw, like: Tensor | None = None) -> Tensor:
    """xyzw quaternion for a rotation of yaw about +z, shape (..., 4)."""
    y = _t(yaw, like)
    h = y / 2
    z = torch.zeros_like(h)
    return torch.stack([z, z, torch.sin(h), torch.cos(h)], -1)


def quat_to_rotvec(q: Tensor) -> Tensor:
    """Shortest-arc rotation vector of an xyzw unit quaternion, shape (..., 3)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = torch.where(q[..., 3:4] < 0, -q, q)               # w >= 0 -> angle in [0, pi]
    v, w = q[..., :3], q[..., 3]
    n = v.norm(dim=-1)
    angle = 2.0 * torch.atan2(n, w)
    small = n < 1e-6
    scale = torch.where(small, 2.0 * torch.ones_like(n), angle / n.clamp_min(1e-12))
    return v * scale.unsqueeze(-1)


def rotvec_to_quat(r: Tensor) -> Tensor:
    """xyzw unit quaternion of a rotation vector, shape (..., 4)."""
    angle = r.norm(dim=-1)
    half = angle / 2
    small = angle < 1e-6
    k = torch.where(small, 0.5 * torch.ones_like(angle), torch.sin(half) / angle.clamp_min(1e-12))
    xyz = r * k.unsqueeze(-1)
    w = torch.cos(half)
    q = torch.cat([xyz, w.unsqueeze(-1)], -1)
    return q / q.norm(dim=-1, keepdim=True)

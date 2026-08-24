import torch
from torch import Tensor

def encode_angles(q: Tensor) -> Tensor:
    """Continuous SO(2) encoding of joint angles: [cos q, sin q]."""
    return torch.cat([torch.cos(q), torch.sin(q)], dim=-1)

def quat_to_matrix(q: Tensor) -> Tensor:
    """xyzw quaternion -> rotation matrix (..., 3, 3)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy),
    ], dim=-1)
    return m.reshape(*q.shape[:-1], 3, 3)

def quat_rotate_inverse(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector v from world frame into the body frame of quaternion q (xyzw)."""
    R = quat_to_matrix(q)                      # (..., 3, 3)
    return torch.einsum("...ji,...j->...i", R, v)  # R^T @ v

def gravity_projection(q: Tensor) -> Tensor:
    """Gravity direction (0,0,-1) expressed in the body frame of q (xyzw)."""
    g = torch.zeros(*q.shape[:-1], 3, device=q.device, dtype=q.dtype)
    g[..., 2] = -1.0
    return quat_rotate_inverse(q, g)

def yaw_from_quat(q: Tensor) -> Tensor:
    """Heading (yaw) angle of an xyzw quaternion, radians in (-pi, pi]."""
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

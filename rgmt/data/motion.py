"""Reference motion loader for Asimov motion-tracking.

Adapted from an earlier internal motion library.

Loads a retargeted .npz (``base_frame_pos``, ``base_frame_wxyz``,
``joint_angles``) written at ``src_fps``, pre-upsamples to ``physics_fps``
using SLERP for orientations and linear interpolation for translations /
joint angles, and computes finite-difference velocities at the upsampled rate.

Quaternion convention: **xyzw** (vector-scalar) throughout. The source .npz
uses ``base_frame_wxyz`` (Pyroki output is wxyz); we convert at load time.

Joint mapping
-------------
The vendored URDF has **25** movable joints: 23 actuated + 2 passive neck
joints (``neck_yaw_joint``, ``neck_pitch_joint``). The .npz ``joint_angles``
array has exactly 23 columns, one per actuated joint in
``ASIMOV_ACTUATED_JOINT_NAMES`` order. We resolve ``actuated_idx`` (indices
into the 25-joint URDF space) and ``passive_idx`` from joint **names** — never
from hardcoded counts — and assert the resolved actuated count is exactly 23 so
a stale URDF cannot silently mis-map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
import yourdfpy

from rgmt.data.joint_map import ASIMOV_ACTUATED_JOINT_NAMES
from rgmt.utils.rotation import encode_angles, quat_rotate_inverse, gravity_projection, quat_to_matrix


# ---------------------------------------------------------------------------
# Quaternion helpers (xyzw convention)
# ---------------------------------------------------------------------------

def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion (xyzw). Shape preserving."""
    return torch.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a * b for xyzw quaternions."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack([x, y, z, w], dim=-1)


def quat_log_xyz(q: torch.Tensor) -> torch.Tensor:
    """Logarithm of a unit quaternion (xyzw) -> 3-vector.

    For q encoding rotation by angle theta about axis u:
    2 * log(q) = theta * u, so this returns (theta/2) * u.
    Combine with a (2/dt) factor to get angular velocity.
    """
    xyz = q[..., :3]
    w = q[..., 3]
    norm_xyz = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    half_angle = torch.atan2(norm_xyz.squeeze(-1), w)          # in [0, pi]
    # Map (pi/2, pi] half-angles into (-pi/2, 0] so the full angle is in (-pi, pi].
    half_angle = torch.where(
        half_angle > torch.pi / 2.0, half_angle - torch.pi, half_angle
    )
    safe_norm = norm_xyz.clamp(min=1e-9)
    return half_angle.unsqueeze(-1) * (xyz / safe_norm)


def slerp(q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation.

    Args:
        q0, q1: (..., 4) xyzw unit quaternions.
        alpha:  (...,) interpolation weight in [0, 1].

    Returns:
        (..., 4) interpolated quaternion.
    """
    # Shortest-path: flip q1 when its dot with q0 is negative.
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)

    theta = torch.acos(dot)                                    # (..., 1)
    sin_theta = torch.sin(theta)
    a = alpha.unsqueeze(-1)
    small = sin_theta < 1e-6
    w0 = torch.where(
        small,
        1.0 - a,
        torch.sin((1.0 - a) * theta) / sin_theta.clamp(min=1e-9),
    )
    w1 = torch.where(
        small,
        a,
        torch.sin(a * theta) / sin_theta.clamp(min=1e-9),
    )
    out = w0 * q0 + w1 * q1
    return out / torch.linalg.norm(out, dim=-1, keepdim=True).clamp(min=1e-9)


# Foot sole sits ~3 cm below the ankle_roll_link origin (MJCF foot capsules at
# z ~= -0.029). Used by ground-normalization to rest the sole at the floor.
_FOOT_SOLE_OFFSET = 0.03


# ---------------------------------------------------------------------------
# MotionRef
# ---------------------------------------------------------------------------

@dataclass
class MotionRef:
    """Pre-upsampled reference motion with finite-difference velocities.

    All tensor fields live on ``device`` with shape ``(n_frames, ...)``.
    """

    src_fps: int                       # source motion FPS (typically 30)
    physics_fps: int                   # simulation FPS (typically 60)
    upsample: int                      # physics_fps // src_fps
    T_src: int                         # raw frames in source npz
    T_up: int                          # = (T_src - 1) * upsample + 1
    n_actuated: int                    # 23 for Asimov v1
    n_urdf_joints: int                 # 25 for Asimov v1 (includes 2 passive neck)
    device: torch.device

    # Per-frame tensors, all shape (T_up, ...) on device.
    base_pos: torch.Tensor             # (T_up, 3)
    base_quat: torch.Tensor            # (T_up, 4) xyzw
    joint_q: torch.Tensor              # (T_up, 23) actuated joints only
    joint_qd: torch.Tensor             # (T_up, 23) actuated joints only
    base_lin_vel: torch.Tensor         # (T_up, 3) world frame
    base_ang_vel: torch.Tensor         # (T_up, 3) world frame

    # Joint index mappings into the 25-joint URDF space.
    actuated_idx: torch.LongTensor     # (23,)
    passive_idx: torch.LongTensor      # (2,)
    actuated_names: list               # ASIMOV_ACTUATED_JOINT_NAMES order

    # Optional keypoint precompute (None when not requested at load time).
    _kp_pos_world: torch.Tensor = None  # (F, Kp, 3) world positions
    _kp_vel_world: torch.Tensor = None  # (F, Kp, 3) finite-diff velocities

    @property
    def n_frames(self) -> int:
        """Number of upsampled frames."""
        return self.T_up

    @classmethod
    def load(
        cls,
        npz_path: Union[str, Path],
        robot_xml: Union[str, Path],
        robot_urdf: Union[str, Path],
        *,
        physics_fps: int = 60,
        src_fps: int = 30,
        device: Union[str, torch.device] = "cpu",
        keypoint_links: list = None,
        ground: bool = False,
    ) -> "MotionRef":
        """Load and pre-process a retargeted motion npz.

        Args:
            npz_path:    Path to .npz with keys base_frame_pos, base_frame_wxyz,
                         joint_angles (all in ASIMOV_ACTUATED_JOINT_NAMES order).
            robot_xml:   Path to Asimov MuJoCo XML (unused at runtime; reserved
                         for future actuator-name cross-check).
            robot_urdf:  Path to Asimov URDF with 25 movable joints.
            physics_fps: Target simulation rate; must be a multiple of src_fps.
            src_fps:     Frame rate of the source npz.
            device:         PyTorch device string or object.
            keypoint_links: Optional list of URDF link names for which to
                            precompute world-frame keypoint positions and
                            finite-difference velocities (shape (F, Kp, 3)).
                            When None, no keypoint data is precomputed.

        Returns:
            Fully initialised MotionRef.

        Raises:
            ValueError: If physics_fps is not a multiple of src_fps, if the
                        URDF does not contain all ASIMOV_ACTUATED_JOINT_NAMES,
                        or if the resolved actuated count != 23.
        """
        device = torch.device(device)
        if physics_fps % src_fps != 0:
            raise ValueError(
                f"physics_fps ({physics_fps}) must be a multiple of src_fps ({src_fps})"
            )
        upsample = physics_fps // src_fps

        # ---- Load source arrays.
        data = np.load(npz_path)
        base_pos_src = torch.tensor(
            data["base_frame_pos"], dtype=torch.float32, device=device
        )
        base_quat_wxyz = torch.tensor(
            data["base_frame_wxyz"], dtype=torch.float32, device=device
        )
        joint_q_src = torch.tensor(
            data["joint_angles"], dtype=torch.float32, device=device
        )

        # wxyz -> xyzw (Pyroki convention -> our convention).
        base_quat_src = torch.stack(
            [
                base_quat_wxyz[..., 1],
                base_quat_wxyz[..., 2],
                base_quat_wxyz[..., 3],
                base_quat_wxyz[..., 0],
            ],
            dim=-1,
        )
        # Normalise (optimisation output may drift slightly off unit length).
        base_quat_src = base_quat_src / torch.linalg.norm(
            base_quat_src, dim=-1, keepdim=True
        ).clamp(min=1e-9)

        T_src = base_pos_src.shape[0]
        n_npz_joints = joint_q_src.shape[1]

        # ---- Resolve actuated / passive joint indices from URDF joint names.
        #
        # The URDF has 25 movable joints: 23 actuated + 2 passive neck.
        # We must NOT use hardcoded counts; resolve entirely from names so a
        # stale URDF cannot silently mis-map.
        urdf = yourdfpy.URDF.load(
            str(robot_urdf),
            mesh_dir=str(Path(robot_urdf).parent / ".." / "assets" / "meshes"),
        )
        urdf_joint_names = list(urdf.actuated_joint_names)
        n_urdf_joints = len(urdf_joint_names)

        actuated_names = list(ASIMOV_ACTUATED_JOINT_NAMES)

        # Verify every actuated name exists in the URDF.
        missing = [n for n in actuated_names if n not in urdf_joint_names]
        if missing:
            raise ValueError(
                f"URDF is missing actuated joints: {missing}\n"
                f"URDF joints: {urdf_joint_names}"
            )

        actuated_idx_list = [urdf_joint_names.index(n) for n in actuated_names]
        actuated_idx = torch.tensor(actuated_idx_list, dtype=torch.long, device=device)

        # CRITICAL: assert exactly 23 actuated joints resolved.
        if len(actuated_idx_list) != 23:
            raise AssertionError(
                f"Expected exactly 23 actuated joints but resolved {len(actuated_idx_list)}. "
                f"Check ASIMOV_ACTUATED_JOINT_NAMES or the URDF.\n"
                f"Resolved: {actuated_names}"
            )

        actuated_set = set(actuated_idx_list)
        passive_idx = torch.tensor(
            [i for i in range(n_urdf_joints) if i not in actuated_set],
            dtype=torch.long,
            device=device,
        )

        # Sanity-check npz joint count matches actuated count.
        if n_npz_joints != 23:
            raise ValueError(
                f"Expected npz joint_angles to have 23 columns (actuated only) "
                f"but got {n_npz_joints}."
            )
        # Column-ORDER contract (review R3): count==23 cannot detect a
        # reordered export upstream (this repo's canonical order is unusual:
        # right arm before left). If the .npz carries its joint-name list,
        # verify it; older exports without the key pass unchecked.
        if "joint_names" in data:
            npz_names = [str(n) for n in data["joint_names"]]
            if npz_names != actuated_names:
                raise ValueError(
                    "npz joint_angles column order does not match "
                    "ASIMOV_ACTUATED_JOINT_NAMES — every joint would silently "
                    f"mis-map. npz: {npz_names[:4]}... expected: "
                    f"{actuated_names[:4]}...")

        # ---- Upsample: build interpolation indices and weights.
        #
        # T_up = (T_src - 1) * upsample + 1 so the first and last source frames
        # are reproduced exactly and the sequence does not overshoot.
        T_up = (T_src - 1) * upsample + 1
        t_idx = torch.arange(T_up, device=device, dtype=torch.float32)
        t_src = t_idx / upsample                                # in [0, T_src-1]
        # clamp ensures i_lo + 1 is always a valid index; for the last upsampled
        # frame t_src == T_src - 1, alpha == 1, and we pull solely from i_lo + 1.
        i_lo = t_src.long().clamp(max=T_src - 2)
        alpha = (t_src - i_lo.float()).unsqueeze(-1)            # (T_up, 1)

        base_pos_up = (
            (1.0 - alpha) * base_pos_src[i_lo] + alpha * base_pos_src[i_lo + 1]
        )
        joint_q_up = (
            (1.0 - alpha) * joint_q_src[i_lo] + alpha * joint_q_src[i_lo + 1]
        )
        base_quat_up = slerp(
            base_quat_src[i_lo], base_quat_src[i_lo + 1], alpha.squeeze(-1)
        )

        # ---- Finite-difference velocities at physics dt.
        dt = 1.0 / physics_fps

        joint_qd_up = torch.zeros_like(joint_q_up)
        joint_qd_up[1:] = (joint_q_up[1:] - joint_q_up[:-1]) / dt
        joint_qd_up[0] = joint_qd_up[1]                        # backward-extrapolate

        base_lin_vel = torch.zeros_like(base_pos_up)
        base_lin_vel[1:] = (base_pos_up[1:] - base_pos_up[:-1]) / dt
        base_lin_vel[0] = base_lin_vel[1]

        # Angular velocity (world frame): omega = (2/dt) * log(q_{t+1} * q_t^-1).xyz
        base_ang_vel = torch.zeros_like(base_pos_up)
        if T_up > 1:
            q_prev_inv = quat_conj(base_quat_up[:-1])
            q_delta = quat_mul(base_quat_up[1:], q_prev_inv)   # (T_up-1, 4)
            base_ang_vel[1:] = (2.0 / dt) * quat_log_xyz(q_delta)
            base_ang_vel[0] = base_ang_vel[1]

        # ---- Optional keypoint FK precompute.
        #
        # For each upsampled frame, build the full URDF joint config (actuated
        # joints from joint_q_up, passive neck joints at 0), run yourdfpy FK,
        # read each keypoint link's transform relative to base_link, then
        # compose with the frame's base pose to get world-frame positions.
        # Velocities are computed by finite difference over physics dt.
        #
        # Performance note: O(F * Kp) Python calls — acceptable for typical
        # motions (~79 frames in synthetic test). For very long motions
        # (thousands of frames), this could be a bottleneck; batched FK or
        # caching in numpy would help but is not implemented here.
        kp_pos_world = None
        kp_vel_world = None
        if keypoint_links is not None:
            Kp = len(keypoint_links)
            kp_pos_arr = np.zeros((T_up, Kp, 3), dtype=np.float32)

            # joint_q_up is (T_up, 23) on device — bring to CPU numpy for FK.
            joint_q_np = joint_q_up.cpu().numpy()
            base_pos_np = base_pos_up.cpu().numpy()
            base_quat_np = base_quat_up.cpu().numpy()  # xyzw

            for f in range(T_up):
                # Build per-actuated-joint config dict.
                cfg_dict = {name: float(joint_q_np[f, j])
                            for j, name in enumerate(actuated_names)}
                urdf.update_cfg(cfg_dict)

                # Base rotation matrix for this frame.
                qxyzw = base_quat_np[f]  # xyzw
                q_t = torch.tensor(qxyzw, dtype=torch.float32)
                R_base = quat_to_matrix(q_t).numpy()  # (3, 3)

                for k, link_name in enumerate(keypoint_links):
                    T_link = urdf.get_transform(link_name)  # 4x4, relative to base
                    link_pos_in_base = T_link[:3, 3]
                    kp_pos_arr[f, k] = base_pos_np[f] + R_base @ link_pos_in_base

            kp_pos_world = torch.tensor(kp_pos_arr, dtype=torch.float32, device=device)

            # Finite-difference velocities (physics dt).
            kp_vel = torch.zeros_like(kp_pos_world)
            if T_up > 1:
                kp_vel[1:] = (kp_pos_world[1:] - kp_pos_world[:-1]) / dt
                kp_vel[0] = kp_vel[1]
            kp_vel_world = kp_vel

        # ---- Ground normalization ----------------------------------------
        # Retargeting often leaves the robot floating; shift each clip in world-z
        # so the lowest foot sole over the trajectory rests at the floor (z=0).
        # Per-clip constant shift -> preserves vertical dynamics (jumps stay
        # jumps). Velocities/orientation are unaffected by a constant translation.
        if ground:
            if kp_pos_world is None:
                raise ValueError("ground=True requires keypoint_links (needs foot FK)")
            foot_idx = [i for i, n in enumerate(keypoint_links) if "ankle_roll" in n]
            if not foot_idx:
                raise ValueError(
                    "ground=True requires an *ankle_roll* keypoint (the foot); "
                    f"none found in {keypoint_links}")
            foot_z = kp_pos_world[:, foot_idx, 2]                     # (T_up, n_feet)
            lowest_sole = float(foot_z.min()) - _FOOT_SOLE_OFFSET     # sole is below the link
            base_pos_up[:, 2] -= lowest_sole
            kp_pos_world[:, :, 2] -= lowest_sole

        return cls(
            src_fps=src_fps,
            physics_fps=physics_fps,
            upsample=upsample,
            T_src=T_src,
            T_up=T_up,
            n_actuated=len(actuated_names),
            n_urdf_joints=n_urdf_joints,
            device=device,
            base_pos=base_pos_up,
            base_quat=base_quat_up,
            joint_q=joint_q_up,
            joint_qd=joint_qd_up,
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            actuated_idx=actuated_idx,
            passive_idx=passive_idx,
            actuated_names=actuated_names,
            _kp_pos_world=kp_pos_world,
            _kp_vel_world=kp_vel_world,
        )

    # -------------------------------------------------------------------------
    # Per-env access
    # -------------------------------------------------------------------------

    def at(
        self, idx: Union[list, torch.Tensor]
    ) -> dict:
        """Slice all per-frame fields at the given indices.

        Args:
            idx: Python list of ints or a LongTensor of shape (B,).

        Returns:
            dict with keys:
                base_pos     (B, 3)
                base_quat    (B, 4)  xyzw
                base_lin_vel (B, 3)
                base_ang_vel (B, 3)
                joint_q      (B, 23) actuated joints only
                joint_qd     (B, 23) actuated joints only
        """
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx, dtype=torch.long, device=self.device)
        return {
            "base_pos":     self.base_pos[idx],
            "base_quat":    self.base_quat[idx],
            "base_lin_vel": self.base_lin_vel[idx],
            "base_ang_vel": self.base_ang_vel[idx],
            "joint_q":      self.joint_q[idx],
            "joint_qd":     self.joint_qd[idx],
        }

    def sample_index(self, n: int, max_lookahead: int) -> torch.LongTensor:
        """Sample n random start indices leaving room for lookahead.

        Args:
            n:            Number of indices to sample.
            max_lookahead: Maximum number of future steps needed; sampled
                          indices are in [0, T_up - max_lookahead).

        Returns:
            LongTensor of shape (n,) on self.device.
        """
        hi = max(1, self.T_up - max_lookahead)
        return torch.randint(0, hi, (n,), device=self.device)

    # -------------------------------------------------------------------------
    # Keypoint accessors
    # -------------------------------------------------------------------------

    def keypoints_at(self, idx) -> tuple:
        """Return world-frame keypoint positions and velocities at given indices.

        Args:
            idx: Python list of ints or LongTensor of shape (B,).

        Returns:
            (pos, vel): each Tensor of shape (B, Kp, 3) on self.device.

        Raises:
            RuntimeError: If MotionRef was loaded without keypoint_links.
        """
        if self._kp_pos_world is None:
            raise RuntimeError(
                "Keypoint data not precomputed. "
                "Pass keypoint_links= when calling MotionRef.load()."
            )
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx, dtype=torch.long, device=self.device)
        return self._kp_pos_world[idx], self._kp_vel_world[idx]

    def keypoints_root_relative_at(self, idx) -> torch.Tensor:
        """Return keypoint positions relative to the base (pelvis) position.

        Args:
            idx: Python list of ints or LongTensor of shape (B,).

        Returns:
            Tensor of shape (B, Kp, 3): keypoint world pos minus base pos.
        """
        pos, _ = self.keypoints_at(idx)
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx, dtype=torch.long, device=self.device)
        root = self.at(idx)["base_pos"].unsqueeze(1)  # (B, 1, 3)
        return pos - root

    # -------------------------------------------------------------------------
    # RGMT command vector
    # -------------------------------------------------------------------------

    def command_at(self, idx: torch.Tensor) -> torch.Tensor:
        """Build the RGMT command vector g_t of dimension 55.

        Components:
            v_ref  (3): base linear velocity rotated into the ref body frame.
            w_ref  (3): base angular velocity rotated into the ref body frame.
            g_ref  (3): gravity direction expressed in the ref body frame.
            q_enc (46): cos/sin encoding of the 23 actuated ref joint angles.

        Args:
            idx: LongTensor of shape (B,) — frame indices into the upsampled
                 motion.

        Returns:
            Tensor of shape (B, 55).
        """
        f = self.at(idx)
        v_body = quat_rotate_inverse(f["base_quat"], f["base_lin_vel"])
        w_body = quat_rotate_inverse(f["base_quat"], f["base_ang_vel"])
        g_ref = gravity_projection(f["base_quat"])
        q_enc = encode_angles(f["joint_q"])
        return torch.cat([v_body, w_body, g_ref, q_enc], dim=-1)  # 3+3+3+46 = 55

    def command_window(self, idx: torch.Tensor, L: int) -> torch.Tensor:
        """Build a centered command window of half-width L.

        Collects command vectors at offsets -L, ..., 0, ..., +L relative to
        each index in ``idx``, clamping out-of-range indices to
        [0, n_frames-1].

        Args:
            idx: LongTensor of shape (B,).
            L:   Half-width of the window (non-negative integer).

        Returns:
            Tensor of shape (B, 2L+1, 55).
        """
        offs = torch.arange(-L, L + 1, device=idx.device)
        grid = (idx.unsqueeze(1) + offs.unsqueeze(0)).clamp(0, self.n_frames - 1)  # (B, S)
        flat = self.command_at(grid.reshape(-1))                                    # (B*S, 55)
        return flat.reshape(idx.shape[0], 2 * L + 1, -1)

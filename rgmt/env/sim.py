"""Newton physics wrapper for the vendored 25-joint Asimov humanoid.

A Newton integration layer built around this repo's vendored Asimov-v1 asset (`rgmt.assets.paths.ROBOT_XML`) and its
canonical joint layout (`rgmt.data.joint_map`).

The robot has a single freejoint plus 25 hinge joints (23 actuated + 2 passive
neck joints). Newton stores all per-world state in flat arrays
(`state.joint_q`, `state.joint_qd`, `state.body_q`, ...). We map between flat
and per-env tensors by slicing fixed-size blocks; the layout per env is:

    joint_q[e * nq_per_env : (e + 1) * nq_per_env]
        = [freejoint_qpos(7), hinge_q[0..24]]
    joint_qd[e * nqd_per_env : (e + 1) * nqd_per_env]
        = [freejoint_qvel(6), hinge_qd[0..24]]

The freejoint qpos in Newton is `(x, y, z, qx, qy, qz, qw)` (Warp's `quat`
type is xyzw). The freejoint qd layout is `(lin_vel(3), ang_vel(3))` — LINEAR
FIRST (empirically verified 2026-07-13; regression-tested in
tests/test_sim_conventions.py — the reverse assumption once silently swapped
every velocity in the pipeline, and this docstring stated it wrong for a
month after the fix).

`joint_q` / `joint_qd` expose all 25 hinge columns; `actuated_idx`
(LongTensor[23]) selects the actuated columns matching
`ASIMOV_ACTUATED_JOINT_NAMES`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import mujoco as _mj
import newton
import numpy as _np
import torch
import warp as wp

from rgmt.assets.paths import ROBOT_XML
from rgmt.data.joint_map import ASIMOV_ACTUATED_JOINT_NAMES


# ---------------------------------------------------------------------------
# mujoco / mujoco-warp / Newton compatibility shim.
#
# Newton's SolverMuJoCo converts a Newton Model into mujoco's MjSpec by calling
# `MjsBody.add_joint(stiffness=..., ref=..., damping=..., ...)`. The values
# come from Warp arrays (float32). In mujoco 3.8 the pybind11 binding for these
# kwargs accepts Python `float` and `np.float64` but rejects `np.float32` with
# `TypeError: stiffness should be a numeric scalar or list`.
#
# Wrap MjsBody.add_joint once globally to coerce float32 scalars to Python
# floats before they hit the binding. Idempotent and harmless on newer mujoco.
# ---------------------------------------------------------------------------

def _install_mjs_add_joint_patch() -> None:
    spec = _mj.MjSpec()
    body_cls = type(spec.worldbody.add_body())
    orig = body_cls.add_joint
    if getattr(orig, "_rgmt_patched", False):
        return

    def _coerce(v):
        if isinstance(v, _np.floating):
            return float(v)
        if isinstance(v, _np.integer):
            return int(v)
        if isinstance(v, _np.bool_):
            return bool(v)
        return v

    def add_joint(self, *args, **kwargs):
        kwargs = {k: _coerce(v) for k, v in kwargs.items()}
        return orig(self, *args, **kwargs)

    add_joint._rgmt_patched = True
    body_cls.add_joint = add_joint


_install_mjs_add_joint_patch()


def _joint_short_name(label: str) -> str:
    """Strip the URDF/MJCF path prefix from a Newton joint label."""
    return label.split("/")[-1]


def _expand_gain(g: Union[float, Sequence[float]], n: int, name: str) -> list[float]:
    """Normalise a PD gain to a length-``n`` per-joint list.

    A scalar broadcasts to every actuated joint; a sequence must already be
    length ``n`` (in ASIMOV_ACTUATED_JOINT_NAMES order)."""
    if isinstance(g, (int, float)):
        return [float(g)] * n
    vec = [float(x) for x in g]
    if len(vec) != n:
        raise ValueError(f"{name} must be a scalar or length {n} (n_actuated), got {len(vec)}")
    return vec


class NewtonSim:
    """Vectorized Asimov simulator. One model, `num_envs` worlds.

    Parameters
    ----------
    num_envs:
        Number of replicated robot worlds.
    kp, kd:
        Position-PD gains applied to the 23 actuated DoF.
    control_decimation:
        Number of physics substeps per `step()` call.
    dt:
        Physics timestep (seconds).
    foot_friction:
        Ground-contact friction coefficient (`default_shape_cfg.mu`).
    keypoint_links:
        Body names (suffix match against `model.body_key`) whose world pose
        and linear velocity are exposed via `keypoint_pos` / `keypoint_lin_vel`.
    device:
        Torch device; warp state lives on the matching device for zero-copy.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        kp: Union[float, Sequence[float]],
        kd: Union[float, Sequence[float]],
        control_decimation: int,
        dt: float,
        foot_friction: float,
        keypoint_links: list[str],
        device: Union[str, torch.device] = "cuda:0",
        effort_limit: Optional[Sequence[float]] = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        # kp/kd may be a scalar (uniform) or a per-actuated-joint vector; the
        # actual expansion to length n_actuated happens once the actuated-joint
        # count is known (below), so per-joint gains can be validated.
        self._kp_arg = kp
        self._kd_arg = kd
        self.foot_friction = float(foot_friction)
        self.control_decimation = int(control_decimation)
        self.keypoint_links = list(keypoint_links)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            _idx = self.device.index if self.device.index is not None else 0
            self._wp_device = f"cuda:{_idx}"
        else:
            self._wp_device = "cpu"
        # Pin the Warp device so model arrays and torch tensors share the same
        # device (required for wp.to_torch zero-copy).
        wp.set_device(self._wp_device)

        actuated_joint_names = list(ASIMOV_ACTUATED_JOINT_NAMES)

        # ---- Single-robot template ----------------------------------------
        single = newton.ModelBuilder()
        # newton 1.4+ defaults rigid_gap/shape gap to 0.1 m and propagates it
        # into MuJoCo geom_gap/pair_gap, creating detection-only contacts that
        # eat into njmax/nconmax at 8k envs. This robot never used gaps: keep 0.
        single.rigid_gap = 0.0
        single.default_shape_cfg.gap = 0.0
        single.add_mjcf(
            str(ROBOT_XML),
            floating=None,
            enable_self_collisions=False,    # MJCF's <contact><exclude> blocks
                                             # aren't reliably preserved through
                                             # Newton's converter; disable global
                                             # self-collision to keep step stable.
        )
        self._joint_names: list[str] = [_joint_short_name(l) for l in single.joint_label]
        self.n_joints_per_env: int = single.joint_count           # incl. freejoint
        self.nq_per_env: int = single.joint_coord_count           # per env
        self.nqd_per_env: int = single.joint_dof_count            # per env

        # Resolve actuated-joint DoF indices in the per-env DoF block.
        # joint_qd_start[i] is the DoF offset (within an env) where joint i's
        # velocities start; for a 1-DoF hinge that single index is its DoF idx.
        joint_qd_start = list(single.joint_qd_start)
        joint_q_start = list(single.joint_q_start)
        name_to_idx = {n: i for i, n in enumerate(self._joint_names)}
        actuated_dof_local: list[int] = []
        actuated_q_local: list[int] = []
        for n in actuated_joint_names:
            if n not in name_to_idx:
                raise KeyError(
                    f"actuated joint {n!r} not found in MJCF "
                    f"(joint names: {self._joint_names[:5]}...)"
                )
            j = name_to_idx[n]
            actuated_dof_local.append(joint_qd_start[j])
            actuated_q_local.append(joint_q_start[j])
        self.n_actuated: int = len(actuated_joint_names)
        self.actuated_joint_names: list[str] = list(actuated_joint_names)
        # Expand PD gains to per-actuated-joint vectors (scalar -> broadcast).
        self.kp_vec = _expand_gain(self._kp_arg, self.n_actuated, "kp")
        self.kd_vec = _expand_gain(self._kd_arg, self.n_actuated, "kd")
        # Representative scalars kept for logging / back-compat callers.
        self.kp = self.kp_vec[0]
        self.kd = self.kd_vec[0]
        self.actuated_dof_local = torch.tensor(
            actuated_dof_local, dtype=torch.long, device=self.device
        )

        # ---- Configure PD per joint --------------------------------------
        # Zero PD on every DoF, then set kp/kd/mode on the actuated DoF only.
        # Passive joints evolve under the armature/damping from the MJCF loader.
        for d in range(single.joint_dof_count):
            single.joint_target_ke[d] = 0.0
            single.joint_target_kd[d] = 0.0
            single.joint_target_mode[d] = int(newton.JointTargetMode.NONE)
        for i, d in enumerate(actuated_dof_local):
            single.joint_target_ke[d] = self.kp_vec[i]
            single.joint_target_kd[d] = self.kd_vec[i]
            single.joint_target_mode[d] = int(newton.JointTargetMode.POSITION)

        # ---- Torque limits ------------------------------------------------
        # None = uncapped (builder default 1e6; checkpoints trained without
        # caps assume it). A length-23 vector (ASIMOV order, Nm) becomes
        # MuJoCo jnt_actfrcrange, clamping the total P+D actuator force.
        self.effort_limit_vec: Optional[list[float]] = None
        if effort_limit is not None:
            lim = [float(x) for x in effort_limit]
            if len(lim) != self.n_actuated:
                raise ValueError(
                    f"effort_limit must be length {self.n_actuated}, got {len(lim)}")
            if any(x <= 0.0 for x in lim):
                raise ValueError("effort_limit entries must be positive")
            self.effort_limit_vec = lim
            for i, d in enumerate(actuated_dof_local):
                single.joint_effort_limit[d] = lim[i]

        # ---- Replicate, ground plane, finalize ----------------------------
        builder = newton.ModelBuilder()
        # Contact stiffness/damping/friction defaults matched to the H1
        # humanoid example — Newton's out-of-box ke is too high for the
        # Asimov's thin foot capsules and the contact solver explodes within a
        # few steps of ground touchdown.
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = self.foot_friction
        builder.rigid_gap = 0.0
        builder.default_shape_cfg.gap = 0.0
        builder.replicate(single, self.num_envs)
        builder.add_ground_plane()
        self.model = builder.finalize()

        # newton 1.5 rewrote replicate() as a single-pass merge. Everything
        # downstream (per-env DoF/body slicing, keypoint resolution, DR index
        # grids) assumes env blocks are contiguous and in env order — assert it.
        n_body = int(self.model.body_count)
        if n_body % self.num_envs != 0:
            raise RuntimeError(
                f"replicate() produced {n_body} bodies not divisible by "
                f"num_envs={self.num_envs} — per-env block layout broken")
        self.bodies_per_env = n_body // self.num_envs
        body_labels = self._body_labels()
        if self.num_envs > 1:
            b0 = body_labels[0].split("/")[-1]
            b1 = body_labels[self.bodies_per_env].split("/")[-1]
            if b0 != b1:
                raise RuntimeError(
                    "replicate() body layout is not [env0 block][env1 block]... "
                    f"(body 0 = {body_labels[0]!r}, body {self.bodies_per_env} "
                    f"= {body_labels[self.bodies_per_env]!r})")

        # ---- Solver + state ----------------------------------------------
        # SolverMuJoCo is the most accurate articulated-robot backend but needs
        # the mujoco_warp extension; fall back to SolverFeatherstone otherwise.
        try:
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                iterations=100,
                ls_iterations=50,
                # njmax raised 200 -> 256: at 2048 envs with recovery poses the
                # constraint count overflowed (nefc ~220), silently truncating
                # contacts. GPU headroom exists (peak ~4.8/8 GB at 2048 envs).
                njmax=256,
                nconmax=400,
            )
        except (ImportError, RuntimeError, TypeError):
            self.solver = newton.solvers.SolverFeatherstone(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Guard the PD-target buffer layout (review R2). actuated_dof_global
        # indexes joint_target_q with DOF indices and an nqd stride, which is
        # only correct in Newton's legacy DOF layout. The flagged future
        # default (newton.use_coord_layout_targets) switches the buffer to
        # COORD shape (nq != nqd with a freejoint: 32 vs 31 here) — every
        # target would silently shift one slot per env. Fail loudly instead.
        _tgt_len = len(self.control.joint_target_q)
        if _tgt_len != self.num_envs * self.nqd_per_env:
            raise RuntimeError(
                f"Control.joint_target_q has {_tgt_len} entries, expected "
                f"num_envs*nqd = {self.num_envs * self.nqd_per_env}. Newton's "
                "target layout changed (use_coord_layout_targets?); the DOF-"
                "indexed PD write path in step() must be migrated first.")

        # Initialise state from the model's joint_q / joint_qd (rest pose).
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ---- Pre-compute index buffers for fast per-env writes -----------
        env_qd_offsets = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            * self.nqd_per_env
        )
        env_q_offsets = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            * self.nq_per_env
        )
        self.actuated_dof_global = (
            env_qd_offsets.unsqueeze(1) + self.actuated_dof_local.unsqueeze(0)
        )  # (num_envs, n_actuated)

        # Hinge slices (all non-free joints, in MJCF order = 25 columns).
        hinge_q_local: list[int] = []
        hinge_qd_local: list[int] = []
        self.hinge_joint_indices: list[int] = []
        hinge_names: list[str] = []
        for i, jt in enumerate(single.joint_type):
            if jt == newton.JointType.FREE:
                continue
            hinge_q_local.append(joint_q_start[i])
            hinge_qd_local.append(joint_qd_start[i])
            self.hinge_joint_indices.append(i)
            hinge_names.append(self._joint_names[i])
        self.n_hinges: int = len(hinge_q_local)
        self.hinge_names: list[str] = hinge_names
        self.hinge_q_local = torch.tensor(hinge_q_local, dtype=torch.long, device=self.device)
        self.hinge_qd_local = torch.tensor(hinge_qd_local, dtype=torch.long, device=self.device)
        self.hinge_q_global = env_q_offsets.unsqueeze(1) + self.hinge_q_local.unsqueeze(0)
        self.hinge_qd_global = env_qd_offsets.unsqueeze(1) + self.hinge_qd_local.unsqueeze(0)

        # actuated_idx: index into the 25 hinge columns selecting the 23
        # actuated joints (in ASIMOV_ACTUATED_JOINT_NAMES order).
        hinge_name_to_col = {n: c for c, n in enumerate(hinge_names)}
        actuated_cols = [hinge_name_to_col[n] for n in actuated_joint_names]
        self.actuated_idx = torch.tensor(
            actuated_cols, dtype=torch.long, device=self.device
        )

        # ---- MJCF joint ref compensation for actuator targets -------------
        # The SolverMuJoCo position servo settles at (target - ref) in Newton
        # joint coordinates: MuJoCo measures qpos relative to each joint's
        # `ref` attribute, while Newton's joint_q (and the URDF-derived
        # reference motions) use the raw body-frame zero. Both elbows carry
        # ref=±0.785398 in asimov.xml, so commanding the reference angle left
        # them 45° off — beyond the ±0.5 rad residual-action authority, i.e.
        # uncorrectable by the policy (empirically confirmed 2026-07-13:
        # target 0 settled at exactly ±0.7856 rad; all ref=0 joints settled
        # at target). Compensate by adding ref to every incoming target so
        # `step(t)` settles at t for all joints.
        import xml.etree.ElementTree as _ET
        _refs = {
            j.get("name"): float(j.get("ref", "0.0") or 0.0)
            for j in _ET.parse(str(ROBOT_XML)).getroot().iter("joint")
            if j.get("name")
        }
        self._target_ref_offset = torch.tensor(
            [_refs.get(n, 0.0) for n in actuated_joint_names],
            dtype=torch.float32, device=self.device,
        )  # (n_actuated,)

        # Freejoint slice indices.
        self.freejoint_q_global = env_q_offsets.unsqueeze(1) + torch.arange(
            7, device=self.device, dtype=torch.long
        ).unsqueeze(0)  # (num_envs, 7)
        self.freejoint_qd_global = env_qd_offsets.unsqueeze(1) + torch.arange(
            6, device=self.device, dtype=torch.long
        ).unsqueeze(0)  # (num_envs, 6)

        # ---- Keypoint body indices ---------------------------------------
        # Resolve each keypoint link name to a per-env Newton body index by
        # suffix-matching model.body_key (mirrors _resolve_pelvis_indices).
        self.keypoint_body_idx = self._resolve_keypoint_indices(self.keypoint_links)

        # ---- External (pelvis) force buffer ------------------------------
        # Per-env world-frame force applied to the pelvis body each substep
        # (fall-recovery assist). Default zero -> no behaviour change.
        # `pelvis_body_idx` is the per-env global Newton body index of the
        # pelvis link, resolved by the same suffix-match used for keypoints.
        self.pelvis_body_idx = self._resolve_keypoint_indices(
            ["pelvis_link"]
        )[:, 0]  # (num_envs,)
        self._external_force = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float32
        )

        # ---- Domain-randomization write surface ---------------------------
        # Per-env dynamics parameters live in the flat Newton Model arrays
        # (replicate() lays envs out contiguously). Setters below write those
        # arrays in place (wp.to_torch views, zero-copy) and OR a ModelFlags
        # bit into a dirty accumulator; nothing reaches the MuJoCo solver until
        # apply_dynamics_changes() issues ONE notify_model_changed per training
        # iteration. BODY_INERTIAL/JOINT_DOF notifies trigger a full mass-
        # matrix constant re-derivation (set_const_0) over all worlds, so
        # per-reset or per-setter notifies would be ruinously expensive.
        self._nominal_body_mass = self._torch(self.model.body_mass).clone()
        self._nominal_body_inv_mass = self._torch(self.model.body_inv_mass).clone()
        self._nominal_body_inertia = self._torch(self.model.body_inertia).clone()
        self._nominal_body_inv_inertia = self._torch(self.model.body_inv_inertia).clone()
        self._nominal_joint_target_ke = self._torch(self.model.joint_target_ke).clone()
        self._nominal_joint_target_kd = self._torch(self.model.joint_target_kd).clone()
        self._nominal_joint_effort_limit = self._torch(self.model.joint_effort_limit).clone()
        # (num_envs, bodies_per_env) global body-index grid; robot bodies only
        # (the ground plane is body-less, so the flat layout is exactly env
        # blocks — asserted after finalize()).
        self.body_idx_grid = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            .unsqueeze(1) * self.bodies_per_env
            + torch.arange(self.bodies_per_env, device=self.device, dtype=torch.long)
            .unsqueeze(0)
        )
        # (num_envs, 8) shape-index grid over the foot contact spheres. These
        # geoms carry MJCF priority=1, so their mu alone decides foot-ground
        # friction (the ground plane's mu — foot_friction — is ignored).
        _foot_geoms = [f"{side}_foot{i}_collision"
                       for side in ("left", "right") for i in (1, 2, 3, 4)]
        self.foot_shape_idx = self._resolve_shape_indices(_foot_geoms)
        self._dyn_dirty_flags: int = 0

    # ----------------------------------------------------------------------
    # Domain randomization setters (see block comment in __init__)
    # ----------------------------------------------------------------------

    def _check_per_env(self, t: torch.Tensor, name: str) -> torch.Tensor:
        if t.shape != (self.num_envs,):
            raise ValueError(f"{name} must be shape ({self.num_envs},), got {tuple(t.shape)}")
        return t.to(device=self.device, dtype=torch.float32)

    def set_foot_friction_per_env(self, mu: torch.Tensor) -> None:
        """Per-env foot-sphere friction coefficient (num_envs,)."""
        mu = self._check_per_env(mu, "mu")
        flat = self._torch(self.model.shape_material_mu)
        flat[self.foot_shape_idx] = mu.unsqueeze(1)
        self._dyn_dirty_flags |= int(newton.ModelFlags.SHAPE_PROPERTIES)

    def set_body_mass_scale(self, scale: torch.Tensor) -> None:
        """Whole-robot mass/inertia scale per env (num_envs,)."""
        s = self._check_per_env(scale, "scale")
        if bool((s <= 0).any()):
            raise ValueError("mass scale must be positive")
        per_body = s.unsqueeze(1)                      # (N, 1)
        mass = self._torch(self.model.body_mass)
        inv_mass = self._torch(self.model.body_inv_mass)
        inertia = self._torch(self.model.body_inertia)
        inv_inertia = self._torch(self.model.body_inv_inertia)
        g = self.body_idx_grid
        mass[g] = self._nominal_body_mass[g] * per_body
        inv_mass[g] = self._nominal_body_inv_mass[g] / per_body
        pb3 = per_body.unsqueeze(-1).unsqueeze(-1)     # (N, 1, 1, 1)
        inertia[g] = self._nominal_body_inertia[g] * pb3
        inv_inertia[g] = self._nominal_body_inv_inertia[g] / pb3
        self._dyn_dirty_flags |= int(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)

    def set_joint_gain_scale(self, kp_scale: torch.Tensor, kd_scale: torch.Tensor) -> None:
        """Per-env multiplicative PD gain scales on the actuated DoF (num_envs,)."""
        kp_s = self._check_per_env(kp_scale, "kp_scale").unsqueeze(1)
        kd_s = self._check_per_env(kd_scale, "kd_scale").unsqueeze(1)
        ke = self._torch(self.model.joint_target_ke)
        kd = self._torch(self.model.joint_target_kd)
        g = self.actuated_dof_global
        ke[g] = self._nominal_joint_target_ke[g] * kp_s
        kd[g] = self._nominal_joint_target_kd[g] * kd_s
        self._dyn_dirty_flags |= int(newton.ModelFlags.JOINT_DOF_PROPERTIES)

    def set_effort_limit_scale(self, scale: torch.Tensor) -> None:
        """Per-env multiplicative torque-limit scale on the actuated DoF.

        Only meaningful when the sim was built with an explicit effort_limit
        (scaling the legacy 1e6 default is a no-op in practice, but allowed).
        """
        s = self._check_per_env(scale, "scale").unsqueeze(1)
        eff = self._torch(self.model.joint_effort_limit)
        g = self.actuated_dof_global
        eff[g] = self._nominal_joint_effort_limit[g] * s
        self._dyn_dirty_flags |= int(newton.ModelFlags.JOINT_DOF_PROPERTIES)

    def apply_dynamics_changes(self) -> int:
        """Flush accumulated DR writes to the solver with ONE notify.

        Returns the flag mask that was applied (0 = nothing pending). No-op on
        solvers without notify support (SolverFeatherstone fallback)."""
        flags = self._dyn_dirty_flags
        if not flags:
            return 0
        self._dyn_dirty_flags = 0
        if hasattr(self.solver, "notify_model_changed"):
            self.solver.notify_model_changed(flags)
        return flags

    def _resolve_shape_indices(self, names: list[str]) -> torch.Tensor:
        """Map shape/geom names to a (num_envs, len(names)) global index grid.

        Mirrors _resolve_keypoint_indices: '/'-anchored suffix match against
        the model shape labels, exactly num_envs hits per name, k-th hit = env k.
        """
        labels: list[str] = []
        for attr in ("shape_label", "shape_key"):
            if hasattr(self.model, attr):
                labels = list(getattr(self.model, attr))
                break
        cols: list[torch.Tensor] = []
        for name in names:
            hits = [i for i, k in enumerate(labels)
                    if k.endswith("/" + name) or k == name]
            if len(hits) != self.num_envs:
                raise ValueError(
                    f"shape {name!r}: expected {self.num_envs} matches, got {len(hits)}")
            cols.append(torch.tensor(hits, dtype=torch.long, device=self.device))
        return torch.stack(cols, dim=1)  # (num_envs, len(names))

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    def _body_labels(self) -> list[str]:
        """Body name strings for the finalized model (Newton path labels)."""
        for attr in ("body_label", "body_key"):
            if hasattr(self.model, attr):
                return list(getattr(self.model, attr))
        return []

    def _resolve_body_indices(self, suffix: str) -> list[int]:
        """All body ids whose path label ends in '/<suffix>' (or equals it).

        Body labels are nested paths like
        'Asimov/worldbody/pelvis_link/left_hip_pitch_link', so a '/'-anchored
        suffix match selects exactly the requested link without over-matching
        deeper descendants.
        """
        hits: list[int] = []
        for i, k in enumerate(self._body_labels()):
            if k.endswith("/" + suffix) or k == suffix:
                hits.append(i)
        return hits

    def _resolve_keypoint_indices(self, names: list[str]) -> torch.Tensor:
        """Map keypoint link names to a (num_envs, Kp) global body-index grid.

        For each link name we expect exactly `num_envs` suffix matches (one per
        replicated world). Newton lays replicated bodies out contiguously, so
        the k-th hit corresponds to env k.
        """
        cols: list[torch.Tensor] = []
        for name in names:
            hits = self._resolve_body_indices(name)
            if len(hits) != self.num_envs:
                raise KeyError(
                    f"keypoint link {name!r}: expected {self.num_envs} body "
                    f"matches, found {len(hits)} (labels sample: "
                    f"{self._body_labels()[:6]})"
                )
            cols.append(torch.tensor(hits, dtype=torch.long, device=self.device))
        # (num_envs, Kp)
        return torch.stack(cols, dim=1)

    # ----------------------------------------------------------------------
    # Zero-copy torch views into Newton state
    # ----------------------------------------------------------------------

    def _torch(self, arr: wp.array) -> torch.Tensor:
        return wp.to_torch(arr)

    @property
    def _joint_q_flat(self) -> torch.Tensor:
        return self._torch(self.state_0.joint_q)

    @property
    def _joint_qd_flat(self) -> torch.Tensor:
        return self._torch(self.state_0.joint_qd)

    @property
    def _body_q_flat(self) -> torch.Tensor:
        # body_q rows are transforms (x, y, z, qx, qy, qz, qw).
        return self._torch(self.state_0.body_q)

    @property
    def _body_qd_flat(self) -> torch.Tensor:
        # body_qd rows are spatial velocities (lin(3), ang(3)) — LINEAR FIRST,
        # same as the freejoint qd (see keypoint_lin_vel; verified 2026-07-13).
        return self._torch(self.state_0.body_qd)

    @property
    def _ctrl_target_flat(self) -> torch.Tensor:
        # DOF-laid-out target buffer (num_envs * nqd entries — NOT coord-
        # shaped; layout guarded by the __init__ assertion, review R2). For
        # this robot the actuated joints are all 1-DoF hinges, so their DoF
        # index (joint_qd_start) is their slot. newton 1.5 removed the
        # joint_target_pos alias, so a missing joint_target_q is a hard error.
        ctrl = self.control
        if not hasattr(ctrl, "joint_target_q"):
            raise AttributeError(
                "newton>=1.5 expected Control.joint_target_q "
                "(joint_target_pos alias was removed upstream)"
            )
        return self._torch(ctrl.joint_target_q)

    # ----- Public per-env views (read-only convenience) -----

    @property
    def joint_q(self) -> torch.Tensor:
        """All hinge joint angles per env: (num_envs, 25)."""
        return self._joint_q_flat[self.hinge_q_global]

    @property
    def joint_qd(self) -> torch.Tensor:
        """All hinge joint velocities per env: (num_envs, 25)."""
        return self._joint_qd_flat[self.hinge_qd_global]

    @property
    def base_pos(self) -> torch.Tensor:
        """Pelvis world position from freejoint qpos: (num_envs, 3)."""
        return self._joint_q_flat[self.freejoint_q_global][:, :3]

    @property
    def base_quat(self) -> torch.Tensor:
        """Pelvis world quaternion xyzw from freejoint qpos: (num_envs, 4)."""
        return self._joint_q_flat[self.freejoint_q_global][:, 3:7]

    @property
    def base_lin_vel(self) -> torch.Tensor:
        """Pelvis linear velocity in world frame: (num_envs, 3)."""
        # Newton freejoint qd layout is (lin_vel(3), ang_vel(3)) — LINEAR
        # FIRST, same convention as body_f's (force, torque). The old
        # (ang, lin) assumption swapped these views AND the reset writes
        # self-consistently for the project's entire history: the policy's
        # "ang_vel" obs was actually linear velocity (actor blind to true
        # angular velocity), and RSI resets injected the reference's walking
        # speed as spin while dropping its forward momentum. Verified
        # empirically 2026-07-13 (translate-vs-rotate probe; regression test
        # in tests/test_sim_conventions.py). NOTE: fixing this changes obs
        # semantics — checkpoints trained before the fix are invalidated.
        return self._joint_qd_flat[self.freejoint_qd_global][:, 0:3]

    @property
    def base_ang_vel(self) -> torch.Tensor:
        """Pelvis angular velocity in world frame: (num_envs, 3)."""
        return self._joint_qd_flat[self.freejoint_qd_global][:, 3:6]

    @property
    def keypoint_pos(self) -> torch.Tensor:
        """World positions of keypoint links: (num_envs, Kp, 3)."""
        # body_q rows: (x, y, z, qx, qy, qz, qw) — take translation.
        return self._body_q_flat[self.keypoint_body_idx][..., 0:3]

    @property
    def keypoint_lin_vel(self) -> torch.Tensor:
        """World linear velocities of keypoint links: (num_envs, Kp, 3)."""
        # body_qd rows: (lin(3), ang(3)) — linear FIRST, matching the
        # freejoint qd convention (verified empirically 2026-07-13). The old
        # [3:6] slice returned ANGULAR velocity, so the r_kpv reward compared
        # rad/s against m/s references for the project's entire history.
        return self._body_qd_flat[self.keypoint_body_idx][..., 0:3]

    # ----------------------------------------------------------------------
    # Step / reset
    # ----------------------------------------------------------------------

    def set_external_force(self, force: torch.Tensor) -> None:
        """Set a per-env world-frame force applied to the pelvis body each step.

        Args:
            force: (num_envs, 3) tensor of world-frame linear forces (N). The
                force persists across steps until overwritten; pass zeros to
                disable. Stored as a copy so the caller may reuse its buffer.
        """
        if force.shape != (self.num_envs, 3):
            raise ValueError(
                f"external force shape {tuple(force.shape)} != ({self.num_envs}, 3)"
            )
        self._external_force.copy_(force.to(self._external_force.dtype))

    def _apply_external_force(self) -> None:
        """Write the stored pelvis force into state_0.body_f (after clear_forces).

        Newton's ``state.body_f`` is a wp.array of spatial vectors, a 6D wrench
        per world body in WORLD frame applied at the body's COM. Per Newton's
        own docs (``newton/_src/sim/state.py`` body_f docstring and
        ``control.py`` joint_f convention), the layout is
        ``(f_x, f_y, f_z, t_x, t_y, t_z)`` — the LINEAR force is the FIRST three
        components ``[0:3]`` and the torque is the last three ``[3:6]``.
        (Verified empirically: writing a +z value into index 2 lifts the pelvis;
        writing into the torque slots only spins/destabilises it.) We therefore
        set only the LINEAR part ``[0:3]`` for each env's pelvis body.
        """
        body_f = wp.to_torch(self.state_0.body_f)  # (n_bodies_total, 6) zero-copy
        body_f[self.pelvis_body_idx, 0:3] = self._external_force.to(body_f.dtype)

    def step(self, actuated_target_pos: torch.Tensor) -> None:
        """Apply PD targets to actuated joints and advance physics.

        actuated_target_pos: (num_envs, n_actuated) torch tensor on self.device.
        """
        if actuated_target_pos.shape != (self.num_envs, self.n_actuated):
            raise ValueError(
                f"actuated_target_pos shape {tuple(actuated_target_pos.shape)} "
                f"!= ({self.num_envs}, {self.n_actuated})"
            )
        targets_flat = self._ctrl_target_flat
        # Add per-joint MJCF ref offsets so the servo settles at the COMMANDED
        # angle in Newton/URDF coordinates (see _target_ref_offset above).
        compensated = actuated_target_pos + self._target_ref_offset
        targets_flat[self.actuated_dof_global] = compensated.to(targets_flat.dtype)

        # collide + substep loop. Substeps = control_decimation.
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.control_decimation):
            self.state_0.clear_forces()
            self._apply_external_force()
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def reset_idx(
        self,
        env_ids: torch.Tensor,
        base_pos: torch.Tensor,
        base_quat_xyzw: torch.Tensor,
        base_lin_vel: torch.Tensor,
        base_ang_vel: torch.Tensor,
        hinge_q: torch.Tensor,
        hinge_qd: torch.Tensor,
    ) -> None:
        """Overwrite per-env state for the given envs.

        Shapes (M envs to reset):
          env_ids:        (M,) long
          base_pos:       (M, 3)
          base_quat_xyzw: (M, 4)
          base_lin_vel:   (M, 3)
          base_ang_vel:   (M, 3)
          hinge_q:        (M, 25)
          hinge_qd:       (M, 25)
        """
        env_ids = env_ids.to(self.device, dtype=torch.long)
        jq = self._joint_q_flat
        jqd = self._joint_qd_flat

        # Freejoint qpos: (x, y, z, qx, qy, qz, qw)
        fq_idx = self.freejoint_q_global[env_ids]                  # (M, 7)
        jq[fq_idx[:, 0:3]] = base_pos.to(jq.dtype)
        jq[fq_idx[:, 3:7]] = base_quat_xyzw.to(jq.dtype)

        # Freejoint qd: (lin(3), ang(3)) — linear first (see base_lin_vel).
        fqd_idx = self.freejoint_qd_global[env_ids]                # (M, 6)
        jqd[fqd_idx[:, 0:3]] = base_lin_vel.to(jqd.dtype)
        jqd[fqd_idx[:, 3:6]] = base_ang_vel.to(jqd.dtype)

        # Hinge qpos/qd (all 25 columns).
        hq_idx = self.hinge_q_global[env_ids]                      # (M, 25)
        hqd_idx = self.hinge_qd_global[env_ids]
        jq[hq_idx] = hinge_q.to(jq.dtype)
        jqd[hqd_idx] = hinge_qd.to(jqd.dtype)

        # Re-evaluate forward kinematics so body_q / body_qd reflect the new
        # joint state (otherwise the next step starts from stale body poses).
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

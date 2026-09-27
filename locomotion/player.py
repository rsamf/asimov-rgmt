"""Playback of corpus frames through a world anchor with inertialized switches.

anchor: planar rigid transform (t, theta):  world_xy = R(theta) clip_xy + t,
        world_yaw = clip_yaw + theta.  Set on every switch so the new clip's
        root lands exactly on the character's current world xy and facing.
offset: inertialization state recorded at a switch (last emitted - raw new)
        for joint angles, root height and root tilt, each with a rate; decays
        with a critically damped spring and is added to the raw clip pose.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from rgmt.data.corpus import MotionCorpus
from locomotion.config import BlendConfig
from locomotion.geom import (quat_conj, quat_from_yaw, quat_mul, quat_to_rotvec, rot2, rot3z,
                             rotvec_to_quat, wrap_angle, yaw_from_quat)
from locomotion.trajectory import decay_spring_damper_exact

KEYS = ("base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "joint_q", "joint_qd")


class Player:
    def __init__(self, corpus: MotionCorpus, cfg: BlendConfig, fps: int, device):
        self.corpus = corpus
        self.cfg = cfg
        self.dt = 1.0 / float(fps)
        self.device = torch.device(device)
        self.frame = 0
        self._emit_current = True
        self._t = torch.zeros(2, device=self.device)
        self._theta = 0.0
        self.last: Optional[Dict[str, torch.Tensor]] = None
        self._zero_offsets()

    # ---- state ----------------------------------------------------------------
    def _zero_offsets(self) -> None:
        z = lambda n: torch.zeros(n, device=self.device)  # noqa: E731
        self._off_q, self._off_qd = z(23), z(23)
        self._off_z, self._off_vz = 0.0, 0.0
        self._off_rot, self._off_w = z(3), z(3)

    @property
    def clip_end(self) -> int:
        return int(self.corpus.clip_end[self.corpus.frame_clip_id[self.frame]])

    @property
    def next_frame(self) -> int:
        return self.frame if self._emit_current else min(self.frame + 1, self.clip_end)

    @property
    def world_xy(self) -> torch.Tensor:
        return self.last["base_pos"][0, :2]

    @property
    def world_yaw(self) -> float:
        return float(yaw_from_quat(self.last["base_quat"][0]))

    # ---- anchor -----------------------------------------------------------------
    def _clip_root(self, frame: int):
        f = self.corpus.at(torch.tensor([frame], device=self.device))
        return f["base_pos"][0, :2], float(yaw_from_quat(f["base_quat"][0]))

    def _set_anchor(self, frame: int, xy, yaw: float) -> None:
        cxy, cyaw = self._clip_root(frame)
        self._theta = wrap_angle(float(yaw) - cyaw)
        xy_t = torch.as_tensor(xy, dtype=torch.float32, device=self.device)
        self._t = xy_t - rot2(self._theta, like=cxy) @ cxy

    def map_frames(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Corpus frames -> world frame through the anchor (no blending)."""
        f = self.corpus.at(idx.to(self.device))
        R2 = rot2(self._theta, like=f["base_pos"])
        R3 = rot3z(self._theta, like=f["base_pos"])
        pos = f["base_pos"].clone()
        pos[:, :2] = f["base_pos"][:, :2] @ R2.T + self._t
        qz = quat_from_yaw(self._theta, like=f["base_quat"]).expand_as(f["base_quat"])
        return {
            "base_pos": pos,
            "base_quat": quat_mul(qz, f["base_quat"]),
            "base_lin_vel": f["base_lin_vel"] @ R3.T,
            "base_ang_vel": f["base_ang_vel"] @ R3.T,
            "joint_q": f["joint_q"], "joint_qd": f["joint_qd"],
        }

    def _raw(self, frame: int) -> Dict[str, torch.Tensor]:
        return self.map_frames(torch.tensor([frame], device=self.device))

    # ---- control ----------------------------------------------------------------
    def reset(self, frame: int, xy=(0.0, 0.0), yaw: float = 0.0) -> None:
        self.frame = int(frame)
        self._set_anchor(self.frame, xy, yaw)
        self._zero_offsets()
        self._emit_current = True
        self.last = {k: v.clone() for k, v in self._raw(self.frame).items()}

    def switch(self, frame: int) -> None:
        """Jump to ``frame``; the next advance emits it, continuous with ``last``."""
        assert self.last is not None, "reset() before switch()"
        xy, yaw = self.world_xy.clone(), self.world_yaw
        last = self.last
        self.frame = int(frame)
        self._set_anchor(self.frame, xy, yaw)
        raw = self._raw(self.frame)
        self._off_q = last["joint_q"][0] - raw["joint_q"][0]
        self._off_qd = last["joint_qd"][0] - raw["joint_qd"][0]
        self._off_z = float(last["base_pos"][0, 2] - raw["base_pos"][0, 2])
        self._off_vz = float(last["base_lin_vel"][0, 2] - raw["base_lin_vel"][0, 2])
        self._off_rot = quat_to_rotvec(quat_mul(last["base_quat"][0], quat_conj(raw["base_quat"][0])))
        self._off_w = last["base_ang_vel"][0] - raw["base_ang_vel"][0]
        self._emit_current = True

    def re_anchor(self, xy, yaw: float) -> None:
        """Physics-mode hook: make the current frame's root coincide with an observed root."""
        self._set_anchor(self.frame, xy, yaw)
        if self.last is not None:
            raw = self._raw(self.frame)
            self.last["base_pos"][0, :2] = raw["base_pos"][0, :2]
            self.last["base_quat"][0] = quat_mul(rotvec_to_quat(self._off_rot), raw["base_quat"][0])

    def advance(self) -> Dict[str, torch.Tensor]:
        if self._emit_current:
            self._emit_current = False
        else:
            self.frame = min(self.frame + 1, self.clip_end)
        raw = self._raw(self.frame)
        out = {k: v.clone() for k, v in raw.items()}
        out["joint_q"][0] += self._off_q
        out["joint_qd"][0] += self._off_qd
        out["base_pos"][0, 2] += self._off_z
        out["base_lin_vel"][0, 2] += self._off_vz
        out["base_quat"][0] = quat_mul(rotvec_to_quat(self._off_rot), raw["base_quat"][0])
        out["base_ang_vel"][0] += self._off_w
        self.last = out
        # decay for the next tick (this tick's emission used the undecayed offset)
        hl = self.cfg.halflife
        self._off_q, self._off_qd = decay_spring_damper_exact(self._off_q, self._off_qd, hl, self.dt)
        self._off_z, self._off_vz = decay_spring_damper_exact(self._off_z, self._off_vz, hl, self.dt)
        self._off_rot, self._off_w = decay_spring_damper_exact(self._off_rot, self._off_w, hl, self.dt)
        return {k: v.clone() for k, v in out.items()}

    def preview(self, k: int) -> Dict[str, torch.Tensor]:
        """Next k raw frames of the current clip through the anchor, clamped at the clip end."""
        start = self.next_frame
        idx = torch.arange(start, start + k, device=self.device).clamp(max=self.clip_end)
        return self.map_frames(idx)

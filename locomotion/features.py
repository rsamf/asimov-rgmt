"""Motion-matching feature database (Holden 2021 feature set).

Per corpus frame, all in that frame's heading frame and relative to the root
ground point:  root linear velocity (3) | left, right foot position (6) |
left, right foot velocity (6) | future root position at each horizon (2H) |
future facing direction at each horizon (2H).  Frames without a full horizon
of future inside their own clip are excluded from search (``valid_idx``) but
keep their features so a query can be built from any playing frame.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from rgmt.data.corpus import MotionCorpus
from locomotion.config import FeaturesConfig
from locomotion.geom import rot2, yaw_from_quat

FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")
POSE_DIM = 15


class FeatureDB:
    def __init__(self, *, horizons: List[int], raw: torch.Tensor, valid_idx: torch.Tensor,
                 mean: torch.Tensor, scale: torch.Tensor, weight: torch.Tensor,
                 clip_id: torch.Tensor, room: torch.Tensor):
        self.horizons = list(horizons)
        self.raw = raw
        self.valid_idx = valid_idx
        self.mean, self.scale, self.weight = mean, scale, weight
        self.n_frames, self.dim = raw.shape
        self.pose_dim = POSE_DIM
        self.device = raw.device
        self._valid_pos = torch.full((self.n_frames,), -1, dtype=torch.long, device=self.device)
        self._valid_pos[valid_idx] = torch.arange(valid_idx.numel(), device=self.device)
        self.feats = self.normalize(raw[valid_idx])
        self._clip_id = clip_id                       # (N,) clip of every frame
        self._room_valid = room[valid_idx]            # (M,) frames left in the clip after each valid frame
        self._clip_valid = clip_id[valid_idx]         # (M,)

    # ---- construction ------------------------------------------------------
    @classmethod
    def build(cls, corpus: MotionCorpus, cfg: FeaturesConfig, device) -> "FeatureDB":
        dev = torch.device(device)
        H = [int(h) for h in cfg.horizons]
        hmax = max(H)
        pos = corpus.base_pos.to(dev)
        quat = corpus.base_quat.to(dev)
        lin = corpus.base_lin_vel.to(dev)
        kp_pos = corpus.kp_pos_world.to(dev)
        kp_vel = corpus.kp_vel_world.to(dev)
        li, ri = (corpus.keypoint_links.index(n) for n in FOOT_LINKS)
        N = pos.shape[0]

        yaw = yaw_from_quat(quat)                                  # (N,)
        Rinv = rot2(-yaw)                                          # (N,2,2)
        ground = pos.clone(); ground[:, 2] = 0.0

        def heading3(v: torch.Tensor) -> torch.Tensor:             # world (N,3) -> heading frame
            out = v.clone()
            out[:, :2] = torch.einsum("nij,nj->ni", Rinv, v[:, :2])
            return out

        root_vel = heading3(lin)
        lf, rf = heading3(kp_pos[:, li] - ground), heading3(kp_pos[:, ri] - ground)
        lfv, rfv = heading3(kp_vel[:, li]), heading3(kp_vel[:, ri])

        idx = torch.arange(N, device=dev)
        clip_id = corpus.frame_clip_id.to(dev)
        end = corpus.clip_end.to(dev)[clip_id]
        valid = idx + hmax <= end
        if not bool(valid.any()):
            raise ValueError(
                f"FeatureDB: no frame has {hmax} frames of future inside its clip; "
                f"clips are too short for horizons {H}")
        traj_pos, traj_dir = [], []
        for h in H:
            j = torch.minimum(idx + h, end)
            dp = pos[j, :2] - pos[:, :2]
            traj_pos.append(torch.einsum("nij,nj->ni", Rinv, dp))
            dy = yaw[j] - yaw
            traj_dir.append(torch.stack([torch.cos(dy), torch.sin(dy)], -1))
        raw = torch.cat([root_vel, lf, rf, lfv, rfv, *traj_pos, *traj_dir], -1).float()

        nh = len(H)
        blocks = [(0, 3, cfg.root_vel), (3, 9, cfg.foot_pos), (9, 15, cfg.foot_vel),
                  (15, 15 + 2 * nh, cfg.traj_pos), (15 + 2 * nh, 15 + 4 * nh, cfg.traj_dir)]
        rv = raw[valid]
        mean = rv.mean(0)
        std = rv.std(0, unbiased=False)
        scale = torch.empty_like(mean); weight = torch.empty_like(mean)
        for a, b, w in blocks:
            scale[a:b] = std[a:b].mean().clamp_min(1e-3)          # one scale per block
            weight[a:b] = float(w)
        return cls(horizons=H, raw=raw, valid_idx=idx[valid], mean=mean, scale=scale, weight=weight,
                   clip_id=clip_id, room=end - idx)

    # ---- queries ------------------------------------------------------------
    def is_valid(self, i: int) -> bool:
        return 0 <= i < self.n_frames and bool(self._valid_pos[i] >= 0)

    def pose_features(self, i: int) -> torch.Tensor:
        return self.raw[i, :POSE_DIM]

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw.to(self.device) - self.mean) / self.scale * self.weight

    @staticmethod
    def traj_raw(traj_pos: torch.Tensor, traj_dir: torch.Tensor) -> torch.Tensor:
        """(H,2),(H,2) -> (4H,) in the stored column order (all positions, then all dirs)."""
        return torch.cat([traj_pos.reshape(-1), traj_dir.reshape(-1)])

    def query(self, pose_raw: torch.Tensor, traj_raw: torch.Tensor,
              current_frame: Optional[int] = None, current_next: Optional[int] = None,
              stay_bias: float = 0.0, jump_margin: int = 0, exclude_window: int = 0) -> int:
        """Nearest valid frame to (pose, trajectory), Holden-style:

        * a JUMP target must leave ``jump_margin`` playable frames beyond the
          horizon so playback does not immediately run out of clip;
        * frames within ``exclude_window`` of ``current_frame`` in the same
          clip are never jump targets (jumping to "now" would stall playback);
        * ``current_next`` (the frame playback reaches on its own) is exempt
          from both rules and gets ``stay_bias`` subtracted, so continuing
          wins ties and small losses.
        """
        q = self.normalize(torch.cat([pose_raw.to(self.device), traj_raw.to(self.device)]))
        d = ((self.feats - q) ** 2).sum(-1)
        blocked = self._room_valid < max(self.horizons) + int(jump_margin)
        if current_frame is not None and 0 <= current_frame < self.n_frames and exclude_window > 0:
            same = self._clip_valid == self._clip_id[current_frame]
            near = (self.valid_idx - int(current_frame)).abs() <= int(exclude_window)
            blocked = blocked | (same & near)
        cont = None
        if current_next is not None and self.is_valid(current_next):
            cont = int(self._valid_pos[current_next])
            d_cont = float(d[cont]) - float(stay_bias)
        d = torch.where(blocked, torch.full_like(d, float("inf")), d)
        if cont is not None:
            d[cont] = d_cont
        k = int(d.argmin())
        if not torch.isfinite(d[k]):                  # nothing jumpable (tiny corpus): best valid frame
            k = int(((self.feats - q) ** 2).sum(-1).argmin())
        return int(self.valid_idx[k])

"""Motion-matching locomotion controller: command in, reference frame out."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from rgmt.data.corpus import MotionCorpus
from locomotion.command import Command, ScriptedSource
from locomotion.config import LocoConfig
from locomotion.corpus import ClipsMeta, SPEED_KEYS
from locomotion.features import FeatureDB
from locomotion.geom import rotate2
from locomotion.player import Player
from locomotion.trajectory import TrajectoryModel


@dataclass
class DebugInfo:
    clip_name: str = ""
    clip_class: str = ""
    clip_frame: int = 0            # frame index within the clip
    frame: int = 0                 # global corpus frame
    searched: bool = False
    vel: torch.Tensor = None       # spring planar velocity, world frame (2,)
    yaw_rate: float = 0.0
    traj_xy: torch.Tensor = None   # predicted world positions (H,2)
    traj_yaw: torch.Tensor = None  # predicted world yaws (H,)
    v_des: torch.Tensor = None
    w_des: float = 0.0


class MotionMatchingController:
    def __init__(self, corpus: MotionCorpus, clips_meta: ClipsMeta, cfg: LocoConfig,
                 device="cpu", feature_db: Optional[FeatureDB] = None):
        self.corpus, self.meta, self.cfg = corpus, clips_meta, cfg
        self.device = torch.device(device)
        self.speeds = self._resolve_speeds()
        self.db = feature_db or FeatureDB.build(corpus, cfg.features, self.device)
        self.traj = TrajectoryModel(cfg.spring, cfg.fps, cfg.features.horizons)
        self.player = Player(corpus, cfg.blend, cfg.fps, self.device)
        self.debug = DebugInfo()
        self._last_cmd = Command()
        self._since_search = 0

    def _resolve_speeds(self) -> Dict[str, float]:
        out = {}
        for k in SPEED_KEYS:
            v = getattr(self.cfg.speeds, k)
            if v is None:
                v = self.meta.speeds.get(k)
            if v is None:
                raise ValueError(
                    f"nominal speed '{k}' is null in config and could not be measured from the corpus "
                    f"(no moving frames of its class); set speeds.{k} in the config")
            out[k] = float(v)
        return out

    # ---- control ------------------------------------------------------------------
    def _first_frame_of_class(self, cls: str) -> int:
        for i, n in enumerate(self.corpus.clip_names):
            if self.meta.class_of(n) == cls:
                return int(self.corpus.clip_start[i])
        return 0

    def reset(self, xy=(0.0, 0.0), yaw: float = 0.0, frame: Optional[int] = None) -> None:
        f = self._first_frame_of_class("idle") if frame is None else int(frame)
        self.player.reset(f, xy, yaw)
        self.traj.reset()
        self._last_cmd = Command()
        self._since_search = self.cfg.search.interval     # search on the first step
        self._update_debug(searched=False, v_des=torch.zeros(2), w_des=0.0)

    def _desired(self, command: Command, yaw: float):
        mx, my = command.move
        vx = mx * (self.speeds["forward"] if mx >= 0 else self.speeds["backward"])
        vy = my * self.speeds["lateral"]
        v_des = rotate2(torch.tensor([vx, vy], dtype=torch.float32), yaw)
        return v_des, command.yaw * self.speeds["turn_rate"]

    def step(self, command: Command, observed_root=None) -> Dict[str, torch.Tensor]:
        if observed_root is not None:
            xy, yaw = observed_root
            self.player.re_anchor(xy, float(yaw))
        yaw0 = self.player.world_yaw
        v_des, w_des = self._desired(command, yaw0)

        s = self.cfg.search
        need = (self._since_search >= s.interval
                or (s.on_command_change and command != self._last_cmd)
                or not self.db.is_valid(self.player.next_frame))
        searched = False
        if need:
            pose = self.db.pose_features(self.player.frame)
            tp, td = self.traj.predict(yaw0, v_des, w_des)
            j = self.db.query(pose, FeatureDB.traj_raw(tp, td),
                              current_next=self.player.next_frame, stay_bias=s.stay_bias)
            if j != self.player.next_frame:
                self.player.switch(j)
            self._since_search = 0
            searched = True
        out = self.player.advance()
        self.traj.step(v_des, w_des)
        self._since_search += 1
        self._last_cmd = command
        self._update_debug(searched, v_des, w_des)
        return out

    def preview(self, k: int) -> Dict[str, torch.Tensor]:
        return self.player.preview(k)

    def _update_debug(self, searched: bool, v_des: torch.Tensor, w_des: float) -> None:
        f = self.player.frame
        c = int(self.corpus.frame_clip_id[f])
        name = self.corpus.clip_names[c]
        xy, yaw = self.player.world_xy, self.player.world_yaw
        traj_xy, traj_yaw = self.traj.predict_world(xy.detach().cpu(), yaw, v_des, w_des)
        self.debug = DebugInfo(clip_name=name, clip_class=self.meta.class_of(name),
                               clip_frame=f - int(self.corpus.clip_start[c]), frame=f, searched=searched,
                               vel=self.traj.vel.clone(), yaw_rate=self.traj.yaw_rate,
                               traj_xy=traj_xy, traj_yaw=traj_yaw, v_des=v_des, w_des=w_des)


def run_script(ctl: MotionMatchingController, script: str, xy=(0.0, 0.0), yaw: float = 0.0
               ) -> List[Dict[str, torch.Tensor]]:
    """Reset, then step through a ScriptedSource; returns every emitted frame."""
    ctl.reset(xy, yaw)
    return [ctl.step(c) for c in ScriptedSource(script, fps=ctl.cfg.fps)]

"""Locomotion command: a heading-frame move vector plus a yaw-rate command.

W/S/A/D sum into ``move`` (x forward, y left), clamped to unit length so a
diagonal walks at nominal speed. Q/E sum into ``yaw`` (+ is counter-clockwise).
``ScriptedSource`` replays a timed key script for headless runs and tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple

KEYS = ("w", "a", "s", "d", "q", "e")
_MOVE = {"w": (1.0, 0.0), "s": (-1.0, 0.0), "a": (0.0, 1.0), "d": (0.0, -1.0)}
_YAW = {"q": 1.0, "e": -1.0}


@dataclass(frozen=True)
class Command:
    move: Tuple[float, float] = (0.0, 0.0)   # heading frame, |move| <= 1
    yaw: float = 0.0                         # [-1, 1], + = turn left (ccw)

    @property
    def is_zero(self) -> bool:
        return self.move == (0.0, 0.0) and self.yaw == 0.0


def from_keys(pressed: Iterable[str]) -> Command:
    keys = {k.lower() for k in pressed}
    x = sum(_MOVE[k][0] for k in keys if k in _MOVE)
    y = sum(_MOVE[k][1] for k in keys if k in _MOVE)
    n = math.hypot(x, y)
    if n > 1.0:
        x, y = x / n, y / n
    yaw = max(-1.0, min(1.0, sum(_YAW[k] for k in keys if k in _YAW)))
    # "+ 0.0" turns -0.0 into 0.0 so opposing keys compare equal to Command()
    return Command(move=(float(x) + 0.0, float(y) + 0.0), yaw=float(yaw) + 0.0)


class ScriptedSource:
    """``"w:2,wd:1,:0.5,q:1"`` -> W for 2 s, W+D for 1 s, nothing for 0.5 s, Q for 1 s.

    Each segment is ``<keys>:<seconds>``; keys are any of w a s d q e (empty
    for no keys). ``next()`` returns the command for the next tick, or None
    once the script is exhausted.
    """

    def __init__(self, script: str, fps: int = 60):
        if not script.strip():
            raise ValueError("ScriptedSource: empty script")
        self._segments: List[Tuple[Command, int]] = []
        for raw in script.split(","):
            seg = raw.strip()
            if seg.count(":") != 1:
                raise ValueError(f"ScriptedSource: segment {seg!r} must be <keys>:<seconds>")
            keys, secs = seg.split(":")
            bad = [k for k in keys.strip().lower() if k not in KEYS]
            if bad:
                raise ValueError(f"ScriptedSource: segment {seg!r} has unknown keys {bad}")
            try:
                seconds = float(secs)
            except ValueError as e:
                raise ValueError(f"ScriptedSource: segment {seg!r} has a non-numeric duration") from e
            if seconds < 0:
                raise ValueError(f"ScriptedSource: segment {seg!r} has a negative duration")
            self._segments.append((from_keys(set(keys.strip().lower())), int(round(seconds * fps))))
        self._i = 0
        self._left = self._segments[0][1]

    @property
    def total_ticks(self) -> int:
        return sum(n for _, n in self._segments)

    def next(self) -> Optional[Command]:
        while self._i < len(self._segments) and self._left <= 0:
            self._i += 1
            if self._i < len(self._segments):
                self._left = self._segments[self._i][1]
        if self._i >= len(self._segments):
            return None
        self._left -= 1
        return self._segments[self._i][0]

    def __iter__(self) -> Iterator[Command]:
        while True:
            c = self.next()
            if c is None:
                return
            yield c

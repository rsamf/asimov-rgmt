"""Critically damped spring-damper primitives (Holden 2021 closed forms) and,
below, the trajectory model that predicts where the character will be.

All functions are exact for any dt, so one call with dt = horizon predicts
the same state as many small steps.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch

from locomotion.config import SpringConfig
from locomotion.geom import rotate2

LN2 = math.log(2.0)


def halflife_to_damping(halflife: float, eps: float = 1e-5) -> float:
    return 4.0 * LN2 / (halflife + eps)


def decay_spring_damper_exact(x, v, halflife: float, dt: float):
    """Decay offset x with rate v toward zero (critically damped). Works on
    tensors or floats; returns (x, v)."""
    y = halflife_to_damping(halflife) / 2.0
    j1 = v + x * y
    eydt = math.exp(-y * dt)
    return eydt * (x + j1 * dt), eydt * (v - j1 * y * dt)


def spring_velocity_update(x, v, a, v_des, halflife: float, dt: float):
    """Advance position x, velocity v, acceleration a by dt while v is pulled
    toward v_des by a critically damped spring. Returns (x, v, a)."""
    y = halflife_to_damping(halflife) / 2.0
    j0 = v - v_des
    j1 = a + j0 * y
    eydt = math.exp(-y * dt)
    x_new = eydt * ((-j1) / (y * y) + (-j0 - j1 * dt) / y) + j1 / (y * y) + j0 / y + v_des * dt + x
    v_new = eydt * (j0 + j1 * dt) + v_des
    a_new = eydt * (a - j1 * y * dt)
    return x_new, v_new, a_new


class TrajectoryModel:
    """Spring state (planar velocity + yaw rate) and analytic future prediction.

    Position and facing are NOT stored: the caller passes the character's
    current world facing (and position, for the overlay) so the prediction
    always starts from the animated root and cannot drift from it.
    """

    def __init__(self, cfg: SpringConfig, fps: int, horizons: Sequence[int]):
        self.cfg = cfg
        self.dt = 1.0 / float(fps)
        self.horizons = [int(h) for h in horizons]
        self.reset()

    def reset(self) -> None:
        self.vel = torch.zeros(2)
        self.acc = torch.zeros(2)
        self.yaw_rate = 0.0
        self.yaw_acc = 0.0

    def step(self, v_des: torch.Tensor, w_des: float) -> None:
        _, self.vel, self.acc = spring_velocity_update(
            torch.zeros(2), self.vel, self.acc, v_des.to(torch.float32), self.cfg.velocity_halflife, self.dt)
        _, self.yaw_rate, self.yaw_acc = spring_velocity_update(
            0.0, self.yaw_rate, self.yaw_acc, float(w_des), self.cfg.rotation_halflife, self.dt)

    def _predict_deltas(self, v_des: torch.Tensor, w_des: float):
        dps, dyaws = [], []
        for h in self.horizons:
            t = h * self.dt
            dp, _, _ = spring_velocity_update(torch.zeros(2), self.vel, self.acc,
                                              v_des.to(torch.float32), self.cfg.velocity_halflife, t)
            dyaw, _, _ = spring_velocity_update(0.0, self.yaw_rate, self.yaw_acc, float(w_des),
                                                self.cfg.rotation_halflife, t)
            dps.append(dp)
            dyaws.append(float(dyaw))
        return torch.stack(dps), torch.tensor(dyaws)      # world-frame displacements, relative yaws

    def predict(self, yaw0: float, v_des: torch.Tensor, w_des: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Future positions (H,2) and facing directions (H,2) in the heading frame of yaw0."""
        dp, dyaw = self._predict_deltas(v_des, w_des)
        pos_h = rotate2(dp, -float(yaw0))
        dir_h = torch.stack([torch.cos(dyaw), torch.sin(dyaw)], -1)
        return pos_h, dir_h

    def predict_world(self, xy0: torch.Tensor, yaw0: float, v_des: torch.Tensor, w_des: float):
        """Future world positions (H,2) and world yaws (H,) for drawing."""
        dp, dyaw = self._predict_deltas(v_des, w_des)
        return xy0.to(torch.float32) + dp, float(yaw0) + dyaw

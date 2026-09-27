"""Critically damped spring-damper primitives (Holden 2021 closed forms) and,
below, the trajectory model that predicts where the character will be.

All functions are exact for any dt, so one call with dt = horizon predicts
the same state as many small steps.
"""
from __future__ import annotations

import math

import torch

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

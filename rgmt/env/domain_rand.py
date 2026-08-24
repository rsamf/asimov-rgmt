"""Domain randomization: per-env dynamics sampling for sim2real.

Pure torch bookkeeping — all sim interaction goes through NewtonSim's DR
setters (`set_foot_friction_per_env`, `set_body_mass_scale`,
`set_joint_gain_scale`, `set_effort_limit_scale`) followed by ONE
`apply_dynamics_changes()` per training iteration (the notify batching rule;
see the DR block comment in sim.py).

Cadence contract (implemented in TrackEnv): draws are EPISODE-consistent —
an env keeps its dynamics until it resets; envs that reset during an
iteration are resampled together at the next iteration boundary. A freshly
reset env therefore runs at most rollout_len (32) steps under its previous
draw, one dynamics switch near episode start that the 10-step history window
recovers from almost immediately. This preserves the history-encoder-as-
implicit-system-ID premise the RGMT architecture is built on.

Ranges are [lo, hi] uniform; None disables a parameter entirely (its setter
is never called, so it contributes nothing to notify cost). `privileged`
appends the normalized draw vector to critic_obs (asymmetric critic sees the
true dynamics; the actor never does — it must infer them from history).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

# MJCF foot_capsule nominal (the value that actually governs foot-ground
# contact — see the foot_friction deprecation note in track_env.py).
NOMINAL_FOOT_MU = 0.6


@dataclass
class DRConfig:
    friction_range: Optional[list] = None       # foot mu, e.g. [0.4, 1.0]
    mass_scale_range: Optional[list] = None     # whole robot, e.g. [0.9, 1.1]
    payload_mass_range: Optional[list] = None   # RESERVED (schema knob, not implemented)
    kp_scale_range: Optional[list] = None       # e.g. [0.9, 1.1]
    kd_scale_range: Optional[list] = None       # e.g. [0.9, 1.1]
    effort_scale_range: Optional[list] = None   # e.g. [0.8, 1.2]
    privileged: bool = False                    # append draws to critic_obs

    def __post_init__(self):
        if self.payload_mass_range is not None:
            raise NotImplementedError(
                "payload_mass_range is a reserved knob — not implemented yet")
        for name in ("friction_range", "mass_scale_range", "kp_scale_range",
                     "kd_scale_range", "effort_scale_range"):
            r = getattr(self, name)
            if r is None:
                continue
            lo, hi = float(r[0]), float(r[1])
            if not (lo <= hi):
                raise ValueError(f"{name}: lo {lo} > hi {hi}")
            if lo <= 0.0 and name != "friction_range":
                raise ValueError(f"{name}: scales must be positive, got lo={lo}")
            if name == "friction_range" and lo < 0.0:
                raise ValueError(f"friction_range: mu must be >= 0, got lo={lo}")


class DomainRand:
    """Per-env draws + normalized privileged observation.

    obs() layout is FIXED (order and width) regardless of which ranges are
    enabled, so priv_dim never depends on range tweaks:
        [mu, mass_scale, payload(reserved), kp_scale, kd_scale, effort_scale]
    each normalized (x - mid)/half_range -> [-1, 1]; disabled params emit 0.
    """

    DR_OBS_DIM = 6
    _PARAMS = (  # (attr, cfg range field, nominal)
        ("mu", "friction_range", NOMINAL_FOOT_MU),
        ("mass_scale", "mass_scale_range", 1.0),
        ("payload", "payload_mass_range", 0.0),
        ("kp_scale", "kp_scale_range", 1.0),
        ("kd_scale", "kd_scale_range", 1.0),
        ("effort_scale", "effort_scale_range", 1.0),
    )

    def __init__(self, cfg: DRConfig, num_envs: int, device) -> None:
        self.cfg = cfg
        self.N = int(num_envs)
        self.device = torch.device(device)
        for attr, _range, nominal in self._PARAMS:
            setattr(self, attr,
                    torch.full((self.N,), float(nominal), device=self.device))

    def _enabled(self, range_field: str) -> Optional[tuple[float, float]]:
        r = getattr(self.cfg, range_field)
        if r is None:
            return None
        return float(r[0]), float(r[1])

    def sample(self, env_ids: torch.Tensor) -> None:
        """Redraw enabled params for ``env_ids`` (others keep their draw)."""
        if env_ids.numel() == 0:
            return
        for attr, range_field, _nominal in self._PARAMS:
            r = self._enabled(range_field)
            if r is None:
                continue
            lo, hi = r
            draw = torch.rand(env_ids.numel(), device=self.device) * (hi - lo) + lo
            getattr(self, attr)[env_ids] = draw

    def write_to_sim(self, sim) -> None:
        """Push current draws through the sim's DR setters (no notify)."""
        if self._enabled("friction_range") is not None:
            sim.set_foot_friction_per_env(self.mu)
        if self._enabled("mass_scale_range") is not None:
            sim.set_body_mass_scale(self.mass_scale)
        kp_on = self._enabled("kp_scale_range") is not None
        kd_on = self._enabled("kd_scale_range") is not None
        if kp_on or kd_on:
            sim.set_joint_gain_scale(self.kp_scale, self.kd_scale)
        if self._enabled("effort_scale_range") is not None:
            sim.set_effort_limit_scale(self.effort_scale)

    def obs(self) -> torch.Tensor:
        """(N, DR_OBS_DIM) normalized draws; disabled params are 0."""
        cols = []
        for attr, range_field, _nominal in self._PARAMS:
            r = self._enabled(range_field)
            if r is None:
                cols.append(torch.zeros(self.N, device=self.device))
                continue
            lo, hi = r
            mid = 0.5 * (lo + hi)
            half = max(0.5 * (hi - lo), 1e-8)
            cols.append((getattr(self, attr) - mid) / half)
        return torch.stack(cols, dim=1)

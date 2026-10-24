"""Physics mode for the locomotion scene: the controller's reference frames
drive a tracking policy (or pure PD) in the simulator.

The actor sees exactly what ``TrackEnv`` builds in training, with the corpus
lookup replaced by the controller:

    obs         proprioception from the sim (``TrackEnv._build_obs``)
    history     the last 10 obs
    cmd_window  21 frames centred on the current reference frame: the last
                10 emitted controller frames, the current one, and the next
                10 the controller would emit if no search switched clips
                (``preview(blended=True)``), with drift features per frame
                when the checkpoint was trained with them

The plant (gains, torque caps, residual scale and filter, dt, observation
options) is read from the checkpoint with ``plant_from_ckpt``; without a
checkpoint the robot runs pure PD on the plant v2 recipe (``physics.pd_gains``
with URDF torque caps).

The reference is a kinematic trajectory that runs ahead on its own. If the
robot falls behind by more than ``physics.max_drift`` metres the controller's
anchor is shifted toward the robot so the drift features stay inside the
range seen in training; heading is left to the policy.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import torch

from rgmt.data.corpus import MotionCorpus, command_from_frames
from rgmt.eval_gated import build_eval_env, plant_from_ckpt
from rgmt.env.gains import gains_profile, urdf_effort_limits
from rgmt.env.track_env import drift_features
from locomotion.command import Command
from locomotion.config import LocoConfig
from locomotion.controller import MotionMatchingController


def load_policy(path: str, device):
    """(model, checkpoint dict) for an RGMT checkpoint, in eval mode."""
    from rgmt.policy.networks import PolicyDims, RGMTActorCritic
    ck = torch.load(path, map_location=device, weights_only=False)
    net = (ck.get("config") or {}).get("network") or {}
    model = RGMTActorCritic(PolicyDims(**ck["dims"]),
                            actor_hidden=tuple(net.get("actor_hidden", [512, 256])),
                            critic_hidden=tuple(net.get("critic_hidden", [512, 256]))).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


class PhysicsDriver:
    def __init__(self, ctl: MotionMatchingController, corpus: MotionCorpus, cfg: LocoConfig,
                 checkpoint: Optional[str], device):
        self.ctl, self.cfg = ctl, cfg
        self.device = torch.device(device)
        self.model = None
        if checkpoint:
            self.model, ck = load_policy(checkpoint, self.device)
            plant = plant_from_ckpt(ck)
            self.iteration = ck.get("iteration")
        else:
            kp, kd = gains_profile(cfg.physics.pd_gains)
            plant = dict(kp=kp, kd=kd, effort_limits=urdf_effort_limits())
            self.iteration = None
        if plant.get("drift_obs_proprio"):
            raise NotImplementedError("checkpoints trained with drift_obs_proprio are not supported here")
        self.env = build_eval_env(corpus, self.device, n_envs=1, **plant)
        self.sim = self.env.sim
        self.L = self.env.L
        self.drift_obs = self.env.cfg.drift_obs
        self.actuated_idx = self.sim.actuated_idx.to(self.device)
        self._ids = torch.arange(1, device=self.device)
        self.past: deque = deque(maxlen=self.L)
        self.cur: Dict[str, torch.Tensor] = {}
        self.falls = 0

    # ---- state ----------------------------------------------------------------
    def reset(self, xy=(0.0, 0.0), yaw: float = 0.0) -> None:
        """Controller to idle at (xy, yaw), robot teleported onto its first frame."""
        self.ctl.reset(xy, yaw)
        self.cur = self.ctl.step(Command())
        self._teleport(self.cur)
        self.past.clear()
        # clip-start clamp, as in training; distinct copies because _leash shifts them in place
        self.past.extend({k: v.clone() for k, v in self.cur.items()} for _ in range(self.L))
        env = self.env
        env.prev_action.zero_()
        env.filt_res.zero_()
        o = env._build_obs()
        env.history[:] = o.unsqueeze(1).expand(-1, env.history.shape[1], -1).clone()

    def _teleport(self, f: Dict[str, torch.Tensor]) -> None:
        n = self.sim.n_hinges
        hq = torch.zeros(1, n, device=self.device); hqd = torch.zeros(1, n, device=self.device)
        hq[:, self.actuated_idx] = f["joint_q"]
        hqd[:, self.actuated_idx] = f["joint_qd"]
        self.sim.reset_idx(self._ids, f["base_pos"], f["base_quat"], f["base_lin_vel"],
                           f["base_ang_vel"], hq, hqd)

    # ---- observation ----------------------------------------------------------
    def window(self) -> Dict[str, torch.Tensor]:
        """The 21 reference frames the command window is built from (past, current, future)."""
        fut = self.ctl.preview(self.L, blended=True)
        rows = list(self.past) + [self.cur]
        return {k: torch.cat([r[k] for r in rows] + [fut[k]], dim=0) for k in self.cur}

    def bundle(self) -> Dict[str, torch.Tensor]:
        win = self.window()
        cmd = command_from_frames(win).unsqueeze(0)                         # (1, 21, 55)
        if self.drift_obs:
            drift = drift_features(win["base_pos"][None], win["base_quat"][None],
                                   self.sim.base_pos, self.sim.base_quat)
            cmd = torch.cat([cmd, drift], dim=-1)                           # (1, 21, 60)
        return {"obs": self.env._build_obs(), "history": self.env.history.clone(), "cmd_window": cmd}

    # ---- control --------------------------------------------------------------
    @torch.no_grad()
    def step(self, command: Command) -> bool:
        """One 60 Hz tick: act on the current reference, advance the controller.
        Returns True when the robot fell (and was reset)."""
        env = self.env
        if self.model is not None:
            action = self.model.act_inference(self.bundle())
        else:
            action = torch.zeros(1, env.n_act, device=self.device)
        # residual on the reference pose, exactly as TrackEnv.step
        res = torch.tanh(action) * env.action_scale
        if env.action_filter_alpha < 1.0:
            env.filt_res = env.action_filter_alpha * res + (1.0 - env.action_filter_alpha) * env.filt_res
            res = env.filt_res
        self.sim.step(self.cur["joint_q"] + res)
        env.prev_action = torch.tanh(action)

        self.past.append(self.cur)
        self.cur = self.ctl.step(command)
        self._leash()
        o = env._build_obs()
        env.history[:] = torch.cat([env.history[:, 1:], o.unsqueeze(1)], dim=1)

        if bool(env._fallen().item()) and self.cfg.physics.auto_reset:
            self.falls += 1
            xy = self.sim.base_pos[0, :2].tolist()
            self.reset(xy, float(self.ctl.player.world_yaw))
            return True
        return False

    def _leash(self) -> None:
        cap = self.cfg.physics.max_drift
        if cap is None:
            return
        robot = self.sim.base_pos[0, :2]
        ref = self.cur["base_pos"][0, :2]
        d = ref - robot
        n = float(d.norm())
        if n <= cap:
            return
        shift = robot + d * (cap / n) - ref
        self.ctl.player.translate(shift)
        for f in [self.cur, *self.past]:
            f["base_pos"][0, :2] += shift

    @property
    def drift(self) -> float:
        """Planar distance (m) between the reference root and the robot root."""
        return float((self.cur["base_pos"][0, :2] - self.sim.base_pos[0, :2]).norm())

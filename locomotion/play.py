"""Kinematic test scene for the motion-matching controller (no physics).

    uv run python -m locomotion.play                      # corpus from configs/local.yaml
    uv run python -m locomotion.play --corpus <dir>       # dir holding cache/ and clips.json
    uv run python -m locomotion.play --viewer null --script "w:2,wd:1,q:1" --seconds 4

Keys in the GL window: W/A/S/D translate in the heading frame, Q/E turn,
Space resets to the origin. A third-person camera follows the robot and the
viewer's own key-driven camera flight is disabled (``--free-camera`` restores
it; the mouse still orbits and zooms). The robot is teleported to every controller frame
through the simulator's kinematic reset (exactly what ``rgmt.view --mode
replay`` does); physics never steps. An overlay draws the predicted
trajectory; the clip/class/speed line is shown in the viewer's stats panel
(or printed once per second when the viewer has no UI).
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import torch

from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.env.sim import NewtonSim
from rgmt.env.track_env import EnvConfig
from rgmt.view import _make_viewer
from locomotion.command import Command, ScriptedSource, from_keys
from locomotion.config import load_config
from locomotion.controller import MotionMatchingController
from locomotion.corpus import load_locomotion_corpus
from locomotion.geom import wrap_angle, yaw_from_quat

_KEYS = ("w", "a", "s", "d", "q", "e")


def _build_sim(device) -> NewtonSim:
    e = EnvConfig(num_envs=1, keypoint_links=KEYPOINT_LINKS)
    return NewtonSim(1, kp=e.kp, kd=e.kd, control_decimation=e.control_decimation, dt=e.dt,
                     foot_friction=e.foot_friction, keypoint_links=KEYPOINT_LINKS, device=device)


def _teleport(sim: NewtonSim, actuated_idx: torch.Tensor, frame: dict, device) -> None:
    hinge_q = torch.zeros(1, sim.n_hinges, device=device); hinge_qd = torch.zeros(1, sim.n_hinges, device=device)
    hinge_q[:, actuated_idx] = frame["joint_q"].to(device)
    hinge_qd[:, actuated_idx] = frame["joint_qd"].to(device)
    sim.reset_idx(torch.arange(1, device=device), frame["base_pos"].to(device), frame["base_quat"].to(device),
                  frame["base_lin_vel"].to(device), frame["base_ang_vel"].to(device), hinge_q, hinge_qd)


class FollowCamera:
    """Third-person camera: behind the robot along its heading, above it,
    looking at the root. Position and heading are smoothed with a half-life so
    turns swing the view rather than snap it. ``attach`` zeroes the viewer's
    key-driven camera speed so WASD/QE drive only the robot."""

    def __init__(self, cfg, fps: int):
        self.cfg = cfg
        self.dt = 1.0 / float(fps)
        self.reset()

    def reset(self) -> None:
        self._xy = None
        self._yaw = 0.0

    def _blend(self) -> float:
        return 1.0 - 0.5 ** (self.dt / max(self.cfg.halflife, 1e-6))

    def update(self, root_xy, yaw: float):
        """Smoothed camera position (x, y, z) and look-at target (x, y, z)."""
        x, y = float(root_xy[0]), float(root_xy[1])
        yaw = float(yaw)
        if self._xy is None:
            self._xy, self._yaw = [x, y], yaw
        else:
            a = self._blend()
            self._xy[0] += (x - self._xy[0]) * a
            self._xy[1] += (y - self._xy[1]) * a
            self._yaw += wrap_angle(yaw - self._yaw) * a
        c = self.cfg
        pos = (self._xy[0] - c.distance * math.cos(self._yaw),
               self._xy[1] - c.distance * math.sin(self._yaw), c.height)
        target = (self._xy[0], self._xy[1], c.target_height)
        return pos, target

    def attach(self, viewer) -> None:
        if hasattr(viewer, "camera_speed"):
            viewer.camera_speed = 0.0

    def apply(self, viewer, root_xy, yaw: float) -> None:
        cam = getattr(viewer, "camera", None)
        if cam is None:
            return
        pos, target = self.update(root_xy, yaw)
        try:
            from pyglet.math import Vec3
            pos, target = Vec3(*pos), Vec3(*target)
        except Exception:  # noqa: BLE001 - fake/headless cameras take plain tuples
            pass
        cam.pos = pos
        cam.look_at(target)


class TrajectoryOverlay:
    """Draws the predicted trajectory (root -> horizon points, plus a facing tick
    at each) with ``viewer.log_lines``. A failure is reported once on stderr and
    the overlay switches itself off instead of failing silently every frame."""

    def __init__(self) -> None:
        self.enabled = True

    def draw(self, viewer, ctl) -> None:
        if not self.enabled or not hasattr(viewer, "log_lines"):
            return
        try:
            import warp as wp
            d = ctl.debug
            root = ctl.player.world_xy.detach().cpu().numpy()
            pts = d.traj_xy.detach().cpu().numpy()
            yaws = d.traj_yaw.detach().cpu().numpy()
            starts, ends = [], []
            prev = root
            for p, y in zip(pts, yaws):
                starts.append((prev[0], prev[1], 0.02)); ends.append((p[0], p[1], 0.02)); prev = p
                starts.append((p[0], p[1], 0.02))
                ends.append((p[0] + 0.15 * math.cos(y), p[1] + 0.15 * math.sin(y), 0.02))
            s = wp.array(np.array(starts, np.float32), dtype=wp.vec3)
            e = wp.array(np.array(ends, np.float32), dtype=wp.vec3)
            viewer.log_lines("loco/trajectory", s, e, (1.0, 0.6, 0.1), width=0.01)
        except Exception as exc:  # noqa: BLE001 - report once, keep the scene running
            self.enabled = False
            print(f"trajectory overlay disabled: {type(exc).__name__}: {exc}", file=sys.stderr)


def run(args) -> int:
    device = torch.device(args.device)
    cfg = load_config(args.config, args.set)
    if args.corpus:
        cfg.corpus.cache_dir = args.corpus
    if args.free_camera:
        cfg.scene.camera.follow = False
    corpus, meta = load_locomotion_corpus(cfg, device)
    ctl = MotionMatchingController(corpus, meta, cfg, device=device)
    ctl.reset()

    sim = _build_sim(device)
    actuated_idx = sim.actuated_idx.to(device)
    viewer = _make_viewer(args.viewer, args.headless)
    viewer.set_model(sim.model)

    status = {"text": ""}
    if hasattr(viewer, "register_ui_callback"):
        try:
            viewer.register_ui_callback(lambda imgui: imgui.text(status["text"]), position="stats")
        except Exception:
            pass
    overlay = TrajectoryOverlay()
    follow = None
    if cfg.scene.camera.follow and getattr(viewer, "camera", None) is not None:
        follow = FollowCamera(cfg.scene.camera, cfg.fps)
        follow.attach(viewer)
    script = ScriptedSource(args.script, fps=cfg.fps) if args.script else None
    can_poll = hasattr(viewer, "is_key_down") and args.viewer == "gl" and not args.headless

    max_steps = int(args.seconds * cfg.fps) if args.seconds is not None else (
        script.total_ticks if script else 10 ** 9)
    frame_dt = 1.0 / max(args.fps, 1e-3)
    step, t_sim, last_print = 0, 0.0, -1.0
    try:
        while step < max_steps and (not hasattr(viewer, "is_running") or viewer.is_running()):
            t_wall = time.perf_counter()
            if script is not None:
                cmd = script.next()
                if cmd is None:
                    break
            elif can_poll:
                if viewer.is_key_down("space"):
                    ctl.reset()
                    if follow is not None:
                        follow.reset()
                cmd = from_keys({k for k in _KEYS if viewer.is_key_down(k)})
            else:
                cmd = Command()
            frame = ctl.step(cmd)
            _teleport(sim, actuated_idx, frame, device)
            if follow is not None:
                follow.apply(viewer, frame["base_pos"][0, :2].tolist(), float(yaw_from_quat(frame["base_quat"][0])))

            d = ctl.debug
            status["text"] = (f"{d.clip_class:9s} {d.clip_name} @ {d.clip_frame}   "
                              f"|v|={float(d.vel.norm()):.2f} m/s  yaw_rate={d.yaw_rate:+.2f} rad/s")
            if not can_poll and t_sim - last_print >= 1.0:
                print(f"t={t_sim:5.1f}s  {status['text']}", flush=True); last_print = t_sim

            viewer.begin_frame(t_sim)
            viewer.log_state(sim.state_0)
            overlay.draw(viewer, ctl)
            viewer.end_frame()
            t_sim += 1.0 / cfg.fps
            step += 1
            if args.viewer == "gl" and not args.headless:
                lag = frame_dt - (time.perf_counter() - t_wall)
                if lag > 0:
                    time.sleep(lag)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(viewer, "close"):
            viewer.close()
    print(f"done ({step} steps)")
    return step


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default=None, help="dir with cache/ and clips.json (default: config corpus.cache_dir)")
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[], help="dotlist overrides, e.g. speeds.forward=0.8")
    p.add_argument("--viewer", choices=["gl", "null", "rerun", "viser"], default="gl")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--script", default=None, help='timed keys, e.g. "w:2,wd:1,:0.5,q:1" (replaces the keyboard)')
    p.add_argument("--seconds", type=float, default=None, help="stop after this many simulated seconds")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fps", type=float, default=60.0, help="real-time pacing for the GL viewer")
    p.add_argument("--free-camera", action="store_true",
                   help="Newton's fly camera instead of the third-person follow camera (scene.camera.follow=false)")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    main()

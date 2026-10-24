"""Headless sim2sim check: the locomotion controller + policy on plain (CPU) MuJoCo.

Mirrors ``locomotion/physics.py:PhysicsDriver`` on ``locomotion_webplayer/assets/asimov_web.xml``
(bake it first with ``scripts/export_web.py``), so it is the Python reference for
the browser loop. Runs a key script with the policy and with pure PD and reports
falls and drift.

    uv run python scripts/mujoco_sim2sim.py --ckpt runs/rgmt_n24_newplant/best_test.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_web import pick_clips, subset_corpus  # noqa: E402

from rgmt.data.corpus import command_from_frames
from rgmt.data.mirror import mirror_corpus
from rgmt.env.track_env import drift_features
from rgmt.utils.rotation import encode_angles, gravity_projection, quat_rotate_inverse
from locomotion.command import Command, ScriptedSource
from locomotion.config import load_config
from locomotion.controller import MotionMatchingController
from locomotion.corpus import load_locomotion_corpus
from locomotion.physics import load_policy

SCRIPT = "w:4,q:2,wq:3,s:2,a:2,d:2,e:3,w:5,wd:3"


class MjDriver:
    def __init__(self, ctl, assets: Path, model=None):
        self.ctl, self.model = ctl, model
        self.p = json.loads((assets / "plant.json").read_text())
        self.m = mujoco.MjModel.from_xml_path(str(assets / "asimov_web.xml"))
        self.d = mujoco.MjData(self.m)
        self.qa, self.va = np.array(self.p["qpos_adr"]), np.array(self.p["qvel_adr"])
        self.ci = np.array(self.p["ctrl_idx"])
        self.pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
        self.head = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, self.p["fall"]["head_body"])
        self.L = self.p["hist_len"]
        self.falls = 0

    # state -----------------------------------------------------------------
    def quat_xyzw(self):
        w, x, y, z = self.d.qpos[3:7]
        return torch.tensor([[x, y, z, w]], dtype=torch.float32)

    def base_pos(self):
        return torch.tensor(self.d.qpos[None, :3], dtype=torch.float32)

    def obs(self):
        q = torch.tensor(self.d.qpos[self.qa], dtype=torch.float32)[None]
        qd = torch.tensor(self.d.qvel[self.va], dtype=torch.float32)[None]
        w = torch.tensor(self.d.qvel[None, 3:6], dtype=torch.float32)   # body frame
        return torch.cat([gravity_projection(self.quat_xyzw()), w, encode_angles(q), qd, self.prev], -1)

    def teleport(self, f):
        d = self.d
        mujoco.mj_resetData(self.m, d)
        x, y, z, w = f["base_quat"][0].tolist()
        d.qpos[:3] = f["base_pos"][0].numpy(); d.qpos[3:7] = [w, x, y, z]
        d.qpos[self.qa] = f["joint_q"][0].numpy()
        d.qvel[:3] = f["base_lin_vel"][0].numpy()
        d.qvel[3:6] = quat_rotate_inverse(f["base_quat"], f["base_ang_vel"])[0].numpy()
        d.qvel[self.va] = f["joint_qd"][0].numpy()
        mujoco.mj_forward(self.m, d)

    def reset(self, xy=(0.0, 0.0), yaw=0.0):
        self.ctl.reset(xy, yaw)
        self.cur = self.ctl.step(Command())
        self.teleport(self.cur)
        self.past = deque(({k: v.clone() for k, v in self.cur.items()} for _ in range(self.L)), maxlen=self.L)
        self.prev = torch.zeros(1, 23); self.filt = torch.zeros(1, 23)
        o = self.obs()
        self.history = o[:, None].repeat(1, self.L, 1)

    def fallen(self):
        d = self.d
        return (d.xpos[self.pelvis, 2] < self.p["fall"]["pelvis_z"]
                or d.xmat[self.pelvis, 8] < self.p["fall"]["up_dot_min"]
                or d.xpos[self.head, 2] < self.p["fall"]["head_z"])

    # loop ----------------------------------------------------------------------
    @torch.no_grad()
    def step(self, command):
        if self.model is not None:
            fut = self.ctl.preview(self.L, blended=True)
            rows = list(self.past) + [self.cur]
            win = {k: torch.cat([r[k] for r in rows] + [fut[k]], 0) for k in self.cur}
            cmd = command_from_frames(win)[None]
            if self.p["drift_obs"]:
                cmd = torch.cat([cmd, drift_features(win["base_pos"][None], win["base_quat"][None],
                                                     self.base_pos(), self.quat_xyzw())], -1)
            a = self.model.act_inference({"obs": self.obs(), "history": self.history, "cmd_window": cmd})
        else:
            a = torch.zeros(1, 23)
        res = torch.tanh(a) * self.p["action_scale"]
        self.filt = self.p["alpha"] * res + (1 - self.p["alpha"]) * self.filt
        self.d.ctrl[self.ci] = (self.cur["joint_q"] + self.filt)[0].numpy()
        mujoco.mj_step(self.m, self.d)
        self.prev = torch.tanh(a)
        self.past.append(self.cur)
        self.cur = self.ctl.step(command)
        self.leash()
        self.history = torch.cat([self.history[:, 1:], self.obs()[:, None]], 1)
        if self.fallen():
            self.falls += 1
            self.reset(tuple(self.d.qpos[:2]), float(self.ctl.player.world_yaw))
            return True
        return False

    def leash(self):
        cap = self.p["max_drift"]
        robot = self.base_pos()[0, :2]
        ref = self.cur["base_pos"][0, :2]
        n = float((ref - robot).norm())
        if n > cap:
            shift = robot + (ref - robot) * (cap / n) - ref
            self.ctl.player.translate(shift)
            for f in [self.cur, *self.past]:
                f["base_pos"][0, :2] += shift

    def drift(self):
        return float((self.cur["base_pos"][0, :2] - self.base_pos()[0, :2]).norm())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--assets", default="locomotion_webplayer/assets")
    ap.add_argument("--script", default=SCRIPT)
    args = ap.parse_args()
    cfg = load_config()
    cfg.corpus.mirror = False
    full, meta = load_locomotion_corpus(cfg, "cpu")
    corpus = mirror_corpus(subset_corpus(full, pick_clips(full, meta, 25, 240, 900)))
    model, _ = load_policy(args.ckpt, "cpu")
    for name, mdl in (("policy", model), ("pure PD", None)):
        ctl = MotionMatchingController(corpus, meta, cfg, device="cpu")
        drv = MjDriver(ctl, Path(args.assets), mdl)
        drv.reset()
        src = ScriptedSource(args.script, fps=cfg.fps)
        drifts, t, start = [], 0, drv.d.qpos[:2].copy()
        while (c := src.next()) is not None:
            drv.step(c)
            drifts.append(drv.drift()); t += 1
        print(f"{name:8s}: {t} ticks ({t / 60:.0f} s), falls={drv.falls}, "
              f"drift mean={np.mean(drifts):.3f} max={np.max(drifts):.3f} m, "
              f"travel={np.linalg.norm(drv.d.qpos[:2] - start):.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

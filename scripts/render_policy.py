"""Side-by-side render: reference replay (left) vs policy rollout in sim (right).

All terminations are disabled so tracking failures stay on camera instead of
teleport-resetting to a random clip (which reads as 'falling'). Renders with
mujoco EGL offscreen, so no display is needed.

Usage:
    uv run --with imageio --with imageio-ffmpeg python scripts/render_policy.py         <ckpt.pt> "<clip name>" <out.mp4>
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
import imageio.v2 as iio

from rgmt.data.corpus import MotionCorpus
from rgmt.data.cache_key import file_sha256
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from rgmt.env.track_env import TrackEnv, EnvConfig
from rgmt.policy.networks import RGMTActorCritic, PolicyDims

CKPT = sys.argv[1]
CLIP = sys.argv[2]
OUT = sys.argv[3]
CACHE = "cache/"
if "--cache" in sys.argv:
    CACHE = sys.argv[sys.argv.index("--cache") + 1]
DEV = "cuda:0"

# ---- policy + env ---------------------------------------------------------
ck = torch.load(CKPT, map_location=DEV, weights_only=False)
dims = PolicyDims(**ck["dims"])
net_cfg = (ck.get("config") or {}).get("network") or {}
model = RGMTActorCritic(dims,
    actor_hidden=tuple(net_cfg.get("actor_hidden", [512, 256])),
    critic_hidden=tuple(net_cfg.get("critic_hidden", [512, 256]))).to(DEV)
model.load_state_dict(ck["model"])
model.eval()
saved_env = (ck.get("config") or {}).get("env") or {}
drift_obs = bool(saved_env.get("drift_obs", False)) or dims.cmd_dim > 55
drift_obs_proprio = bool(saved_env.get("drift_obs_proprio", False)) or dims.obs_dim > 98

corpus = MotionCorpus.load_cache(
    CACHE, output_device=DEV, urdf_hash=file_sha256(ROBOT_URDF),
    physics_fps=60, src_fps=30, keypoint_links=KEYPOINT_LINKS)

# ALL terminations disabled: a tracking failure teleport-resets the env to a
# random clip mid-video, which reads as the robot 'falling'. For an honest
# render the robot must stay on camera, drift and all.
# plant + action semantics must match training (per-joint gains / action_scale)
cfg = EnvConfig(num_envs=1, keypoint_links=KEYPOINT_LINKS, episode_len=10**9,
                noise_level=0.0, recovery_fraction=0.0,
                kp=saved_env.get("kp", 100.0), kd=saved_env.get("kd", 5.0),
                action_scale=float(saved_env.get("action_scale", 0.5)),
                action_filter_alpha=float(saved_env.get("action_filter_alpha", 1.0)),
                effort_limits=saved_env.get("effort_limits", None),
                dt=float(saved_env.get("dt", 1.0 / 60.0)),
                control_decimation=int(saved_env.get("control_decimation", 1)),
                root_err_done=1e9, z_dev_done=1e9, joint_err_done=1e9,
                z_fall=-1.0, up_dot_min=-2.0, head_z_min=-10.0,
                drift_obs=drift_obs, drift_obs_proprio=drift_obs_proprio)
env = TrackEnv(cfg, corpus, device=DEV, train=False)

ci = corpus.clip_names.index(CLIP)
start = int(corpus.clip_start[ci])
end = int(corpus.clip_end[ci])
n = end - start  # steps

# rewind to clip start (rgmt.view pattern)
ids = torch.arange(1, device=DEV)
env.reset_idx(ids)
env.idx[:] = start
env._write_ref_frame(ids, env.idx)
o = env._build_obs()
env.history[:] = o.unsqueeze(1).expand(-1, 10, -1).clone()
bundle = env._bundle()

# ---- rollout, recording sim qpos ------------------------------------------
def xyzw_to_wxyz(q):
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)

sim_qpos = []
with torch.no_grad():
    for t in range(n):
        a = model.act_inference(bundle)
        bundle, _, done, _ = env.step(a)
        pos = env.sim.base_pos[0].cpu().numpy()
        quat = env.sim.base_quat[0].cpu().numpy()          # xyzw
        hinges = env.sim.joint_q[0].cpu().numpy()          # (25,)
        sim_qpos.append(np.concatenate([pos, xyzw_to_wxyz(quat), hinges]))
sim_qpos = np.stack(sim_qpos)

# ---- reference qpos over the same frames -----------------------------------
idx = torch.arange(start + 1, end + 1, device=DEV)  # sim state at t corresponds to ref idx t+1
ref = corpus.at(idx)
ref_hinges = np.zeros((n, 25), dtype=np.float32)
act_idx = env.actuated_idx.cpu().numpy()
ref_hinges[:, act_idx] = ref["joint_q"].cpu().numpy()
ref_qpos = np.concatenate([
    ref["base_pos"].cpu().numpy(),
    xyzw_to_wxyz(ref["base_quat"].cpu().numpy()),
    ref_hinges,
], axis=-1)

# ---- render side by side ----------------------------------------------------
m = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
assert m.nq == sim_qpos.shape[1], f"nq {m.nq} != qpos width {sim_qpos.shape[1]}"
d = mujoco.MjData(m)
ren = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera()
cam.azimuth, cam.elevation, cam.distance = 135, -10, 2.2

# common origin shift so both stay in frame
origin = ref_qpos[0, :2].copy()
sim_qpos[:, :2] -= origin
ref_qpos[:, :2] -= origin

def frame(q):
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    cam.lookat[:] = [q[0], q[1], q[2] - 0.1]
    ren.update_scene(d, camera=cam)
    return ren.render()

w = iio.get_writer(OUT, fps=60, quality=8)
for t in range(n):
    left = frame(ref_qpos[t])
    right = frame(sim_qpos[t])
    combo = np.hstack([left, right])
    combo[:, 638:642] = 30  # divider
    w.append_data(combo)
w.close()
print(f"wrote {OUT} ({n} frames, {n/60:.1f}s)  clip='{CLIP}'  drift_obs={drift_obs}")

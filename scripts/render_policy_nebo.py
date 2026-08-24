"""Nebo 3D-action render: reference replay vs policy rollout, one scene per clip.

The nebo counterpart of ``scripts/render_policy.py``. Instead of rasterizing a
side-by-side MP4 with an offscreen MuJoCo renderer, each clip becomes one nebo
**action scene** carrying two instances -- ``reference`` (the retargeted motion)
and ``policy`` (the checkpoint's rollout) -- plus per-step tracking metrics on
the same playhead.

All terminations are disabled, exactly as in ``render_policy.py``: a tracking
failure teleport-resets the env to a random clip mid-episode, which reads as the
robot 'falling'. For an honest render the robot must stay in the scene, drift
and all.

Usage:
    uv run --no-sync python scripts/render_policy_nebo.py
    uv run --no-sync python scripts/render_policy_nebo.py --ckpt runs/<name>/best_test.pt \
        --clip "CMU__05__05_19" --clip "..." --cache cache/
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

import nebo as nb
from nebo.extras.robotics import mj_pose

from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from rgmt.data.cache_key import file_sha256
from rgmt.data.corpus import MotionCorpus
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.env.track_env import EnvConfig, TrackEnv
from rgmt.policy.networks import PolicyDims, RGMTActorCritic
from rgmt.utils.rotation import quat_to_matrix

DEV = "cuda:0"
# The checkpoint records the cache it trained on, so prefer that; this
# default is only a last resort.
DEFAULT_CACHE = "cache/"
DEFAULT_CKPT = "runs/rgmt_medium/best_test.pt"

# Clip sets picked from a per-clip eval dump of the released policy on
# the held-out test split (3 noisy passes each). Each set spans the failure
# spectrum -- clean successes, partials, outright failures -- with short clips on
# purpose: this is a render, not an eval campaign. Comments are
# "<passes>/3, <MPKPE under noise>".
CLIP_SETS = {
    "main": [
        "BMLmovi__Subject_24_F_MoSh__Subject_24_F_3",            # 3/3, 33.8 mm
        "Transitions_mocap__mazen_c3d__walksideways_walk",       # 2/3, 273.0 mm
        "ACCAD__Female1Running_c3d__C9 -  run backwards turn run forward",  # 0/3, 386.8 mm
    ],
    # Second batch: more range to scrub through. Partials are genuinely scarce in
    # this dump (only two clips score 1/3 or 2/3 under ~400 frames), so the 1/3
    # entry is the longest clip here by some margin.
    "extra": [
        "BioMotionLab_NTroje__rub058__0004_motorcycle",          # 3/3,  62.8 mm
        "CMU__108__108_11",                                      # 3/3,  93.0 mm
        "ACCAD__s009__Sprint1",                                  # 2/3, 447.6 mm
        "Transitions_mocap__mazen_c3d__walksideways_kick",       # 1/3, 347.6 mm
        "CMU__102__102_11",                                      # 0/3, 161.7 mm
        "BioMotionLab_NTroje__rub069__0028_jumping1",            # 0/3,  56.7 mm
    ],
}
DEFAULT_CLIPS = CLIP_SETS["main"]


def scene_label(clip: str) -> str:
    """``DATASET__subject__take`` -> ``DATASET/take`` (a nebo stream path)."""
    parts = [p.strip() for p in clip.split("__") if p.strip()]
    head, tail = (parts[0], parts[-1]) if len(parts) > 1 else ("clip", parts[0])
    return f"{head}/{' '.join(tail.split())}"


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


# ---------------------------------------------------------------------------
# pipeline nodes
# ---------------------------------------------------------------------------
@nb.fn(ui={"default_tab": "info"})
def load_policy(ckpt_path: str) -> dict:
    """Load the actor-critic and the env config it was trained with."""
    ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    dims = PolicyDims(**ck["dims"])
    net_cfg = (ck.get("config") or {}).get("network") or {}
    model = RGMTActorCritic(
        dims,
        actor_hidden=tuple(net_cfg.get("actor_hidden", [512, 256])),
        critic_hidden=tuple(net_cfg.get("critic_hidden", [512, 256])),
    ).to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    saved_env = (ck.get("config") or {}).get("env") or {}
    iteration = int(ck.get("iteration", -1))
    motion_cache = (ck.get("config") or {}).get("motion_cache")
    nb.log_cfg({
        "ckpt": ckpt_path,
        "iteration": iteration,
        "dims": ck["dims"],
        "action_scale": saved_env.get("action_scale"),
        "effort_limits": saved_env.get("effort_limits"),
        "dr": bool(saved_env.get("dr")),
    })
    nb.log_text("checkpoint", f"{ckpt_path} @ iteration {iteration}")
    return {
        "model": model,
        "dims": dims,
        "saved_env": saved_env,
        "iteration": iteration,
        "path": ckpt_path,
        "motion_cache": motion_cache,
    }


@nb.fn(ui={"default_tab": "info"})
def load_corpus(cache_dir: str) -> MotionCorpus:
    """Load the baked motion corpus (the reference side of every scene)."""
    corpus = MotionCorpus.load_cache(
        cache_dir, output_device=DEV, urdf_hash=file_sha256(ROBOT_URDF),
        physics_fps=60, src_fps=30, keypoint_links=KEYPOINT_LINKS)
    nb.log_text("corpus", f"{len(corpus.clip_names)} clips from {cache_dir}")
    return corpus


@nb.fn(ui={"default_tab": "info"})
def build_env(corpus: MotionCorpus, policy: dict) -> TrackEnv:
    """Single-env TrackEnv with every termination disabled (see module docstring)."""
    saved = policy["saved_env"]
    dims = policy["dims"]
    drift_obs = bool(saved.get("drift_obs", False)) or dims.cmd_dim > 55
    drift_obs_proprio = bool(saved.get("drift_obs_proprio", False)) or dims.obs_dim > 98
    cfg = EnvConfig(
        num_envs=1, keypoint_links=KEYPOINT_LINKS, episode_len=10**9,
        noise_level=0.0, recovery_fraction=0.0,
        kp=saved.get("kp", 100.0), kd=saved.get("kd", 5.0),
        action_scale=float(saved.get("action_scale", 0.5)),
        action_filter_alpha=float(saved.get("action_filter_alpha", 1.0)),
        effort_limits=saved.get("effort_limits", None),
        dt=float(saved.get("dt", 1.0 / 60.0)),
        control_decimation=int(saved.get("control_decimation", 1)),
        root_err_done=1e9, z_dev_done=1e9, joint_err_done=1e9,
        z_fall=-1.0, up_dot_min=-2.0, head_z_min=-10.0,
        drift_obs=drift_obs, drift_obs_proprio=drift_obs_proprio)
    env = TrackEnv(cfg, corpus, device=DEV, train=False)
    nb.log_text("env", "terminations disabled; drift_obs=%s drift_obs_proprio=%s"
                % (drift_obs, drift_obs_proprio))
    return env


@nb.fn(ui={"default_tab": "actions"})
def publish_robot() -> "nb.BodyModelRef":
    """Compile the Asimov v1 MJCF to a GLB once and publish it to the run."""
    ref = nb.log_body_model("asimov", mjcf=str(ROBOT_XML))
    nb.log_text("model", f"{ref.name}: {len(ref.body_names)} bodies "
                         f"({ref.source_format}), model_id={ref.model_id}")
    return ref


@nb.fn(ui={"default_tab": "actions"})
def rollout_clip(env: TrackEnv, policy: dict, model_ref, clip: str) -> dict:
    """Roll one clip and log it as a two-instance 3D scene + tracking metrics."""
    corpus = env.motion
    model = policy["model"]
    label = scene_label(clip)

    ci = corpus.clip_names.index(clip)
    start, end = int(corpus.clip_start[ci]), int(corpus.clip_end[ci])
    n = end - start

    # rewind to the clip start (rgmt.view / render_policy pattern)
    ids = torch.arange(1, device=DEV)
    env.reset_idx(ids)
    env.idx[:] = start
    env._write_ref_frame(ids, env.idx)
    o = env._build_obs()
    env.history[:] = o.unsqueeze(1).expand(-1, 10, -1).clone()
    bundle = env._bundle()

    sim_qpos, kpe_mm, pose_mm, root_mm = [], [], [], []
    with torch.no_grad():
        for _ in range(n):
            a = model.act_inference(bundle)
            bundle, _, done, _ = env.step(a)
            if bool(done[0]):
                # idx hit the clip's last frame: step() has already auto-reset
                # the env, so this frame is the teleported RSI pose, not ours.
                break
            pos = env.sim.base_pos[0].cpu().numpy()
            quat = env.sim.base_quat[0].cpu().numpy()      # xyzw
            hinges = env.sim.joint_q[0].cpu().numpy()      # (25,)
            sim_qpos.append(np.concatenate([pos, xyzw_to_wxyz(quat), hinges]))

            # tracking error at the advanced idx (same convention as eval_gated)
            ref_kp, _ = corpus.keypoints_at(env.idx)
            rob_kp = env.sim.keypoint_pos
            kpe_mm.append(float((rob_kp - ref_kp).norm(dim=-1).mean()) * 1000.0)
            ref_now = corpus.at(env.idx)
            R_s = quat_to_matrix(env.sim.base_quat).transpose(1, 2)
            R_r = quat_to_matrix(ref_now["base_quat"]).transpose(1, 2)
            pose_mm.append(float((
                torch.einsum("nij,nkj->nki", R_s, rob_kp - env.sim.base_pos[:, None, :])
                - torch.einsum("nij,nkj->nki", R_r, ref_kp - ref_now["base_pos"][:, None, :])
            ).norm(dim=-1).mean()) * 1000.0)
            root_mm.append(float(
                (env.sim.base_pos - ref_now["base_pos"]).norm(dim=-1).mean()) * 1000.0)
    sim_qpos = np.stack(sim_qpos)
    frames = sim_qpos.shape[0]

    # reference qpos over the same frames (sim state at t == ref idx start+1+t)
    idx = torch.arange(start + 1, start + 1 + frames, device=DEV)
    ref = corpus.at(idx)
    ref_hinges = np.zeros((frames, 25), dtype=np.float32)
    ref_hinges[:, env.actuated_idx.cpu().numpy()] = ref["joint_q"].cpu().numpy()
    ref_qpos = np.concatenate([
        ref["base_pos"].cpu().numpy(),
        xyzw_to_wxyz(ref["base_quat"].cpu().numpy()),
        ref_hinges,
    ], axis=-1)

    # common planar origin shift so both instances stay near the scene origin
    origin = ref_qpos[0, :2].copy()
    sim_qpos[:, :2] -= origin
    ref_qpos[:, :2] -= origin

    # ---- MuJoCo forward kinematics -> world body poses ---------------------
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    assert m.nq == sim_qpos.shape[1], f"nq {m.nq} != qpos width {sim_qpos.shape[1]}"
    assert m.nbody == len(model_ref.body_names), (
        f"mj nbody {m.nbody} != body model {len(model_ref.body_names)}")
    d = mujoco.MjData(m)

    def pose_of(q):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        return mj_pose(m, d)

    for t in range(frames):
        nb.log_body_transform(label, model_ref, {
            "reference": pose_of(ref_qpos[t]),
            "policy": pose_of(sim_qpos[t]),
        }, step=t)
        nb.log_line(f"{label}/kpe_mm", kpe_mm[t], step=t)
        nb.log_line(f"{label}/pose_mm", pose_mm[t], step=t)
        nb.log_line(f"{label}/root_err_mm", root_mm[t], step=t)

    summary = {
        "clip": clip, "scene": label, "frames": frames, "clip_steps": n,
        "mean_kpe_mm": round(float(np.mean(kpe_mm)), 1),
        "final_kpe_mm": round(kpe_mm[-1], 1),
        "mean_pose_mm": round(float(np.mean(pose_mm)), 1),
        "max_root_err_mm": round(float(np.max(root_mm)), 1),
    }
    nb.log_text(
        "summary",
        f"{clip}\n  scene '{label}'  {frames} frames ({frames / 60:.1f}s @60Hz)\n"
        f"  MPKPE {summary['mean_kpe_mm']} mm (final {summary['final_kpe_mm']} mm)\n"
        f"  root-relative pose {summary['mean_pose_mm']} mm, "
        f"max root drift {summary['max_root_err_mm']} mm",
    )
    print(f"  {label}: {frames} frames, MPKPE {summary['mean_kpe_mm']} mm, "
          f"max root drift {summary['max_root_err_mm']} mm")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--clip", action="append", dest="clips", default=None)
    ap.add_argument("--preset", choices=sorted(CLIP_SETS), default="main",
                    help="named clip set (ignored when --clip is given)")
    ap.add_argument("--uri", default=".nebo/")
    args = ap.parse_args()
    clips = args.clips or CLIP_SETS[args.preset]

    if not torch.cuda.is_available():
        print("CUDA unavailable -- the rollout needs a GPU. Aborting.", file=sys.stderr)
        return 2
    nb.init(uri=args.uri)
    nb.ui(view="flat", tracker="step")

    policy = load_policy(args.ckpt)
    # The checkpoint knows which corpus cache it was trained against; honour it.
    cache = args.cache or policy["motion_cache"] or DEFAULT_CACHE
    if not Path(cache).is_dir():
        print(f"motion cache not found: {cache}", file=sys.stderr)
        return 2
    print(f"motion cache: {cache}")
    nb.md(
        f"# Policy vs reference -- 3D action render\n\n"
        f"`{args.ckpt}` at **iteration {policy['iteration']}** "
        f"(best-on-test checkpoint), rolled out "
        f"greedily on {len(clips)} held-out clips from `{cache}`.\n\n"
        f"Each clip is one **Actions** scene with two instances of the Asimov v1 "
        f"humanoid:\n\n"
        f"- **reference** -- the GMR-retargeted motion clip, replayed frame by frame "
        f"(what the policy is asked to track).\n"
        f"- **policy** -- the checkpoint rolled out in Newton physics from the clip's "
        f"first frame, greedy actions, no noise.\n\n"
        f"Both are placed by world body transforms from MuJoCo forward kinematics, "
        f"shifted by a common planar origin so they share the scene. **All "
        f"terminations are disabled** (`root_err_done`/`z_dev_done`/`joint_err_done` "
        f"= 1e9, no fall gates), so a tracking failure drifts on camera instead of "
        f"teleport-resetting -- the failing clips are supposed to look bad.\n\n"
        f"Per-step metrics share the tracker playhead with the scenes: `kpe_mm` "
        f"(mean keypoint error), `pose_mm` (root-relative pose error) and "
        f"`root_err_mm` (world root drift). Step *t* is the same frame index in "
        f"every scene, so pressing play advances all clips together.\n\n"
        f"Clips:\n\n" + "\n".join(f"- `{c}`" for c in clips) + "\n"
    )

    corpus = load_corpus(cache)
    env = build_env(corpus, policy)
    model_ref = publish_robot()

    missing = [c for c in clips if c not in corpus.clip_names]
    if missing:
        print(f"clips not in corpus: {missing}", file=sys.stderr)
        return 2

    summaries = [rollout_clip(env, policy, model_ref, c) for c in clips]

    nb.log_bar("mean_kpe_mm", {s["scene"]: s["mean_kpe_mm"] for s in summaries})
    nb.log_bar("max_root_err_mm", {s["scene"]: s["max_root_err_mm"] for s in summaries})
    nb.log_text("run", "rolled %d clips, %d scene frames total"
                % (len(summaries), sum(s["frames"] for s in summaries)))
    nb.flush(timeout=120.0)
    run_id = nb.run_id()
    newest = max(Path(args.uri).glob("*.nebo"), key=lambda f: f.stat().st_mtime,
                 default=None)
    print(f"run_id: {run_id}\nfile:   {newest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

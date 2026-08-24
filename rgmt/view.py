"""Interactive robot viewer — see the Asimov robot in the Newton simulator.

Three input scripts (modes):

  replay  — kinematic playback of a reference clip (ground truth: what the
            motion data looks like on the robot, no physics).
  pd      — physics rollout with ZERO residual action: pure PD tracking of
            the reference joint targets (the policy-free baseline).
  policy  — physics rollout driven by a trained checkpoint (greedy, no
            command noise).

Examples:
  uv run python -m rgmt.view --mode replay --clip "CMU__41__41_05"
  uv run python -m rgmt.view --mode pd     --clip "CMU__41__41_05"
  uv run python -m rgmt.view --mode policy --ckpt <run>/ckpt_000350.pt --clip "CMU__41__41_05"

By default this loads the preprocessed corpus cache and opens a Newton GL
window (close the window or Ctrl-C to exit). Use ``--viewer null`` for a
headless smoke run, ``--viewer rerun|viser`` for those backends, and
``--list`` to print available clip names.
"""

from __future__ import annotations

import argparse
import time

import torch

from rgmt.assets.paths import ROBOT_URDF, ROBOT_XML
from rgmt.data.cache_key import file_sha256
from rgmt.data.corpus import MotionCorpus
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.data.motion import MotionRef
from rgmt.env.track_env import TrackEnv, EnvConfig

def _load_corpus(args, device) -> MotionCorpus:
    if not args.cache and not args.motion_path:
        raise SystemExit("pass --cache <corpus dir> or --motion-path <clip.npz>")
    if args.motion_path:
        ref = MotionRef.load(args.motion_path, ROBOT_XML, ROBOT_URDF, device="cpu",
                             keypoint_links=KEYPOINT_LINKS)
        return MotionCorpus.from_clips([ref], ["clip0"], KEYPOINT_LINKS, output_device=device)
    return MotionCorpus.load_cache(
        args.cache, output_device=device, urdf_hash=file_sha256(ROBOT_URDF),
        physics_fps=60, src_fps=30, keypoint_links=KEYPOINT_LINKS)


def _make_viewer(kind: str, headless: bool):
    import newton.viewer as nv
    if kind == "gl":
        return nv.ViewerGL(headless=headless)
    if kind == "null":
        return nv.ViewerNull(num_frames=10 ** 9)
    if kind == "rerun":
        return nv.ViewerRerun()
    if kind == "viser":
        return nv.ViewerViser()
    raise ValueError(f"unknown viewer kind: {kind}")


def _clip_range(corpus: MotionCorpus, name: str | None) -> tuple[int, int, str]:
    if name is None:
        c = 0
    else:
        if name not in corpus.clip_names:
            raise SystemExit(
                f"clip '{name}' not in corpus ({len(corpus.clip_names)} clips; --list to print)")
        c = corpus.clip_names.index(name)
    return int(corpus.clip_start[c]), int(corpus.clip_end[c]), corpus.clip_names[c]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["replay", "pd", "policy"], default="replay")
    p.add_argument("--cache", default=None, help="preprocessed corpus dir")
    p.add_argument("--motion-path", default=None, help="single raw .npz instead of the cache")
    p.add_argument("--clip", default=None, help="clip name (filename stem); default: first clip")
    p.add_argument("--ckpt", default=None, help="checkpoint .pt for --mode policy")
    p.add_argument("--viewer", choices=["gl", "null", "rerun", "viser"], default="gl")
    p.add_argument("--headless", action="store_true", help="GL without a window")
    p.add_argument("--steps", type=int, default=None, help="max steps (default: clip length / loop)")
    p.add_argument("--fps", type=float, default=60.0, help="real-time pacing for the GL viewer")
    p.add_argument("--no-loop", action="store_true", help="stop at clip end instead of looping")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--list", action="store_true", help="print clip names and exit")
    args = p.parse_args()

    device = torch.device(args.device)
    corpus = _load_corpus(args, device)
    if args.list:
        for n in corpus.clip_names:
            print(n)
        return
    start, end, clip_name = _clip_range(corpus, args.clip)
    n_frames = end - start + 1
    print(f"mode={args.mode}  clip='{clip_name}'  frames={n_frames}  "
          f"({n_frames / 60:.1f}s @60fps)")

    model = None
    drift_obs = False
    drift_obs_proprio = False
    if args.mode == "policy":
        if not args.ckpt:
            raise SystemExit("--mode policy requires --ckpt")
        from rgmt.policy.networks import RGMTActorCritic, PolicyDims
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        dims = PolicyDims(**ck["dims"])
        model = RGMTActorCritic(dims).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        # Ckpts trained with drift-feedback obs need the env built to match.
        saved_env = (ck.get("config") or {}).get("env") or {}
        drift_obs = bool(saved_env.get("drift_obs", False)) or dims.cmd_dim > 55
        drift_obs_proprio = (bool(saved_env.get("drift_obs_proprio", False))
                             or dims.obs_dim > 98)
        print(f"loaded policy from {args.ckpt} (iteration {ck.get('iteration')})")

    cfg = EnvConfig(num_envs=1, keypoint_links=KEYPOINT_LINKS, episode_len=10 ** 9,
                    noise_level=0.0, recovery_fraction=0.0, drift_obs=drift_obs,
                    drift_obs_proprio=drift_obs_proprio)
    env = TrackEnv(cfg, corpus, device=device, train=False)
    ids = torch.arange(1, device=device)

    def rewind() -> dict:
        """Place the env at the clip start and return a fresh bundle."""
        env.reset_idx(ids)
        env.idx[:] = start
        env._write_ref_frame(ids, env.idx)
        o = env._build_obs()
        env.history[:] = o.unsqueeze(1).expand(-1, 10, -1).clone()
        return env._bundle()

    viewer = _make_viewer(args.viewer, args.headless)
    viewer.set_model(env.sim.model)

    bundle = rewind()
    frame_dt = 1.0 / max(args.fps, 1e-3)
    t_sim = 0.0
    step = 0
    max_steps = args.steps if args.steps is not None else (n_frames if args.no_loop else 10 ** 9)

    try:
        while step < max_steps and (not hasattr(viewer, "is_running") or viewer.is_running()):
            t_wall = time.perf_counter()

            if args.mode == "replay":
                # Kinematic: teleport the robot to the reference frame.
                frame = start + (step % n_frames)
                env.idx[:] = frame
                env._write_ref_frame(ids, env.idx)
            else:
                if args.mode == "pd":
                    action = torch.zeros(1, 23, device=device)
                else:
                    with torch.no_grad():
                        action = model.act_inference(bundle)
                bundle, _rew, done, _info = env.step(action)
                at_end = bool((env.idx >= end).item()) or bool(done.item())
                if at_end:
                    if args.no_loop:
                        break
                    bundle = rewind()

            viewer.begin_frame(t_sim)
            viewer.log_state(env.sim.state_0)
            viewer.end_frame()

            t_sim += 1.0 / 60.0
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


if __name__ == "__main__":
    main()

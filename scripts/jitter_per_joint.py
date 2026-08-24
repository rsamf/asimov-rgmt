"""Per-joint commanded-jitter breakdown for a checkpoint.

Rolls N test clips greedily (no noise, matching the render) and reports, per
actuated joint, the mean |Δ commanded target| per control step in mrad —
i.e. the same signal as eval's `jitter_mrad` but decomposed by joint. Use it
to decide WHERE anti-jitter effort should go (global filter vs per-joint
action scale / gains).

Usage:
    uv run python scripts/jitter_per_joint.py <ckpt.pt> \
        [--cache DIR] [--split split.json] [--n-clips 40] [--n-envs 64]
"""
import argparse

import torch

from rgmt.data.corpus import MotionCorpus
from rgmt.data.cache_key import file_sha256
from rgmt.data.joint_map import KEYPOINT_LINKS, ASIMOV_ACTUATED_JOINT_NAMES
from rgmt.assets.paths import ROBOT_URDF
from rgmt.eval_gated import build_eval_env, load_split, split_clip_ids
from rgmt.policy.networks import RGMTActorCritic, PolicyDims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--cache", default="cache/")
    ap.add_argument("--split", default=None, help="restrict to this split's test clips")
    ap.add_argument("--n-clips", type=int, default=40)
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    dev = a.device
    corpus = MotionCorpus.load_cache(
        a.cache, output_device=dev, urdf_hash=file_sha256(ROBOT_URDF),
        physics_fps=60, src_fps=30, keypoint_links=KEYPOINT_LINKS)

    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    dims = PolicyDims(**ck["dims"])
    net_cfg = (ck.get("config") or {}).get("network") or {}
    model = RGMTActorCritic(dims,
        actor_hidden=tuple(net_cfg.get("actor_hidden", [512, 256])),
        critic_hidden=tuple(net_cfg.get("critic_hidden", [512, 256]))).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    senv = (ck.get("config") or {}).get("env") or {}
    env = build_eval_env(
        corpus, dev, a.n_envs,
        drift_obs=bool(senv.get("drift_obs", False)) or dims.cmd_dim > 55,
        drift_obs_proprio=bool(senv.get("drift_obs_proprio", False)) or dims.obs_dim > 98,
        kp=senv.get("kp", 100.0), kd=senv.get("kd", 5.0),
        action_scale=float(senv.get("action_scale", 0.5)),
        action_filter_alpha=float(senv.get("action_filter_alpha", 1.0)),
        effort_limits=senv.get("effort_limits", None),
        dt=float(senv.get("dt", 1.0 / 60.0)),
        control_decimation=int(senv.get("control_decimation", 1)))

    if a.split:
        ids = split_clip_ids(corpus, load_split(a.split), "test")[: a.n_clips]
    else:
        ids = list(range(min(a.n_clips, corpus.n_clips)))

    n_act = env.n_act
    jit_sum = torch.zeros(n_act, device=dev)
    steps = 0
    ids_all = torch.arange(a.n_envs, device=dev)
    for w0 in range(0, len(ids), a.n_envs):
        wave = ids[w0:w0 + a.n_envs]
        k = len(wave)
        starts = torch.tensor([int(corpus.clip_start[c]) for c in wave], device=dev)
        env.reset_idx(ids_all)
        env.idx[:k] = starts
        env._write_ref_frame(ids_all[:k], env.idx[:k])
        o = env._build_obs()
        env.history[:] = o.unsqueeze(1).expand(-1, 10, -1).clone()
        env.ep_step.zero_()
        bundle = env._bundle()
        max_len = max(int(corpus.clip_end[c]) - int(corpus.clip_start[c]) for c in wave)
        resolved = torch.zeros(a.n_envs, dtype=torch.bool, device=dev)
        resolved[k:] = True
        prev = None
        with torch.no_grad():
            for t in range(max_len + 2):
                act = model.act_inference(bundle)
                tgt = env.action_scale * torch.tanh(act)
                if prev is not None:
                    live = (~resolved).float().unsqueeze(1)
                    jit_sum += ((tgt - prev).abs() * live).sum(0)
                    steps += int(live.sum())
                prev = tgt
                bundle, _, done, info = env.step(act)
                resolved |= done
                if bool(resolved.all()):
                    break

    per = (jit_sum / max(steps, 1) * 1000).cpu()
    order = per.argsort(descending=True)
    total = float(per.mean())
    print(f"\nper-joint commanded jitter (mrad/step), {len(ids)} clips, "
          f"{steps} live steps — mean {total:.1f}:")
    for i in order.tolist():
        name = ASIMOV_ACTUATED_JOINT_NAMES[i]
        bar = "#" * int(round(float(per[i]) / max(float(per[order[0]]), 1e-9) * 40))
        print(f"  {name:28} {float(per[i]):6.1f}  {bar}")


if __name__ == "__main__":
    main()

"""Clean numerical eval of a trained checkpoint.

Greedy policy, NO command noise, NO recovery envs -> measures real tracking
quality (keypoint error in meters, root-height error, fall rate) rather than
the noisy stochastic training reward.

Usage: python scripts/eval_ckpt.py <ckpt.pt> [n_envs] [steps] [cache_dir]
"""
import sys
import torch

from rgmt.data.corpus import MotionCorpus
from rgmt.data.cache_key import file_sha256
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_URDF
from rgmt.env.track_env import TrackEnv, EnvConfig
from rgmt.policy.networks import RGMTActorCritic, PolicyDims
from rgmt.utils.rotation import quat_to_matrix

DEFAULT_CACHE = "cache/"


def main(ckpt_path, n_envs=256, steps=400, cache=DEFAULT_CACHE):
    dev = "cuda:0"
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    dims = PolicyDims(**ck["dims"])
    net_cfg = (ck.get("config") or {}).get("network") or {}
    model = RGMTActorCritic(
        dims,
        actor_hidden=tuple(net_cfg.get("actor_hidden", [512, 256])),
        critic_hidden=tuple(net_cfg.get("critic_hidden", [512, 256]))).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    corpus = MotionCorpus.load_cache(
        cache, output_device=dev, urdf_hash=file_sha256(ROBOT_URDF),
        physics_fps=60, src_fps=30, keypoint_links=KEYPOINT_LINKS)

    # Ckpts trained with drift-feedback obs (cmd_dim 60 / obs_dim 103) need the
    # env must be built to match or the widths silently mismatch the network.
    saved_env = (ck.get("config") or {}).get("env") or {}
    drift_obs = bool(saved_env.get("drift_obs", False)) or dims.cmd_dim > 55
    drift_obs_proprio = bool(saved_env.get("drift_obs_proprio", False)) or dims.obs_dim > 98

    # eval plant MUST match the trained plant (per-joint gains, action_scale,
    # filter, torque caps) — read them from the checkpoint's stored config.
    cfg = EnvConfig(
        num_envs=n_envs,
        control_decimation=int(saved_env.get("control_decimation", 1)),
        dt=float(saved_env.get("dt", 1.0 / 60.0)),
        kp=saved_env.get("kp", 100.0), kd=saved_env.get("kd", 5.0),
        action_scale=float(saved_env.get("action_scale", 0.5)),
        action_filter_alpha=float(saved_env.get("action_filter_alpha", 1.0)),
        effort_limits=saved_env.get("effort_limits", None),
        foot_friction=0.75, K=9, L=10, episode_len=100000,
        z_fall=0.12, up_dot_min=0.0, head_z_min=0.3, joint_err_done=100.0,
        root_err_done=1.0, noise_level=0.0,  # NO command noise
        keypoint_links=KEYPOINT_LINKS, recovery_fraction=0.0,  # NO recovery envs
        drift_obs=drift_obs, drift_obs_proprio=drift_obs_proprio)
    env = TrackEnv(cfg, corpus, device=dev, train=False)  # train=False -> no noise

    bundle = env.reset_all()
    mpkpe_sum = z_err_sum = fall_sum = upright_cnt = tot = 0.0
    # Termination-cause counts from the env's PRE-reset info (the auto-reset
    # makes post-hoc fall counting blind — the old fall metric was always ~0).
    ep_ended = falls = trackfails = 0.0
    tf_zdev = tf_rootxy = tf_joint = 0.0
    # Drift breakdown: KPE bucketed by episode age (steps since that env's
    # reset). Open-loop drift shows up as late-age error >> early-age error.
    early_sum = early_cnt = late_sum = late_cnt = 0.0
    rel_sum = rootxy_sum = rootang_sum = relf_sum = 0.0
    perkp_sum = torch.zeros(len(KEYPOINT_LINKS), device=dev)  # per-keypoint (upright)
    with torch.no_grad():
        for t in range(steps):
            action = model.act_inference(bundle)
            bundle, reward, done, info = env.step(action)
            ref_pos, _ = env.motion.keypoints_at(env.idx)          # (N,Kp,3)
            rob_pos = env.sim.keypoint_pos                          # (N,Kp,3)
            perkp = (rob_pos - ref_pos).norm(dim=-1)                # (N,Kp) meters
            perenv_kpe = perkp.mean(dim=1)                          # (N,) meters
            ref_root = env.motion.at(env.idx)["base_pos"]           # (N,3)
            rel_err = ((rob_pos - env.sim.base_pos[:, None, :])
                       - (ref_pos - ref_root[:, None, :])).norm(dim=-1).mean(dim=1)
            rootxy = (env.sim.base_pos[:, :2] - ref_root[:, :2]).norm(dim=-1)
            ref_quat = env.motion.at(env.idx)["base_quat"]
            qdot = (env.sim.base_quat * ref_quat).sum(-1).abs().clamp(max=1.0)
            root_ang = 2.0 * torch.acos(qdot)                       # (N,) rad
            R_s = quat_to_matrix(env.sim.base_quat).transpose(1, 2)
            R_r = quat_to_matrix(ref_quat).transpose(1, 2)
            relf = (torch.einsum("nij,nkj->nki", R_s, rob_pos - env.sim.base_pos[:, None, :])
                    - torch.einsum("nij,nkj->nki", R_r, ref_pos - ref_root[:, None, :])
                    ).norm(dim=-1).mean(dim=1)                      # root-FRAME pose err
            z_err = (env.sim.base_pos[:, 2] - env.motion.at(env.idx)["base_pos"][:, 2]).abs()
            dc = info.get("done_causes", {})
            ep_ended += float(done.float().sum())
            falls += dc.get("fallen", 0.0) * n_envs
            trackfails += dc.get("tracking", 0.0) * n_envs
            tf_zdev += dc.get("tracking_zdev", 0.0) * n_envs
            tf_rootxy += dc.get("tracking_rootxy", 0.0) * n_envs
            tf_joint += dc.get("tracking_joint", 0.0) * n_envs
            fallen = env._fallen()
            up = ~fallen
            # tracking error measured on still-upright envs only
            if up.any():
                mpkpe_sum += float(perenv_kpe[up].mean()) * int(up.sum())
                rel_sum += float(rel_err[up].mean()) * int(up.sum())
                rootxy_sum += float(rootxy[up].mean()) * int(up.sum())
                rootang_sum += float(root_ang[up].mean()) * int(up.sum())
                relf_sum += float(relf[up].mean()) * int(up.sum())
                z_err_sum += float(z_err[up].mean()) * int(up.sum())
                upright_cnt += int(up.sum())
                perkp_sum += perkp[up].sum(dim=0)
                age = env.ep_step
                m_early = up & (age < 100)
                m_late = up & (age >= 300)
                if m_early.any():
                    early_sum += float(perenv_kpe[m_early].sum()); early_cnt += int(m_early.sum())
                if m_late.any():
                    late_sum += float(perenv_kpe[m_late].sum()); late_cnt += int(m_late.sum())
            fall_sum += float(fallen.float().mean())
            tot += 1

    print(f"  MPKPE (upright envs):   {mpkpe_sum/max(upright_cnt,1)*1000:6.1f} mm")
    print(f"  MPKPE ep-age <100:      {early_sum/max(early_cnt,1)*1000:6.1f} mm  (n={int(early_cnt)})")
    print(f"  MPKPE ep-age >=300:     {late_sum/max(late_cnt,1)*1000:6.1f} mm  (n={int(late_cnt)})")
    print(f"  MPKPE root-relative:    {rel_sum/max(upright_cnt,1)*1000:6.1f} mm")
    print(f"  root-XY err:            {rootxy_sum/max(upright_cnt,1)*1000:6.1f} mm")
    print(f"  root orientation err:   {rootang_sum/max(upright_cnt,1)*57.296:6.1f} deg")
    print(f"  pose err (root frame):  {relf_sum/max(upright_cnt,1)*1000:6.1f} mm")
    print(f"  root-z err (upright):   {z_err_sum/max(upright_cnt,1)*1000:6.1f} mm")
    print(f"  fall fraction (steps):  {fall_sum/tot*100:6.1f} %  (frac of env-steps fallen)")
    print(f"  upright env-steps:      {upright_cnt/(n_envs*tot)*100:6.1f} %")
    print(f"  episodes ended:         {int(ep_ended)}  "
          f"(falls {int(falls)} = {falls/max(ep_ended,1)*100:.1f}%, "
          f"tracking-fail {int(trackfails)} = {trackfails/max(ep_ended,1)*100:.1f}%, "
          f"rest = clip end)")
    if trackfails > 0:
        print(f"    tracking-fail split:  z-dev {tf_zdev/trackfails*100:.1f}%  "
              f"root-XY {tf_rootxy/trackfails*100:.1f}%  joint {tf_joint/trackfails*100:.1f}%")
    perkp_mm = (perkp_sum / max(upright_cnt, 1) * 1000).tolist()
    for name, v in sorted(zip(KEYPOINT_LINKS, perkp_mm), key=lambda x: -x[1]):
        print(f"    kp {name:<24s} {v:6.1f} mm")


if __name__ == "__main__":
    ck = sys.argv[1]
    ne = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    st = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    cache = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_CACHE
    print(f"=== eval {ck} (n_envs={ne}, steps={st}, cache={cache}) ===")
    main(ck, ne, st, cache)

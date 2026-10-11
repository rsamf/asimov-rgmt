"""Reproduce the released medium policy (asimov-rgmt-medium).

This is the exact training recipe of the published checkpoint: the RGMT
architecture trained on the easy and medium training clips of the asimov-gmr
reference release (https://github.com/rsamf/asimov-gmr), under domain
randomization, URDF torque caps, and the split-defined in-loop robust eval.
It builds an OmegaConf config directly (no Hydra CLI) so it runs cleanly
detached in the background.

Point --cache at a corpus baked by `rgmt.preprocess` from the reference
release's training set. The full run takes 34,000 iterations at 8192 envs.

Usage:
    uv run python scripts/train_medium.py --cache cache/           # full run
    uv run python scripts/train_medium.py --cache cache/ --smoke   # pipeline smoke
"""
import argparse
import os

from omegaconf import OmegaConf

from rgmt.train import run_training
from rgmt.env.gains import gains_profile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT = os.path.join(REPO, "rgmt", "data", "splits", "medium.json")


def build_cfg(cache: str, log_dir: str, smoke: bool, device: str, seed: int,
              mirror: bool = False, gains: str = "leg_weighted",
              body_frame_ang_vel: bool = False, exp_name: str | None = None,
              split_json: str | None = None, effort_dr: tuple[float, float] = (0.8, 1.2)):
    kp, kd = gains_profile(gains)
    exp_name = exp_name or "rgmt_medium"
    return OmegaConf.create(dict(
        experiment_name=f"{exp_name}_smoke" if smoke else exp_name,
        seed=seed,
        device=device,
        log_dir=log_dir,
        motion_cache=cache,
        motion_dir=None,
        motion_path=None,
        mirror_augment=mirror,   # off in the released recipe
        env=dict(
            num_envs=256 if smoke else 8192,
            control_decimation=1,
            dt=1.0 / 60.0,
            kp=kp,
            kd=kd,
            action_scale=0.3,
            foot_friction=0.75,
            K=9,
            L=10,
            episode_len=1000,
            z_fall=0.12,
            up_dot_min=0.0,
            head_z_min=0.3,
            joint_err_done=100.0,
            root_err_done=0.25,
            z_dev_done=0.2,
            noise_level=1.0,
            noise=dict(v=[0.5, 0.5, 0.2], w=0.52, g=0.05, q=0.1),
            keypoint_links=None,
            recovery_fraction=0.0,
            drift_obs=True,
            # Plant v2 recipe: omega in the pelvis frame, like the command
            # window and a real IMU (plant v1 checkpoints used world frame).
            body_frame_ang_vel=body_frame_ang_vel,
            action_filter_alpha=0.7,
            # Sim2real plant: per-joint datasheet torque caps from the URDF,
            # plus episode-consistent domain randomization with a privileged
            # critic (the actor infers dynamics from its 10-step history).
            effort_limits="urdf",
            dr=dict(
                friction_range=[0.4, 1.0],
                mass_scale_range=[0.9, 1.1],
                kp_scale_range=[0.9, 1.1],
                kd_scale_range=[0.9, 1.1],
                effort_scale_range=list(effort_dr),
                privileged=True,
            ),
        ),
        algo=dict(
            iterations=12 if smoke else 34000,
            rollout_len=32,
            lr=1.5e-4,
            n_epochs=1,
            mb_size=16384,
            clip=0.2,
            value_coef=1.0,
            entropy_coef=0.0003,
            entropy_final=1.0e-4,
            entropy_anneal_frac=0.3,
            lambda_smooth=0.1,
            smooth_slack=2.5,
            smooth_eps_floor=0.02,
            lambda_spatial=0.01,
            spatial_noise_std=0.05,
            spatial_slack=2.5,
            spatial_floor=0.30,
            max_grad_norm=1.0,
            dual_clip=3.0,
            target_kl=0.02,
            kl_shock_factor=3.0,
            # Plain MSE value loss plus separate actor/critic grad clips.
            # The clipped value loss rate-limited the critic, and a shared
            # global grad clip scaled the actor's step by ~0.005.
            value_clip=False,
            grad_clip_per_module=True,
            gamma=0.995,
            lam=0.95,
            lr_final=None,
            kl_adaptive=True,
            lr_min=1.0e-5,
            lr_max=1.0e-3,
            assist_anneal_iters=None,
        ),
        network=dict(n_embd=256, n_heads=8,
                     actor_hidden=[1024, 512], critic_hidden=[1024, 512]),
        reward=dict(
            # Global-to-local rebalance: root drift is otherwise triple-counted
            # (world keypoints inherit it while the root terms charge for it
            # again), so weight shifts toward root-relative pose.
            w_kp=0.6,
            w_rel=0.6,
            w_rq=0.4,
            w_rp=0.25,
            w_kpv=0.5,
            w_arate=0.01,
        ),
        eval=dict(
            every=6 if smoke else 100,
            greedy=True,
            robust_every=12 if smoke else 600,
            robust_envs=256,
            robust_noise=0.05,
            robust_repeats=1 if smoke else 3,
            split_json=split_json or SPLIT,
            mining=True,
            mining_ema=0.5,
            mining_boost=3.0,
        ),
    ))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True,
                    help="preprocessed corpus dir (rgmt.preprocess output)")
    ap.add_argument("--log-dir", default=os.path.join(REPO, "runs"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="12-iteration pipeline smoke run at 256 envs")
    ap.add_argument("--mirror", action="store_true",
                    help="left-right mirror augmentation (2x train clips); off in the released recipe")
    ap.add_argument("--gains", default="leg_weighted",
                    help="PD gain profile: leg_weighted (plant v1 recipe) | caps_v2 | caps:<e_sat> (plant v2)")
    ap.add_argument("--body-frame-ang-vel", action="store_true",
                    help="observe pelvis angular velocity in the body frame (plant v2 recipe)")
    ap.add_argument("--exp-name", default=None, help="experiment name (default rgmt_medium)")
    ap.add_argument("--effort-dr", nargs=2, type=float, default=[0.8, 1.2], metavar=("LO", "HI"),
                    help="torque-cap DR scale range (plant v2 recipe: 0.9 1.2, from the PD sweep)")
    ap.add_argument("--split", default=None,
                    help="train/test split json (default rgmt/data/splits/medium.json)")
    args = ap.parse_args()
    if args.smoke:
        os.environ.setdefault("NEBO_NO_STORE", "1")
    stats = run_training(build_cfg(args.cache, args.log_dir, args.smoke,
                                   args.device, args.seed, mirror=args.mirror,
                                   gains=args.gains, body_frame_ang_vel=args.body_frame_ang_vel,
                                   exp_name=args.exp_name, split_json=args.split,
                                   effort_dr=tuple(args.effort_dr)))
    print("TRAINING COMPLETE. final stats:", stats)

"""GPU smoke test: two training iterations end-to-end."""

import pytest
import torch

_HAS = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not _HAS, reason="needs CUDA")

from omegaconf import OmegaConf
from rgmt.train import run_training
from rgmt.assets.paths import ROBOT_XML  # noqa: F401
from tests.fixtures.make_synthetic_motion import make_synthetic_motion, make_synthetic_clips
from rgmt.preprocess import run_preprocess


def test_two_iterations(tmp_path):
    p = make_synthetic_motion(tmp_path / "m.npz", n_frames=200)
    cfg = OmegaConf.create(dict(
        experiment_name="smoke",
        seed=0,
        device="cuda:0",
        log_dir=str(tmp_path / "runs"),
        motion_path=str(p),
        env=dict(
            num_envs=64,
            control_decimation=1,
            dt=0.016666,
            kp=100.0,
            kd=5.0,
            action_scale=0.5,
            foot_friction=0.75,
            K=9,
            L=10,
            episode_len=200,
            z_fall=0.12,
            up_dot_min=0.0,
            head_z_min=0.3,
            joint_err_done=100.0,
            root_err_done=5.0,
            noise_level=1.0,
            noise=dict(v=[0.5, 0.5, 0.2], w=0.52, g=0.05, q=0.1),
            keypoint_links=None,
            recovery_fraction=0.15,
            # Sim2real plant + DR path: URDF torque caps and privileged domain
            # randomization exercised end-to-end (priv_dim 187 + 6).
            effort_limits="urdf",
            dr=dict(friction_range=[0.4, 1.0], kp_scale_range=[0.9, 1.1],
                    effort_scale_range=[0.8, 1.2], privileged=True),
        ),
        algo=dict(
            iterations=2,
            rollout_len=8,
            lr=3e-4,
            n_epochs=2,
            mb_size=256,
            clip=0.2,
            value_coef=1.0,
            entropy_coef=0.005,
            max_grad_norm=1.0,
            dual_clip=3.0,
            target_kl=0.02,
            gamma=0.99,
            lam=0.95,
        ),
        network=dict(
            n_embd=128,
            n_heads=4,
            actor_hidden=[512, 256],
            critic_hidden=[512, 256],
        ),
        reward=dict(),
        eval=dict(every=50, greedy=True),
    ))
    stats = run_training(cfg)
    assert "avg_return" in stats
    assert torch.isfinite(torch.tensor(stats["avg_return"])), (
        f"avg_return is not finite: {stats['avg_return']}"
    )
    assert "assist_scale" in stats, (
        f"run_training must return 'assist_scale'; got keys: {list(stats.keys())}"
    )


def test_two_iterations_from_cache(tmp_path):
    if not torch.cuda.is_available():
        import pytest; pytest.skip("needs CUDA")
    src = tmp_path / "src"; make_synthetic_clips(src, [("a", 120), ("b", 90)])
    run_preprocess(OmegaConf.create(dict(motion_dir=str(src), motion_path=None,
        cache_dir=str(tmp_path / "cache"), physics_fps=60, src_fps=30,
        keypoint_links=None, force=False)))
    cfg = OmegaConf.create(dict(
        experiment_name="smoke_cache", seed=0, device="cuda:0", log_dir=str(tmp_path / "runs"),
        motion_cache=str(tmp_path / "cache"), motion_dir=None, motion_path=None,
        env=dict(num_envs=64, control_decimation=1, dt=0.016666, kp=100.0, kd=5.0,
                 action_scale=0.5, foot_friction=0.75, K=9, L=10, episode_len=200,
                 z_fall=0.12, up_dot_min=0.0, head_z_min=0.3, joint_err_done=100.0, root_err_done=5.0,
                 noise_level=1.0, noise=dict(v=[0.5, 0.5, 0.2], w=0.52, g=0.05, q=0.1),
                 keypoint_links=None, recovery_fraction=0.15),
        algo=dict(iterations=2, rollout_len=8, lr=3e-4, n_epochs=2, mb_size=256, clip=0.2,
                  value_coef=1.0, entropy_coef=0.005, max_grad_norm=1.0, dual_clip=3.0,
                  target_kl=0.02, gamma=0.99, lam=0.95, assist_anneal_iters=2),
        network=dict(n_embd=128, n_heads=4, actor_hidden=[512, 256], critic_hidden=[512, 256]),
        reward=dict(), eval=dict(every=50, greedy=True)))
    stats = run_training(cfg)
    assert "avg_return" in stats

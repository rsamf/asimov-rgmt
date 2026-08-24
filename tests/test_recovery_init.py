"""Tests for recovery-env designation and randomized unstable initialization.

~15% of parallel envs are designated "recovery" envs at init time.
Recovery envs reset to randomized UNSTABLE poses rather than RSI from a clean
reference frame (paper §II-D).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"
)

from rgmt.env.track_env import TrackEnv, EnvConfig
from rgmt.data.motion import MotionRef
from rgmt.data.corpus import MotionCorpus
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from tests.fixtures.make_synthetic_motion import make_synthetic_motion


def test_recovery_fraction(tmp_path):
    """Bernoulli(0.15) mask: expect fraction in [0.10, 0.20] for N=1000."""
    p = make_synthetic_motion(tmp_path / "m.npz", n_frames=120)
    m = MotionRef.load(
        p, ROBOT_XML, ROBOT_URDF, device="cuda:0", keypoint_links=KEYPOINT_LINKS
    )
    env = TrackEnv(
        EnvConfig(num_envs=1000, recovery_fraction=0.15, keypoint_links=KEYPOINT_LINKS),
        m,
        "cuda:0",
    )
    frac = env.is_recovery.float().mean().item()
    assert 0.10 < frac < 0.20, (
        f"Expected recovery fraction in (0.10, 0.20), got {frac:.4f}"
    )


def _make_recovery_env(tmp_path, n_envs=64, n_frames=240, **cfg_kwargs):
    p = make_synthetic_motion(tmp_path / "m.npz", n_frames=n_frames)
    m = MotionRef.load(
        p, ROBOT_XML, ROBOT_URDF, device="cuda:0", keypoint_links=KEYPOINT_LINKS
    )
    corpus = MotionCorpus.from_clips(
        [m], ["clip0"], KEYPOINT_LINKS, output_device="cuda:0"
    )
    kwargs = dict(num_envs=n_envs, keypoint_links=KEYPOINT_LINKS)
    kwargs.update(cfg_kwargs)
    return TrackEnv(EnvConfig(**kwargs), corpus, "cuda:0")


def test_recovery_spawn_near_reference(tmp_path):
    """Recovery spawns are centred on the reference root at the env's idx.

    World-frame reward + root_err_done make an origin-spawn unlearnable; the
    unstable pose must sit inside the survivable radius of ITS reference frame.
    """
    env = _make_recovery_env(tmp_path, recovery_fraction=1.0)
    env.reset_all()
    ref = env.motion.at(env.idx)
    dxy = (env.sim.base_pos[:, :2] - ref["base_pos"][:, :2].to(env.device)).abs()
    assert (dxy <= 0.25 + 1e-5).all(), f"spawn xy jitter exceeded 0.25: {dxy.max():.3f}"
    z = env.sim.base_pos[:, 2]
    assert (z >= 0.10 - 1e-5).all() and (z <= 0.45 + 1e-5).all()
    # joint angles come from THIS env's reference frame +- 0.5
    q_act = env.sim.joint_q[:, env.actuated_idx]
    dq = (q_act - ref["joint_q"].to(env.device)).abs()
    assert (dq <= 0.5 + 1e-4).all(), f"joint noise exceeded 0.5: {dq.max():.3f}"


def test_recovery_window_suppresses_instability_termination(tmp_path):
    """Within the 3 s window, fallen/tracking criteria must not terminate.

    All-recovery env, unstable spawns -> many envs are 'fallen' immediately,
    yet done_causes must show zero fallen/tracking terminations in-window.
    """
    env = _make_recovery_env(
        tmp_path, recovery_fraction=1.0, recovery_window_s=3.0
    )
    env.reset_all()
    saw_fallen = False
    for _ in range(5):
        a = torch.zeros(env.N, env.n_act, device=env.device)
        _, _, done, info = env.step(a)
        saw_fallen = saw_fallen or bool(info["fallen"].any())
        assert info["done_causes"]["fallen"] == 0.0
        assert info["done_causes"]["tracking"] == 0.0
        assert not bool(done.any()), "no env should terminate this early in-window"
    assert saw_fallen, "unstable spawns should trip the fallen flag (else the test is vacuous)"


def test_recovery_reward_shield(tmp_path):
    """In-window fallen envs are not fined (fall_pen 0, alive granted)."""
    env = _make_recovery_env(
        tmp_path, recovery_fraction=1.0, recovery_reward_shield=True
    )
    env.reset_all()
    a = torch.zeros(env.N, env.n_act, device=env.device)
    _, _, _, info = env.step(a)
    assert bool(info["fallen"].any()), "unstable spawns should include fallen states"
    assert float(info["terms"]["fall_pen"].abs().max()) == 0.0
    assert (info["terms"]["alive"] > 0).all()

    env_off = _make_recovery_env(
        tmp_path, recovery_fraction=1.0, recovery_reward_shield=False
    )
    env_off.reset_all()
    _, _, _, info_off = env_off.step(a)
    assert float(info_off["terms"]["fall_pen"].max()) > 0.0, (
        "with shield off, in-window fallen envs must still be fined"
    )


def test_recovery_spawn_knobs(tmp_path):
    """Spawn tilt/z ranges are configurable."""
    env = _make_recovery_env(
        tmp_path, recovery_fraction=1.0,
        recovery_spawn_tilt_max=0.0,
        recovery_spawn_z_min=0.40, recovery_spawn_z_max=0.44,
    )
    env.reset_all()
    z = env.sim.base_pos[:, 2]
    assert (z >= 0.40 - 1e-5).all() and (z <= 0.44 + 1e-5).all()
    # tilt_max=0 -> identity orientation -> body-up.z == 1
    from rgmt.utils.rotation import quat_to_matrix
    up_z = quat_to_matrix(env.sim.base_quat)[:, 2, 2]
    assert (up_z > 0.999).all()


def test_assist_force_scaling(tmp_path):
    """Assist force: recovery-only, U[0, assist_force_max * scale], zero at scale 0."""
    env = _make_recovery_env(
        tmp_path, n_envs=256, recovery_fraction=0.5, assist_force_max=200.0
    )
    env.set_assist_scale(1.0)
    f = env._assist_force
    assert (f[~env.is_recovery] == 0).all()
    rec_fz = f[env.is_recovery, 2]
    assert (rec_fz >= 0).all() and (rec_fz <= 200.0).all() and rec_fz.max() > 50.0
    assert (f[:, :2] == 0).all(), "assist is world +z only"
    env.set_assist_scale(0.25)
    assert env._assist_force[:, 2].max() <= 50.0 + 1e-5
    env.set_assist_scale(0.0)
    assert (env._assist_force == 0).all()


def test_recovery_local_reward_contract(tmp_path):
    """In-window reward equals 2*r_rel + alive - act_pen - arate_pen."""
    env = _make_recovery_env(
        tmp_path, recovery_fraction=1.0, recovery_reward_shield=True,
        recovery_local_reward=True,
    )
    env.reset_all()
    a = torch.zeros(env.N, env.n_act, device=env.device)
    _, reward, _, info = env.step(a)
    t = info["terms"]
    expected = 2.0 * t["r_rel"] + t["alive"] - t["act_pen"] - t["arate_pen"]
    assert torch.allclose(reward, expected, atol=1e-5), (
        f"in-window local reward mismatch: max dev "
        f"{float((reward - expected).abs().max()):.2e}"
    )

"""Action low-pass filter (EnvConfig.action_filter_alpha) semantics."""
import pytest, torch
_HAS = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not _HAS, reason="needs CUDA")
from rgmt.env.track_env import TrackEnv, EnvConfig
from rgmt.data.motion import MotionRef
from rgmt.data.corpus import MotionCorpus
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from tests.fixtures.make_synthetic_motion import make_synthetic_motion


def _env(tmp_path, alpha):
    p = make_synthetic_motion(tmp_path / "m.npz", n_frames=120)
    m = MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device="cuda:0", keypoint_links=KEYPOINT_LINKS)
    cfg = EnvConfig(num_envs=2, keypoint_links=KEYPOINT_LINKS, action_filter_alpha=alpha)
    corpus = MotionCorpus.from_clips([m], ["clip0"], KEYPOINT_LINKS, output_device="cuda:0")
    env = TrackEnv(cfg, corpus, device="cuda:0")
    env.reset_all()
    return env


def test_filter_state_converges_geometrically(tmp_path):
    env = _env(tmp_path, alpha=0.5)
    a = torch.ones(2, 23, device="cuda:0")           # constant action
    r = torch.tanh(a[0, 0]) * env.action_scale       # steady-state residual
    env.step(a)
    assert torch.allclose(env.filt_res, 0.5 * r * torch.ones_like(env.filt_res), atol=1e-6)
    env.step(a)
    # second step: 0.5*r + 0.5*(0.5*r) = 0.75*r  (unless an env auto-reset zeroed it)
    live = env.ep_step.squeeze() > 1 if env.ep_step.dim() > 1 else env.ep_step > 1
    if bool(live.any()):
        assert torch.allclose(env.filt_res[live], 0.75 * r * torch.ones_like(env.filt_res[live]),
                              atol=1e-6)


def test_alpha_one_keeps_filter_state_zero(tmp_path):
    env = _env(tmp_path, alpha=1.0)
    env.step(torch.ones(2, 23, device="cuda:0"))
    # filter path is skipped entirely at alpha=1.0 — state remains zero
    assert torch.all(env.filt_res == 0.0)


def test_reset_zeroes_filter_state(tmp_path):
    env = _env(tmp_path, alpha=0.5)
    env.step(torch.ones(2, 23, device="cuda:0"))
    assert bool((env.filt_res != 0).any())
    env.reset_idx(torch.tensor([0], device="cuda:0"))
    assert torch.all(env.filt_res[0] == 0.0)
    assert bool((env.filt_res[1] != 0).any())

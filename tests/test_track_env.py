import pytest, torch
_HAS = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not _HAS, reason="needs CUDA")
from rgmt.env.track_env import TrackEnv, EnvConfig
from rgmt.data.motion import MotionRef
from rgmt.data.corpus import MotionCorpus
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from tests.fixtures.make_synthetic_motion import make_synthetic_motion

def test_step_bundle_shapes(tmp_path):
    p = make_synthetic_motion(tmp_path / "m.npz", n_frames=120)
    m = MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device="cuda:0", keypoint_links=KEYPOINT_LINKS)
    cfg = EnvConfig(num_envs=4, keypoint_links=KEYPOINT_LINKS)
    corpus = MotionCorpus.from_clips([m], ["clip0"], KEYPOINT_LINKS, output_device="cuda:0")
    env = TrackEnv(cfg, corpus, device="cuda:0")
    b = env.reset_all()
    assert b["obs"].shape == (4, 98) and b["history"].shape == (4, 10, 98)
    assert b["cmd_window"].shape == (4, 21, 55) and b["critic_obs"].shape == (4, env.priv_dim)
    b2, rew, done, info = env.step(torch.zeros(4, 23, device="cuda:0"))
    assert rew.shape == (4,) and done.shape == (4,)
    # finiteness
    for k, v in b2.items():
        assert torch.isfinite(v).all(), f"non-finite in bundle[{k}]"
    assert torch.isfinite(rew).all()

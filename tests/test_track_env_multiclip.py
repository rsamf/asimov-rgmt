"""GPU test: TrackEnv multi-clip corpus integration with clip-boundary termination."""
import pytest, torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from rgmt.env.track_env import TrackEnv, EnvConfig, _clip_aware_grid
from rgmt.data.motion import MotionRef
from rgmt.data.corpus import MotionCorpus
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from tests.fixtures.make_synthetic_motion import make_synthetic_clips


def test_noisy_window_qblock_stays_in_clip(tmp_path):
    """Grid indices built by _clip_aware_grid must never exceed per-env clip bounds.

    Directly tests the helper extracted from ``_command_window_noisy`` to prove
    that frames near a clip boundary are clamped to that clip and do not index
    into an adjacent clip.  A global clamp (.clamp(0, n_frames-1)) would allow
    frames near clip a's end to reach into clip b — this test catches that.
    """
    paths = make_synthetic_clips(tmp_path, [("a", 40), ("b", 50)])
    clips = [MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device="cuda:0", keypoint_links=KEYPOINT_LINKS)
             for p in paths]
    corpus = MotionCorpus.from_clips(clips, ["a", "b"], KEYPOINT_LINKS, output_device="cuda:0")

    # Place query indices near the boundary between clip a and clip b.
    # With L=10, a global clamp would pull window frames from clip b for idx near end0.
    end0 = int(corpus.clip_end[0])   # last global frame of clip a (0-indexed)
    idx = torch.tensor([end0 - 2, end0 - 1, end0, end0 + 5])  # last env is inside clip b
    L = 10

    grid = _clip_aware_grid(idx, L, corpus)  # (4, 21)

    # For each env, every grid index must stay within the owning clip's bounds.
    cid = corpus.frame_clip_id[idx]
    lo = corpus.clip_start[cid]   # (4,)
    hi = corpus.clip_end[cid]     # (4,)

    assert grid.shape == (4, 2 * L + 1), f"unexpected grid shape {grid.shape}"

    for i in range(4):
        env_lo, env_hi = int(lo[i]), int(hi[i])
        env_min = int(grid[i].min())
        env_max = int(grid[i].max())
        assert env_min >= env_lo, (
            f"env {i}: grid min {env_min} < clip_start {env_lo} (cross-clip bleed!)"
        )
        assert env_max <= env_hi, (
            f"env {i}: grid max {env_max} > clip_end {env_hi} (cross-clip bleed!)"
        )


def test_multiclip_step_and_clip_end(tmp_path):
    paths = make_synthetic_clips(tmp_path, [("a", 40), ("b", 60), ("c", 30)])
    clips = [MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device="cuda:0", keypoint_links=KEYPOINT_LINKS)
             for p in paths]
    corpus = MotionCorpus.from_clips(clips, ["a", "b", "c"], KEYPOINT_LINKS, output_device="cuda:0")
    env = TrackEnv(EnvConfig(num_envs=8, episode_len=10_000, keypoint_links=KEYPOINT_LINKS),
                   corpus, device="cuda:0")
    b = env.reset_all()
    assert b["obs"].shape == (8, 98) and b["critic_obs"].shape == (8, env.priv_dim)
    # force every env to its clip end and confirm motion_end triggers a reset (idx jumps back into a clip)
    env.idx[:] = corpus.clip_end_of(env.idx)
    b2, rew, done, info = env.step(torch.zeros(8, 23, device="cuda:0"))
    assert done.all()                       # all were at clip end -> motion_end
    assert torch.isfinite(rew).all()
    # after auto-reset, every env idx is a valid in-corpus frame
    assert torch.all(env.idx >= 0) and torch.all(env.idx < corpus.n_frames)

"""Physics ground truth for the mirror augmentation: Newton FK on the MJCF.

Reset one env with a per-joint-distinct pose and a tilted root, another with
the mirrored pose/root, and demand the simulator's body positions come out as
the mirror image. This is the only check that involves the sim's own asset
(the MJCF) rather than the URDF the reference FK uses.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from rgmt.assets.paths import ROBOT_URDF, ROBOT_XML
from rgmt.data import mirror as mr
from rgmt.data.corpus import MotionCorpus
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.data.motion import MotionRef
from rgmt.env.sim import NewtonSim
from rgmt.env.track_env import EnvConfig, TrackEnv
from tests.fixtures.make_synthetic_motion import make_synthetic_motion

DEV = "cuda:0"
ATOL = 1e-3   # model asymmetries up to 0.58 mm (docs/reports/asimov-model-asymmetries-2026-07-13.md)


@pytest.fixture(scope="module")
def sim():
    return NewtonSim(2, kp=100.0, kd=5.0, control_decimation=1, dt=1.0 / 60.0,
                     foot_friction=0.75, keypoint_links=KEYPOINT_LINKS, device=DEV)


def test_sim_fk_of_mirrored_pose_is_mirror_of_fk(sim):
    dev = torch.device(DEV)
    g = torch.Generator().manual_seed(3)
    q = (0.05 * (torch.arange(23) + 1) * (-1.0) ** torch.arange(23)).to(dev)   # distinct per joint
    quat = torch.randn(4, generator=g); quat = (quat / quat.norm()).to(dev)
    pos = torch.tensor([0.3, 0.2, 1.2], device=dev)                              # airborne
    hq = torch.zeros(2, sim.n_hinges, device=dev)
    hq[0, sim.actuated_idx] = q
    hq[1, sim.actuated_idx] = mr.mirror_joints(q)
    zeros = torch.zeros(2, 3, device=dev)
    sim.reset_idx(torch.arange(2, device=dev),
                  torch.stack([pos, mr.mirror_vec(pos)]),
                  torch.stack([quat, mr.mirror_quat(quat)]),
                  zeros, zeros, hq, torch.zeros(2, sim.n_hinges, device=dev))
    kp = sim.keypoint_pos                                                        # (2, Kp, 3)
    err = (kp[1] - mr.mirror_keypoints(kp[0])).norm(dim=-1)
    assert err.max() < ATOL, f"max keypoint mirror error {err.max():.2e} m; per-link {err.tolist()}"
    # sanity: the pose is genuinely asymmetric, so a wrong sign would be caught
    assert (kp[1] - kp[0]).norm(dim=-1).max() > 0.05


def test_track_env_reset_on_mirrored_clip_matches_mirrored_reference(tmp_path):
    """Ties the corpus mirror, URDF FK reference keypoints and Newton FK together."""
    p = make_synthetic_motion(tmp_path / "m.npz", n_frames=60, asymmetric=True)
    ref = MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device="cpu", keypoint_links=KEYPOINT_LINKS)
    corpus = mr.mirror_corpus(
        MotionCorpus.from_clips([ref], ["m"], KEYPOINT_LINKS, output_device=DEV))
    env = TrackEnv(EnvConfig(num_envs=2, keypoint_links=KEYPOINT_LINKS, recovery_fraction=0.0),
                   corpus, device=DEV, train=False)
    F = ref.n_frames
    f = 17
    idx = torch.tensor([f, f + F], device=DEV)
    env.reset_all()
    env.idx[:] = idx
    env._write_ref_frame(torch.arange(2, device=DEV), idx)
    kp_sim = env.sim.keypoint_pos
    kp_ref, _ = corpus.keypoints_at(idx)
    # sim body positions are the mirror of each other
    assert (kp_sim[1] - mr.mirror_keypoints(kp_sim[0])).norm(dim=-1).max() < ATOL
    # and each matches its own reference keypoints (URDF FK vs Newton FK)
    assert (kp_sim - kp_ref).norm(dim=-1).max() < ATOL

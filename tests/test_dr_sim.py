"""DR sim integration: per-env parameter writes, notify batching, dynamics.

CUDA-gated (NewtonSim). CPU-side sampling math is in test_domain_rand.py.
"""
import pytest
import torch

torch.manual_seed(0)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NewtonSim requires CUDA")

try:
    import warp  # noqa: F401
    _HAS_WARP = True
except ImportError:
    _HAS_WARP = False
if not _HAS_WARP:
    pytestmark = pytest.mark.skip(reason="warp not installed")

from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.env.gains import urdf_effort_limits


def _sim(n=4, effort=None):
    from rgmt.env.sim import NewtonSim
    return NewtonSim(num_envs=n, kp=100.0, kd=5.0, control_decimation=1,
                     dt=1.0 / 60.0, foot_friction=0.75,
                     keypoint_links=list(KEYPOINT_LINKS), device="cuda:0",
                     effort_limit=effort)


def _flat(sim, arr):
    import warp as wp
    return wp.to_torch(arr)


def test_effort_limit_lands_per_env():
    eff = urdf_effort_limits()
    sim = _sim(3, effort=eff)
    flat = _flat(sim, sim.model.joint_effort_limit)
    per_env = flat[sim.actuated_dof_global]           # (3, 23)
    want = torch.tensor(eff, device=per_env.device)
    for e in range(3):
        assert torch.allclose(per_env[e], want)


def test_setters_write_per_env_and_batch_one_notify():
    sim = _sim(4)
    calls = []
    orig = sim.solver.notify_model_changed
    sim.solver.notify_model_changed = lambda f: (calls.append(f), orig(f))[1]

    mu = torch.tensor([0.4, 0.6, 0.8, 1.0], device="cuda:0")
    kp_s = torch.tensor([0.9, 1.0, 1.1, 1.0], device="cuda:0")
    kd_s = torch.ones(4, device="cuda:0")
    ms = torch.tensor([0.9, 1.0, 1.1, 1.05], device="cuda:0")

    sim.set_foot_friction_per_env(mu)
    sim.set_joint_gain_scale(kp_s, kd_s)
    sim.set_body_mass_scale(ms)
    assert calls == [], "setters must not notify"
    flags = sim.apply_dynamics_changes()
    assert len(calls) == 1 and calls[0] == flags and flags != 0
    assert sim.apply_dynamics_changes() == 0, "clean accumulator -> no notify"
    assert len(calls) == 1

    # per-env values landed in the flat Newton arrays
    mu_flat = _flat(sim, sim.model.shape_material_mu)
    got_mu = mu_flat[sim.foot_shape_idx]              # (4, 8)
    assert torch.allclose(got_mu, mu.unsqueeze(1).expand(-1, 8))
    ke = _flat(sim, sim.model.joint_target_ke)[sim.actuated_dof_global]
    nominal = sim._nominal_joint_target_ke[sim.actuated_dof_global]
    assert torch.allclose(ke, nominal * kp_s.unsqueeze(1))
    mass = _flat(sim, sim.model.body_mass)[sim.body_idx_grid]
    nom_mass = sim._nominal_body_mass[sim.body_idx_grid]
    assert torch.allclose(mass, nom_mass * ms.unsqueeze(1))

    # and in the per-world MJWarp arrays after the notify
    import warp as wp
    mjw_mass = wp.to_torch(sim.solver.mjw_model.body_mass)     # (4, nbody)
    ratio = mjw_mass[:, 1:] / mjw_mass[0:1, 1:]                # skip world body
    assert torch.allclose(ratio.mean(dim=1), ms / ms[0], atol=1e-5)


def test_low_effort_env_collapses_nominal_stands():
    # Leg-weighted gains: the profile that actually stands at the zero pose
    # (uniform kp=100/kd=5 collapses within 2 s even WITHOUT torque caps, so
    # it can't discriminate the effort-limit effect).
    from rgmt.env.sim import NewtonSim
    from rgmt.env.gains import leg_weighted_gains
    KP, KD = leg_weighted_gains()
    eff = urdf_effort_limits()
    sim = NewtonSim(num_envs=2, kp=KP, kd=KD, control_decimation=1,
                    dt=1.0 / 60.0, foot_friction=0.75,
                    keypoint_links=list(KEYPOINT_LINKS), device="cuda:0",
                    effort_limit=eff)
    scale = torch.tensor([1.0, 0.02], device="cuda:0")   # env1: ~1-3 Nm caps
    sim.set_effort_limit_scale(scale)
    sim.apply_dynamics_changes()
    tgt = torch.zeros(2, 23, device="cuda:0")
    for _ in range(120):
        sim.step(tgt)
    z = sim.base_pos[:, 2]
    assert float(z[0]) > 0.35, f"nominal env should stand, base_z={float(z[0]):.3f}"
    assert float(z[1]) < 0.30, f"torque-starved env should collapse, base_z={float(z[1]):.3f}"


def test_track_env_dr_cadence(tmp_path):
    from tests.fixtures.make_synthetic_motion import make_synthetic_motion
    from rgmt.data.motion import MotionRef
    from rgmt.data.corpus import MotionCorpus
    from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
    from rgmt.env.track_env import TrackEnv, EnvConfig

    npz = tmp_path / "clip.npz"
    make_synthetic_motion(npz, n_frames=60)
    ref = MotionRef.load(npz, ROBOT_XML, ROBOT_URDF, device="cpu",
                         keypoint_links=list(KEYPOINT_LINKS))
    corpus = MotionCorpus.from_clips([ref], ["clip"], list(KEYPOINT_LINKS),
                                     output_device="cuda:0")
    cfg = EnvConfig(num_envs=4, keypoint_links=list(KEYPOINT_LINKS),
                    noise_level=0.0, recovery_fraction=0.0,
                    dr=dict(friction_range=[0.4, 1.0], privileged=True))
    env = TrackEnv(cfg, corpus, device="cuda:0", train=True)
    assert env.priv_dim == 187 + 6
    b = env.reset_all()
    assert b["critic_obs"].shape[-1] == env.priv_dim

    env.resample_dr()                       # reset_all marked everything pending
    before = env.dr.mu.clone()
    env.reset_idx(torch.tensor([1, 3], device="cuda:0"))
    env.resample_dr()
    after = env.dr.mu
    unchanged = torch.tensor([0, 2])
    assert torch.equal(after[unchanged].cpu(), before[unchanged].cpu()), \
        "non-reset envs must keep their draw bitwise"
    assert not torch.equal(after.cpu(), before.cpu()), \
        "reset envs should (a.s.) get new draws"

    # eval-style env: DR off, priv_dim unchanged from config width
    cfg_eval = EnvConfig(num_envs=2, keypoint_links=list(KEYPOINT_LINKS),
                         noise_level=0.0, recovery_fraction=0.0,
                         dr=dict(friction_range=[0.4, 1.0], privileged=True))
    env_eval = TrackEnv(cfg_eval, corpus, device="cuda:0", train=False)
    assert env_eval.dr is None
    be = env_eval.reset_all()
    assert be["critic_obs"].shape[-1] == 187 + 6   # zeros appended, width stable

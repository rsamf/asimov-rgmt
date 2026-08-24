"""Drift-feedback observations (EnvConfig.drift_obs).

The drift features append [dp_body(3), cos/sin heading err(2)] per command
frame (cmd_dim 55 -> 60) so planar/yaw drift — unobservable under the paper's
yaw-invariant g_t — becomes closed-loop. These tests pin the shape contract,
the reset-time semantics (robot spawned ON the reference -> zero drift at the
center frame, window offsets equal to reference root displacement), and that
the default-off path keeps the original 55/187 contract.
"""
import pytest, torch
_HAS = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not _HAS, reason="needs CUDA")
from rgmt.env.track_env import TrackEnv, EnvConfig
from rgmt.data.motion import MotionRef
from rgmt.data.corpus import MotionCorpus
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from tests.fixtures.make_synthetic_motion import make_synthetic_motion

DEV = "cuda:0"


def _make_env(tmp_path, *, drift=True, train=False, n_envs=4, noise=None):
    p = make_synthetic_motion(tmp_path / "m.npz", n_frames=120)
    m = MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device=DEV, keypoint_links=KEYPOINT_LINKS)
    corpus = MotionCorpus.from_clips([m], ["clip0"], KEYPOINT_LINKS, output_device=DEV)
    kwargs = dict(num_envs=n_envs, keypoint_links=KEYPOINT_LINKS,
                  drift_obs=drift, recovery_fraction=0.0)
    if noise is not None:
        kwargs["noise"] = noise
    cfg = EnvConfig(**kwargs)
    return TrackEnv(cfg, corpus, device=DEV, train=train)


def test_shapes_and_finite_noisy(tmp_path):
    # legacy noise dict WITHOUT the drift p/yaw keys must still work (merge over defaults)
    legacy_noise = {"v_xy": 0.5, "v_z": 0.2, "w": 0.52, "g": 0.05, "q": 0.1}
    env = _make_env(tmp_path, drift=True, train=True, noise=legacy_noise)
    assert env.cmd_dim == 60 and env.priv_dim == 98 + 60 + (1 + 3 * env.Kp + 3)
    b = env.reset_all()
    assert b["cmd_window"].shape == (4, 21, 60)
    assert b["critic_obs"].shape == (4, env.priv_dim)
    b2, rew, done, info = env.step(torch.zeros(4, 23, device=DEV))
    for k, v in b2.items():
        assert torch.isfinite(v).all(), f"non-finite in bundle[{k}]"


def test_drift_semantics_at_reset(tmp_path):
    env = _make_env(tmp_path, drift=True, train=False)  # clean path, RSI = ref frame
    b = env.reset_all()
    feats = b["cmd_window"][..., 55:60]                  # (N, 21, 5)
    L = env.L
    # Center frame: robot sits exactly on the reference -> zero displacement,
    # zero heading error (fixture yaw is identity).
    center = feats[:, L, :]
    assert torch.allclose(center[:, 0:3], torch.zeros_like(center[:, 0:3]), atol=1e-4)
    assert torch.allclose(center[:, 3], torch.ones_like(center[:, 3]), atol=1e-4)   # cos
    assert torch.allclose(center[:, 4], torch.zeros_like(center[:, 4]), atol=1e-4)  # sin
    # Window offsets: with robot == ref[idx] and identity yaw, dp at offset k
    # must equal the reference root displacement ref[idx+k] - ref[idx] (clamped
    # to the clip). The fixture translates along +x only.
    store = env.motion.base_pos.device                   # corpus storage may be CPU
    idx = env.idx.to(store)                              # (N,)
    offs = torch.arange(-L, L + 1, device=store)
    grid = (idx.unsqueeze(1) + offs.unsqueeze(0)).clamp(0, env.motion.n_frames - 1)
    expected = (env.motion.base_pos[grid.reshape(-1)].reshape(4, 21, 3)
                - env.motion.base_pos[idx][:, None, :]).to(feats.device)
    assert torch.allclose(feats[..., 0:3], expected, atol=1e-4)
    # heading error stays identity across the whole window
    assert torch.allclose(feats[..., 3], torch.ones_like(feats[..., 3]), atol=1e-4)


def test_critic_obs_carries_drift(tmp_path):
    env = _make_env(tmp_path, drift=True, train=False)
    b = env.reset_all()
    # critic layout: [o_t(98), g_t_clean(60), h_ref(1), kp_rel(3Kp), v_base(3)]
    drift_slice = b["critic_obs"][:, 98 + 55: 98 + 60]
    assert torch.allclose(drift_slice[:, 0:3], torch.zeros_like(drift_slice[:, 0:3]), atol=1e-4)
    assert torch.allclose(drift_slice[:, 3], torch.ones_like(drift_slice[:, 3]), atol=1e-4)


def test_default_off_keeps_contract(tmp_path):
    env = _make_env(tmp_path, drift=False, train=False)
    assert env.cmd_dim == 55 and env.priv_dim == 98 + 55 + (1 + 3 * env.Kp + 3)
    b = env.reset_all()
    assert b["cmd_window"].shape == (4, 21, 55)


def test_policy_forward_with_drift(tmp_path):
    from rgmt.policy.networks import RGMTActorCritic, PolicyDims
    env = _make_env(tmp_path, drift=True, train=True)
    dims = PolicyDims(priv_dim=env.priv_dim, cmd_dim=env.cmd_dim)
    model = RGMTActorCritic(dims).to(DEV)
    b = env.reset_all()
    a, logp, v = model.act(b)
    assert a.shape == (4, 23) and logp.shape == (4,) and v.shape == (4,)


def test_proprio_drift_and_done_causes(tmp_path):
    from rgmt.policy.networks import RGMTActorCritic, PolicyDims
    p = make_synthetic_motion(tmp_path / "m.npz", n_frames=120)
    m = MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device=DEV, keypoint_links=KEYPOINT_LINKS)
    corpus = MotionCorpus.from_clips([m], ["clip0"], KEYPOINT_LINKS, output_device=DEV)
    cfg = EnvConfig(num_envs=4, keypoint_links=KEYPOINT_LINKS, drift_obs=True,
                    drift_obs_proprio=True, recovery_fraction=0.0)
    env = TrackEnv(cfg, corpus, device=DEV, train=False)
    assert env.obs_dim == 103 and env.priv_dim == 103 + 60 + (1 + 3 * env.Kp + 3)
    b = env.reset_all()
    assert b["obs"].shape == (4, 103) and b["history"].shape == (4, 10, 103)
    # at RSI reset the proprio drift block is [0,0,0,1,0]
    assert torch.allclose(b["obs"][:, 98:101], torch.zeros(4, 3, device=DEV), atol=1e-4)
    assert torch.allclose(b["obs"][:, 101], torch.ones(4, device=DEV), atol=1e-4)
    dims = PolicyDims(priv_dim=env.priv_dim, obs_dim=env.obs_dim, cmd_dim=env.cmd_dim)
    model = RGMTActorCritic(dims).to(DEV)
    a, logp, v = model.act(b)
    b2, rew, done, info = env.step(a)
    assert b2["obs"].shape == (4, 103)
    dc = info["done_causes"]
    assert set(dc) == {"fallen", "tracking", "tracking_zdev", "tracking_rootxy",
                       "tracking_joint", "motion_end", "timeout"}
    assert all(0.0 <= x <= 1.0 for x in dc.values())
    # the split partitions the tracking bucket
    assert abs(dc["tracking"] - (dc["tracking_zdev"] + dc["tracking_rootxy"]
                                 + dc["tracking_joint"])) < 1e-6


def test_push_perturbation(tmp_path):
    p = make_synthetic_motion(tmp_path / "m.npz", n_frames=120)
    m = MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device=DEV, keypoint_links=KEYPOINT_LINKS)
    corpus = MotionCorpus.from_clips([m], ["clip0"], KEYPOINT_LINKS, output_device=DEV)
    cfg = EnvConfig(num_envs=8, keypoint_links=KEYPOINT_LINKS, recovery_fraction=0.0,
                    push_force_max=100.0, push_interval_s=0.1, push_duration_s=0.05,
                    noise_level=0.0)
    env = TrackEnv(cfg, corpus, device=DEV, train=True)
    env.reset_all()
    saw_push = False
    for _ in range(20):
        b, r, d, info = env.step(torch.zeros(8, 23, device=DEV))
        f = env.sim._external_force
        if float(f[:, :2].abs().max()) > 0:
            saw_push = True
        assert float(f[:, 2].abs().max()) == 0.0  # horizontal only (assist off)
        assert torch.isfinite(b["obs"]).all()
    assert saw_push, "no push ever became active with 0.1s interval over 20 steps"
    # eval env (train=False) must never push
    env2 = TrackEnv(cfg, corpus, device=DEV, train=False)
    env2.reset_all()
    for _ in range(10):
        env2.step(torch.zeros(8, 23, device=DEV))
    assert float(env2.sim._external_force.abs().max()) == 0.0


def test_bundle_history_is_a_snapshot(tmp_path):
    """Regression: bundle['history'] must NOT alias the env ring buffer.

    The training loop stores the bundle AFTER env.step() mutates history in
    place; an aliased tensor silently becomes the next step's (or next
    episode's) history, corrupting PPO's logp re-evaluation.
    """
    env = _make_env(tmp_path, drift=False, train=True)
    b = env.reset_all()
    h_before = b["history"].clone()
    env.step(torch.zeros(4, env.n_act, device=DEV))
    assert torch.equal(b["history"], h_before), (
        "bundle['history'] mutated across env.step() — it aliases the ring buffer"
    )

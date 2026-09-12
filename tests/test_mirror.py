"""Left-right (sagittal, y -> -y) mirror augmentation: pure math + corpus tests.

CPU-only. The physics ground truth (Newton FK on the MJCF) lives in
tests/test_mirror_sim.py.
"""
import xml.etree.ElementTree as ET

import pytest
import torch

from rgmt.assets.paths import ROBOT_URDF, ROBOT_XML
from rgmt.data.joint_map import ASIMOV_ACTUATED_JOINT_NAMES, KEYPOINT_LINKS
from rgmt.data import mirror as mr
from rgmt.utils.rotation import (
    encode_angles, gravity_projection, quat_rotate_inverse, quat_to_matrix)

M = torch.diag(torch.tensor([1.0, -1.0, 1.0]))


# ---------------------------------------------------------------- (a) perms
def test_joint_perm_is_involution_and_swaps_left_right_by_name():
    perm = mr.MIRROR_JOINT_PERM
    assert perm.shape == (23,)
    assert torch.equal(perm[perm], torch.arange(23))
    for i, name in enumerate(ASIMOV_ACTUATED_JOINT_NAMES):
        assert ASIMOV_ACTUATED_JOINT_NAMES[int(perm[i])] == mr.mirror_name(name)
    fixed = [ASIMOV_ACTUATED_JOINT_NAMES[i] for i in range(23) if int(perm[i]) == i]
    assert fixed == ["waist_yaw_joint"]


def test_keypoint_perm_is_involution_with_midline_fixed_points():
    perm = mr.MIRROR_KP_PERM
    assert perm.shape == (len(KEYPOINT_LINKS),)
    assert torch.equal(perm[perm], torch.arange(len(KEYPOINT_LINKS)))
    for i, name in enumerate(KEYPOINT_LINKS):
        assert KEYPOINT_LINKS[int(perm[i])] == mr.mirror_name(name)
    fixed = sorted(KEYPOINT_LINKS[i] for i in range(len(KEYPOINT_LINKS)) if int(perm[i]) == i)
    assert fixed == ["neck_pitch_link", "pelvis_link"]


def test_mirror_name_swaps_prefix_only():
    assert mr.mirror_name("left_hip_roll_joint") == "right_hip_roll_joint"
    assert mr.mirror_name("right_wrist_yaw_link") == "left_wrist_yaw_link"
    assert mr.mirror_name("waist_yaw_joint") == "waist_yaw_joint"


def test_perm_from_names_raises_on_missing_partner():
    with pytest.raises(ValueError):
        mr._perm_from_names(["left_a", "left_b", "right_a"])


# ------------------------------------------------ (b) asset justifies the sign
def _vec(s, default):
    return torch.tensor([float(x) for x in (s or default).split()])


def _mjcf_joints():
    root = ET.parse(ROBOT_XML).getroot()
    out = {}
    for j in root.iter("joint"):
        if j.get("class") != "motor":
            continue
        out[j.get("name")] = dict(
            axis=_vec(j.get("axis"), "0 0 1"),
            range=_vec(j.get("range"), "0 0"),
            ref=float(j.get("ref") or 0.0),
        )
    return out


def _urdf_axes():
    root = ET.parse(ROBOT_URDF).getroot()
    out = {}
    for j in root.iter("joint"):
        ax = j.find("axis")
        lim = j.find("limit")
        if ax is None:
            continue
        out[j.get("name")] = dict(
            axis=_vec(ax.get("xyz"), "1 0 0"),
            range=torch.tensor([float(lim.get("lower")), float(lim.get("upper"))]),
        )
    return out


def test_mjcf_axes_ranges_refs_are_mirror_consistent_and_justify_sign():
    """For a hinge about axis a, the y-reflection of R(a, q) is R(M a, -q).
    So if the partner joint's axis is exactly M a, the mirrored angle is -q
    (sign -1); if it is -M a, the angle is preserved (sign +1)."""
    J = _mjcf_joints()
    names = ASIMOV_ACTUATED_JOINT_NAMES
    expected_sign = torch.zeros(23)
    for i, n in enumerate(names):
        p = names[int(mr.MIRROR_JOINT_PERM[i])]
        a_i, a_p = J[n]["axis"], J[p]["axis"]
        if torch.allclose(a_p, M @ a_i, atol=1e-6):
            expected_sign[i] = -1.0
        elif torch.allclose(a_p, -(M @ a_i), atol=1e-6):
            expected_sign[i] = 1.0
        else:
            raise AssertionError(f"{n}/{p}: axes {a_i.tolist()} / {a_p.tolist()} not mirror images")
        s = expected_sign[i]
        lo, hi = J[n]["range"].tolist()
        lo_p, hi_p = J[p]["range"].tolist()
        exp_range = (s * hi, s * lo) if s < 0 else (lo, hi)
        assert (lo_p, hi_p) == pytest.approx(exp_range, abs=1e-6), n
        assert J[p]["ref"] == pytest.approx(s * J[n]["ref"], abs=1e-6), n
    assert torch.equal(expected_sign, mr.MIRROR_JOINT_SIGN)


def test_urdf_axes_agree_with_mjcf_on_sign():
    """MotionRef FK runs on the URDF, the sim on the MJCF: both must agree."""
    U = _urdf_axes()
    names = ASIMOV_ACTUATED_JOINT_NAMES
    for i, n in enumerate(names):
        p = names[int(mr.MIRROR_JOINT_PERM[i])]
        s = float(mr.MIRROR_JOINT_SIGN[i])
        assert torch.allclose(U[p]["axis"], -s * (M @ U[n]["axis"]), atol=1e-6), n
        lo, hi = U[n]["range"].tolist()
        exp = (-hi, -lo) if s < 0 else (lo, hi)
        assert U[p]["range"].tolist() == pytest.approx(exp, abs=1e-6), n


# ------------------------------------------------ (c)/(d) vector functions
def _rand_frames(n=64, kp=len(KEYPOINT_LINKS), gen=None):
    g = gen or torch.Generator().manual_seed(0)
    q = torch.randn(n, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    return {
        "base_pos": torch.randn(n, 3, generator=g),
        "base_quat": q,
        "joint_q": torch.randn(n, 23, generator=g),
        "joint_qd": torch.randn(n, 23, generator=g),
        "base_lin_vel": torch.randn(n, 3, generator=g),
        "base_ang_vel": torch.randn(n, 3, generator=g),
        "kp_pos_world": torch.randn(n, kp, 3, generator=g),
        "kp_vel_world": torch.randn(n, kp, 3, generator=g),
    }


def test_mirror_frames_is_an_involution():
    f = _rand_frames()
    ff = mr.mirror_frames(mr.mirror_frames(f))
    assert set(ff) == set(f)
    for k in f:
        assert torch.allclose(ff[k], f[k], atol=1e-6), k


def test_mirror_frames_changes_every_tensor():
    f = _rand_frames()
    m = mr.mirror_frames(f)
    for k in f:
        assert not torch.allclose(m[k], f[k]), k


def test_mirror_quat_equals_matrix_conjugation():
    g = torch.Generator().manual_seed(1)
    q = torch.randn(1000, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    R = quat_to_matrix(q)
    Rm = quat_to_matrix(mr.mirror_quat(q))
    assert torch.allclose(Rm, M @ R @ M, atol=1e-6)


def test_mirror_joints_permutes_and_negates():
    q = torch.arange(23.0)
    m = mr.mirror_joints(q)
    assert torch.equal(m, mr.MIRROR_JOINT_SIGN * q[mr.MIRROR_JOINT_PERM])
    assert torch.allclose(mr.mirror_joints(m), q)


def test_mirror_keypoints_uses_link_list():
    kp = torch.randn(5, 10, 3)
    m = mr.mirror_keypoints(kp)
    assert torch.allclose(m[:, :, 1], -kp[:, mr.MIRROR_KP_PERM, 1])
    assert torch.allclose(m[:, :, 0], kp[:, mr.MIRROR_KP_PERM, 0])
    # explicit link list: 3 links, one pair + one midline
    kp3 = torch.randn(2, 3, 3)
    m3 = mr.mirror_keypoints(kp3, links=["left_x", "mid", "right_x"])
    assert torch.allclose(m3[:, 0, 0], kp3[:, 2, 0]) and torch.allclose(m3[:, 1, 1], -kp3[:, 1, 1])


def _command_of(f):
    """Reproduces MotionCorpus.command_at on a frames dict."""
    v = quat_rotate_inverse(f["base_quat"], f["base_lin_vel"])
    w = quat_rotate_inverse(f["base_quat"], f["base_ang_vel"])
    g = gravity_projection(f["base_quat"])
    return torch.cat([v, w, g, encode_angles(f["joint_q"])], dim=-1)


def test_mirror_command_commutes_with_mirror_frames():
    """Proves the body-frame v/w/g signs and the cos/sin permutation."""
    f = _rand_frames(256)
    lhs = mr.mirror_command(_command_of(f))
    rhs = _command_of(mr.mirror_frames(f))
    assert lhs.shape == (256, 55)
    assert torch.allclose(lhs, rhs, atol=1e-5)


# ------------------------------------------------------- (e) mirror_corpus
from rgmt.data.motion import MotionRef
from rgmt.data.corpus import MotionCorpus, TENSOR_KEYS
from tests.fixtures.make_synthetic_motion import make_synthetic_clips


def _corpus(tmp_path, output_device="cpu"):
    paths = make_synthetic_clips(tmp_path, [("a", 20), ("b", 35), ("c", 12)], asymmetric=True)
    clips = [MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device="cpu", keypoint_links=KEYPOINT_LINKS)
             for p in paths]
    return MotionCorpus.from_clips(clips, ["a", "b", "c"], KEYPOINT_LINKS, output_device=output_device)


def test_mirror_corpus_doubles_clips_with_consistent_bookkeeping(tmp_path):
    c = _corpus(tmp_path)
    m = mr.mirror_corpus(c)
    C, F = c.n_clips, c.n_frames
    assert m.n_clips == 2 * C and m.n_frames == 2 * F
    assert m.clip_names == ["a", "b", "c", "a__mirror", "b__mirror", "c__mirror"]
    assert torch.equal(m.clip_len, torch.cat([c.clip_len, c.clip_len]))
    assert torch.equal(m.clip_start[:C], c.clip_start)
    assert torch.equal(m.clip_start[C:], c.clip_start + F)
    assert torch.equal(m.clip_end[C:], c.clip_end + F)
    assert torch.equal(m.frame_clip_id[:F], c.frame_clip_id)
    assert torch.equal(m.frame_clip_id[F:], c.frame_clip_id + C)
    assert m._frame_sample_weights is None and m._valid_starts_cache == {}
    assert m.output_device == c.output_device and m.storage_device == c.storage_device
    assert m.keypoint_links == c.keypoint_links and m.actuated_names == c.actuated_names
    for k in TENSOR_KEYS:                      # originals are untouched, mirrored differ
        assert torch.equal(getattr(m, k)[:F], getattr(c, k))
        assert not torch.allclose(getattr(m, k)[F:], getattr(c, k))
    # the source corpus was not mutated
    assert c.n_clips == C and c.clip_names == ["a", "b", "c"]


def test_mirror_corpus_second_half_is_mirror_of_first(tmp_path):
    c = _corpus(tmp_path)
    m = mr.mirror_corpus(c)
    F = c.n_frames
    idx = torch.tensor([0, 7, 38, 39, 100, F - 1])
    orig = m.at(idx); mirr = m.at(idx + F)
    assert torch.allclose(mirr["base_pos"], mr.mirror_vec(orig["base_pos"]))
    assert torch.allclose(mirr["base_quat"], mr.mirror_quat(orig["base_quat"]))
    assert torch.allclose(mirr["base_lin_vel"], mr.mirror_vec(orig["base_lin_vel"]))
    assert torch.allclose(mirr["base_ang_vel"], mr.mirror_angvel(orig["base_ang_vel"]))
    assert torch.allclose(mirr["joint_q"], mr.mirror_joints(orig["joint_q"]))
    assert torch.allclose(mirr["joint_qd"], mr.mirror_joints(orig["joint_qd"]))
    p0, v0 = m.keypoints_at(idx); p1, v1 = m.keypoints_at(idx + F)
    assert torch.allclose(p1, mr.mirror_keypoints(p0)) and torch.allclose(v1, mr.mirror_keypoints(v0))
    # command windows (incl. clip-boundary clamping) are the mirrored commands
    w0 = m.command_window(idx, L=5); w1 = m.command_window(idx + F, L=5)
    assert torch.allclose(w1, mr.mirror_command(w0), atol=1e-5)
    assert torch.equal(m.clip_end_of(idx + F), m.clip_end_of(idx) + F)


def test_mirror_corpus_rejects_double_mirroring(tmp_path):
    m = mr.mirror_corpus(_corpus(tmp_path))
    with pytest.raises(ValueError, match="__mirror"):
        mr.mirror_corpus(m)


def test_mirror_corpus_sampling_weights_by_mirrored_name(tmp_path):
    m = mr.mirror_corpus(_corpus(tmp_path))
    m.set_clip_sampling_weights({"a__mirror": 0.0, "c": 0.0, "c__mirror": 0.0})
    idx = m.sample_index(2000, max_lookahead=2)
    cid = m.frame_clip_id[idx]
    banned = torch.tensor([m.clip_names.index(n) for n in ["a__mirror", "c", "c__mirror"]])
    assert not torch.isin(cid, banned).any()
    assert torch.isin(cid, torch.tensor([m.clip_names.index("b__mirror")])).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mirror_corpus_on_gpu_storage_matches_cpu(tmp_path):
    c = _corpus(tmp_path, output_device="cuda:0")
    m_cpu = mr.mirror_corpus(c)
    m_gpu = mr.mirror_corpus(c.to_storage("cuda:0"))
    assert m_gpu.storage_device.type == "cuda"
    for k in TENSOR_KEYS:
        assert torch.allclose(getattr(m_cpu, k).cuda(), getattr(m_gpu, k), atol=1e-6), k
    assert torch.equal(m_cpu.frame_clip_id.cuda(), m_gpu.frame_clip_id)
    s = m_gpu.sample_index(200, 4)
    assert torch.all(m_gpu.clip_end_of(s) - s >= 4)


# ---------------------------------------------------------- (g) mirror_split
def test_mirror_split_expands_train_only_and_preserves_other_keys():
    split = {"train": ["a", "b"], "test": ["t"], "difficulty": {"a": "easy"}, "n_train": 2}
    out = mr.mirror_split(split)
    assert out["train"] == ["a", "b", "a__mirror", "b__mirror"]
    assert out["test"] == ["t"]
    assert out["difficulty"] == {"a": "easy"} and out["n_train"] == 2
    assert split["train"] == ["a", "b"]          # input not mutated


# --------------------------------------------- (f-cpu) URDF FK ground truth
from tests.fixtures.make_synthetic_motion import make_synthetic_motion, mirror_synthetic_npz


def test_mirrored_npz_through_urdf_fk_matches_mirrored_tensors(tmp_path):
    """Independent oracle: write the mirrored .npz straight from the math,
    run the real loader (slerp, finite-difference velocities, yourdfpy FK) on
    both, and check the mirror functions reproduce it. The 1e-3 m keypoint
    tolerance covers the model's <0.6 mm right-side asymmetries."""
    a = make_synthetic_motion(tmp_path / "a.npz", n_frames=40, asymmetric=True)
    b = mirror_synthetic_npz(a, tmp_path / "b.npz")
    A = MotionRef.load(a, ROBOT_XML, ROBOT_URDF, device="cpu", keypoint_links=KEYPOINT_LINKS, ground=False)
    B = MotionRef.load(b, ROBOT_XML, ROBOT_URDF, device="cpu", keypoint_links=KEYPOINT_LINKS, ground=False)
    assert torch.allclose(B.base_pos, mr.mirror_vec(A.base_pos), atol=1e-6)
    RA, RB = quat_to_matrix(A.base_quat), quat_to_matrix(B.base_quat)
    assert torch.allclose(RB, quat_to_matrix(mr.mirror_quat(A.base_quat)), atol=1e-5)
    assert torch.allclose(B.joint_q, mr.mirror_joints(A.joint_q), atol=1e-6)
    assert torch.allclose(B.joint_qd, mr.mirror_joints(A.joint_qd), atol=1e-4)
    assert torch.allclose(B.base_lin_vel, mr.mirror_vec(A.base_lin_vel), atol=1e-4)
    assert torch.allclose(B.base_ang_vel, mr.mirror_angvel(A.base_ang_vel), atol=1e-4)
    assert A.base_ang_vel.abs().max() > 0.1          # the oracle actually exercises the sign
    assert torch.allclose(B._kp_pos_world, mr.mirror_keypoints(A._kp_pos_world), atol=1e-3)
    assert torch.allclose(B._kp_vel_world, mr.mirror_keypoints(A._kp_vel_world), atol=5e-2)

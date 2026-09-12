"""Left-right mirroring of reference motion ("RandomHorizontalFlip" for animation).

Reflects a clip across the robot's sagittal plane, the world x-z plane
(``M = diag(1, -1, 1)``). The Asimov v1 is bilaterally symmetric to well under
a millimetre (docs/reports/asimov-model-asymmetries-2026-07-13.md), so the
mirror image of every clip is a physically valid reference. Used as a
train-time data augmentation by duplicating the corpus (``mirror_corpus``):
frame-uniform RSI over the doubled corpus mirrors each episode with p = 0.5.

Transforms (all pure sign patterns, no FK):

    position, linear velocity, keypoints   (x, y, z)     -> (x, -y, z)
    quaternion xyzw                        (x, y, z, w)  -> (-x, y, -z, w)   [= M R M]
    angular velocity (pseudovector)        (wx, wy, wz)  -> (-wx, wy, -wz)
    joints                                 q'[i] = SIGN[i] * q[PERM[i]]
    keypoints                              permute left/right links, then (x, -y, z)

Everything here is device-agnostic torch.
"""
from typing import Dict, List, Optional

import torch

from rgmt.data.joint_map import ASIMOV_ACTUATED_JOINT_NAMES, KEYPOINT_LINKS

MIRROR_SUFFIX = "__mirror"


def mirror_name(name: str) -> str:
    """Swap a ``left_``/``right_`` prefix; midline names map to themselves."""
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    return name


def _perm_from_names(names: List[str]) -> torch.LongTensor:
    """perm[i] = index of names[i]'s mirror partner. Raises if a partner is
    missing; the result is an involution by construction."""
    index = {n: i for i, n in enumerate(names)}
    perm = []
    for n in names:
        p = mirror_name(n)
        if p not in index:
            raise ValueError(f"mirror: no partner '{p}' for '{n}' in {names}")
        perm.append(index[p])
    perm_t = torch.tensor(perm, dtype=torch.long)
    assert torch.equal(perm_t[perm_t], torch.arange(len(names)))
    return perm_t


MIRROR_JOINT_PERM: torch.LongTensor = _perm_from_names(ASIMOV_ACTUATED_JOINT_NAMES)
MIRROR_KP_PERM: torch.LongTensor = _perm_from_names(KEYPOINT_LINKS)

# Per-joint sign of the mirrored angle. A hinge R(a, q) reflected by M becomes
# R(M a, -q). In this asset every right-side axis is exactly the y-reflection
# of its left twin (right pitch axes are `0 -1 0` vs left `0 1 0`; roll, yaw
# and the oblique wrist axes are identical on both sides and have zero y
# component), every midline axis (waist_yaw `0 0 1`) has zero y component,
# joint ranges are negated-and-swapped, and the elbow `ref` offsets are
# opposite (+/-0.785398). Hence the mirrored angle is -q for all 23 joints.
# An asset whose right-side pitch axes were NOT negated would need +1 on those
# pairs; tests/test_mirror.py derives this vector from the MJCF and URDF.
MIRROR_JOINT_SIGN: torch.Tensor = -torch.ones(len(ASIMOV_ACTUATED_JOINT_NAMES))


def mirror_vec(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) polar vector: (x, y, z) -> (x, -y, z)."""
    out = v.clone()
    out[..., 1] = -out[..., 1]
    return out


def mirror_angvel(w: torch.Tensor) -> torch.Tensor:
    """(..., 3) pseudovector: (x, y, z) -> (-x, y, -z)."""
    out = w.clone()
    out[..., 0] = -out[..., 0]
    out[..., 2] = -out[..., 2]
    return out


def mirror_quat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) xyzw: (x, y, z, w) -> (-x, y, -z, w), i.e. R -> M R M."""
    out = q.clone()
    out[..., 0] = -out[..., 0]
    out[..., 2] = -out[..., 2]
    return out


def mirror_joints(q: torch.Tensor) -> torch.Tensor:
    """(..., 23) joint angles or velocities in ASIMOV_ACTUATED_JOINT_NAMES order."""
    perm = MIRROR_JOINT_PERM.to(q.device)
    sign = MIRROR_JOINT_SIGN.to(q.device, q.dtype)
    return sign * q[..., perm]


def mirror_keypoints(kp: torch.Tensor, links: Optional[List[str]] = None) -> torch.Tensor:
    """(..., Kp, 3) keypoint positions or velocities; ``links`` names the Kp
    axis (default KEYPOINT_LINKS)."""
    perm = MIRROR_KP_PERM if links is None else _perm_from_names(list(links))
    return mirror_vec(kp[..., perm.to(kp.device), :])


def mirror_frames(frames: Dict[str, torch.Tensor],
                  keypoint_links: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    """Mirror a dict of per-frame corpus tensors (the MotionCorpus TENSOR_KEYS)."""
    fns = {
        "base_pos": mirror_vec,
        "base_quat": mirror_quat,
        "joint_q": mirror_joints,
        "joint_qd": mirror_joints,
        "base_lin_vel": mirror_vec,
        "base_ang_vel": mirror_angvel,
        "kp_pos_world": lambda t: mirror_keypoints(t, keypoint_links),
        "kp_vel_world": lambda t: mirror_keypoints(t, keypoint_links),
    }
    unknown = set(frames) - set(fns)
    if unknown:
        raise KeyError(f"mirror_frames: unknown keys {sorted(unknown)}")
    return {k: fns[k](v) for k, v in frames.items()}


def mirror_command(cmd: torch.Tensor) -> torch.Tensor:
    """(..., 55) MotionCorpus command [v(3), w(3), g(3), cos q(23), sin q(23)],
    all in the reference root frame. Body-frame vectors mirror like their
    world-frame originals (R -> M R M commutes with the reflection)."""
    n = MIRROR_JOINT_PERM.numel()
    perm = MIRROR_JOINT_PERM.to(cmd.device)
    sign = MIRROR_JOINT_SIGN.to(cmd.device, cmd.dtype)
    v = mirror_vec(cmd[..., 0:3])
    w = mirror_angvel(cmd[..., 3:6])
    g = mirror_vec(cmd[..., 6:9])
    cos = cmd[..., 9:9 + n][..., perm]                 # cos(s q) = cos(q)
    sin = sign * cmd[..., 9 + n:9 + 2 * n][..., perm]  # sin(s q) = s sin(q)
    return torch.cat([v, w, g, cos, sin], dim=-1)


# ------------------------------------------------------------ corpus / split
def mirror_split(split: dict, suffix: str = MIRROR_SUFFIX) -> dict:
    """Return a copy of a train/test split whose ``train`` list also carries
    the mirrored name of every train clip. ``test`` (and every other key) is
    untouched: mirrored copies of test clips are therefore in neither role and
    get RSI weight 0, so the held-out numbers stay comparable and leak-free."""
    out = dict(split)
    out["train"] = list(split["train"]) + [n + suffix for n in split["train"]]
    return out


def mirror_corpus(corpus, suffix: str = MIRROR_SUFFIX):
    """Return a NEW MotionCorpus holding ``corpus``'s clips followed by their
    left-right mirrors, named ``<name><suffix>``. Same storage/output device,
    no sampling weights, fresh valid-starts cache. Pure tensor ops (no FK):
    exact up to the sub-millimetre asymmetries of the model.

    Frame-uniform RSI over the result mirrors each episode with p = 0.5. Not
    saveable with ``save_cache`` (mirrored clips have no source file); build it
    in memory at training-corpus load time.
    """
    from rgmt.data.corpus import MotionCorpus, TENSOR_KEYS, clip_bookkeeping

    already = [n for n in corpus.clip_names if n.endswith(suffix)]
    if already:
        raise ValueError(
            f"mirror_corpus: corpus already contains {len(already)} '{suffix}' clips "
            f"(e.g. {already[:2]}); refusing to mirror twice")
    orig = {k: getattr(corpus, k) for k in TENSOR_KEYS}
    mirr = mirror_frames(orig, keypoint_links=corpus.keypoint_links)
    cat = {k: torch.cat([orig[k], mirr[k]], dim=0) for k in TENSOR_KEYS}
    lens = torch.cat([corpus.clip_len, corpus.clip_len])
    starts, ends, frame_clip_id = clip_bookkeeping(lens)
    return MotionCorpus(
        physics_fps=corpus.physics_fps, src_fps=corpus.src_fps,
        actuated_idx=corpus.actuated_idx, actuated_names=list(corpus.actuated_names),
        keypoint_links=list(corpus.keypoint_links), output_device=corpus.output_device,
        clip_start=starts, clip_len=lens, clip_end=ends, frame_clip_id=frame_clip_id,
        clip_names=list(corpus.clip_names) + [n + suffix for n in corpus.clip_names],
        **cat,
    )

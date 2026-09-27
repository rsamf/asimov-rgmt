import pytest
import torch
from rgmt.data.corpus import MotionCorpus, TENSOR_KEYS, clip_bookkeeping
from locomotion.config import FeaturesConfig
from locomotion.features import FeatureDB
from locomotion.geom import quat_from_yaw, quat_mul, rot3z
from tests.fixtures.loco_corpus import make_loco_corpus


def _db(seconds=2.0, **kw):
    corpus, _ = make_loco_corpus(seconds=seconds)
    return corpus, FeatureDB.build(corpus, FeaturesConfig(**kw), device="cpu")


def test_shapes_and_valid_mask():
    corpus, db = _db()
    H = 3
    assert db.pose_dim == 15 and db.dim == 15 + 4 * H
    assert db.raw.shape == (corpus.n_frames, db.dim)
    per_clip_valid = int(corpus.clip_len[0]) - 60
    assert db.valid_idx.numel() == per_clip_valid * corpus.n_clips
    end0 = int(corpus.clip_end[0])
    assert db.is_valid(end0 - 60) and not db.is_valid(end0 - 59)
    assert db.feats.shape == (db.valid_idx.numel(), db.dim)


def test_query_recovers_own_frame():
    corpus, db = _db()
    i = int(corpus.clip_start[0]) + 30          # mid walk_fwd
    j = db.query(db.raw[i, :15], db.raw[i, 15:])
    assert torch.allclose(db.raw[j], db.raw[i], atol=1e-5)


def test_stay_bias_breaks_ties_toward_current():
    corpus, db = _db()
    idle0 = int(corpus.clip_start[corpus.clip_names.index("idle")])
    q = db.raw[idle0 + 10]
    j = db.query(q[:15], q[15:], current_next=idle0 + 20, stay_bias=0.5)
    assert j == idle0 + 20                      # idle frames tie; bias picks the continuing one


def _transform_corpus(corpus, yaw, offset):
    R = rot3z(yaw)
    qz = quat_from_yaw(torch.tensor(yaw))
    off = torch.tensor(offset, dtype=torch.float32)
    d = {k: getattr(corpus, k).clone() for k in TENSOR_KEYS}
    d["base_pos"] = d["base_pos"] @ R.T + off
    d["base_quat"] = quat_mul(qz.expand_as(d["base_quat"]), d["base_quat"])
    d["base_lin_vel"] = d["base_lin_vel"] @ R.T
    d["base_ang_vel"] = d["base_ang_vel"] @ R.T
    d["kp_pos_world"] = d["kp_pos_world"] @ R.T + off
    d["kp_vel_world"] = d["kp_vel_world"] @ R.T
    starts, ends, fcid = clip_bookkeeping(corpus.clip_len)
    return MotionCorpus(physics_fps=corpus.physics_fps, src_fps=corpus.src_fps,
                        actuated_idx=corpus.actuated_idx, actuated_names=corpus.actuated_names,
                        keypoint_links=corpus.keypoint_links, output_device=corpus.output_device,
                        clip_start=starts, clip_len=corpus.clip_len, clip_end=ends,
                        frame_clip_id=fcid, clip_names=list(corpus.clip_names), **d)


def test_features_invariant_to_world_rotation_and_translation():
    corpus, db = _db()
    moved = _transform_corpus(corpus, 1.1, (3.0, -2.0, 0.0))
    db2 = FeatureDB.build(moved, FeaturesConfig(), device="cpu")
    assert torch.allclose(db.raw[db.valid_idx], db2.raw[db2.valid_idx], atol=1e-4)


def test_walk_y_matches_walk_fwd_features():
    corpus, db = _db()
    a = int(corpus.clip_start[corpus.clip_names.index("walk_fwd")]) + 20
    b = int(corpus.clip_start[corpus.clip_names.index("walk_y")]) + 20
    assert torch.allclose(db.raw[a], db.raw[b], atol=1e-3)


def test_normalisation_is_per_block_and_weighted():
    _, db = _db(traj_dir=2.0, foot_pos=0.5)
    assert torch.allclose(db.scale[3:9], db.scale[3].expand(6))
    assert torch.allclose(db.scale[-6:], db.scale[-6].expand(6))
    assert torch.allclose(db.weight[-6:], torch.full((6,), 2.0))
    assert torch.allclose(db.weight[3:9], torch.full((6,), 0.5))
    z = db.normalize(db.mean)
    assert torch.allclose(z, torch.zeros_like(z), atol=1e-6)


def test_traj_raw_ordering_matches_columns():
    corpus, db = _db()
    i = int(corpus.clip_start[0]) + 10
    H = len(db.horizons)
    pos = db.raw[i, 15:15 + 2 * H].reshape(H, 2)
    d = db.raw[i, 15 + 2 * H:].reshape(H, 2)
    assert torch.allclose(FeatureDB.traj_raw(pos, d), db.raw[i, 15:])


def test_build_raises_when_no_frame_has_future():
    corpus, _ = make_loco_corpus(seconds=0.5)   # 30 frames < 60-frame horizon
    with pytest.raises(ValueError) as ei:
        FeatureDB.build(corpus, FeaturesConfig(), device="cpu")
    assert "60" in str(ei.value)

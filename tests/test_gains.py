"""Per-joint PD gain profile + the sim's scalar/vector expansion."""
import pytest

from rgmt.data.joint_map import ASIMOV_ACTUATED_JOINT_NAMES
from rgmt.env.gains import leg_weighted_gains
from rgmt.env.sim import _expand_gain

N = len(ASIMOV_ACTUATED_JOINT_NAMES)


def test_leg_weighted_shapes():
    kp, kd = leg_weighted_gains()
    assert len(kp) == N and len(kd) == N
    assert all(v > 0 for v in kp + kd)


def test_knees_ankles_are_stiffest_and_most_damped():
    kp, kd = leg_weighted_gains()
    g = dict(zip(ASIMOV_ACTUATED_JOINT_NAMES, zip(kp, kd)))

    def by(sub):
        return [g[n] for n in ASIMOV_ACTUATED_JOINT_NAMES if sub in n]

    knee_kd = [kd_ for _, kd_ in by("knee")]
    ankle_kd = [kd_ for _, kd_ in by("ankle")]
    arm_kd = [kd_ for n, (kp_, kd_) in g.items() for _ in [0]
              if any(s in n for s in ("shoulder", "elbow", "wrist"))]
    # legs damped harder than arms (the anti-jitter intent)
    assert min(knee_kd + ankle_kd) > max(arm_kd)
    # knees/ankles stiffer than arms too
    knee_ankle_kp = [kp_ for n, (kp_, _) in g.items() if "knee" in n or "ankle" in n]
    arm_kp = [kp_ for n, (kp_, _) in g.items()
              if any(s in n for s in ("shoulder", "elbow", "wrist"))]
    assert min(knee_ankle_kp) >= max(arm_kp)


def test_expand_gain_scalar_and_vector():
    assert _expand_gain(5.0, N, "kd") == [5.0] * N
    v = list(range(N))
    assert _expand_gain(v, N, "kp") == [float(x) for x in v]


def test_expand_gain_wrong_length_raises():
    with pytest.raises(ValueError):
        _expand_gain([1.0, 2.0], N, "kp")

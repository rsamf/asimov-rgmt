"""Per-joint PD gains — leg-weighted, for jitter reduction.

The uniform baseline was kp=100, kd=5 on all 23 actuated joints. Foot/shin
chatter concentrates in the distal leg, so this profile:
  - raises stiffness AND damping in the knees and ankles (the load-bearing,
    balance-critical, chatter-prone joints),
  - keeps the hips elevated and the waist moderate,
  - makes the arms more compliant (lower kp) with light damping — they are not
    balance-critical and high arm gains show up directly as arm jitter.

Ankle *pitch* gets the most damping (that is where foot-ground contact chatter
appears). Gains are classified by substring so they are robust to the
``_joint`` suffix / left-right naming.

`leg_weighted_gains()` returns (kp_list, kd_list) in ASIMOV_ACTUATED_JOINT_NAMES
order — feed straight into EnvConfig.kp / EnvConfig.kd (which now accept a
length-23 vector as well as a scalar).
"""
import xml.etree.ElementTree as ET

from rgmt.assets.paths import ROBOT_URDF
from rgmt.data.joint_map import ASIMOV_ACTUATED_JOINT_NAMES


def _gain_for(name: str) -> tuple[float, float]:
    """(kp, kd) for one actuated joint, by substring class."""
    n = name
    if "knee" in n:
        return (200.0, 10.0)
    if "ankle_pitch" in n:
        return (200.0, 12.0)
    if "ankle_roll" in n:
        return (150.0, 10.0)
    if "hip_pitch" in n or "hip_roll" in n:
        return (150.0, 5.0)
    if "hip_yaw" in n:
        return (120.0, 4.0)
    if "waist" in n:
        return (150.0, 6.0)
    if "shoulder_yaw" in n:
        return (80.0, 3.0)
    if "shoulder" in n:          # pitch / roll
        return (100.0, 3.0)
    if "elbow" in n:
        return (100.0, 3.0)
    if "wrist" in n:
        return (60.0, 2.0)
    raise ValueError(f"no gain rule for joint {name!r}")


def leg_weighted_gains() -> tuple[list[float], list[float]]:
    """(kp, kd) vectors in ASIMOV_ACTUATED_JOINT_NAMES order."""
    pairs = [_gain_for(n) for n in ASIMOV_ACTUATED_JOINT_NAMES]
    kp = [p[0] for p in pairs]
    kd = [p[1] for p in pairs]
    return kp, kd


def urdf_effort_limits() -> list[float]:
    """Per-joint torque limits (Nm) from the URDF, in ASIMOV order.

    The MJCF carries no <actuator> section, so Newton's importer leaves the
    effort limit at its 1e6 builder default, so the sim runs uncapped unless
    told otherwise. The URDF's <limit effort="..."> values are the
    motor datasheet numbers and are the deterministic base that DR's
    effort_scale_range multiplies. Raises if an actuated joint is missing or
    has a non-positive effort (a silent 0 would freeze the joint).
    """
    root = ET.parse(ROBOT_URDF).getroot()
    efforts: dict[str, float] = {}
    for j in root.findall("joint"):
        limit = j.find("limit")
        if limit is not None and "effort" in limit.attrib:
            efforts[j.attrib["name"]] = float(limit.attrib["effort"])
    out: list[float] = []
    for name in ASIMOV_ACTUATED_JOINT_NAMES:
        if name not in efforts:
            raise ValueError(f"urdf_effort_limits: no <limit effort> for {name!r}")
        e = efforts[name]
        if e <= 0.0:
            raise ValueError(f"urdf_effort_limits: non-positive effort {e} for {name!r}")
        out.append(e)
    return out


if __name__ == "__main__":
    kp, kd = leg_weighted_gains()
    eff = urdf_effort_limits()
    for n, a, b, e in zip(ASIMOV_ACTUATED_JOINT_NAMES, kp, kd, eff):
        print(f"  {n:26} kp={a:5.0f}  kd={b:5.1f}  effort={e:5.0f}")

import xml.etree.ElementTree as ET
from rgmt.data.joint_map import ASIMOV_ACTUATED_JOINT_NAMES, PASSIVE_JOINT_NAMES, KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_XML

def test_counts():
    assert len(ASIMOV_ACTUATED_JOINT_NAMES) == 23
    assert PASSIVE_JOINT_NAMES == ["neck_yaw_joint", "neck_pitch_joint"]

def test_names_match_mjcf_motor_joints_in_order():
    root = ET.parse(ROBOT_XML).getroot()
    motor = [j.get("name") for j in root.iter("joint") if j.get("class") == "motor"]
    assert motor == ASIMOV_ACTUATED_JOINT_NAMES

def test_keypoint_links_exist_in_mjcf():
    root = ET.parse(ROBOT_XML).getroot()
    bodies = {b.get("name") for b in root.iter("body")}
    for k in KEYPOINT_LINKS:
        assert k in bodies, k

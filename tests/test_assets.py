import xml.etree.ElementTree as ET
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF, MESH_DIR

def test_assets_exist():
    assert ROBOT_XML.is_file() and ROBOT_URDF.is_file() and MESH_DIR.is_dir()

def test_urdf_movable_joints_are_exactly_the_actuated_set():
    from rgmt.data.joint_map import ASIMOV_ACTUATED_JOINT_NAMES
    root = ET.parse(ROBOT_URDF).getroot()
    joints = [j for j in root.findall("joint") if j.get("type") != "fixed"]
    names = [j.get("name") for j in joints]
    assert sorted(names) == sorted(ASIMOV_ACTUATED_JOINT_NAMES), names

def test_urdf_meshes_all_exist():
    root = ET.parse(ROBOT_URDF).getroot()
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename").split("/")[-1]
        assert (MESH_DIR / fn).is_file(), fn

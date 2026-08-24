import xml.etree.ElementTree as ET
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF, MESH_DIR

def test_assets_exist():
    assert ROBOT_XML.is_file() and ROBOT_URDF.is_file() and MESH_DIR.is_dir()

def test_urdf_has_25_joints_no_toe():
    root = ET.parse(ROBOT_URDF).getroot()
    joints = [j for j in root.findall("joint") if j.get("type") != "fixed"]
    names = [j.get("name") for j in joints]
    assert len(names) == 25, names
    assert not any("toe" in n for n in names)

def test_urdf_meshes_all_exist():
    root = ET.parse(ROBOT_URDF).getroot()
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename").split("/")[-1]
        assert (MESH_DIR / fn).is_file(), fn

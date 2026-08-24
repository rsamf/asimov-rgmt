from pathlib import Path

_ASSET_ROOT = Path(__file__).resolve().parent / "asimov-v1"
ROBOT_XML = _ASSET_ROOT / "xmls" / "asimov.xml"
ROBOT_URDF = _ASSET_ROOT / "xmls" / "asimov.urdf"
MESH_DIR = _ASSET_ROOT / "assets" / "meshes"

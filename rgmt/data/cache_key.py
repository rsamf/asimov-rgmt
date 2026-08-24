"""Content/parameter hashing for the motion cache (fail-loud staleness)."""
import hashlib
import json
from pathlib import Path
from typing import Union

SCHEMA_VERSION = 3   # v2: clips ground-normalized at preprocess (see MotionRef.load)
                     # v3: `ground` is part of the cache key + manifest (otherwise
                     #     a ground_clips=false rebuild previously produced byte-
                     #     identical keys and passed every staleness check)

def file_sha256(path: Union[str, Path]) -> str:
    """SHA-256 of a file's bytes, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def params_key(*, source_hash: str, urdf_hash: str, physics_fps: int, src_fps: int,
               keypoint_links, schema_version: int, ground: bool = True) -> str:
    """Stable cache key over everything that affects a clip's processed tensors."""
    payload = json.dumps(
        {"source_hash": source_hash, "urdf_hash": urdf_hash,
         "physics_fps": physics_fps, "src_fps": src_fps,
         "keypoint_links": list(keypoint_links), "schema_version": schema_version,
         "ground": bool(ground)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

"""Offline motion preprocessing: clips -> per-clip safetensors cache + manifest."""
import json
from pathlib import Path
import hydra
import nebo as nb
from omegaconf import OmegaConf
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.data.motion import MotionRef
from rgmt.data.corpus import MotionCorpus, _write_clip_file, TENSOR_KEYS
from rgmt.data.cache_key import SCHEMA_VERSION, file_sha256, params_key

def run_preprocess(cfg) -> dict:
    cache_dir = Path(cfg.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    kp_links = list(cfg.keypoint_links) if cfg.get("keypoint_links") else list(KEYPOINT_LINKS)
    urdf_hash = file_sha256(ROBOT_URDF)
    if cfg.get("motion_dir"):
        npz_files = sorted(Path(cfg.motion_dir).glob("*.npz"))
    else:
        npz_files = [Path(cfg.motion_path)]
    if not npz_files:
        raise ValueError(f"preprocess: no .npz found under {cfg.get('motion_dir') or cfg.motion_path}")

    man_path = cache_dir / "manifest.json"
    existing = {}
    if man_path.exists() and not cfg.force:
        existing = {c["name"]: c for c in json.loads(man_path.read_text()).get("clips", [])}

    clips_meta, processed, skipped = [], 0, 0
    processed_ref = None
    nb.init()
    with nb.start_run(name="preprocess", config=OmegaConf.to_container(cfg, resolve=True)):
        for npz in nb.track(npz_files, name="clips"):
            name = npz.stem
            sh = file_sha256(npz)
            key = params_key(source_hash=sh, urdf_hash=urdf_hash, physics_fps=cfg.physics_fps,
                             src_fps=cfg.src_fps, keypoint_links=kp_links,
                             schema_version=SCHEMA_VERSION,
                             ground=cfg.get("ground_clips", True))
            prev = existing.get(name)
            if (not cfg.force and prev and prev["cache_key"] == key
                    and (cache_dir / prev["file"]).exists()):
                clips_meta.append(prev); skipped += 1
                continue
            ref = MotionRef.load(npz, ROBOT_XML, ROBOT_URDF, physics_fps=cfg.physics_fps,
                                 src_fps=cfg.src_fps, device="cpu", keypoint_links=kp_links,
                                 ground=cfg.get("ground_clips", True))
            tensors = {k: getattr(ref, "_kp_pos_world" if k == "kp_pos_world"
                                  else "_kp_vel_world" if k == "kp_vel_world" else k)
                       for k in TENSOR_KEYS}
            fn = _write_clip_file(cache_dir, name, tensors)
            clips_meta.append({"name": name, "file": fn, "n_frames": ref.n_frames,
                               "source_hash": sh, "cache_key": key})
            processed_ref = ref
            processed += 1
            nb.log_line("preprocess/processed", float(processed))

    # write merged manifest — source actuated metadata from the last processed clip,
    # the existing manifest (all-skip incremental run), or fall back to a fresh load.
    if processed_ref is not None:
        act_names = list(processed_ref.actuated_names)
        act_idx = processed_ref.actuated_idx.tolist()
    elif man_path.exists():
        _prev_man = json.loads(man_path.read_text())
        act_names = list(_prev_man["actuated_names"])
        act_idx = list(_prev_man["actuated_idx"])
    else:
        ref0 = MotionRef.load(npz_files[0], ROBOT_XML, ROBOT_URDF, physics_fps=cfg.physics_fps,
                              src_fps=cfg.src_fps, device="cpu", keypoint_links=kp_links)
        act_names = list(ref0.actuated_names)
        act_idx = ref0.actuated_idx.tolist()
    manifest = {"schema_version": SCHEMA_VERSION, "physics_fps": cfg.physics_fps,
                "src_fps": cfg.src_fps, "keypoint_links": kp_links, "urdf_hash": urdf_hash,
                "ground": bool(cfg.get("ground_clips", True)),
                "actuated_names": act_names,
                "actuated_idx": act_idx, "clips": clips_meta}
    man_path.write_text(json.dumps(manifest, indent=2))
    return {"processed": processed, "skipped": skipped, "n_clips": len(clips_meta)}

@hydra.main(version_base=None, config_path="configs", config_name="preprocess")
def main(cfg):
    print(run_preprocess(cfg))

if __name__ == "__main__":
    main()

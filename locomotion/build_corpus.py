"""Bake the locomotion corpus: resolve the frozen subset, symlink the npz
files, run rgmt.preprocess, measure nominal speeds, write clips.json.

    uv run python -m locomotion.build_corpus [--out DIR] [--subset JSON] [--dirs D1 D2] [--force]

Output layout (``--out`` defaults to corpus.cache_dir from the config):
    <out>/npz/<name>.npz    symlinks to the resolved sources
    <out>/cache/            MotionCorpus cache (manifest.json + safetensors)
    <out>/clips.json        {clips: {name: {class, source}}, speeds: {...}, meta: {...}}
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf

from rgmt.assets.paths import ROBOT_URDF
from rgmt.data.cache_key import file_sha256
from rgmt.data.corpus import MotionCorpus
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.preprocess import run_preprocess
from locomotion.config import LocoConfig, load_config
from locomotion.corpus import (DEFAULT_SEARCH_DIRS, DEFAULT_SUBSET_JSON, ClipsMeta, measure_speeds,
                               resolve_clips)


def build(out: Path, subset_json: Path, search_dirs: Sequence[Path], cfg: LocoConfig,
          force: bool = False) -> ClipsMeta:
    out = Path(out)
    exclude = {e.name: e.reason for e in cfg.corpus.exclude}
    entries = resolve_clips(subset_json, search_dirs, exclude)
    npz_dir = out / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    wanted = {f"{e.name}.npz" for e in entries}
    for stale in npz_dir.glob("*.npz"):
        if stale.name not in wanted and stale.is_symlink():      # only ever remove our own links
            stale.unlink()
    for e in entries:
        link = npz_dir / f"{e.name}.npz"
        if link.exists() and not link.is_symlink():
            raise FileExistsError(
                f"build_corpus: {link} is a regular file, not a symlink; refusing to overwrite it. "
                f"Move it away or choose another --out.")
        if link.is_symlink():
            link.unlink()
        link.symlink_to(e.path.resolve())

    pre = OmegaConf.create({"motion_dir": str(npz_dir), "motion_path": None, "cache_dir": str(out / "cache"),
                            "physics_fps": cfg.fps, "src_fps": cfg.src_fps, "keypoint_links": None,
                            "ground_clips": True, "force": bool(force)})
    print(run_preprocess(pre))

    corpus = MotionCorpus.load_cache(out / "cache", output_device="cpu", urdf_hash=file_sha256(ROBOT_URDF),
                                     physics_fps=cfg.fps, src_fps=cfg.src_fps, keypoint_links=KEYPOINT_LINKS)
    classes = {e.name: e.cls for e in entries}
    subset = json.loads(Path(subset_json).read_text())
    meta = ClipsMeta(classes=classes, sources={e.name: str(e.path) for e in entries},
                     speeds=measure_speeds(corpus, classes),
                     extra={"subset_json": str(subset_json), "rules_version": subset.get("rules_version"),
                            "search_dirs": [str(d) for d in search_dirs], "excluded": exclude,
                            "created": _dt.date.today().isoformat()})
    meta.save(out / "clips.json")
    counts = Counter(classes.values())
    print("clips per class:", dict(sorted(counts.items())), "total", len(classes))
    print("nominal speeds:", meta.speeds)
    return meta


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="output dir (default: corpus.cache_dir from the config)")
    p.add_argument("--subset", default=str(DEFAULT_SUBSET_JSON))
    p.add_argument("--dirs", nargs="+", default=[str(d) for d in DEFAULT_SEARCH_DIRS],
                   help="npz directories searched in order")
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[], help="dotlist overrides, e.g. corpus.mirror=false")
    p.add_argument("--force", action="store_true", help="re-bake every clip")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    build(Path(args.out or cfg.corpus.cache_dir), Path(args.subset), [Path(d) for d in args.dirs], cfg,
          force=args.force)


if __name__ == "__main__":
    main()

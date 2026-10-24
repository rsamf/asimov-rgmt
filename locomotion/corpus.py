"""Locomotion corpus: which clips, where they come from, and how to load them.

The frozen selection (asimov-gmr ``curation/locomotion_subset.json``) is the
contract: every name maps verbatim to ``$ROBOTICS_ROOT/train/<name>.npz``
(names contain spaces and hyphens and are never normalised). The corpus root
comes from ``corpus.robotics_root`` or the ``ROBOTICS_ROOT`` environment
variable; machine paths live in the gitignored ``configs/local.yaml``. ``ClipsMeta`` (``clips.json`` next to the cache) keeps
each clip's gait class and source plus the nominal speeds measured from the
corpus at build time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import torch

from rgmt.assets.paths import ROBOT_URDF
from rgmt.data.cache_key import file_sha256
from rgmt.data.corpus import MotionCorpus
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.data.mirror import MIRROR_SUFFIX, mirror_corpus
from locomotion.config import LocoConfig, require

SPEED_KEYS = ("forward", "backward", "lateral", "turn_rate")


@dataclass
class ClipEntry:
    name: str
    cls: str
    path: Path


def train_dir(robotics_root: Path) -> Path:
    """The corpus training directory: one ``<name>.npz`` per clip."""
    return Path(robotics_root) / "train"


def resolve_clips(subset_json: Path, robotics_root: Path,
                  exclude: Mapping[str, str]) -> List[ClipEntry]:
    """Map every name in the subset to ``<robotics_root>/train/<name>.npz``.
    Excluded names are skipped. Raises FileNotFoundError listing names with
    no npz (the subset guarantees there are none, so this means a wrong root
    or a stale subset)."""
    data = json.loads(Path(subset_json).read_text())
    d = train_dir(robotics_root)
    entries, missing = [], []
    for cls, names in data["clips"].items():
        for name in names:
            if name in exclude:
                continue
            p = d / f"{name}.npz"
            if p.exists():
                entries.append(ClipEntry(name=name, cls=cls, path=p))
            else:
                missing.append(name)
    if missing:
        raise FileNotFoundError(
            f"resolve_clips: {len(missing)} clip(s) have no npz in {d}: "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
    return entries


@dataclass
class ClipsMeta:
    classes: Dict[str, str]
    sources: Dict[str, str] = field(default_factory=dict)
    speeds: Dict[str, Optional[float]] = field(default_factory=dict)
    extra: Dict = field(default_factory=dict)

    def class_of(self, name: str) -> str:
        if name.endswith(MIRROR_SUFFIX):
            name = name[: -len(MIRROR_SUFFIX)]
        return self.classes.get(name, "unknown")

    def save(self, path: Path) -> None:
        clips = {n: {"class": c, "source": self.sources.get(n)} for n, c in self.classes.items()}
        Path(path).write_text(json.dumps(
            {"clips": clips, "speeds": self.speeds, "meta": self.extra}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "ClipsMeta":
        d = json.loads(Path(path).read_text())
        return cls(classes={n: v["class"] for n, v in d["clips"].items()},
                   sources={n: v.get("source") for n, v in d["clips"].items()},
                   speeds=dict(d.get("speeds", {})), extra=dict(d.get("meta", {})))


def measure_speeds(corpus: MotionCorpus, classes: Mapping[str, str],
                   min_speed: float = 0.1, min_rate: float = 0.1) -> Dict[str, Optional[float]]:
    """Median planar root speed of walk_fwd / walk_back / strafe frames and
    median |yaw rate| of turn frames, ignoring stand-still frames. None when a
    class has no moving frames."""
    cls_list = [classes.get(n, "unknown") for n in corpus.clip_names]
    names = sorted(set(cls_list))
    cls_id = torch.tensor([names.index(c) for c in cls_list])
    per_frame = torch.repeat_interleave(cls_id, corpus.clip_len.cpu())
    planar = corpus.base_lin_vel[:, :2].cpu().norm(dim=-1)
    rate = corpus.base_ang_vel[:, 2].cpu().abs()

    def med(cls: str, vals: torch.Tensor, floor: float) -> Optional[float]:
        if cls not in names:
            return None
        v = vals[per_frame == names.index(cls)]
        v = v[v >= floor]
        return float(v.median()) if v.numel() else None

    return {"forward": med("walk_fwd", planar, min_speed),
            "backward": med("walk_back", planar, min_speed),
            "lateral": med("strafe", planar, min_speed),
            "turn_rate": med("turn", rate, min_rate)}


def load_locomotion_corpus(cfg: LocoConfig, device) -> tuple[MotionCorpus, ClipsMeta]:
    """Open ``<cache_dir>/cache`` and ``<cache_dir>/clips.json``; mirror if configured."""
    root = Path(require(cfg.corpus.cache_dir, "corpus.cache_dir"))
    corpus = MotionCorpus.load_cache(
        root / "cache", output_device=device, urdf_hash=file_sha256(ROBOT_URDF),
        physics_fps=cfg.fps, src_fps=cfg.src_fps, keypoint_links=KEYPOINT_LINKS)
    corpus = corpus.to_storage(device)
    if cfg.corpus.mirror:
        corpus = mirror_corpus(corpus)
    meta_path = root / "clips.json"
    if meta_path.exists():
        meta = ClipsMeta.load(meta_path)
    else:
        meta = ClipsMeta(classes={n: "unknown" for n in corpus.clip_names})
    return corpus, meta

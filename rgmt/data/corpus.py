"""CPU-resident multi-clip motion corpus with clip-boundary-aware access."""
from dataclasses import dataclass, field
from typing import List
import json
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
from rgmt.utils.rotation import encode_angles, quat_rotate_inverse, gravity_projection
from rgmt.data.cache_key import SCHEMA_VERSION, params_key

TENSOR_KEYS = ["base_pos", "base_quat", "joint_q", "joint_qd",
               "base_lin_vel", "base_ang_vel", "kp_pos_world", "kp_vel_world"]

@dataclass
class MotionCorpus:
    physics_fps: int
    src_fps: int
    actuated_idx: torch.LongTensor       # (23,) CPU
    actuated_names: list
    keypoint_links: list
    output_device: torch.device

    base_pos: torch.Tensor               # all (TotalFrames, ...) on CPU
    base_quat: torch.Tensor
    joint_q: torch.Tensor
    joint_qd: torch.Tensor
    base_lin_vel: torch.Tensor
    base_ang_vel: torch.Tensor
    kp_pos_world: torch.Tensor           # (Total, Kp, 3)
    kp_vel_world: torch.Tensor

    clip_start: torch.LongTensor         # (C,)
    clip_len: torch.LongTensor
    clip_end: torch.LongTensor           # (C,) inclusive last global idx
    frame_clip_id: torch.LongTensor      # (Total,)
    clip_names: list

    _valid_starts_cache: dict = field(default_factory=dict, compare=False)
    # Optional per-frame RSI sampling weights (n_frames,). None = uniform.
    _frame_sample_weights: object = field(default=None, compare=False)

    # ---- properties
    @property
    def n_frames(self) -> int:
        return int(self.base_pos.shape[0])

    @property
    def n_clips(self) -> int:
        return len(self.clip_names)

    @property
    def Kp(self) -> int:
        return int(self.kp_pos_world.shape[1])

    # ---- construction
    @classmethod
    def from_clips(cls, clips: List, clip_names: List[str], keypoint_links: List[str],
                   output_device="cpu") -> "MotionCorpus":
        if len(clips) == 0:
            raise ValueError("from_clips: no clips")
        ref = clips[0]
        for c in clips:
            if c.physics_fps != ref.physics_fps or c.src_fps != ref.src_fps:
                raise ValueError("from_clips: clips disagree on fps")
            if c._kp_pos_world is None:
                raise ValueError("from_clips: clip loaded without keypoint_links")
        cat = {k: torch.cat([_clip_tensor(c, k).cpu() for c in clips], dim=0) for k in TENSOR_KEYS}
        lens = torch.tensor([c.n_frames for c in clips], dtype=torch.long)
        starts = torch.zeros(len(clips), dtype=torch.long)
        starts[1:] = torch.cumsum(lens, 0)[:-1]
        ends = starts + lens - 1
        frame_clip_id = torch.repeat_interleave(torch.arange(len(clips)), lens)
        return cls(
            physics_fps=ref.physics_fps, src_fps=ref.src_fps,
            actuated_idx=ref.actuated_idx.cpu(), actuated_names=list(ref.actuated_names),
            keypoint_links=list(keypoint_links),
            output_device=torch.device(output_device),
            clip_start=starts, clip_len=lens, clip_end=ends, frame_clip_id=frame_clip_id,
            clip_names=list(clip_names), **cat,
        )

    # ---- storage device -----------------------------------------------------
    @property
    def storage_device(self) -> torch.device:
        """Where the concatenated corpus tensors live (CPU or GPU)."""
        return self.base_pos.device

    def to_storage(self, device) -> "MotionCorpus":
        """Move corpus storage to `device` (e.g. GPU when it fits: ~0.5 GB for
        a 4-hour corpus). Accessors gather on the storage device and return on
        output_device — with both on GPU every per-step copy disappears
        (profiled at ~40% of env.step time when CPU-resident)."""
        device = torch.device(device)
        for k in TENSOR_KEYS:
            setattr(self, k, getattr(self, k).to(device))
        self.clip_start = self.clip_start.to(device)
        self.clip_len = self.clip_len.to(device)
        self.clip_end = self.clip_end.to(device)
        self.frame_clip_id = self.frame_clip_id.to(device)
        self.actuated_idx = self.actuated_idx.to(device)
        self._valid_starts_cache.clear()
        if self._frame_sample_weights is not None:
            self._frame_sample_weights = self._frame_sample_weights.to(device)
        return self

    # ---- RSI sampling weights ------------------------------------------------
    def set_clip_sampling_weights(self, weights_by_name: dict, default: float = 1.0) -> dict:
        """Weight RSI start-frame sampling per clip (uniform within a clip).

        ``weights_by_name`` maps clip name -> relative weight. Corpus clips
        absent from the dict get ``default``. Unknown names in the dict raise
        (typo protection — a silently ignored weight file would look exactly
        like a null result). Returns {matched, total_clips, weighted_mass_share}
        so callers can log what the reweighting actually did.
        """
        unknown = set(weights_by_name) - set(self.clip_names)
        if unknown:
            raise ValueError(
                f"set_clip_sampling_weights: {len(unknown)} names not in corpus, "
                f"e.g. {sorted(unknown)[:3]}")
        w_clip = torch.full((self.n_clips,), float(default), dtype=torch.float32)
        for i, name in enumerate(self.clip_names):
            if name in weights_by_name:
                w_clip[i] = float(weights_by_name[name])
        if (w_clip <= 0).all():
            raise ValueError("set_clip_sampling_weights: all weights <= 0")
        dev = self.storage_device
        self._frame_sample_weights = torch.repeat_interleave(
            w_clip.to(dev), self.clip_len.to(dev))
        mass = self._frame_sample_weights
        boosted = torch.repeat_interleave(
            (w_clip > float(default)).to(dev), self.clip_len.to(dev))
        return {
            "matched": len(set(weights_by_name) & set(self.clip_names)),
            "total_clips": self.n_clips,
            "boosted_mass_share": float(mass[boosted].sum() / mass.sum()),
        }

    # ---- accessors (idx = global LongTensor; results on output_device)
    def at(self, idx) -> dict:
        i = idx.to(self.storage_device); d = self.output_device
        return {
            "base_pos": self.base_pos[i].to(d), "base_quat": self.base_quat[i].to(d),
            "base_lin_vel": self.base_lin_vel[i].to(d), "base_ang_vel": self.base_ang_vel[i].to(d),
            "joint_q": self.joint_q[i].to(d), "joint_qd": self.joint_qd[i].to(d),
        }

    def command_at(self, idx) -> torch.Tensor:
        f = self.at(idx)
        v = quat_rotate_inverse(f["base_quat"], f["base_lin_vel"])
        w = quat_rotate_inverse(f["base_quat"], f["base_ang_vel"])
        g = gravity_projection(f["base_quat"])
        q = encode_angles(f["joint_q"])
        return torch.cat([v, w, g, q], dim=-1)            # (B,55)

    def command_window(self, idx, L: int) -> torch.Tensor:
        i = idx.to(self.storage_device)
        cid = self.frame_clip_id[i]
        lo = self.clip_start[cid].unsqueeze(1)            # (B,1)
        hi = self.clip_end[cid].unsqueeze(1)
        offs = torch.arange(-L, L + 1, device=self.storage_device)
        grid = (i.unsqueeze(1) + offs.unsqueeze(0))       # (B,S)
        grid = torch.maximum(grid, lo)
        grid = torch.minimum(grid, hi)                    # clamp within owning clip
        flat = self.command_at(grid.reshape(-1))          # (B*S,55) on device
        return flat.reshape(idx.shape[0], 2 * L + 1, -1)

    def keypoints_at(self, idx):
        i = idx.to(self.storage_device); d = self.output_device
        return self.kp_pos_world[i].to(d), self.kp_vel_world[i].to(d)

    def keypoints_root_relative_at(self, idx):
        pos, _ = self.keypoints_at(idx)
        root = self.at(idx)["base_pos"].unsqueeze(1)
        return pos - root

    def clip_end_of(self, idx):
        i = idx.to(self.storage_device)
        return self.clip_end[self.frame_clip_id[i]].to(self.output_device)

    def sample_index(self, n: int, max_lookahead: int) -> torch.LongTensor:
        vs = self._valid_starts_cache.get(max_lookahead)
        if vs is None:
            room = self.clip_end[self.frame_clip_id] - torch.arange(
                self.n_frames, device=self.storage_device)
            vs = torch.nonzero(room >= max_lookahead, as_tuple=False).squeeze(1)
            if vs.numel() == 0:
                raise ValueError(
                    f"no clip has >= {max_lookahead} lookahead room; clips too short")
            self._valid_starts_cache[max_lookahead] = vs
        if self._frame_sample_weights is not None:
            sel = torch.multinomial(
                self._frame_sample_weights[vs], n, replacement=True)
        else:
            sel = torch.randint(0, vs.shape[0], (n,), device=vs.device)
        return vs[sel].to(self.output_device)

    # ---- cache IO
    def save_cache(self, cache_dir, *, source_hashes: dict, urdf_hash: str,
                   ground: bool = True) -> None:
        cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
        clips_meta = []
        for c in range(self.n_clips):
            s = int(self.clip_start[c]); e = int(self.clip_end[c]) + 1
            name = self.clip_names[c]
            tensors = {k: getattr(self, k)[s:e] for k in TENSOR_KEYS}
            fn = _write_clip_file(cache_dir, name, tensors)
            key = params_key(source_hash=source_hashes[name], urdf_hash=urdf_hash,
                             physics_fps=self.physics_fps, src_fps=self.src_fps,
                             keypoint_links=self.keypoint_links,
                             schema_version=SCHEMA_VERSION, ground=ground)
            clips_meta.append({"name": name, "file": fn, "n_frames": e - s,
                               "source_hash": source_hashes[name], "cache_key": key})
        manifest = {
            "schema_version": SCHEMA_VERSION, "physics_fps": self.physics_fps,
            "src_fps": self.src_fps, "keypoint_links": list(self.keypoint_links),
            "urdf_hash": urdf_hash, "ground": bool(ground),
            "actuated_names": list(self.actuated_names),
            "actuated_idx": self.actuated_idx.tolist(), "clips": clips_meta,
        }
        (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load_cache(cls, cache_dir, *, output_device, urdf_hash, physics_fps, src_fps,
                   keypoint_links, ground: bool = True) -> "MotionCorpus":
        cache_dir = Path(cache_dir)
        man_path = cache_dir / "manifest.json"
        if not man_path.exists():
            raise RuntimeError(f"motion cache: no manifest at {man_path}")
        man = json.loads(man_path.read_text())
        if man.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                f"motion cache: schema {man.get('schema_version')} != {SCHEMA_VERSION}; "
                f"rebuild with preprocess force=true")
        if (man["physics_fps"], man["src_fps"], list(man["keypoint_links"]), man["urdf_hash"]) != \
           (physics_fps, src_fps, list(keypoint_links), urdf_hash):
            raise RuntimeError(
                "motion cache: corpus params/urdf differ from request; rebuild with force=true")
        if bool(man.get("ground", True)) != bool(ground):
            raise RuntimeError(
                f"motion cache: ground={man.get('ground')} but caller expects "
                f"ground={ground} (different z-datum); rebuild with force=true")
        parts = {k: [] for k in TENSOR_KEYS}
        names, lens = [], []
        for cm in man["clips"]:
            exp = params_key(source_hash=cm["source_hash"], urdf_hash=urdf_hash,
                             physics_fps=physics_fps, src_fps=src_fps,
                             keypoint_links=keypoint_links,
                             schema_version=SCHEMA_VERSION, ground=ground)
            if cm["cache_key"] != exp:
                raise RuntimeError(f"motion cache: clip '{cm['name']}' stale (cache_key mismatch); "
                                   f"rebuild with force=true")
            fp = cache_dir / cm["file"]
            if not fp.exists():
                raise RuntimeError(f"motion cache: clip '{cm['name']}' missing file {fp}")
            t = load_file(str(fp))
            actual = t[TENSOR_KEYS[0]].shape[0]
            if actual != cm["n_frames"]:
                raise RuntimeError(
                    f"motion cache: clip '{cm['name']}' frame count {actual} "
                    f"!= manifest {cm['n_frames']}; rebuild with force=true")
            for k in TENSOR_KEYS:
                parts[k].append(t[k])
            names.append(cm["name"]); lens.append(cm["n_frames"])
        cat = {k: torch.cat(parts[k], dim=0) for k in TENSOR_KEYS}
        lens_t = torch.tensor(lens, dtype=torch.long)
        starts = torch.zeros(len(names), dtype=torch.long)
        starts[1:] = torch.cumsum(lens_t, 0)[:-1]
        ends = starts + lens_t - 1
        frame_clip_id = torch.repeat_interleave(torch.arange(len(names)), lens_t)
        return cls(
            physics_fps=physics_fps, src_fps=src_fps,
            actuated_idx=torch.tensor(man["actuated_idx"], dtype=torch.long),
            actuated_names=list(man["actuated_names"]), keypoint_links=list(keypoint_links),
            output_device=torch.device(output_device),
            clip_start=starts, clip_len=lens_t, clip_end=ends, frame_clip_id=frame_clip_id,
            clip_names=names, **cat,
        )


def _write_clip_file(cache_dir, name: str, tensors: dict) -> str:
    fn = f"clip_{name}.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(Path(cache_dir) / fn))
    return fn


def _clip_tensor(clip, key: str) -> torch.Tensor:
    """Pull a per-frame tensor off a MotionRef by the corpus key name."""
    return getattr(clip, "_kp_pos_world" if key == "kp_pos_world"
                   else "_kp_vel_world" if key == "kp_vel_world" else key)

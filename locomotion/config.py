"""Configuration for the locomotion controller.

Every scalar coefficient (nominal speeds, spring half-lives, feature weights,
search cadence, blend half-life) lives here. Defaults come from
``configs/default.yaml``; ``load_config`` layers an optional user YAML and
``key=value`` dotlist overrides on top and returns a typed ``LocoConfig``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from omegaconf import OmegaConf

DEFAULT_YAML = Path(__file__).resolve().parent / "configs" / "default.yaml"


@dataclass
class SpeedsConfig:
    # None -> use the corpus median recorded in clips.json at build time.
    forward: Optional[float] = None      # m/s, W
    backward: Optional[float] = None     # m/s, S
    lateral: Optional[float] = None      # m/s, A / D
    turn_rate: Optional[float] = None    # rad/s, Q / E


@dataclass
class SpringConfig:
    velocity_halflife: float = 0.2       # s, planar velocity toward the desired velocity
    rotation_halflife: float = 0.1       # s, yaw rate toward the desired yaw rate


@dataclass
class FeaturesConfig:
    horizons: List[int] = field(default_factory=lambda: [20, 40, 60])   # frames ahead at 60 fps
    root_vel: float = 1.0
    foot_pos: float = 0.75
    foot_vel: float = 1.0
    traj_pos: float = 1.0
    traj_dir: float = 1.5


@dataclass
class SearchConfig:
    interval: int = 6                    # frames between searches (6 = 10 Hz)
    stay_bias: float = 0.1               # normalised-feature units subtracted from the continuing frame
    on_command_change: bool = True       # also search the tick the command changes
    jump_margin: int = 20                # a jump target must keep this many frames beyond the horizon
    exclude_window: int = 20             # frames around the current one (same clip) never jumped to


@dataclass
class BlendConfig:
    halflife: float = 0.1                # s, inertialization decay of the switch offset


@dataclass
class ExcludeEntry:
    name: str
    reason: str


@dataclass
class CorpusConfig:
    # Directory holding cache/ (MotionCorpus cache) and clips.json (see build_corpus.py).
    cache_dir: str = "/mnt/shared/robotics/asimov/caches/loco"
    mirror: bool = True                  # append left-right mirrored clips at load time
    exclude: List[ExcludeEntry] = field(default_factory=list)


@dataclass
class CameraConfig:
    follow: bool = True                  # third-person follow camera in play.py (False = Newton fly camera)
    distance: float = 3.0                # m behind the robot along its heading
    height: float = 1.6                  # m above the ground
    target_height: float = 0.8           # m above the ground the camera looks at (root height)
    halflife: float = 0.25               # s, smoothing of the followed position and heading


@dataclass
class SceneConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)


@dataclass
class LocoConfig:
    fps: int = 60
    src_fps: int = 30
    speeds: SpeedsConfig = field(default_factory=SpeedsConfig)
    spring: SpringConfig = field(default_factory=SpringConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)


def load_config(path: str | None = None, overrides: Sequence[str] = ()) -> LocoConfig:
    """Dataclass defaults < default.yaml < user yaml (optional) < dotlist overrides."""
    conf = OmegaConf.merge(OmegaConf.structured(LocoConfig), OmegaConf.load(DEFAULT_YAML))
    if path is not None:
        conf = OmegaConf.merge(conf, OmegaConf.load(path))
    if overrides:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_object(conf)

import math
from locomotion.config import CameraConfig, load_config
from locomotion.play import FollowCamera


def _cam(**kw):
    return FollowCamera(CameraConfig(**kw), fps=60)


def test_config_has_scene_camera_defaults():
    cfg = load_config()
    assert cfg.scene.camera.follow is True
    assert cfg.scene.camera.distance == 3.0 and cfg.scene.camera.height == 1.6
    assert cfg.scene.camera.target_height == 0.8 and cfg.scene.camera.halflife == 0.25


def test_first_update_snaps_behind_robot_along_heading():
    cam = _cam(distance=3.0, height=1.6, target_height=0.8)
    yaw = math.pi / 2                                   # robot faces +y
    pos, target = cam.update((2.0, 1.0), yaw)
    assert abs(pos[0] - 2.0) < 1e-6 and abs(pos[1] - (1.0 - 3.0)) < 1e-6 and abs(pos[2] - 1.6) < 1e-6
    assert tuple(round(t, 6) for t in target) == (2.0, 1.0, 0.8)


def test_yaw_step_is_followed_gradually():
    cam = _cam(halflife=0.25)
    cam.update((0.0, 0.0), 0.0)
    pos1, _ = cam.update((0.0, 0.0), math.pi / 2)       # robot snaps to face +y
    assert pos1[0] < -2.5 and pos1[1] > -0.5              # still essentially behind the old heading
    for _ in range(60 * 3):                              # 3 s = 12 half-lives
        pos, _ = cam.update((0.0, 0.0), math.pi / 2)
    assert abs(pos[0]) < 0.02 and abs(pos[1] + 3.0) < 0.02


def test_yaw_wrap_takes_the_short_way():
    cam = _cam(halflife=0.25)
    cam.update((0.0, 0.0), math.pi - 0.05)
    pos, _ = cam.update((0.0, 0.0), -math.pi + 0.05)     # 0.1 rad apart across the wrap
    assert all(math.isfinite(v) for v in pos)
    assert abs(pos[1]) < 0.2                             # did not swing through yaw 0


def test_position_is_smoothed_and_reset_snaps():
    cam = _cam(halflife=0.25)
    cam.update((0.0, 0.0), 0.0)
    pos, target = cam.update((10.0, 0.0), 0.0)
    assert target[0] < 1.0                               # lags the jump
    cam.reset()
    pos, target = cam.update((10.0, 0.0), 0.0)
    assert abs(target[0] - 10.0) < 1e-6


class _FakeCamera:
    def __init__(self):
        self.pos = (0.0, 0.0, 0.0); self.target = None

    def look_at(self, target):
        self.target = tuple(target)


class _FakeViewer:
    camera_speed = 4.0

    def __init__(self):
        self.camera = _FakeCamera()


def test_apply_moves_viewer_camera_and_disables_key_flight():
    v = _FakeViewer(); cam = _cam(distance=2.0, height=1.0, target_height=0.5)
    cam.attach(v)
    assert v.camera_speed == 0.0
    cam.apply(v, (1.0, 0.0), 0.0)
    assert tuple(round(x, 6) for x in v.camera.pos) == (-1.0, 0.0, 1.0)
    assert tuple(round(x, 6) for x in v.camera.target) == (1.0, 0.0, 0.5)


def test_apply_is_noop_without_camera():
    class NoCam:
        pass
    cam = _cam()
    cam.apply(NoCam(), (0.0, 0.0), 0.0)                  # must not raise

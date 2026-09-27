import torch
from locomotion.play import TrajectoryOverlay


class _Ctl:
    class player:
        world_xy = torch.zeros(2)
    class debug:
        traj_xy = torch.tensor([[0.3, 0.0], [0.6, 0.0], [1.0, 0.0]])
        traj_yaw = torch.zeros(3)


class _RaisingViewer:
    calls = 0

    def log_lines(self, *a, **k):
        self.calls += 1
        raise TypeError("signature changed")


class _NoLinesViewer:
    pass


def test_overlay_warns_once_then_disables(capsys):
    v = _RaisingViewer(); ov = TrajectoryOverlay()
    for _ in range(3):
        ov.draw(v, _Ctl())
    assert v.calls == 1 and not ov.enabled
    err = capsys.readouterr().err
    assert err.count("trajectory overlay disabled") == 1 and "signature changed" in err


def test_overlay_is_noop_without_log_lines(capsys):
    ov = TrajectoryOverlay()
    ov.draw(_NoLinesViewer(), _Ctl())
    assert ov.enabled and capsys.readouterr().err == ""

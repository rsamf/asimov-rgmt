import math
import pytest
from locomotion.command import Command, from_keys, ScriptedSource


def test_opposing_keys_cancel():
    assert from_keys({"w", "s"}) == Command()
    assert from_keys({"a", "d"}).move == (0.0, 0.0)
    assert from_keys({"q", "e"}).yaw == 0.0
    assert from_keys(set()).is_zero


def test_axes_and_signs():
    assert from_keys({"w"}).move == (1.0, 0.0)
    assert from_keys({"s"}).move == (-1.0, 0.0)
    assert from_keys({"a"}).move == (0.0, 1.0)      # left is +y
    assert from_keys({"d"}).move == (0.0, -1.0)
    assert from_keys({"q"}).yaw == 1.0              # ccw
    assert from_keys({"e"}).yaw == -1.0


def test_diagonal_is_clamped_to_unit_norm():
    c = from_keys({"w", "d"})
    assert math.isclose(math.hypot(*c.move), 1.0, abs_tol=1e-6)
    assert math.isclose(c.move[0], -c.move[1], abs_tol=1e-6) and c.move[0] > 0
    assert c.yaw == 0.0


def test_unknown_keys_are_ignored():
    assert from_keys({"w", "x", "space"}) == from_keys({"w"})


def test_scripted_source_sequence():
    src = ScriptedSource("w:0.05,wd:0.05,:0.05,q:0.05", fps=60)
    assert src.total_ticks == 12
    seq = [src.next() for _ in range(13)]
    assert seq[0] == from_keys({"w"}) and seq[2] == from_keys({"w"})
    assert seq[3] == from_keys({"w", "d"})
    assert seq[6] == Command()
    assert seq[9] == from_keys({"q"})
    assert seq[12] is None
    assert list(ScriptedSource("e:0.05", fps=60)) == [from_keys({"e"})] * 3


@pytest.mark.parametrize("bad", ["w:abc", "w", "x:1", "w:-1", ""])
def test_scripted_source_rejects_bad_segments(bad):
    with pytest.raises(ValueError) as ei:
        ScriptedSource(bad, fps=60)
    assert bad == "" or bad.split(":")[0] in str(ei.value)

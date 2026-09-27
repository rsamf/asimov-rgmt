import textwrap
from locomotion.config import load_config, LocoConfig


def test_defaults_load_from_yaml():
    cfg = load_config()
    assert isinstance(cfg, LocoConfig)
    assert cfg.fps == 60 and cfg.src_fps == 30
    assert cfg.speeds.forward is None            # null -> corpus median
    assert cfg.spring.velocity_halflife == 0.2
    assert cfg.features.horizons == [20, 40, 60]
    assert cfg.search.interval == 6
    assert cfg.blend.halflife == 0.1
    assert cfg.corpus.mirror is True
    names = [e.name for e in cfg.corpus.exclude]
    assert "ACCAD__MartialArtsWalksTurns_c3d__E9 - side step left" in names
    assert all(e.reason for e in cfg.corpus.exclude)


def test_dotlist_overrides_win():
    cfg = load_config(overrides=["speeds.forward=0.8", "blend.halflife=0.15", "corpus.mirror=false"])
    assert cfg.speeds.forward == 0.8
    assert cfg.blend.halflife == 0.15
    assert cfg.corpus.mirror is False


def test_yaml_file_overrides_defaults_then_dotlist(tmp_path):
    y = tmp_path / "my.yaml"
    y.write_text(textwrap.dedent("""
        speeds: {forward: 0.7}
        search: {interval: 12}
    """))
    cfg = load_config(str(y), overrides=["search.interval=3"])
    assert cfg.speeds.forward == 0.7
    assert cfg.search.interval == 3              # dotlist beats file
    assert cfg.spring.rotation_halflife == 0.1   # untouched default survives

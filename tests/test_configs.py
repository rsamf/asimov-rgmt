from hydra import initialize, compose


def test_compose():
    with initialize(version_base=None, config_path="../rgmt/configs"):
        cfg = compose(config_name="train")
    assert cfg.env.num_envs > 0
    assert cfg.network.n_embd == 128
    assert cfg.algo.lr > 0
    assert cfg.env.K == 9 and cfg.env.L == 10

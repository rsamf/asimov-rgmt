"""KL-shock rollback guard + split-defined RSI universe.

A long training run once died to a single destructive update the 1.5x early
stop could only see
after it was applied. kl_shock_factor > 0 snapshots model+optimizer at update
start and rolls the whole iteration back when any minibatch's approx_kl
exceeds factor*target_kl.
"""
import torch

from rgmt.algos.ppo import PPOTrainer, PPOConfig
from rgmt.policy.networks import RGMTActorCritic, PolicyDims
from rgmt.train import compose_rsi_weights


def _data(d, B=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return dict(
        obs=r(B, d.obs_dim), history=r(B, d.hist_len, d.obs_dim),
        cmd_window=r(B, d.cmd_len, d.cmd_dim), critic_obs=r(B, d.priv_dim),
        actions=r(B, 23), logp=r(B), returns=r(B),
        advantages=r(B), values=r(B))


def _params(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_shock_rolls_back_model_and_optimizer():
    torch.manual_seed(0)
    d = PolicyDims(priv_dim=8)
    model = RGMTActorCritic(d)
    # random logp makes the very first minibatch's approx_kl nonzero, so a
    # near-zero threshold guarantees the guard fires on this batch
    ppo = PPOTrainer(model, PPOConfig(n_epochs=1, mb_size=16, target_kl=0.02,
                                      kl_shock_factor=1e-9))
    before = _params(model)
    stats = ppo.update(_data(d))
    assert stats["kl_shock"] == 1.0
    assert _same(before, _params(model)), "params must be restored on shock"
    # optimizer state was restored too: a fresh Adam snapshot has no step
    # counts, so state should be exactly the pre-update (empty) state
    assert ppo.opt.state_dict()["state"] == {}


def test_no_shock_below_threshold_params_move():
    torch.manual_seed(0)
    d = PolicyDims(priv_dim=8)
    model = RGMTActorCritic(d)
    ppo = PPOTrainer(model, PPOConfig(n_epochs=1, mb_size=16, target_kl=0.02,
                                      kl_shock_factor=1e9))
    before = _params(model)
    stats = ppo.update(_data(d))
    assert stats["kl_shock"] == 0.0
    assert not _same(before, _params(model)), "update must apply when no shock"


def test_guard_off_is_silent_and_applies():
    torch.manual_seed(0)
    d = PolicyDims(priv_dim=8)
    model = RGMTActorCritic(d)
    ppo = PPOTrainer(model, PPOConfig(n_epochs=1, mb_size=16, target_kl=0.02))
    before = _params(model)
    stats = ppo.update(_data(d))
    assert "kl_shock" not in stats, "disabled guard must not change the log surface"
    assert not _same(before, _params(model))


# ---- split-defined RSI universe (train.py) --------------------------------

def test_rsi_weights_zero_unlisted_clips():
    corpus_names = ["a", "b", "c", "d", "e"]
    split = {"train": ["a", "b"], "test": ["c"]}
    fail_ema = {"a": 1.0}          # always-failing train clip
    w = compose_rsi_weights(corpus_names, split, fail_ema, mining_boost=3.0)
    assert w["a"] == 4.0           # 1 + 3*EMA mining weight
    assert "b" not in w            # default weight 1.0 via corpus default
    assert w["c"] == 0.0           # test clip excluded
    assert w["d"] == 0.0 and w["e"] == 0.0   # neither-role clips excluded


def test_rsi_weights_no_split_passthrough():
    w = compose_rsi_weights(["a", "b"], None, {"a": 0.5}, mining_boost=3.0)
    assert w == {"a": 2.5}

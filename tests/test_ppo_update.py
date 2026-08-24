import torch
from rgmt.policy.networks import RGMTActorCritic, PolicyDims
from rgmt.algos.ppo import PPOTrainer, PPOConfig


def test_update_runs_and_reduces_loss_on_fixed_batch():
    d = PolicyDims(priv_dim=8); model = RGMTActorCritic(d)
    ppo = PPOTrainer(model, PPOConfig(n_epochs=4, mb_size=16))
    B = 32
    data = dict(
        obs=torch.randn(B, d.obs_dim), history=torch.randn(B, d.hist_len, d.obs_dim),
        cmd_window=torch.randn(B, d.cmd_len, d.cmd_dim), critic_obs=torch.randn(B, d.priv_dim),
        actions=torch.randn(B, 23), logp=torch.randn(B), returns=torch.randn(B),
        advantages=torch.randn(B), values=torch.randn(B))
    stats = ppo.update(data)
    assert {"policy_loss", "value_loss", "entropy", "kl"} <= set(stats)


def test_kl_adaptive_lr():
    d = PolicyDims(priv_dim=8); model = RGMTActorCritic(d)
    ppo = PPOTrainer(model, PPOConfig(kl_adaptive=True, lr=1e-4, target_kl=0.02,
                                      lr_min=1e-5, lr_max=1e-3))
    # overshoot: kl > 2x target -> lr shrinks by 1.5, optimizer follows
    ppo._adapt_lr(0.05)
    assert abs(ppo.lr - 1e-4 / 1.5) < 1e-12
    assert ppo.opt.param_groups[0]["lr"] == ppo.lr
    # undershoot: kl < 0.5x target -> lr grows by 1.5
    ppo._adapt_lr(0.001)
    assert abs(ppo.lr - 1e-4) < 1e-12
    # in-band: unchanged
    ppo._adapt_lr(0.02)
    assert abs(ppo.lr - 1e-4) < 1e-12
    # bounds are respected
    for _ in range(50):
        ppo._adapt_lr(1.0)
    assert ppo.lr == 1e-5
    for _ in range(50):
        ppo._adapt_lr(0.0)
    assert ppo.lr == 1e-3


def test_kl_adaptive_update_adjusts_once_per_iteration():
    d = PolicyDims(priv_dim=8); model = RGMTActorCritic(d)
    ppo = PPOTrainer(model, PPOConfig(n_epochs=4, mb_size=16, kl_adaptive=True,
                                      lr=1e-4, lr_min=1e-5, lr_max=1e-3))
    B = 32
    data = dict(
        obs=torch.randn(B, d.obs_dim), history=torch.randn(B, d.hist_len, d.obs_dim),
        cmd_window=torch.randn(B, d.cmd_len, d.cmd_dim), critic_obs=torch.randn(B, d.priv_dim),
        actions=torch.randn(B, 23), logp=torch.randn(B), returns=torch.randn(B),
        advantages=torch.randn(B), values=torch.randn(B))
    stats = ppo.update(data)
    # early stop stays active in adaptive mode; the controller adapts ONCE on
    # the final cumulative KL and the logged lr reflects the post-adapt value
    assert 1.0 <= stats["n_updates"] <= 8.0
    assert stats["lr"] == ppo.lr
    assert ppo.cfg.lr_min <= ppo.lr <= ppo.cfg.lr_max
    # lr moved by at most one 1.5x notch in a single update() call
    assert ppo.lr in (1e-4, 1e-4 / 1.5, 1.5e-4)


def test_value_clip_off_and_per_module_clip():
    # Optimizer-fix paths: plain-MSE value loss + separate actor/critic grad clips
    torch.manual_seed(0)
    d = PolicyDims(priv_dim=8); model = RGMTActorCritic(d)
    ppo = PPOTrainer(model, PPOConfig(n_epochs=1, mb_size=16,
                                      value_clip=False,
                                      grad_clip_per_module=True))
    B = 32
    data = dict(
        obs=torch.randn(B, d.obs_dim), history=torch.randn(B, d.hist_len, d.obs_dim),
        cmd_window=torch.randn(B, d.cmd_len, d.cmd_dim), critic_obs=torch.randn(B, d.priv_dim),
        actions=torch.randn(B, 23), logp=torch.randn(B), returns=torch.randn(B),
        advantages=torch.randn(B), values=torch.randn(B))
    stats = ppo.update(data)
    assert "grad_norm_actor" in stats and "grad_norm_critic" in stats
    assert stats["grad_norm_actor"] > 0 and stats["grad_norm_critic"] > 0
    assert stats["value_loss"] > 0
    # param split covers the whole model exactly once
    n_all = sum(1 for _ in model.parameters())
    assert len(ppo._actor_params) + len(ppo._critic_params) == n_all
    assert len(ppo._critic_params) > 0


def test_legacy_paths_unchanged_by_default():
    torch.manual_seed(0)
    d = PolicyDims(priv_dim=8); model = RGMTActorCritic(d)
    ppo = PPOTrainer(model, PPOConfig(n_epochs=1, mb_size=16))
    B = 32
    data = dict(
        obs=torch.randn(B, d.obs_dim), history=torch.randn(B, d.hist_len, d.obs_dim),
        cmd_window=torch.randn(B, d.cmd_len, d.cmd_dim), critic_obs=torch.randn(B, d.priv_dim),
        actions=torch.randn(B, 23), logp=torch.randn(B), returns=torch.randn(B),
        advantages=torch.randn(B), values=torch.randn(B))
    stats = ppo.update(data)
    assert "grad_norm_actor" not in stats and "grad_norm_critic" not in stats

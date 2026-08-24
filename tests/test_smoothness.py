"""Action-smoothness regularization: temporal budget + variable Lipschitz bound.

Covers the loss math, the pair masking, the noise-injection loophole, gradient
isolation, and the off-by-default contract.
"""
import pytest
import torch

from rgmt.algos.ppo import PPOConfig, PPOTrainer
from rgmt.algos.rollout import RolloutBuffer
from rgmt.policy.networks import PolicyDims, RGMTActorCritic

_HAS = torch.cuda.is_available()

ACT = 23
OBS = 98
CMD = 55
PRIV = 187


def _dims():
    return PolicyDims(priv_dim=PRIV, obs_dim=OBS, cmd_dim=CMD)


def _model(device="cpu"):
    torch.manual_seed(0)
    return RGMTActorCritic(_dims()).to(device)


def _bundle(n, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(1)
    return {
        "obs": torch.randn(n, OBS, generator=g).to(device),
        "history": torch.randn(n, 10, OBS, generator=g).to(device),
        "cmd_window": torch.randn(n, 21, CMD, generator=g).to(device),
        "critic_obs": torch.randn(n, PRIV, generator=g).to(device),
    }


# --------------------------------------------------------------------------
# Loss math (pure tensor ops, mirrors ppo.update)
# --------------------------------------------------------------------------

def _temporal_loss(tanh_mu, prev, ref_qd, pair, *, slack, floor, dt, scale):
    budget = (ref_qd.abs() * dt) / scale * slack + floor
    excess = ((tanh_mu - prev).abs() - budget).clamp_min(0.0)
    w = pair.unsqueeze(-1)
    denom = (w.sum() * tanh_mu.shape[-1]).clamp_min(1.0)
    return (excess.pow(2) * w).sum() / denom


def test_temporal_one_sided_below_budget_is_zero():
    """A delta strictly inside the budget contributes exactly nothing."""
    tanh_mu = torch.full((4, ACT), 0.10)
    prev = torch.full((4, ACT), 0.09)          # delta 0.01
    ref_qd = torch.zeros(4, ACT)               # budget = floor only
    pair = torch.ones(4)
    loss = _temporal_loss(tanh_mu, prev, ref_qd, pair,
                          slack=2.5, floor=0.02, dt=1 / 60, scale=0.3)
    assert loss.item() == 0.0


def test_temporal_penalizes_only_the_excess():
    """Above budget, the loss is the SQUARED excess — not the squared delta."""
    delta, floor = 0.05, 0.02
    tanh_mu = torch.full((1, ACT), delta)
    prev = torch.zeros(1, ACT)
    loss = _temporal_loss(tanh_mu, prev, torch.zeros(1, ACT), torch.ones(1),
                          slack=2.5, floor=floor, dt=1 / 60, scale=0.3)
    assert loss.item() == pytest.approx((delta - floor) ** 2, rel=1e-6)


def test_temporal_velocity_budget_scales_with_slack_and_refqd():
    """Doubling slack doubles the velocity allowance; slack=0 leaves the floor."""
    tanh_mu = torch.full((1, ACT), 0.5)
    prev = torch.zeros(1, ACT)
    ref_qd = torch.full((1, ACT), 3.0)         # rad/s
    kw = dict(floor=0.02, dt=1 / 60, scale=0.3)
    l1 = _temporal_loss(tanh_mu, prev, ref_qd, torch.ones(1), slack=2.5, **kw)
    l2 = _temporal_loss(tanh_mu, prev, ref_qd, torch.ones(1), slack=5.0, **kw)
    l0 = _temporal_loss(tanh_mu, prev, ref_qd, torch.ones(1), slack=0.0, **kw)
    assert l2 < l1 < l0                        # bigger budget -> smaller penalty
    # slack=0 must equal the zero-velocity case (floor only)
    flat = _temporal_loss(tanh_mu, prev, torch.zeros(1, ACT), torch.ones(1),
                          slack=2.5, **kw)
    assert l0.item() == pytest.approx(flat.item(), rel=1e-6)


def test_temporal_masking_excludes_reset_pairs():
    """pair_valid=0 rows contribute nothing and are excluded from the mean."""
    tanh_mu = torch.full((2, ACT), 0.9)
    prev = torch.zeros(2, ACT)
    ref_qd = torch.zeros(2, ACT)
    both = _temporal_loss(tanh_mu, prev, ref_qd, torch.ones(2),
                          slack=2.5, floor=0.02, dt=1 / 60, scale=0.3)
    one = _temporal_loss(tanh_mu, prev, ref_qd, torch.tensor([1.0, 0.0]),
                         slack=2.5, floor=0.02, dt=1 / 60, scale=0.3)
    # masking a row must not dilute the per-sample mean
    assert one.item() == pytest.approx(both.item(), rel=1e-6)
    none = _temporal_loss(tanh_mu, prev, ref_qd, torch.zeros(2),
                          slack=2.5, floor=0.02, dt=1 / 60, scale=0.3)
    assert none.item() == 0.0


# --------------------------------------------------------------------------
# Rollout pairing / masking
# --------------------------------------------------------------------------

def test_buffer_emits_smooth_fields_only_when_enabled():
    d = _dims()
    off = RolloutBuffer(2, 3, d, "cpu")
    on = RolloutBuffer(2, 3, d, "cpu", store_smooth=True)
    b = _bundle(2)
    for buf in (off, on):
        for _ in range(3):
            buf.add(b, torch.zeros(2, ACT), torch.zeros(2), torch.zeros(2),
                    torch.zeros(2), torch.zeros(2),
                    tanh_mu=torch.zeros(2, ACT), ref_qd=torch.zeros(2, ACT))
    keys_off = set(off.compute_gae(torch.zeros(2), 0.99, 0.95))
    keys_on = set(on.compute_gae(torch.zeros(2), 0.99, 0.95))
    assert "prev_tanh_mu" not in keys_off
    assert {"prev_tanh_mu", "ref_qd", "pair_valid"} <= keys_on
    assert keys_on - keys_off == {"prev_tanh_mu", "ref_qd", "pair_valid"}


def test_buffer_pairing_shift_and_done_masking():
    """prev_tanh_mu is the previous timestep, same env; t=0 and post-done invalid."""
    T, N = 3, 2
    buf = RolloutBuffer(N, T, _dims(), "cpu", store_smooth=True)
    b = _bundle(N)
    dones = [torch.tensor([0.0, 1.0]),   # env1 terminates at t=0
             torch.tensor([0.0, 0.0]),
             torch.tensor([0.0, 0.0])]
    for t in range(T):
        buf.add(b, torch.zeros(N, ACT), torch.zeros(N), torch.zeros(N),
                torch.zeros(N), dones[t],
                tanh_mu=torch.full((N, ACT), float(t + 1)),
                ref_qd=torch.zeros(N, ACT))
    out = buf.compute_gae(torch.zeros(N), 0.99, 0.95)
    prev = out["prev_tanh_mu"].reshape(T, N, ACT)
    valid = out["pair_valid"].reshape(T, N)
    # shift: prev at t is tanh_mu from t-1
    assert torch.allclose(prev[1], torch.full((N, ACT), 1.0))
    assert torch.allclose(prev[2], torch.full((N, ACT), 2.0))
    # t=0 has no predecessor
    assert torch.equal(valid[0], torch.zeros(N))
    # env0 continuous; env1's pair at t=1 spans its reset at t=0
    assert valid[1, 0] == 1.0 and valid[1, 1] == 0.0
    assert torch.equal(valid[2], torch.ones(N))


# --------------------------------------------------------------------------
# Noise injection: the history-duplication loophole
# --------------------------------------------------------------------------

def test_noise_off_is_identity_and_consumes_no_rng():
    m = _model()
    b = _bundle(4)
    with torch.no_grad():
        clean = m._features(b)
        state = torch.random.get_rng_state()
        again = m._features(b, obs_noise_std=0.0)
    assert torch.equal(clean, again)
    assert torch.equal(state, torch.random.get_rng_state()), \
        "obs_noise_std=0 must not draw from the RNG (bit-identity contract)"


def test_noise_perturbs_both_obs_and_last_history_token():
    """The loophole regression test.

    history[:, -1] IS obs (the env rolls the fresh obs in and rebuilds it as
    `obs`). If only `obs` were noised, the actor could route current-state
    dependence through the history encoder and make the penalty measure nothing.
    Assert the same perturbation reaches both copies.
    """
    m = _model()
    b = _bundle(4)
    # make the duplication explicit, as the env does
    b["history"][:, -1] = b["obs"]

    torch.manual_seed(7)
    with torch.no_grad():
        noisy = m._features(b, obs_noise_std=0.1)

    # Reconstruct what the obs-only variant would have produced, using the same
    # noise draw; it must differ -> the history path really is perturbed too.
    torch.manual_seed(7)
    with torch.no_grad():
        o_n = m.obs_rms.normalize(b["obs"])
        eps = torch.randn_like(o_n) * 0.1
        hist_clean = m.obs_rms.normalize(b["history"])
        cmd = m.cmd_rms.normalize(b["cmd_window"])
        h_obs_only = m.history_enc(hist_clean)
        obs_only = torch.cat(
            [o_n + eps, h_obs_only, m.command_enc(h_obs_only, cmd)], dim=-1
        )
    assert not torch.allclose(noisy, obs_only, atol=1e-7), \
        "history[:, -1] was not perturbed — the evasion path is open"


def test_actor_mean_matches_act_inference_when_clean():
    m = _model()
    b = _bundle(4)
    with torch.no_grad():
        assert torch.equal(m.actor_mean(b), m.act_inference(b))


def test_return_mean_flags_are_opt_in():
    m = _model()
    b = _bundle(4)
    with torch.no_grad():
        assert len(m.act(b)) == 3
        assert len(m.act(b, return_mean=True)) == 4
        a = torch.zeros(4, ACT)
        assert len(m.evaluate(b, a)) == 3
        logp, ent, val, mu = m.evaluate(b, a, return_mean=True)
        assert torch.equal(mu, m.act_inference(b))


# --------------------------------------------------------------------------
# Gradient isolation
# --------------------------------------------------------------------------

def test_penalties_do_not_touch_critic_or_log_std():
    """Aux losses must reach actor + encoders only."""
    m = _model()
    b = _bundle(8)
    mu = m.actor_mean(b)
    mu_noisy = m.actor_mean(b, obs_noise_std=0.05)
    tanh_mu = torch.tanh(mu)
    prev = torch.zeros_like(tanh_mu)
    loss = ((tanh_mu - prev).abs() - 0.0).clamp_min(0).pow(2).mean() \
        + (torch.tanh(mu_noisy) - tanh_mu).abs().pow(2).mean()
    loss.backward()

    for name, p in m.named_parameters():
        if name.startswith("critic") or name == "log_std":
            assert p.grad is None or torch.all(p.grad == 0), \
                f"{name} received gradient from an actor-only penalty"
    assert m.actor[0].weight.grad is not None
    assert torch.any(m.actor[0].weight.grad != 0)
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in m.history_enc.parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in m.command_enc.parameters())


# --------------------------------------------------------------------------
# PPOTrainer wiring
# --------------------------------------------------------------------------

def _fake_data(n, device="cpu", smooth=True):
    b = _bundle(n, device)
    data = dict(b)
    data["actions"] = torch.zeros(n, ACT, device=device)
    data["logp"] = torch.zeros(n, device=device)
    data["values"] = torch.zeros(n, device=device)
    data["advantages"] = torch.randn(n, device=device)
    data["returns"] = torch.zeros(n, device=device)
    if smooth:
        data["prev_tanh_mu"] = torch.zeros(n, ACT, device=device)
        data["ref_qd"] = torch.zeros(n, ACT, device=device)
        data["pair_valid"] = torch.ones(n, device=device)
    return data


def test_disabled_update_adds_no_metrics_and_needs_no_extra_keys():
    m = _model()
    ppo = PPOTrainer(m, PPOConfig(n_epochs=1, mb_size=8))
    stats = ppo.update(_fake_data(16, smooth=False))
    assert not any(k.startswith(("smooth/", "spatial/")) for k in stats)


def test_enabled_update_logs_metrics():
    m = _model()
    cfg = PPOConfig(n_epochs=1, mb_size=8, lambda_smooth=0.1, lambda_spatial=0.01,
                    action_scale=0.3, ctrl_dt=1 / 60)
    stats = PPOTrainer(m, cfg).update(_fake_data(16))
    for k in ("smooth/loss", "smooth/excess_mean", "smooth/frac_over",
              "smooth/sat_frac", "spatial/loss", "spatial/excess_mean",
              "spatial/frac_over", "spatial/gain_p50", "spatial/gain_p95"):
        assert k in stats, f"missing metric {k}"
        assert stats[k] == stats[k]            # not NaN


def test_action_scale_must_be_set_when_enabled():
    m = _model()
    cfg = PPOConfig(n_epochs=1, mb_size=8, lambda_smooth=0.1)  # action_scale None
    with pytest.raises(ValueError, match="action_scale"):
        PPOTrainer(m, cfg).update(_fake_data(16))


def test_multi_epoch_rejected_with_temporal_penalty():
    """prev_tanh_mu is the behaviour policy's mean — multi-epoch would anchor stale."""
    m = _model()
    cfg = PPOConfig(n_epochs=5, mb_size=8, lambda_smooth=0.1, action_scale=0.3)
    with pytest.raises(ValueError, match="n_epochs"):
        PPOTrainer(m, cfg).update(_fake_data(16))

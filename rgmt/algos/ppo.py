"""PPO clipped-surrogate update adapted to structured policy bundles.

Adapted from an earlier internal PPO implementation with the following
changes:
- Data arrives as a flat dict of tensors (M rows) rather than a RolloutData
  object with time x env dimensions.
- Minibatch indices slice ALL bundle fields (obs, history, cmd_window,
  critic_obs) and the slice is passed as a dict to model.evaluate().
- model.evaluate(bundle, actions) returns (logp, entropy, value) matching
  RGMTActorCritic.evaluate() signature.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    lr: float = 3.0e-4
    n_epochs: int = 5
    mb_size: int = 16384
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 1.0
    dual_clip: float = 3.0       # dual-clip PPO floor for adv<0 (Ye et al. 2020)
    target_kl: float = 0.02
    # KL-shock rollback: the 1.5x early stop checks AFTER a minibatch update
    # is applied, so one catastrophic update slips through before the stop
    # fires (observed once in a 30k-iteration run: kl 0.08, avg_return
    # 297->7, unrecoverable at floored LR with entropy annealed). At
    # kl_shock_factor > 0 the update() snapshots model+optimizer state at
    # iteration start and, when any minibatch's approx_kl exceeds
    # factor*target_kl, RESTORES the snapshot and discards the whole
    # iteration's update (logged as train/kl_shock). One iteration of
    # progress traded for surviving the shock; 0 disables (bit-identical).
    kl_shock_factor: float = 0.0
    # Measured on a trained run: the +-clip trust region on V bound 95% of
    # samples. Returns are O(350) with |R-V| median ~2.7 vs clip 0.2, so the
    # critic's gradient zeroed after moving 0.2 toward targets 3-15 away,
    # leaving it permanently rate-limited. And the critic's raw grad norm
    # (~187) vs the actor's (~2) under ONE shared global clip scaled the
    # actor's step by ~0.005 (alone it would be ~0.5). Defaults preserve the
    # legacy behavior; the released training recipe sets value_clip=False,
    # grad_clip_per_module=True.
    value_clip: bool = True
    grad_clip_per_module: bool = False
    # KL-adaptive LR, adapted ONCE PER ITERATION on the final (cumulative)
    # KL. approx_kl measures divergence from the ROLLOUT policy, so it grows
    # across minibatch updates by construction; adapting on it per minibatch
    # misreads normal accumulation as overshoot and pins the LR at
    # lr_min. Instead: after the update epochs finish (or early-stop), shrink
    # the LR when the iteration's realized KL exceeded 1.5x target and grow
    # it when the iteration finished under 0.5x target. The hard early stop
    # stays active as shock protection; the controller then right-sizes the
    # NEXT iteration. When enabled the controller owns the LR; external
    # schedules must not override it.
    kl_adaptive: bool = False
    lr_min: float = 1.0e-5
    lr_max: float = 1.0e-3

    # ---- Action-smoothness regularization (both OFF by default) -------------
    # Two one-sided hinge penalties on the actor's MEAN action, in tanh space
    # (tanh(mu)*action_scale = commanded radians, so the budgets below are in
    # normalized-action units and convert cleanly).
    #
    # TEMPORAL (lambda_smooth): penalizes |tanh(mu_t) - tanh(mu_{t-1})| beyond a
    #   per-joint budget
    #       budget_j = |qd_ref_j| * ctrl_dt / action_scale * smooth_slack + smooth_eps_floor
    #   Actions here are residuals on the MOVING reference pose, so the ideal
    #   residual is not constant — it is the PD offset compensating dynamic
    #   torques, which scales with reference speed. Hence a speed-proportional
    #   allowance plus a floor that clamps quasi-static chatter. smooth_slack=0
    #   degenerates to a constant floor budget.
    #   This supersedes reward-side w_arate (same quantity, worse pathway: acts on
    #   SAMPLED actions so exploration noise adds a ~2*sigma^2 floor, and is
    #   diluted through GAE). Reduce w_arate when enabling this — do not stack.
    #
    # SPATIAL (lambda_spatial): a VARIABLE Lipschitz constraint. Bounds the local
    #   gain |d tanh(mu)| / |d obs| against a state-dependent K. Estimated by
    #   finite difference over a Gaussian ball on the NORMALIZED observation:
    #   for eps ~ N(0, s^2 I), E||f(s+eps)-f(s)||^2 = s^2 * ||J_f||_F^2 + O(s^4),
    #   i.e. the FD form IS a single-probe Hutchinson estimator of the squared
    #   Jacobian norm at one extra forward pass. Chosen over autograd.grad
    #   deliberately: double-backward forces the MultiheadAttention encoders off
    #   their fused SDPA path (~2.5-3x actor cost), the cheap
    #   autograd.grad(mu.sum(), obs) trick measures the Jacobian ROW SUM where
    #   symmetric left/right joint responses cancel, and exact-norm/Hutchinson
    #   variants have the same variance as FD while still paying double-backward.
    #   Regularizing over a finite noise ball also matches THE metric, which
    #   scores robustness under action noise 0.05.
    #   Do NOT "upgrade" this to a real gradient penalty without re-reading the above.
    lambda_smooth: float = 0.0
    smooth_slack: float = 2.5
    smooth_eps_floor: float = 0.02
    lambda_spatial: float = 0.0
    spatial_noise_std: float = 0.05
    spatial_slack: float = 2.5
    spatial_floor: float = 0.0
    # Plant quantities the budgets need. ctrl_dt = env dt * control_decimation.
    # action_scale must be set from the TRAINED env whenever a penalty is on
    # (EnvConfig's dataclass default is 0.5 while the training scripts run
    # 0.3; a silently guessed value would rescale every budget). Asserted in
    # update().
    ctrl_dt: float = 1.0 / 60.0
    action_scale: float | None = None


class PPOTrainer:
    """PPO trainer that operates on structured policy bundle data."""

    def __init__(self, model: nn.Module, cfg: PPOConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.lr = cfg.lr
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        # Param split for grad_clip_per_module: the critic MLP is
        # fully separate from the actor path, so clipping them jointly lets
        # the critic's gradient scale set the actor's step size.
        self._critic_params = [p for n, p in model.named_parameters()
                               if n.startswith("critic")]
        self._actor_params = [p for n, p in model.named_parameters()
                              if not n.startswith("critic")]

    def _adapt_lr(self, final_kl: float) -> None:
        """Per-iteration KL-adaptive LR step on the realized cumulative KL.

        ``final_kl`` is the last approx_kl of the update phase — the total
        policy movement this iteration relative to the rollout policy.
        """
        cfg = self.cfg
        if final_kl > 1.5 * cfg.target_kl:
            self.lr = max(cfg.lr_min, self.lr / 1.5)
        elif 0.0 <= final_kl < 0.5 * cfg.target_kl:
            self.lr = min(cfg.lr_max, self.lr * 1.5)
        for g in self.opt.param_groups:
            g["lr"] = self.lr

    def update(self, data: dict[str, torch.Tensor]) -> dict[str, float]:
        """Run PPO update epochs over data and return averaged stats.

        Args:
            data: dict with keys obs (M,obs_dim), history (M,T,obs_dim),
                  cmd_window (M,L,cmd_dim), critic_obs (M,priv_dim),
                  actions (M,act_dim), logp (M,), returns (M,),
                  advantages (M,), values (M,).

        Returns:
            Stats dict with at minimum: policy_loss, value_loss, entropy,
            kl, clip_frac, grad_norm, n_updates, lr.
        """
        cfg = self.cfg

        # Pull flat tensors; ensure float32.
        b_logp_old = data["logp"].float()
        b_adv = data["advantages"].float()
        b_val_old = data["values"].float()
        b_ret = data["returns"].float()
        b_act = data["actions"].float()

        # Bundle keys that feed the model.
        bundle_keys = ("obs", "history", "cmd_window", "critic_obs")

        # Advantage normalisation over the full batch.
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n = b_logp_old.shape[0]
        mb = min(cfg.mb_size, n)

        # Determine device from first bundle tensor.
        device = data["obs"].device

        stats: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "clip_frac": 0.0,
            "grad_norm": 0.0,
        }
        n_updates = 0
        early_stop = False

        # ---- KL-shock rollback snapshot (only when the guard is enabled) -----
        shock_thresh = cfg.kl_shock_factor * cfg.target_kl
        shock = False
        snapshot = None
        if cfg.kl_shock_factor > 0.0:
            snapshot = {
                "model": {
                    k: v.detach().clone()
                    for k, v in self.model.state_dict().items()
                },
                "opt": copy.deepcopy(self.opt.state_dict()),
            }

        # ---- Smoothness penalties (short-circuited when their lambda is 0) ---
        # The guards matter for bit-identity: the spatial term draws from the RNG
        # (randn_like), so entering that branch at lambda 0 would shift the stream
        # and change every subsequently sampled action.
        smooth_on = cfg.lambda_smooth > 0.0
        spatial_on = cfg.lambda_spatial > 0.0
        if smooth_on or spatial_on:
            if cfg.action_scale is None:
                raise ValueError(
                    "PPOConfig.action_scale must be set from the env when a "
                    "smoothness penalty is enabled (budgets divide by it)."
                )
            # scalar or per-joint vector -> broadcastable tensor
            a_scale = torch.as_tensor(
                cfg.action_scale, dtype=torch.float32, device=device
            )
            b_ref_qd = data["ref_qd"].float()
            # |qd_ref| * dt / scale: reference motion per control step, expressed
            # in normalized-action units. Constant w.r.t. the policy.
            ref_rate = (b_ref_qd.abs() * cfg.ctrl_dt) / a_scale
        if smooth_on:
            if cfg.n_epochs != 1:
                raise ValueError(
                    "lambda_smooth > 0 requires n_epochs == 1: prev_tanh_mu is the "
                    "behaviour policy's mean, so multi-epoch updates would anchor "
                    f"the penalty to a stale iterate (got n_epochs={cfg.n_epochs})."
                )
            b_prev_tanh_mu = data["prev_tanh_mu"].float()
            b_pair = data["pair_valid"].float()
            budget_t = ref_rate * cfg.smooth_slack + cfg.smooth_eps_floor
            for k in ("smooth/loss", "smooth/excess_mean", "smooth/frac_over",
                      "smooth/sat_frac"):
                stats[k] = 0.0
        if spatial_on:
            # Gain bound K, same variable-budget shape but a DIFFERENT dimension
            # (gain per unit normalized-obs, not rate per step) — calibrate
            # spatial_floor/spatial_slack independently of the temporal ones.
            budget_s = ref_rate * cfg.spatial_slack + cfg.spatial_floor
            for k in ("spatial/loss", "spatial/excess_mean", "spatial/frac_over",
                      "spatial/gain_p50", "spatial/gain_p95"):
                stats[k] = 0.0

        for _epoch in range(cfg.n_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, mb):
                idx = perm[start : start + mb]

                # Build minibatch bundle for the model.
                mb_bundle = {k: data[k][idx].float() for k in bundle_keys}

                # model.evaluate returns (logp, entropy, value[, mu])
                if smooth_on or spatial_on:
                    logp, entropy, value, mu = self.model.evaluate(
                        mb_bundle, b_act[idx], return_mean=True
                    )
                    tanh_mu = torch.tanh(mu)
                else:
                    logp, entropy, value = self.model.evaluate(mb_bundle, b_act[idx])

                adv = b_adv[idx]
                ratio = (logp - b_logp_old[idx]).exp()

                # Standard clipped surrogate.
                surr1 = ratio * adv
                surr2 = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * adv
                clipped = torch.min(surr1, surr2)

                # Dual-clip PPO (Ye et al. 2020): bound the negative-advantage
                # term so tail samples with huge ratios can't explode the loss.
                dual = torch.where(
                    adv < 0.0,
                    torch.max(clipped, cfg.dual_clip * adv),
                    clipped,
                )
                policy_loss = -dual.mean()

                # Value loss. The clipped form (clamp v around v_old, max of
                # two MSE) is a +-cfg.clip trust region on the VALUE; at this
                # task's return scale it bound 95% of samples, so
                # value_clip=False (the released recipe) uses plain MSE.
                if cfg.value_clip:
                    value_clipped = b_val_old[idx] + (value - b_val_old[idx]).clamp(
                        -cfg.clip, cfg.clip
                    )
                    value_loss = 0.5 * torch.max(
                        (value - b_ret[idx]) ** 2,
                        (value_clipped - b_ret[idx]) ** 2,
                    ).mean()
                else:
                    value_loss = 0.5 * ((value - b_ret[idx]) ** 2).mean()

                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_mean
                )

                # ---- Temporal smoothness: one-sided, velocity-budgeted -------
                # prev_tanh_mu / ref_qd / pair_valid are stored rollout tensors
                # (no graph), so the budget and the anchor are constants and the
                # gradient flows only through tanh(mu) -> actor + encoders.
                if smooth_on:
                    d_t = (tanh_mu - b_prev_tanh_mu[idx]).abs()
                    excess_t = (d_t - budget_t[idx]).clamp_min(0.0)
                    w = b_pair[idx].unsqueeze(-1)                  # (mb, 1)
                    denom_t = (w.sum() * tanh_mu.shape[-1]).clamp_min(1.0)
                    loss_smooth = (excess_t.pow(2) * w).sum() / denom_t
                    loss = loss + cfg.lambda_smooth * loss_smooth
                    with torch.no_grad():
                        stats["smooth/loss"] += float(loss_smooth.item())
                        stats["smooth/excess_mean"] += float(
                            ((excess_t * w).sum() / denom_t).item()
                        )
                        stats["smooth/frac_over"] += float(
                            (((excess_t > 0).float() * w).sum() / denom_t).item()
                        )
                        # Bang-bang evasion watchdog: at the tanh rails a dither
                        # produces ~zero delta and becomes penalty-invisible.
                        stats["smooth/sat_frac"] += float(
                            (tanh_mu.abs() > 0.95).float().mean().item()
                        )

                # ---- Spatial: variable Lipschitz bound on the local gain -----
                if spatial_on:
                    mu_noisy = self.model.actor_mean(
                        mb_bundle, obs_noise_std=cfg.spatial_noise_std
                    )
                    # Dividing by sigma makes the bound sigma-independent, so
                    # retuning spatial_noise_std doesn't silently rescale K.
                    gain = (torch.tanh(mu_noisy) - tanh_mu).abs() / cfg.spatial_noise_std
                    excess_s = (gain - budget_s[idx]).clamp_min(0.0)
                    loss_spatial = excess_s.pow(2).mean()
                    loss = loss + cfg.lambda_spatial * loss_spatial
                    with torch.no_grad():
                        stats["spatial/loss"] += float(loss_spatial.item())
                        stats["spatial/excess_mean"] += float(excess_s.mean().item())
                        stats["spatial/frac_over"] += float(
                            (excess_s > 0).float().mean().item()
                        )
                        # Percentiles calibrate spatial_floor (set it near p50 so
                        # the hinge clips the tail, not the whole distribution).
                        q = torch.quantile(
                            gain.flatten().float(),
                            torch.tensor([0.5, 0.95], device=gain.device),
                        )
                        stats["spatial/gain_p50"] += float(q[0].item())
                        stats["spatial/gain_p95"] += float(q[1].item())

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip_per_module:
                    # Separate budgets: the critic's (much larger) gradient no
                    # longer scales the actor's step down with it.
                    ga = nn.utils.clip_grad_norm_(
                        self._actor_params, cfg.max_grad_norm
                    ).item()
                    gc = nn.utils.clip_grad_norm_(
                        self._critic_params, cfg.max_grad_norm
                    ).item()
                    grad_norm = (ga ** 2 + gc ** 2) ** 0.5
                    stats["grad_norm_actor"] = stats.get("grad_norm_actor", 0.0) + ga
                    stats["grad_norm_critic"] = stats.get("grad_norm_critic", 0.0) + gc
                else:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.max_grad_norm
                    ).item()
                self.opt.step()

                with torch.no_grad():
                    # Schulman k3 KL approximation; use median to avoid domination
                    # by a small tail of high-ratio outliers (same logic as reference).
                    logratio = (logp - b_logp_old[idx]).clamp(-10.0, 10.0)
                    kl_per = logratio.exp() - 1.0 - logratio
                    approx_kl = kl_per.median().item()
                    clip_frac = (
                        ((ratio - 1.0).abs() > cfg.clip).float().mean().item()
                    )

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_mean.item())
                stats["kl"] += approx_kl
                stats["clip_frac"] += clip_frac
                stats["grad_norm"] += grad_norm
                n_updates += 1

                # KL-shock rollback: discard the entire iteration's update.
                # approx_kl at minibatch k reflects the movement applied by
                # minibatches 1..k-1, so the destructive step is detected one
                # minibatch late — rollback undoes it regardless. The
                # measured (shocked) kl still feeds stats and the adaptive
                # LR, which shrinks for the next iteration.
                if snapshot is not None and approx_kl > shock_thresh:
                    self.model.load_state_dict(snapshot["model"])
                    self.opt.load_state_dict(snapshot["opt"])
                    shock = True
                    early_stop = True
                    break

                # Early stop if KL divergence exceeds 1.5x target. approx_kl
                # is cumulative vs the rollout policy, so this bounds the
                # iteration's TOTAL movement (also under kl_adaptive).
                if approx_kl > 1.5 * cfg.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        # Average stats over all minibatch updates performed.
        for k in stats:
            stats[k] /= max(1, n_updates)
        if cfg.kl_adaptive and n_updates > 0:
            # Adapt on the iteration's realized (final cumulative) KL, not
            # the per-update average — see PPOConfig.kl_adaptive.
            self._adapt_lr(approx_kl)
        stats["lr"] = self.opt.param_groups[0]["lr"]
        stats["n_updates"] = float(n_updates)
        if cfg.kl_shock_factor > 0.0:
            stats["kl_shock"] = 1.0 if shock else 0.0
        return stats

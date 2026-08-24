"""Vectorised PPO rollout buffer with structured bundle storage and GAE advantage estimation."""

from __future__ import annotations

import torch

from rgmt.policy.networks import PolicyDims


class RolloutBuffer:
    """Stores `rollout_len` steps over `num_envs` envs, holding structured policy bundles,
    and computes GAE (Generalised Advantage Estimation).

    Bundle fields stored per step (T, N, ...):
        obs          (T, N, obs_dim)
        history      (T, N, hist_len, obs_dim)
        cmd_window   (T, N, cmd_len, cmd_dim)
        critic_obs   (T, N, priv_dim)

    Additional per-step fields:
        action  (T, N, act_dim)
        logp    (T, N)
        value   (T, N)
        reward  (T, N)
        done    (T, N)
    """

    def __init__(
        self,
        num_envs: int,
        rollout_len: int,
        dims: PolicyDims,
        device: str | torch.device,
        store_smooth: bool = False,
    ) -> None:
        self.num_envs = num_envs
        self.rollout_len = rollout_len
        self.dims = dims
        self.device = torch.device(device)
        # Extra per-step fields for the temporal-smoothness penalty. Nothing is
        # allocated when off, so a run with lambda_smooth=0 is untouched.
        self.store_smooth = bool(store_smooth)

        T, N = rollout_len, num_envs
        d = dims

        # Bundle fields
        self.obs = torch.zeros(T, N, d.obs_dim, device=self.device)
        self.history = torch.zeros(T, N, d.hist_len, d.obs_dim, device=self.device)
        self.cmd_window = torch.zeros(T, N, d.cmd_len, d.cmd_dim, device=self.device)
        self.critic_obs = torch.zeros(T, N, d.priv_dim, device=self.device)

        # Action / policy fields
        self.actions = torch.zeros(T, N, d.act_dim, device=self.device)
        self.logp = torch.zeros(T, N, device=self.device)
        self.values = torch.zeros(T, N, device=self.device)
        self.rewards = torch.zeros(T, N, device=self.device)
        self.dones = torch.zeros(T, N, device=self.device)

        # Smoothness fields (allocated only when enabled).
        if self.store_smooth:
            self.tanh_mu = torch.zeros(T, N, d.act_dim, device=self.device)
            self.ref_qd = torch.zeros(T, N, d.act_dim, device=self.device)

        self.ptr = 0

    def add(
        self,
        bundle: dict,
        action: torch.Tensor,
        logp: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        tanh_mu: torch.Tensor | None = None,
        ref_qd: torch.Tensor | None = None,
    ) -> None:
        """Write one step of data into the current slot and advance the pointer.

        ``tanh_mu``/``ref_qd`` are keyword-optional so the existing positional call
        sites (train.py, scripts/profile_throughput.py, tests/test_rollout.py) are
        unaffected.
        """
        t = self.ptr
        self.obs[t] = bundle["obs"]
        self.history[t] = bundle["history"]
        self.cmd_window[t] = bundle["cmd_window"]
        self.critic_obs[t] = bundle["critic_obs"]
        self.actions[t] = action
        self.logp[t] = logp
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done.float() if done.dtype == torch.bool else done
        if self.store_smooth:
            if tanh_mu is not None:
                self.tanh_mu[t] = tanh_mu
            if ref_qd is not None:
                self.ref_qd[t] = ref_qd
        self.ptr += 1

    def clear(self) -> None:
        """Reset the write pointer (data is overwritten on next rollout)."""
        self.ptr = 0

    def compute_gae(
        self,
        last_value: torch.Tensor,
        gamma: float,
        lam: float,
    ) -> dict:
        """Compute GAE advantages and returns, then return flattened (T*N, ...) tensors.

        Returns a dict with keys:
            obs, history, cmd_window, critic_obs  -- bundle fields flattened to (T*N, ...)
            actions, logp, values                 -- flattened (T*N, ...) policy fields
            advantages, returns                   -- flattened (T*N,) GAE outputs
        """
        T = self.rollout_len
        adv = torch.zeros_like(self.rewards)  # (T, N)
        gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_v * non_terminal - self.values[t]
            gae = delta + gamma * lam * non_terminal * gae
            adv[t] = gae

        returns = adv + self.values  # (T, N)

        # Flatten (T, N, ...) -> (T*N, ...)
        def flat(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(-1, *x.shape[2:])

        out = dict(
            obs=flat(self.obs),
            history=flat(self.history),
            cmd_window=flat(self.cmd_window),
            critic_obs=flat(self.critic_obs),
            actions=flat(self.actions),
            logp=flat(self.logp),
            values=flat(self.values),
            advantages=flat(adv),
            returns=flat(returns),
        )

        if self.store_smooth:
            # Pair sample t with its predecessor t-1 in the SAME env. The flat
            # index is m = t*N + e, so a time-shift here stays aligned with every
            # other key under the shared minibatch `idx`.
            prev = torch.zeros_like(self.tanh_mu)
            prev[1:] = self.tanh_mu[:-1]
            # Valid pairs only: t=0 has no in-buffer predecessor (~3% of pairs at
            # rollout_len=32), and done[t-1] means the env auto-reset inside step,
            # so action t belongs to a fresh episode. Use the FULL done — truncated
            # episodes (clip end / timeout) also reset, and prev_action/filt_res are
            # zeroed there, so the pair is discontinuous regardless of cause.
            valid = torch.zeros_like(self.dones)
            valid[1:] = 1.0 - self.dones[:-1]
            out["prev_tanh_mu"] = flat(prev)
            out["ref_qd"] = flat(self.ref_qd)
            out["pair_valid"] = flat(valid)

        return out

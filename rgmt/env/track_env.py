"""TrackEnv — vectorized RGMT motion-tracking environment.

Composes the already-built pieces into the policy input bundle:
  * ``NewtonSim``          (rgmt.env.sim)    — physics + robot state / keypoints
  * ``MotionRef``          (rgmt.data.motion) — reference command / keypoints
  * ``compute_reward``     (rgmt.env.reward)  — keypoint tracking reward

This module *composes*; it does not re-derive physics or motion math. All
shape contracts below are exact because the PPO loop, the runner, and the
config construct and consume this interface verbatim.

Bundle (per step / reset), keys EXACT — must match ``RGMTActorCritic``:
    obs        (N, obs_dim)       obs_dim = 98, or 103 with cfg.drift_obs_proprio
    history    (N, 10, obs_dim)
    cmd_window (N, 21, cmd_dim)   cmd_dim = 55, or 60 with cfg.drift_obs
    critic_obs (N, priv_dim)   where priv_dim = obs_dim + cmd_dim + (1 + 3*Kp + 3)
                               = 187 for Kp=10 at the paper's 98/55

Observation ``o_t`` (N,98):
    [ gravity_projection(base_quat)      (3)
      base_ang_vel                       (3)
      encode_angles(joint_q[actuated])  (46)
      joint_qd[actuated]                (23)
      prev_action                       (23) ]

Critic ``s_t`` (N, priv_dim) — paper eq.4, all noise-free:
    [ o_t (current proprio obs, no noise)               (98)
      g_t_clean (noise-free command at current frame)   (55)
      o_priv:
        h_ref (motion ref base height at idx)            (1)
        robot root-relative keypoints (sim, flattened) (3*Kp)
        base_lin_vel (sim)                               (3) ]

Command noise (Table II, training only, scaled by ``cfg.noise_level``):
    v_ref [0:3]  += U[-0.5,0.5] on x/y, U[-0.2,0.2] on z
    w_ref [3:6]  += U[-0.52,0.52]
    g_ref [6:9]  += U[-0.05,0.05]
    q_ref [9:55] : the FAITHFUL form perturbs the 23 joint *angles* by
                   U[-0.1,0.1] *before* the cos/sin encoding. We therefore
                   do NOT add noise to the encoded cos/sin block directly;
                   instead we re-encode noised angles via ``_qblock_noisy``
                   (see below) so the cos^2+sin^2=1 manifold is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import torch

from rgmt.env.sim import NewtonSim
from rgmt.env.domain_rand import DomainRand, DRConfig
from rgmt.data.corpus import MotionCorpus
from rgmt.env.reward import RewardState, RewardRef, RewardWeights, compute_reward
from rgmt.utils.rotation import (
    gravity_projection,
    encode_angles,
    quat_to_matrix,
    yaw_from_quat,
)


def _clip_aware_grid(
    idx: torch.Tensor, L: int, corpus: "MotionCorpus"
) -> torch.Tensor:
    """Build a (N, 2L+1) LongTensor of frame indices clamped within each env's clip.

    Mirrors ``MotionCorpus.command_window`` clamping exactly: the window for
    env i is centred on ``idx[i]`` but never steps outside
    ``[clip_start[cid], clip_end[cid]]`` for the clip that owns that frame.

    Args:
        idx:    (N,) LongTensor (any device) — current frame indices.
        L:      command-window half-width (window size = 2L+1).
        corpus: ``MotionCorpus``; the grid is built on its storage device
                (CPU or GPU), so it can index ``corpus.joint_q`` directly.

    Returns:
        (N, 2L+1) LongTensor on the corpus storage device.
    """
    dev = corpus.storage_device
    i = idx.to(dev)
    cid = corpus.frame_clip_id[i]
    lo = corpus.clip_start[cid].unsqueeze(1)   # (N, 1)
    hi = corpus.clip_end[cid].unsqueeze(1)     # (N, 1)
    offs = torch.arange(-L, L + 1, device=dev)  # (2L+1,)
    grid = i.unsqueeze(1) + offs.unsqueeze(0)   # (N, 2L+1)
    grid = torch.maximum(grid, lo)
    grid = torch.minimum(grid, hi)
    return grid


def compute_fallen(base_z, up_z, head_z, z_fall, up_dot_min, head_z_min):
    """Instability/fallen flag (paper §II-D, three criteria).

    Args:
        base_z:      pelvis height, (N,).
        up_z:        body up-axis z-component = R[:,2,2] (cos of tilt from vertical), (N,).
        head_z:      head keypoint height, (N,), or None to skip that criterion.
        z_fall:      pelvis-height threshold (below -> fallen).
        up_dot_min:  body-up.z threshold (below -> tilted past the limit).
        head_z_min:  head-height threshold (below -> abnormally low key body part).
    Returns:
        BoolTensor (N,): fallen if pelvis too low, tilted past the limit, or (when
        head_z is given) the head is abnormally low.
    """
    fallen = (base_z < z_fall) | (up_z < up_dot_min)
    if head_z is not None:
        fallen = fallen | (head_z < head_z_min)
    return fallen


def _default_noise() -> dict:
    """Table II command-noise amplitudes (uniform half-widths)."""
    return {
        "v_xy": 0.5,   # v_ref x/y  ~ U[-0.5, 0.5]
        "v_z": 0.2,    # v_ref z    ~ U[-0.2, 0.2]
        "w": 0.52,     # w_ref      ~ U[-0.52, 0.52]
        "g": 0.05,     # g_ref      ~ U[-0.05, 0.05]
        "q": 0.1,      # joint angle (pre-encode) ~ U[-0.1, 0.1]
        # Drift-feedback channels (only used when EnvConfig.drift_obs):
        "p": 0.05,     # drift dp (m, per axis)        ~ U[-0.05, 0.05]
        "yaw": 0.05,   # drift heading error (rad, pre-encode) ~ U[-0.05, 0.05]
    }


@dataclass
class EnvConfig:
    """Configuration for :class:`TrackEnv`.

    Field set is EXACT; the training loop constructs this with exactly these names.
    """

    num_envs: int
    # scalar (uniform) or a per-actuated-joint vector of length 23 in
    # ASIMOV_ACTUATED_JOINT_NAMES order (see rgmt.env.gains.leg_weighted_gains).
    kp: Union[float, Sequence[float]] = 100.0
    kd: Union[float, Sequence[float]] = 5.0
    # EMA low-pass on the commanded residual: applied = a*new + (1-a)*prev.
    # 1.0 = off (back-compat). At 0.7/60Hz the -3dB point is ~11 Hz: above the
    # 2-5 Hz balance band, below the chatter band (a per-joint diagnostic
    # showed residual jitter = hip/ankle balance chatter). Policy trains
    # THROUGH it.
    action_filter_alpha: float = 1.0
    control_decimation: int = 1
    dt: float = 1.0 / 60.0
    action_scale: float = 0.5
    K: int = 9                       # reserved (history-encoder kernel); not used here
    L: int = 10                      # command-window half-width -> 2L+1 = 21
    foot_friction: float = 0.75   # GROUND-PLANE mu only. A no-op for foot-ground
                                  # contact: the MJCF foot spheres carry
                                  # priority=1, friction=0.6, and mujoco-warp
                                  # takes friction solely from the higher-
                                  # priority geom. Randomize dr.friction_range
                                  # (foot-shape mu) instead. Kept for
                                  # back-compat with recorded configs.
    # None = uncapped (builder default 1e6) | "urdf" (per-joint
    # <limit effort> from the URDF) | explicit length-23 list (Nm, ASIMOV order).
    # Part of the PLANT: eval must use the value the policy trained under.
    effort_limits: Union[None, str, list] = None
    episode_len: int = 1000
    z_fall: float = 0.12          # pelvis sits ~0.08 on the ground -> <0.12 = collapsed
    up_dot_min: float = 0.0       # body-up.z < 0 -> tilted past horizontal
    head_z_min: float = 0.3       # head keypoint abnormally low (paper's 3rd criterion)
    joint_err_done: float = 100.0
    root_err_done: float = 0.25    # SQUARED planar err (m^2) -> 0.5 m radius (was 1.0)
    z_dev_done: float = 0.2        # |root_z - ref_z| > 0.2 m = tracking failure (paper's criterion)
    noise: dict = field(default_factory=_default_noise)
    noise_level: float = 1.0
    keypoint_links: list = field(default_factory=list)
    reward: Optional[dict] = None
    # --- Fall-recovery curriculum (paper §II-D) ---
    # Fraction of envs designated as recovery envs (fixed Bernoulli mask at init).
    recovery_fraction: float = 0.15
    # Duration (seconds) of the recovery assist window.
    recovery_window_s: float = 3.0
    # Max upward assist force (N) for recovery envs, sampled U[0, max] * anneal
    # scale per iteration. Paper: 200 N on the ~35 kg G1 (~0.6x body weight);
    # Asimov is 32.15 kg so the same 200 N keeps the same relative assist.
    assist_force_max: float = 200.0
    # Inside the recovery grace window, pay BODY-LOCAL tracking (the
    # root-relative keypoint kernel r_rel) instead of the world-frame
    # product. Earlier recovery experiments identified this as the structural
    # reason the paper's recovery curriculum did not transfer: their reward
    # is body-local, so a fallen/displaced robot still receives dense pose
    # gradient; our world-frame kernels are ~0 there, leaving only sparse
    # alive/fall signals. Requires recovery_fraction > 0 to matter.
    recovery_local_reward: bool = False
    # Do not fine the fallen state inside the recovery grace window.
    # Earlier runs showed recovery envs sitting fallen for the whole 3 s window at
    # -w_fall/step — a return sink that dominated value targets (value_loss
    # ~34, clip_frac 0.45) and collapsed tracking success 6.9% -> 0.2%. The
    # paper's body-local reward keeps paying a fallen robot for pose matching;
    # shielding fall_pen (and granting alive) in-window is our equivalent, so
    # the get-up incentive comes from the tracking kernels + post-window
    # survival instead of a near-constant per-step fine.
    recovery_reward_shield: bool = False
    # Recovery spawn distribution knobs. Defaults reproduce the paper's
    # full range (tilt up to pi = fully inverted, z down to near-ground).
    # Milder settings target the NEAR-FALL basin our death states show
    # (median death: z 0.21 below ref, tilt 34 deg) when full get-up proves
    # unlearnable under our reward.
    recovery_spawn_tilt_max: float = 3.14159265
    recovery_spawn_z_min: float = 0.10
    recovery_spawn_z_max: float = 0.45
    # --- drift feedback ---
    # Append per-command-frame drift features [dp in the robot heading frame
    # (3), heading error cos/sin (2)] -> cmd_dim 55 -> 60. The paper's g_t is
    # yaw-invariant by design and carries NO positional feedback, so planar +
    # yaw drift is unobservable to the policy and integrates open-loop over
    # long horizons (a ~210 mm eval floor in early runs). These features close
    # that loop while staying invariant to rigid world transforms of the
    # (robot, reference) pair.
    drift_obs: bool = False
    # --- drift feedback in the PROPRIO path ---
    # Append the CURRENT-frame drift features (same 5 values, noise-free) to
    # o_t -> obs_dim 98 -> 103, so they reach the actor directly and the
    # history encoder (drift TREND over 10 steps becomes visible). The
    # command-window drift alone taught orientation control but not
    # XY; hypothesis: the cross-attention over 21 command frames dilutes the
    # positional signal, and XY correction needs the trend, not a snapshot.
    drift_obs_proprio: bool = False
    # --- push-perturbation training (stumble robustness) ---
    # Death-state diagnostics showed 96% of z-dev terminations are near-falls
    # (robot LOW and tilted ~34 deg at death) on kinematically ordinary clips
    # — a balance-robustness gap, not height tracking. Random horizontal
    # pelvis pushes during training (train=True only) are the standard
    # disturbance-rejection lever. 0 disables (default; contracts unchanged).
    # --- N2: failure-weighted RSI (hard-example mining) ---
    # Path to a JSON {clip_name: relative_weight}; RSI start frames are then
    # drawn clip-weighted (uniform within a clip) instead of corpus-uniform.
    # Weights are produced offline from a robust-eval --dump-clips file.
    sampling_weights_json: Optional[str] = None
    push_force_max: float = 0.0      # N; per-push magnitude ~ U[0.3, 1.0] * max
    push_interval_s: float = 3.0     # mean seconds between pushes per env
    push_duration_s: float = 0.15    # seconds each push lasts
    # --- domain randomization (sim2real) ---
    # Dict of DRConfig kwargs (rgmt/env/domain_rand.py), e.g.
    #   dr: {friction_range: [0.4, 1.0], mass_scale_range: [0.9, 1.1],
    #        kp_scale_range: [0.9, 1.1], kd_scale_range: [0.9, 1.1],
    #        effort_scale_range: [0.8, 1.2], privileged: true}
    # None = off (bit-identical to the non-randomized plant). Train-gated: eval envs
    # never randomize. privileged appends 6 normalized params to critic_obs
    # (changes priv_dim -> old ckpts' critics won't load).
    dr: Optional[dict] = None


class TrackEnv:
    """Vectorized RGMT tracking environment over ``num_envs`` worlds."""

    def __init__(
        self,
        cfg: EnvConfig,
        motion: MotionCorpus,
        device: str | torch.device = "cuda:0",
        *,
        train: bool = True,
    ) -> None:
        self.cfg = cfg
        self.motion = motion
        self.device = torch.device(device)
        self.train = train

        self.N = int(cfg.num_envs)
        self.L = int(cfg.L)
        self.Kp = len(cfg.keypoint_links)
        self.action_scale = float(cfg.action_scale)
        # Head keypoint index for the "abnormally low key body part" criterion.
        self._head_kp_idx = (cfg.keypoint_links.index("neck_pitch_link")
                             if "neck_pitch_link" in cfg.keypoint_links else None)

        # Command feature width: 55 (paper) or 60 with drift feedback.
        self.cmd_dim = 60 if cfg.drift_obs else 55
        # Proprio width: 98 (paper) or 103 with proprio drift feedback.
        self.obs_dim = 103 if cfg.drift_obs_proprio else 98
        # priv_dim = o_t(obs_dim) + g_t_clean(cmd_dim) + o_priv[h_ref(1) + kp_rel(3*Kp) + v_base(3)]
        # Paper eq.4: s_t = [o_t, g_t, o_priv], all noise-free. For Kp=10, 98/55: 98+55+34=187.
        self.priv_dim = self.obs_dim + self.cmd_dim + (1 + 3 * self.Kp + 3)
        # Privileged DR: the critic additionally sees the sampled
        # dynamics params (computed from cfg, not self.dr, so the width is
        # consistent even for eval-built envs with train=False).
        self._dr_privileged = bool(cfg.dr and dict(cfg.dr).get("privileged"))
        if self._dr_privileged:
            self.priv_dim += DomainRand.DR_OBS_DIM

        # Older configs may carry a noise dict without the drift channels;
        # merge over defaults so lookups are always safe.
        self.cfg.noise = {**_default_noise(), **dict(cfg.noise or {})}

        # ---- Build sim ----------------------------------------------------
        # Resolve the effort-limit sentinel: "urdf" -> per-joint datasheet Nm.
        _eff = cfg.effort_limits
        if isinstance(_eff, str):
            if _eff != "urdf":
                raise ValueError(f"effort_limits: unknown sentinel {_eff!r}")
            from rgmt.env.gains import urdf_effort_limits
            _eff = urdf_effort_limits()
        self.sim = NewtonSim(
            self.N,
            kp=cfg.kp,
            kd=cfg.kd,
            control_decimation=cfg.control_decimation,
            dt=cfg.dt,
            foot_friction=cfg.foot_friction,
            keypoint_links=cfg.keypoint_links,
            device=self.device,
            effort_limit=_eff,
        )
        # actuated columns into the 25-wide hinge buffer (LongTensor[23]).
        self.actuated_idx = self.sim.actuated_idx.to(self.device)
        self.n_act = int(self.actuated_idx.numel())  # 23

        # ---- Domain randomization (train-only; see domain_rand.py) --------
        # Draws are episode-consistent: reset envs are marked pending and
        # resampled together at the next iteration boundary (resample_dr(),
        # called from the train loop next to set_assist_scale) so the solver
        # sees exactly ONE notify_model_changed per iteration.
        self.dr: Optional[DomainRand] = None
        if cfg.dr and train:
            self.dr = DomainRand(DRConfig(**dict(cfg.dr)), self.N, self.device)
            all_ids = torch.arange(self.N, device=self.device)
            self.dr.sample(all_ids)
            self.dr.write_to_sim(self.sim)
            self.sim.apply_dynamics_changes()
        self._dr_pending = torch.zeros(self.N, dtype=torch.bool, device=self.device)

        # ---- Per-env episode bookkeeping ---------------------------------
        self.idx = torch.zeros(self.N, dtype=torch.long, device=self.device)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=self.device)
        self.prev_action = torch.zeros(self.N, self.n_act, device=self.device)
        self.action_filter_alpha = float(cfg.action_filter_alpha)
        # low-pass filter state: the last APPLIED residual (post-filter)
        self.filt_res = torch.zeros(self.N, self.n_act, device=self.device)
        # history ring buffer (N, 10, obs_dim); index 0 = oldest, -1 = newest.
        self.history = torch.zeros(self.N, 10, self.obs_dim, device=self.device)

        # reward weights (overridable via cfg.reward dict).
        if cfg.reward:
            self.reward_weights = RewardWeights(**cfg.reward)
        else:
            self.reward_weights = RewardWeights()

        # max future lookahead needed for RSI sampling (command window + a step).
        self._max_lookahead = self.L + 1

        if cfg.sampling_weights_json:
            import json as _json
            with open(cfg.sampling_weights_json) as f:
                stats = self.motion.set_clip_sampling_weights(_json.load(f))
            print(f"[TrackEnv] clip sampling weights: {stats['matched']}/"
                  f"{stats['total_clips']} clips weighted, boosted mass share "
                  f"{stats['boosted_mass_share']:.3f}")

        # --- Fall-recovery curriculum (paper §II-D) ---
        # Fixed Bernoulli(recovery_fraction) mask sampled once at init.
        # Simpler and more reproducible than per-reset sampling; the recovery assist adds
        # the assist force + delayed termination that uses recovery_window_s.
        self.is_recovery: torch.Tensor = (
            torch.rand(self.N, device=self.device) < cfg.recovery_fraction
        )  # BoolTensor[N]

        # Annealed upward assist force (world +z), applied to the
        # pelvis via the sim. Per-env magnitude is resampled U[0,200]*scale each
        # time set_assist_scale() is called (i.e. once per training iteration),
        # held fixed across the rollout. Zero for non-recovery envs.
        self._assist_force: torch.Tensor = torch.zeros(
            self.N, 3, device=self.device
        )  # (N, 3) world-frame force pushed to sim.set_external_force

        # Delayed-termination recovery window. Within this many env steps after
        # a recovery env's reset, instability/fallen does NOT terminate the
        # episode (the policy gets time to stand up). The per-env timer counts
        # steps elapsed since reset; reset to 0 in reset_idx.
        # One env step advances control_decimation substeps of dt each.
        step_dt = cfg.dt * max(int(cfg.control_decimation), 1)
        self._recovery_window_steps: int = int(
            round(cfg.recovery_window_s / step_dt)
        )
        self._recovery_steps_elapsed: torch.Tensor = torch.zeros(
            self.N, dtype=torch.long, device=self.device
        )

        # --- push-perturbation state ---
        step_dt_push = cfg.dt * max(int(cfg.control_decimation), 1)
        self._push_interval_steps = max(int(round(cfg.push_interval_s / step_dt_push)), 1)
        self._push_duration_steps = max(int(round(cfg.push_duration_s / step_dt_push)), 1)
        # steps until each env's next push starts (randomized phase)...
        self._push_timer = torch.randint(
            1, self._push_interval_steps + 1, (self.N,), device=self.device
        )
        # ...and steps remaining in the currently-active push (0 = none).
        self._push_remaining = torch.zeros(self.N, dtype=torch.long, device=self.device)
        self._push_force = torch.zeros(self.N, 3, device=self.device)

    # ----------------------------------------------------------------------
    # Observation / command / privileged construction
    # ----------------------------------------------------------------------

    def _build_obs(self) -> torch.Tensor:
        """Assemble o_t (N, obs_dim) from current sim state + prev_action.

        98 base features; +5 current-frame drift features (noise-free, like
        the rest of o_t) when cfg.drift_obs_proprio.
        """
        g_proj = gravity_projection(self.sim.base_quat)                  # (N, 3)
        omega = self.sim.base_ang_vel                                    # (N, 3)
        q_act = self.sim.joint_q[:, self.actuated_idx]                   # (N, 23)
        qd_act = self.sim.joint_qd[:, self.actuated_idx]                 # (N, 23)
        q_enc = encode_angles(q_act)                                     # (N, 46)
        parts = [g_proj, omega, q_enc, qd_act, self.prev_action]
        if self.cfg.drift_obs_proprio:
            parts.append(self._drift_features(self.idx.unsqueeze(1), noisy=False).squeeze(1))
        return torch.cat(parts, dim=-1)

    def _qblock_noisy(self, ref_angles: torch.Tensor) -> torch.Tensor:
        """Re-encode reference joint angles with pre-encode uniform noise.

        ``ref_angles`` is (..., 23). Adds U[-q, q]*noise_level to the raw
        angles THEN encodes, faithfully matching Table II (noise lives on the
        angle, not on cos/sin). Returns (..., 46).
        """
        amp = self.cfg.noise["q"] * self.cfg.noise_level
        noise = (torch.rand_like(ref_angles) * 2.0 - 1.0) * amp
        return encode_angles(ref_angles + noise)

    def _drift_features(self, grid: torch.Tensor, noisy: bool) -> torch.Tensor:
        """Drift-feedback features (N, S, 5) for reference frames ``grid``.

        [dp_x, dp_y, dp_z, cos(dyaw), sin(dyaw)] where dp is the world
        displacement from the robot's CURRENT root to the reference root at
        each window frame, rotated into the robot's heading (yaw-only) frame,
        and dyaw = reference heading - robot heading. Invariant to a rigid
        yaw/translation of the (robot, reference) pair; this is exactly the
        positional feedback the yaw-invariant g_t lacks. Noise (train only):
        U[-p,p] per dp axis, U[-yaw,yaw] on dyaw BEFORE the cos/sin encode.
        """
        flat = grid.reshape(-1).to(self.motion.base_pos.device)
        d = self.device
        ref_pos = self.motion.base_pos[flat].to(d).reshape(*grid.shape, 3)    # (N,S,3)
        ref_quat = self.motion.base_quat[flat].to(d).reshape(*grid.shape, 4)  # (N,S,4)
        dp = ref_pos - self.sim.base_pos[:, None, :]                          # (N,S,3) world
        yaw_r = yaw_from_quat(self.sim.base_quat)                             # (N,)
        dyaw = yaw_from_quat(ref_quat) - yaw_r[:, None]                       # (N,S)
        if noisy:
            nl = self.cfg.noise_level
            dp = dp + (torch.rand_like(dp) * 2.0 - 1.0) * (self.cfg.noise["p"] * nl)
            dyaw = dyaw + (torch.rand_like(dyaw) * 2.0 - 1.0) * (self.cfg.noise["yaw"] * nl)
        cy, sy = torch.cos(yaw_r)[:, None], torch.sin(yaw_r)[:, None]
        dx = cy * dp[..., 0] + sy * dp[..., 1]
        dy = -sy * dp[..., 0] + cy * dp[..., 1]
        return torch.stack(
            [dx, dy, dp[..., 2], torch.cos(dyaw), torch.sin(dyaw)], dim=-1)

    def _command_window_noisy(self, idx: torch.Tensor) -> torch.Tensor:
        """Centered command window (N, 21, cmd_dim), with Table II noise if training.

        Clean path delegates to ``motion.command_window``. The noisy path
        rebuilds the q-encoded block from noised *angles* (not noised cos/sin)
        and adds uniform noise to the raw v/w/g slices. With ``drift_obs``,
        the 5 drift-feedback features are appended per frame (55 -> 60).
        """
        win = self.motion.command_window(idx, self.L)  # (N, 21, 55)
        noisy = self.train and self.cfg.noise_level != 0.0
        grid = None
        if noisy:
            win = win.clone()
            nl = self.cfg.noise_level
            n = self.cfg.noise

            def _u(shape, amp):
                return (torch.rand(shape, device=win.device) * 2.0 - 1.0) * amp

            N, S = win.shape[0], win.shape[1]
            # v_ref [0:3] — x/y vs z amplitudes.
            win[..., 0:2] += _u((N, S, 2), n["v_xy"] * nl)
            win[..., 2:3] += _u((N, S, 1), n["v_z"] * nl)
            # w_ref [3:6]
            win[..., 3:6] += _u((N, S, 3), n["w"] * nl)
            # g_ref [6:9]
            win[..., 6:9] += _u((N, S, 3), n["g"] * nl)
            # q_ref [9:55] — perturb angles BEFORE encoding.
            # Recover the raw reference angles for every (env, window-offset)
            # frame using per-env clip-aware clamping so the window never
            # bleeds across clip boundaries (matching
            # MotionCorpus.command_window behaviour).
            grid = _clip_aware_grid(idx, self.L, self.motion)    # (N, S) on corpus storage device
            ref_angles = self.motion.joint_q[grid.reshape(-1)].to(win.device)  # (N*S, 23)
            win[..., 9:55] = self._qblock_noisy(ref_angles).reshape(N, S, 46)
        if self.cfg.drift_obs:
            if grid is None:
                grid = _clip_aware_grid(idx, self.L, self.motion)
            win = torch.cat([win, self._drift_features(grid, noisy)], dim=-1)
        return win

    def _critic_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Critic input s_t (N, priv_dim) — paper eq.4, all noise-free.

        s_t = cat([o_t, g_t_clean, o_priv]) where:
          o_t       = current proprioceptive obs (N, 98) — passed in, no noise.
          g_t_clean = noise-free command at current frame (N, 55).
          o_priv    = [h_ref(N,1), kp_rel(N,3*Kp), v_base(N,3)].
        """
        # g_t_clean: call command_at directly (noise-free path).
        g_t_clean = self.motion.command_at(self.idx)                     # (N, 55)
        if self.cfg.drift_obs:
            drift = self._drift_features(
                self.idx.unsqueeze(1), noisy=False).squeeze(1)           # (N, 5)
            g_t_clean = torch.cat([g_t_clean, drift], dim=-1)            # (N, 60)
        # o_priv components.
        h_ref = self.motion.at(self.idx)["base_pos"][:, 2:3]            # (N, 1)
        # Robot root-relative keypoints from the sim.
        kp_rel = self.sim.keypoint_pos - self.sim.base_pos[:, None, :]  # (N, Kp, 3)
        kp_rel = kp_rel.reshape(self.N, 3 * self.Kp)                    # (N, 3*Kp)
        v_base = self.sim.base_lin_vel                                   # (N, 3)
        parts = [obs, g_t_clean, h_ref, kp_rel, v_base]
        if self._dr_privileged:
            # Sampled dynamics params, pre-normalized to [-1, 1]. Eval envs
            # (train=False -> self.dr is None) emit zeros: harmless, since
            # eval uses act_inference only and never reads the critic.
            parts.append(self.dr.obs() if self.dr is not None
                         else torch.zeros(self.N, DomainRand.DR_OBS_DIM,
                                          device=self.device))
        critic_obs = torch.cat(parts, dim=-1)
        assert critic_obs.shape[-1] == self.priv_dim, (
            f"critic_obs last dim {critic_obs.shape[-1]} != priv_dim {self.priv_dim}"
        )
        return critic_obs

    def _bundle(self) -> dict:
        obs = self._build_obs()
        return {
            "obs": obs,
            # .clone() is LOAD-BEARING: self.history is a ring buffer mutated
            # in place by step()/reset_idx(). Without the copy, a bundle held
            # across env.step() (the training loop stores it in the rollout
            # buffer AFTER stepping; info["terminal_obs"] outlives the
            # auto-reset) silently morphs into the NEXT step's — or next
            # episode's — history, so PPO re-evaluated logp on inputs the
            # rollout policy never saw (a structural phantom-KL floor,
            # found in the 2026-07-13 audit). obs/cmd_window/critic_obs are
            # freshly allocated each call and need no copy.
            "history": self.history.clone(),
            "cmd_window": self._command_window_noisy(self.idx),
            "critic_obs": self._critic_obs(obs),
        }

    # ----------------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------------

    @staticmethod
    def _axis_angle_to_quat_xyzw(
        axis: torch.Tensor, angle: torch.Tensor
    ) -> torch.Tensor:
        """Convert axis-angle to xyzw quaternion.

        axis:  (..., 3) unit vectors
        angle: (...,) angles in radians
        Returns (..., 4) xyzw.
        """
        half = angle / 2.0
        s = torch.sin(half).unsqueeze(-1)
        w = torch.cos(half).unsqueeze(-1)
        return torch.cat([axis * s, w], dim=-1)

    def _write_recovery_frame(self, env_ids: torch.Tensor) -> None:
        """Reset recovery envs to a randomized UNSTABLE pose near their reference.

        Called AFTER ``reset_idx`` has assigned the new RSI phase, so
        ``self.idx[env_ids]`` is the motion frame these envs must resume
        tracking. The unstable pose is centred on the REFERENCE root at that
        frame: the paper's reward/commands are body-local, so world
        placement is irrelevant there — but our recipe tracks world-frame
        keypoints and terminates on root_err_done, so a spawn at the origin
        gets ~zero tracking gradient and dies by rootxy at exactly window
        expiry. Spawning inside the survivable radius makes stand-up-AND-
        resume-tracking an experienceable trajectory.

        Pose distribution (all uniform):
          base_pos:  xy = ref_xy + U[-0.25, 0.25] (inside the 0.5 m
                     root_err_done radius), z from [0.10, 0.45] (low/fallen)
          base_quat: random tilt via axis-angle: axis∈S², angle∈[0, π] to cover
                     full range including fallen; normalised to unit quaternion
          base_lin_vel: zero (starts from rest; avoids wild numerical blowups)
          base_ang_vel: U[-0.5, 0.5]³
          joint_q:   ref angles at THIS env's phase idx + U[-0.5, 0.5] noise
          joint_qd:  zero (avoids NaN from large joint velocity transients)

        The height range [0.10, 0.45] keeps the base above the ground plane
        so Newton does not start with interpenetration (which can produce NaN
        forces). A typical standing height is ~0.63 m; values < 0.45 m place
        the robot in a clearly non-upright regime.
        """
        M = int(env_ids.numel())
        if M == 0:
            return

        dev = self.device
        ref = self.motion.at(self.idx[env_ids])

        # --- base position: reference-centred xy, low / non-upright height ---
        ref_xy = ref["base_pos"][:, :2].to(dev)
        xy = ref_xy + (torch.rand(M, 2, device=dev) * 2.0 - 1.0) * 0.25
        z_lo, z_hi = self.cfg.recovery_spawn_z_min, self.cfg.recovery_spawn_z_max
        z = torch.rand(M, 1, device=dev) * (z_hi - z_lo) + z_lo
        base_pos = torch.cat([xy, z], dim=-1)             # (M, 3)

        # --- base orientation: random axis-angle up to recovery_spawn_tilt_max ---
        # Sample axis uniformly on S² via normal-then-normalise.
        axis = torch.randn(M, 3, device=dev)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        # Angle up to pi covers the full range including fully fallen.
        angle = torch.rand(M, device=dev) * self.cfg.recovery_spawn_tilt_max
        base_quat = self._axis_angle_to_quat_xyzw(axis, angle)   # (M, 4) xyzw

        # --- velocities ---
        base_lin_vel = torch.zeros(M, 3, device=dev)
        base_ang_vel = (torch.rand(M, 3, device=dev) * 2.0 - 1.0) * 0.5

        # --- joint angles: this env's reference frame + large noise ---
        ref_q = ref["joint_q"].to(dev)                             # (M, 23)
        noise = (torch.rand(M, 23, device=dev) * 2.0 - 1.0) * 0.5
        act_q = ref_q + noise                                      # (M, 23)

        hinge_q = torch.zeros(M, 25, device=dev, dtype=torch.float32)
        hinge_qd = torch.zeros(M, 25, device=dev, dtype=torch.float32)
        hinge_q[:, self.actuated_idx] = act_q.to(torch.float32)

        self.sim.reset_idx(
            env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel,
            hinge_q, hinge_qd,
        )

    def _write_ref_frame(self, env_ids: torch.Tensor, new_idx: torch.Tensor) -> None:
        """Reset given envs to the motion reference frame at ``new_idx`` (RSI).

        Scatters the 23 actuated ref joint angles into the 25-wide hinge
        buffer at ``actuated_idx`` (passive neck columns left at 0).
        """
        ref = self.motion.at(new_idx)
        M = int(env_ids.numel())
        hinge_q = torch.zeros(M, 25, device=self.device, dtype=torch.float32)
        hinge_qd = torch.zeros(M, 25, device=self.device, dtype=torch.float32)
        hinge_q[:, self.actuated_idx] = ref["joint_q"].to(self.device)
        hinge_qd[:, self.actuated_idx] = ref["joint_qd"].to(self.device)
        self.sim.reset_idx(
            env_ids,
            ref["base_pos"],
            ref["base_quat"],
            ref["base_lin_vel"],
            ref["base_ang_vel"],
            hinge_q,
            hinge_qd,
        )

    def reset_idx(self, env_ids: torch.Tensor, *, rsi: bool = True) -> None:
        """Reset the given envs (RSI: random start index) and refill history.

        For envs in ``self.is_recovery``, writes a randomized unstable pose
        instead of the clean RSI reference frame (paper §II-D).  The motion
        phase index (self.idx) is still set to a valid RSI index so the command
        window and critic obs remain meaningful.
        """
        env_ids = env_ids.to(self.device, dtype=torch.long)
        M = int(env_ids.numel())
        if M == 0:
            return
        if rsi:
            new_idx = self.motion.sample_index(M, self._max_lookahead).to(self.device)
        else:
            new_idx = torch.zeros(M, dtype=torch.long, device=self.device)

        self.idx[env_ids] = new_idx
        self.ep_step[env_ids] = 0
        self.prev_action[env_ids] = 0.0
        self.filt_res[env_ids] = 0.0
        # Restart the delayed-termination grace window for these envs.
        self._recovery_steps_elapsed[env_ids] = 0
        # Mark for a fresh DR draw at the next iteration boundary (resample_dr).
        self._dr_pending[env_ids] = True

        # --- Split into recovery vs. normal envs ---
        is_rec_mask = self.is_recovery[env_ids]          # (M,) bool
        rec_ids = env_ids[is_rec_mask]
        norm_ids = env_ids[~is_rec_mask]
        norm_new_idx = new_idx[~is_rec_mask]

        if int(rec_ids.numel()) > 0:
            self._write_recovery_frame(rec_ids)
        if int(norm_ids.numel()) > 0:
            self._write_ref_frame(norm_ids, norm_new_idx)

        # Refill history for reset envs with the fresh reset-frame o_t.
        o = self._build_obs()                                          # (N, 98)
        self.history[env_ids] = o[env_ids].unsqueeze(1).expand(-1, 10, -1).clone()

    def reset_all(self) -> dict:
        """Reset every env via RSI and return the initial bundle."""
        self.reset_idx(torch.arange(self.N, device=self.device), rsi=True)
        return self._bundle()

    # ----------------------------------------------------------------------
    # Step
    # ----------------------------------------------------------------------

    def resample_dr(self) -> None:
        """Redraw dynamics for envs that reset since the last call (train loop
        hook, once per iteration next to set_assist_scale).

        A reset env runs at most rollout_len steps under its previous draw
        before this fires — one dynamics switch near episode start, accepted
        so the solver sees exactly ONE notify_model_changed per iteration
        (JOINT_DOF/BODY_INERTIAL notifies re-derive mass-matrix constants over
        every world; per-reset notifies would dominate step time)."""
        if self.dr is None:
            return
        ids = self._dr_pending.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return   # nothing reset -> no writes, no notify
        self.dr.sample(ids)
        self._dr_pending.zero_()
        self.dr.write_to_sim(self.sim)
        self.sim.apply_dynamics_changes()

    def set_assist_scale(self, scale: float) -> None:
        """Set the annealed upward assist force for recovery envs.

        For each RECOVERY env, samples a world +z force magnitude
        ``U[0, assist_force_max] * scale`` Newtons (resampled on every call —
        i.e. once per training iteration — and held fixed across that rollout).
        Non-recovery envs get zero force. The resulting per-env (N,3) force is
        pushed to the sim, which applies it to the pelvis body each substep.

        Args:
            scale: anneal scale in [0, 1] (1.0 = full assist, 0.0 = none).
        """
        scale = float(scale)
        self._assist_force.zero_()
        if scale > 0.0:
            mag = torch.rand(self.N, device=self.device) * (
                self.cfg.assist_force_max * scale
            )  # (N,)
            # Apply only to recovery envs, on the world +z axis.
            self._assist_force[:, 2] = torch.where(
                self.is_recovery, mag, torch.zeros_like(mag)
            )
        self.sim.set_external_force(self._assist_force)

    def _update_pushes(self) -> None:
        """Advance per-env push timers; write assist+push forces to the sim.

        Every ~push_interval_s (uniform phase per env) a horizontal force of
        magnitude U[0.3, 1.0] * push_force_max is applied to the pelvis for
        push_duration_s. Combined additively with the recovery assist force.
        """
        self._push_timer -= 1
        start = self._push_timer <= 0
        if bool(start.any()):
            n = int(start.sum())
            theta = torch.rand(n, device=self.device) * (2.0 * torch.pi)
            mag = (0.3 + 0.7 * torch.rand(n, device=self.device)) * self.cfg.push_force_max
            f = torch.zeros(n, 3, device=self.device)
            f[:, 0] = mag * torch.cos(theta)
            f[:, 1] = mag * torch.sin(theta)
            self._push_force[start] = f
            self._push_remaining[start] = self._push_duration_steps
            # next push in ~U[0.5, 1.5] * interval
            self._push_timer[start] = (
                (0.5 + torch.rand(n, device=self.device))
                * self._push_interval_steps
            ).long().clamp(min=1)
        active = self._push_remaining > 0
        self._push_remaining.clamp_(min=0)
        self._push_remaining -= active.long()
        total = self._assist_force + self._push_force * active.unsqueeze(-1).float()
        self.sim.set_external_force(total)

    def _fallen(self) -> torch.Tensor:
        """fallen = pelvis z < z_fall OR body-up-z < up_dot_min OR head z < head_z_min."""
        base_z = self.sim.base_pos[:, 2]
        R = quat_to_matrix(self.sim.base_quat)                          # (N, 3, 3)
        up_z = R[:, 2, 2]   # body up-axis z-component (cos tilt from vertical)
        head_z = (self.sim.keypoint_pos[:, self._head_kp_idx, 2]
                  if self._head_kp_idx is not None else None)
        return compute_fallen(base_z, up_z, head_z,
                              self.cfg.z_fall, self.cfg.up_dot_min, self.cfg.head_z_min)

    def step(self, action: torch.Tensor):
        """Advance one control step.

        Returns ``(bundle, reward[N], done[N], info)``.
        """
        action = action.to(self.device).reshape(self.N, self.n_act)

        # --- residual action on the reference pose ---
        ref = self.motion.at(self.idx)
        res = torch.tanh(action) * self.action_scale
        if self.action_filter_alpha < 1.0:
            self.filt_res = (self.action_filter_alpha * res
                             + (1.0 - self.action_filter_alpha) * self.filt_res)
            res = self.filt_res
        q_tar = ref["joint_q"] + res
        if self.cfg.push_force_max > 0.0 and self.train:
            self._update_pushes()
        self.sim.step(q_tar)

        # --- advance phase (clamp at owning clip's last frame) ---
        self.ep_step += 1
        clip_end = self.motion.clip_end_of(self.idx)          # (N,) on device
        self.idx = torch.minimum(self.idx + 1, clip_end)

        # --- recovery grace window (paper §II-D) ---
        # Advance the per-env timer (steps since reset) and compute the
        # in-window mask ONCE — it shields both the reward-side fall fine
        # (recovery_fall_shield, if enabled) and instability termination below.
        self._recovery_steps_elapsed += 1
        in_window = self.is_recovery & (
            self._recovery_steps_elapsed <= self._recovery_window_steps
        )

        # --- reward (robot vs motion ref at the advanced idx) ---
        fallen = self._fallen()
        fallen_for_reward = (
            fallen & ~in_window if self.cfg.recovery_reward_shield else fallen
        )
        kp_pos = self.sim.keypoint_pos
        kp_vel = self.sim.keypoint_lin_vel
        root_pos = self.sim.base_pos
        root_quat = self.sim.base_quat
        root_h = root_pos[:, 2]

        ref_kp_pos, ref_kp_vel = self.motion.keypoints_at(self.idx)
        ref_now = self.motion.at(self.idx)
        ref_root_pos = ref_now["base_pos"]
        ref_root_quat = ref_now["base_quat"]
        ref_root_h = ref_root_pos[:, 2]

        rstate = RewardState(
            kp_pos=kp_pos, kp_vel=kp_vel, root_pos=root_pos,
            root_quat=root_quat, root_h=root_h,
            action=torch.tanh(action), prev_action=self.prev_action,
            fallen=fallen_for_reward,
        )
        rref = RewardRef(
            kp_pos=ref_kp_pos, kp_vel=ref_kp_vel,
            root_pos=ref_root_pos, root_quat=ref_root_quat, root_h=ref_root_h,
        )
        reward, terms = compute_reward(rstate, rref, self.reward_weights)
        if self.cfg.recovery_local_reward and bool(in_window.any()):
            # Body-local reward for recovery envs inside the grace window
            # (see EnvConfig.recovery_local_reward). 2.0 matches the
            # tracking product's scale; r_rel at exponent 1.0 keeps a sharp
            # standalone gradient toward matching the reference pose in the
            # robot's own root frame, regardless of world displacement.
            local = (2.0 * terms["r_rel"]
                     + terms["alive"] - terms["act_pen"] - terms["arate_pen"])
            reward = torch.where(in_window, local, reward)

        # --- termination ---
        # tracking error gates.
        q_act = self.sim.joint_q[:, self.actuated_idx]
        joint_err = ((q_act - ref_now["joint_q"]) ** 2).sum(-1)
        root_planar_err = ((root_pos[:, :2] - ref_root_pos[:, :2]) ** 2).sum(-1)
        motion_end = self.idx >= clip_end
        timeout = self.ep_step >= self.cfg.episode_len
        # Tracking-failure termination. A stander collecting alive-bonus is
        # only unlearnable if failing to track ENDS the episode. Mirrors the
        # paper's success criterion (root height deviating >0.2 m = failure).
        root_z_dev = (root_pos[:, 2] - ref_root_pos[:, 2]).abs()
        track_done = (
            (joint_err > self.cfg.joint_err_done)
            | (root_planar_err > self.cfg.root_err_done)
            | (root_z_dev > self.cfg.z_dev_done)
        )

        # --- delayed-termination recovery window (paper §II-D) ---
        # Recovery envs still inside the grace window (mask computed above):
        # suppress instability/fallen-driven termination (fallen + track_done)
        # so the policy gets time to stand up. Normal termination
        # (motion_end/timeout) still fires, and after the window expires
        # fallen/track_done terminate as usual.
        instability_done = fallen | track_done
        instability_done = instability_done & ~in_window
        done = instability_done | motion_end | timeout

        # update prev_action AFTER reward (which consumed the old prev_action).
        self.prev_action = torch.tanh(action).detach()

        # Per-term reward means (scalar floats for logging).
        reward_terms = {k: float(v.mean()) for k, v in terms.items()}

        # Recovery success: fraction of recovery envs that are currently upright
        # (not fallen). Defined as ~fallen among is_recovery envs.
        if self.is_recovery.any():
            recovery_success = float(
                (~fallen[self.is_recovery]).float().mean()
            )
        else:
            recovery_success = 0.0

        # Termination-cause rates, measured PRE-reset (auto-reset below makes
        # post-hoc inspection blind — post-reset states are always upright).
        # Fractions of all envs terminating this step, by cause; "fallen"
        # takes precedence over tracking failure when both fire, and the
        # tracking split attributes each failure to its firing criterion
        # (z_dev > root_err > joint_err precedence when several fire at once).
        _track_attr = instability_done & track_done & ~fallen
        _t_zdev = _track_attr & (root_z_dev > self.cfg.z_dev_done)
        _t_rootxy = (_track_attr & ~_t_zdev
                     & (root_planar_err > self.cfg.root_err_done))
        _t_joint = _track_attr & ~_t_zdev & ~_t_rootxy
        done_causes = {
            "fallen": float((instability_done & fallen).float().mean()),
            "tracking": float(_track_attr.float().mean()),
            "tracking_zdev": float(_t_zdev.float().mean()),
            "tracking_rootxy": float(_t_rootxy.float().mean()),
            "tracking_joint": float(_t_joint.float().mean()),
            "motion_end": float((motion_end & ~instability_done).float().mean()),
            "timeout": float((timeout & ~instability_done & ~motion_end).float().mean()),
        }

        info = {
            "terms": terms,
            "fallen": fallen,
            "reward_terms": reward_terms,
            "recovery_success": recovery_success,
            "done_causes": done_causes,
            # Reference joint velocity of the transition this action spans.
            # `ref` was fetched PRE-advance (top of step), and corpus joint_qd is a
            # BACKWARD difference, so this is joint_qd[t] = (q[t]-q[t-1])/dt — the
            # velocity of the reference over exactly the interval the action pair
            # (mu_{t-1}, mu_t) covers. Consumed by the smoothness budgets; free
            # (already gathered for the PD target).
            "ref_joint_qd": ref["joint_qd"],
            # Per-env cause masks (pre-reset) for per-clip attribution.
            "done_cause_masks": {
                "fallen": instability_done & fallen,
                "zdev": _t_zdev,
                "rootxy": _t_rootxy,
                "motion_end": motion_end & ~instability_done,
                "timeout": timeout & ~instability_done & ~motion_end,
                # Truncations (successful clip end / time limit): episodes cut
                # by the protocol, NOT by failure — GAE must bootstrap V(s_T)
                # for these instead of assuming zero future value.
                "truncated": (motion_end | timeout) & ~instability_done,
            },
            # Pre-reset death-state signals (valid where a mask is set):
            # signed height error and body tilt at the terminal step — lets
            # diagnostics distinguish stumble dips (low + tilted) from
            # over-standing (high + upright) etc.
            "death_state": {
                "z_signed": root_pos[:, 2] - ref_root_pos[:, 2],
                "up_z": quat_to_matrix(self.sim.base_quat)[:, 2, 2],
            },
        }

        # --- stash pre-reset terminal bundle, then auto-reset done envs (RSI) ---
        if bool(done.any()):
            info["terminal_obs"] = self._bundle()
            self.reset_idx(done.nonzero(as_tuple=False).squeeze(-1), rsi=True)
        else:
            # roll history with the latest o_t (no reset overwrote it).
            pass

        # roll history for all NON-reset envs; reset envs were just refilled.
        not_done = ~done
        if bool(not_done.any()):
            o = self._build_obs()
            nd = not_done.nonzero(as_tuple=False).squeeze(-1)
            self.history[nd] = torch.cat(
                [self.history[nd, 1:], o[nd].unsqueeze(1)], dim=1
            )

        return self._bundle(), reward, done, info

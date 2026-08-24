"""RGMT training entry point.

Provides ``run_training(cfg) -> dict`` (callable without the Hydra CLI, for
testing) and a ``@hydra.main`` ``main`` entry point for real training runs.
"""

from __future__ import annotations

import os

# Reduce CUDA allocator fragmentation (must be set before torch initializes
# CUDA). The 3070 fit-test OOMed at 4096 envs with ~1 GB "reserved but
# unallocated" — expandable segments reclaim that.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import time

import hydra
import torch
from dataclasses import asdict
from pathlib import Path
from omegaconf import OmegaConf

import nebo as nb
import warp as wp

from rgmt.data.motion import MotionRef
from rgmt.data.corpus import MotionCorpus
from rgmt.data.cache_key import file_sha256
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_XML, ROBOT_URDF
from rgmt.env.track_env import TrackEnv, EnvConfig
from rgmt.policy.networks import RGMTActorCritic, PolicyDims
from rgmt.algos.rollout import RolloutBuffer
from rgmt.algos.ppo import PPOTrainer, PPOConfig

# Module-level nebo setup — sets workflow description and UI defaults.
nb.md(
    "# RGMT Motion Tracking\n"
    "PPO trainer for the Asimov v1 humanoid (23 DoF), "
    "dynamics-conditioned command aggregation."
)
nb.ui(view="dag", tracker="step")


def assist_scale_at(it: int, total: int) -> float:
    """Linear anneal of the upward assist-force scale: ``1 -> 0`` over ``total``.

    ``it=0`` -> 1.0, ``it=total/2`` -> 0.5, ``it>=total`` -> 0.0.
    Guards against ``total <= 0`` (no annealing window -> immediately 0.0).
    """
    if total <= 0:
        return 0.0
    return max(0.0, 1.0 - it / total)


def _build_noise_dict(noise_cfg) -> dict:
    """Normalise noise config to the EnvConfig-expected keys.

    The Hydra / test config may arrive as either:
      - ``{v_xy: float, v_z: float, w: float, g: float, q: float}``  (canonical)
      - ``{v: [x, y, z], w: float, g: float, q: float}``            (legacy brief form)

    Always returns a dict with keys ``{v_xy, v_z, w, g, q}``.
    """
    if hasattr(noise_cfg, "_metadata"):
        # OmegaConf DictConfig — convert to plain dict first.
        noise_cfg = OmegaConf.to_container(noise_cfg, resolve=True)

    if isinstance(noise_cfg, dict):
        if "v_xy" in noise_cfg:
            return dict(noise_cfg)
        if "v" in noise_cfg:
            v = noise_cfg["v"]
            return {
                "v_xy": float(v[0]),
                "v_z": float(v[2]),
                "w": float(noise_cfg.get("w", 0.52)),
                "g": float(noise_cfg.get("g", 0.05)),
                "q": float(noise_cfg.get("q", 0.1)),
            }
    # Fallback: return defaults (EnvConfig will also fall back).
    return {}


def _build_corpus(cfg, device, kp_links) -> MotionCorpus:
    """Build a MotionCorpus from a cache dir, a .npz dir, or a single .npz path.

    Priority order:
      1. ``cfg.motion_cache`` (preprocessed cache dir) → ``MotionCorpus.load_cache``
      2. ``cfg.motion_dir``   (.npz clip dir)          → ``MotionCorpus.from_clips``
      3. ``cfg.motion_path``  (single .npz)            → 1-clip corpus

    Clips for paths 2 and 3 are loaded on CPU; the corpus exposes tensors on
    ``device`` via ``output_device``. Corpus STORAGE lives on
    ``cfg.corpus_device`` (default: the training device — GPU-resident storage
    removes the per-step CPU gather + H2D copy, profiled at ~40% of env.step;
    a 4 h corpus is only ~0.5 GB. Set ``corpus_device: cpu`` for corpora that
    don't fit). ``physics_fps``/``src_fps`` are fixed at 60/30, matching
    preprocess defaults.
    """
    storage = cfg.get("corpus_device") or device
    if cfg.get("motion_cache"):
        corpus = MotionCorpus.load_cache(
            cfg.motion_cache,
            output_device=device,
            urdf_hash=file_sha256(ROBOT_URDF),
            physics_fps=60,
            src_fps=30,
            keypoint_links=kp_links,
        )
        return corpus.to_storage(storage)
    if cfg.get("motion_dir"):
        npz = sorted(Path(cfg.motion_dir).glob("*.npz"))
        if not npz:
            raise ValueError(f"motion_dir has no .npz: {cfg.motion_dir}")
        clips = [
            # ground=True matches the preprocess default: the raw-clip path
            # previously used a different z-datum than cache-based training
            # confounding z_fall/z_dev_done/h_ref across paths.
            MotionRef.load(p, ROBOT_XML, ROBOT_URDF, device="cpu",
                           keypoint_links=kp_links, ground=True)
            for p in npz
        ]
        corpus = MotionCorpus.from_clips(clips, [p.stem for p in npz], kp_links, output_device=device)
        return corpus.to_storage(storage)
    ref = MotionRef.load(cfg.motion_path, ROBOT_XML, ROBOT_URDF, device="cpu",
                         keypoint_links=kp_links, ground=True)
    return MotionCorpus.from_clips([ref], ["clip0"], kp_links, output_device=device).to_storage(storage)


def compose_rsi_weights(corpus_names, split, fail_ema, mining_boost) -> dict:
    """RSI weight dict: mining weights on train clips, 0 elsewhere.

    The split defines the training universe: test clips AND any corpus clip
    listed in neither role get weight 0. The neither-role case only bites when
    a cache is deliberately a superset of the split (difficulty subsets
    sharing one cache); without this rule those clips would silently train at
    weight 1.
    """
    w = {n: 1.0 + mining_boost * e for n, e in fail_ema.items()}
    if split is not None:
        train_names = set(split["train"])
        w.update({n: 0.0 for n in corpus_names if n not in train_names})
    return w


def run_training(cfg) -> dict:
    """Run the PPO training loop and return final stats dict.

    Args:
        cfg: OmegaConf DictConfig (or compatible) with top-level keys:
             experiment_name, seed, device, env, algo, network, reward, log_dir,
             and one of the following motion source keys (checked in order):

             motion_cache : str, optional
                 Path to a preprocessed corpus directory (written by
                 ``run_preprocess``).  Loaded directly via
                 ``MotionCorpus.load_cache``; fastest path — no FK on startup.
             motion_dir : str, optional
                 Directory of raw ``.npz`` clip files.  All clips are loaded
                 and concatenated into a corpus on the fly.
             motion_path : str
                 Single ``.npz`` clip file.  Used when neither ``motion_cache``
                 nor ``motion_dir`` is set.  Builds a 1-clip corpus.

             Precedence: ``motion_cache`` > ``motion_dir`` > ``motion_path``.

    Returns:
        dict with at minimum ``avg_return`` plus PPO update stats.
    """
    device = cfg.device
    wp.set_device(device)
    torch.manual_seed(cfg.seed)

    # ---- Motion corpus -------------------------------------------------------
    kp_links = list(cfg.env.keypoint_links or []) or KEYPOINT_LINKS
    corpus = _build_corpus(cfg, device, kp_links)

    # ---- Environment -------------------------------------------------------
    noise = _build_noise_dict(cfg.env.noise)
    reward_dict = OmegaConf.to_container(cfg.reward, resolve=True) if cfg.reward else {}

    env_cfg = EnvConfig(
        num_envs=cfg.env.num_envs,
        kp=cfg.env.kp,
        kd=cfg.env.kd,
        control_decimation=cfg.env.control_decimation,
        dt=cfg.env.dt,
        action_scale=cfg.env.action_scale,
        action_filter_alpha=cfg.env.get("action_filter_alpha", 1.0),
        K=cfg.env.K,
        L=cfg.env.L,
        foot_friction=cfg.env.foot_friction,
        # Plant-defining: must be forwarded or the dataclass default silently
        # wins (the known unforwarded-field bug class).
        effort_limits=cfg.env.get("effort_limits", None),
        episode_len=cfg.env.episode_len,
        z_fall=cfg.env.z_fall,
        up_dot_min=cfg.env.up_dot_min,
        head_z_min=cfg.env.get("head_z_min", 0.3),
        joint_err_done=cfg.env.joint_err_done,
        root_err_done=cfg.env.root_err_done,
        z_dev_done=cfg.env.get("z_dev_done", 0.2),
        noise=noise,
        noise_level=cfg.env.noise_level,
        keypoint_links=kp_links,
        reward=reward_dict if reward_dict else None,
        # BUG FIX: these were never forwarded, so EnvConfig silently used
        # its dataclass defaults (recovery_fraction=0.15) regardless of config.
        recovery_fraction=cfg.env.get("recovery_fraction", 0.15),
        recovery_window_s=cfg.env.get("recovery_window_s", 3.0),
        assist_force_max=cfg.env.get("assist_force_max", 200.0),
        recovery_reward_shield=cfg.env.get("recovery_reward_shield", False),
        recovery_local_reward=cfg.env.get("recovery_local_reward", False),
        recovery_spawn_tilt_max=cfg.env.get("recovery_spawn_tilt_max", 3.14159265),
        recovery_spawn_z_min=cfg.env.get("recovery_spawn_z_min", 0.10),
        recovery_spawn_z_max=cfg.env.get("recovery_spawn_z_max", 0.45),
        drift_obs=cfg.env.get("drift_obs", False),
        drift_obs_proprio=cfg.env.get("drift_obs_proprio", False),
        sampling_weights_json=cfg.env.get("sampling_weights_json", None),
        push_force_max=cfg.env.get("push_force_max", 0.0),
        push_interval_s=cfg.env.get("push_interval_s", 3.0),
        push_duration_s=cfg.env.get("push_duration_s", 0.15),
        # Domain randomization (dict of DRConfig kwargs; None = off).
        dr=(OmegaConf.to_container(cfg.env.dr, resolve=True)
            if cfg.env.get("dr") else None),
    )
    env = TrackEnv(env_cfg, corpus, device=device)

    # ---- Train/test split + in-loop robust eval + failure-weighted mining ---
    # (2026-07-15) Training and evaluation are coupled: every eval.robust_every
    # iterations the loop runs THE metric (success-gated, noisy repeats) on the
    # held-out test split, then a single train-split pass whose failures drive
    # the RSI sampling weights (EMA-smoothed so single-pass flicker doesn't
    # whipsaw the curriculum). Test clips are excluded from RSI from iteration
    # 0. This supersedes env.sampling_weights_json when both are set.
    from rgmt.eval_gated import build_eval_env, run_gated_eval, load_split, split_clip_ids
    split_path = OmegaConf.select(cfg, "eval.split_json", default=None)
    robust_every = int(OmegaConf.select(cfg, "eval.robust_every", default=0) or 0)
    mining_on = bool(OmegaConf.select(cfg, "eval.mining", default=True))
    mining_ema = float(OmegaConf.select(cfg, "eval.mining_ema", default=0.5))
    robust_noise = float(OmegaConf.select(cfg, "eval.robust_noise", default=0.05))
    robust_repeats = int(OmegaConf.select(cfg, "eval.robust_repeats", default=3))
    robust_envs = int(OmegaConf.select(cfg, "eval.robust_envs", default=256))
    # RSI weight = 1 + mining_boost*EMA. At the default 3.0 a clip failing
    # every pass converges to a 4x sampling weight; raise it to mine harder
    # once the curriculum plateaus (observed: a flat 74-77% band for 3600
    # iterations at 3.0 with the failure pool stuck near 22%).
    mining_boost = float(OmegaConf.select(cfg, "eval.mining_boost", default=3.0))

    split = load_split(split_path) if split_path else None
    # Optional clip -> difficulty label map (if the split carries one) for per-class in-loop
    # eval curves (eval/test_success_<label> etc.).
    _diff_map = (split or {}).get("difficulty") or {}
    fail_ema: dict = {}     # clip name -> EMA of single-pass failure (0..1)
    eval_last: dict = {}    # latest in-loop eval summary (goes into ckpts)

    def _apply_sampling_weights():
        if split is None and not fail_ema:
            return
        w = compose_rsi_weights(corpus.clip_names, split, fail_ema, mining_boost)
        if split is not None:
            n_off = sum(1 for v in w.values() if v == 0.0) - len(split["test"])
            if n_off:
                print(f"[train] split excludes {n_off} corpus clips in "
                      f"neither role (+ {len(split['test'])} test)")
        stats = corpus.set_clip_sampling_weights(w)
        print(f"[train] RSI weights: {stats['matched']}/{stats['total_clips']} "
              f"clips set, boosted mass {stats['boosted_mass_share']:.3f}")

    if split is not None:
        split_clip_ids(corpus, split, "test")   # validate names early (raises)
        if cfg.env.get("sampling_weights_json"):
            print("[train] WARNING: eval.split_json supersedes env.sampling_weights_json")
        _apply_sampling_weights()

    # ---- Policy ------------------------------------------------------------
    dims = PolicyDims(
        priv_dim=env.priv_dim,
        obs_dim=env.obs_dim,
        cmd_dim=env.cmd_dim,
        n_embd=cfg.network.n_embd,
        n_heads=cfg.network.n_heads,
    )
    model = RGMTActorCritic(
        dims,
        actor_hidden=tuple(cfg.network.actor_hidden),
        critic_hidden=tuple(cfg.network.critic_hidden),
    ).to(device)

    # ---- PPO trainer -------------------------------------------------------
    ppo_cfg = PPOConfig(
        lr=cfg.algo.lr,
        n_epochs=cfg.algo.n_epochs,
        mb_size=cfg.algo.mb_size,
        clip=cfg.algo.clip,
        value_coef=cfg.algo.value_coef,
        entropy_coef=cfg.algo.entropy_coef,
        max_grad_norm=cfg.algo.max_grad_norm,
        dual_clip=cfg.algo.dual_clip,
        target_kl=cfg.algo.target_kl,
        kl_adaptive=cfg.algo.get("kl_adaptive", False),
        kl_shock_factor=cfg.algo.get("kl_shock_factor", 0.0),
        value_clip=cfg.algo.get("value_clip", True),
        grad_clip_per_module=cfg.algo.get("grad_clip_per_module", False),
        lr_min=cfg.algo.get("lr_min", 1.0e-5),
        lr_max=cfg.algo.get("lr_max", 1.0e-3),
        # Action-smoothness penalties (0 = off, bit-identical to pre-2026-07-29).
        lambda_smooth=cfg.algo.get("lambda_smooth", 0.0),
        smooth_slack=cfg.algo.get("smooth_slack", 2.5),
        smooth_eps_floor=cfg.algo.get("smooth_eps_floor", 0.02),
        lambda_spatial=cfg.algo.get("lambda_spatial", 0.0),
        spatial_noise_std=cfg.algo.get("spatial_noise_std", 0.05),
        spatial_slack=cfg.algo.get("spatial_slack", 2.5),
        spatial_floor=cfg.algo.get("spatial_floor", 0.0),
        # Plant quantities the budgets need — taken from the env, never guessed.
        ctrl_dt=cfg.env.dt * max(int(cfg.env.control_decimation), 1),
        action_scale=cfg.env.action_scale,
    )
    ppo = PPOTrainer(model, ppo_cfg)
    _smooth_on = ppo_cfg.lambda_smooth > 0.0

    # ---- Rollout buffer ----------------------------------------------------
    buf = RolloutBuffer(cfg.env.num_envs, cfg.algo.rollout_len, dims, device,
                        store_smooth=_smooth_on)

    # ---- Checkpointing -----------------------------------------------------
    ckpt_dir = Path(cfg.log_dir) / cfg.experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_every = int(OmegaConf.select(cfg, "eval.every", default=50))

    def _save_ckpt(iteration: int, stats: dict | None = None,
                   names: tuple | None = None) -> None:
        opt = getattr(ppo, "opt", None) or getattr(ppo, "optimizer", None)
        payload = {
            "model": model.state_dict(),
            "opt": opt.state_dict() if opt is not None else None,
            # KL-adaptive controller state: without this, a resumed adaptive
            # run snaps the LR back to cfg.algo.lr on its first adaptation.
            "ppo_lr": ppo.lr,
            "iteration": iteration,
            "dims": asdict(dims),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "stats": stats,
            # In-loop eval state: the ckpt is self-describing — resuming a
            # mining run reconstructs its RSI distribution from clip_fail_ema
            # without any external weights file.
            "clip_fail_ema": dict(fail_ema) if fail_ema else None,
            "split_json": split_path,
            "eval_last": dict(eval_last) if eval_last else None,
        }
        # Atomic writes: a crash mid-save must not leave a truncated file
        # (a partially written checkpoint once cost a whole run).
        for name in names or (f"ckpt_{iteration:06d}.pt", "latest.pt"):
            tmp = ckpt_dir / (name + ".tmp")
            torch.save(payload, tmp)
            os.replace(tmp, ckpt_dir / name)

    # ---- Resume ------------------------------------------------------------
    start_it = 0
    resume_from = cfg.get("resume_from")
    if resume_from:
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt = getattr(ppo, "opt", None) or getattr(ppo, "optimizer", None)
        if opt is not None and ck.get("opt") is not None:
            opt.load_state_dict(ck["opt"])
        # Restore the adaptive-LR controller state (older ckpts lack the key
        # — fall back to the config LR, the previous behavior). Push it into
        # the param groups so fixed-LR resumes also continue where they left
        # off; cosine-scheduled runs overwrite it next iteration anyway.
        ppo.lr = float(ck.get("ppo_lr", ppo.lr))
        if opt is not None:
            for g in opt.param_groups:
                g["lr"] = ppo.lr
        start_it = int(ck.get("iteration", 0))
        # Restore the mining state so a resumed run continues its curriculum
        # (and its test exclusion) instead of silently reverting to uniform.
        if ck.get("clip_fail_ema"):
            fail_ema.update(ck["clip_fail_ema"])
            _apply_sampling_weights()
        eval_last = dict(ck.get("eval_last") or {})
        print(f"resumed from {resume_from} at iteration {start_it} (lr {ppo.lr:.2e})")

    # ---- Nebo run ----------------------------------------------------------
    final: dict = {}

    with nb.start_run(
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    ):
        nb.log_cfg({
            "num_envs": cfg.env.num_envs,
            "rollout_len": cfg.algo.rollout_len,
            "iterations": cfg.algo.iterations,
            "lr": cfg.algo.lr,
            "n_epochs": cfg.algo.n_epochs,
            "mb_size": cfg.algo.mb_size,
            "clip": cfg.algo.clip,
            "gamma": cfg.algo.gamma,
            "lam": cfg.algo.lam,
            "n_embd": cfg.network.n_embd,
            "actor_hidden": list(cfg.network.actor_hidden),
            "critic_hidden": list(cfg.network.critic_hidden),
            "priv_dim": env.priv_dim,
        })

        bundle = env.reset_all()

        # Assist-force anneal window. Defaults to the full training run so the
        # upward fall-recovery assist fades from 1.0 -> 0.0 over all iterations.
        # Explicit None check: `or` coerced an intentional 0 (= no assist,
        # assist_scale_at returns 0.0 for total<=0) into a full-run anneal.
        _anneal_cfg = OmegaConf.select(cfg, "algo.assist_anneal_iters", default=None)
        assist_anneal_iters = (
            int(_anneal_cfg) if _anneal_cfg is not None else int(cfg.algo.iterations)
        )

        _last_kl_alert_it = -10**9
        # In-loop eval state: env built lazily on first eval (a second, small
        # NewtonSim shares the process; kernels are already compiled). Best
        # tracking seeds from the resumed ckpt's last eval so best_test.pt
        # isn't clobbered by a worse post-resume first eval.
        _eval_env = [None]
        _eval_ids = [None, None]
        _best_test = [float(eval_last.get("test_rate", 0.0) or 0.0)]
        try:
            for it in range(start_it, cfg.algo.iterations):
                buf.clear()

                # Anneal and push the upward assist force into the env (no-op for
                # non-recovery envs; see TrackEnv.set_assist_scale).
                assist_scale = assist_scale_at(it, assist_anneal_iters)

                # Exploration-noise anneal: linearly take entropy_coef ->
                # entropy_final over the LAST entropy_anneal_frac of the run.
                # Rationale: the mean is optimized under std~0.8
                # sampling noise all run, which biases it toward twitchy
                # corrections; relaxing the entropy bonus late lets log_std
                # shrink so the mean sharpens for the low-noise regime it is
                # actually deployed in.
                entropy_final = cfg.algo.get("entropy_final", None)
                if entropy_final is not None:
                    frac = float(cfg.algo.get("entropy_anneal_frac", 0.3))
                    p = (it / max(cfg.algo.iterations - 1, 1) - (1.0 - frac)) / max(frac, 1e-9)
                    ppo.cfg.entropy_coef = (cfg.algo.entropy_coef
                                            + max(0.0, min(p, 1.0))
                                            * (float(entropy_final) - cfg.algo.entropy_coef))

                # Cosine LR decay lr -> lr_final over the run (gated on lr_final).
                # Early runs peaked around iteration 500-600 then degraded under fixed lr.
                # Disabled when the PPO KL-adaptive LR owns the learning rate —
                # overwriting it here would undo the adaptation every iteration.
                lr_final = OmegaConf.select(cfg, "algo.lr_final", default=None)
                if lr_final is not None and not ppo_cfg.kl_adaptive:
                    import math as _math
                    # Schedule horizon defaults to this config's iterations. For
                    # RESUMED runs with an extended iterations count, that horizon
                    # change silently HOT-RESTARTS the cosine (observed: a
                    # continuation resumed at ~1.65e-4 instead of the base
                    # run's final 3e-5, a 5.5x jump). Set
                    # algo.lr_schedule_total to the ORIGINAL horizon to continue
                    # at the schedule's tail instead; frac clamps at 1.0 so
                    # training past the horizon holds lr_final.
                    sched_total = int(
                        OmegaConf.select(cfg, "algo.lr_schedule_total", default=None)
                        or cfg.algo.iterations
                    )
                    frac = min(it / max(sched_total - 1, 1), 1.0)
                    lr_now = lr_final + 0.5 * (cfg.algo.lr - lr_final) * (
                        1.0 + _math.cos(_math.pi * frac))
                    opt = getattr(ppo, "opt", None) or getattr(ppo, "optimizer", None)
                    if opt is not None:
                        for g in opt.param_groups:
                            g["lr"] = lr_now
                env.set_assist_scale(assist_scale)
                # DR: redraw dynamics for envs that reset last iteration
                # (one batched notify; no-op when DR is off or nothing reset).
                env.resample_dr()

                # Accumulators for rollout-mean metrics.
                _reward_term_sums: dict[str, float] = {}
                _done_cause_sums: dict[str, float] = {}
                _recovery_success_sum: float = 0.0
                _rollout_steps = 0
                _t_roll0 = time.perf_counter()

                # Collect rollout_len steps.
                for _ in range(cfg.algo.rollout_len):
                    with torch.no_grad():
                        if _smooth_on:
                            action, logp, value, mu = model.act(bundle, return_mean=True)
                        else:
                            action, logp, value = model.act(bundle)
                    bundle_next, reward, done, step_info = env.step(action)
                    # Truncation bootstrapping: motion_end
                    # (successful clip completion!) and timeout cut episodes by
                    # protocol, not by failure. GAE zeroes the bootstrap at every
                    # done, which valued completing a clip like dying. Fold the
                    # terminal state's value into the reward for truncated envs —
                    # exact for GAE since done still masks the recursion.
                    trunc = step_info["done_cause_masks"].get("truncated")
                    if trunc is not None and bool(trunc.any()):
                        with torch.no_grad():
                            v_term = model.value(step_info["terminal_obs"])
                        reward = reward + cfg.algo.gamma * v_term * trunc.float()
                    if _smooth_on:
                        # tanh(mu) is the commanded residual in normalized units
                        # (the env applies action_scale); detached — it is the
                        # constant anchor for the next step's temporal penalty.
                        buf.add(bundle, action, logp, value, reward, done,
                                tanh_mu=torch.tanh(mu).detach(),
                                ref_qd=step_info["ref_joint_qd"])
                    else:
                        buf.add(bundle, action, logp, value, reward, done)
                    bundle = bundle_next

                    # Accumulate per-term reward means.
                    for term_k, term_v in step_info.get("reward_terms", {}).items():
                        _reward_term_sums[term_k] = (
                            _reward_term_sums.get(term_k, 0.0) + term_v
                        )
                    # Accumulate termination-cause rates (pre-reset truth from the
                    # env; the auto-reset makes post-hoc fall counting blind).
                    for dc_k, dc_v in step_info.get("done_causes", {}).items():
                        _done_cause_sums[dc_k] = _done_cause_sums.get(dc_k, 0.0) + dc_v
                    _recovery_success_sum += step_info.get("recovery_success", 0.0)
                    _rollout_steps += 1

                _t_roll = time.perf_counter() - _t_roll0

                # Bootstrap last value.
                with torch.no_grad():
                    last_v = model.value(bundle)

                # GAE + flatten.
                data = buf.compute_gae(last_v, cfg.algo.gamma, cfg.algo.lam)

                # PPO update.
                _t_upd0 = time.perf_counter()
                stats = ppo.update(data)
                _t_upd = time.perf_counter() - _t_upd0

                # RMS update for observation normalisation.
                model.update_rms(data)

                avg_return = float(data["returns"].mean())
                final = dict(avg_return=avg_return, **{k: float(v) for k, v in stats.items()})

                # Per-reward-term rollout means.
                denom = max(_rollout_steps, 1)
                reward_term_means = {k: v / denom for k, v in _reward_term_sums.items()}
                recovery_success = _recovery_success_sum / denom

                # Throughput metrics (saturation tuning; see scripts/profile_throughput.py).
                final["time/rollout_s"] = _t_roll
                final["time/update_s"] = _t_upd
                final["time/env_steps_per_s"] = (
                    cfg.env.num_envs * cfg.algo.rollout_len / max(_t_roll, 1e-9)
                )

                # Merge new metrics into final stats.
                final["assist_scale"] = assist_scale
                final["entropy_coef"] = float(ppo.cfg.entropy_coef)
                final["recovery_success"] = recovery_success
                for term_k, term_v in reward_term_means.items():
                    final[f"reward/{term_k}"] = term_v
                # Termination-cause rates (mean per-step fraction of envs ending
                # by each cause) + the fall share of all terminations.
                _dc_means = {k: v / denom for k, v in _done_cause_sums.items()}
                _dc_total = sum(_dc_means.values())
                for dc_k, dc_v in _dc_means.items():
                    final[f"done/{dc_k}"] = dc_v
                final["done/fall_share"] = (
                    _dc_means.get("fallen", 0.0) / _dc_total if _dc_total > 0 else 0.0
                )

                # Log everything to nebo.
                for k, v in final.items():
                    nb.log_line(f"train/{k}", v, step=it)

                # KL-shock alert: wakes any `nebo runs wait` supervisor mid-run
                # (verified file-mode on nebo 0.3.0). Rate-limited so a sustained
                # shock doesn't spam; 5x target is well past early-stop territory.
                if final.get("kl", 0.0) > 5.0 * ppo_cfg.target_kl and (
                    it - _last_kl_alert_it
                ) > 200:
                    _last_kl_alert_it = it
                    nb.alert(
                        f"kl-shock: {cfg.experiment_name}",
                        f"iter {it}: kl={final['kl']:.4f} (target {ppo_cfg.target_kl})",
                        level=nb.AlertLevel.WARN,
                    )

                # In-loop robust eval (THE metric) + mining-weight refresh.
                if robust_every and ((it + 1) % robust_every == 0
                                     or (it + 1) == cfg.algo.iterations):
                    _t_ev0 = time.perf_counter()
                    model.eval()
                    if _eval_env[0] is None:
                        _eval_env[0] = build_eval_env(
                            corpus, device, robust_envs,
                            drift_obs=env_cfg.drift_obs,
                            drift_obs_proprio=env_cfg.drift_obs_proprio,
                            kp=env_cfg.kp, kd=env_cfg.kd,
                            action_scale=env_cfg.action_scale,
                            action_filter_alpha=env_cfg.action_filter_alpha,
                            effort_limits=env_cfg.effort_limits,
                            dt=env_cfg.dt,
                            control_decimation=env_cfg.control_decimation)
                        _eval_ids[0] = (split_clip_ids(corpus, split, "test")
                                        if split else None)
                        _eval_ids[1] = (split_clip_ids(corpus, split, "train")
                                        if split else list(range(corpus.n_clips)))
                    # THE metric on the held-out test split (fixed seed:
                    # the same noise stream every eval, so the curve moves
                    # only when the policy does).
                    r_test = run_gated_eval(
                        model, _eval_env[0], _eval_ids[0],
                        action_noise=robust_noise, repeats=robust_repeats,
                        seed=0, per_clip=bool(_diff_map))
                    # One train-split pass drives the mining weights; the
                    # seed varies per eval so the EMA integrates independent
                    # noise draws instead of one frozen stream.
                    if mining_on:
                        r_train = run_gated_eval(
                            model, _eval_env[0], _eval_ids[1],
                            action_noise=robust_noise, repeats=1,
                            seed=1000 + it, per_clip=True)
                        for name, rc in r_train["clips"].items():
                            failed = 1.0 - float(rc["success"])
                            prev = fail_ema.get(name, failed)
                            fail_ema[name] = (1.0 - mining_ema) * prev + mining_ema * failed
                        _apply_sampling_weights()
                        # Per-clip outcomes for offline failure autopsy
                        # (scripts/analyze_failures.py); overwritten each eval.
                        tmp = ckpt_dir / "clips_train_latest.json.tmp"
                        tmp.write_text(json.dumps(
                            {"iteration": it + 1, "clips": r_train["clips"]}))
                        os.replace(tmp, ckpt_dir / "clips_train_latest.json")
                    model.train()
                    eval_s = time.perf_counter() - _t_ev0
                    eval_last = dict(
                        iteration=it + 1, test_rate=r_test["rate"],
                        test_mpkpe_mm=r_test["mpkpe_mm"],
                        test_pose_mm=r_test["pose_mm"],
                        test_jitter_mrad=r_test["jitter_mrad"],
                        train_fail_frac=(1.0 - r_train["rate"]) if mining_on else None)
                    nb.log_line("eval/test_success", r_test["rate"] * 100.0, step=it)
                    nb.log_line("eval/test_mpkpe_mm", r_test["mpkpe_mm"], step=it)
                    nb.log_line("eval/test_pose_mm", r_test["pose_mm"], step=it)
                    nb.log_line("eval/test_jitter_mrad", r_test["jitter_mrad"], step=it)
                    # Per-difficulty breakdown (when the split carries a clip ->
                    # label map). Success is pass-weighted; MPKPE/pose are
                    # CLIP-weighted survivor means (the headline aggregate is
                    # step-weighted — close but not identical).
                    if _diff_map:
                        by_d: dict = {}
                        for name, rc in r_test["clips"].items():
                            lbl = _diff_map.get(name)
                            if lbl is None:
                                continue
                            a = by_d.setdefault(lbl, [0, 0, [], []])
                            a[0] += int(rc["success"]); a[1] += int(rc["passes"])
                            if rc["success"] > 0:
                                a[2].append(rc["surv_kpe_mm"])
                                a[3].append(rc["surv_pose_mm"])
                        for lbl, (s, p, kpes, poses) in sorted(by_d.items()):
                            nb.log_line(f"eval/test_success_{lbl}",
                                        100.0 * s / max(p, 1), step=it)
                            if kpes:
                                nb.log_line(f"eval/test_mpkpe_mm_{lbl}",
                                            sum(kpes) / len(kpes), step=it)
                                nb.log_line(f"eval/test_pose_mm_{lbl}",
                                            sum(poses) / len(poses), step=it)
                        eval_last["test_success_by_difficulty"] = {
                            lbl: round(100.0 * s / max(p, 1), 2)
                            for lbl, (s, p, _, _) in sorted(by_d.items())}
                    if mining_on:
                        nb.log_line("eval/train_fail_pct", (1.0 - r_train["rate"]) * 100.0, step=it)
                    nb.log_line("train/time/eval_robust_s", eval_s, step=it)
                    _by_d = eval_last.get("test_success_by_difficulty") or {}
                    _d_str = ("  [" + " ".join(f"{k}:{v:.0f}%" for k, v in _by_d.items()) + "]"
                              if _by_d else "")
                    print(f"[eval @{it+1}] test {r_test['rate']*100:.1f}% "
                          f"mpkpe {r_test['mpkpe_mm']:.1f}mm "
                          f"jitter {r_test['jitter_mrad']:.1f}mrad ({eval_s:.0f}s){_d_str}")
                    if r_test["rate"] > _best_test[0]:
                        _best_test[0] = r_test["rate"]
                        _save_ckpt(it + 1, final, names=("best_test.pt",))

                # Checkpoint every eval.every iters and on the final iter.
                if (it + 1) % ckpt_every == 0 or (it + 1) == cfg.algo.iterations:
                    _save_ckpt(it + 1, final)
        except Exception as exc:
            # Crash alert: land the failure in the run stream so a
            # `nebo runs wait` supervisor wakes immediately instead of
            # discovering the corpse via checkpoint mtimes.
            nb.alert(
                f"training-crashed: {cfg.experiment_name}",
                f"{type(exc).__name__}: {exc}",
                level=nb.AlertLevel.ERROR,
            )
            raise
        nb.alert(
            f"training-complete: {cfg.experiment_name}",
            f"{cfg.algo.iterations} iters, avg_return "
            f"{final.get('avg_return', float('nan')):.2f}",
            level=nb.AlertLevel.INFO,
        )


    return final


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()

"""Reusable success-gated eval core (THE metric — see scripts/eval_success_gated.py).

Factored out of the standalone script (2026-07-15) so the training loop can run
the same protocol in-process every ``eval.robust_every`` iterations without
paying per-invocation sim-build/module-load costs, and so train/test splits
evaluate through one code path.

RNG is hermetic: the whole eval runs inside ``torch.random.fork_rng`` seeded
from ``seed``, so calling it mid-training neither perturbs nor is perturbed by
the training RNG stream, and results are reproducible for a given (ckpt,
clip set, seed).
"""
from typing import Optional, Sequence

import torch

from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.env.track_env import TrackEnv, EnvConfig
from rgmt.utils.rotation import quat_to_matrix


def build_eval_env(corpus, device, n_envs: int = 256, *,
                   drift_obs: bool = False, drift_obs_proprio: bool = False,
                   kp=100.0, kd=5.0, action_scale: float = 0.5,
                   action_filter_alpha: float = 1.0,
                   effort_limits=None,
                   dt: float = 1.0 / 60.0,
                   control_decimation: int = 1) -> TrackEnv:
    """The eval-protocol env: noise-free, no recovery/pushes, root_err gate
    relaxed to 1.0 (the historical protocol constant — do not tighten without
    re-baselining every number).

    kp/kd/action_scale/effort_limits MUST match the values the policy was
    trained under (they define the plant + action semantics); the defaults are
    the historical uniform scalars / uncapped torques, so old runs evaluate
    unchanged. Pass the trained values (from the checkpoint config) for any
    run that used per-joint gains, a different action_scale, or torque caps."""
    cfg = EnvConfig(
        num_envs=n_envs,
        # dt/decimation are plant-defining like kp/kd — hard-coding them here
        # would silently score a differently-clocked plant (review R1).
        control_decimation=int(control_decimation), dt=float(dt), kp=kp, kd=kd,
        action_scale=action_scale, action_filter_alpha=action_filter_alpha,
        effort_limits=effort_limits,
        foot_friction=0.75, K=9, L=10, episode_len=100000,
        z_fall=0.12, up_dot_min=0.0, head_z_min=0.3, joint_err_done=100.0,
        root_err_done=1.0, noise_level=0.0,
        keypoint_links=KEYPOINT_LINKS, recovery_fraction=0.0,
        drift_obs=drift_obs, drift_obs_proprio=drift_obs_proprio)
    return TrackEnv(cfg, corpus, device=device, train=False)


def run_gated_eval(model, env: TrackEnv, clip_ids: Optional[Sequence[int]] = None, *,
                   action_noise: float = 0.0, repeats: int = 1, seed: int = 0,
                   per_clip: bool = True) -> dict:
    """Roll every clip in ``clip_ids`` (default: all) start->end; success = the
    episode reaches motion_end without a failure termination.

    model: RGMTActorCritic in eval mode, or None for pure-PD (zero actions).
    Returns dict(success, total, rate, per_pass, mpkpe_mm, pose_mm, jitter_mrad,
                 clips={name: {success, steps, kpe_mm, pose_mm, ...}}).
    """
    corpus = env.motion
    dev = env.device
    n_envs = env.N
    if clip_ids is None:
        clip_ids = range(len(corpus.clip_names))
    # longest-first so each wave's straggler is as short as possible
    order = sorted(clip_ids,
                   key=lambda c: -(int(corpus.clip_end[c]) - int(corpus.clip_start[c])))
    n_clips = len(order)
    ids_all = torch.arange(n_envs, device=dev)

    succ = fail = 0
    mpkpe_sum = pose_sum = jitter_sum = 0.0
    surv_steps = 0
    jitter_steps = 0
    pass_succ = [0] * repeats
    clip_results = {}

    devices = [dev] if torch.device(dev).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)  # hermetic: does not touch the caller's stream
        for rep in range(repeats):
            for w0 in range(0, n_clips, n_envs):
                wave = order[w0:w0 + n_envs]
                k = len(wave)
                starts = torch.tensor([int(corpus.clip_start[c]) for c in wave], device=dev)
                env.reset_idx(ids_all)                     # fresh sim states everywhere
                env.idx[:k] = starts
                env._write_ref_frame(ids_all[:k], env.idx[:k])
                o = env._build_obs()
                env.history[:] = o.unsqueeze(1).expand(-1, 10, -1).clone()
                env.ep_step.zero_()
                bundle = env._bundle()

                resolved = torch.zeros(n_envs, dtype=torch.bool, device=dev)
                resolved[k:] = True                        # unused slots
                kpe_acc = torch.zeros(n_envs, device=dev)
                pose_acc = torch.zeros(n_envs, device=dev)
                step_acc = torch.zeros(n_envs, device=dev)
                jitter_acc = torch.zeros(n_envs, device=dev)
                # jitter has one fewer term than steps (pairwise deltas) —
                # count its terms separately (review minor: the old code
                # divided steps-1 sums by surv_steps).
                jitter_step_acc = torch.zeros(n_envs, device=dev)
                prev_tgt = None
                max_len = max(int(corpus.clip_end[c]) - int(corpus.clip_start[c]) for c in wave)
                with torch.no_grad():
                    for t in range(max_len + 2):
                        a_clean = (model.act_inference(bundle) if model is not None
                                   else torch.zeros(n_envs, env.n_act, device=dev))
                        action = a_clean
                        if action_noise > 0.0:
                            action = a_clean + torch.randn_like(a_clean) * action_noise
                        # Commanded joint target of the DETERMINISTIC policy
                        # (noise-free, matches the greedy render): the residual is
                        # action_scale * tanh(a). Its step-to-step change is the
                        # jitter signal — using a_clean, not the noised action, so
                        # the injected eval noise doesn't swamp it.
                        tgt = env.action_scale * torch.tanh(a_clean)
                        bundle, _, done, info = env.step(action)
                        ref_pos, _ = env.motion.keypoints_at(env.idx)
                        rob_pos = env.sim.keypoint_pos
                        kpe = (rob_pos - ref_pos).norm(dim=-1).mean(dim=1)
                        ref_root = env.motion.at(env.idx)["base_pos"]
                        ref_quat = env.motion.at(env.idx)["base_quat"]
                        R_s = quat_to_matrix(env.sim.base_quat).transpose(1, 2)
                        R_r = quat_to_matrix(ref_quat).transpose(1, 2)
                        pose = (torch.einsum("nij,nkj->nki", R_s, rob_pos - env.sim.base_pos[:, None, :])
                                - torch.einsum("nij,nkj->nki", R_r, ref_pos - ref_root[:, None, :])
                                ).norm(dim=-1).mean(dim=1)
                        live = ~resolved
                        # Exclude the resolution step from the error accumulators
                        # step() auto-resets done envs BEFORE
                        # returning, so a done env's post-step state is the
                        # teleported fresh-RSI pose (kpe ~ 0) — accumulating it
                        # gave every successful pass one bogus near-zero step.
                        counted = (live & ~done).float()
                        kpe_acc += kpe * counted
                        pose_acc += pose * counted
                        step_acc += counted
                        if prev_tgt is not None:
                            jitter_acc += (tgt - prev_tgt).abs().mean(dim=1) * counted
                            jitter_step_acc += counted
                        prev_tgt = tgt
                        clean = info["done_cause_masks"]["motion_end"] & live
                        failed = done & live & ~clean
                        if bool(clean.any()):
                            succ += int(clean.sum())
                            pass_succ[rep] += int(clean.sum())
                            mpkpe_sum += float(kpe_acc[clean].sum())
                            pose_sum += float(pose_acc[clean].sum())
                            jitter_sum += float(jitter_acc[clean].sum())
                            surv_steps += int(step_acc[clean].sum())
                            jitter_steps += int(jitter_step_acc[clean].sum())
                        fail += int(failed.sum())
                        if per_clip:
                            for mask, ok in ((clean, True), (failed, False)):
                                for s in mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                                    name = corpus.clip_names[wave[s]]
                                    prev = clip_results.get(name) or dict(
                                        success=0, steps=0, kpe_mm=0.0, pose_mm=0.0,
                                        passes=0, surv_passes=0,
                                        _kpe_surv=0.0, _pose_surv=0.0, _surv_steps=0)
                                    st = int(step_acc[s])
                                    kv = float(kpe_acc[s]); pv = float(pose_acc[s])
                                    prev["success"] += int(ok)
                                    prev["passes"] += 1
                                    prev["steps"] = st           # last resolution (back-compat)
                                    prev["kpe_mm"] = round(kv / max(st, 1) * 1000, 1)
                                    prev["pose_mm"] = round(pv / max(st, 1) * 1000, 1)
                                    if ok:                       # survivor-gated aggregates
                                        prev["surv_passes"] += 1
                                        prev["_kpe_surv"] += kv
                                        prev["_pose_surv"] += pv
                                        prev["_surv_steps"] += st
                                    clip_results[name] = prev
                        resolved |= done
                        if bool(resolved.all()):
                            break

    # Fold survivor-gated per-clip sums into means (None if the clip never
    # completed a pass — MPKPE-on-survivors is undefined there). Strip privates.
    for rc in clip_results.values():
        ss = rc.pop("_surv_steps", 0)
        ksum = rc.pop("_kpe_surv", 0.0)
        psum = rc.pop("_pose_surv", 0.0)
        rc["surv_kpe_mm"] = round(ksum / ss * 1000, 1) if ss > 0 else None
        rc["surv_pose_mm"] = round(psum / ss * 1000, 1) if ss > 0 else None

    total = succ + fail
    return dict(
        success=succ, total=total, n_clips=n_clips, repeats=repeats,
        rate=succ / max(total, 1), per_pass=pass_succ,
        mpkpe_mm=mpkpe_sum / max(surv_steps, 1) * 1000,
        pose_mm=pose_sum / max(surv_steps, 1) * 1000,
        # jitter: mean |Δ commanded joint target| per control step over survivor
        # steps, in milliradians. Lower = smoother. Noise-free (deterministic
        # policy command), so comparable across action_scale and to the render.
        jitter_mrad=jitter_sum / max(jitter_steps, 1) * 1000,
        clips=clip_results,
    )


def load_split(path: str) -> dict:
    """Load a split JSON ({train: [names], test: [names], ...})."""
    import json
    with open(path) as f:
        return json.load(f)


def split_clip_ids(corpus, split: dict, role: str) -> list:
    """Corpus clip indices for split[role]; raises on names missing from the
    corpus (a silently shrunken split would look like a metric shift)."""
    name_to_id = {n: i for i, n in enumerate(corpus.clip_names)}
    missing = [n for n in split[role] if n not in name_to_id]
    if missing:
        raise ValueError(f"split[{role}]: {len(missing)} names not in corpus, "
                         f"e.g. {missing[:3]}")
    return [name_to_id[n] for n in split[role]]

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This is a PPO motion-tracking trainer for the Asimov v1 humanoid (23 actuated degrees of freedom), implementing *Robust and Generalized Humanoid Motion Tracking* (RGMT, arXiv:2601.23080) on top of Newton physics. Reference motions are GMR-retargeted clips from the asimov-gmr pipeline. The policy is a dynamics-conditioned command-aggregation network that outputs residual actions on top of PD tracking.

## Commands

Everything runs through `uv` (Python 3.12, torch cu128):

```bash
uv run pytest                                  # all tests
uv run pytest tests/test_reward.py -k name     # single file or test
uv run python -m rgmt.preprocess motion_dir=motions/ cache_dir=cache/   # bake corpus cache (once)
uv run python -m rgmt.train motion_cache=cache/                          # train (Hydra CLI)
uv run python scripts/train_medium.py --cache cache/                     # released-policy recipe
uv run python -m rgmt.view --mode replay|pd|policy --cache cache/ [--clip NAME] [--ckpt ckpt.pt]
uv run python scripts/eval_success_gated.py <ckpt.pt|pd> [n_envs] \
    --action-noise 0.05 --repeats 3            # THE metric (robust success rate)
uv run python scripts/eval_ckpt.py <ckpt.pt>   # clean numerical eval (greedy, no noise)
uv run python scripts/profile_throughput.py    # env-step and update timing breakdown
```

Many tests are CUDA-gated (`skipif not torch.cuda.is_available()`) and skip cleanly on CPU-only machines. There is no linter configured.

Motion source precedence in train and preprocess is `motion_cache > motion_dir > motion_path`. `resume_from=<ckpt.pt>` resumes the model, optimizer, and iteration counter. Checkpoints go to `<log_dir>/<experiment_name>/ckpt_XXXXXX.pt` plus `latest.pt`, and `best_test.pt` tracks the best in-loop test result. For detached full runs, use `scripts/train_medium.py`, which builds an `OmegaConf` config directly without the Hydra CLI.

## Evaluation protocol

The headline number is the success-gated robust evaluation: every clip is rolled start to end (no RSI sampling), success means reaching the clip end without a failure termination, and success is measured under action noise with repeats (`--action-noise 0.05 --repeats 3`) because single deterministic passes proved fragile. Tracking error (MPKPE) is reported only over successful episodes, so precision and survival are not conflated.

- The core lives in `rgmt/eval_gated.py`, which uses hermetic RNG (`fork_rng`). `build_eval_env` pins the historical protocol constants (for example `root_err_done=1.0`); do not tighten them without re-baselining every recorded number. `scripts/eval_success_gated.py` is the thin CLI (`--split/--role` restricts evaluation to one side of a train/test split, and `--dump-clips` writes per-clip outcomes).
- The training loop runs the same protocol in-process every `eval.robust_every` iterations on the held-out test split (`eval.split_json`, for example `rgmt/data/splits/medium.json`). With `eval.mining: true`, a train-split pass refreshes failure-weighted RSI sampling weights (EMA via `eval.mining_ema`); `eval.split_json` supersedes `env.sampling_weights_json`.
- `scripts/analyze_failures.py <clips.json>` autopsies a per-clip dump: it buckets clips by pass-success count, clusters failures by source dataset, length, survival depth, and reference speed, and picks representative clips for `scripts/render_policy.py`.

## Architecture

Data flow: `.npz` clips are loaded by `MotionRef.load` (upsample 30 to 60 fps, finite-difference velocities, FK keypoints), `rgmt.preprocess` bakes per-clip safetensors plus `manifest.json`, `MotionCorpus` serves them to `TrackEnv`, and `PPOTrainer` optimizes `RGMTActorCritic`.

- **`rgmt/data/`**: `joint_map.py` is the canonical source of truth for the 23 actuated joint names in MJCF order (right arm before left arm) plus `KEYPOINT_LINKS`. `motion.py` (`MotionRef.load`) enforces the `.npz` contract (see the README section "Dataset format" for exact shapes, wxyz quaternions, and the 23 joint columns). `corpus.py` holds all clips as concatenated flat tensors with clip-boundary bookkeeping. `cache_key.py` keys the cache on source hash, URDF hash, fps, keypoint links, and schema version, so preprocess skips unchanged clips. `splits/` holds train/test split JSONs.
- **`rgmt/env/`**: `sim.py` (`NewtonSim`) wraps Newton and mujoco-warp around the vendored asset in `rgmt/assets/asimov-v1/` (a freejoint plus 25 hinges: 23 actuated and 2 passive neck joints). `track_env.py` (`TrackEnv`) composes the simulator, corpus, and `reward.py` into the policy bundle with exact shape contracts documented in its docstring: `obs (N,98)`, `history (N,10,98)`, `cmd_window (N,21,55)`, `critic_obs (N,187)` in the paper-base configuration. Training-time command noise (paper Table II) and the termination criteria live here.
- **`rgmt/policy/`**: `RGMTActorCritic`. The `HistoryEncoder` (causal self-attention over the 10-step proprio history, max-pooled) produces a dynamics embedding that the `CommandEncoder` uses as the cross-attention query over the 21-frame command window. The critic is asymmetric, on privileged noise-free observations. `RunningMeanStd` observation normalization is part of the checkpoint.
- **`rgmt/algos/`**: `PPOTrainer` (dual-clip PPO, target-KL early stop, optional KL-shock rollback, optional cosine or KL-adaptive LR) operating on flat bundle dicts from `RolloutBuffer`.
- **`rgmt/train.py`** exposes `run_training(cfg) -> dict`, callable without the Hydra CLI (used by tests and `scripts/train_medium.py`).

Config is Hydra: `rgmt/configs/train.yaml` with groups `env/track`, `algo/ppo`, `network/rgmt`, and `reward/keypoint`. Comments in the config files record the reasoning behind tuning decisions; read them before changing reward weights or PPO hyperparameters. The campaign findings are summarized in `docs/results.md`.

## Conventions and gotchas

- **Quaternions:** input `.npz` is `wxyz` (w first) and is converted to `xyzw` on load. Newton's freejoint qpos is `(x, y, z, qx, qy, qz, qw)`, and its qd is `(lin_vel, ang_vel)`, LINEAR FIRST, like `body_f`'s `(force, torque)`; `body_qd` is also linear-first. The reverse was once assumed and silently swapped every velocity in the pipeline; this is regression-tested in `tests/test_sim_conventions.py`, so trust that test over any documentation. Rotation helpers are in `rgmt/utils/rotation.py`.
- **Joint indexing:** any per-joint tensor is in `ASIMOV_ACTUATED_JOINT_NAMES` order. Mis-ordering silently mis-maps every reference target.
- The corpus is stored **on the GPU by default** (`corpus_device: null` resolves to the training device); set `corpus_device: cpu` only for corpora that do not fit (roughly 0.5 GB per 4 hours of motion).
- **Domain randomization:** `env.dr` (a dict of `DRConfig` kwargs, see `rgmt/env/domain_rand.py`) randomizes foot friction, mass, PD-gain scale, and torque-limit scale per environment. It is train-gated (eval never randomizes); `privileged: true` appends 6 dims to `critic_obs`, so critics of checkpoints trained without it will not load. Draws are episode-consistent and resampled at iteration boundaries so the sim sees **exactly one `notify_model_changed` per iteration** (`JOINT_DOF`/`BODY_INERTIAL` notifies re-derive mass-matrix constants over all worlds, so never notify per reset; setters accumulate flags and `apply_dynamics_changes()` flushes them).
- `env.effort_limits: "urdf"` caps per-joint torque from the URDF datasheet values. This is plant-defining like kp and kd, so eval must match the trained value; all eval scripts read it from the checkpoint config. `foot_friction` is a **no-op for foot-ground contact** (the MJCF foot spheres carry `priority=1, friction=0.6`, and mujoco-warp uses the higher-priority geom's friction exclusively); the real knob is `dr.friction_range`.
- Experiment tracking is **nebo** (`nb.log_line`, per-iteration timings under `train/time/*`); the module-level `nb.md`/`nb.ui` calls in `train.py` run at import. `scripts/nebo_read.py` reads run data back.
- **Directory roles:** `scripts/` holds standalone eval, analysis, and launch tools; `outputs/` is for generated artifacts only (eval dumps, triage CSVs, renders), never scripts, and is gitignored; `docs/` holds public documentation; `data/` (gitignored) is for local datasets only.

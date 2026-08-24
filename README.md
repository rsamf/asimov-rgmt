# Asimov Motion Tracking (RGMT)

A PPO motion-tracking trainer for the [Asimov v1 humanoid](https://github.com/menloresearch/asimov-1) (23 actuated degrees of freedom), implementing [*Robust and Generalized Humanoid Motion Tracking*](https://arxiv.org/abs/2601.23080) (RGMT) on top of [Newton](https://github.com/newton-physics/newton) physics. Reference motions are GMR-retargeted AMASS clips produced by [asimov-gmr](https://github.com/rsamf/asimov-gmr); this repository trains a dynamics-conditioned command-aggregation policy that outputs residual actions on top of PD tracking.

## Pretrained model

A trained policy is published at [`rsamf/asimov-rgmt-medium`](https://huggingface.co/rsamf/asimov-rgmt-medium). It was trained on the easy and medium training splits of the asimov-gmr reference release only, under domain randomization and per-joint URDF torque caps, so its torque commands are ones the real robot can deliver. On the frozen 180-clip held-out test set (60 easy, 60 medium, 60 hard) it scores:

| Metric | Value |
|---|---|
| Robust success rate (action noise 0.05, 3 repeats) | 75.4% ± 0.9 |
| Greedy success rate (no noise) | 75.0% |
| Easy / medium / hard success | 93.9% / 77.2% / 55.0% |
| MPKPE over successful episodes | 96.0 mm |
| Root-relative pose error | 51.9 mm |
| Commanded jitter | 20.5 mrad |

The hard clips were never seen in training, so the 55.0% hard number is zero-shot. See [docs/results.md](docs/results.md) for the evaluation protocol, more detailed results, and the main findings from the training campaign.

```bash
# Download the checkpoint and evaluate it:
hf download rsamf/asimov-rgmt-medium asimov-rgmt-medium.pt --local-dir models/
uv run python scripts/eval_success_gated.py models/asimov-rgmt-medium.pt \
    --split rgmt/data/splits/medium.json --role test --action-noise 0.05 --repeats 3
```

## Quick start

Everything runs through `uv` (Python 3.12, CUDA 12.8 PyTorch wheels).

```bash
# 1. Bake a directory of retargeted clips into an on-disk cache (offline, run once):
uv run python -m rgmt.preprocess motion_dir=motions/ cache_dir=cache/

# 2. Train, sampling across all cached clips:
uv run python -m rgmt.train motion_cache=cache/

# 3. Or reproduce the released policy exactly (full recipe, 34,000 iterations):
uv run python scripts/train_medium.py --cache cache/
```

`rgmt.train` can also build a corpus on the fly from `motion_dir=` (a directory of `.npz` clips) or a single `motion_path=` clip; the precedence is `motion_cache > motion_dir > motion_path`. The offline cache is strongly preferred because it precomputes the expensive per-clip upsampling and forward-kinematics keypoints once. `resume_from=<ckpt.pt>` resumes a run (model, optimizer, and iteration counter).

To obtain training data, follow the [asimov-gmr](https://github.com/rsamf/asimov-gmr) pipeline. It turns your own AMASS copy into the exact reference release this repository trains on, including difficulty labels and the train/test split. AMASS itself carries a non-commercial research license and cannot be redistributed, which is why no motion data ships with either repository.

## Evaluation

The headline number is the success-gated robust evaluation: every clip is rolled from start to end (no reference-state-initialization sampling), and an episode succeeds if it reaches the clip end without a failure termination. Success is measured under action noise with repeats (`--action-noise 0.05 --repeats 3`) because single deterministic passes proved fragile as a metric. Tracking error (MPKPE) is reported only over successful episodes, so precision and survival are not conflated.

```bash
# THE metric (robust success rate):
uv run python scripts/eval_success_gated.py <ckpt.pt|pd> [n_envs] \
    --action-noise 0.05 --repeats 3

# Clean numerical eval (greedy, no noise):
uv run python scripts/eval_ckpt.py <ckpt.pt>

# Environment-step and update timing breakdown:
uv run python scripts/profile_throughput.py
```

`--split rgmt/data/splits/medium.json --role test` restricts evaluation to the held-out test set, and `--dump-clips` writes per-clip outcomes that `scripts/analyze_failures.py` can autopsy (it buckets clips by pass count and clusters failures by source dataset, length, survival depth, and reference speed).

## Visualization

See the robot in the Newton simulator with three input modes:

```bash
# Kinematic playback of a clip (what the motion data looks like, no physics):
uv run python -m rgmt.view --mode replay --cache cache/ --clip "CMU__41__41_05"

# Physics with zero residual action (pure PD tracking, the policy-free baseline):
uv run python -m rgmt.view --mode pd --cache cache/ --clip "CMU__41__41_05"

# A trained checkpoint tracking the clip (greedy, no command noise):
uv run python -m rgmt.view --mode policy --cache cache/ --ckpt models/asimov-rgmt-medium.pt
```

This opens a Newton GL window by default. Close it or press Ctrl-C to exit; clips loop unless `--no-loop` is passed.

| Flag | Meaning |
|---|---|
| `--mode replay\|pd\|policy` | Input mode: kinematic, zero-action PD, or trained checkpoint. |
| `--cache DIR` | Preprocessed corpus directory (or use `--motion-path clip.npz` for one raw clip). |
| `--clip NAME` | Clip to play (filename stem). The default is the first clip; `--list` prints all names. |
| `--ckpt FILE.pt` | Checkpoint for `--mode policy`. |
| `--viewer gl\|null\|rerun\|viser` | Viewer backend; `gl` opens a window and `null` is a headless smoke run. |
| `--steps N` / `--no-loop` | Limit the step count, or stop at the clip end instead of looping. |
| `--fps F` / `--headless` / `--device` | GL pacing, windowless GL, and CUDA device selection. |

`scripts/render_policy.py` renders a side-by-side reference-versus-policy MP4 offscreen, with no display needed.

## Architecture

Data flows as follows: `.npz` clips are loaded by `MotionRef.load` (upsampling 30 to 60 fps, finite-difference velocities, forward-kinematics keypoints), `rgmt.preprocess` bakes per-clip safetensors plus a manifest, `MotionCorpus` holds all clips as concatenated flat tensors, `TrackEnv` composes the simulator, corpus, and reward into the policy input bundle, and `PPOTrainer` optimizes `RGMTActorCritic`.

- **`rgmt/data/`** holds the data layer. `joint_map.py` is the canonical source of truth for the 23 actuated joint names in MJCF order plus the keypoint links. `motion.py` enforces the `.npz` contract described below. `corpus.py` provides clip-boundary bookkeeping over flat tensors, and `cache_key.py` keys the preprocess cache on source hash, URDF hash, frame rates, keypoint links, and schema version so unchanged clips are skipped.
- **`rgmt/env/`** holds the simulation layer. `sim.py` wraps Newton and mujoco-warp around the vendored robot asset in `rgmt/assets/asimov-v1/` (a freejoint plus 25 hinges: 23 actuated and 2 passive neck joints). `track_env.py` composes the simulator, corpus, and `reward.py` into the policy bundle, with exact shape contracts documented in its docstring: `obs (N,98)`, `history (N,10,98)`, `cmd_window (N,21,55)`, `critic_obs (N,187)` in the paper-base configuration. Training-time command noise (paper Table II) and the termination criteria live here. `domain_rand.py` implements per-environment dynamics randomization for sim-to-real transfer.
- **`rgmt/policy/`** holds `RGMTActorCritic`. A causal self-attention `HistoryEncoder` over the 10-step proprioception history produces a dynamics embedding, which the `CommandEncoder` uses as the cross-attention query over the 21-frame command window. The critic is asymmetric and sees privileged noise-free observations. The `RunningMeanStd` observation normalizer is part of the checkpoint.
- **`rgmt/algos/`** holds `PPOTrainer`: dual-clip PPO with a target-KL early stop, an optional KL-shock rollback guard, optional cosine or KL-adaptive learning-rate control, and optional action-smoothness regularizers.

Configuration is Hydra: `rgmt/configs/train.yaml` with groups `env/track`, `algo/ppo`, `network/rgmt`, and `reward/keypoint`. Comments in the config files record the reasoning behind tuning decisions; read them before changing reward weights or PPO hyperparameters.

## Throughput notes

- The corpus is stored on the GPU by default (`corpus_device: null` resolves to the training device; a 4-hour corpus is roughly 0.5 GB). This removes the per-step CPU gather and host-to-device copy, which accounted for about 40% of environment-step time when the corpus was CPU-resident. Set `corpus_device: cpu` only for corpora that do not fit.
- `train.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce allocator fragmentation. As a reference point, an RTX 3070 (8 GB) reaches about 70k environment steps per second at 2048 environments and about 104k at 4096 environments at roughly 5.9 GB peak memory.
- Per-iteration timings are logged through [nebo](https://pypi.org/project/nebo/) (`train/time/rollout_s`, `train/time/update_s`, `train/time/env_steps_per_s`), and `scripts/profile_throughput.py` gives a step-time breakdown (simulation versus command window versus policy) for tuning new hardware.

## Dataset format

Reference motion data is **not** generated by this repository. You supply GMR-retargeted clips (the [asimov-gmr](https://github.com/rsamf/asimov-gmr) pipeline produces them); the format below is the contract `rgmt.data.motion.MotionRef.load` enforces.

### Per-clip file: a NumPy `.npz`

There is one `.npz` per motion clip, each containing exactly three arrays that share the same first dimension `F` (the source frame count):

| Key | Shape | Meaning | Convention |
|---|---|---|---|
| `base_frame_pos`  | `(F, 3)` | Root (pelvis) world position | Meters, world frame `(x, y, z)`. |
| `base_frame_wxyz` | `(F, 4)` | Root orientation quaternion | **`wxyz`** (w first); converted to `xyzw` and re-normalized on load. |
| `joint_angles`    | `(F, 23)`| Actuated joint angles | Radians, in `ASIMOV_ACTUATED_JOINT_NAMES` order. |

The dtype is flexible; all arrays are cast to `float32` on load.

The loader enforces two checks and raises if they are violated:
- `joint_angles` must have exactly 23 columns: the actuated joints only, not 25 (no neck) and not 27 (no toes).
- The quaternion is read as `wxyz` (w first), not `xyzw`. GMR output is already `wxyz`, so do not pre-convert it.

### Joint-angle column order (the 23 actuated joints)

```
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll,
right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll,
waist_yaw,
right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_yaw,
left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_yaw
```

Note the right-arm-before-left-arm ordering, which is the canonical MJCF order. The columns must match this exactly, or every reference target is mis-mapped. The canonical list lives in `rgmt/data/joint_map.py::ASIMOV_ACTUATED_JOINT_NAMES`.

### Frame rate

`src_fps` must evenly divide `physics_fps` (the defaults are 30 and 60), so clips should be at 30 fps (or 60). The loader SLERP- or linearly upsamples to the physics rate and computes velocities by finite difference. You supply only per-frame pose, no velocities.

### What you do not provide

- No velocity arrays (they are computed by finite difference).
- No keypoint or link Cartesian positions (they are computed by forward kinematics at preprocess time).
- No neck or toe joints, only the 23 actuated joints.

### Corpus layout

For multi-clip training, place clips in a directory. The filename stem becomes the clip name, and the cache key is per clip:

```
motions/
  walk_01.npz
  dance_03.npz
  ...
```

### Minimal example of writing a valid clip

```python
import numpy as np

np.savez(
    "walk_01.npz",
    base_frame_pos  = pos.astype(np.float32),        # (F, 3) meters, world frame
    base_frame_wxyz = quat_wxyz.astype(np.float32),  # (F, 4) quaternion, w first
    joint_angles    = q.astype(np.float32),          # (F, 23) radians, canonical order
)
```

### Caveats

- **Grounding.** `base_frame_pos[:, 2]` is the root height used by reference-state initialization and the height and keypoint rewards. Retargeting that leaves the robot floating above the ground feeds that float straight into training: reference-state initialization drops the robot from the air, and the keypoint reward asks for feet above the floor. Ground the clips at the source by shifting each clip's `z` so the lowest foot-sole contact sits at the floor.
- **Clip length.** A clip must survive upsampling with command-window lookahead room. With `L=10`, sampling needs roughly 21 or more upsampled frames of room, so very short clips (a few frames) are unusable. Aim for clips of at least one second.

## Tests

```bash
uv run pytest                                  # all tests
uv run pytest tests/test_reward.py -k name     # single file or test
```

Many tests are CUDA-gated and skip cleanly on CPU-only machines.

## License

The code is licensed under the [Apache License 2.0](LICENSE). The published model weights are likewise Apache 2.0. Motion data is not included; AMASS and the SMPL-X body models carry their own non-commercial research licenses.

## Citation

This repository is an independent implementation of the RGMT paper. If you build on it, please cite the paper:

```bibtex
@article{ma2026rgmt,
  title   = {Robust and Generalized Humanoid Motion Tracking},
  author  = {Ma, Yubiao and Yu, Han and Xie, Jiayin and Lv, Changtai and Luo, Qiang and Zhang, Chi and Yin, Yunpeng and Xing, Boyang and Ren, Xuemei and Zheng, Dongdong},
  journal = {arXiv preprint arXiv:2601.23080},
  year    = {2026}
}
```

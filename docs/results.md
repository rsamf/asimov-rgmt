# Results and findings

This document summarizes the training campaign behind the released checkpoint, [`rsamf/asimov-rgmt-medium`](https://huggingface.co/rsamf/asimov-rgmt-medium), and the findings that shaped the current defaults. All numbers were measured with the evaluation protocol below on the [asimov-gmr reference release](https://github.com/rsamf/asimov-gmr): 3,221 training clips (8.50 hours at 30 fps) with frozen easy, medium, and hard difficulty labels, and a frozen 180-clip held-out test set with 60 clips per difficulty.

## Evaluation protocol

The headline metric is the success-gated robust success rate:

- Every clip is rolled from its start to its end. There is no reference-state-initialization sampling at evaluation time.
- An episode succeeds if it reaches the clip end without a failure termination (a fall, or a tracking-failure termination such as excessive root drift).
- Success is measured under Gaussian action noise (standard deviation 0.05 in normalized action units) with three repeats per clip, because single deterministic passes proved fragile as a metric. A policy is scored on its success neighborhood, which is the quantity that transfers.
- The greedy (zero-noise) success rate is reported alongside it. Injected evaluation noise can act as beneficial dither for some policies, so quoting both keeps the deterministic behavior honest.
- MPKPE (mean per-keypoint position error) is reported only over successful episodes, so precision and survival are not conflated. Root-relative pose error reports the same quantity in the root frame, which removes global drift from the number.
- Evaluation always runs the deterministic plant. Domain randomization is training-only, but plant-defining settings such as torque limits must match the values the policy trained under. The evaluation scripts read them from the checkpoint config.

## The released policy

The released checkpoint was trained on the easy and medium training clips only (2,389 clips). The hard training clips were excluded entirely, so every hard number below is zero-shot. The run carried the full sim-to-real configuration:

- Per-joint torque caps taken from the URDF datasheet values. Earlier policies trained with uncapped actuators, which a real robot cannot deliver.
- Domain randomization over foot friction (uniform on [0.4, 1.0]), whole-robot mass scale ([0.9, 1.1]), PD-gain scale ([0.9, 1.1]), and torque-limit scale ([0.8, 1.2]). Draws are episode-consistent and resampled at iteration boundaries.
- A privileged critic that sees the sampled dynamics parameters, while the actor must infer them from its 10-step proprioception history. This exercises the RGMT architecture's system-identification premise directly.
- An exact reproduction recipe is provided in `scripts/train_medium.py` (34,000 iterations at 8,192 environments).

Results on the frozen 180-clip test set:

| Difficulty | Test success | Train-side success |
|---|---|---|
| Easy | 93.9% | 99.0% (trained) |
| Medium | 77.2% | 89.2% (trained) |
| Hard (zero-shot) | 55.0% | 61.8% (never trained) |
| All | **75.4% ± 0.9** (greedy 75.0%) | 85.7% |

Precision on successful test episodes: 96.0 mm MPKPE, 51.9 mm root-relative pose error, and 20.5 mrad commanded jitter.

Two aspects of these numbers are worth calling out:

1. **The robust and greedy rates agree** (75.4% versus 75.0%). For a comparable policy trained without domain randomization, injected evaluation noise inflated the robust rate by roughly three points over greedy. The agreement here is itself evidence that the policy is genuinely noise-robust rather than dither-assisted.
2. **The sim-to-real configuration costs about two points.** A policy trained with the same recipe but without domain randomization and without torque caps reaches 77.4% robust success (78.3% greedy) with 84.9 mm MPKPE and 45.4 mm pose error. The two-point gap buys robustness across a family of dynamics instead of one exact plant, and torque commands a real robot can deliver.

## Findings from the campaign

The campaign's main lessons, roughly in order of impact:

**Data quality dominates.** The single largest gains came from corpus curation, not from algorithm changes: retargeting fixes, grounding, glitch filtering, difficulty labeling, and per-clip failure analysis feeding the next dataset round. The asimov-gmr repository documents that pipeline. Failure-weighted reference-state-initialization mining (upweighting clips the policy currently fails) also refreshes itself from the in-loop evaluation.

**A KL-shock rollback guard makes long runs survivable.** PPO's standard target-KL early stop checks after a minibatch update has been applied, so a single catastrophic update can slip through. One such update (approximate KL of 0.08) killed a 30,000-iteration run outright during the entropy anneal, dropping the average return from 297 to 7. The guard (`kl_shock_factor: 3.0`) snapshots model and optimizer state at iteration start and rolls the whole iteration back when any minibatch exceeds three times the target KL. On one reward configuration the guard rolled back 102 destructive updates over 34,000 iterations with zero collapses. It is a precondition for some configurations, not a safety net.

**The critic was silently rate-limited by two defaults.** With returns of order 350 and a value-clip window of 0.2, the clipped value loss bound 95% of samples, so the critic could not take real steps. Separately, a single shared gradient-norm clip let the critic's large gradient norm (about 187) scale the actor's effective step by roughly 0.005. Switching to a plain MSE value loss with separate actor and critic clip budgets (`value_clip: false`, `grad_clip_per_module: true`) halved the value loss over the following run and reduced KL-shock rollbacks from 102 to 5 under an otherwise comparable setup.

**Rebalancing global versus local reward terms bought local fidelity for free.** Root drift was effectively triple-counted: world-frame keypoints inherit root drift into every keypoint while the dedicated root-position and root-orientation terms charge for it again. Shifting weight from world keypoints toward root-relative keypoints (`w_kp` 1.0 to 0.6, `w_rel` 0.3 to 0.6, `w_rq` 0.6 to 0.4) improved root-relative pose error by 15% while world-frame MPKPE stayed flat, at the cost of about 3 mrad of extra jitter. The rebalance also made training markedly shockier, which is what promoted the KL-shock guard from optional to load-bearing.

**Action smoothness is best regularized on the actor's mean, not the sampled action.** Two hinge penalties in tanh space, both with speed-proportional budgets, replaced a reward-side action-rate penalty: a temporal penalty on the change of the mean action between steps, and a spatial penalty implementing a variable Lipschitz bound on the local input-output gain, estimated by finite differences over a noise ball on the normalized observation. Together with a first-order low-pass filter on the commanded residual (`action_filter_alpha: 0.7`, trained through, so the filter is part of the plant), commanded jitter dropped by roughly 40% in the most smoothness-focused configuration, at a cost of a few points of success in that configuration. The released recipe uses moderate settings of both penalties (`lambda_smooth: 0.1`, `lambda_spatial: 0.01`).

**Planar drift must be made observable.** The paper's command representation is yaw-invariant and carries no positional feedback, so planar and heading drift are unobservable to the policy and integrate open-loop over long horizons. Appending five drift-feedback features per command frame (body-frame position offset and heading-error cosine and sine) closes that loop while remaining invariant to rigid world transforms of the robot and reference pair.

**Precision improves through the success plateau.** Success rate saturates first; tracking error keeps improving for thousands of iterations afterward. Ending runs at the success plateau leaves precision on the table.

**The entropy anneal is productive once de-risked.** Annealing the entropy bonus to a small floor over the last 30% of the run sharpens the mean policy for the low-noise regime it is deployed in. Without the shock guard this phase destroyed runs; with it, the anneal reliably produces the best checkpoints of each run.

## Reproducing

```bash
# Build the corpus with asimov-gmr, preprocess it, then:
uv run python scripts/train_medium.py --cache cache/

# Evaluate any checkpoint on the held-out test set:
uv run python scripts/eval_success_gated.py runs/rgmt_medium/best_test.pt \
    --split rgmt/data/splits/medium.json --role test --action-noise 0.05 --repeats 3
```

Training numbers were produced on a single RTX 5090; the full 34,000-iteration run takes roughly two days at 8,192 environments. The in-loop robust evaluation runs every 600 iterations on the held-out test split and drives both checkpoint selection (`best_test.pt`) and the failure-mining weights.

# Idea: Modest Label Smoothing on the Validated Long-Plateau Schedule

## Summary

Compose one low-cost generalization intervention with the successful EXP-002 optimizer policy: change hard-label cross-entropy to cross-entropy with `label_smoothing=0.05`, while leaving the 80% high-learning-rate plateau, standard momentum, terminal cosine refinement, architecture, augmentation, data loading, evaluation schedule, and seed unchanged.

EXP-002 raised `best_test_acc` from 91.67% to 91.83%, but the model still drives its late smoothed training loss to about 0.06 while final test loss remains 0.2843. Those losses are not directly comparable because they are measured on augmented training batches versus the fixed test set, yet the large separation is consistent with highly confident fitting of the training labels and remaining generalization error. A modest soft-target penalty directly addresses that gap without consuming a meaningful fraction of the fixed 300-second training budget.

## Diagnosis

The optimizer horizon is no longer the most obvious limiter. EXP-002 established that this model benefits from preserving high-LR exploration for 80% of counted time and then stepping to `lr=0.01` before a cosine decay to `1e-4`. It reached 91.83%, improved over the prior baseline by 0.16 percentage points, and had only a 0.01-point best-versus-final accuracy gap. The schedule therefore converges reliably enough to serve as a controlled composition base.

The remaining signal is generalization rather than failure to optimize:

- Late training loss near 0.06 under hard-label CE implies that the network assigns very high probability to augmented training targets.
- Final test loss of 0.2843 and final accuracy of 91.82% show that this confidence does not transfer perfectly.
- The model already has random crop and horizontal flip, but no explicit target-space regularization.
- Label smoothing penalizes extreme class probabilities and can reduce sensitivity to idiosyncratic or ambiguous training labels at effectively zero memory and throughput cost.

This diagnosis is suggestive, not a proof of calibration error: the train and test losses use different samples and transforms. The experiment is therefore framed as a small, falsifiable regularization test rather than an assumption that stronger smoothing must help.

## Mechanism

For ten classes and smoothing strength `epsilon=0.05`, replace the hard one-hot target with the distribution used by PyTorch cross-entropy:

```text
q_correct = 1 - epsilon + epsilon / 10 = 0.955
q_other   = epsilon / 10               = 0.005
```

Equivalently, the loss mixes 95% ordinary negative log likelihood with 5% cross-entropy against a uniform class distribution. The uniform component prevents the optimizer from receiving an unchecked incentive to increase an already-correct logit margin. This can reduce overconfident memorization, encourage less brittle representations, and improve accuracy on held-out images.

The proposed `0.05` strength is intentionally below the commonly used `0.1` cited in the label-smoothing literature. ResNet-20 has limited capacity and this run provides only about 100 epochs; excessive smoothing could weaken the class-discriminative signal or combine with high-LR implicit regularization to underfit. A 5% mixture is large enough to test the mechanism while keeping 95.5% target mass on the correct class.

Label smoothing changes the numerical scale and floor of training loss. With `epsilon=0.05` and ten classes, even a prediction matching the smoothed target has cross-entropy of roughly 0.28, so the new logged training loss must not be compared directly with EXP-002's hard-label value near 0.06. Test loss remains ordinary evaluator loss and is directly comparable.

## Exact Candidate Setting

- `LABEL_SMOOTHING = 0.05`
- Preserve EXP-002 exactly:
  - `BATCH_SIZE = 128`
  - `LR = 0.1`
  - `LR_HOLD_FRACTION = 0.8`
  - discontinuous transition to `ANNEAL_START_LR = 0.01`
  - elapsed-budget cosine decay from `0.01` to `MIN_LR = 1e-4` over the final 20%
  - ordinary SGD momentum `0.9`, without Nesterov
  - weight decay `1e-4`
  - persistent DataLoader workers
  - evaluation checkpoints `(0.2, 0.4, 0.6, 0.7)` plus dense once-per-epoch evaluation during the final 20%
  - seed 42 and all current model, initialization, transform, normalization, and loader behavior
- Change only the training loss call from:

  ```python
  loss = F.cross_entropy(outputs, targets)
  ```

  to:

  ```python
  loss = F.cross_entropy(
      outputs,
      targets,
      label_smoothing=LABEL_SMOOTHING,
  )
  ```

No change is made to `Eval.evaluate()`. Evaluation must continue using the fixed test labels and ground-truth loss in `prepare.py`.

## Evidence

- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/002/04-analysis.md`: EXP-002 produced 91.83% best accuracy, 91.82% final accuracy, 0.2843 final test loss, 38,629 steps in 300.0 seconds, and only a 0.01-point best/final gap. Its next-step recommendation is an isolated regularization intervention on the validated schedule.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/03-experiment-learnings.md`: the 80% high-LR plateau plus terminal refinement is a medium-importance validated pattern and should be preserved; early annealing is a recorded failed approach.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv`: 91.83% is the moving baseline, so EXP-003 must reach at least 91.93% to count as an improvement.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/knowledge/papers/label-smoothing.md`: Muller, Kornblith, and Hinton report that soft targets often improve generalization and calibration and produce tighter within-class representation clusters; the distillation downside is irrelevant here.
- `train.py`: the current loss is hard-label `F.cross_entropy`, and no other soft-label regularizer is active.

## Expected Impact

**Primary hypothesis:** `label_smoothing=0.05` will raise `best_test_acc` from 91.83% to at least 91.93%, with a plausible result of approximately 91.95-92.10% (+0.12 to +0.27 percentage points), by reducing excessive confidence without disrupting the validated optimization horizon.

**Secondary expectations:** final test loss may decrease even though the training loss increases substantially by construction; best and final accuracy should remain close because the EXP-002 tail already converges smoothly. Optimizer steps, counted training time, total runtime, peak VRAM, and evaluation count should remain effectively unchanged.

The expected gain is deliberately modest. Label smoothing does not add new examples or model capacity, and EXP-002 is already reasonably regularized by crop/flip augmentation and a long high-LR phase. Its value is the favorable cost-to-risk ratio under a strict fixed-time benchmark.

## Failure Modes and Interpretation

- **Over-regularization:** if best accuracy falls while late accuracy remains smooth, 0.05 may be too strong in combination with the long high-LR plateau. The clean follow-up is `epsilon=0.025`, not a simultaneous scheduler or augmentation change.
- **Insufficient effect:** a result within noise of 91.83% may mean the remaining error is not driven by confidence, or that 0.05 is too weak. Do not reroll the seed. A future strength test should be predeclared and retain all other settings.
- **Slower fitting:** smoothing reduces the gradient incentive for large class margins. If checkpoint accuracies lag EXP-002 throughout the high-LR phase and never recover, the finite 300-second horizon may be too short for this intervention.
- **Misread training loss:** the smoothed training loss cannot approach the old 0.06 scale. Treating its expected higher floor as failed optimization would be an analysis error; compare accuracy trajectories and fixed-evaluator test loss instead.
- **Test-loss/accuracy divergence:** smoothing may improve calibration and lower test loss without changing top-1 accuracy. That is scientifically informative but is still a no-improvement under the declared `best_test_acc` objective.
- **Throughput regression:** PyTorch's built-in implementation should have negligible overhead, but verify steps and runtime. An unexpected material drop in step count could erase the regularization benefit under the fixed budget.
- **Run variance near threshold:** the required gain is only 0.10 percentage points. Use the fixed seed and single predeclared run; do not repeat until a favorable sample appears.

## Why This Experiment Is Isolated

EXP-001 bundled an early cosine schedule with Nesterov and regressed, making causal attribution difficult. EXP-002 deliberately repaired that ambiguity and established a successful optimizer base. EXP-003 should now add exactly one mechanism: target smoothing.

Specifically, this experiment does not change the plateau fraction, terminal LR curve, momentum variant, weight decay, batch size, augmentation, model capacity, evaluation cadence, worker configuration, precision, averaging, seed, or data order logic. It also does not combine label smoothing with Mixup, because both create soft targets and may over-regularize this small model. This de-bundling makes the comparison actionable: any accuracy or throughput difference from EXP-002 is attributable to the loss intervention, subject to ordinary deterministic-runtime variation.

## Implementation Sketch

1. Add one hyperparameter beside the other training constants:

   ```python
   LABEL_SMOOTHING = 0.05
   ```

2. Pass it to the existing training-only `F.cross_entropy` call as shown above.
3. Leave evaluation untouched. Do not apply smoothing in `Eval.evaluate()` and do not modify `prepare.py`.
4. Optionally print the smoothing constant once at startup for provenance; avoid per-step diagnostic work.
5. Run Ruff/static checks and inspect the diff to confirm that only these loss-related lines changed in `train.py`.

## Fixed-Budget Feasibility

The intervention uses PyTorch's existing cross-entropy API, adds no dependency, allocates no persistent model-sized state, and requires no extra forward pass, backward pass, batch, epoch, or evaluation. Its arithmetic overhead is tiny relative to the ResNet forward/backward pass. It should therefore preserve approximately 38,600 optimizer steps, 300 seconds of counted training, the 336-second EXP-002 total runtime, and roughly 330 MB peak VRAM, comfortably under the 600-second cap on one H20.

## Verification

Before execution:

- Confirm exactly one NVIDIA H20 with approximately 98 GB VRAM is visible.
- Confirm the moving baseline is 91.83% and the acceptance threshold is therefore 91.93%.
- Confirm seed 42 is unchanged, no stale run log exists, and `git diff` shows only `train.py` loss-related edits.
- Run syntax/Ruff/pre-commit checks appropriate to the repo.

Execute exactly once with the required redirected command:

```bash
uv run train.py > run.log 2>&1
```

Terminate and mark failure if total runtime exceeds 10 minutes. After completion:

- Require a valid numeric summary and `training_seconds` approximately 300 seconds.
- Require `best_test_acc >= 91.93%` for an improvement over EXP-002.
- Confirm `num_steps`, `total_seconds`, and `peak_vram_mb` remain close to EXP-002; investigate any material throughput loss before attributing the metric solely to smoothing.
- Compare the fixed-evaluator accuracy trajectory, final test loss, best/final gap, and final accuracy with EXP-002. Do not compare raw smoothed training loss magnitude to the hard-label run.
- Confirm validation ran no more than once in any epoch and the process stayed below 600 seconds.
- Record the result without retries or seed changes, then remove `run.log` as required before any later experiment.

## Decision Rule and Follow-Up

- **Improvement:** retain `label_smoothing=0.05` only if `best_test_acc >= 91.93%` and all integrity checks pass.
- **No improvement with clear underfitting:** revert EXP-003 and consider a separately planned `0.025` smoothing test.
- **No improvement with unchanged optimization trajectory:** revert EXP-003 and move to a different isolated generalization mechanism rather than stacking more soft-target regularization.
- **Invalid/crash/timeout:** fix only the implementation or protocol fault and rerun the same predeclared experiment; do not alter the smoothing strength opportunistically.

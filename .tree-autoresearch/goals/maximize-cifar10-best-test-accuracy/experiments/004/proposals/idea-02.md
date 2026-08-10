# Proposal: Clean-Only Two-Stage Label Smoothing

## Summary

Extend the accepted EXP-002 recipe with mild label smoothing only on clean batches during the first 75% of charged training time. Keep existing CutMix batches on their current two-label objective, and use ordinary hard-label cross-entropy for every batch during the final 25%. The intervention is deliberately narrow: it adds an output-space regularizer to early clean supervision without further softening CutMix targets or disturbing the validated clean late-refinement phase.

## Mechanism

EXP-002 improved `best_test_acc` from 94.62% to 95.23% by applying CutMix to 10,257 of 20,668 eligible early batches, but the other roughly half of early batches still used fully hard targets (`experiments/002/04-analysis.md`, Results). Small label smoothing on those clean batches changes the target from a one-hot vector to a mixture of the one-hot target and the uniform class distribution. With PyTorch's convention and 10 classes, `epsilon=0.05` assigns probability `0.955` to the correct class and `0.005` to each other class. This discourages extreme clean-example logits and promotes the tighter class representations associated with improved classification generalization in Muller et al. (`experiments/004/papers/when-label-smoothing-helps.md`).

The restriction to clean batches matters. A CutMix loss already represents the soft target `lambda*y_a + (1-lambda)*y_b`; applying label smoothing to both constituent cross-entropies would instead optimize `(1-epsilon)*(lambda*y_a + (1-lambda)*y_b) + epsilon*uniform`, adding a second softening mechanism with no evidence that it is additive. Clean-only smoothing gives every early batch a generalization intervention while preserving CutMix's validated target semantics.

The two stages are tied to charged-time progress rather than epoch or step count:

- **Stage 1, `0.00 <= progress < 0.75`**: CutMix-selected batches retain the existing area-corrected two-term loss with no label smoothing. Non-CutMix batches use hard targets with `label_smoothing=0.05`.
- **Stage 2, `0.75 <= progress <= 1.00`**: CutMix is already disabled and label smoothing is also disabled, so all batches use the exact EXP-002 hard-target loss for low-LR refinement.

The `0.75` boundary matches both `CUTMIX_END` and `DROP_PATH_DECAY_START`. It therefore preserves EXP-002's successful phase structure rather than introducing another transition. `epsilon=0.05` is intentionally conservative because CutMix and drop path already regularize the early phase; the proposal tests a distinct mechanism, not a broad regularization-strength escalation like the unstable EXP-003 grid.

## Concrete Change

Modify only `train.py` from the EXP-002 base.

1. Add explicit constants near the existing CutMix hyperparameters:

   ```python
   LABEL_SMOOTHING = 0.05
   LABEL_SMOOTHING_END = CUTMIX_END
   ```

2. Include both values and `label_smoothing_scope=clean_only` in the startup configuration log. This makes the preregistered recipe auditable without changing any required summary key.

3. Leave the CutMix gate, `cutmix_batch`, target pairing, rectangle-area correction, and dedicated CPU/CUDA generators unchanged. After the single model forward, replace only the clean branch of the current loss selection:

   ```python
   if targets_b is None:
       smoothing = LABEL_SMOOTHING if progress < LABEL_SMOOTHING_END else 0.0
       loss = F.cross_entropy(
           outputs,
           targets_a,
           label_smoothing=smoothing,
       )
   else:
       loss = adjusted_lam * F.cross_entropy(outputs, targets_a)
       loss += (1.0 - adjusted_lam) * F.cross_entropy(outputs, targets_b)
   ```

   The mixed branch must not pass `label_smoothing`; its behavior must remain bit-for-logic identical to EXP-002. The strict `<` boundary makes the first batch at or beyond 75% fully hard-label, matching the existing CutMix cutoff.

4. Optionally maintain an integer `label_smoothed_clean_batches` counter and print it in a non-summary audit line at the end. This counter must increment only when `targets_b is None` and `progress < LABEL_SMOOTHING_END`. Based on EXP-002 exposure, approximately 10,411 early clean batches should be smoothed, although the exact count is determined by the unchanged fixed CutMix RNG stream and achieved step throughput.

No model, optimizer, LR schedule, drop-path schedule, data transform, evaluator, validation cadence, timer, seed, or required final-summary field changes.

## RNG Preservation

`F.cross_entropy(..., label_smoothing=0.05)` is deterministic and consumes no random numbers. The implementation must not introduce a new generator, random gate, or stochastic per-example smoothing decision. Because the CutMix decision remains before the loss branch and its dedicated seed-42 CPU/CUDA generator calls are unchanged, every CutMix gate, lambda, center, and permutation draw follows the same stream as EXP-002 for a given sequence of steps. The global seed-42 stream used by data loading and drop path is likewise not advanced by this change. Different gradients will change parameters, as intended, but not RNG consumption.

## Fixed-Budget and Runtime Protocol

The smoothed clean loss remains inside the existing `t0` through `torch.cuda.synchronize()` interval, so all computation is charged to the unchanged `TIME_BUDGET_S=300` training budget. It requires no second forward pass, additional model, dependency, or persistent GPU tensor. Native cross-entropy adds an `O(batch_size * num_classes)` uniform-loss term on only about 37.5% of all steps (`75%` early window times roughly `50%` clean gate), which is negligible beside the WRN convolutions. Expected step-time overhead is below 1%; optimizer exposure should remain close to EXP-002's 27,950 steps, subject to normal runtime variation. Peak VRAM should remain approximately EXP-002's 1,178.9 MiB.

Execute exactly one experiment from the repository root with physical GPU 0 exposed:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
```

The run must retain one frozen `Eval.evaluate` call per epoch, stop after the fixed charged budget, complete within the 10-minute outer limit, and print the complete summary. Do not tune from intermediate test accuracy, retry a seed, or alter the recipe after observing the metric.

## Evidence

- **Direct parent evidence**: EXP-002 reached 95.23%, improved final test loss to 0.2044, and verified that an early soft-label intervention plus a clean final quarter raises generalization despite 840 fewer steps (`experiments/002/04-analysis.md`). This proposal preserves that entire validated path.
- **Mechanism evidence**: Muller, Kornblith, and Hinton report that label smoothing can improve generalization and calibration by changing learned class representations (`experiments/004/papers/when-label-smoothing-helps.md`). The paper distillation specifically recommends small smoothing on clean examples and warns against assuming additivity with mixed soft labels.
- **Negative local evidence**: EXP-003 found that changing only CutMix probability and drop-path strength did not produce a confirmed improvement; selected gains regressed on confirmation (`experiments/003/04-analysis.md`). Clean-only label smoothing is a qualitatively distinct output-space mechanism, not another point in that failed scalar grid.
- **Code fit**: `train.py` already exposes the exact clean-versus-CutMix branch and charged-time `progress`, so the intervention is a small loss-only change using the installed PyTorch API. No evaluator or data-pipeline modification is needed.

## Strongest Risk

The main risk is redundant early soft supervision. With this change, every batch in the first 75% receives either strong two-class CutMix targets or mild uniform label smoothing, while drop path remains active. That could reduce useful fitting, especially because EXP-003 showed that apparent differences around this operating point can be noisy. Restricting smoothing to clean batches, fixing `epsilon=0.05`, and removing it completely for the final quarter limit this risk without bundling a compensating optimizer or architecture change. A result below 95.33% should be treated as no improvement, not followed by strength or seed retries in the same experiment.

A secondary interpretability risk is that label smoothing raises training cross-entropy by design, so the smoothed loss cannot be compared numerically with EXP-002's hard-target training loss as an overfitting diagnostic. Primary and frozen test metrics remain comparable.

## Tests and Verification

Before the full run:

1. Run a deterministic CPU loss smoke with fixed logits and targets. Verify PyTorch's clean smoothed loss equals `(1-0.05) * NLL + 0.05 * mean_negative_log_probability` within tolerance.
2. Verify the loss branch matrix: early clean uses `0.05`; early CutMix uses the unchanged two-term loss; clean at exactly `progress=0.75` uses `0.0`; late clean uses `0.0`.
3. Snapshot the global CPU RNG state and both CutMix generator states around the clean label-smoothing loss smoke and verify they do not change. Do not add any smoke execution to the timed production path.
4. Compile `train.py` and check the startup configuration reports `label_smoothing=0.05`, `label_smoothing_end=0.75`, and `clean_only`.

After the full run, verify GPU 0 identity, exit code, absence of NaN/Inf or CUDA errors, `training_seconds` near 300, total runtime below 600 seconds, one evaluation per completed epoch, all required summary keys, unchanged `num_params=2,748,890`, and the expected CutMix exposure near 0.5. If the optional smoothing counter is implemented, verify it equals eligible early batches minus applied CutMix batches.

## Testable Hypothesis

Relative to the accepted EXP-002 parent at 95.23%, clean-only `epsilon=0.05` label smoothing during the first 75% of charged time will improve output-space generalization without disrupting CutMix or late fitting, producing `best_test_acc >= 95.33%` in the single fixed-seed run. The expected gain is modest (approximately 0.10-0.30 percentage points); success requires the primary threshold, not merely improved calibration or final loss.

## Effort

**Low.** The implementation is a pair of constants, one logged configuration field, and a change confined to the existing clean loss branch, plus focused smoke checks and the standard single GPU run.

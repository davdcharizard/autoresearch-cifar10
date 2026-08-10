# Proposal: Front-Loaded CutMix with Time-Normalized Cosine Decay and EMA

## Summary

Use a deliberately small regularization package: probabilistic CutMix during the first 70% of training time, a cosine learning-rate schedule normalized to the fixed 300-second budget, and a sparsely updated exponential moving average (EMA) for evaluation. Keep the ResNet-20 architecture, batch size, optimizer, weight decay, seed, and existing crop/flip transforms unchanged.

This is a coherent combination rather than a collection of independent tricks. Early CutMix shapes the representation while the learning rate is high; removing CutMix for the final 30% restores direct clean-example supervision as the cosine schedule settles the solution; EMA then reduces the checkpoint noise that is especially visible when `best_test_acc` is selected from only 89 epoch-end evaluations. Do not add label smoothing in this experiment: CutMix already creates soft targets, and applying both would increase the risk of underfitting this 272K-parameter model while making the result harder to attribute.

## Baseline and Limiter

- Parent: `BASE` at `91.51% best_test_acc`.
- Measured horizon supplied for this proposal: 34,435 steps in 89 epochs under the 300-second training budget.
- The current `MultiStepLR` milestones are 32,000 and 48,000 steps. Thus the baseline trains at LR 0.1 for about 93% of the realized run, reaches only the first decay, and never reaches the second. It has very little low-LR refinement.
- The baseline uses only random crop/flip and hard-label cross-entropy. It has no mixed-sample regularization or weight averaging.

The likely limiter is therefore not raw throughput or capacity alone, but poor use of the short horizon: limited early regularization followed by an LR schedule whose intended late phases do not fit inside the measured run.

## Mechanism

### 1. Early probabilistic CutMix

For each training batch while `progress < 0.70`, apply CutMix with probability `0.5`. Draw `lambda ~ Beta(1, 1)`, which is simply a uniform draw and needs no extra dependency. Randomly permute the batch, replace one rectangular region with pixels from the paired examples, recompute lambda from the clipped rectangle area, and use

```text
loss = lambda_adjusted * CE(logits, targets_a)
     + (1 - lambda_adjusted) * CE(logits, targets_b)
```

Otherwise use ordinary cross-entropy. This retains clean batches throughout the early phase and uses only clean batches in the final phase. It approximates a clean-plus-mixed objective without the second forward pass required by RegMixup.

CutMix is preferred over pure Mixup here because CIFAR-10 objects are spatially localized and CutMix preserves local pixel statistics. The implementation adds only a batch permutation, one rectangular copy, and a second scalar CE reduction over the same logits.

### 2. Time-normalized cosine learning rate

Remove `MultiStepLR`. Before every optimization step, compute

```python
progress = min(total_training_time / TIME_BUDGET_S, 1.0)
lr = MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))
```

with `LR = 0.1` and `MIN_LR = 0.001`, and assign this LR to the optimizer parameter groups. Do not add warmup: SGD at batch size 128 is already stable, and warmup would spend scarce early steps below the baseline LR.

Approximate LR values are 0.0855 at 25% of the budget, 0.0505 at 50%, 0.0214 when CutMix turns off at 70%, 0.0034 at 90%, and 0.001 at the end. Normalizing by measured training time rather than step count makes the intended schedule survive the small throughput cost of CutMix and EMA.

### 3. Sparse EMA for evaluation

Create a detached copy of the initialized model. At the first step after `progress >= 0.10`, copy the online parameters into the EMA model, then every 10 optimizer steps update parameters with

```python
ema_param.lerp_(online_param, 1.0 - EMA_DECAY)
```

using `EMA_DECAY = 0.99`. Updating every 10 steps gives an effective smoothing window of roughly 1,000 online steps while avoiding meaningful per-step overhead. Immediately before each evaluation, copy BatchNorm buffers (running mean, running variance, and batch counter) from the online model into the EMA model; do not average integer counters or stale early BatchNorm statistics.

Evaluate the online model before EMA activation and the EMA model thereafter. There must still be exactly one call to `evaluator.evaluate` per epoch, and `best_test_acc` must be computed from whichever single model was evaluated that epoch. Do not evaluate both models in the same epoch.

## Exact `train.py` Changes

1. Import `copy` and `math`; add constants `MIN_LR = 1e-3`, `CUTMIX_PROB = 0.5`, `CUTMIX_END = 0.70`, `EMA_START = 0.10`, `EMA_DECAY = 0.99`, and `EMA_EVERY = 10`.
2. Add a dependency-free CutMix helper operating on already-GPU-resident tensors. Use `torch.randperm(batch_size, device=inputs.device)`, a uniform `torch.rand((), device=inputs.device)` lambda, uniformly sampled rectangle center coordinates, clipped integer bounds, and area-corrected lambda. Avoid one-hot targets.
3. After constructing `model`, create `ema_model = copy.deepcopy(model)`, set all EMA parameters to `requires_grad_(False)`, and keep it in eval mode.
4. Delete the `MultiStepLR` creation and `scheduler.step()`. Set the cosine LR from `total_training_time / TIME_BUDGET_S` immediately before the forward pass.
5. Apply CutMix only under the time/probability gate, then compute either the two-term CutMix CE or the unchanged hard-label CE.
6. After `optimizer.step()`, activate/reset EMA once at 10% progress and update it every 10 steps thereafter under `torch.no_grad()`.
7. At epoch end, copy online buffers into EMA and make exactly one evaluator call. Leave all final-summary fields and formatting unchanged.
8. Keep `BATCH_SIZE=128`, `MOMENTUM=0.9`, `WEIGHT_DECAY=1e-4`, `MAX_STEPS`, the model, data loader, transforms, and seed unchanged. This isolates the proposed regularization/schedule package.

## Evidence

- [A Unified Analysis of Mixed Sample Data Augmentation](../papers/mixed-sample-analysis.md) analyzes CutMix and Mixup as pixel-level loss and first-layer regularizers and supports mixed-sample training as a low-arithmetic-cost intervention. Its analysis also supports treating Mixup and CutMix as distinct choices rather than stacking them blindly.
- [Using Mixup as a Regularizer Can Surprisingly Improve Accuracy & OOD Robustness](../papers/regmixup.md) reports that retaining ordinary clean-example supervision alongside mixed examples can outperform replacing it completely. A 50% CutMix gate provides this clean/mixed balance without RegMixup's extra forward pass, which is important under a fixed wall-clock budget.
- [Time Matters in Regularizing Deep Networks](../papers/time-matters-regularization.md) finds that augmentation and Mixup matter most in an early critical period and that removing regularization late can preserve or improve generalization. This directly motivates disabling CutMix at 70% rather than applying it through the final low-LR refinement phase.
- EMA is the least directly supported component in the experiment-local paper packet, so it is intentionally conservative and sparse. Its role is checkpoint smoothing, not stronger data regularization; it should be ablated first if the full package regresses.

## Expected Impact

The experiment should exceed the required `91.61%` threshold. A reasonable single-run expectation is `91.9-92.4% best_test_acc`, or approximately `+0.4` to `+0.9` percentage points over BASE. Most of the expected gain should come from the better-matched LR trajectory and early CutMix; EMA is expected to contribute a smaller improvement or stabilize which epoch realizes the best score.

The compute overhead should be small enough to retain close to the baseline's 34,435 steps. Accept up to about a 3% step-count loss if accuracy improves, but a materially larger loss indicates an inefficient implementation. EMA adds one extra model copy, negligible relative to the 98 GB H20 memory capacity.

## Risks and Mitigations

- **Underfitting from excessive regularization:** CutMix plus cosine decay may lower training fit too early. Mitigation: probability 0.5, clean batches interleaved throughout, and a fully clean final 30%; no label smoothing and no increased weight decay.
- **Cosine decays too aggressively:** The baseline holds LR 0.1 almost to the end, whereas cosine reaches roughly 0.05 halfway through. If accuracy stalls early, the next experiment should delay cosine rather than compensate by adding more methods.
- **EMA/BatchNorm mismatch:** Averaged weights with incompatible running statistics can reduce accuracy. Copy online BatchNorm buffers immediately before evaluation; never average `num_batches_tracked`.
- **EMA lags useful late changes:** A decay of 0.99 per 10 steps is intentionally much less sticky than 0.999 per step. Resetting EMA at 10% avoids averaging random initialization into the final model.
- **CutMix implementation error:** The loss lambda must be recalculated from the clipped rectangle area, and paired targets must use the exact permutation used for pixels.
- **Throughput regression:** Avoid a second model forward, one-hot label allocation, CPU image manipulation, or EMA updates every step.

## Follow-Up Ablations

The first run should test the full coherent package. If more experiments are available, use this order to identify the source of the result:

1. Time-normalized cosine only. This establishes whether repairing the mismatched baseline schedule is sufficient.
2. Cosine plus scheduled CutMix, no EMA. Compare with the full run to isolate checkpoint averaging.
3. Cosine plus EMA, no CutMix. This tests whether CutMix is too strong for the short ResNet-20 horizon.
4. If CutMix helps but the full package underfits, reduce `CUTMIX_END` from 0.70 to 0.55 before changing its strength or adding label smoothing.
5. Consider label smoothing (`0.05`) only as a replacement for CutMix, not an addition, if CutMix consistently hurts throughput or accuracy.

SWA is not proposed: the horizon contains little time after convergence, and correct SWA use would require choosing a late collection window plus BatchNorm-statistics handling. EMA begins providing usable averaged weights much earlier and is simpler under the one-validation-per-epoch rule.

## Verification

1. Confirm physical GPU 0 and expose only it:

   ```bash
   nvidia-smi -i 0 --query-gpu=index,name,memory.total --format=csv,noheader
   CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1
   ```

2. Confirm the complete summary and key metrics:

   ```bash
   grep "^best_test_acc:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:" run.log
   tail -n 12 run.log
   ```

3. Pass criteria: clean completion within the outer 10-minute limit, `training_seconds` approximately 300 seconds, exactly one validation per completed epoch, `best_test_acc >= 91.61%`, and the full final summary present.
4. Sanity-check that step count remains near 34,435, peak VRAM is plausible for two small model copies, and the log's LR reaches the intended low-LR phase.
5. Remove `run.log` after recording the experiment result. Only `train.py` may be changed for execution; this proposal is planning material, not part of the experimental code diff.

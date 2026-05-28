# Report EXP-020: CosineAnnealingLR with Correct T_max
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-020.md
- **Plan**: plans/plan-020.md
- **Log**: logs/exp-log-020.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, %, higher is better). Baseline: 95.91% (EXP-019, commit 355391a). Target: best_test_acc > 96.01% (baseline + 0.1pp).

## Idea & Hypothesis

Replace the wall-clock-fractional MultiStepLR schedule (drops at 50%/75% of 300s training budget, multipliers 0.1/0.01, final LR ~0.002) with a step-level cosine annealing schedule that decays LR from 0.2 to ~0 over ~100 estimated epochs, retaining 5-epoch linear warmup. This was the strongest candidate because EXP-000's CosineAnnealingLR failure was conclusively attributed to a T_max=200 mismatch (not cosine decay itself), and high-accuracy CIFAR-10 recipes (airbench96 at 96%) use smooth LR decay to zero. Hypothesis: smooth cosine decay to a lower final LR (~0 vs 0.002) would produce tighter convergence in the final training phase and avoid the AMP instability observed at the intermediate LR=0.02 regime from abrupt 10x drops.

## Approach

Removed the old wall-clock-fractional schedule infrastructure: `_lr_progress` cell, `_epoch_count` cell, `_wall_clock_fractional_step_decay` function, their training-loop updates, and the 14-line stale comment block. Replaced with a single `LambdaLR` using a `_warmup_cosine` closure that linearly ramps from 1/warmup_steps to 1.0 during the first 5 epochs (step-granularity), then cosine decays from 1.0 to 0.0 over the remaining steps. Total steps estimated as `ESTIMATED_EPOCHS=100 * steps_per_epoch`. The `import math` was placed inside `main()` to keep the diff minimal. No other code or hyperparameters changed.

## Execution

Single run, local execution on H20 GPU. Training completed 99 epochs in 300.0s — within 1 epoch of the ESTIMATED_EPOCHS=100 estimate. No errors, no retries, no adjustments needed. TTA evaluation added ~108s overhead (total 408.2s).

## Results

- **Primary metric**: 96.46% (baseline: 95.91%, delta: +0.55pp, +0.57%)
- **Observations**: Best accuracy achieved at the final epoch (epoch 99), confirming that cosine decay to near-zero LR produced tight convergence at training end. No AMP instability observed — smooth LR transitions eliminated the oscillation seen with step decay at the intermediate LR=0.02 regime. Throughput identical to baseline (~194 steps/epoch, 864.6 MB peak VRAM). The +0.55pp gain exceeded the hypothesis range of +0.1-0.3pp.
- **Analysis**: The hypothesis was confirmed and the effect was larger than expected. Two mechanisms likely contributed: (1) the lower final LR (~0 vs 0.002) allowed tighter convergence in the polish phase, and (2) the smooth cosine transition through intermediate LR regimes avoided the transient instability from abrupt 10x drops. The best=final pattern strongly supports mechanism (1) — the model was still improving at epoch 99. This suggests the old schedule's final LR of 0.002 was still too high for optimal convergence.
- **Key Learning**: Cosine decay to ~0 LR adds +0.55pp over step decay to 0.002 — the final LR floor matters more than the schedule shape in the middle.

## Verification

- **Conditions**: All 3 passed
  1. best_test_acc > 96.01%: PASS (96.46%)
  2. Full 10-field summary block: PASS (10/10 fields)
  3. Eval count ≤ num_epochs: PASS (99 ≤ 99)
- **Review Notes**: Results confirmed trustworthy. The 96.46% value is plausible given the mechanism (lower final LR → tighter convergence) and consistent with the best=final epoch pattern. No anomalies in throughput, VRAM, or epoch count.
- **Verdict**: improvement
- **Verdict Basis**: All 3 verification conditions passed, primary metric 96.46% exceeds baseline 95.91% by +0.55pp (well above the 0.1pp threshold).

## Unexplored Avenues

- **Eta_min > 0**: The current schedule decays to exactly 0. A small eta_min (e.g., 1e-5) might prevent the optimizer from completely stalling on the last few steps, though the best=final pattern suggests this isn't a problem yet.
- **Longer warmup**: 5 epochs is ~5% of training. A 10-epoch warmup might improve stability in the early phase, though no instability was observed.
- **OneCycleLR**: A triangular schedule (warmup to peak, then cosine decay) with a higher peak LR might accelerate early convergence, but evidence from brainstorm review was mixed for CIFAR-10.

## Next Steps

1. **Deeper architecture (NUM_BLOCKS=4, ResNet-26)** (medium confidence): With the LR schedule now optimized, the next axis is model capacity via depth. Adds ~1.4M params, costs ~15 epochs but gains more nonlinear capacity per resolution. The cosine schedule adapts naturally to fewer epochs.
2. **Cutout replacing RandomErasing** (low-medium confidence): Substitute the occlusion augmentation — airbench96's 12px Cutout may be more effective than RandomErasing for CIFAR-10's spatial scale.
3. **AdamW optimizer** (low confidence): SGD+momentum has been the optimizer throughout. AdamW with cosine schedule is the modern default and may find a better optimum, though the evidence for CIFAR-10 specifically is weaker.

## Exit Action Results

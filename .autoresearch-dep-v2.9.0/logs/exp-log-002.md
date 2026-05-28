# EXP-002: TrivialAugmentWide + RandomErasing on Width-2x Baseline

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-002.md
- **Plan**: plans/plan-002.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-002
- **Commit**: 651d57c
- **PR**: (failed — token permissions; user can create manually from autoresearch/exp-002 → main)
- **Outcome**: completed

## Implementation Notes

### Summary

Two lines added to the `train_tf` transforms.Compose list in train.py: `transforms.TrivialAugmentWide()` inserted between `RandomHorizontalFlip()` and `ToTensor()` (PIL-level operation), and `transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))` appended after `Normalize()` (tensor-level operation). No other changes — architecture, schedule, optimizer, and all other settings remain at EXP-001 values. Ruff check and format-check both pass.

### Surprises & Discoveries

None — the two-line change applied cleanly with no edge cases.

### Decisions

No deviations from the plan. The change is exactly as specified.

## Experimental Adjustments

(none)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local run, background Bash)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27T15:24:00Z
- **Ended**: 2026-05-27T15:30:00Z

Description:
- Running the width-2x ResNet-20 with the augmented training pipeline (TrivialAugmentWide + RandomErasing) under `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. Expected: best_test_acc in the 92.8-93.5% range, ~60-65 epochs in 300s due to augmentation overhead. The wall-clock-fractional schedule fires drops at 50%/75% of budget regardless of epoch count.

Observations:
- Params 1,073,962 (identical to EXP-001 — architecture unchanged)
- Step time ~11ms (augmentation overhead negligible on H20)
- 68 epochs completed (vs EXP-001's 69 — 1 fewer, minimal augmentation overhead)
- Early accuracy ~3pp below EXP-001 at same epochs (augmentation makes training harder as expected)
- First LR drop at ~epoch 34: +3.8pp jump (86.45→90.24%), then rapid convergence to 92.40% by epoch 49
- Second LR drop at ~epoch 52: pushed from 92.40% to 92.92% peak — more gain from polish than EXP-001's +0.02pp
- Final best 92.92% at epoch 65, final epoch 92.88%

Key Metrics:
- best_test_acc: 92.92% (source: run.log summary block)
- final_test_acc: 92.88% (source: run.log summary block)
- final_test_loss: 0.2163 (source: run.log summary block)
- training_seconds: 300.0 (source: run.log summary block)
- total_seconds: 354.4 (source: run.log summary block)
- startup_seconds: 1.1 (source: run.log summary block)
- peak_vram_mb: 598.7 (source: run.log summary block)
- num_epochs: 68 (source: run.log summary block)
- num_steps: 26420 (source: run.log summary block)
- num_params: 1,073,962 (source: run.log summary block)

## Verification Results

### Conditions Checked

- **Condition 1**: PASS — best_test_acc 92.92% > 92.39% threshold (baseline 92.29% + 0.1pp)
- **Condition 2**: PASS — summary block complete: `---` separator present, all 10 metric lines present
- **Condition 3**: PASS — eval_count (68) == num_epochs (68), validation runs exactly once per epoch

### Informational Metrics

- training_seconds: 300.0
- total_seconds: 354.4
- peak_vram_mb: 598.7
- num_epochs: 68
- num_steps: 26420
- num_params: 1,073,962

## Errors & Dead Ends

## Human Notes

> (autopilot — no user interaction)

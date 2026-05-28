# EXP-021: Pre-activation ResNet Blocks (BN-ReLU-Conv)

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-021.md
- **Plan**: plans/plan-021.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-021
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Converted the ResNet-20 model from post-activation (Conv-BN-ReLU) to pre-activation (BN-ReLU-Conv) ordering per He et al. 2016. Three changes to `train.py`: (1) BasicBlock.forward rewritten to `bn1(x) → relu → conv1 → bn2 → relu → conv2` with shortcut taken from raw `x` before BN/ReLU — no final relu after addition, preserving clean identity mapping. bn1 was already `BatchNorm2d(in_channels)` from a prior edit so no __init__ change needed. (2) Removed `self.bn1 = nn.BatchNorm2d(16 * WIDTH_MULT)` from ResNet.__init__ since the first block's bn1 handles normalization of stem output. Added `self.bn_final = nn.BatchNorm2d(64 * WIDTH_MULT)` after layer3 definition. (3) ResNet.forward stem simplified to raw `self.conv1(x)` (no BN/ReLU), ending changed to `F.relu(self.bn_final(out)) → pool → fc`.

### Surprises & Discoveries

The existing BasicBlock already had `self.bn1 = nn.BatchNorm2d(in_channels)` — identical to what pre-activation requires. The __init__ needed no change; only the forward pass ordering changed. This made the diff smaller than expected.

### Decisions

None — implementation followed the plan exactly.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/.autoresearch/logs/exp-021-run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27
- **Ended**: 2026-05-27

Description:
- Running the full training script with pre-activation ResNet blocks replacing post-activation blocks. Training config is identical to EXP-020 baseline (WIDTH_MULT=4, batch 256, AMP, label_smoothing=0.2, TrivialAugmentWide+RandomErasing, TTA evaluation, cosine warmup+decay LR schedule). The only change is the block ordering: BN-ReLU-Conv (pre-activation) instead of Conv-BN-ReLU (post-activation). Expecting ~99 epochs in 300s with identical throughput. Target: best_test_acc > 96.56%.

Observations:
- Training completed 93 epochs in 300.0s (vs 99 epochs in EXP-020 baseline) — a ~6% throughput regression
- 18,083 total steps vs 19,198 in EXP-020, avg ~16.6ms/step vs ~15.6ms/step baseline
- Throughput regression likely due to reduced kernel fusion opportunities with BN→ReLU→Conv ordering compared to Conv→BN→ReLU (cuDNN fuses Conv+BN+ReLU but not BN+ReLU+Conv)
- Training loss converged normally, no instability or divergence observed
- Best accuracy 96.23% achieved during training, a -0.23pp regression from the 96.46% baseline
- Fewer epochs (93 vs 99) means the model had less training, compounding any architectural effect

Key Metrics:
- best_test_acc: 96.23%
- final_test_acc: 96.23%
- final_test_loss: 0.2291
- training_seconds: 300.0
- total_seconds: 408.3
- startup_seconds: 1.2
- peak_vram_mb: 1449.2
- num_epochs: 93
- num_steps: 18083
- num_params: 4,286,026

## Verification Results

### Conditions Checked

**Condition 1: best_test_acc > 96.56%** — **FAIL**
- Command: `grep '^best_test_acc:' .autoresearch/logs/exp-021-run.log`
- Actual: best_test_acc = 96.23%
- 96.23 < 96.56 threshold (baseline 96.46 + 0.1pp)
- Source: `.autoresearch/logs/exp-021-run.log` final summary block

**Condition 2: Full 10-field summary block printed** — **PASS**
- Command: `grep -c -E '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|startup_seconds|peak_vram_mb|num_epochs|num_steps|num_params):' .autoresearch/logs/exp-021-run.log`
- Actual: 10
- Source: `.autoresearch/logs/exp-021-run.log` final summary block

**Condition 3: Eval count ≤ num_epochs** — **PASS**
- EVAL_COUNT = 93, NUM_EPOCHS = 93
- 93 ≤ 93
- Source: `.autoresearch/logs/exp-021-run.log`

**Overall: FAIL** — Condition 1 not met. Verdict: no-improvement.

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> 

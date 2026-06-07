# EXP-000: Modern Training Recipe (Cosine LR + CutOut + Label Smoothing)

## Execution

Overall Status & Info:
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-000.md
- **Plan**: plans/plan-000.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-000
- **Commit**: 288af5c
- **PR**: https://github.com/davdcharizard/autoresearch-cifar10/pull/1
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented all planned changes to train.py: (1) replaced MultiStepLR with SequentialLR chaining LinearLR warmup (5 epochs, start_factor=0.1) and CosineAnnealingLR (T_max=90), stepping per-epoch instead of per-step; (2) added CutOut augmentation class (16x16 patches) after Normalize in the training transform pipeline; (3) replaced F.cross_entropy with nn.CrossEntropyLoss(label_smoothing=0.1); (4) removed MAX_STEPS cap so TIME_BUDGET_S is the sole termination condition. Added numpy import for CutOut's random coordinate generation.

### Surprises & Discoveries
Initial T_max=200 was too large — with only ~90 epochs training, the cosine schedule barely decayed the LR by end of training (still ~0.058 vs baseline's 0.01 after step 32k). This caused a 2.55% accuracy regression to 89.26%.

### Decisions
Changed T_max from 200 to 90 after Run 1 failure. With the SequentialLR structure (5 warmup + 85 cosine), the cosine phase now spans the actual training duration, ensuring LR decays to near zero by the final epoch.

## Experimental Adjustments

- **T_max 200→90**: Run 1 with T_max=200 produced 89.26% (regression from 91.81% baseline). Root cause: LR stayed too high throughout training because cosine only completed 45% of its cycle. Fixed by matching T_max to actual epoch count. (ref: Run 1 — accuracy regression)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local run)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log (overwritten by Run 2)
- **WandB**: N/A
- **Status**: failed
- **Started**: 2026-05-28
- **Ended**: 2026-05-28

Description:
- Running the modified ResNet-20 training with cosine LR schedule (warmup 5 epochs + cosine decay T_max=200), CutOut augmentation (16px patches), and label smoothing (0.1).

Observations:
- best_test_acc=89.26%, a 2.55% regression from baseline. LR at epoch 90 was still ~0.058 due to T_max=200 causing the cosine to only complete ~45% of its cycle. The model never entered the low-LR fine-tuning phase that the baseline achieves.

Key Metrics:
- best_test_acc: 89.26% (REGRESSION from 91.81% baseline)

### Run 2

Metadata:
- **Job ID**: N/A (local run)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-28
- **Ended**: 2026-05-28

Description:
- Retry with T_max=90 to match actual training duration. All other changes unchanged (CutOut 16px, label smoothing 0.1, warmup 5 epochs).

Observations:
- best_test_acc=92.10%, an improvement of 0.29% over the 91.81% baseline. Training completed 89 epochs in 300s. LR properly decayed to near zero by final epoch.

Key Metrics:
- best_test_acc: 92.10% @ epoch 89 (source: run.log)
- final_test_acc: 91.91% @ epoch 89 (source: run.log)
- final_test_loss: 0.2971 (source: run.log)
- peak_vram_mb: 330.1 (source: run.log)
- num_params: 269,722 (source: run.log)

## Verification Results

### Conditions Checked

1. **Run completion**: Exit code 0, `best_test_acc` line present in run.log — **PASS**
2. **Time budget**: training_seconds=300.0 <= 300 — **PASS**
3. **Accuracy improvement**: best_test_acc=92.10% >= 91.91% (baseline 91.81% + 0.1% threshold) — **PASS**
4. **Eval frequency**: 89 eval lines = 89 epochs (num_epochs=89) — **PASS**

### Informational Metrics

- final_test_acc: 91.91%
- final_test_loss: 0.2971
- training_seconds: 300.0
- total_seconds: 375.0
- startup_seconds: 1.1
- peak_vram_mb: 330.1
- num_epochs: 89
- num_steps: 34458
- num_params: 269,722

## Errors & Dead Ends

### 2026-05-28 — T_max=200 too large, LR didn't decay enough
- Error: best_test_acc regressed to 89.26% (from 91.81% baseline)
- Root cause: CosineAnnealingLR T_max=200 was far larger than the actual ~90 training epochs, so LR remained too high throughout training (~0.058 at epoch 90 vs baseline's 0.01 at same point)
- Source: Run 1 log
- Do NOT retry: avoid T_max significantly larger than expected epoch count; always match T_max to actual training duration

## Human Notes

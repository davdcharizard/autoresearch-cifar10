# Report EXP-019: Channels_last + T_max=49 + extended TTA (spatial shifts)
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-019.md
- **Plan**: plans/plan-019.md
- **Log**: logs/exp-log-019.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, higher is better) within 300s single-GPU training budget. Current baseline: 96.39% (EXP-016).

## Idea & Hypothesis

Combine channels_last (NHWC) memory format for training speedup with original T_max=49 and extended 6-view TTA (original + hflip + ±1px spatial shifts). Hypothesis: extra epochs at near-zero LR would refine the model, and 6-view TTA would reduce prediction variance, together pushing accuracy to ~96.5-96.7%.

## Approach

Three changes to train.py: (1) channels_last on model (before EMA deepcopy) and training inputs, (2) 6-view TTA in eval forward with reflect-padded ±1px spatial shifts, (3) COSINE_T_MAX kept at 49 (unchanged from baseline).

## Execution

Single run, completed normally. 64 epochs in 300s (~18% speedup from channels_last, more than EXP-018's 9%).

## Results

- **Primary metric**: 96.28% (baseline: 96.39%, delta: -0.11%, -0.11%)
- **Observations**: The critical discovery: **CosineAnnealingLR is periodic** — after T_max=49 cosine steps, the LR does NOT stay at minimum. It restarts and increases back toward the peak LR. With 64 total epochs (59 cosine steps, 10 past T_max), the LR rose from 0 back to ~0.01 by epoch 64. This caused model degradation: best=96.28% at some middle epoch vs final=95.38% (0.9% gap). The prior assumption that "extra epochs at near-zero LR provide free refinement" was incorrect — extra epochs with rising LR actively hurt.
- **Analysis**: The experiment confounded three variables (channels_last, LR restart, extended TTA). The LR restart is the dominant failure mode — 0.9% best/final gap proves the model peaked mid-training then degraded. The 6-view TTA cannot be evaluated independently because the model state at peak epoch may differ from the baseline's peak state. The key protocol finding: any experiment using channels_last MUST account for CosineAnnealingLR's periodic behavior when more epochs exceed T_max.
- **Key Learning**: CosineAnnealingLR restarts after T_max (periodic); channels_last's extra epochs cause LR to rise, degrading the model. Must clamp LR at minimum or match T_max to actual epochs.

## Verification

- **Conditions**: best_test_acc >= 96.49% FAILED (actual: 96.28%)
- **Review Notes**: Results trustworthy — degradation explained by LR restart mechanism
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric 96.28% below baseline 96.39% + 0.1% threshold

## Unexplored Avenues

- **Extended TTA (spatial shifts) WITHOUT channels_last** — isolate TTA effect on the exact baseline model. No LR restart confound. This is the cleanest next experiment.
- **Channels_last with T_max matching actual cosine epochs + eta_min clamp** — use CosineAnnealingLR with T_max=59 (matching actual cosine epochs under channels_last speedup) and verify LR doesn't restart. But EXP-018 showed T_max=55 hurts (slower decay), so the tension between optimal decay rate and avoiding restart may be irreconcilable.
- **Replace CosineAnnealingLR with CosineAnnealingWarmRestarts or manual schedule** — use a schedule that explicitly clamps at minimum after T_max steps. Could enable channels_last speedup without restart degradation.

## Next Steps

1. **Extended TTA only (no channels_last)** (high confidence) — test 6-view TTA on the exact baseline. Zero training changes, no LR restart risk. Cleanly isolates whether spatial-shift TTA adds value beyond hflip.
2. **Channels_last + LR clamp** (medium confidence) — use channels_last but add `scheduler.step(); lr = max(optimizer.param_groups[0]['lr'], 0)` or set eta_min to prevent restart. Requires careful implementation.
3. **Gradient clipping** (low confidence) — try gradient clipping to stabilize late-stage training. Orthogonal to schedule issues.

## Exit Action Results

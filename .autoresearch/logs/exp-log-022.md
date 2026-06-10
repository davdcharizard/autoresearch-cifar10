# EXP-022: WRN-style dropout in the residual blocks (p=0.1)

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-022.md
- **Plan**: plans/plan-022.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-022
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (run clean; verification Cond 1 failed → no-improvement verdict, rendered in analyze)

## Implementation Notes

### Summary
Three changes to the clean EXP-012 baseline `train.py` (Milestone 1): (1) added `DROPOUT_P = 0.1` constant next to the other hyperparameters; (2) added `self.dropout = nn.Dropout(p=DROPOUT_P)` to `BasicBlock.__init__`; (3) inserted `out = self.dropout(out)` in `BasicBlock.forward` between the first conv+ReLU and the second conv — the exact WRN placement (Zagoruyko & Komodakis 2016). Everything else identical to EXP-012 (k=4 4.3M params, PEAK_LR 0.2 cosine-to-0, batch 128, Nesterov, WD 1e-4, LS 0.1, TrivialAugment + Cutout(16), torch.compile reduce-overhead, bf16, channels_last, seed 42). `uv run ruff check` passed, AST parses, `git diff --name-only` = train.py only.

### Surprises & Discoveries
None. `nn.Dropout` is core torch (no new dep). Eval correctness is automatic: the frozen `Eval.evaluate()` calls `model.eval()` so dropout becomes identity at test time, and the training loop calls `model.train()` each epoch — no manual toggling and no eval contamination.

### Decisions
Chose p=0.1 (not the WRN paper's 0.3) as a budget-appropriate first probe: the recipe is already heavily regularized (TA+Cutout+LS+WD) and the budget is short (~92 ep vs WRN's 200), so a strong dropout risks the under-fitting that sank CutMix (EXP-018). Used plain elementwise `nn.Dropout` (matches the paper) rather than channel-wise `Dropout2d`.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID recorded at launch — background bash task)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08
- **Ended**: 2026-06-08

Description:
- Runs the EXP-012 recipe with WRN-style dropout (p=0.1) inserted between the two convs of every residual block, testing whether intermediate-feature regularization — a locus untouched by all prior experiments — reduces the residual generalization gap and lifts best_test_acc above the 96.32 bar. The model is generalization-bound at fixed k=4 capacity in 300s. Expected: a small top-1 gain (loss DOWN + acc UP = dropout helps) OR graceful no-improvement (loss UP = under-fit/over-regularized at the ~92-ep budget).

Observations:
- Clean run: params 4,299,866 (unchanged), clean compile (no graph break on nn.Dropout), no traceback, no NaN (source: run.log L2, summary block).
- best_test_acc 94.85% vs baseline 96.22 = **−1.37pp**, a LARGE regression. final_test_loss ROSE sharply 0.195→**0.2236**. This is clear UNDER-fitting / over-regularization — the predicted failure mode: even mild p=0.1 dropout stacked on the already-saturated TA+Cutout+LS+WD recipe prevents convergence at the budget (source: run.log eval lines + summary).
- num_epochs 84 (vs baseline 91): dropout's extra per-step RNG mask cost a few epochs, but the loss rise (0.224 ≫ 0.195) is far too large to be an epoch-count confound — it is genuine under-fit, not marginal under-training.

Key Metrics:
- best_test_acc: 94.85% (source: run.log `best_test_acc:` line)
- final_test_loss: 0.2236 (source: run.log) — vs baseline 0.195, ROSE sharply (under-fit signature)
- final_test_acc: 94.84%; num_epochs: 84; num_steps: 32681; total_seconds: 400.7; peak_vram_mb: 537.9; num_params: 4,299,866 (source: run.log summary)

## Verification Results

### Conditions Checked

- **Cond 1 — primary metric clears bar (best_test_acc ≥ 96.32)**: **FAILED**. best_test_acc = 94.85% < 96.32 (−1.37pp vs baseline 96.22). (source: run.log `best_test_acc: 94.85%`)
- **Cond 2 — clean completion within budget**: skipped — aborted after Cond 1 failed. (Would pass: total_seconds=400.7 < 600, Traceback count=0, metrics present.)
- **Cond 3 — no constraint violations**: skipped — aborted after Cond 1 failed. (Would pass: git diff = train.py only, num_params=4,299,866, eval-count=84 == num_epochs=84, nn.Dropout is core torch / no new deps, seed 42 intact.)

Verdict basis: first necessary condition failed → no-improvement; remaining conditions not evaluated.

### Informational Metrics

- Not collected (only when all conditions pass). For the record: peak_vram_mb=537.9, num_epochs=84 (vs baseline 91), num_steps=32681, final_test_loss=0.2236 (ROSE sharply vs baseline 0.195 → under-fit / over-regularization signature).

## Errors & Dead Ends

<!-- none -->

## Human Notes

> (none — autopilot)

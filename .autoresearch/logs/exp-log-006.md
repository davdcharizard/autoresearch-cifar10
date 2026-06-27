# EXP-006: torch.compile with pre-loop warmup

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-006.md
- **Plan**: plans/plan-006.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-006
- **Commit**: 1990397
- **PR**: N/A — no git remote configured (intentional per TASK.md)
- **Outcome**: completed

## Implementation Notes

### Summary
Three edits per plan: (1) `base_model = model` eager reference kept, then `model = torch.compile(model)` (default inductor mode); (2) warmup block after optimizer creation and before `t_start_training` — 3 forward+backward passes on a synthetic (512,3,32,32) channels_last batch under bf16 autocast with label-smoothed CE, grads zeroed afterwards, no optimizer.step(), so weights are untouched and the one-time compile cost lands in startup_seconds (excluded from the 300s training budget by construction); (3) eval call switched to `evaluator.evaluate(base_model, device)` so the frozen Eval path runs eager — no second compilation, eval timing byte-identical to baseline. Zero hyperparameter changes. Ruff clean; only train.py modified.

### Surprises & Discoveries
- None at implementation time.

### Decisions
- None beyond the plan (the eager-eval and warmup designs were already specified there).

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash task b572fc30j (local)
- **Log file(s)**: run.log (project root)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 05:10
- **Ended**: 2026-06-10 05:18

Description:
- Compiled training step on the 4x doubly-regularized baseline — pure systems experiment testing whether inductor fusion raises steps/s enough (~114 → ≥130 epochs hypothesized) to convert into accuracy under the augmented recipe (pass bar ≥ 96.33). Regardless of verdict, num_epochs is the img/s datapoint that decides whether the aligned-width (6x/8x) direction is viable. Expect startup_seconds 60–120 (compile), total ≤ ~580s.

Observations:
- Compile clean — no traceback; params 4,286,026 unchanged (source: run.log L2)
- Step timing: dt ~22ms, ~22.9k img/s vs 27ms/~19k eager → ~1.21x throughput, on track for ~138 epochs (source: run.log step lines, epoch ~10)
- Epoch-1 eval: test_acc 35.11% — healthy, close to EXP-004's eager 34.26% (source: run.log first `eval ep` line)

Key Metrics:
- best_test_acc: 96.71% | final_test_acc: 96.65% | final_test_loss: 0.1837 (source: run.log summary block)
- training_seconds: 300.0 | total_seconds: 491.1 | startup_seconds: 22.8 (source: run.log summary block)
- peak_vram_mb: 1639.5 | num_epochs: 139 | num_steps: 13,462 | num_params: 4,286,026 (source: run.log summary block)
- Throughput: 139 epochs vs 114 eager = 1.22x (dt 22ms vs 27ms; ~22.9k img/s); compile cost only 22.8s, fully in startup — budget integrity verified (training_seconds exactly 300.0)

## Verification Results

### Conditions Checked

1. **Run completes without crashing within the time budget (≤ 10 min total)** — PASS
   - `best_test_acc:` present; total_seconds = 491.1 ≤ 600; exit 0 (source: run.log summary; task b572fc30j)
2. **best_test_acc ≥ baseline + 0.1 pp (≥ 96.33)** — PASS
   - best_test_acc = 96.71% vs baseline 96.23% → +0.48 pp (source: run.log summary; exp-index.sh baseline)
3. **Validation executed at most once per epoch** — PASS
   - eval lines = 139 = num_epochs (source: run.log)

### Informational Metrics

- num_epochs: 139 (vs 114 eager — 1.22x throughput; THE datapoint that re-opens the aligned-width direction)
- startup_seconds: 22.8 (compile cost fully absorbed in startup; training_seconds exactly 300.0)
- peak_vram_mb: 1639.5 (vs 1620.7 eager — inductor workspace negligible)
- num_params: 4,286,026 (unchanged — pure systems experiment)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

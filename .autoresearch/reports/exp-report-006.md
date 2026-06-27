# Report EXP-006: torch.compile with pre-loop warmup
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-006.md
- **Plan**: plans/plan-006.md
- **Log**: logs/exp-log-006.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, %, higher is better) of train.py within the fixed 300s training budget (≤10 min total wall clock). Baseline at experiment start: **96.23%** @ 1174e0d (4x ResNet-20 + one-cycle + TA + RandomErasing). Question tested: does compiled execution of the training step buy enough throughput (epochs) to convert into accuracy — and is the compile/width path viable at all?

## Idea & Hypothesis

Chosen over batch-1024+LR-0.8 and Mixup/CutMix because throughput was the demonstrated gate (count-2 failed approach: 8x@40ep, 5x@52ep both starved) and compile is the only untried throughput lever that doesn't touch the frozen data pipeline. Two design subtleties were identified up front: (1) train.py bills per-step wall time against the 300s budget, so compile cost must be absorbed pre-loop via a warmup with no optimizer.step(); (2) passing the eager `base_model` reference to the frozen evaluator avoids a second compilation of the eval graph.

**Hypothesis**: ≥15% steps/s gain (114 → ≥130 epochs) at unchanged dynamics lifts best_test_acc from 96.23% to ≥96.33%; startup absorbs 60–120s of compile; total ≤ 580s.

## Approach

Three edits to train.py, zero hyperparameter changes: `base_model = model; model = torch.compile(model)` after model creation; a 3-iteration warmup block (synthetic (512,3,32,32) channels_last batch, bf16 autocast, label-smoothed CE, backward, grads zeroed, no step) before `t_start_training`; eval call switched to `evaluator.evaluate(base_model, device)`.

## Execution

Single run, no retries, no errors. Compile completed without graph breaks. Total 491.1s.

## Results

- **Primary metric**: best_test_acc = **96.71%** (baseline: 96.23, delta: **+0.48pp**, +0.50%)
- **Observations**:
  - Throughput: dt 27ms → 22ms (~22.9k img/s), num_epochs 114 → **139** (1.22x) — hypothesis premise met with margin.
  - startup_seconds = 22.8 — compile was far cheaper than the 60–120s estimate, and training_seconds is exactly 300.0: the budget-integrity design worked as intended and is verifiable from the log.
  - final_test_loss 0.1947 → 0.1837 and final (96.65) ≈ best (96.71): the longer schedule converged cleanly; no instability anywhere in the run.
  - The +0.48pp gain EXCEEDED both augmentation steps' predictions for "more epochs convert weakly": at 114 epochs the augmented recipe was actually still epoch-limited — EXP-004's "final=best means converged" inference was too optimistic.
  - peak VRAM 1639.5MB (+19MB) — inductor workspace negligible.
- **Analysis**: Trajectory: 91.97 → 93.16 → 95.23 → 96.06 → 96.23 → **96.71**. The result revises the project's mental model in two ways. First, epochs were still a binding constraint at 4x under heavy augmentation — meaning the augmentation gains and capacity gains were both being throttled by throughput, and the diminishing-returns reading of EXP-004 partially reflected epoch starvation, not regularization exhaustion. Second, the width direction is genuinely re-opened: with 1.22x throughput, an aligned 6x net (96/192/384) would land at ~75–80 epochs — above the ~70-epoch starvation floor estimated from EXP-002/005 — making it the natural next probe. Mixup/CutMix also gets a small re-promotion (more epochs = more room for stronger regularization).
- **Key Learning**: The compile lever was pure profit (1.22x img/s for 22.8s of startup), and its success exposed that "converged" one-cycle runs at fixed time can still be epoch-starved — throughput gains convert into accuracy even when final=best.

## Verification

- **Conditions**: all passed
  1. Clean completion within budget: total_seconds 491.1 ≤ 600, exit 0 — PASS
  2. best_test_acc ≥ baseline + 0.1pp (≥ 96.33): 96.71 — PASS
  3. Eval at most once per epoch: 139 eval lines = 139 epochs — PASS
- **Review Notes**: results trustworthy — eval ran the unmodified eager path (base_model shares weights with the compiled wrapper) on the frozen prepare.py Eval; the gain mechanism (more epochs at identical hyperparameters) is fully visible in the log; budget integrity proven by training_seconds=300.0 + startup_seconds=22.8.
- **Verdict**: improvement
- **Verdict Basis**: all conditions passed + improvement well above the 0.1pp bar (+0.48pp)

## Unexplored Avenues

- **Aligned width 6x (96/192/384) on the compiled recipe**: ~75–80 epochs projected (1.22x throughput, aligned channels avoid the EXP-005 penalty) — now above the starvation floor; the highest-value untried capacity probe.
- **compile mode="max-autotune"**: more aggressive kernel search; could add another few % img/s for ~1–3 min extra startup (wall clock has ~109s headroom). Marginal but cheap to test alongside a width change? No — keep single-variable.
- **Mixup/CutMix at 139 epochs**: re-promoted from sub-noise to plausible (+0.1–0.3pp) since epochs were shown to still bind; lower priority than width.
- **Batch 1024 under compile**: fewer launches matter less post-fusion; still likely marginal.

## Next Steps

1. **WIDTH_MULT 4 → 6 (96/192/384, aligned) on the compiled recipe** (medium-high confidence: capacity gains were throughput-throttled, alignment rule respected, projected ~75–80 epochs above the floor; clean failure decisively closes width).
2. **Mixup or CutMix at the longer schedule** (medium confidence; after the width question).
3. **max-autotune compile mode** (low-medium; cheap incremental img/s if width needs more headroom).

## Exit Action Results

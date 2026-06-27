# EXP-009: Mixup (α=0.2) on the compiled 4x recipe

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-009.md
- **Plan**: plans/plan-009.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-009
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (verification condition 2 — metric below baseline + 0.1pp)

## Implementation Notes

### Summary
Per plan: added `MIXUP_ALPHA = 0.2` constant and a ~10-line mixup block in the training step. lam is Beta(0.2, 0.2)-sampled per step on CPU; `perm = torch.randperm(batch)` on GPU; inputs mixed via `torch.lerp(inputs[perm], inputs, lam)` with an explicit `.contiguous(channels_last)` to guard against index_select dropping the memory format; loss is the two-term lam-weighted label-smoothed CE. All mixing is outside the compiled module (input shape/dtype unchanged → no recompile; warmup block untouched); eval path untouched (eager `base_model` on clean test data). Architecture and every other hyperparameter byte-identical to baseline 1990397. Diff: +13/−1 lines, train.py only. Ruff clean; GPU 0 idle at launch.

### Surprises & Discoveries
- None at implementation time.

### Decisions
- Used `torch.lerp(b, a, lam)` (= lam*a + (1−lam)*b) rather than the arithmetic expression — single fused kernel, same semantics as the plan's formula.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash task b52gvbaa9 (local)
- **Log file(s)**: run.log (project root)
- **WandB**: N/A
- **Status**: completed (metric below bar)
- **Started**: 2026-06-10 06:12
- **Ended**: 2026-06-10 06:21

Description:
- Mixup α=0.2 added to the otherwise-frozen compiled 4x TA+RE recipe — first probe of the input-label-space regularization axis after capacity was closed bidirectionally (EXP-007/008). Expect dt ~22ms (mixing ~free), ~135–139 epochs, VRAM ~1.65GB, total ~490–500s. Pass bar ≥ 96.81; hypothesis ≥ 96.81 via reduced train/test gap. Health signatures: epoch-1 ≥ 15%, train-loss trace sits higher than prior runs (mixed targets — expected, not a bug); bug signature would be epoch-1 ≈ 10%.

Observations:
- Params 4,286,026 — unchanged as required (source: run.log L2)
- Epoch-1 eval: test_acc 33.87% vs EXP-006's 35.11% — healthy; mixup costs only ~1.2pp of early convergence, nowhere near the ≈10% bug signature (source: run.log first `eval ep` line)
- dt 23ms at steps 100–150 (~22.2k img/s) vs 22ms baseline — mixing is ~free as predicted, projects ~133–139 epochs (source: run.log step lines)
- Smoothed train loss 2.13 at step 150 sits above prior runs at similar progress — expected from mixed targets, per plan (source: run.log step lines)

Key Metrics:
- best_test_acc: 96.25% | final_test_acc: 96.15% | final_test_loss: 0.3042 (source: run.log summary block)
- training_seconds: 300.0 | total_seconds: 482.8 | startup_seconds: 11.1 (source: run.log summary block)
- peak_vram_mb: 1613.0 | num_epochs: 137 | num_steps: 13,224 | num_params: 4,286,026 (source: run.log summary block)
- Throughput confirmed free: 137 epochs vs 139 baseline (dt 23ms vs 22ms); startup only 11.1s (warm inductor cache)
- Trajectory: tracked ~3–5pp below EXP-006 mid-schedule (ep 40: 75.3 vs ~85; ep 120: 95.58), and the anneal did NOT close the gap — last 5 epochs are a noisy plateau 96.13–96.25 (converged, not starved). final_test_loss 0.3042 vs baseline 0.1837 (softer predictions under mixup) (source: run.log eval lines)

## Verification Results

### Conditions Checked

1. **Run completes without crashing within the time budget (≤ 10 min total)** — PASS
   - `best_test_acc:` present; total_seconds = 482.8 ≤ 600; exit 0 (source: run.log summary; task b52gvbaa9)
2. **best_test_acc ≥ baseline + 0.1 pp (≥ 96.81)** — FAIL
   - best_test_acc = 96.25% vs baseline 96.71% → −0.46 pp (source: run.log summary; exp-index.sh baseline)
3. **Validation executed at most once per epoch** — skipped — aborted after prior failure (informally: 137 eval lines = 137 epochs, compliant)

### Informational Metrics

- (not collected — necessary condition failed; values noted in Key Metrics for the record)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

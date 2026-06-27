# EXP-008: Wider-but-shallower — NUM_BLOCKS 2 + WIDTH_MULT 6 (ResNet-14)

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-008.md
- **Plan**: plans/plan-008.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-008
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (verification condition 2 — metric below baseline + 0.1pp)

## Implementation Notes

### Summary
Two constant changes per plan: `NUM_BLOCKS = 3` → `2` (ResNet-14 = 6*2+2) and `WIDTH_MULT = 4` → `6` (stage widths 96/192/384, all 32-aligned per the project-insights H20 alignment rule), both inline comments updated. The compiled doubly-regularized recipe (torch.compile + pre-loop warmup, TA, RandomErasing, time-keyed one-cycle, batch 512, selective WD, label smoothing) stays byte-identical to baseline 1990397. The `_make_layer` path supports any block count — no structural edits needed. This is a justified retry of the High count-3 failed approach (capacity/epoch starvation): depth FLOPs pay for the width (net ~1.46x FLOPs for 1.43x params), the ~80–85 epoch projection uses EXP-007's *measured compiled* scaling, and a binding dt gate (kill if dt ≥ 47ms at step ~100) caps the residual risk — see plan-008 § Failed-Approach Retry Justification. Ruff clean; only train.py modified; GPU 0 idle at launch (0 MiB used).

### Surprises & Discoveries
- None at implementation time. (Note: git checkout to the new branch invalidated a stale in-context copy of train.py from EXP-007 — re-read before editing; no impact.)

### Decisions
- None beyond the plan.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash task bglu7etqy (local)
- **Log file(s)**: run.log (project root)
- **WandB**: N/A
- **Status**: completed (metric below bar)
- **Started**: 2026-06-10 05:55
- **Ended**: 2026-06-10 06:02

Description:
- ResNet-14 at 6x width (96/192/384, ~6.15M params) on the compiled TA+RE recipe — the depth-for-width capacity probe after uniform widening was closed (count-3 High). Expect dt ~36–40ms → ~80–85 epochs, VRAM ~2GB, total ~450–470s. Pass bar ≥ 96.81; hypothesis ≥ 96.85. Binding early gate: dt ≥ 47ms sustained at step ~100 → kill (foregone starvation). Failure signature that closes capacity entirely: depressed accuracy with final≈best at ~80 epochs.

Observations:
- Params 6,149,866 — matches ~6.15M prediction; "ResNet-14 (6x wide)" header correct (source: run.log L2)
- Epoch-1 eval: test_acc 39.58% — healthy, the strongest epoch-1 of any experiment (37.87% at EXP-007, 35.11% at EXP-006); shallow-wide optimizes fast early (source: run.log first `eval ep` line)
- **dt gate PASSED**: dt 38ms steady at steps 100–150 (~13.5k img/s), inside the predicted 36–40ms window and below the 47ms kill bar; projects 139 x 22/38 ≈ 80 epochs — above the ~70 starvation floor as planned (source: run.log step lines)

Key Metrics:
- best_test_acc: 95.76% | final_test_acc: 95.73% | final_test_loss: 0.2034 (source: run.log summary block)
- training_seconds: 300.0 | total_seconds: 411.1 | startup_seconds: 19.0 (source: run.log summary block)
- peak_vram_mb: 1709.7 | num_epochs: 83 | num_steps: 7,980 | num_params: 6,149,866 (source: run.log summary block)
- Throughput: dt 38ms steady (~13.5k img/s) — the measured-regime projection was exact (predicted 36–40ms, 80–85 epochs; got 83)
- **PLATEAU signature, not starvation**: epochs 79–83 all flat at 95.64–95.76 (best 95.76 at ep 80, final 95.73) — the model CONVERGED to a lower ceiling. This is the opposite of the EXP-002/005/007 climbing-at-cutoff signature: epochs were sufficient; the architecture itself is worse (source: run.log last 5 eval lines)

## Verification Results

### Conditions Checked

1. **Run completes without crashing within the time budget (≤ 10 min total)** — PASS
   - `best_test_acc:` present; total_seconds = 411.1 ≤ 600; exit 0 (source: run.log summary; task bglu7etqy)
2. **best_test_acc ≥ baseline + 0.1 pp (≥ 96.81)** — FAIL
   - best_test_acc = 95.76% vs baseline 96.71% → −0.95 pp (source: run.log summary; exp-index.sh baseline)
3. **Validation executed at most once per epoch** — skipped — aborted after prior failure (informally: 83 eval lines = 83 epochs, compliant)

### Informational Metrics

- (not collected — necessary condition failed; values noted in Key Metrics for the record)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

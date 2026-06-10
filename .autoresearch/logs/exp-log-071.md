# EXP-071: BatchNorm eps 1e-5 → 1e-3

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-071.md
- **Plan**: plans/plan-071.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-071
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Applied the plan's Milestone 1 verbatim: added `BN_EPS = 1e-3` as a named constant after L28 (`CUTOUT_SIZE`), then passed `eps=BN_EPS` to all four `nn.BatchNorm2d(...)` construction sites — BasicBlock `bn1`/`bn2`, the downsample-shortcut BN inside the `nn.Sequential`, and the ResNet stem `bn1`. Everything else byte-identical to EXP-054 (AugMix-p0.5 + GPU Cutout16, cosine peak0.2/warmup0.05, Nesterov m0.9, WD1e-4, LS0.1, batch128, seed42, compile reduce-overhead). Smoke test passed: AST OK; `ResNet(3,10,width_mult=4)` forward `(2,3,32,32)→(2,10)`; num_params UNCHANGED at 4,299,866 (eps adds no parameters); all 22 BatchNorm2d modules report `.eps == 1e-3`; `git diff --name-only` == `train.py` only.

### Surprises & Discoveries
- The net has 22 BatchNorm2d modules (3 blocks/stage × 2 BN per block × 3 stages = 18, + 1 stem + 3 downsample-shortcut BNs in the stride-2/channel-change blocks = 22). The four *construction sites* in source cover all 22 instantiated modules because `_make_layer` builds BasicBlocks in a loop — so editing the 4 sites correctly propagated eps to every BN, as the smoke assertion confirmed.

### Decisions
- No deviations from the plan. The named-constant approach (vs. inlining `eps=1e-3` four times) keeps the probe a single auditable knob, matching the plan's rationale.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (background bash, PID recorded at launch)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-06-10
- **Ended**: 2026-06-10

Description:
- Running the EXP-071 BN-eps probe: `CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1` on idle GPU 1 (0% util, 0 MiB at launch; GPU 0 lightly used at 10%/1043 MiB). Tests whether raising the BatchNorm numerical floor 1e-5→1e-3 moves best_test_acc off the 96.45 baseline. Expected: a within-noise null (96.2–96.45), completing the BN-estimator axis map — eps is the last untested static BN knob and is optimization-/logit-scale-/dt-neutral, so a clean benign null is the overwhelmingly likely outcome.

Observations:
- Launch healthy; params printed 4,299,866 (source: run.log L2). 390 batches/epoch, time budget 300s.
- **Early gate (≤ep3) PASSED**: dt steady 8ms (occasional 9ms), img/s ~15,300 — no cudagraph break, no GPU contention. Eval climbing NORMALLY: ep1 test_acc 34.53%, ep2 45.88% — NOT stuck at random ~10% (the EXP-070 divergence signature is ABSENT). Train loss declining smoothly 2.78→1.64. LR warmed to peak 0.200 by ep2 as expected. Trajectory tracks EXP-054 — consistent with the predicted inert BN-eps null. (source: run.log eval ep1/ep2 lines, step 00050–01000)

Key Metrics:
- ep1 test_acc: 34.53% (source: run.log "eval ep 1")
- ep2 test_acc: 45.88% (source: run.log "eval ep 2")
- **best_test_acc: 95.92%** (−0.53pp vs baseline 96.45) (source: run.log "best_test_acc:")
- final_test_loss: 0.2050; training_seconds: 300.0; total_seconds: 584.4; num_epochs: 89; num_steps: 34343; num_params: 4,299,866; peak_vram_mb: 461.5 (source: run.log summary block)

## Verification Results

### Conditions Checked

- **Necessary condition 1 — `best_test_acc >= 96.55`**: best_test_acc = **95.92** < 96.55. **FAILED** (−0.63pp below bar, −0.53pp below baseline 96.45). Stop at first failed condition.
- **Necessary condition 2 — clean completion within budget**: NOT formally required after cond-1 failure, but recorded for completeness: training_seconds 300.0 ✓, total_seconds 584.4 < 600 ✓, num_params 4,299,866 UNCHANGED ✓, 0 nan/traceback/error ✓, 89 epochs.
- **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == train.py only ✓; no new deps; seed 42; evaluate() once/epoch; ran uncontended (dt steady 8ms).

**Verdict: no-improvement.** Clean valid run (Σdt=300.0 respected, wall 584.4 < 600, dt 8ms / no graph break, train.py only) that missed the bar. Results trustworthy — no NaN/error, metric parsed directly from summary. NOT invalid (no constraint breach, eps adds no params) and NOT crash (produced a real interpretable metric).

### Informational Metrics

- num_epochs 89 / num_steps 34343 (throughput unchanged vs EXP-054's ~91 ep — eps is compute-free; the 2-epoch difference is base-recipe wall variance).
- final_test_loss 0.2050 (vs EXP-054's 0.1968 — slightly higher, consistent with the mild eps under-utilization).
- peak_vram_mb 461.5.
- **Key observation**: the BN-eps probe was NOT the predicted exact null — it landed −0.53pp, squarely in the "every scalar/static-knob retune lands −0.2 to −0.6pp" band (EXP-067 insight). A 100× larger BN eps mildly dampens low-variance channels enough to cost ~0.5pp, rather than being inert. The early trajectory was healthy (no EXP-070-style divergence — eps is optimization-stable as predicted), so the loss is a uniform mild degradation across the run, not an early-epoch catastrophe.

## Errors & Dead Ends

<!-- none -->

## Human Notes

> (none — autopilot)

# EXP-022: Lookahead-Wrapped Momentum SGD

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-022
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Implemented the reviewed Lookahead point in tracked `train.py` only. The accepted width-2 model, SGD configuration, N1/M7 plus p=0.5 alpha-1 CutMix phase, weak hard-label tail, LR schedule, seed, worker lifecycle, timer, and evaluator remain unchanged. The candidate keeps one detached slow tensor per optimizer parameter and, after completed steps 5, 10, 15, etc., applies fused `torch._foreach_lerp_` followed by `torch._foreach_copy_` before the existing CUDA synchronization so all overhead is counted.

### Surprises & Discoveries

The first plan draft's conceptual per-parameter interpolation would have launched many tiny CUDA kernels. Adversarial plan review identified that this alone could exceed the 1% cost gate. Installed PyTorch 2.9.1 exposes both `_foreach_lerp_` and `_foreach_copy_`, allowing the exact registered recurrence to use two multi-tensor operations per synchronization.

### Decisions

- Retain inner SGD momentum buffers unchanged across parameter synchronization, matching the selected proposal; do not interpolate/reset optimizer state or BatchNorm running buffers.
- Append `lookahead_offset = step % 5` to existing evaluation lines for weight/BN-buffer phase attribution without moving or adding evaluations.
- Keep the final fast/slow relative-distance calculation outside counted training, since it is a diagnostic after the scored budget and does not affect any evaluation.

## Experimental Adjustments

- **Fused synchronization implementation**: Replaced the conceptual per-tensor loop with foreach lerp/copy to prevent launch overhead from testing an unnecessarily slow implementation. (ref: `02-plan-review.md` concern 1)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local preflight; scored production not launched)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/022/preflight-report.json`
- **WandB**: N/A
- **Status**: failed
- **Started**: 2026-08-06 14:32 UTC
- **Ended**: 2026-08-06 14:33 UTC

Description:
- Exact-corpus H20 preflight of the accepted CIFAR-10 recipe wrapped with Lookahead (`k=5`, `alpha=0.5`, persistent momentum). Two byte-aligned arms consumed one persisted set of 200 actual post-N1/M7/CutMix batches, with synchronization-local diagnostics through step 50 and safety tracking through step 200. The scored seed-42 production run was authorized only if this screen passed.

Observations:

- The materialized corpus contained 94 hard and 106 CutMix probability-target batches; parameters were byte-identical through steps 1-4, the fused recurrence agreed with the algebraic reference within `2.98e-08`, momentum buffers persisted at synchronization, and step 6 updated them. (source: `preflight-report.json` fields `hard_batches`, `soft_batches`, `steps_1_to_4_bitwise_equal`, `recurrence_max_abs_error`, `momentum_persisted_at_sync`, `step6_momentum_changed`)
- Candidate-only single-class concentration reached 95.3125% at step 7 versus 50.78125% control, and again 95.3125% at step 13 versus 85.9375% control. This crossed the registered >95% safety veto twice. (source: `preflight-report.json` field `candidate_only_concentration_failures`)
- Lower candidate terminal loss did not override the collapse veto: loss EMA was 1.91333 candidate versus 2.02415 control. (source: `preflight-report.json` fields `candidate_terminal_loss_ema`, `control_terminal_loss_ema`)
- The controller first raised on a harmless `2.98e-08` FP32 lerp-reference difference because that assertion preceded the concentration assertion; the already-serialized report exposed the independent research failure, so no controller rerun was performed.

Key Metrics:

- candidate/control 200-step loss-EMA ratio: 0.945253 (source: `preflight-report.json`)
- candidate-only concentration events: 2, at steps 7 and 13; maximum candidate share 95.3125% (source: `preflight-report.json`)
- scored `best_test_acc`: not measured — production blocked by the pre-registered safety gate.

## Verification Results

### Conditions Checked

- Scored verification skipped — execution failed at the mandatory pre-production safety condition, so timing and production were not run.

### Informational Metrics

## Errors & Dead Ends

### 2026-08-06 — Persistent-momentum Lookahead caused early class concentration
- Error: `candidate-only predicted-class share 0.953125 at steps 7 and 13`
- Root cause: Pulling parameters halfway toward the slow point every five updates while retaining fast-path momentum created an unsafe early location/velocity transient despite finite state and lower loss.
- Source: `preflight-report.json` field `candidate_only_concentration_failures`; exact persisted corpus `preflight-corpus.pt`.
- Do NOT retry: Retire exact `k=5`, `alpha=0.5` Lookahead with persistent momentum at LR 0.1; do not weaken the concentration gate, reroll data/seed, or proceed to timing/production.

## Human Notes

> Autopilot session; no human intervention requested.

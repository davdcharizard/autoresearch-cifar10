# EXP-020: Convolution-only official-order Gradient Centralization

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-020
- **Base Node**: 002
- **Commit**: `cf33559`
- **Outcome**: failed — complete valid run, primary metric below the parent-relative improvement gate

## Implementation Notes

### Summary

Implemented the planned single-variable eligibility ablation in `train.py`. Coupled `1e-4` L2 is materialized on all 44 gradients before momentum, while exact row-mean projection applies only to the 16 convolution weights; the inherited classifier, normalization, and bias directions therefore retain ordinary coupled-decay SGD semantics. Added cadence-512 fixed-device-scalar audits for total and phase-resolved convolution energy, raw/L2/cross removed-row-mean decomposition, path/inventory reconciliation, final-state finiteness, and final-16 evaluation context without adding an evaluation or forward pass.

### Surprises & Discoveries

The planned inventory arithmetic matched the live model exactly: 16 convolution tensors contain 2,742,704 elements and 2,256 rows; the 28 excluded tensors contain 6,186 elements. The raw/L2 decomposition needs the effective FP32 contribution (`regularized_mean - raw_mean`) rather than an idealized `weight_decay * mean(weight)` to describe the exact applied arithmetic.

### Decisions

The report-only raw/L2/cross reconstruction has a hybrid-scaled status but is not part of metric-run integrity, following the plan review: its numerical anomaly cannot invalidate an otherwise correct applied update. In contrast, applied-energy orthogonality, residual, phase reconciliation, and nonfiniteness remain hard integrity gates. Phase selection is computed before the helper call and drives only audit routing/counters; it never gates GC.

## Experimental Adjustments

- **No implementation adjustment after tests**: deterministic smoke passed exact inventory, audit-update bitwise parity, excluded two-step parity, zero evaluator calls, component error 0, applied-energy error `3.11e-11`, and max residual `5.96e-8`. The first complete preflight passed without repair: median ratio `1.019440`, p90 `1.023965`, MAD/median `0.003304`, parent drift `0.008854`, projected 27,417 steps / 141 epochs, zero live-allocation growth, and zero evaluator calls. (ref: `/tmp/exp020_gc_smoke.py` stdout; `/tmp/exp020_preflight.log` complete JSON)

## Run Log

### Run 1

Metadata:
- **Job ID**: local unified-exec session 54791
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed (research no-improvement)
- **Started**: 2026-08-06 16:07:05 UTC
- **Ended**: 2026-08-06 16:15:03 UTC

Description:
- The sole metric run will execute the fixed-seed EXP020 candidate on physical GPU 0 only after deterministic math and accuracy-blind throughput/allocation gates pass. It tests whether excluding the classifier from otherwise official-order, full-run GC improves EXP002 beyond 95.33% while preserving exposure. The expected outcome is a complete 300-second charged run with exact GC integrity diagnostics and one evaluation per epoch.

Observations:
- The sole metric process exited 0 after exactly 300.0 charged seconds and 461.1 total seconds. It completed 28,090 steps across 145 epochs with exactly 145 evaluations; no traceback, OOM, runtime error, or integrity failure appeared. (source: `run.log` L297-L316; evaluation-line count from `rg -c '^  eval ep' run.log` = 145)
- All 28,090 completed steps received convolution GC: 10,292 CutMix, 10,453 early-clean, and 7,345 late-clean, summing exactly to the call count. The 55 cadence audits reconciled across phase buckets and final integrity was `PASS`. (source: `run.log` L297-L304)
- GC removed 4.262522861646 of 24.44482219761 audited convolution squared-energy units, a 41.7580% norm ratio. The phase norm ratios were 40.7105% CutMix, 42.7511% early-clean, and 39.8024% late-clean, giving no evidence of a uniquely excessive late-phase projection. (source: `run.log` L299-L302)
- Effective L2 common-mode contribution was negligible: raw removed energy 4.262502784533, L2-only `1.1880089e-05`, cross `8.1970246e-06`; reconstruction error was `1.04e-15`. This makes raw-order GC numerically almost identical for the audited component, though it does not itself test accuracy causality. (source: `run.log` L303)
- The final-16 evaluations averaged 95.12125%, ranged 95.03-95.24%, and ended at 95.19%; the 95.24 best premium over that plateau was 0.11875 points. (source: `run.log` L305-L308)

Key Metrics:
- `best_test_acc`: 95.24% (parent EXP002 95.23%, +0.01 point; formal threshold 95.33%, failed) (source: `run.log` L307; `tree.sh show ... 002`)
- `final_test_acc`: 95.19%; `final_test_loss`: 0.2029 (source: `run.log` L308-L309)
- Exposure: 28,090 steps / 145 epochs; charged 300.0 s; total 461.1 s; peak VRAM 1,180.0 MiB; 2,748,890 parameters (source: `run.log` L310-L316)
- Mechanism integrity: PASS; applied decomposition error `3.0371e-9`; max absolute post-projection row mean `3.7253e-9`; phase energy error `1.3080e-15`; zero audited/final-state nonfinite values (source: `run.log` L299-L304)

## Verification Results

### Conditions Checked

- **Execution integrity and constraints — PASS**: physical GPU 0 was NVIDIA H20 UUID `GPU-b1bc897d-2183-dad2-8302-8800bc02a633` with one visible device; exit 0; 300.0 charged seconds; 461.1 total seconds; 145 evaluations for 145 epochs; unchanged parameter count; only tracked `train.py` changed; GC integrity `PASS`. (source: pre-launch GPU commands; `run.log` L297-L316; `git diff --name-only a36dc09`)
- **Primary metric — FAIL**: 95.24% is only +0.01 point over parent EXP002's 95.23% and is below the required 95.33%. This falls in the preregistered unresolved positive-movement band, so it is a formal no-improvement but does not trigger the `<=95.23%` official-order closure rule. (source: `run.log` L307; `tree.sh show ... 002`)
- **Remaining verification**: stopped after the primary necessary condition failed; raw evidence remains for the analysis-phase adversarial result audit.

### Informational Metrics

- Not collected as a verification-success appendix because the necessary metric condition failed; all durable run values are transcribed under Run 1 above for analysis.

## Errors & Dead Ends

- None.

## Independent Result Audit

- Claude returned `AUDIT_VERDICT: PASS`, independently recomputing the 95.24 maximum and formal no-improvement, freshness/scope, 145/145 evaluation cadence, full budget/summary, inventory/dose/audit arithmetic, energy and tail statistics, and absence of evaluator or reward-hacking changes (`04-result-review.md`).

## Human Notes

> Autopilot session; no execution-phase human intervention.

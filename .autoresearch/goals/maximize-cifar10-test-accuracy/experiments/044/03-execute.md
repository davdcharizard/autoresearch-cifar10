# EXP-044: Exact-Neutral Spatial-Dispersion Input

## Execution

Overall Status & Info:
- **Created**: 2026-07-27
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-044
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - offline/local-only run
- **Outcome**: failed - valid normal-exposure metric miss

## Implementation Notes

### Summary

Appended one exactly zero bias-free `128->64` adapter after all accepted pooled-head initialization inside the restoring CPU RNG fork. The final block preserves accepted GAP and unrolls the accepted MLP only to add the adapter's population-spatial-std contribution to its hidden preactivation. The tracked diff is 13 additions and 5 removals in `train.py`; data, objective, optimizer construction, schedule, RNG, and evaluator are unchanged.

### Surprises & Discoveries

The plan review exposed that zero-start checks alone cannot qualify the statistic backward because zero adapter weights block that path. The harness therefore includes a nonzero-adapter analytic derivative oracle and derives full CE/mixup adapter gradients without reading candidate hidden autograd.

### Decisions

Runtime hooks capture the real adapter input and pooled-head operations without adding production diagnostics. Timing interleaves early/hard accepted/candidate windows within each local block, so every combined retention pairs measurements from the same nearby GPU timeline.

## Experimental Adjustments

- None.

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session `64391`
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-27 06:48 UTC
- **Ended**: 2026-07-27 06:55 UTC

Description:
- Sole fixed-seed score of the exact zero-start population-dispersion adapter against accepted 94.48%. It launches only if source/state/statistic/gradient/update gates and paired H20 exposure qualification pass. Goal improvement requires best accuracy at least94.58%; mechanism support additionally requires at least127 realized passes.

Observations:
- Semantic preflight passed directly: source diff13/5, all common startup state/logits/gradients/RNG exact, only one zero `[64,128]` adapter appended, total1,011,674 parameters, and optimizer membership/options correct (source: semantic preflight stdout, 2026-07-27).
- Independent statistic backward errors were `1.39e-17` FP64 and at most `7.45e-9` FP32. Adapter gradient norms were `0.1571/0.1843` early/hard, mean/std correlations `0.8541/0.8352`, epsilon-floor ratios `0.00680/0.00697`, and maximum update error `2.98e-8`; diagnostics were not used to tune (source: semantic preflight stdout, 2026-07-27).
- Interleaved complete-step timing passed: accepted early/hard medians were `11.7519/11.4024ms`, candidate medians `11.9269/11.5748ms`; window CVs <=`0.3582%`, ratio CVs `0.4443%/0.0746%`, paired retentions `0.98157-0.98957`, median retention `0.984150`, projected `128.2387` passes, and candidate peak `626.693MiB` (source: timing preflight stdout, 2026-07-27).
- The sole score exited zero with one finite summary. Mixup stopped once at step16,147/195.0s and RandAugment after epoch-83 exhaustion at step16,185/195.5s; 26 evaluation epochs were unique every-fifth plus final129 and no error signature appeared (source: `run.log` L1-L71).
- Best accuracy reached `93.95%` at epoch125 and ended `93.83%`/`0.2637`; normal exposure makes this an attributable representation miss rather than timing or infrastructure failure (source: `run.log` L58-L71).

Key Metrics:
- `best_test_acc`: `93.95%`, `0.53` below baseline and `0.63` below threshold (source: `run.log` L62).
- `final_test_acc` / `final_test_loss`: `93.83%` / `0.2637` versus accepted `94.45%` / `0.2456` (source: `run.log` L63-L64).
- Exposure: `25,139` steps = `128.71168` passes across129 epochs (source: `run.log` L69-L70).
- Counted/wall/startup: `300.0/342.5/1.1s`; peak VRAM `1,096.5MiB`; params `1,011,674` (source: `run.log` L65-L71).

## Verification Results

### Conditions Checked

- **Completion/resource contract - PASS**: exit0; one H20; one finite summary;300.0s counted/342.5s wall; correct transitions;26 unique evaluations;1,011,674 params;128.71168 passes; no errors (source: `run.log` L1-L71 and local audit).
- **Primary metric improvement - FAIL**: best93.95% is below baseline94.48% and required94.58% (source: `run.log` L62).
- **Hypothesis support - FAIL**: exposure cleared127 but accuracy did not clear94.58%; the exact normal-exposure treatment is rejected.
- **Corroboration - skipped after metric failure**: final93.83% and loss0.2637 are recorded but not alternate criteria.

### Informational Metrics

- Skipped under fail-fast verification after primary failure; raw values remain above.

## Errors & Dead Ends

- None.

## Human Notes

> User requested uninterrupted autopilot and offline/local-only execution.

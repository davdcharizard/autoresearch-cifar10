# EXP-043: Convolution-Only Gradient Centralization

## Execution

Overall Status & Info:
- **Created**: 2026-07-27
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-043
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - offline/local-only run
- **Outcome**: failed - valid normal-exposure metric miss

## Implementation Notes

### Summary

Added one pure `centralize_convolution_gradients` helper implementing the preregistered per-output-filter projection, cached all convolution weights once after the accepted model reaches CUDA, and called the helper between the sole backward and unchanged SGD step. The tracked diff is 11 added lines in `train.py`; the model, forward graph, loss, data path, optimizer construction, schedule, RNG, timing, and evaluator are unchanged.

### Surprises & Discoveries

The adversarial plan review identified that an inline-only production loop could not be independently invoked by preflight without duplicating the treatment. Factoring the exact five-line operation into a pure helper closes that gap and includes one Python function call in both timing and the score. The accepted timer begins after DataLoader yield, so worker/RandAugment delivery is both unchanged and correctly excluded from the paired counted-step probe.

### Decisions

The semantic harness imports and calls the production helper directly while an AST/diff audit proves the scored call lies between backward and SGD. Timing uses a precomputed convolution list just like production, synchronized wall time rather than CUDA events, four local accepted/candidate pairs per regime, a 1% paired-ratio CV ceiling, and the conservative requirement that every paired whole-run retention clears the 127-pass floor.

## Experimental Adjustments

- **Inserted the project root into the ignored harness import path**: Direct execution from `experiments/043/` omitted the repository root from `sys.path`; adding the discovered absolute root enables `prepare`/candidate import without changing production or treatment semantics. (ref: 2026-07-27 verifier error below)

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session `66512`
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-27 05:57 UTC
- **Ended**: 2026-07-27 06:04 UTC

Description:
- Sole fixed-seed local score of exact full-run convolution-only gradient centralization against the accepted 94.48% frontier. It may launch only after evaluator-free source/state/projection/Nesterov qualification and the paired complete-step H20 exposure gate pass. Goal improvement requires best accuracy at least 94.58%; normal-exposure mechanism support additionally requires at least 127 realized passes.

Observations:
- Semantic preflight passed after one harness-only import-path correction. Source additions/removals were exactly the reviewed 11/0 lines; accepted/candidate state, forward, raw gradients, RNG, optimizer, constants, mixup, LR, temporal controls, and cadence matched. The cache contained exactly 18 convolution tensors, 983,472 values, and 1,392 filters (source: semantic preflight stdout, 2026-07-27).
- Projection/update oracles passed in both regimes. Maximum residual filter mean was `3.75e-8`, idempotence error `4.47e-8`, FP64 error `3.75e-8`, and fresh/preseeded parameter/buffer errors were at most `2.98e-8`. The stem lost `98.80%` early and `97.85%` hard gradient norm; deeper removed fractions ranged roughly `19-48%`, recorded without tuning (source: semantic preflight stdout, 2026-07-27).
- Pre-timing audit confirmed baseline `94.48` at `a7c42dc`, exactly one idle H20 with `0 MiB` used, correct EXP043 branch, only `train.py` tracked, frozen `prepare.py`/`pyproject.toml`, local CIFAR, ignored harness, clean diff, and absent `run.log` (source: local audit stdout, 2026-07-27).
- Paired complete-step timing passed. Early/hard accepted medians were `11.7703/11.4378 ms` and candidate medians `11.9123/11.5719 ms`; window CVs were at most `0.2974%`, paired-ratio CVs `0.3781%/0.1856%`, all four whole-run retentions were `0.98518-0.99306`, median retention was `0.987468`, projected exposure `128.671` passes, and candidate peak `622.712 MiB` (source: timing preflight stdout, 2026-07-27).
- The sole score exited zero with one finite summary. Mixup stopped once at step `16,257`/`195.0s`; RandAugment stopped once after epoch-84 exhaustion at step `16,380`/`196.4s`. Evaluations were unique at every fifth epoch plus final epoch 131, and no error signature appeared (source: `run.log` L1-L73).
- The run retained normal exposure but trailed throughout the clean tail, peaking at `93.88%` in epoch 125 and ending `93.87%`/`0.2661`. This is an attributable accuracy miss, not a timing or infrastructure failure (source: `run.log` L58-L73).

Key Metrics:
- `best_test_acc`: `93.88%`, `0.60` below baseline and `0.70` below threshold (source: `run.log` L64).
- `final_test_acc` / `final_test_loss`: `93.87%` / `0.2661` versus accepted `94.45%` / `0.2456` (source: `run.log` L65-L66).
- Exposure: `25,353` steps = `129.80736` passes across 131 epochs (source: `run.log` L71-L72).
- Counted/wall/startup: `300.0/341.2/1.1s`; peak VRAM `1,096.4 MiB`; parameters `1,003,482` (source: `run.log` L67-L73).

## Verification Results

### Conditions Checked

- **Completion/resource contract - PASS**: exit 0; one H20; one finite summary; `300.0s` counted, `341.2s` wall; correct transitions; 27 unique evaluations; `1,003,482` parameters; `129.80736` passes; no error signature (source: `run.log` L1-L73 and post-run local audit, 2026-07-27).
- **Primary metric improvement - FAIL**: best `93.88%` is below both baseline `94.48%` and required `94.58%` (source: `run.log` L64).
- **Hypothesis support - FAIL**: exposure cleared 127 passes, but accuracy did not clear 94.58%; this exact normal-exposure mechanism is rejected (source: `run.log` L64/L72).
- **Corroboration - skipped after necessary metric failure**: final `93.87%` and loss `0.2661` remain recorded above but are not alternate success criteria (source: `run.log` L65-L66).

### Informational Metrics

- Skipped under the fail-fast verification procedure after the primary metric failure; raw values are preserved in Run 1 Key Metrics.

## Errors & Dead Ends

### 2026-07-27 - Nested preflight could not import project modules
- Error: `ModuleNotFoundError: No module named 'prepare'`
- Root cause: Python placed the nested script directory, rather than the current project root, at the front of `sys.path`.
- Source: semantic preflight traceback at `preflight.py:38` before module/model construction
- Do NOT retry: Do not execute a nested harness without explicitly adding its independently resolved project root to `sys.path`.

## Human Notes

> User requested uninterrupted autopilot and offline/local-only execution.

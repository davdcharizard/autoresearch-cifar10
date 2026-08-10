# EXP-036: Reflection-Padded Strong and Weak Crops

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-036
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed — paired reflection trajectory exceeded global logit/gradient bounds; loader timing and production not authorized

## Implementation Notes

### Summary

Added `padding_mode="reflect"` to exactly the accepted weak and strong four-pixel RandomCrop constructors. No transform order, crop geometry, label mixing, model, optimizer, schedule, loader lifecycle, timer, evaluator, logging, or summary line changed. All counterfactual data and loader instrumentation remains ignored and experiment-local.

### Surprises & Discoveries

The production implementation is exactly two keyword additions. The nontrivial work is evidence: reflection changes input pixels by design, so paired checks must align stochastic decisions and targets rather than assert tensor equality.

### Decisions

Both phases use reflection to isolate one consistent boundary prior; no strong-only/weak-only mode, padding width, `edge`/`symmetric` alternative, transform reordering, or tensor-pipeline conversion was introduced.

## Experimental Adjustments

- **Stopped after candidate-specific global safety failure**: Both accepted control calibrations passed prospectively, but reflection reached 20.72x logit and 9.81x gradient ratios on aligned paired views. The frozen gate blocks loader timing and production; no threshold, persistence definition, or corpus was changed. (ref: `preflight-report.json`)

## Run Log

### Run 1

Metadata:
- **Job ID**: local preflight; no production PID
- **Log file(s)**: `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/036/preflight.log`; `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/036/loader-timing.log`; root `run.log` only if authorized
- **WandB**: N/A
- **Status**: failed preflight; loader timing and production skipped
- **Started**: 2026-08-06 21:47 UTC
- **Ended**: 2026-08-06 21:41 UTC

Description:
- The reflection candidate first proves exact two-keyword scope, 81-offset border semantics, matched transform RNG/target decisions, control-qualified global trajectory safety, and fresh strong/weak loader wall margin. Only complete passes authorize one seed-42 production run. The expected benefit is removal of an artificial crop-position cue; the primary implementation risk is NumPy-backed PIL reflection overhead in forkserver workers.

Observations:
- Static scope passed: compile, Ruff, format, pre-commit, exact two-keyword diff, and one idle 97,871-MiB H20; only tracked `train.py` changed and `data/` remained untouched. (source: static command output; `git diff`)
- Exhaustive offset semantics passed. The uniform-offset expected padding area was 13.4066% and observed constant/reflection changed area was 13.3295%; center crops were exact and no interior source pixel changed. (source: `preflight-report.json` `offsets`)
- The paired corpus preserved all RNG/target contracts: 32 strong batches split 16 hard/16 CutMix plus 16 weak hard batches, identical outgoing RNG and bitwise-equal targets. Full post-policy views differed in 23.6893% of elements on average after downstream transforms/mixing. (source: `preflight-report.json` `corpus`)
- Both accepted control/control calibrations passed before candidate authority. Their maximum logit/gradient/update ratios were 1.2328/1.6810/1.2482 and 1.8000/1.7508/1.0416, with no persistent concentration and terminal phase EMA ratios below 1.0. (source: `preflight-report.json` `control_calibrations`)
- Reflection stayed finite and had no candidate-only concentration step, but failed the frozen global geometry gates: maximum candidate/control logit/gradient/update ratios were 20.7200/9.8057/2.9802. Logit failures persisted from strong steps17-30 and gradient failures occurred at steps11 and17. (source: `preflight-report.json` `trajectory`)
- Whole update/parameter and preceding-median maxima remained bounded at 0.04719/1.3630, and strong/weak terminal loss EMA ratios were only 1.0499/1.0957. These favorable averages cannot waive the preregistered global veto. (source: `preflight-report.json` `trajectory`)
- Loader timing and production were correctly skipped. No `loader-timing.log`, `loader-timing.json`, or root `run.log` was created.

Key Metrics:
- primary `best_test_acc`: unavailable — production not authorized.
- offset expected/changed area: 13.4066%/13.3295%; full paired-view changed fraction: 23.6893%. (source: `preflight-report.json`)
- candidate/control maxima: logit20.7200x, gradient9.8057x, update2.9802x, update/parameter0.04719, preceding median1.3630; concentration steps none. (source: `preflight-report.json`)
- candidate/control strong/weak terminal loss EMA: 1.0499x/1.0957x. (source: `preflight-report.json`)
- candidate/controller SHA-256: `baf8760babca1fd1d215c32e1d06f010feb85f1230dc857c53015bf9f5fe1fc9` / `bd2912dc4d872d5ea1210184984c099bb3ca5e9f23b156b4ef0175e1f6ca1af8`.
- corpus/report/log SHA-256: `df7156774dbe62dd2e0d934eeab9835ef1fbd4c5c34f72db8fc44ca2f30330c6` / `b89ccea220e7a6394337e51c6ae612086b1f2b2d7349a62fb0d39721698dd570` / `8df27718755cb3a59e88411f97dc2da234ec1c5a2ef4883ea00168f7cdc1ccd9`.

## Verification Results

### Conditions Checked

- **Baseline/source/static semantics — pass**: current 94.15% baseline at `7c1e7d8`, exact two-keyword scope, code quality, transform parity, offset geometry, and RNG semantics passed.
- **Paired corpus/control qualification — pass**: aligned source/RNG/target corpus persisted, both accepted calibrations passed frozen bounds before candidate replay.
- **Candidate global trajectory safety — fail**: logit RMS reached 20.72x and gradient norm 9.81x control on paired strong views, exceeding the fixed 5x limit.
- **Loader timing/production/metric — skipped**: aborted immediately after the candidate-specific safety failure; no primary metric exists.

### Informational Metrics

## Errors & Dead Ends

### 2026-08-06 — Reflection paired strong-view geometry veto
- Error: `logit ratio` exceeded 5x on strong steps17-30 and `gradient ratio` exceeded 5x at steps11/17; maxima 20.7200x/9.8057x.
- Root cause: replacing the artificial constant border with reflected texture changed roughly 23.7% of post-policy tensor elements after RandAugment/CutMix propagation and drove persistent output-scale divergence, despite aligned RNG/targets and mild average loss degradation.
- Source: `preflight-report.json` SHA-256 `b89ccea220e7a6394337e51c6ae612086b1f2b2d7349a62fb0d39721698dd570`.
- Do NOT retry: do not restrict reflection to one phase, change padding width/mode, relax the 5x bound, select offsets/images, reroll corpus/seed, or proceed to loader timing/production.

## Human Notes

> Autopilot; no human intervention.

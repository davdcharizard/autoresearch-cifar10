# EXP-029: Conv2d-Weight-Only Data-Gradient Centralization

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-029
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Implemented the reviewed GC intervention in tracked `train.py` only. A no-grad helper discovers Conv2d modules, requires each weight gradient, and subtracts its mean across input/spatial dimensions. The helper is called exactly once after backward and immediately before the unchanged PyTorch momentum-SGD step, preserving accepted `P(g)+lambda*w` decay ordering and counting all new work inside the fixed timer.

### Surprises & Discoveries

The production diff is only eleven added lines and needs no parameter, buffer, optimizer group, RNG, forward, data, or evaluator change. This provides unusually clean attribution, but the minimal literal implementation still launches one reduction and one subtraction for each of 19 Conv gradients; the reviewed 1% timing gate is therefore a live likely veto rather than a formality.

### Decisions

- Keep module-type eligibility discovery inside the production helper exactly as proposed; do not precompute weights or introduce a faster semantically equivalent variant after observing timing.
- Keep all mechanism norm/finiteness diagnostics in ignored controllers so timing and production execute the identical minimal helper without synchronizing instrumentation.
- Treat the 26,629-step estimate as hypothesis evidence after production, not a way to invalidate a formal fixed-budget metric improvement.

## Experimental Adjustments

- **Protocol constants bound explicitly**: Read-only `prepare.py` import confirmed `NUM_WORKERS=8` and `TIME_BUDGET_S=300` before implementation. (ref: `02-plan-review.md` concern 4)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local preflight)
- **Log file(s)**: `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/029/preflight.log`, `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/029/preflight-report.json`
- **WandB**: N/A
- **Status**: failed (controller fixture)
- **Started**: 2026-08-06 18:32:35 UTC
- **Ended**: 2026-08-06 18:33:01 UTC

Description:
- Semantic and immutable-corpus H20 preflight of all-Conv data-gradient centralization against accepted momentum SGD. It will verify exact projection and decay/momentum ordering, then compare aligned models over the registered 200 strong and 64 weak batches. Timing is authorized only if every trajectory gate passes.

Observations:

- All 264 candidate trajectory steps passed the substantive safety gates: zero concentration events, maximum candidate/control update ratio 1.27408, strong/weak loss-EMA ratios 0.999624/1.003094, valid BN/state/RNG/corpora, and production-corpus projected filter means at most `1.49e-08`. (source: first `preflight-report.json` fields under `trajectory`)
- The controller rejected only its synthetic projection fixture: gradients with artificial offsets up to several units left `1.86e-06` FP32 reduction residual and `1.91e-06` second-projection change against an absolute `1e-07` check. The manual installed-SGD recurrence was exact. This is independent of candidate behavior and the immutable data trajectory. (source: first `preflight-report.json` fields `projection_fixture`, `recurrence_fixture`; `preflight.log`)

Key Metrics:

- first-attempt trajectory max update ratio: 1.274075; concentration events: 0 (source: first `preflight-report.json`)
- first-attempt strong/weak loss-EMA ratios: 0.999624 / 1.003094 (source: first `preflight-report.json`)

### Run 2

Metadata:
- **Job ID**: N/A (local preflight code retry)
- **Log file(s)**: `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/029/preflight.log`, `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/029/preflight-report.json`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 18:34:08 UTC
- **Ended**: 2026-08-06 18:34:42 UTC

Description:
- One allowed controller-code retry after scaling the deterministic synthetic fixture into ordinary FP32 gradient magnitudes. Candidate code, projection semantics, gates, model state, and both immutable corpora are unchanged. The retry must reproduce the same 264-step candidate trajectory while allowing the fixture to test implementation rather than extreme reduction precision.

Observations:

- The scaled fixture passed with FP64 reference error `1.18e-08`, idempotence residual `1.12e-08`, and exact five-step installed/manual parameter and momentum recurrence. The 264-step trajectory reproduced Run 1 exactly and passed every safety gate. (source: final `preflight-report.json` fields `projection_fixture`, `recurrence_fixture`, and `trajectory`)
- GC removed a nontrivial gradient component rather than being BN-negligible. Strong removed/raw fractions were stem 0.8727, stage1 0.3669, stage2 0.4449, stage3 0.5865; weak fractions were 0.9150, 0.4725, 0.5711, 0.4509. (source: final `preflight-report.json` field `trajectory.stage_gradient_fractions`)

Key Metrics:

- preflight status: pass; report SHA-256 `dd04816f...63ee` (source: `preflight-report.json`)
- max/median candidate-control update ratio: 1.274075 / 0.880251; concentration events: 0 (source: `preflight-report.json`)
- strong/weak loss-EMA ratios: 0.999624 / 1.003094 (source: `preflight-report.json`)

### Run 3

Metadata:
- **Job ID**: N/A (local paired timing)
- **Log file(s)**: `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/029/timing.log`, `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/029/timing-report.json`
- **WandB**: N/A
- **Status**: failed
- **Started**: 2026-08-06 18:37:46 UTC
- **Ended**: 2026-08-06 18:42:50 UTC

Description:
- One device-conditioning process followed by five counterbalanced fresh-process control/candidate pairs using real production loaders. Each scored arm measures 800 strong and 200 weak complete synchronized steps after warmup, with candidate execution calling the exact production helper. Production is authorized only if the aggregate candidate/control ratio is at most 1.01 and all secondary stability/lifecycle gates pass.

Observations:

- Five paired ratios were `1.014498, 1.042778, 1.027132, 1.005090, 1.009153`. Their aggregate candidate/control ratio was `1.019749`, exceeding 1.01, and the maximum pair exceeded 1.04. Raw trials were serialized before the controller raised. (source: `timing-report.json` fields `trials`, `overall_weighted_ratio`, `max_pair_ratio`; `timing.log`)
- Mean path ratios were 1.017750 strong-hard, 1.018790 strong-soft, and 1.025693 weak-hard. The effect was therefore broad rather than isolated to CutMix target handling. Historical projection fell from 26,898 to 26,377 steps. (source: `timing-report.json` per-path trial means and `historical_projected_steps`)
- Secondary gates passed: control/candidate CV 0.770%/1.830%, peak 598.68 MiB, loader delivery 112.70x consumption, max weak rebuild 2.903s, lifecycle wall/count 1.00968, projected total 333.60s, and all workers stopped. (source: `timing-report.json`)
- The registered timing failure blocked the sole production run. No fused kernel, layer subset, coefficient, phase restriction, or threshold rescue was attempted. (source: `timing.log`)

Key Metrics:

- aggregate candidate/control step ratio: 1.019749 (gate <=1.01; failed) (source: `timing-report.json`)
- maximum paired ratio: 1.042778 (gate <=1.04; failed) (source: `timing-report.json`)
- projected steps: 26,377 versus accepted 26,898 (source: `timing-report.json`)
- scored `best_test_acc`: not measured — production blocked by timing gates.

## Verification Results

### Conditions Checked

- Scored verification skipped — execution failed at mandatory paired timing conditions before production.

### Informational Metrics

## Errors & Dead Ends

### 2026-08-06 — Synthetic fixture magnitude made absolute projection tolerance non-diagnostic
- Error: `preflight gates failed: ['FP64 projection reference', 'projection idempotence']`
- Root cause: The fixture added parameter-index-dependent offsets up to several units, so ordinary FP32 mean subtraction left about `1.9e-06` rounding residual despite exact recurrence and `1.49e-08` production-gradient residual. The absolute-only check measured artificial reduction scale, not wrong dimensions/semantics.
- Source: first `preflight-report.json` fields `projection_fixture` and `trajectory.max_post_projection_filter_mean`; `preflight.log`.
- Do NOT retry: Do not weaken production-corpus gates or change the candidate. Use the same deterministic nonzero-offset fixture at ordinary gradient scale and permit only this one controller-code retry.

### 2026-08-06 — Literal all-Conv GC exceeded fixed-budget timing gates
- Error: `timing gates failed: ['aggregate timing ratio', 'paired timing ratio']`
- Root cause: Nineteen separate mean reductions and in-place subtractions added 1.97% aggregate full-step cost on the H20; overhead appeared consistently across strong hard/soft and weak hard paths, with one pair at 4.28%.
- Source: `timing.log`; `timing-report.json` fields `overall_weighted_ratio`, `max_pair_ratio`, `trials`, and `historical_projected_steps`.
- Do NOT retry: Retire this literal full-strength all-Conv helper for EXP029; do not fuse, subset layers, delay phases, change strength, or relax timing thresholds as a rescue. Any lower-overhead GC formulation requires a new reviewed experiment.

## Human Notes

> Autopilot session; no human intervention requested.

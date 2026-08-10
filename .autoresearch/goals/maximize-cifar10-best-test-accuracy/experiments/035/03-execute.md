# EXP-035: Fixed SiLU Throughout ResNet-20

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-035
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed — formal site-gradient trajectory gate; timing and production not authorized

## Implementation Notes

### Summary

Replaced exactly the three source-level functional ReLU calls in `train.py` with fixed beta-1 `F.silu`, producing 19 dynamic SiLU sites across the stem and nine residual blocks. No module, parameter, initialization, data, optimizer, schedule, timer, evaluator, logging, or summary line changed. All semantic instrumentation remains isolated in ignored experiment-local artifacts.

### Surprises & Discoveries

The tracked implementation is exactly the reviewed three-call substitution; no production helper or structural accommodation was needed.

### Decisions

Kept the accepted ReLU-oriented Kaiming initialization unchanged to isolate activation semantics and avoid repeating EXP034's unstable initialization reparameterization. Used functional out-of-place SiLU exactly as specified; no learned beta, site exception, gain, approximation, or fusion was introduced.

## Experimental Adjustments

- **No retry after non-specific formal gate failures**: The absolute tensor-relative rule was undefined for zero-initialized BN biases, and both accepted control/control calibrations exceeded the same 5x site-gradient ratio used to veto the candidate. These findings weaken interpretation but do not authorize post-observation threshold or statistic changes. (ref: `preflight-report.json`; Errors & Dead Ends)

## Run Log

### Run 1

Metadata:
- **Job ID**: local preflight; production PID pending authorization
- **Log file(s)**: `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/035/preflight.log`; `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/035/timing.log`; production root `run.log` only if authorized
- **WandB**: N/A
- **Status**: failed preflight; timing and production skipped
- **Started**: 2026-08-06 21:19 UTC
- **Ended**: 2026-08-06 21:25 UTC

Description:
- The fixed SiLU candidate first undergoes exact source/topology/RNG/oracle checks, controller identity tests, two production-default control/control calibrations, and a real immutable-corpus trajectory screen. Only a complete safety pass authorizes seven paired fixed-budget timing trials, and only a timing pass authorizes one seed-42 production run. The expected benefit is smoother signed representation flow; the main risks are pooled-feature cancellation, strong-phase underfit, and activation-backward overhead.

Observations:
- Static scope passed: compile, Ruff, format, pre-commit, diff-check, exact three-call diff, and an idle 97,871-MiB H20 all passed; only tracked `train.py` changed and user-owned `data/` remained untouched. (source: preflight command output; `git diff`)
- Construction and oracle checks passed: 19 Conv/19 BN/one Linear, 1,073,962 parameters, identical initial state and CPU/CUDA RNG, 19 dynamic activation sites, and SiLU value/derivative maximum errors of zero with derivative 0.5 at zero. (source: `preflight-report.json` `construction`/`self_test`)
- Initial real hard/soft gates passed. Candidate/control loss, logit-RMS, pooled-RMS, and gradient-norm ratios were 0.6183/0.6434/0.5867/0.4430 and 0.6121/0.6441/0.5867/0.4420. Control began 100% one-class while SiLU shares were 87.5%/91.41%, so there was no candidate-only concentration. (source: `preflight-report.json` `initial_function`)
- Corpus integrity passed before and after replay: registered 200 strong batches (94 hard/106 soft) and 64 weak batches retained their file/tensor declarations. (source: `preflight-report.json` `corpus`)
- The candidate had no candidate-only >95% class-share step, stayed finite with exact 264 BN counters, and remained within global gates: max candidate/control logit/pooled/gradient/update ratios were 2.4685/1.2207/2.1666/1.6355; whole update/parameter and preceding-median maxima were 0.01068/1.3214. Strong/weak terminal loss EMA ratios were favorable at 0.9333/0.8751. (source: `preflight-report.json` `trajectory`)
- Production was nevertheless vetoed by 18 site-gradient ratio failures at steps 9-58 and four absolute per-tensor relative-update failures at steps 1-4. The site-gradient maximum was 8.2441x, but accepted control/control maxima were also 9.5078x and 5.6247x; the per-tensor maximum was inflated to `4.09e28` by division through zero-initialized parameter norms, while controls reached `9.03e28`. (source: `preflight-report.json` `trajectory`/`control_control_calibrations`)
- Under the frozen protocol, those formal failures block timing and production even though they are not candidate-specific. No `timing.log`, `timing-silu.json`, or root `run.log` was created.

Key Metrics:
- primary `best_test_acc`: unavailable — production not authorized.
- candidate/control maxima: logit 2.4685x; pooled 1.2207x; gradient 2.1666x; update 1.6355x; site gradient 8.2441x. (source: `preflight-report.json`)
- candidate/control terminal strong/weak EMA: 0.9333x/0.8751x; candidate/control minimum BN variance: 0.021640/0.056358. (source: `preflight-report.json`)
- candidate source SHA-256: `d80faf3628593a194765a0f33ea0b3bd1cb11e7972ced176ece3b080007ff94b`; controller SHA-256: `ccf14047ae0db07b6cdfdb483219b11dbd03ac2b8f61e22d09bff41457ae2cb4`. (source: `preflight-report.json`)
- report/log SHA-256: `ad129e253686aa27872a3c89aea5dea10c625952eef82d46dc5bc1d31e7eaf99` / `c5acd0486348193c9cf717f6b5c47baf39aa017f1f71942a55f9cb02c8ab090e`.

## Verification Results

### Conditions Checked

- **Baseline/source/static semantics — pass**: current baseline 94.15% at `7c1e7d8`; exact three-call activation-only tracked diff and all code-quality/topology/RNG/oracle checks passed.
- **Immutable corpus — pass**: both registered real post-transform corpora matched hashes, schemas, counts, ordering, and post-replay declarations.
- **Trajectory safety — fail**: candidate exceeded the formal `[0.20,5.0]` site-gradient ratio at 18 steps and the absolute per-tensor relative-update bound at four steps; timing and production stopped immediately.
- **Timing/production/metric — skipped**: aborted after the trajectory failure; no primary metric exists.

### Informational Metrics

## Errors & Dead Ends

### 2026-08-06 — Zero-norm per-tensor relative-update statistic
- Error: `max_tensor_relative_update` reached `4.09e28` candidate and `9.03e28` accepted control, tripping steps 1-4.
- Root cause: BatchNorm biases are initialized at exact zero, so an absolute `update.norm / parameter.norm` statistic is undefined for those tensors; the `1e-30` denominator clamp converts ordinary first updates into meaningless huge ratios.
- Source: `preflight-report.json` `trajectory.maxima` and both `control_control_calibrations`.
- Do NOT retry: future protocols must define zero-norm tensor handling before candidate execution (absolute update, delayed denominator, or exclude analytically zero-start tensors); do not amend EXP035 after observing its candidate.

### 2026-08-06 — Non-specific site-gradient ratio veto
- Error: candidate site-gradient ratio exceeded 5x at 18 steps, maximum 8.2441x.
- Root cause: per-site ratios divide by small, CUDA-variable accepted gradients; both predeclared accepted control/control calibrations also crossed the same nominal bound at 9.5078x and 5.6247x. The frozen gate is therefore formally failed but cannot support a candidate-specific instability claim.
- Source: `preflight-report.json` `trajectory.failures` and `control_control_calibrations`.
- Do NOT retry: do not relax the 5x bound, aggregate sites differently, select steps, enable deterministic backends, reroll controls/corpus/seed, or run timing/production after the veto.

## Human Notes

> Autopilot; no human intervention.

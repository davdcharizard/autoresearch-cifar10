# EXP-017: Learned Pool-First Transition Shortcuts

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-017
- **Commit**: (pending - committed on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Replaced only the two CIFAR ResNet transition shortcuts with exact average-pool, learned bias-free 1x1 projection, and BatchNorm sequences, while representing all seven same-shape shortcuts as identity modules. A marker convolution class keeps the accepted model-wide initialization stream unchanged: constructor draws are isolated, both new tensors receive sequential Kaiming draws from one seed-derived CPU generator, and the later initializer skips only those tensors. Ignored structural/numerical/loader and paired timing controllers implement the approved conjunctive preflight protocol. External Claude reviewed the exact production diff and controller sources and returned `APPROVED` before any controller execution.

### Surprises & Discoveries

External review empirically confirmed that all accepted shared tensors and the post-construction global CPU RNG state remain bitwise equal to the control, despite the two additional learned projections. The 202-batch deterministic production augmentation stream has exactly 50% CutMix, and an eight-worker lifecycle probe shut down cleanly during the review. Claude identified only fail-safe timing-dispersion risk and the possibility that actual exposure lands just below the 25,500 attribution floor; neither can create a false metric pass.

### Decisions

The new projection generator is derived directly from the active `torch.initial_seed()` and consumed sequentially for layer2 then layer3. This preserves run determinism and the global RNG stream, but the new draws are seed-coupled rather than statistically independent of the shared initializer; attribution will state that explicitly. Timing-controller reruns are permitted only for a documented dispersion-only false failure, as approved in the plan and external review; a valid production run remains strictly non-rerunnable.

## Experimental Adjustments

- **No source adjustment after implementation review**: Claude found no blocking correctness defect and explicitly approved the exact sources. (ref: `02-plan-review-implementation-addendum.md`)

## Run Log

### Run 1

Metadata:
- **Job ID**: local supervisor PID 2353666; exec session 74421
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 09:30:36 UTC
- **Ended**: 2026-08-06 09:36:01 UTC

Description:
- One fixed-seed production run will test whether learned normalized pool-first transition shortcuts improve CIFAR-10 best test accuracy over the accepted 94.15 baseline. Launch is conditional on every reviewed structural, numerical, timing, exposure, memory, and loader gate. The expected formal threshold is at least 94.25, while 94.25-94.35 remains noise-consistent weak evidence and mechanism attribution additionally requires at least 25,500 actual steps.

Observations:
- Before production launch, mandatory controller source review completed with exit code 0 and `APPROVED`. (source: `02-plan-review-implementation-addendum.md`)
- Structural gate passed with exact topology, parameters, shared-state equality, RNG equality, pooling semantics, and hard/soft gradient recruitment. (source: `preflight-structural.json`)
- Numerical gate passed on the fixed real-batch stream with no candidate-only divergence or concentration failure. (source: `preflight-numerical.json`)
- Five paired timing trials and the loader lifecycle gate passed every conjunctive launch criterion. (source: `timing-shortcut.json`, `preflight-loader.json`)
- The sole seed-42 production run exited zero with a complete summary. Switch and first-weak accuracy were 90.20% and 93.45%, but best accuracy peaked at 94.09%, below both the 94.15 frontier and 94.25 acceptance threshold. (source: `run.log` evaluation trajectory and final summary)

Key Metrics:
- parameters: 1,084,586; transition/identity shortcuts: 2/7; shared-state and RNG equality: true/true (source: `preflight-structural.json`)
- CutMix rate: 48.51%; hard/soft shortcut RMS ratios: 0.699-0.713; hard/soft projection update ratios: 0.140-0.216%; terminal candidate/control loss-EMA ratio: 0.9263 (source: `preflight-numerical.json`)
- training ratio: 1.02803; projected steps: 26,164; p95 ratio: 1.04748; expected evaluations: 18; projected wall: 317.92 s (source: `timing-shortcut.json`)
- loader CutMix: 50%; wait median/p95: 0.168/0.205 ms; candidate step median: 14.028 ms; strong/weak workers stopped: 8/8; weak target ndim: 1 (source: `preflight-loader.json`)
- best/final accuracy: 94.09%/94.05%; final NLL: 0.2024; switch/first-weak accuracy: 90.20%/93.45% (source: `run.log` evaluation trajectory and final summary)
- training/total/startup: 300.0/331.5/1.0 s; peak VRAM: 598.8 MiB; epochs/steps: 69/26,557; parameters: 1,084,586 (source: `run.log` final summary)

## Verification Results

### Conditions Checked

- **Primary improvement**: failed - `best_test_acc=94.09%` is below the required `94.25%` and 0.06 points below the 94.15 moving baseline. Remaining result-quality verification stops at this failed necessary condition; integrity checks below establish that the result is valid rather than invalid. (source: official index query; `run.log` final summary)
- **Complete numeric summary**: passed - all 10 expected fields were present and finite after an exit-zero run. (source: `run.log` final summary)
- **Time and resource protocol**: passed - training was exactly 300.0 s, total wall 331.5 s, startup 1.0 s, and one H20 was used under the 600 s supervisor. (source: `run.log` final summary; launch metadata)
- **Scope and model**: passed - only `train.py` was tracked and the run reported exactly 1,084,586 parameters. (source: prelaunch `git diff --name-only`; `run.log` final summary)
- **Lifecycle, targets, and evaluator**: passed - switch occurred once at 80.0%, all eight strong workers stopped, 10,590/21,279 batches used CutMix (49.77%), the weak loader produced hard targets, and 19 evaluations occurred on 19 unique epochs with at most one per epoch. (source: `run.log`; `preflight-loader.json`)
- **Attribution exposure**: passed - 26,557 actual optimizer steps exceeded the 25,500 mechanism-support floor. (source: `run.log` final summary)

### Informational Metrics

- switch/first-weak/best/final accuracy: 90.20%/93.45%/94.09%/94.05% (source: `run.log`)
- final test loss: 0.2024; best-final gap: 0.04 points (source: `run.log`)
- training/total/startup: 300.0/331.5/1.0 s; peak VRAM: 598.8 MiB (source: `run.log`)
- epochs/steps/evaluations: 69/26,557/19; parameters: 1,084,586 (source: `run.log`)
- realized CutMix: 49.77%; strong workers stopped: 8 (source: `run.log` augmentation switch)

## Errors & Dead Ends

## Human Notes

> Autopilot execution; no human intervention requested.

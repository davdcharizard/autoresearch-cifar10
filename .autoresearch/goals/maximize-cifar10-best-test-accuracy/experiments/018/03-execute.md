# EXP-018: Late Arithmetic SWA with In-Budget BN Recalibration

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-018
- **Commit**: (pending - committed on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Added importable production helpers in `train.py` for uniform FP32 parameter averaging, synchronized charged snapshots, ordered mean installation, cumulative BatchNorm reset/refresh, and buffer validation. The accepted online path remains unchanged until 86%; weak epoch-endpoint snapshots are collected before the epoch's sole evaluation, SGD stops once charged progress reaches 98%, and the remaining counted budget installs SWA and refreshes BN on the existing hard weak loader before one terminal SWA evaluation. One provenance line records online/SWA attribution and integrity without changing the ten-field summary.

### Surprises & Discoveries

External plan review showed that seven snapshots in an 88-98% window had zero timing margin, so the reviewed window widened to 86-98% and preflight now requires eight projected endpoints. External source review also found that the first timing controller scaled historical steps instead of measured speed and used constant evaluation/wall gates; these were corrected before any controller ran. Snapshot placement moved immediately before the epoch evaluation so charged work crossing 98% cannot create an online-plus-SWA double evaluation.

### Decisions

The accumulator stores flat detached vectors for compact exact arithmetic but installs by ordered tensor copies, avoiding parameter-storage aliasing. BN refresh temporarily uses `momentum=None` and restores original momenta, matching cumulative SWA-update semantics. Production requires 4.5 seconds remaining, seven actual snapshots, quantitative spread floors, at least 390 refresh batches, and `install_step == num_steps`. A mergeable result additionally requires final SWA accuracy itself to clear 94.25 and the recorded online best.

## Experimental Adjustments

- **Widened snapshot window to 86-98%**: Provides one-epoch margin while avoiding the earliest rapidly adapting weak checkpoints. (ref: `02-plan-review.md`)
- **Bound controllers to production helpers**: Arithmetic and refresh gates import exact `train.py` logic rather than copies. (ref: `02-plan-review.md`)
- **Replaced decorative timing projections**: Steps/evaluations/wall now depend on fresh synchronized measurements. (ref: `02-plan-review-implementation-addendum.md`)

## Run Log

### Run 1

Metadata:
- **Job ID**: local supervisor PID 2371672; exec session 79864
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 11:00:30 UTC
- **Ended**: 2026-08-06 11:05:53 UTC

Description:
- One fixed-seed production run will test whether a uniform average of weak-tail endpoint parameters improves the accepted late solution. Launch requires externally approved exact sources plus arithmetic, cumulative BN refresh, lifecycle, timing, snapshot-margin, exposure, evaluator-count, memory, and wall gates. The final SWA model itself must reach at least 94.25% and no less than the pre-install online best; the run is never rerolled.

Observations:
- Mandatory external idea, plan, revised-plan, implementation, and corrected-source reviews all exited zero; no fallback reviewer was used. (source: EXP-018 review artifacts)
- Arithmetic gate matched a seven-snapshot FP64 reference within `1.19e-7`, preserved online model/optimizer/gradient/RNG state, and rejected reordered parameters. (source: `preflight-arithmetic.json`)
- Real cumulative refresh processed 780 batches across two iterator traversals in 3.001 s, aligned all BN counters, restored momentum, left parameters/optimizer/gradients exact, and stopped eight workers. (source: `preflight-refresh.json`)
- Five fresh timing children passed every joint launch gate with one projected snapshot of margin. (source: `timing-swa.json`)
- The sole production run exited zero with a complete finite summary and all SWA/process-integrity gates. Online best reached 94.02%; final SWA fell to 93.85% with NLL 0.2037, so both accuracy gates failed. (source: `run.log`)

Key Metrics:
- arithmetic reference snapshots/error: 7 / `1.19e-7`; real snapshot: 0.504 ms; state/RNG equality: true (source: `preflight-arithmetic.json`)
- refresh batches/counter/time: 780/780/3.001 s; install+finish: 0.0108 s; workers stopped: 8 (source: `preflight-refresh.json`)
- measured/conservative step: 10.894/11.128 ms; projected snapshots/steps/evals/wall: 8/26,412/18/324.99 s (source: `timing-swa.json`)
- projected snapshot work/390-refresh/peak memory: 0.0736/1.395 s/610.96 MiB; trial CV: 0.60% (source: `timing-swa.json`)
- online best/final SWA/final NLL: 94.02%/93.85%/0.2037; switch/first weak: 88.09%/93.21% (source: `run.log`)
- snapshots/range/spread: 8 / 87.14-97.30% / median `7.26e-3`, first-last `2.67e-2` (source: `run.log` SWA line)
- install/final steps: 26,453/26,453; refresh/BN batches: 1,624/1,624; snapshot/install/refresh: 0.0292/0.0084/5.9911 s (source: `run.log`)
- training/total/startup: 300.0/332.4/1.2 s; peak VRAM: 611.0 MiB; epochs/evaluations: 69/18 (source: `run.log`)

## Verification Results

### Conditions Checked

- **Primary improvement**: failed - `best_test_acc=94.02%` is 0.13 points below the 94.15 frontier and 0.23 below the required 94.25%. (source: official index query; `run.log`)
- **Final SWA metric integrity**: failed - final SWA accuracy 93.85% is below 94.25% and 0.17 points below the pre-install online best of 94.02%. (source: `run.log` SWA line)
- **Completion/summary/time/resource**: passed - exit zero, 10 finite fields, 300.0 counted seconds, 332.4 total seconds, 1,073,962 parameters, and one H20. (source: process status; `run.log`)
- **SWA arithmetic/exposure/BN**: passed - eight snapshots, spread floors exceeded, 26,453 steps with `install_step == num_steps`, and 1,624 aligned cumulative BN batches. (source: `run.log` SWA line and summary)
- **Lifecycle/targets/evaluator**: passed - one 80.0% switch, eight stopped workers, 10,746/21,581 CutMix batches (49.79%), hard refresh targets, and 18 evaluations on 18 unique epochs. (source: `run.log`; controller artifacts)

### Informational Metrics

- final SWA NLL: 0.2037 versus EXP-010's 0.1934; worse by 0.0103. (source: `run.log`; EXP-010 analysis)
- online best/final SWA gap: -0.17 points; best/final summary gap: 0.17 points. (source: `run.log`)
- snapshot median consecutive/first-last RMS: `7.2556e-3` / `2.6714e-2`. (source: `run.log` SWA line)
- training/total/startup/VRAM: 300.0 s / 332.4 s / 1.2 s / 611.0 MiB. (source: `run.log`)

## Errors & Dead Ends

## Human Notes

> Autopilot execution; no human intervention requested.

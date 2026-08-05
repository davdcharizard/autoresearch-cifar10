# EXP-033: Three-Point Terminal Parameter Average

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-033
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - local-only run
- **Outcome**: failed - valid result below margin

## Implementation Notes

### Summary

Added two detached device-resident snapshots of all trainable parameters at the first post-update states whose pre-step counted times reach 95% and 97.5%. At budget exhaustion, the implementation fully materializes and finite-checks the uniform average of those snapshots and terminal live parameters, temporarily installs it for the existing terminal evaluator call, and restores every terminal live parameter in `finally` with an elementwise equality check.

### Surprises & Discoveries

Counting snapshot clone work necessarily shifts subsequent time-derived LR/progress slightly, so accepted trajectory identity is valid only through the first snapshot. The reviewed protocol now measures and bounds that intended timer-only divergence instead of claiming impossible full-run identity.

### Decisions

The snapshot-due policy is a pure helper so the verifier exercises production threshold semantics directly. Averaged tensors are fully computed and checked before any live overwrite; terminal BN buffers, parameter objects, optimizer references/state, and evaluator cadence remain untouched. Evaluation-consumed RNG is preserved exactly, while backup/average/install/restore arithmetic must consume none.

## Experimental Adjustments

- **Hardened review contracts**: Replaced full-run trajectory identity with exact-prefix plus timer-offset bounds, separated evaluator RNG from arithmetic RNG, replaced scalar restoration checks with elementwise equality, and defined timing per one clone sweep. (ref: `02-plan-review.md`)
- **Normalized verifier device identity**: Compared CUDA device type rather than `cuda:0` against unspecialized `cuda`; production was unchanged. (ref: first semantic preflight error)

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 1359480 (timeout PID 1359476)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-07-27 00:32:21 UTC
- **Ended**: 2026-07-27 00:38:36 UTC

Description:
- This will be the sole local fixed-seed score only if semantic and H20 overhead qualification pass. The intervention keeps all accepted live training, earlier evaluation, augmentation, model, and optimizer choices, then evaluates a predetermined uniform `[95%,97.5%,100%]` trainable-parameter average at the terminal epoch with terminal live BN buffers. Primary success is `best_test_acc >=94.42%`; `final_test_acc >=94.42%` is separate mechanism corroboration.

Observations:
- Static audit passed: production compiles, only tracked `train.py` differs from `67c8e98`, `prepare.py` is unchanged, CIFAR-10 is local, one H20 is idle, and no scored log exists. (source: setup commands)
- Semantic preflight passed after the verifier-only device fix: exact accepted construction/model/optimizer/RNG prefix, 987,098 parameters, 50 trainable tensors and optimizer states, ordered 285.0/292.5 thresholds, finite fixed-order arithmetic, terminal-buffer and evaluator-RNG semantics, injected-failure restoration, and exact live restoration. (source: semantic preflight stdout)
- H20 overhead timing passed: two snapshots add 0.000682 seconds total, retention is 0.9999977, projected exposure is 133.00706 passes, maximum LR offset is 1.225e-7, and the terminal sequence adds 0.00540 seconds for a 345.305-second wall projection. Candidate/control/terminal CVs were 0.0855%/0.4440%/2.2804%. (source: timing preflight stdout)
- The sole score completed without numerical, CUDA, worker, or evaluator error. Mixup disabled at step 16,587/195.0 s and RandAugment after iterator exhaustion at step 16,770/197.1 s, a valid 183-step lag. (source: `run.log` lines 40-42)
- Snapshots occurred exactly once at step 24,549/pre-step 285.002 s and step 25,211/pre-step 292.509 s. The existing final epoch used the three-point average once and every terminal live parameter restored elementwise exactly. (source: `run.log` lines 60-69)
- Evaluations occurred once at each fifth epoch through 130 plus final partial epoch 133; all 27 epochs were unique. The averaged endpoint equaled the prior live epoch-130 top-1 at 93.87% while loss changed from 0.2560 to 0.2606. (source: `run.log` lines 6-69)

Key Metrics:
- **Preflight projected passes**: 133.00706; **retention**: 0.9999977.
- **Snapshot timer offset**: 0.000682 s total; **lost-step bound**: 1; **max LR offset**: 1.225e-7.
- **Projected wall**: 345.305 s; **timing peak allocation**: 19.87 MiB.
- **Score**: best/final 93.87%/93.87%; delta -0.45/-0.35 points versus accepted 94.32%/94.22%.
- **Final loss**: 0.2606; +0.0083 versus accepted 0.2523; best-final gap 0.00 points.
- **Execution**: 25,873 steps, 133 epochs, 132.46976 passes, 300.0 counted / 342.8 total / 1.1 startup seconds.
- **Resources/state**: 1,096.3 MiB peak VRAM; 987,098 parameters; `terminal_restore_exact=true`.

## Verification Results

### Conditions Checked

- **Run integrity**: PASS - exit 0, one H20, one finite summary, 300.0 counted and 342.8 total seconds, 132.46976 passes, exact topology/scope, two ordered snapshots, correct temporal transitions, 27 unique evaluations, one averaged terminal call, and exact live restoration. (source: `run.log`; final source audit)
- **Primary metric**: FAIL - `best_test_acc=93.87%` is 0.45 points below baseline and 0.55 below the required 94.42%. No rerun. (source: `run.log` lines 71-80)
- **Mechanism corroboration**: FAIL - averaged `final_test_acc=93.87%` is below 94.42%, and final loss 0.2606 is worse than 0.2523. (source: `run.log` lines 69-73)

### Informational Metrics

- Skipped as a formal success-only collection after the primary condition failed; all execution values needed for analysis are preserved above.

## Errors & Dead Ends

### 2026-07-26 - Verifier compared indexed and unspecialized CUDA devices
- Error: `AssertionError` while requiring snapshot tensors on `DEVICE`; tensors reported `cuda:0` while `DEVICE` was `cuda`.
- Root cause: the ignored harness compared full `torch.device` identities instead of the shared CUDA device type.
- Source: first semantic preflight traceback at `experiments/033/preflight.py:278`.
- Do NOT retry: compare `value.device.type` for an unspecialized single-GPU device contract.

## Human Notes

> Autopilot local-only execution; no user intervention requested.

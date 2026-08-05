# EXP-026: Worker-Safe Early-Only RandAugment

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-026
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - offline/local-only session
- **Outcome**: failed

## Implementation Notes

### Summary

Added the fixed torchvision one-operation magnitude-5 policy after accepted crop/flip, controlled by an unlocked shared byte passed through the same explicit multiprocessing context as the persistent DataLoader. RandAugment owns a worker-local RNG stream swapped in only around the torchvision call, so its operation randomness does not advance accepted crop/flip streams. The main process disables the policy only after a normally exhausted epoch at or beyond 65% counted time.

### Surprises & Discoveries

The host's verified multiprocessing default is `forkserver`, not Linux's older `fork` default. A context-created shared `Value` is still transferred as shared memory under forkserver; the marker preflight tests actual propagation rather than assuming it from the start-method name.

### Decisions

The plan review's RNG-confound concern was addressed by isolating RandAugment randomness without adding a seed: each worker lazily clones its current state into a private RandAugment stream, updates that stream per call, and restores the accepted state in `finally`. Transition validity is measured as a step lag below one 195-batch epoch rather than a brittle fixed wall-time window.

## Experimental Adjustments

- **Isolated RandAugment worker RNG**: adopted after adversarial plan review to preserve exact accepted crop/flip and clean-tail trajectories while retaining standard stochastic operations (ref: `02-plan-review.md`, concern 1).

## Run Log

### Run 1

Metadata:
- **Job ID**: PID 1324933 (exec session 72202)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-26 20:48:51 UTC
- **Ended**: 2026-07-26 20:54 UTC

Description:
- One offline local H20 score of the accepted WRN/mixup recipe with fixed early-only `N=1,M=5` RandAugment. The augmentation is removed at the first exhausted epoch boundary after the accepted 65% mixup cutoff, leaving all later epochs on accepted crop/flip and hard labels. The run launches only after exact worker-RNG/cutoff replay and conservative wall-time preflights pass.

Observations:
- Semantic retry passed: forkserver shared-state propagation, 14-operation fixed policy, unchanged model/RNG/logits/parameters, isolated worker augmentation RNG, exact accepted clean-tail replay, and next-epoch cutoff markers all passed (source: `preflight.py --semantics`, 2026-07-26).
- Balanced real-loader timing passed. Accepted/candidate median paced epoch times were 2.652912/2.691617 seconds with CV 0.003955/0.044253; inactive boundary epochs were 2.658194/2.656913 seconds. Historical-differential and live-absolute total projections were 346.735/426.101 seconds, below 500 (source: `preflight.py --loader-timing`, 2026-07-26).
- The sole scored run exited 0 with no worker, CUDA, non-finite, or timeout error. Mixup stopped at step 17,859 / 195.0s and RandAugment stopped after epoch 92 exhausted at step 17,940 / 195.9s, an 81-step lag below one epoch (source: `run.log` L42-L44).
- The run completed 27,822 steps in 300.0 counted / 345.2 total seconds and scored 94.12%, below the required 94.17%; no rerun or policy adjustment was made (source: `run.log` L66-L77).

Key Metrics:
- semantic preflight: PASS; context: forkserver; policy operations: 14; parameters: 691,674 (source: semantic preflight output).
- loader preflight: PASS; accepted/candidate epoch median 2.652912/2.691617s; projected totals 346.735/426.101s (source: loader-timing output).
- best/final accuracy: 94.12% / 94.12%; final loss: 0.2574 (source: `run.log` L68-L70).
- training/total/startup: 300.0s / 345.2s / 1.1s; epochs/steps/passes: 143 / 27,822 / 142.44864 (source: `run.log` L71-L76).
- peak VRAM / parameters: 1,094.0 MiB / 691,674 (source: `run.log` L74-L77).

## Verification Results

### Conditions Checked

- Baseline: PASS - 94.07% at `eb08811`, threshold 94.17% (source: `exp-index.sh baseline`, 2026-07-26).
- Scope/device/compile: PASS - one H20, only tracked `train.py` changed, clean diff/compile (source: preflight audit, 2026-07-26).
- Semantic/cutoff integrity: PASS - exact policy/context, model/RNG identity, accepted clean-tail replay, and marker no-leak proof (source: semantic preflight output).
- Loader feasibility: PASS - CVs <=5%; 346.735/426.101s projections <=500s (source: loader-timing output).
- Completion/process integrity: PASS - exit 0; 300.0 counted / 345.2 wall seconds; 142.44864 passes; unique eval epochs; one mixup and one correctly ordered 81-step-lag RandAugment transition; no error markers (source: `run.log` L42-L77).
- Primary metric: FAIL - `best_test_acc=94.12% <94.17%` (source: `run.log` L68). Verification stopped at this necessary-condition failure.

### Informational Metrics

- Not promoted as success-only informational metrics; complete values are preserved in Run 1 Key Metrics above.

## Errors & Dead Ends

### 2026-07-26 - Inference probe reused for preflight backward
- Error: `RuntimeError: Inference tensors cannot be saved for backward`
- Root cause: The semantic harness created `probe` inside `torch.inference_mode()` for logit equality, then reused it for the optimizer smoke backward.
- Source: first `preflight.py --semantics` attempt, traceback at preflight line 214.
- Do NOT retry: Do not reuse inference tensors in autograd checks; construct a normal backward-only probe.

## Human Notes

> Autopilot run; no intervention requested.

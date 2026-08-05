# EXP-036: Scaled Pooled-Feature Residual MLP Head

## Execution

Overall Status & Info:
- **Created**: 2026-07-27
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-036
- **Commit**: a7c42dc
- **PR**: N/A - offline/local-only session
- **Outcome**: completed

## Implementation Notes

### Summary

Added the fixed bias-free `128 -> 64 -> 128` ReLU residual branch after global
pooling at scale 0.1. The branch is registered only after every accepted tensor
has been initialized, under an isolated CPU-default-generator seed, so accepted
common tensor bytes and both global RNG streams can remain exact.

### Surprises & Discoveries

The reviewed proposal originally used `torch.manual_seed` inside a CPU-only RNG
fork. That API also seeds CUDA and would have contaminated the scored mixup
trajectory; the plan critic caught this before implementation.

### Decisions

Use `torch.random.default_generator.manual_seed(36036)` inside
`torch.random.fork_rng(devices=[])`. The preflight must prove both CPU restoration
and byte-identical CUDA state. A completed score below 130 passes remains a
valid nonrepeatable goal result, explicitly superseding the proposal's invalid
classification, while making the head mechanism operationally inconclusive.

## Experimental Adjustments

- **Initialize bias-free weights directly**: the first semantic run exposed
  that the accepted `_weights_init` assumes every `nn.Linear` has a bias and
  called `zeros_(None)`. Applying its exact Kaiming-normal matrix operation to
  each new weight fixes the mechanical incompatibility without changing the
  reviewed topology, seed, or treatment (ref: error below).

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 33944
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed (full summary; local process exited)
- **Started**: 2026-07-27 02:07 UTC
- **Ended**: 2026-07-27 02:14 UTC

Description:
- The sole fixed-seed score will run locally on one H20 only after semantic,
  diagnostic, and complete-body timing gates pass. It tests whether a small
  nonlinear remapping of accepted pooled features improves generalization
  while preserving the spatial learner and at least 130 projected passes.
  Primary success is 94.42% best test accuracy.

Observations:

- Semantic preflight passed after the bias-free initializer fix: all 987,098
  common parameter bytes and CPU/CUDA RNG states matched; total parameters
  were 1,003,482. Initial residual/direct norm ratio was 0.120864 and logit RMS
  perturbation 0.069719; all grouped gradients were finite/nonzero (source:
  local semantic preflight stdout, 2026-07-27 02:06 UTC).
- Timing preflight passed with retention 0.982799, projected 130.71955 passes,
  maximum CV 0.004325, and 610.2 MiB candidate peak after reset (source: local
  timing preflight stdout, 2026-07-27 02:07 UTC).
- The sole score produced one complete finite summary with no numerical, CUDA,
  worker, or evaluator error. Mixup disabled at step 16,336/195.0 s and
  RandAugment at the epoch-84 iterator boundary at step 16,380/195.5 s, a
  valid 44-step lag (source: `run.log` lines 38-40).
- Evaluations occurred once every fifth epoch through 130 plus final epoch 131;
  all 27 evaluation epochs were unique (source: `run.log` lines 6-62).

Key Metrics:

- **Score**: best/final 94.48%/94.45%; best delta +0.16 points over baseline
  94.32 and +0.06 over the 94.42 threshold; final delta +0.23 over accepted
  94.22 and +0.13 over the corroboration floor (source: `run.log` lines 64-65).
- **Final loss**: 0.2456, -0.0067 versus accepted 0.2523; best-final gap 0.03
  points (source: `run.log` lines 64-66).
- **Execution**: 25,450 steps, 131 epochs, 130.304 passes, 300.0 counted /
  343.9 total / 1.1 startup seconds (source: `run.log` lines 67-72).
- **Resources/model**: 1,096.4 MiB peak VRAM and 1,003,482 parameters (source:
  `run.log` lines 70-73).
- **Exposure projection**: 130.71955 projected versus 130.304 realized, a
  -0.41555-pass residual; the run stayed above the protected 130-pass regime.

## Verification Results

### Conditions Checked

- **Run integrity**: PASS - one H20, one finite full summary, 300.0 counted and
  343.9 total seconds, 130.304 passes, exact topology/source, correct temporal
  transitions, and 27 unique evaluations (source: `run.log`; final source audit).
- **Primary metric**: PASS - `best_test_acc=94.48%` exceeds baseline 94.32 by
  0.16 points and required 94.42 by 0.06 (source: `run.log` lines 64-65).
- **Mechanism corroboration**: PASS - final accuracy 94.45 exceeds 94.32 and
  loss 0.2456 improves on 0.2523 (source: `run.log` lines 65-66).

### Informational Metrics

- Peak VRAM 1,096.4 MiB; final accuracy 94.45%; final loss 0.2456; 300.0
  counted / 343.9 total seconds; 131 epochs; 25,450 steps; 1,003,482 parameters
  (source: `run.log` lines 65-73).

## Errors & Dead Ends

### 2026-07-27 - Accepted Linear initializer assumes a bias tensor
- Error: `AttributeError: 'NoneType' object has no attribute 'zero_'` while
  applying `_weights_init` to the bias-free pooled-head linears.
- Root cause: the accepted initializer always calls `init.zeros_(m.bias)` for
  `nn.Linear`, while the reviewed head explicitly uses `bias=False`.
- Source: first semantic preflight traceback at `preflight.py:141`, production
  `train.py:141`.
- Do NOT retry: initialize bias-free head matrices directly with the accepted
  `init.kaiming_normal_` weight operation; do not add biases to satisfy a helper.

## Human Notes

> Autopilot session; no execution-time intervention requested.

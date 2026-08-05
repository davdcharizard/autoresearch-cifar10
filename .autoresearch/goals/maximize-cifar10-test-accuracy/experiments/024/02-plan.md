# Plan EXP-024: Two Diagonal Conditional Stage-3 Gates
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Implement exact-neutral self-gating
- [x] Create branch `autoresearch/maximize-cifar10-test-accuracy-024` from accepted `eb08811`; modify only `train.py`.
- [x] Add a 128-channel diagonal conditional gate with zero `weight`/`bias` vectors and scale `2*sigmoid(weight*pool(residual)+bias)`; attach one to each stage-3 residual branch before shortcut addition.
- [x] Construct and initialize the entire accepted WRN first, then attach both zero-only gates without random draws; preserve every accepted parameter/buffer and post-construction CPU/CUDA RNG exactly.
- [x] Compile and audit the 692,186-parameter diagnostic-free model.

### Milestone 2: Pass semantic and timing gates
- [x] Semantic preflight passed exact state/RNG/logits, two placements, unit scales, shortcuts, device/dtype, no-decay grouping, and finite nonzero aggregate gradients for all vectors.
- [x] Timing preflight measured 0.978872 retention and 138.901932 projected passes with every CV below 0.98%, passing the gate.
- [x] Status, root Python files, `git diff --check`, and complete diff audit showed only planned `train.py` model changes.

### Milestone 3: Execute once and verify
- [x] Confirm one H20, remove stale `run.log`, and run exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Exit 0 with finite summary, 300.0 counted/341.7 wall seconds, one mixup transition, 692,186 parameters, unique epoch evaluations, and no errors.
- [x] `best_test_acc=93.91` failed the 94.17 threshold; diagonal self-gating is closed without rescue or rerun.

## Code Changes
- **`train.py` / `DiagonalStage3Gate`**: register zero `weight` and `bias` vectors of length 128; global-average-pool the signed residual, compute channel scales with `2*sigmoid(weight*pooled+bias)`, and multiply the residual only.
- **`train.py` / `PreActBlock`**: add an optional `gate` module and apply it after `conv2` before the existing shortcut addition.
- **`train.py` / `WideResNet`**: keep accepted construction and model-wide initialization unchanged; optionally attach a new diagonal gate to both existing stage-3 blocks afterward. Zero tensor construction must consume no RNG and preserve accepted common state.
- **`train.py` / production**: instantiate the gated model and retain the existing topology log plus new parameter count; add no gate diagnostics or training-loop changes.

## Configuration Changes
- Attention: none -> two diagonal conditional gates on 128-channel stage-3 residuals.
- Parameters: 691,674 -> 692,186; all 512 new vector parameters follow the existing `ndim < 2` no-decay policy.
- Model widths/depth, accepted common initialization, FP32, seed 42, data/crop/flip, batch-shared alpha-0.2 mixup through 65%, LR/floor, Nesterov momentum, weight decay, batch/workers, counted-time accounting, and evaluation cadence remain unchanged.
- Mechanism interpretation: a score below 94.17 rejects this exact per-channel self-gating treatment and strengthens, but does not prove, the case for global cross-channel interaction. A pass satisfies the fixed goal criterion but does not establish multi-seed causal sufficiency. Neither outcome invites parameter rescue.

## Execution Environment
- Method: offline local evaluator-free semantic/timing preflight, followed only on pass by one scored command.
- Resources: exactly one NVIDIA H20, local CIFAR-10, existing `uv`, persistent workers; no network, package install, remote service, W&B, GitHub, or `gh`.
- Estimated runtime: preflight under 3 minutes; score about 340 seconds wall with hard 600-second timeout.
- Log output: project-root `run.log`, retained through analysis only.
- Tool skill: none.

## Abort Criteria
- Abort before scoring on wrong gate count/placement/shape, any nonzero initial vector, non-unit scale, accepted common state/logit/RNG mismatch, shortcut gating, device/dtype error, gate parameter absent from no-decay group, missing/zero/non-finite aggregate first-step vector gradient norm, syntax/scope error, timing CV >5%, projected passes <138, OOM, or non-finite state.
- During scoring abort/classify on nonzero exit, timeout, OOM, traceback/non-finite output, wrong count, missing summary, missing/multiple mixup transition, or duplicate epoch evaluation.
- Never tune vectors, add cross-channel terms, change placement/schedule, alter thresholds, or rerun a completed valid score.

## Verification Protocol

### Verification Procedure

1. Query `exp-index.sh baseline`; within 10 seconds require 94.07 at `eb08811` and threshold 94.17.
2. Run H20, compile, status/untracked/root-Python, and full-diff audits; within 30 seconds require one H20 and only planned tracked `train.py` changes.
3. Run `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/024/preflight.py --semantics` with dummy `Eval`, timeout 120 seconds. Construct accepted/gated models from identical CPU/CUDA states and require counts 691,674/692,186; every accepted state tensor equal; identical serialized post-construction CPU/CUDA RNG; exactly two length-128 zero weight/bias gates on `layer3[0:2]` residuals; exact unit scales and accepted logits; unchanged shortcuts; gate device/dtype after `.to`; all four vectors in the no-decay group; finite nonzero aggregate first-backward gradient norms on all four vectors.
4. Run the same ignored experiment-scoped preflight with `--throughput`, timeout 180 seconds. Reproduce accepted/candidate production timed bodies with pinned copies, LR/group writes, private CUDA RNG, Beta/randperm mixup or hard labels, forward/backward/Nesterov step, and synchronize. After >=20 warm steps, measure three balanced >=40-step windows per model/regime; compute median means, CVs, `0.65*mixup+0.35*hard` aggregates, and `141.9*accepted_ms/candidate_ms`. Require every CV <=0.05, projected passes >=138, exact counts, finite output/loss, and explicit pass. Retain the approved floor but record realized exposure explicitly because projections have overestimated prior scores.
5. Remove stale `run.log`; execute exactly `timeout 600s uv run train.py > run.log 2>&1` once. Do not score again after valid completion.
6. Parse `run.log` with `rg`/`awk`; require exit 0, complete finite summary, `best_test_acc >=94.17`, counted seconds >=300, total <600, 692,186 parameters, one transition at 65%, unique eval epochs, and no error text. Stop at first failed necessary condition.
7. On metric pass only, audit final `train.py` diff/status and collect informational metrics. On metric failure, record no-improvement and close diagonal self-gating.

### Informational Metrics (Optional)
- Final summary: peak VRAM, final accuracy/loss, counted/wall seconds, epochs, steps, and parameters.
- Derived from `run.log`: effective passes, evaluation count, best epoch, best/final gap.
- Preflight stdout: regime window means/CVs, weighted retention, projected passes, and peak memory.

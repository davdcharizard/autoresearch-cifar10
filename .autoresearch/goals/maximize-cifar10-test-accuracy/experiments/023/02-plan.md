# Plan EXP-023: Selective Width with Full Two-Gate SE
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Implement exact-neutral composed architecture
- [x] Create branch `autoresearch/maximize-cifar10-test-accuracy-023` from accepted `eb08811`; modify only `train.py`.
- [x] Replace uniform final width with explicit `STAGE_WIDTHS=(32,64,160)` while preserving both high-resolution stages and all accepted training logic.
- [x] Add diagnostic-free ratio-16 SE to both stage-3 residual branches before shortcut addition. Build and initialize the width-only network first, then attach `160->10->160` gates from CPU seed 23017 inside a restored RNG fork; zero the second projection for exact unit scales.
- [x] Compile and log the exact stage topology/968,302 trainable parameters without adding runtime gate observation.

### Milestone 2: Pass semantic and production-path feasibility gates
- [x] Semantic preflight passed exact topology/counts, accepted early-stage topology, composed/width-only common state and RNG, exact unit logits, placement, grouping, and two-step gradient opening.
- [x] Timing was stable but projected 126.206224 passes, below the fixed 127-pass floor; scoring aborted without changing the threshold.
- [x] Status, root Python files, `git diff --check`, and complete diff audit showed only planned `train.py` production changes.

### Milestone 3: Execute once and verify
- [x] Confirmed one H20; scored command was not launched because the preregistered exposure gate failed.
- [x] Scored-run completion conditions were not reached.
- [x] `best_test_acc` is unavailable; recorded the fail-closed preflight outcome without tuning or rerun.

## Code Changes
- **`train.py` constants/model widths**: replace `WIDEN_FACTOR=2` with `STAGE_WIDTHS=(32,64,160)` and validate/use the explicit three-stage tuple for blocks, final BatchNorm, classifier, and topology logging.
- **`train.py` / `Stage3SE`**: global-pool the signed residual, apply biased `Linear(160,10)`, ReLU, biased `Linear(10,160)`, and `2*sigmoid` scaling; multiply only the residual before unchanged shortcut addition.
- **`train.py` / construction**: construct and initialize the complete width-only model first. Inside `torch.random.fork_rng(devices=[])`, seed only `torch.random.default_generator` with 23017, attach a gate to each stage-3 block, Kaiming-initialize first projections, and zero first biases plus second weights/biases. Do not perturb caller CPU/CUDA RNG.
- **`train.py` / logging**: print `WRN-16 stages=[32, 64, 160]` and the parameter count only; no gate diagnostics or result-conditioned control flow.

## Configuration Changes
- Stage widths: `[32,64,128] -> [32,64,160]`.
- Attention: none -> two exact-neutral ratio-16 stage-3 SE gates, seed 23017.
- Parameters: 691,674 -> 968,302 (width-only reference: 961,562; gates add 6,740).
- FP32, depth `[2,2,2]`, seed 42, crop/flip, batch-shared alpha-0.2 mixup through 65%, LR/floor, Nesterov momentum, selective weight decay, batch/workers, counted-time accounting, and evaluation cadence remain unchanged.
- Evidence interpretation: `160->10->160` gates operate on a new feature distribution and are not assumed to inherit EXP-017's +0.09. The hypothesis explicitly tests whether full conditional selection transfers and becomes complementary to added width. Both gates remain because later EXP-018/019 disproved final-only and static approximations.

## Execution Environment
- Method: offline local evaluator-free preflight followed, only on pass, by one scored local command.
- Resources: exactly one NVIDIA H20, local CIFAR-10, existing `uv`, persistent DataLoader workers; no network, packages, remote services, W&B, GitHub, or `gh`.
- Estimated runtime: preflight under 4 minutes; score about 340 seconds wall, hard timeout 600 seconds.
- Log output: project-root `run.log`, retained through analysis only.
- Tool skill: none.

## Abort Criteria
- Abort before scoring on wrong topology/count/gate placement or shape, non-unit initial gate, composed logits/state mismatch versus width-only, accepted early-stage topology mismatch, width-only/composed CPU/CUDA RNG drift, shortcut gating, wrong device/dtype or optimizer groups, missing/non-finite two-step gradients, syntax/scope error, timing CV >5%, projected passes <127, OOM, or non-finite state.
- During scoring abort/classify on nonzero exit, timeout, OOM, traceback/non-finite output, wrong topology/count, missing summary, missing/multiple mixup transition, or duplicate epoch evaluation.
- Never tune width, ratio, seed, placement, initialization, schedule, or threshold after preflight/results; never rerun a completed valid score.

## Verification Protocol

### Verification Procedure

1. Query `exp-index.sh baseline`; within 10 seconds require baseline 94.07 at `eb08811` and threshold 94.17.
2. Run `nvidia-smi --query-gpu=name --format=csv,noheader`, `uv run python -m py_compile train.py`, `git status --short --untracked-files=all`, root-Python and full-diff audits; within 30 seconds require one H20 and only planned tracked `train.py` production changes.
3. Run an ignored experiment-scoped semantic preflight with dummy `Eval`, timeout 120 seconds. Instantiate accepted `[32,64,128]`, width-only `[32,64,160]`, and composed `[32,64,160]+SE` under reset seed 42. Require counts 691,674/961,562/968,302; exact block/projection/BN/classifier topology with unchanged 32/64 early stages; all composed non-gate tensors equal width-only; serialized width-only/composed post-construction CPU/CUDA RNG equal; exactly two `160->10->160` gates on residuals; scale/logit identity versus width-only; shortcuts unchanged; gate device/dtype correct; matrices in decay and biases in no-decay; finite nonzero second-projection gradients on step one and first-projection gradients after one optimizer step.
4. Run an ignored warm throughput preflight with timeout 180 seconds. Reproduce the exact production timed body for accepted and composed models with private RNG streams, pinned copies, LR/group writes, Beta/randperm mixup or hard labels, forward/backward/Nesterov step, and synchronize. Measure three balanced >=40-step windows per model/regime after >=20 warm steps; use median window means and `0.65*mixup + 0.35*hard` aggregates. Require each CV <=0.05, projected passes `141.9 * accepted_ms / candidate_ms >=127`, finite outputs/loss, exact counts, and explicit pass. The 127 floor is stricter than the idea review's minimum 125, retains 89.5% of accepted exposure, and avoids scoring at the standalone-overhead product estimate; it is still lower than standalone SE because this experiment explicitly tests a higher-ceiling composed model.
5. Remove stale `run.log`; run exactly `timeout 600s uv run train.py > run.log 2>&1` once. Do not score again after a valid completion.
6. Parse `run.log` with `rg`/`awk`; require exit 0, complete finite summary, `best_test_acc >=94.17`, counted time >=300, total <600, exact topology/968,302 parameters, exactly one transition at the 65% boundary, unique eval epochs, and no errors. Stop immediately at the first failed necessary condition.
7. On metric pass only, perform final `git diff -- train.py`/status audit and collect informational metrics. On a valid metric failure, record no-improvement and interpret the composition as sub-additive without rescue.

### Informational Metrics (Optional)
- `peak_vram_mb`, final accuracy/loss, counted/wall seconds, epochs, steps, and parameters: final `run.log` summary.
- Effective passes: `num_steps * 256 / 50000`; evaluation count and best/final gap from `run.log`.
- Preflight accepted/candidate mixup/hard timings, CVs, weighted retention, projected passes, and peak memory: preflight stdout.

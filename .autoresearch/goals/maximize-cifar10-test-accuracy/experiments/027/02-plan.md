# Plan EXP-027: Extra Final Block Plus Early RandAugment
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Compose the two exact prior treatments
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-027` from `eb08811`; modify only `train.py` in production.
- [x] Reproduce EXP-011 exactly with `STAGE_BLOCKS=(2,2,3)`, strict three-count validation, unchanged block internals/widths, direct seed-42 whole-model construction/initialization, 987,098 parameters, and exact topology logging.
- [x] Reproduce EXP-026 exactly with fixed early `N=1,M=5` bilinear/mean-fill RandAugment, private worker RNG swapping, one explicit forkserver shared byte/loader context, and an exhausted-epoch cutoff at or after 65%.
- [x] Preserve accepted FP32 SGD, batch, LR/floor/warmup, decay, alpha-0.2 batch-shared mixup, seed, time budget, and evaluator cadence; compile and audit the complete diff.

### Milestone 2: Prove component identity and composition feasibility
- [x] Create ignored `experiments/027/preflight.py` with a fail-closed evaluator; verify exact EXP-011 topology/parameter/init oracle and exact EXP-026 transform/RNG/cutoff/no-leak oracle, including unchanged model state when transform plumbing is constructed before it.
- [x] Run matched complete counted-body H20 timing for accepted `[2,2,2]` and composed `[2,2,3]` models across mixup/hard regimes; require every CV <=2% and >=130 projected passes from 141.9 accepted. This gate verifies only the deep-model counted exposure regime; loader/wall risk is separate.
- [x] Run balanced real-loader timing at the composed GPU-consumer pace; require CV <=5%, clean boundary/workers, finite correct batches, and both historical-differential and live-absolute wall projections <=500 seconds.

### Milestone 3: Run the sole score
- [x] Confirm one H20, remove stale `run.log`, and launch exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Monitor worker/CUDA/numerical health and both transitions without reacting to interim accuracy; never change either component or rerun a valid completion.
- [x] Require exit 0, 300 counted seconds, total <600, 987,098 parameters, >=130 realized passes, unique evaluation epochs, one mixup transition at/after 195 seconds, and one subsequent exhausted-epoch RandAugment transition with lag in `[0,195)` steps.

### Milestone 4: Verify score and interaction
- [x] Require both `best_test_acc >=94.17%` and predetermined endpoint `final_test_acc >=94.17%` versus baseline 94.07%, reducing max-over-evaluations noise risk.
- [x] Record all summary metrics and compare final loss to EXP-011's 0.2782: lower supports sub-additive generalization harm; equal/higher falsifies that interaction explanation but does not override the primary metric verdict.
- [x] Accept only on goal conditions. Any valid miss closes the exact composition; never tune depth, policy, cutoff, fill, interpolation, seed, LR, or exposure gate.

## Code Changes
- **`train.py` / architecture**: replace scalar `NUM_BLOCKS=2` with `STAGE_BLOCKS=(2,2,3)`; strictly validate three positive non-Boolean integers; build each stage from its count; instantiate directly under seed 42 and log `widths=[32,64,128] blocks=[2,2,3]`. This must reproduce EXP-011, including its shape-dependent direct initialization trajectory, rather than introducing accepted-first mutation.
- **`train.py` / augmentation**: copy EXP-026's `EarlyRandAugment` semantics exactly. Use torchvision's standard fixed policy after crop/flip; lazily clone a worker-local augmentation RNG, swap/update it only around RandAugment, and restore the accepted worker state in `finally`. Use the same explicit multiprocessing context for an unlocked byte and DataLoader, explicit prefetch 2, and flip once only after a normal iterator exhausts at/after 65% counted time. This stacks early invariance on the deeper learner; it does not alter the later clean-tail distribution.
- **`.autoresearch/.../experiments/027/preflight.py`**: ignored verification-only harness combining the prior topology, worker replay, GPU timing, and real-loader timing oracles; never touches evaluator/test data.

## Configuration Changes
- Stage block counts: `[2,2,2] -> [2,2,3]`; parameters 691,674 -> 987,098; widths remain `[32,64,128]`.
- Training images: accepted crop/flip -> fixed early RandAugment after crop/flip, then exact accepted crop/flip after the first exhausted epoch ending at/after 65%.
- Loader: implicit context/prefetch -> same effective verified forkserver context and explicit prefetch 2; eight workers, persistence, shuffle, pinning, drop-last unchanged.
- Every optimizer, schedule, batch, mixup, numeric, seed, time, and evaluation setting remains accepted.

## Execution Environment
- Method: offline local preflights, then one local score only on pass; no remote/network/install/W&B/GitHub/`gh`.
- Resources: one NVIDIA H20, local CIFAR-10, installed environment, eight persistent forkserver workers.
- Estimated runtime: preflights 2-4 minutes; scored run about 340-430 seconds wall, hard timeout 600.
- Log output: scored stdout/stderr only in root `run.log`, retained through analysis then removed.
- Tool skill: none.

## Abort Criteria
- Abort before scoring on any topology/count/init mismatch with EXP-011; any policy/order/RNG/cutoff/marker/context mismatch with EXP-026; model RNG drift caused by transform setup; wrong optimizer/constants; scope/syntax failure; evaluator/test access.
- Abort before scoring if GPU timing CV >2%, projected passes <130, loader CV >5%, malformed/non-finite batches, worker/boundary failure, or either projected wall time >500. The 130-pass floor is intentionally close to EXP-011's regime and must not be relaxed.
- During score classify on nonzero exit, timeout, OOM/resource/worker/non-finite error, missing/stale summary, wrong count/topology, <130 passes, invalid/repeated transitions, duplicate eval epoch, or total >=600. Never rerun a valid score.

## Verification Protocol

### Verification Procedure

1. Query `exp-index.sh baseline`; within 10 seconds require 94.07 at `eb08811`, threshold 94.17.
2. Run one-H20, local-data, scope/status, `git diff --check`, full diff, and `uv run python -m py_compile train.py` audits; within 30 seconds require only tracked production change `train.py` and frozen `prepare.py`.
3. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/027/preflight.py --semantics`; within 180 seconds require strict stage-count validation, exact `[2,2,3]` topology and 987,098 params, and byte-equal full model state against an independent direct EXP-011 seed-42 construction made without transform plumbing. Require equal post-construction CPU/CUDA RNG, initial logits, first-epoch sampler labels, and CUDA mixup stream; correct optimizer groups and finite step. Also require exact EXP-026 transform order/14-op policy/context, private worker RNG, accepted tail replay, exhausted-boundary marker no-leak, and one non-reenable source path.
4. Run `uv run python .../experiments/027/preflight.py --throughput`; within 240 seconds measure accepted/candidate complete counted paths with private RNG, balanced three-window mixup/hard timing and CV <=0.02. Require `141.9 * accepted_weighted_ms / candidate_weighted_ms >=130.0`, finite state, correct logits/counts, and pass. This establishes only model-side counted exposure.
5. Run `uv run python .../experiments/027/preflight.py --loader-timing`; within 300 seconds use balanced fresh base/composed loader arms paced at the candidate counted-step median, one warmup plus three epochs each. Require CV <=0.05, clean inactive boundary/workers, and both `338.5 + max(0, candidate_epoch_median-base_epoch_median)*134 <=500` and `38.5 + candidate_epoch_median*134 <=500`, anchored to EXP-011's total/counting time and epochs.
6. Remove stale log and execute exactly `timeout 600s uv run train.py > run.log 2>&1` once; require exit 0 and no result-conditioned rerun.
7. Parse with `rg`; require one finite summary, 300 counted seconds, total <600, 987,098 params, `num_steps*256/50000 >=130`, unique accepted-cadence eval epochs, one mixup transition at/after 195 seconds and one later exhausted RandAugment transition with step lag `[0,195)`, and no error markers.
8. Require both `best_test_acc >=94.17` and `final_test_acc >=94.17`; stop on either failure. Record final loss relative to 0.2782 and audit the final diff regardless of that informational mechanism result.

### Informational Metrics (Optional)
- Final summary accuracy/loss, counted/total/startup time, epochs, steps/passes, VRAM, parameters; source `run.log`.
- Transition epoch/step/time and lag; source `run.log`.
- Final-loss interaction: candidate minus EXP-011 0.2782; lower supports the interaction, higher falsifies it.
- Preflight GPU/loader window values, CVs, projected passes, boundary times, and wall projections.

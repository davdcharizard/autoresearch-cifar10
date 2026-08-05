# Plan EXP-032: Reflection-Padded Random Crops
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Implement the exact crop-geometry treatment
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-032` from accepted commit `67c8e98`; modify only `train.py` in production and keep `prepare.py`/evaluator frozen.
- [x] Change exactly `transforms.RandomCrop(32, padding=4)` to `transforms.RandomCrop(32, padding=4, padding_mode="reflect")`; preserve every other source line, compile, and audit the one-line diff.

### Milestone 2: Prove decision isolation, intended pixels, and loader stability
- [x] Create ignored `experiments/032/preflight.py` with a fail-closed evaluator and independent `git show 67c8e98:train.py` oracle; prove accepted construction/model/optimizer/RNG identity and 987,098 parameters.
- [x] Reconstruct fixed crop offsets and flip decisions from independently restored RNG states; before RandAugment, prove exact equality away from padding-derived output pixels against an independent NumPy reflection oracle, intended confinement on touched pixels with a consistently flipped mask, and measure padded-window incidence over >=100,000 draws.
- [x] Exercise preflight-only instrumented transforms inside forkserver workers and require per-sample accepted/candidate equality of sampler index, worker id, crop `(i,j)`, flip bit, active flag, decoded RandAugment op/sign, private-state hashes, target, and post-iterator main RNG across an exhausted active epoch and the inactive next epoch.
- [ ] Run real-loader active/inactive paced timing with raw output before assertions; require finite 195-batch epochs, every CV <=5%, no worker starvation at accepted consumer pace, unchanged counted-body exposure projection 133.00736, and explicit differential/absolute wall projections <500 seconds.

### Milestone 3: Run the sole fixed-seed score
- [ ] Confirm baseline 94.32 at `67c8e98`, one idle H20, local data, frozen evaluator, exact scope, no stale `run.log`, and passing preflights; execute exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [ ] Monitor numerical/CUDA/worker health and accepted mixup/RandAugment transitions without reacting to interim accuracy; never rerun a valid completion or change padding mode/width/crop size/seed.
- [ ] Require exit 0, one finite summary, 300.0-300.1 counted seconds, total <600, 987,098 parameters, one ordered transition per policy, and unique accepted-cadence evaluations. Record realized passes: `<130` neither invalidates nor authorizes rerunning the result; primary success still enters the frontier, while a primary miss closes exact reflection but leaves the broader geometry mechanism inconclusive.

### Milestone 4: Classify boundary-quality outcome
- [ ] Classify objective improvement solely by `best_test_acc >=94.42%`; separately report `final_test_acc >=94.32%` and `final_test_loss <=0.2523` as corroboration without overriding the primary metric.
- [ ] Record best/final/loss deltas, best-final gap, steps/epochs/passes, transition lag, evaluation count, VRAM, counted/wall time, padded-window incidence, and final source audit.
- [ ] Accept into the frontier whenever the primary metric and hard task constraints pass, regardless of the experiment-specific exposure interpretation. A valid >=130-pass miss closes reflection, symmetric, replicate, and alternate padding-width/crop-size geometry; a `<130` miss closes exact reflection only, and no same-loop rescue is allowed.

## Code Changes
- **`train.py` / `make_train_transform`**: add only `padding_mode="reflect"` to the existing 32x32 crop with four-pixel padding. This changes pixels sampled outside the original image from zeros to reflected edge content while leaving padded dimensions, crop-coordinate RNG calls, flip, early RandAugment, tensor conversion, and normalization order unchanged.
- **`.autoresearch/.../experiments/032/preflight.py`**: ignored verification-only harness for exact source scope, decision-stream and pre-RandAugment pixel-confinement checks, instrumented forkserver crop/flip/RandAugment/cutoff/sampler controls, and real-loader stability. Replace `prepare.Eval` before importing either candidate or oracle, guard any `CIFAR10(train=False)` construction, keep modules in separate namespaces, and never write `run.log`.

## Configuration Changes
- Crop padding mode: torchvision default `constant` with fill 0 -> `reflect`; padding remains 4 and output remains 32x32.
- Temporal scope: reflection applies throughout all training epochs, including the hard-label/RandAugment-disabled tail. Do not claim late pixel identity to accepted; only RandAugment becomes inactive.
- Model/optimization/data decisions: unchanged `(2,2,3)`, batch 256, FP32, LR `0.2 ->0.002`, Nesterov momentum 0.9, matrix decay `5e-4`, batch-shared alpha-0.2 mixup through 65%, crop offsets, flip decisions, early N1/M5 RandAugment draws, seed 42, worker setup, budget, and evaluator.

## Execution Environment
- Method: offline local semantic and loader preflights, then one local score only on pass; no remote, network, installs, W&B, GitHub, or `gh`.
- Resources: one NVIDIA H20 for the score, local CIFAR-10, installed environment, eight persistent forkserver workers.
- Estimated runtime: preflights under 3 minutes; score about 340-355 seconds wall with a 600-second hard timeout.
- Log output: scored stdout/stderr only in root `run.log`, retained through analysis then removed.
- Tool skill: none.

## Abort Criteria
- Abort before loader timing on any scope/frozen-file/syntax failure; construction/model/optimizer/global-RNG mismatch; wrong transform order/parameters; crop/flip/private-RandAugment decision or post-call RNG mismatch; pre-RandAugment pixel difference outside a correctly flipped padding mask; mismatch to the independent NumPy reflection oracle; missing intended touched-pixel difference; invalid incidence; worker per-sample trace/sampler/target/main-RNG/cutoff error; evaluator/test access; or non-finite data.
- Abort before scoring on loader error/starvation, wrong batch count/shape, any CV >5%, or either explicit wall projection >=500. Counted-body source/shape identity fixes the pre-score exposure projection at accepted 133.00736; loader timing is used only for wall feasibility. Emit raw epochs/incidence/projections before assertions and never repeat a stable miss.
- During score stop/classify on timeout, nonzero exit, OOM/worker/resource error, non-finite loss, no output for 60 seconds, malformed/missing/duplicate summary, wrong topology, invalid/repeated transition, duplicate evaluation epoch, or total >=600. Never rerun a valid score; below-target realized exposure is reported separately rather than erased.

## Verification Protocol

### Verification Procedure

1. Query `exp-index.sh baseline`; within 10 seconds require 94.32 at `67c8e98`, so the primary threshold is 94.42.
2. Within 30 seconds query `nvidia-smi`, local CIFAR presence, branch/status, `git diff --check`, `git diff --unified=0 67c8e98 -- train.py`, `git diff --exit-code 67c8e98 -- prepare.py`, and `uv run python -m py_compile train.py .../experiments/032/preflight.py`. Require one idle H20, only tracked `train.py`, and exactly one added crop argument.
3. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/032/preflight.py --semantics`. Require independent accepted/candidate model state, construction CPU/CUDA RNG, optimizer groups/settings, all unchanged constants/source sections, transform order, crop size/padding, and 987,098 parameters.
4. On deterministic asymmetric nonzero PIL fixtures, save one pre-transform CPU RNG state and independently restore it before (a) accepted crop+flip, (b) candidate crop+flip, and (c) manual decision replay. Assert constant PIL padding and independent `numpy.pad(..., mode="reflect")` consume no torch RNG; replay the two `torch.randint` crop draws plus flip `torch.rand`, require shared `(i,j)`/flip/terminal states, and require production outputs to equal their manual oracles.
5. For >=100,000 separately seeded coordinate draws, print incidence first and require padding contact in `[0.985,0.990]` (theoretical `80/81=0.987654`). Across exhaustive `(i,j)` offsets and both flip decisions with RandAugment bypassed, flip the padding-derived mask with the image and require equality on every mapped original-image pixel, all differences confined to the mask, and at least one difference for every touching offset.
6. From independently restored main RNG states, apply accepted-like and candidate `EarlyRandAugment` wrappers to corresponding cropped images for >=64 calls. Before each production call, independently decode the chosen op index and sign from the effective private state using torchvision's fixed RandAugment draw sequence; require matching decoded decisions, main RNG restoration, byte-equal private state, and exact no-advance bypass when inactive. Pixel confinement is not asserted after active RandAugment because geometric operations may move/interpolate reflected content.
7. Build a preflight-only top-level instrumented transform/dataset that follows the production pad -> crop -> flip -> `EarlyRandAugment` sequence and returns sample index, worker id, `(i,j)`, flip bit, active flag, decoded op/sign, private pre/post-state hashes, and target alongside the image. Run fresh accepted-like and candidate forkserver arms from independently restored seed-42 states in production construction order, tear down each loader before resetting for the other arm, exhaust one active epoch, then sample the inactive next epoch. Require every per-sample trace and target to match, active/inactive image hashes to differ only as intended from reflection, shared-byte cutoff propagation, and terminal main RNG equality. Guard `prepare.Eval` before importing either module and raise on any `datasets.CIFAR10(train=False)` construction.
8. Run `timeout 240s uv run python .../experiments/032/preflight.py --loader-timing`. At fixed accepted consumer time per batch, compare accepted-like and candidate active/inactive real loaders in balanced order with one warm and three measured complete epochs per arm. Print arm order, all epoch windows, medians, CVs, weighted values, and projections first; require exactly 195 finite batches/49,920 examples per epoch, every CV <=.05, and no candidate epoch >1.10x its matched accepted median. Define `accepted_weighted=0.65*accepted_active_median+0.35*accepted_inactive_median` and candidate likewise; with `projected_epochs=133.00736*50000/49920`, require `differential=345.3+max(0,candidate_weighted-accepted_weighted)*projected_epochs <500` and `absolute=45.3+candidate_weighted*projected_epochs <500`. Report pre-score counted exposure as unchanged 133.00736 because the GPU body/shapes are source-identical.
9. Reconfirm audit and one idle H20, remove stale log, execute exactly `timeout 600s uv run train.py > run.log 2>&1` once, record PID/start, and never launch a second valid score.
10. Parse with `rg`; require one finite summary, 300.0-300.1 counted seconds, total <600, 987,098 parameters, and no traceback/OOM/non-finite/worker errors. Record `num_steps*256/50000`; `<130` does not invalidate or authorize rerunning the sole completed score. Primary success still enters the frontier; primary miss closes exact reflection only and leaves broader padding geometry inconclusive.
11. Require mixup disable exactly once at the first >=195-second step and one later RandAugment disable after iterator exhaustion with step lag `[0,195)` and no re-enable. Require unique every-fifth-epoch evaluations plus one final partial epoch.
12. Classify goal success only by `best_test_acc >=94.42%`. Independently report final-accuracy corroboration `>=94.32%` and loss corroboration `<=0.2523`; neither can overturn the primary metric. Audit final source regardless of verdict.

### Informational Metrics (Optional)
- Final summary: `run.log` - best/final accuracy, final loss, counted/total/startup seconds, epochs, steps, passes, VRAM, and parameters.
- Transitions/cadence: `run.log` - mixup/RandAugment epoch/step/time, transition lag, and unique evaluation epochs/count.
- Preflight: direct output - padding-contact incidence, pixel confinement, decision/RNG alignment, loader epoch windows/CVs, and wall/exposure projections.
- Mechanism: best/final/loss deltas from accepted 94.32/94.22/0.2523 and best-final gap.

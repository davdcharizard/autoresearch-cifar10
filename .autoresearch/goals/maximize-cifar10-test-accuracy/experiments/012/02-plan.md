# Plan EXP-012: Exact 8x8 Bottleneck Residual Refinement
- **Created**: 2026-07-24

## Milestones

### Milestone 1: Exact bottleneck implementation
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-012` from clean accepted commit `eb08811`.
- [x] Modify only project source `train.py`: add the exact pre-activation `128->64->64->128` identity bottleneck, initialize the accepted model first and isolate refinement construction/initialization in a restoring CPU RNG fork, execute it once between `layer3` and final BN, and log exact topology/count.
- [x] Compile, audit the complete source diff, require byte-identical `prepare.py`, and pass topology, initialization, identity-shortcut, hook-order, parameter, MAC, shape, backward, accepted-tensor equality, and post-construction RNG-equality assertions.

### Milestone 2: Matched production-path feasibility gate
- [x] Confirm exactly one H20 and local CIFAR files, then import final code through a fail-closed evaluator stub.
- [x] Benchmark exact accepted/no-refinement and candidate/rank-64 models with pinned transfers and complete FP32 mixup/hard-label production steps in the fixed interleaved order.
- [x] Require counts 691,674/745,434, every regime CV ratio <=0.05, aggregate retention >=0.92, and calibrated projection >=130.5 passes before scoring.

### Milestone 3: Single scored run
- [x] Remove stale `run.log`; run exactly once with `timeout 600s uv run train.py > run.log 2>&1`.
- [x] Monitor startup, finite loss, exposure, one mixup transition, cadence, and completion without changing the run from interim accuracy.
- [x] Require exit 0 and a complete final summary.

### Milestone 4: Result audit
- [x] Verify one H20, exact FP32 topology/count, 300.0-300.5 counted seconds, total <=600, one transition, accepted optimizer/schedule, and unique fifth-plus-terminal evaluation cadence.
- [x] Record realized passes against 120, 130.5, and accepted 141.9, plus best/final accuracy and loss, epochs, steps, VRAM, and best/final gap.
- [x] Accept only `best_test_acc >=94.17%`; a stable negative with >=120 passes rejects only this exact rank-64 placement and initialization.

## Code Changes
- **`train.py`**: add `BOTTLENECK_WIDTH=64`; add `PreActBottleneck` with `BN(128)-ReLU-Conv1x1(128,64)`, `BN(64)-ReLU-Conv3x3(64,64)`, and `BN(64)-ReLU-Conv1x1(64,128)`, literal identity shortcut, and no post-add activation. Extend `WideResNet` with optional strictly validated refinement width. Construct and initialize every accepted module with the unchanged `self.apply` first; only then construct and initialize refinement inside `torch.random.fork_rng(devices=[])`, restoring global CPU RNG exactly. Call it once after `layer3`; instantiate production with 64 and log it. No accepted block, width, optimizer, schedule, data, or evaluator logic changes.
- **`experiments/012/preflight.py`**: local ignored research artifact containing the exact fail-closed semantic/timing protocol; never imported by production and excluded from git/source scope. Run as `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/012/preflight.py`; preserve it with recorded stdout for reproducibility.

## Configuration Changes
- Post-stage-3 refinement: none -> fixed half-width 64 bottleneck; parameters `691,674 -> 745,434`; MACs/image `101,106,944 -> 104,514,816`.
- FP32, `[32,64,128]`, `[2,2,2]`, batch 256, LR/floor/warmup, momentum, selective decay, alpha-0.2 mixup through 65%, seed, transforms, loader, and evaluation cadence: unchanged.
- No endpoint zeroing, EMA, ratio/placement search, dense extra block, width change, augmentation, fusion, precision, LR, or adaptive fallback.

## Execution Environment
- Method: local/offline evaluator-free feasibility preflight, then only on pass one scored `timeout 600s uv run train.py > run.log 2>&1`.
- Resources: exactly one NVIDIA H20, existing environment and local CIFAR files; no network, install, remote service, GitHub, or `gh`.
- Estimated runtime: preflight under one minute; scored run about 340 seconds total.
- Log output: project-root `run.log`; record preflight stdout values in `03-execute.md`; remove `run.log` after analysis.
- Tool skill: none.

## Abort Criteria
- Do not score if semantic/topology/RNG/MAC checks, counts, CV ratio <=0.05, 0.92 retention, or 130.5-pass projection fail. Do not substitute a ratio, placement, initialization, or other finalist.
- Stop/classify scoring on timeout 124, traceback, CUDA/OOM, non-finite loss, wrong topology/count, missing H20, or total wall time >=600 seconds.
- Do not stop for low interim accuracy or retry/reroll a valid result; no bottleneck, initialization, optimizer, LR, batch, precision, or regularization rescue.

## Verification Protocol

### Verification Procedure
1. Run `exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; require baseline 94.07 at `eb08811` and threshold 94.17. Require exactly one H20, local `data/cifar-10-batches-py`, and clean accepted base.
2. Compile with `uv run python -m py_compile train.py`; require `git diff --name-only eb08811` returns only `train.py`, `git diff --quiet eb08811 -- prepare.py` exits 0, and `git diff --check` exits 0. Audit the full diff against the exact block/constructor/log scope.
3. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/012/preflight.py`. It must replace `prepare.Eval` before importing final `train.py`, and the stub must fail if evaluation is called. Reject bottleneck widths of zero, negative, float, Boolean, and wrong container/type semantics. Under identical saved/reset CPU RNG states, build accepted/no-refinement and candidate/64 models; require exact counts 691,674/745,434, byte-identical values for every accepted parameter/buffer key, and bitwise-equal post-construction CPU RNG states. This proves refinement constructor/default initialization/custom initialization do not alter accepted weights, DataLoader shuffle/worker seeds, augmentation, or later mixup randomness.
4. Assert unchanged stem/stages `[32,64,128]` and `[2,2,2]`, accepted strides/projections/shortcuts, final BN `128`, classifier `128->10`, and exact bottleneck BN/channel/kernel/stride/padding/bias values. Hooks must observe exactly one refinement call whose input is the `layer3` output and whose input/output are `[N,128,8,8]`. Zeroing only a test copy's three residual convolutions must make block output bitwise equal input; no post-add activation/module is permitted. Require 53,760 added/745,434 total parameters and 3,407,872 added/104,514,816 total MACs.
5. In the same named preflight, require finite FP32 `[256,10]` logits and a finite backward/accepted SGD step. Construct exact accepted and candidate models/optimizers under matched initialization and independent initially equal training RNG streams. Use fixed pinned host tensors and one timed function reproducing production copies, LR writes, progress/mixup branch, zero-grad, Beta/randperm, forward/loss/finite guard/backward, optimizer step, and synchronize.
6. Warm each path for 25 mixup steps. At 50% progress measure three 50-step windows in order `accepted-A,candidate-A,candidate-B,accepted-B,accepted-C,candidate-C`; repeat identically at 80% for hard labels. Restore/update each path's private RNG around each window. Record all window means, peak memory, and require correct logits/finite states.
7. For each path/regime use median window mean and CV ratio `statistics.pstdev/mean`. Compute `aggregate=0.65*mixup_median+0.35*hard_median`, `retention=accepted_aggregate/candidate_aggregate`, and `projection=141.9*retention`. Require every CV ratio <=0.05, retention >=0.92, projection >=130.5, exact counts/topology, and no OOM. The preflight may not inspect accuracy.
8. Run the sole scored command `rm -f run.log` then `timeout 600s uv run train.py > run.log 2>&1`; require exit 0 or classify from the final 50 lines with no result-conditioned alternative.
9. Require `Device: cuda`, exact rank-64 topology and 745,434 count, counted seconds `[300.0,300.5]`, total <=600, `num_steps<64000`, finite loss, exactly one switch near 195 seconds/LR 0.0612, and unique evaluations only every fifth epoch plus terminal.
10. Compute passes `num_steps*256/50000`; require at least 120 for stable mechanism interpretation and record against 130.5/141.9. Extract summary metrics and require `best_test_acc>=94.17%`. Any lower valid score is no-improvement with no rerun. Below 120 realized passes, the exact mechanism is operationally confounded but the formal verdict still follows accuracy.

### Informational Metrics (Optional)
- `run.log` final summary: peak VRAM, final accuracy/loss, training/total/startup seconds, epochs, steps, parameters.
- Derived from `run.log`: passes, exposure retention, best epoch, best/final gap, evaluation and transition counts.
- Preflight stdout: all windows/CVs, aggregate times, retention, projection, peak memory, semantic assertions.

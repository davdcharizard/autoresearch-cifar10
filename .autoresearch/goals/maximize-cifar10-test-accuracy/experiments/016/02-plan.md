# Plan EXP-016: Fixed-MAC Stage-Depth Redistribution
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Implement controlled `[1,2,3]` topology
- [x] Modify only `train.py`: expose scored depths `[1,2,3]`, but make the production constructor always build and fully initialize the accepted `[2,2,2]` graph before any topology mutation; never pass `[1,2,3]` into `_make_layer` directly.
- [x] Remove `layer1[1]`; inside `torch.random.fork_rng(devices=[])`, set only `torch.random.default_generator` to preregistered `NEW_BLOCK_INIT_SEED = 16016`, construct one `128->128` `PreActBlock`, and explicitly call the exact accepted `_weights_init` routine on it before appending as `layer3[2]`; restore global CPU/CUDA RNG exactly.
- [x] Preserve every training setting and log exact stage depths; compile with `uv run python -m py_compile train.py`.

### Milestone 2: Pass evaluator-free semantic and timing gates
- [x] Create ignored `experiments/016/preflight.py`; import `train.py` with a dummy evaluator that raises on evaluation.
- [x] Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/016/preflight.py --semantics`; require exact topology, parameter/MAC counts, common-state/RNG equality, finite forward/backward, and new-block gradients.
- [x] Audit scope with `git diff --name-only eb08811 --`, `git status --short --untracked-files=all`, `git diff --check`, and root Python-file enumeration; require only tracked `train.py` changed and no importable extras.
- [x] Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/016/preflight.py --throughput`; require all CVs <=5%, weighted retention >=97%, and projected passes >=137.6.

### Milestone 3: Run exactly once
- [x] Confirm exactly one `NVIDIA H20`, remove stale `run.log`, and run `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Require exit 0, finite complete summary, 300 counted seconds, total below 600 seconds, and one mixup transition near 195 seconds.

### Milestone 4: Verify and preserve evidence
- [x] Evaluate `best_test_acc >= 94.17`; observed 93.82, so the necessary metric condition failed; final loss was 0.2778.
- [x] Audit 35 unique evaluation epochs, exact parameter count 968,538, stage log `[1,2,3]`, 171.6992 realized passes, and the final production diff.
- [x] Record all implementation, preflight, run, and verification evidence in `03-execute.md`.

## Code Changes
- **`train.py` configuration/model construction**: introduce `STAGE_BLOCKS = (1, 2, 3)` as the scored final topology and `NEW_BLOCK_INIT_SEED = 16016`. `WideResNet` must first construct `[2,2,2]`, run its unchanged whole-model `self.apply(_weights_init)`, and only then exchange one initialized stage-1 identity block for a new stage-3 block. Within the CPU RNG fork, construction is followed by `new_block.apply(self._weights_init)` because module constructors use defaults; keep all surviving state bitwise identical and isolate the new block seed from global RNG.
- **`train.py` logging**: print widths and exact depths rather than a misleading scalar-derived WRN name. No per-step diagnostics or synchronization is added.
- The ignored preflight is not production code, never accesses test data, and may implement an accepted reference only for comparison.

## Configuration Changes
- Stage depths: `[2,2,2]` -> `[1,2,3]`; total residual blocks remain six.
- Parameters: 691,674 -> 968,538; convolution/linear MACs remain exactly 101,106,944 per image.
- Architecture widths, block implementation, optimizer, LR schedule/floor, batch size, FP32, augmentation, alpha-0.2 batch-shared mixup, 65% cutoff, hard-label tail, seed 42, workers, and evaluation cadence remain unchanged.

## Execution Environment
- Method: offline local execution; no network, remote job, package install, W&B, GitHub, or `gh`.
- Resources: one NVIDIA H20, local CIFAR-10, existing `uv`, eight persistent workers.
- Estimated runtime: preflight under 3 minutes; scored wall time about 340 seconds with a 600-second hard limit.
- Log output: scored stdout/stderr to project-root `run.log`; retain until analysis.
- Tool skill: none.

## Abort Criteria
- Abort before scoring on any scope/syntax failure; local seed other than exact integer 16016; direct `[1,2,3]` construction before accepted initialization; wrong depths/block/shortcut count; parameter count other than 968,538; MAC count other than 101,106,944; common-state or post-construction CPU/CUDA RNG mismatch; bitwise mismatch between the new block and an independently constructed-and-exactly-initialized seed-16016 oracle; non-finite/missing new-block gradients; evaluator access; or optimizer membership error.
- Abort before scoring if any timing CV exceeds 5%, weighted retention is below 97%, projected passes are below 137.6, or any path OOMs/diverges.
- During scoring, let `timeout 600s` terminate hangs. Treat nonzero exit, error/non-finite output, missing summary, wrong topology/count, duplicate evaluation epoch, or missing/multiple transition as crash/invalid. Never rerun a valid score or substitute `[2,1,3]`.

## Verification Protocol

### Verification Procedure

1. Query baseline with `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; require 94.07 at `eb08811`, hence threshold 94.17. Timeout 10 seconds.
2. Run hardware/scope/syntax commands: `nvidia-smi --query-gpu=name --format=csv,noheader`, `git diff --name-only eb08811 --`, `git status --short --untracked-files=all`, `git ls-files --others --exclude-standard`, `find . -maxdepth 1 -type f -name '*.py' -printf '%f\n'`, `git diff --check`, and `uv run python -m py_compile train.py`. Require one H20, only modified tracked `train.py`, no untracked/importable extras, and exit 0. Timeout 30 seconds.
3. Run semantic preflight command from Milestone 2. Require source/runtime seed exactly 16016; accepted-first `[2,2,2]` construction and initialization before mutation; final `[1,2,3]`, six blocks, three projection shortcuts, expected stage shapes, 968,538 parameters, and exact equal MACs. From the same saved pre-construction state, compare every named surviving parameter/buffer bitwise against an unmodified accepted reference and compare the new block bitwise against an independent seed-16016 oracle that performs constructor defaults followed by exact `_weights_init`; require post-construction CPU/CUDA RNG equality, finite logits/loss/gradients, and updated new-block weights. Timeout 120 seconds.
4. Run throughput preflight command from Milestone 2. Benchmark production mixup and hard-label steps in balanced warm windows (>=25 warmup, three >=50-step windows each), including transfer, RNG, forward/backward, SGD, and synchronization. Require weighted retention >=0.97, projected passes >=137.6, CVs <=0.05, and explicit `THROUGHPUT PASS`. Timeout 180 seconds.
5. Remove `run.log` and execute exactly `timeout 600s uv run train.py > run.log 2>&1`; require exit 0 and never rerun a valid result.
6. Parse with `rg -n '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|startup_seconds|peak_vram_mb|num_epochs|num_steps|num_params):' run.log`, `rg -n '^  eval ep' run.log`, `rg -n 'Mixup disabled' run.log`, and error-pattern `rg`. Require one summary, `best_test_acc >=94.17`, 300 counted seconds, total <600, 968,538 parameters, one transition near 195 seconds, unique evaluation epochs, and no error match. Stop verification immediately on a necessary-condition failure.
7. Run `git diff eb08811 -- train.py`; confirm only the approved topology/RNG-control/logging change, frozen evaluator, seed 42, accepted loss/schedule, and existing once-per-epoch evaluation condition.

### Informational Metrics (Optional)
- Collect peak VRAM, final accuracy/loss, training/total/startup seconds, epochs, steps, and parameter count from the final summary only if necessary conditions pass; compute passes as `num_steps * 256 / 50000`.

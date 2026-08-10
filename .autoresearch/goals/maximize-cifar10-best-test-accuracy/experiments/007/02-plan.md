# Plan EXP-007: Width-2 ResNet-20
- **Created**: 2026-08-05

## Baseline and Hypothesis

- Moving baseline: `best_test_acc = 92.30%` at commit `11f8469`; formal improvement requires at least `92.40%`.
- Intervention: widen only the accepted post-activation ResNet-20 from stage channels 16/32/64 to 32/64/128. Preserve depth, Option-A shortcuts, N1/M7 through 80%, weak hard-label tail, optimizer, schedule, seed, evaluator, and worker lifecycle.
- Hypothesis: if a repeated fresh-process benchmark confirms width 2 is operationally feasible under the fixed timer, 3.98x model capacity will outweigh its expected roughly 31% update loss and raise `best_test_acc` to at least 92.40%.
- Evidence boundary: EXP-004's plateau train-loss EMA was not persisted, so capacity under strong views is an unconfirmed premise being tested, not an established bottleneck. EXP-001's near-zero weak-recipe train loss proves clean-data memorization capacity but does not resolve the interaction with N1/M7.

## Milestones

### Milestone 1: Isolated width implementation
- [x] Create branch `autoresearch/maximize-cifar10-best-test-accuracy-007` from integration commit `11f8469`; preserve the untracked `data/` cache.
- [x] Add `WIDTH_MULTIPLIER = 2`, make `ResNet` accept a width multiplier, and derive only its stem/stage/classifier channel dimensions from 16/32/64.
- [x] Keep `BasicBlock`, raw Option-A shortcuts, initialization, and every non-model training line unchanged.
- [x] Pass compilation, Ruff, pre-commit, diff, tracked-scope, output-shape, exact parameter-count, and transition-padding checks.

### Milestone 2: Repeated H20 timing gate
- [x] Run a disposable `/tmp/exp007_width_bench.py` in a fresh process on the idle H20 using the implemented `ResNet(..., width_multiplier=1|2)` and the exact pinned-transfer/SGD timed region.
- [x] Run three paired trials with fresh model/optimizer state, alternating width order; each configuration gets 50 warmup and 200 synchronized timed steps. Record mean, median, p95, coefficient of variation, peak allocation, raw projections, ratio, and calibrated projection.
- [x] Launch no full experiment unless width 2 has exactly 1,073,962 parameters, candidate/control mean ratio at most 1.67, calibrated projection at least 23,000 steps, raw throughput at least 70 steps/s, timing CV below 5%, finite loss/gradients, and peak allocation below 2 GB. These are doom/instability gates with margin, not point-estimate accuracy gates.

### Milestone 3: Single fixed-seed run
- [x] Reconfirm baseline, exact one-H20 idleness, clean tracked scope, and absence of stale `run*.log` files after preflight.
- [x] Launch exactly once under a 600-second supervisor with all output redirected to `run.log`; record PID/start time and do not reroll or alter width after observing results.
- [x] Monitor concise tails for finite loss/progress, one 80% worker transition, CUDA/DataLoader failures, timeout, and the weak-tail trajectory without streaming the full log.

### Milestone 4: Integrity, metric, and trajectory verification
- [x] Require exit zero, one complete ten-field numeric summary, 300 seconds counted training, total runtime below 600 seconds, exactly 1,073,962 parameters, and at most one evaluation per epoch.
- [x] Require one `randaugment->base` switch near 80.0%, eight stopped workers, and actual exposure reported against the 26,500 projection and EXP-004's 38,358 steps.
- [x] Accept only `best_test_acc >= 92.40%`; regardless of verdict, persist the last strong checkpoint/loss, first weak checkpoint, all tail accuracies, last-three-evaluation slope, best/final gap, and whether the best occurs at the final epoch.

## Code Changes

- **`train.py` only**:
  - Add `WIDTH_MULTIPLIER = 2` near the other model hyperparameters.
  - Change `ResNet.__init__` to accept `width_multiplier=1` and derive:
    ```python
    c1, c2, c3 = (width_multiplier * channels for channels in (16, 32, 64))
    ```
  - Replace only hard-coded model dimensions with `c1/c2/c3`: stem `3 -> c1`, stem BN `c1`, stages `c1->c1`, `c1->c2`, `c2->c3`, and classifier `c3 -> num_classes`.
  - Instantiate the training model as `ResNet(NUM_BLOCKS, NUM_CLASSES, WIDTH_MULTIPLIER)`.

Do not edit `BasicBlock`. Its two bias-free 3x3 convolutions, BN/ReLU order, raw shortcut, stride slicing, and append-only channel padding stay exact. At width 2, transition padding must become 32 and 64 channels through the existing arithmetic; no projection shortcut is added.

The model must have exactly 1,073,962 trainable parameters:

| Component | Parameters |
|---|---:|
| Stem convolution + BN affine | 928 |
| Stage 1 convolutions + BN affine | 55,680 |
| Stage 2 convolutions + BN affine | 203,520 |
| Stage 3 convolutions + BN affine | 812,544 |
| Classifier with bias | 1,290 |
| **Total** | **1,073,962** |

No startup label change, logging addition, loss diagnostic, or unrelated refactor belongs in the tracked diff. Existing progress output already provides debiased train-loss EMA and all required timing.

## Configuration Changes

- `WIDTH_MULTIPLIER`: implicit 1 -> explicit 2.
- Stage channels: 16/32/64 -> 32/64/128.
- Parameters: 269,722 -> 1,073,962.
- Expected synchronized step: approximately 7.5 ms -> 10.9 ms in the initial diagnostic.
- Expected actual exposure: 38,358 -> approximately 26,500-28,000 steps; about 67-72 epochs and 13-14 weak-tail epochs.
- Expected peak VRAM: 330.1 MB -> approximately 599 MB.

All data, batch size 128, hard-label cross-entropy, ordinary SGD (`lr=0.1`, momentum 0.9, weight decay `1e-4`), 80% hold, `0.01 -> 1e-4` cosine tail, seed, checkpoints, and loader settings are unchanged. In particular, do not import preactivation, dropout, Nesterov, `5e-4` decay, projection shortcuts, batch scaling, AMP, compilation, or WRN's longer schedule.

## Execution Environment

- Method: local single-GPU run from the project root: `timeout --signal=TERM --kill-after=10s 600s uv run train.py > run.log 2>&1`.
- Resources: exactly one idle NVIDIA H20 with approximately 97,871 MiB; cached CIFAR-10 in preserved `data/`; no environment or dependency changes.
- Estimated runtime: 300 counted training seconds and approximately 330-380 seconds total, hard limit 600 seconds.
- Log output: full stdout/stderr only in `run.log`; concise `tail`/`grep` monitoring. Keep the log through analysis, then remove it before the next experiment.
- Tool skill: local execution only.

## Repeated Timing Preflight

The disposable script imports the implemented `ResNet` and constructs fresh width-1 and width-2 models without modifying training state. For each paired trial:

1. Reconfirm the H20 is idle; seed 42 separately for each fresh model.
2. Allocate reusable pinned host input `[128,3,32,32]` and integer targets `[128]`.
3. Use exact SGD (`lr=0.1`, momentum 0.9, weight decay `1e-4`) and hard cross-entropy.
4. Time nonblocking H2D transfer, zero-grad, forward, loss, backward, optimizer step, and terminal CUDA synchronization, matching `train.py`'s counted region.
5. Warm 50 steps and collect 200 step times. Alternate trial order `1x,2x`; `2x,1x`; `1x,2x` to reduce clock/order bias.
6. For each width calculate mean, median, p95, timing CV, raw `300 / mean` steps, examples, and peak VRAM. Use the median of the three trial means as the control/candidate time.
7. Calculate `calibrated_steps = 38_358 * control_time / candidate_time`.

Hard launch gates are all conjunctive: width-2 parameter count 1,073,962; output shape `(2,10)`; candidate/control ratio <=1.67; `calibrated_steps >= 23_000`; raw candidate rate >=70 steps/s; CV <5% for both widths; finite losses/gradients; peak width-2 allocation <2 GB. These boundaries reject a materially slower or unstable implementation while leaving substantial margin around the prior 26,563-step point estimate. A projection in the 26,000-26,500 range is valid evidence about the declared capacity/update trade and must not be rejected due to microbenchmark noise. Persist measurements in `03-execute.md`, delete the temporary script, and leave tracked code in place for execution only if every gate passes.

## Abort Criteria

- Abort before launch on any operational timing gate failure, wrong parameter/shape/padding result, non-finite diagnostic, wrong or busy GPU, tracked diff outside `train.py`, static-check failure, stale log conflict, or a moving baseline other than 92.30 at `11f8469`.
- Do not change to width 1.5 inside EXP-007 after a failed width-2 preflight. Record preflight-infeasible and route to analysis; width 1.5 requires a separately reviewed experiment.
- During the full run, terminate on the 600-second supervisor, non-zero process failure, CUDA/OOM/DataLoader error, non-finite loss/metric, 120 seconds without progress, or worker lifecycle failure. Do not abort merely for low intermediate accuracy or a late best still rising.
- Once launched, any mechanically valid step count is a result to analyze, not grounds for a retry. The fixed-time goal judges net accuracy; report exposure and do not rerun for a more favorable clock state.

## Verification Protocol

### Verification Procedure

1. Query baseline using:
   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv
   ```
   Require `baseline=92.30` and `baseline_commit=11f8469` before preflight and immediately before launch.
2. Query hardware with `nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits`; require exactly one idle H20 near 97,871 MiB.
3. Run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, `uv run pre-commit run --files train.py`, `git diff --check -- train.py`, and inspect `git diff -- train.py`. Each static command has a 120-second timeout. Require only the declared model-width changes.
4. In a fresh CPU check, instantiate width 1 and width 2. Require parameter counts 269,722 and 1,073,962, output `(2,10)` for both, nine blocks for both, and transition `pad_channels` 32/64 for width 2. Run the repeated H20 timing preflight and require every gate above.
5. Confirm `find . -maxdepth 1 -type f -name 'run*.log' -print` has no output, then launch the supervised command exactly once. Supervisor timeout is 600 seconds.
6. After exit, extract:
   ```bash
   grep -E '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|startup_seconds|peak_vram_mb|num_epochs|num_steps|num_params):' run.log
   ```
   Require exit zero and exactly one finite value for all ten fields; `300.0 <= training_seconds < 310.0`, `total_seconds < 600`, and `num_params = 1073962` after removing the separator. The 10-second counted-time slack matches the accepted EXP-004 protocol and prevents one slow boundary step from invalidating a sound run.
7. Inspect `grep -E '^augmentation_switch:|eval ep' run.log`. Require exactly one `randaugment->base` switch at 80.0-80.2%, eight workers stopped, unique evaluation epoch numbers, no more than one evaluation per epoch, and terminal evaluation matching `num_epochs`.
8. Compare `best_test_acc` numerically with 92.30. `>=92.40` is improvement; lower is no-improvement. Actual steps below the reviewed proposal's 26,000 expected-exposure boundary do not invalidate the fixed-time metric, but make update loss a primary explanatory mechanism.
9. Persist all summary metrics plus phase diagnostics before deleting the log: nearest progress loss before switch, last strong accuracy, first weak accuracy, every tail accuracy, best epoch, final accuracy/loss, last-three-evaluation least-squares or endpoint slope, and whether the best was final. Compare exposure, epochs, tail length, VRAM, and timing to EXP-004.

### Informational Metrics

- Final summary: `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params` from `run.log`.
- Capacity/exposure: actual step retention and samples versus EXP-004; actual mean counted step time `training_seconds / num_steps`; peak VRAM per additional parameter.
- Phase trajectory: last strong checkpoint/loss, first weak checkpoint, tail evaluation series, best epoch, last-three-evaluation slope, best/final gap, and final-epoch-best flag.

## Decision and Follow-Up Rules

- **Improvement**: `best_test_acc >= 92.40%` and all integrity conditions pass. Commit only `train.py`; accept width 2 as the moving fixed-time recipe. Describe the gain as the net capacity/exposure effect, not width at equal compute.
- **No improvement, advisory optimization-lag signature**: revert. If exposure is at least 26,000 and the best is final or the last-three slope is positive, treat the coarse 13-14-point tail series as evidence favoring, not proving, a separately reviewed width-1.5 experiment.
- **No improvement, advisory overfit signature**: revert. If train-loss EMA is low while test accuracy clearly plateaus or degrades, treat that as evidence favoring separately reviewed width 2 plus `5e-4`; do not claim the single run cleanly diagnosed capacity or bundle decay into EXP-007.
- **No improvement with actual steps below 26,000**: revert and identify update loss as a primary mechanism in the valid fixed-time result. Do not rerun EXP-007; use the result plus fresh timing to judge a width-1.5 proposal.
- **Preflight no-go**: record EXP-007 as invalid/preflight-infeasible with no full metric and route to analysis. Do not silently substitute width 1.5 or weight decay.
- **Crash or integrity failure**: record crash/invalid as appropriate, fix only a mechanical implementation defect within retry limits, and never change width, schedule, or seed during a retry.

## Adversarial Review Refinements

The mandatory external Claude plan review completed successfully with exit code 0 and is preserved in `02-plan-review.md`; no fallback reviewer was used. The plan adopts its substantive concerns: point-estimate-pinned launch gates were replaced with margin-bearing operational feasibility gates, the post-run analysis boundary was restored to the proposal's 26,000 steps, counted-time slack now matches EXP-004, and a null result is explicitly unable to falsify the unresolved capacity premise cleanly. Fewer once-per-epoch tail evaluations are accepted as a downward sampling bias on `best_test_acc`; evaluation cadence is not changed because the goal caps validation at once per epoch and isolation requires preserving the accepted evaluator policy.

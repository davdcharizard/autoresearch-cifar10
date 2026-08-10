# Plan EXP-006: Plateau-Only Fixed-Square Cutout
- **Created**: 2026-08-05

## Baseline and Hypothesis

- Moving baseline: `best_test_acc = 92.30%` at commit `11f8469`, queried with `exp-index.sh baseline`.
- Improvement threshold: `best_test_acc >= 92.40%`.
- Hypothesis: replacing the accepted plateau-only RandAugment operation with one post-normalization, mean-valued 16x16 Cutout patch on every crop/flip view through the existing 80% boundary will encourage part-distributed evidence, preserve at least 98.5% of EXP-004's 38,358 optimizer steps, and improve best test accuracy to at least 92.40% after the unchanged weak low-LR tail.
- Interpretation constraint: the experiment tests whether this one canonical Cutout configuration beats N1/M7 RandAugment for this model, seed, and fixed-time recipe. The literature does not establish that head-to-head result. Cutout may also buy more optimizer steps because it is cheaper than PIL RandAugment; that is a legitimate fixed-time recipe benefit but prevents attributing a gain solely to occlusion. A marginal 92.40-92.45% pass is formally successful but is not a reliable effect-size claim under a single fixed seed.

## Milestones

### Milestone 1: Isolated Cutout implementation
- [x] Create branch `autoresearch/maximize-cifar10-best-test-accuracy-006` from integration commit `11f8469` and confirm the only pre-existing untracked path is the preserved `data/` cache.
- [x] In `train.py`, remove the pre-`ToTensor` PIL RandAugment entry and append fixed-square `RandomErasing` after `Normalize`; do not perform a positional literal swap because RandomErasing requires a tensor. Preserve the weak transform and every model, optimizer, schedule, timing, seed, and evaluator choice.
- [x] Mechanically rename the phase state and switch provenance from RandAugment to Cutout in both predicates and the single switch block.
- [x] Add setup-time assertions proving the live plateau loader's dataset references the exact `cutout_train_tf` object and that its ordered stack ends in the declared `RandomErasing` with no `RandAugment`; do not sample a loader batch or consume training RNG for this check.
- [x] Pass syntax, Ruff, pre-commit, diff, transform-semantics, model-parameter, and tracked-scope checks.

### Milestone 2: Loader feasibility and lifecycle preflight
- [x] Run a disposable `/tmp/exp006_loader_bench.py` in a fresh process with the exact dataset, transform, batch size, eight persistent forkserver workers, pinning, shuffling, and drop-last settings.
- [x] Warm one Cutout epoch, time three full Cutout epochs, explicitly shut down its iterator, verify all old worker PIDs exit, build the weak loader, and measure rebuild-to-first-batch latency.
- [x] Require 390 batches per epoch, slowest Cutout epoch at least 160 batches/s, all old workers stopped, switch latency below 5 seconds, finite batches, and a projected total runtime comfortably below 600 seconds. Record results in `03-execute.md`, then delete the disposable script.

### Milestone 3: Single fixed-seed experiment
- [x] Confirm exactly one idle NVIDIA H20 with approximately 98 GB VRAM and confirm no stale `run.log` or renamed run-log variant exists.
- [x] Run exactly once under a 600-second supervisor with all stdout/stderr redirected to `run.log`; do not tee, reroll, or change the candidate after observing results.
- [x] Monitor concise log tails for progress, non-finite loss, resource failures, timeout, and the one Cutout-to-weak transition without streaming the full log.

### Milestone 4: Integrity and metric verification
- [x] Require exit code zero, one complete numeric summary, 300 seconds of counted training, total runtime below 600 seconds, and at most one evaluation per epoch.
- [x] Require exactly one `cutout->base` switch at approximately 80%, eight stopped workers, no RandAugment provenance, at least 37,783 optimizer steps, and unchanged 269,722 parameters.
- [x] Compare `best_test_acc` with 92.30%; accept only at 92.40% or higher. Record plateau/final diagnostics regardless of verdict and retain `run.log` until analysis persists all metrics.

## Code Changes

- **`train.py` only**:
  - Rename `strong_train_tf` to `cutout_train_tf` and replace `transforms.RandAugment(num_ops=1, magnitude=7)` with:
    ```python
    transforms.RandomErasing(
        p=1.0,
        scale=(0.25, 0.25),
        ratio=(1.0, 1.0),
        value=0,
        inplace=True,
    )
    ```
  - Keep the exact order `RandomCrop -> RandomHorizontalFlip -> ToTensor -> Normalize -> RandomErasing`. With the existing unit standard deviation, a post-normalization zero fill is the per-channel mean. Fixed area and aspect yield one contained 16x16 patch, masking exactly 256 of 1,024 spatial positions.
  - Initialize the first loader with `cutout_train_tf`.
  - Immediately after construction, assert `train_loader.dataset.transform is cutout_train_tf`; assert the five transform types are crop, flip, tensor conversion, normalization, and RandomErasing in that order; assert the eraser's declared probability, scale, ratio, value, and in-place setting; and assert no RandAugment remains. These setup checks prove the live training dataset is wired to the reviewed transform without drawing a sample or perturbing the fixed RNG stream.
  - Rename `randaugment_enabled` to `cutout_enabled` in its initialization, the mid-epoch boundary predicate, and the epoch-end switch predicate. Change the provenance line to `augmentation_switch: cutout->base`.
  - Retain `shutdown_train_loader`, its liveness assertion, explicit iterator clearing, garbage collection, and weak-loader reconstruction exactly. No change to the crossing-batch rule: it uses `lr=0.1`, breaks once counted time first reaches 80%, evaluates, shuts down, and the next weak batch enters the annealed tail.

No other tracked file may change. The `data/` cache is untracked infrastructure and must be preserved.

## Configuration Changes

- Plateau augmentation family: `RandAugment(num_ops=1, magnitude=7)` -> `RandomErasing(p=1.0, scale=(0.25, 0.25), ratio=(1.0, 1.0), value=0, inplace=True)`.
- Plateau duration: unchanged at `LR_HOLD_FRACTION = 0.8`.
- Weak refinement transform: unchanged crop/flip/tensor/normalize.
- Optimizer and loss: unchanged hard-label cross entropy and SGD (`lr=0.1`, momentum `0.9`, weight decay `1e-4`, no Nesterov).
- Tail schedule: unchanged immediate `0.01` entry after 80%, cosine to `1e-4`.
- Seed, batch size, workers, model, evaluator, and checkpoints: unchanged.

The canonical 16x16, `p=1.0` setting follows CIFAR Cutout's one-mask, fixed-square formulation and gives a strong, easily auditable category test. It may be too aggressive for the roughly 80-epoch plateau; this underfitting risk is accepted a priori and will not be tuned within EXP-006.

## Execution Environment

- Method: local experiment from the project root. Full run command: `timeout --signal=TERM --kill-after=10s 600s uv run train.py > run.log 2>&1`.
- Resources: exactly one idle NVIDIA H20 with approximately 97,871 MiB total VRAM; cached CIFAR-10 under `data/`; no dependency or environment changes.
- Estimated runtime: 300 seconds counted training, approximately 340-350 seconds total, hard limit 600 seconds.
- Log output: all full-run output goes only to `run.log`; concise `tail`/`grep` checks are used while monitoring. The log remains until the analyzing phase has persisted results and is then removed before the next experiment.
- Tool skill: local execution only; no remote submission skill.

## Preflight Procedure

1. Confirm branch and scope with `git branch --show-current`, `git status --short`, and `git diff -- train.py`. Reject any tracked diff outside `train.py` and preserve `data/`.
2. Run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, and `uv run pre-commit run --files train.py` with a 120-second timeout each.
3. In a fresh Python process, construct the exact transform and apply its `RandomErasing` operation to at least 32 synthetic normalized tensors initialized entirely to the nonzero value `1.0`. For every output, require exactly 256 all-channel-zero spatial positions with a 16x16 bounding box and require all remaining values to stay `1.0`; require finite output. Confirm the full transform order places `RandomErasing` after `Normalize`. The nonzero input makes the global zero count unambiguous.
4. Instantiate `ResNet(NUM_BLOCKS, NUM_CLASSES)` on CPU and require output shape `(2, 10)` for `(2, 3, 32, 32)` input and exactly 269,722 parameters.
5. Run the disposable loader diagnostic described in Milestone 2. It is a feasibility gate only, not an accuracy proxy. If any threshold fails, do not alter worker count or Cutout strength inside this experiment; treat the candidate as preflight-infeasible and analyze the failure.

## Abort Criteria

- Abort before launch if the GPU query does not show exactly one idle H20 near 97,871 MiB, a stale experiment log cannot be resolved, any tracked file besides `train.py` is modified, static checks fail, mask semantics differ from one contained 16x16 mean-valued square, or model parameters change.
- Abort before launch if Cutout's slowest warmed epoch is below 160 batches/s, any old persistent worker remains alive, loader switching takes 5 seconds or more, or projected total runtime is not comfortably below 600 seconds. Do not compensate by changing workers, mask strength, or timing.
- Terminate the full run if it reaches 600 seconds, exits non-zero, reports CUDA/OOM/DataLoader errors, produces non-finite loss or metrics, stops making training progress for 120 seconds, or violates the single-switch worker lifecycle.
- Do not abort for weak intermediate accuracy alone. Pre-register EXP-004's last strong-phase checkpoint (84.60%) and the logged debiased plateau train-loss EMA as underfitting diagnostics, but let a mechanically valid run finish because the weak tail creates the decisive accuracy transition.

## Verification Protocol

### Verification Procedure

1. Query the moving baseline before execution:
   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv
   ```
   Require `baseline=92.30` and `baseline_commit=11f8469`; otherwise stop and rebase the plan on the new accepted recipe.
2. Query hardware with:
   ```bash
   nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
   ```
   Require exactly one line naming NVIDIA H20, total memory near 97,871 MiB, and no material active allocation before launch.
3. Confirm the log precondition using `find . -maxdepth 1 -type f -name 'run*.log' -print`; require no output. Then launch the supervised command from the project root. The supervisor timeout is 600 seconds.
4. After completion, extract the summary with:
   ```bash
   grep -E '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|startup_seconds|peak_vram_mb|num_epochs|num_steps|num_params):' run.log
   ```
   Require exit code zero and exactly one finite numeric value for all ten fields. Require `300.0 <= training_seconds < 301.0`, `total_seconds < 600`, `num_steps >= 37783`, and `num_params = 269722` after removing the thousands separator.
5. Inspect lifecycle and evaluation provenance with `grep -E '^augmentation_switch:|eval ep' run.log`. Require exactly one `augmentation_switch: cutout->base`, progress between 80.0% and 80.2%, `workers_stopped: 8`, no `randaugment->base` line, unique evaluation epoch numbers, and no more than one evaluation per epoch.
6. Compare the primary metric numerically with the moving baseline. `best_test_acc >= 92.40` is an improvement; any lower value is no-improvement. The run must still be recorded if valid. A gain accompanied by fewer than 37,783 steps fails the predeclared throughput-equivalence hypothesis and must be analyzed as confounded rather than accepted as a reusable recipe. A run above 38,933 steps (more than 1.5% over EXP-004) remains valid and may be accepted under the fixed-time goal, but its gain must be described as the combined statistical and computational benefit of the Cutout recipe, not evidence that occlusion alone beats RandAugment.
7. Record the final strong-phase checkpoint, the nearest logged debiased train-loss EMA before the switch, immediate post-switch accuracy, best/final accuracy and loss, steps, epochs, timing, VRAM, parameter count, switch provenance, and evaluation count in `04-analysis.md`. Compare plateau loss/checkpoint to EXP-004 to diagnose whether full-strength Cutout underfit; do not use that diagnosis to rerun EXP-006.

### Informational Metrics (Optional)

- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params`: the ten-field final `run.log` summary.
- Plateau diagnostics: last `step ... loss:` line and `eval ep` line before `augmentation_switch: cutout->base`.
- Transition response: first `eval ep` line after the Cutout-to-weak switch.
- Evaluation density and uniqueness: all `eval ep` records in `run.log`, grouped by epoch number.

## Decision Rules

- **Improvement**: all integrity and throughput conditions pass and `best_test_acc >= 92.40%`. Commit only `train.py` and merge the experiment branch into the integration branch.
- **No improvement**: integrity and throughput pass but `best_test_acc < 92.40%`. Revert to `11f8469` and conclude only that canonical fixed-square Cutout does not beat N1/M7 under this exact protocol.
- **Invalid/crash**: preflight infeasibility, lifecycle violation, throughput below 37,783 steps, malformed/non-finite summary, non-zero exit, or timeout. Do not infer a statistical verdict and do not introduce an unplanned fallback implementation.

## Adversarial Review Refinements

The mandatory external Claude plan review completed successfully with exit code 0 and is preserved in `02-plan-review.md`; no fallback reviewer was used. The plan adopts all four actionable concerns: live-loader transform identity/type assertions close the wiring-verification hole without consuming RNG, extra optimizer exposure is explicitly treated as part of fixed-time performance rather than pure mechanism evidence, the PIL-to-tensor positional move is explicit, and mask tests use guaranteed-nonzero synthetic inputs.

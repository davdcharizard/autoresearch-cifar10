# Plan EXP-012: Canonical Full Preactivation
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement the reviewed full-preactivation architecture
- [x] Create experiment branch `autoresearch/maximize-cifar10-best-test-accuracy-012` from accepted integration commit `7c1e7d8` and modify only `train.py`.
- [x] Reorder all nine residual units to `BN-ReLU-Conv-BN-ReLU-Conv-add`, remove every post-add ReLU, preserve raw `x` shortcuts for ordinary units, and use the preactivated tensor only for the first and two transition shortcuts.
- [x] Remove the stem BN/ReLU, add final `BatchNorm2d(128)` plus ReLU before pooling, and retain all shared convolution/classifier shapes and Kaiming initialization.
- [x] Pass compilation, Ruff, formatting, pre-commit, exact-scope, parameter-count, RNG-alignment, shortcut-boundary, activation-placement, gradient, target-format, and evaluator-cadence checks.

### Milestone 2: Pass fixed-time feasibility gates
- [x] On one idle H20, run five alternating paired control/candidate training trials with fresh model/optimizer state, 100 warmups and 500 synchronized timed steps per trial.
- [x] Require stable timing, candidate/control median trial-mean ratio <=1.03, projected exposure >=26,091 steps, and conservative total runtime <540 seconds; record memory without imposing a non-goal threshold short of OOM.
- [x] Benchmark inference separately and require stable trials and no material evaluator-cost increase that would violate the 600-second total limit.

### Milestone 3: Execute and verify one fixed-seed run
- [x] Confirm one idle 97,871 MiB H20 and no stale completed log, then run exactly once under a 600-second timeout with stdout/stderr redirected exclusively to `run.log`.
- [x] Verify the numeric summary, 300-second counted budget, total runtime, parameter count, single augmentation switch, stopped workers, CutMix provenance, and unique evaluation epochs.
- [x] Compare `best_test_acc` with 94.15%; improvement requires at least 94.25%. Record trajectory diagnostics without using them as retry or acceptance gates.

## Code Changes
- **`train.py` / `BasicBlock` construction**: change `bn1` from `BatchNorm2d(out_channels)` to `BatchNorm2d(in_channels)` while keeping `conv1` as `in_channels -> out_channels` with its existing stride; keep `bn2` and `conv2` at `out_channels`. Add an explicit first-unit flag so the shortcut-source exception cannot be inferred ambiguously from shape alone.
- **`train.py` / `BasicBlock.forward`**: compute `pre = ReLU(bn1(x))`, then `residual = conv1(pre)` and `residual = conv2(ReLU(bn2(residual)))`. For the first network unit and the two dimension-changing units set the shortcut source to `pre`; otherwise use raw `x`. Apply the existing Option-A `[:, :, ::stride, ::stride]` slice and high-channel zero pad only to transition shortcut sources. Return `residual + shortcut` directly with no post-add activation.
- **`train.py` / `ResNet`**: make the stem only `Conv2d(3, 32, 3, padding=1, bias=False)`, mark only the first layer-1 block as the special first unit, add `BatchNorm2d(128)` followed by ReLU after layer 3, then preserve adaptive pooling and the classifier. Pin shared randomized-module construction order exactly to control: stem conv, each block's conv1 then conv2 in layer order, then classifier; BN construction and boolean flags draw no RNG.
- No training-loop, data, optimizer, schedule, timer, evaluator, seed, logging, worker-lifecycle, or summary edits are permitted.

## Configuration Changes
- Residual layout: `Conv-BN-ReLU-Conv-BN-add-ReLU` -> `BN-ReLU-Conv-BN-ReLU-Conv-add` (paper-faithful full preactivation and unrectified residual sums).
- Stem/final boundary: stem `Conv-BN-ReLU`, no terminal normalization -> stem `Conv`, terminal `BN-ReLU` before pooling.
- Shortcut boundary rule: raw identity for six ordinary non-first same-shape blocks; preactivated shortcut for the first network block and both stride-2 Option-A transition blocks.
- Unchanged: 1,073,962 parameters; 19 convolutions, 19 BN calls, and 19 ReLU calls; width 2; p=0.5 alpha-1 plateau CutMix; N1/M7 through 80%; weak hard-label tail; batch 128; single all-parameter SGD group with momentum 0.9 and decay `1e-4`; LR schedule; seed 42; FP32; timer; workers; and evaluator cadence.
- Deliberately excluded: zero-gamma initialization. In canonical preactivation, zeroing the BN immediately before ReLU/conv2 would make the residual branch dead because ReLU's derivative at zero is zero.
- Experiment-level hypothesis: the complete canonical full-preactivation package, including reordered residual units, first/transition shortcut boundary semantics, unrectified sums, and terminal feature conditioning, can reach at least 94.25%. The run cannot attribute an outcome to any one component of that package.
- Evidence strength: a 94.25-94.35 result formally passes the fixed user-defined gate but is weak causal evidence at the one-seed noise floor. Shared initialization/RNG alignment improves pairing but does not remove data-order or CUDA variance; no variance-estimation reruns are allowed.

## Adversarial Review Response
- Mandatory Claude plan review completed successfully and is preserved in `02-plan-review.md`; no fallback reviewer was used.
- Accepted concern 1: explicitly classify a narrow pass as weak single-run causal evidence. The +0.10 gate and no-reroll policy are hard goal constraints, so replication cannot be added to EXP-012.
- Accepted concern 2: remove the 650 MB/64 MB hard abort thresholds. Peak allocation remains an informational metric; only OOM blocks or invalidates execution on the 98 GB H20.
- Accepted concern 3: state the hypothesis and analysis at the full-preactivation-package level. Shortcut transport, terminal conditioning, and residual ordering are not separately identified.
- Accepted concern 4: distinguish preflight projection from realized exposure. The completed run remains valid under the goal protocol regardless of realized steps; falling below 26,091 weakens mechanism attribution and is reported, never used to justify a rerun.
- Concern 5 is addressed without demotion: the code specification now pins randomized-module construction order. BN construction and flags do not consume RNG, so bitwise shared-weight and post-construction CPU-RNG alignment is an achievable correctness gate rather than a reason to alter the candidate.

## Execution Environment
- Method: local `timeout 600s uv run train.py > run.log 2>&1` after every correctness and feasibility gate passes.
- Resources: exactly one idle NVIDIA H20 with approximately 97,871 MiB; existing eight DataLoader workers; no package or dependency changes.
- Estimated runtime: 325-370 seconds total; exactly 300 counted training seconds; projected 26,091-27,300 optimizer steps.
- Log output: full stdout/stderr only in project-root `run.log`; monitoring uses bounded `tail`/targeted summary patterns and process state, never `tee` or streamed full output.
- Tool skill: none; local execution.

## Abort Criteria
- Do not launch if the tracked diff includes anything except `train.py`, shared Conv/Linear initialization or global RNG state is misaligned with `7c1e7d8`, parameter count differs from 1,073,962, or any architecture/gradient/target correctness assertion fails.
- Do not launch if paired timing has CV >2% for either model, candidate/control median trial-mean ratio >1.03, projected exposure <26,091 steps, non-finite loss/gradients, OOM, or conservative projected total runtime >=540 seconds.
- During execution stop on wrong or busy GPU, traceback, OOM, non-finite loss, malformed target format, failed worker shutdown, missing progress for 90 seconds while the process remains alive, or any process exceeding 600 seconds.
- Do not stop, tune, or rerun because an intermediate checkpoint is low. The fixed tail must finish; one valid seed-42 run determines the verdict.
- No rescue changes to batch size, compilation, precision, shortcut semantics, initialization, schedule, augmentation, evaluator, or seed after timing or accuracy observations.

## Verification Protocol

### Verification Procedure
1. Query the moving baseline with `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv`; require `baseline=94.15` and `baseline_commit=7c1e7d8`, giving a 94.25 success threshold.
2. Require `git diff --name-only 7c1e7d8` to print only `train.py`; run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, `uv run ruff format --check train.py`, `uv run pre-commit run --all-files`, and `git diff --check`. Use a targeted diff to confirm CutMix, transforms, loaders, optimizer, LR/timing logic, worker lifecycle, evaluator call, seed, and summary are byte-identical.
3. In a disposable fresh process construct accepted and candidate models from cloned seed-42 CPU RNG states. Require every same-named Conv/Linear tensor bitwise equal, global CPU RNG states byte-identical after construction, exactly 1,073,962 trainable parameters, nine blocks, first-BN feature widths equal each block's input width, second-BN widths equal output width, all BN scales one/biases zero, and logits shape `[128,10]` with finite values.
4. Zero residual convolution weights and use eval-mode seeded tensors to prove shortcut semantics exactly: layer1 block1 output equals its `ReLU(BN(x))`; six ordinary same-shape block outputs equal raw `x`; layer2/3 transition outputs equal the preactivated tensor sliced at `::2` and padded with 32/64 high-side zero channels. Require no post-add ReLU and one final network BN-ReLU immediately before pooling.
5. Run ordinary hard-target `[128]` and soft probability-target `[128,10]` forward/backward tests. Require finite loss/output and finite nonzero gradients for both convolutions and both BN affine tensors in every block. Confirm all parameters belong once to the unchanged single SGD group with decay `1e-4`.
6. On an idle H20 benchmark commit `7c1e7d8` control and candidate in fresh alternating processes. For each of five paired trials, use the same pinned `[128,3,32,32]` batch and alternating hard/soft targets, 100 warmups, then 500 timed steps covering nonblocking H2D, zero-grad, forward, CE, backward, SGD, and synchronization. Require per-model trial-mean CV <=2%, candidate/control median trial-mean ratio <=1.03, and `floor(26898 * control_mean / candidate_mean) >=26091`. Record control/candidate peak allocation; only OOM is a memory veto.
7. Benchmark eval-mode inference in the same alternating/fresh-state discipline. Combine the measured evaluator delta with EXP-010's 330.7-second total as a conservative projection and require <540 seconds. Preserve training and inference timing summaries in `03-execute.md`; timing is a feasibility gate, not an experiment result.
8. Confirm exactly one idle H20 using `nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader`, require only the preserved untracked `data/`, and ensure no completed `run.log` or renamed run-log variant remains. Execute `timeout 600s uv run train.py > run.log 2>&1`; exit 0 is required, 124 is timeout failure, and any other non-zero status is a crash.
9. Parse the ten final summary fields and targeted trajectory lines. Require `training_seconds` approximately 300.0, `total_seconds < 600`, `num_params = 1,073,962`, one `randaugment+cutmix->base` switch near 80%, eight stopped workers, strong mixed fraction 45-55%, and no weak-phase soft-target assertion failure. Parse evaluation epochs and require every epoch unique, enforcing no more than one evaluation per epoch.
10. Parse `best_test_acc` and require >=94.25 for improvement; a complete lower result is valid no-improvement and cannot be rerun. Compare actual steps with EXP-010's 26,898 and the 26,091 preflight projection floor. If realized steps fall below 26,091, keep the formal metric verdict but mark the representation comparison exposure-confounded and do not rerun. Record the 80% checkpoint, first weak checkpoint versus 93.16%, final loss versus 0.1934, endpoint slope, best/final gap, and VRAM as diagnostics that cannot override the primary gate. Report a 94.25-94.35 pass as weak evidence at the single-seed noise floor.

### Informational Metrics (Optional)
- Final summary metrics: targeted final lines in `run.log` for final accuracy/loss, training/startup/total seconds, peak VRAM, epochs, steps, and parameters.
- Trajectory: evaluation and augmentation-switch lines for the 80% checkpoint, first weak checkpoint, best epoch, final slope, and best/final gap.
- Feasibility: paired training/inference timing distributions, projected/actual step retention, parameter count, and peak allocation preserved in `03-execute.md`.

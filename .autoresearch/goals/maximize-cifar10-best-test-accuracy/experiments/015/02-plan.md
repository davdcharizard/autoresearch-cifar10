# Plan EXP-015: Same-Width Residual Identity Initialization
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement the selective identity initialization
- [x] Create `autoresearch/maximize-cifar10-best-test-accuracy-015` from integration commit `7c1e7d8`; modify only `train.py`.
- [x] Add an explicit `zero_init_residual` block flag and apply it to exactly the six non-entry stride-1 equal-width blocks; leave all three stage-entry gammas one.
- [x] Pass syntax, Ruff, formatting, pre-commit, scope, state/RNG, identity/transition, target-format, schedule, evaluator-cadence, and parameter-membership checks.

### Milestone 2: Prove first-update safety and compute neutrality
- [x] On fresh hard and probability-target pairs, prove bounded nonzero first gamma updates, no replay collapse, active padded transitions, and recruitment of every gated residual convolution on backward two.
- [x] Materialize one identical 64-step strong-view hard/soft stream; require finite control/candidate trajectories and candidate terminal loss EMA <=1.5x control.
- [x] Run five alternating fresh-process H20 training and inference pairs; require >=99% projected exposure, <=1.01 training/inference ratios, stable p95/CV, finite values, and unchanged memory/runtime class.

### Milestone 3: Execute and verify the one seed-42 run
- [x] Confirm one idle H20, no stale run-log variant, exact scope, and baseline 94.15; launch once under the 600-second watchdog with output only in `run.log`.
- [x] Verify exit zero, the complete summary, 300-second counted budget, wall limit, actual exposure, one switch, eight stopped workers, CutMix/target provenance, and unique evaluation epochs.
- [x] Require `best_test_acc >=94.25%` for improvement; record strong/weak fit, final NLL, exposure/evaluation attribution, and best/final gap without adaptive tuning or rerun.

## Code Changes

- **`train.py` / `BasicBlock.__init__`**: add literal keyword-only argument `*, zero_init_residual=False`. After constructing `self.bn2`, call `init.zeros_(self.bn2.weight)` only when the flag is true. `zeros_` consumes no RNG, adds no state, and does not change any forward operation.
- **`train.py` / `ResNet._make_layer`**: enumerate the three blocks in each stage and pass `zero_init_residual=True` exactly when `block_index > 0`, stride is one, and current/output channels are equal. This selects `layer{1,2,3}.{1,2}.bn2.weight` and deliberately excludes every `.0` stage-entry block.
- No production logging, optimizer, loss, data, augmentation, CutMix, batch, LR/decay, model graph, shortcut, timer, evaluator, worker, seed, or dependency change is permitted.

## Configuration Changes

- Selected within-stage final BN scales: `1.0 -> 0.0` for exactly 448 existing scalars: 32 + 32 + 64 + 64 + 128 + 128.
- Stage-entry final BN scales: unchanged at one for `layer1.0`, `layer2.0`, and `layer3.0`; the two padded Option-A transitions must remain active from backward one.
- Parameters/state/RNG: unchanged 1,073,962 parameters and state keys/shapes; state differs from control only in the six declared gamma tensors; CPU/CUDA post-construction RNG is bitwise aligned.
- Complete EXP-010 recipe remains fixed: postactivation width 2; batch 128; N1/M7 plus p=0.5 alpha-1 CutMix through 80%; hard weak tail; SGD momentum 0.9 and all-parameter decay `1e-4`; LR 0.1 then 0.01 cosine to `1e-4`; seed 42.
- Deliberately excluded: zeroing `layer1.0` or either padded transition, small positive gamma, gamma-specific LR/decay, warmup, preactivation, projection shortcuts, pooling, EMA, BF16, or any rescue after preflight/accuracy observation.

## Adversarial Review Response

- Mandatory external Claude plan review completed successfully in `02-plan-review.md`; no fallback reviewer was used.
- Production gamma provenance was removed because the selected proposal permits only initialization plumbing. First/second-step recruitment remains proved in disposable preflight, while the full run logs only the accepted schema.
- The launch pins the sole confirmed idle H20 at visible index 0. A valid run's formal verdict follows the goal's accuracy/completion/time conditions. Fewer than 26,629 steps or fewer than 18 evaluations weakens compute-neutral comparability but does not erase a genuine pass; more than 19 evaluations is invalid because it grants extra max-metric opportunities.
- Safety thresholds are deliberately loose catastrophic-collapse gates, not tuned quality gates: EXP-014 jumped to 1.221 relative branch weight, 9.92x replay loss, and 100% one-class predictions after one update. EXP-015 limits max gamma to 0.25, replay loss to 2x, class concentration below 95%, and sustained 64-step loss to 1.5x control.
- A 94.25-94.35 result is a formal but low-confidence single-seed improvement. Preflight recruitment and switch fit are mechanism corroboration; they do not override the user's primary verdict rule.

## Execution Environment

- Method: local `CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1`, exactly once after every preflight gate passes and index 0 is confirmed as the sole idle H20.
- Resources: exactly one idle NVIDIA H20 with approximately 97,871 MiB; existing eight persistent DataLoader workers; frozen installed dependencies.
- Estimated runtime: 325-360 seconds total; exactly 300 counted training seconds; projected 26,629-27,170 updates.
- Log output: complete stdout/stderr only in project-root `run.log`; monitor bounded targeted progress/error/summary lines, never `tee` or full output.
- Tool skill: none; local execution.

## Abort Criteria

- Do not launch if scope extends beyond `train.py`, selected gamma names/count/values differ, any `.0`/padded block is zero, state/RNG/parameter membership differs unexpectedly, identity/transition semantics fail, or hard/soft targets are malformed/nonfinite.
- Do not launch if any first gamma gradient is zero/nonfinite, any gated convolution has a first-step data gradient, post-step `max(abs(gamma)) >0.25`, replay loss exceeds 2x candidate pre-update or aligned control replay, candidate class concentration exceeds 95% without the control doing so, any gated convolution fails recruitment on backward two, or the 64-step terminal loss EMA exceeds 1.5x control. These loose gates reject only EXP-014-class immediate or sustained collapse; they are not accuracy tuning.
- Do not launch if training ratio >1.01, projected steps <26,629, p95 ratio >1.02, either trial-mean CV >2%, inference ratio >1.01, candidate peak >=650 MiB or >16 MiB over control, or projected total >=540 seconds.
- During the full run stop on wrong/busy GPU, traceback, OOM, nonfinite loss/gradient signal, malformed weak target, failed worker shutdown, no progress for 90 seconds while alive, or timeout at 600 seconds.
- Do not stop for a low finite checkpoint. A switch below 87.08% diagnoses short-horizon underfit but the weak tail must finish. Run only seed 42 once; never alter the zeroed block set, gamma value, LR, or add a fallback after any observation.

## Verification Protocol

### Verification Procedure

1. Query the moving baseline with `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv`; require 94.15 at `7c1e7d8`, hence 94.25 for success.
2. Require `git diff --name-only 7c1e7d8` to print only `train.py`. Run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, `uv run ruff format --check train.py`, `uv run pre-commit run --all-files`, and `git diff --check`; every command must exit zero. Confirm one static `evaluator.evaluate` call and unchanged schedule/data/timer/optimizer text outside the reviewed initialization plumbing.
3. In a disposable seed-42 construction, require exactly nine blocks, six zero gamma tensors named `layer{1,2,3}.{1,2}.bn2.weight`, three all-one `.0` gammas, and no zero padded transition. Require each selected block to be stride one/equal-width/no-pad, unchanged 1,073,962 parameters/state keys/shapes/optimizer membership, bitwise equal nonselected state, and identical post-construction CPU/CUDA RNG.
4. Prove initial train/eval output equals input exactly for each selected block on nonnegative inputs; prove the three entry blocks match control, both padded high-channel halves are nonzero/trainable, exactly 448 gamma scalars are selected by construction, and hard `[128]` plus probability `[128,10]` CE paths are finite.
5. On fresh identical hard and soft batches, require all six gamma gradient norms finite/nonzero and all selected `conv1`, `bn1`, and `conv2` loss-gradient norms exactly zero before step one. After ordinary SGD require every gamma nonzero and max absolute <=0.25. Replay the batch and enforce both 2x loss bounds and the 95% class-concentration bound; on backward two require finite nonzero `conv1`/`conv2` gradients in all six blocks.
6. Materialize 64 identical real strong-view batches once and train aligned control/candidate models separately over the same alternating hard/probability sequence. Require finite state throughout, all selected gammas recruited, no one-class collapse, and candidate terminal loss EMA <=1.5x control.
7. On one idle H20, run five alternating fresh-process control/candidate training pairs with 100 warmups and 500 synchronized steps including H2D, forward, CE, backward, SGD, and synchronize. Require ratio <=1.01, projected steps >=26,629, CV <=2%, p95 <=1.02x, finite values, candidate peak <650 MiB and delta <=16 MiB. Run five separate eval-mode inference pairs with the same warmup/measurement counts; require ratio <=1.01, CV <=2%, and conservative total <540 seconds. Confirm loader length 390 and project 18-19 unique evaluation epochs, never >19.
8. Confirm visible index 0 is the only idle H20, only preserved untracked `data/`, and no completed `run.log` variant. Execute `CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1`; require exit zero and never retry a valid run.
9. Parse the ten standard summary fields. Require 300.0 counted seconds, total <600, 1,073,962 parameters, one switch near 80%, eight workers stopped, 45-55% strong CutMix, hard weak targets, and unique evaluation epochs. Expect at least 26,629 steps and 18-19 evaluations; a lower count weakens attribution but is not itself a goal failure. More than 19 evaluations is invalid because it adds max-metric opportunities.
10. Parse `best_test_acc`; >=94.25 with completion/time integrity is formal improvement and lower is valid no-improvement. Compare switch accuracy with 89.73% and the 87.08 underfit marker, first weak with 93.16%, final NLL with 0.1934, best/final gap, and steps with 26,898. Bounded preflight recruitment and healthy fit support the mechanism, but diagnostics cannot override the primary rule; a bare 94.25-94.35 pass is formal low-confidence evidence.

### Informational Metrics (Optional)

- Final summary: `rg '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|startup_seconds|peak_vram_mb|num_epochs|num_steps|num_params)' run.log` after successful completion.
- Trajectory/lifecycle: `tr '\r' '\n' < run.log | rg 'eval ep|augmentation_switch'` for switch, first weak, terminal slope, best/final gap, worker shutdown, and CutMix rate.
- Preflight: first gamma/conv gradient norms, post-step max gamma, replay loss/concentration, second-step recruitment, 64-step relative fit, paired training/inference distributions, projected exposure, and memory.

# Plan EXP-014: Zero-Initialized Average-Max Readout
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement an exact-baseline-floor max readout
- [x] Create branch `autoresearch/maximize-cifar10-best-test-accuracy-014` from `7c1e7d8` and modify only `train.py`.
- [x] Preserve the accepted average classifier and add a bias-free 128-to-10 `max_fc` after global initialization, constructed inside a CPU RNG fork and initialized exactly zero.
- [x] Pool both average and maximum final features and sum `fc(avg) + max_fc(max)`, which is algebraically a zero-max-half concatenated classifier.
- [x] Pass compilation, Ruff, formatting, pre-commit, exact-scope, parameter/RNG/logit/gradient identity, max-path trainability, target-format, and evaluator-cadence checks.

### Milestone 2: Pass fixed-time feasibility gates
- [x] Run five alternating fresh-process control/candidate H20 training pairs with 100 warmups and 500 timed hard/soft steps per trial.
- [x] Require stable trials, candidate/control ratio <=1.025, projected exposure >=26,242 steps, candidate p95 <=1.07x control, finite gradients, and modest memory.
- [x] Benchmark inference separately, verify unchanged batch-128 epoch/evaluation count, and require conservative total runtime below 540 seconds.

### Milestone 3: Execute and verify one fixed-seed run
- [x] Confirm one idle H20 and no stale log, then run once under 600 seconds with stdout/stderr exclusively in `run.log`.
- [x] Verify summary, 300-second budget, wall limit, parameter count, max-path provenance, one augmentation switch, stopped workers, CutMix rate, and unique evaluation epochs.
- [x] Require `best_test_acc >=94.25%` for improvement over 94.15%; record switch/weak-tail/max-path diagnostics without adaptive tuning or retry.

## Code Changes
- **`train.py` / `ResNet.__init__`**: retain construction of `self.fc = nn.Linear(128, 10)` and the existing `self.apply(self._weights_init)` byte-for-byte. After that apply pass, enter `torch.random.fork_rng(devices=[])`, construct `self.max_fc = nn.Linear(128, 10, bias=False)`, and zero its weight. Default constructor draws are restored by the fork; zeroing consumes no RNG. The original classifier tensor/identity/init remains accepted and the new branch contributes exactly zero at initialization.
- **`train.py` / `ResNet.forward`**: after `layer3`, flatten global average and adaptive maximum features separately and return `self.fc(avg_features) + self.max_fc(max_features)`. This equals a 256-to-10 concatenated classifier whose average half is the accepted classifier, max half is zero, and only the average half owns the accepted bias.
- **`train.py` / provenance logging**: at each existing evaluation and in the final summary, report `max_readout_ratio = ||max_fc.weight|| / ||fc.weight||`. This adds no evaluation and runs outside the counted training step. It makes branch engagement visible without inventing a post-hoc accuracy or effect-size threshold.
- No model block, optimizer, data, CutMix, batch, LR, decay, schedule, timer, evaluator cadence, worker lifecycle, or seed change is permitted.

## Configuration Changes
- Final aggregation/readout: global average -> learned class-specific average plus max logits.
- New parameters: one bias-free `Linear(128,10)` with 1,280 zero-initialized weights; total parameters 1,075,242.
- Initialization: accepted average path exactly preserved; max branch exactly zero; global CPU/CUDA RNG after construction identical to control.
- First backward: candidate logits and all accepted-path gradients must equal control; `max_fc.weight.grad` must be finite and nonzero, while zero max weights prevent gradient feedback into max features until the first optimizer update.
- Unchanged: complete EXP-010 p=0.5 CutMix/N1-M7 recipe, postactivation width-2 blocks, Option-A shortcuts, batch 128, LR/optimizer/decay, 80% transition, seed 42, 390 batches/epoch, and accepted 19-evaluation behavior.
- Deliberately excluded: fixed mixing coefficient, learned scalar gate, GeM, classifier concatenation in a single reinitialized layer, max-branch bias, dropout, normalization/rescaling, partial freezing, or post-observation max-path control.

## Adversarial Review Response
- Mandatory Claude plan review completed successfully and is preserved in `02-plan-review.md`; no fallback reviewer was used.
- Accepted concerns 1-2: log max/average weight-norm ratio at every production evaluation and final summary so analysis can distinguish an engaged branch from a near no-op. No arbitrary minimum ratio is a formal metric gate because no evidence supports choosing one; the preflight nonzero-gradient/update gate remains mandatory.
- Accepted concern 3: always report final accuracy and best/final gap. A bare pass or isolated spike is a formal user-defined improvement but weak causal evidence, especially if max-path ratio is tiny.
- Accepted concern 4: require 18-19 unique evaluations rather than exactly 19. Batch/epoch structure is unchanged; a slower candidate may conservatively lose one look, but it may never exceed EXP-010's 19.
- Accepted concern 5: tighten the paired mean ratio from 1.03 to 1.025 and projected exposure from 26,091 to 26,242 steps, requiring measurable headroom inside the original 97% floor.

## Execution Environment
- Method: local `timeout 600s uv run train.py > run.log 2>&1` after every structural and timing gate passes.
- Resources: exactly one idle NVIDIA H20 with approximately 97,871 MiB; existing eight DataLoader workers; frozen dependencies.
- Estimated runtime: 330-375 seconds total; exactly 300 counted training seconds; projected at least 26,242 updates.
- Log output: full stdout/stderr only in project-root `run.log`; monitor bounded targeted lines/process state, never `tee` or full output.
- Tool skill: none; local execution.

## Abort Criteria
- Do not launch if any accepted state tensor or post-construction RNG differs, initial logits/accepted gradients are not exact, max weight is nonzero at construction, max gradient is zero/nonfinite, parameter count differs from 1,075,242, or the tracked diff extends beyond the reviewed readout.
- Do not launch if timing CV >2%, training ratio >1.025, projected steps <26,242, p95 ratio >1.07, inference ratio >1.05, OOM/nonfinite values occur, or projected total >=540 seconds.
- During execution stop on wrong/busy GPU, traceback, OOM, non-finite loss, malformed targets, failed worker shutdown, no progress for 90 seconds while alive, or timeout at 600 seconds.
- Do not stop for a low finite checkpoint. A switch below 87.08% diagnoses that learned peak readout harmed strong fit; finish the weak tail.
- Exactly one valid seed-42 run. Do not freeze/rescale the max branch, change its initialization, alter pooling, add regularization, or reroll after any observation.

## Verification Protocol

### Verification Procedure
1. Query baseline via `exp-index.sh baseline`; require 94.15 at `7c1e7d8`, hence a 94.25 success threshold.
2. Require `git diff --name-only 7c1e7d8` to print only `train.py`; run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, `uv run ruff format --check train.py`, `uv run pre-commit run --all-files`, and `git diff --check`. Confirm all non-readout mechanics and the single evaluator call are unchanged.
3. In a disposable paired construction from seed 42, require every accepted state tensor bitwise equal, identical post-construction CPU/CUDA RNG, original `fc` object shape/value, new zero `[10,128]` max weight with no bias, and 1,075,242 total parameters. Confirm every parameter appears once in the unchanged single SGD group with momentum 0.9/decay `1e-4`.
4. On identical seeded hard and probability-target batches, require initial candidate/control logits bitwise equal, equal losses, and bitwise-equal gradients for every accepted parameter after the first backward. Require finite nonzero `max_fc.weight.grad`; after one identical optimizer step require max weights become nonzero, and after the second backward require finite nonzero gradient reaches at least one non-maximum final-map location through the unchanged average path and the selected max locations through the learned max path.
5. Prove constant-map semantics: candidate/control logits remain equal initially for arbitrary spatially constant final features; adaptive-max indices are deterministic; helper construction/forward consumes no RNG. Exercise hard `[128]` and probability `[128,10]` CE, finite logits/loss/gradients, and exact batch/target shapes.
6. On one idle H20 run five alternating fresh-process control/candidate training pairs. Recreate state each trial, warm 100 steps, then time 500 synchronized steps including H2D, forward, CE, backward, SGD, and synchronize. Require trial-mean CV <=2%, median ratio <=1.025, `floor(26898 * control_mean / candidate_mean) >=26242`, candidate p95 <=1.07x control, finite values, and candidate peak allocation <620 MB.
7. Run five alternating eval-mode inference pairs with 100 warmups/500 forwards. Require CV <=2%, candidate/control mean <=1.05, and conservative EXP-010 total plus incremental cost across 19 evaluator passes <540 seconds. Confirm loader length remains 390 and a synthetic/full trace projects 18-19 unique evaluation epochs, never more than EXP-010's 19.
8. Confirm exactly one idle H20, only preserved untracked `data/`, and no completed `run.log` variant. Execute `timeout 600s uv run train.py > run.log 2>&1`; require exit 0.
9. Parse all ten standard summary fields plus `max_readout_ratio`. Require 300.0 counted seconds, total <600, 1,075,242 parameters, one switch near 80%, eight workers stopped, 45-55% strong CutMix, hard weak targets, and 18-19 unique evaluation epochs (never >19).
10. Parse `best_test_acc`; >=94.25 is improvement, lower is valid no-improvement without rerun. Compare 80% accuracy with EXP-010's 89.73% and the 87.08 underfit marker, first weak with 93.16%, final NLL with 0.1934, actual steps with 26,898/26,242, max-readout ratio trajectory, and final slope/best gap. These diagnostics cannot override the primary gate. A 94.25-94.35 pass, isolated best spike, or pass with tiny max-path ratio is formal improvement but weak causal evidence.

### Informational Metrics (Optional)
- Final summary: final accuracy/loss, training/startup/total seconds, VRAM, epochs, steps, parameters from targeted `run.log` lines.
- Trajectory: switch, first weak, best epoch, terminal slope, best/final gap, final NLL.
- Max-path provenance: first-step gradient/update norms and initial output/shared-gradient identity from preflight; per-evaluation and final max/average weight-norm ratios from production logs.
- Feasibility: paired training/inference distributions, projected/actual exposure, evaluator count, memory, and total projection.

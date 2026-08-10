# Plan EXP-005: Early Weak-Phase Adaptation
- **Created**: 2026-08-05

## Milestones

### Milestone 1: Boundary decoupled and statically verified
- [x] Create branch `autoresearch/maximize-cifar10-best-test-accuracy-005` from accepted commit `11f8469`.
- [x] Add `AUG_SWITCH_FRACTION = 0.75`; change precisely the post-batch break predicate (`total_training_time >= fraction * TIME_BUDGET_S`) and loader-switch predicate (`progress >= fraction`). Leave the LR computation and `dense_tail_due` predicates keyed to `LR_HOLD_FRACTION=0.8`.
- [x] Pass compilation, Ruff, pre-commit, diff, and tracked-scope checks; assert the diff contains no other behavioral change.

### Milestone 2: Single fixed-seed run executed
- [x] Confirm moving baseline `92.30%`, no stale log, and one idle H20 with approximately 98 GB and no compute process.
- [x] Launch exactly once with seed 42 using `CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1`.
- [x] Confirm one worker switch at 75%, all eight old workers stop, training resumes, and logged steps from 75-80% remain at `lr=0.1` before the unchanged tail begins.

### Milestone 3: Ordered verification completed
- [x] Require `best_test_acc >= 92.40%`; stop formal verification on failure.
- [ ] If accuracy passes, require exit 0, ten unique finite summary keys, `300.0 <= training_seconds < 310.0`, and `total_seconds < 600`.
- [ ] If prior conditions pass, require unique evaluations ending at `num_epochs`, exactly one switch at 75-76%, unchanged `num_params=269722`, and explicit high-LR weak-phase evidence.
- [ ] Collect informational metrics only after all necessary conditions pass.

## Code Changes
- **`train.py`**: Add one augmentation-specific phase constant and use it only for the existing post-batch break and loader-switch predicates. This creates 15 counted seconds of weak crop/flip training at `lr=0.1` before the accepted LR transition, without changing RandAugment strength, worker lifecycle, model, loss, schedule, seed, or evaluator. Main risk: less strong-view exposure may reduce the invariance benefit that produced EXP-004's gain.

## Configuration Changes
- `AUG_SWITCH_FRACTION`: implicit coupling to `0.80` -> explicit `0.75`.
- `LR_HOLD_FRACTION`: remains `0.80`; `lr=0.1` through that boundary, then `0.01` to `1e-4` cosine.
- RandAugment remains one operation at magnitude 7; all EXP-004 controls remain fixed.
- Evaluation cadence deliberately remains unchanged, with no new 75% evaluation. Adding a checkpoint unavailable to EXP-004 would increase model-selection opportunity and confound the best-accuracy comparison; phase wiring is verified from switch/LR logs instead.

## Execution Environment
- Method: one local H20 run under `timeout 600s`.
- Resources: one idle NVIDIA H20, eight persistent forkserver loader workers, cached data, locked environment.
- Estimated runtime: approximately 340 seconds total and 300 seconds counted training.
- Log output: all output redirected to project-root `run.log`; compact monitoring only.
- Tool skill: local execution.

## Abort Criteria
- Abort before launch on stale log, wrong/occupied GPU, multiple exposed GPUs, tracked changes outside `train.py`, or scope/diff mismatch.
- Kill on 600-second timeout, traceback, OOM, non-finite loss, no steps after 120 seconds, missing switch by 80%, worker shutdown failure, or no resumed steps within 60 seconds.
- Do not abort on intermediate accuracy or because early weak-phase accuracy is below EXP-004; use the completed predeclared run.

## Verification Protocol

### Verification Procedure

1. Query `exp-index.sh baseline`; require `92.30` at `11f8469`, otherwise recompute threshold as current baseline plus 0.10.
2. Run static/scope checks and idle-H20 queries exactly as EXP-004. Require only the intended constant and two-predicate diff in `train.py`. Use `rg -n 'AUG_SWITCH_FRACTION|LR_HOLD_FRACTION|dense_tail_due' train.py` to confirm: break and loader switch use 0.75; LR schedule and dense evaluation still use 0.8.
3. Launch one fixed-seed run under 600 seconds; record PID, timestamps, selected GPU, switch line, and compact liveness/error observations.
4. Parse all ten summary keys. First require `best_test_acc >= 92.40%`; if it fails, record `no-improvement` and skip remaining formal necessary conditions.
5. If accuracy passes, require exit 0, each finite key once, counted-time band, and wall limit.
6. Parse evaluation, switch, and step lines. Require unique eval epochs ending at summary epoch, one switch at 75-76%, at least one post-switch/pre-80% step logging `lr: 0.1000`, later low-LR evidence, and unchanged parameter count.
7. Do not rerun or change seed. The official verdict follows the user's predeclared moving-baseline rule and fixed seed 42; additional seeds would not be comparable with the single-seed baseline. A 0.10-point boundary result satisfies that protocol but must be reported as weak causal evidence because augmentation draws change with the phase boundary. The LR log check proves implementation fidelity, not efficacy.

### Informational Metrics (Optional)
- All summary metrics from `run.log`.
- Switch epoch/progress and stopped-worker count.
- Weak high-LR interval evidence and first low-LR checkpoint.
- Step delta, runtime delta, and trajectory versus EXP-004.

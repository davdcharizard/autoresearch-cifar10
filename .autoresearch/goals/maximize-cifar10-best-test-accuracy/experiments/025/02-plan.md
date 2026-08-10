# Plan EXP-025: Identity-Initialized Final-Stage ECA
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Exact-function ECA implementation
- [x] Create experiment branch `autoresearch/maximize-cifar10-best-test-accuracy-025` from integration and confirm baseline 94.15%/threshold 94.25%.
- [x] Before editing, build the unmodified `7c1e7d8` model from seed 42 and persist its full state plus post-construction CPU/CUDA RNG hashes as the identity oracle.
- [x] Modify only `train.py`: add three zero-kernel length-5 ECA gates to `layer3` residual branches immediately before shortcut addition.
- [x] Prove 1,073,977 parameters, three `Conv1d`s/15 zero weights, unchanged nine blocks/19 `Conv2d`s/stage shapes/Option-A, bitwise shared-state/RNG/logit equality, exact BN changes, and per-tensor shared-gradient `max_abs<=1e-7` plus relative norm `<=1e-6` on hard and soft inputs.
- [x] Pass compile, Ruff, format, diff, and scope checks.

### Milestone 2: Immutable-corpus recruitment safety
- [x] Adapt the EXP-024 three-bucket corpus controller under `experiments/025/preflight_eca.py`, importing production `make_train_loader` and `cutmix_collate` and duplicating only the exact source-matched transform declarations; assert loader/collator settings, shapes, target row sums, root-path setup, deterministic cuBLAS environment, and serialize evidence before assertions.
- [x] On separate first hard/soft updates require finite nonzero ECA movement, max weight <=0.25, gate range `[0.75,1.25]`, and per-block mean `[0.95,1.05]`.
- [ ] Across 200 interleaved strong hard/soft steps require finite state, no candidate-only >95% class share, loss-EMA ratio <=1.5, nonzero hard/soft gate gradients, gate range `[0.5,1.5]`, and per-block means `[0.85,1.15]`; report gate quantiles separately by target type.

### Milestone 3: Paired H20 timing
- [ ] On one idle H20, run an unscored conditioning process then five alternating fresh-process pairs with 100 warmups and exactly 400 strong-hard/400 strong-soft/200 weak-hard steps per arm.
- [ ] Pass weighted ratio <=1.035, every pair <1.05, both CVs <2%, candidate p95 <1.08x control mean, projected exposure >=26,000, peak <700 MiB, evaluator ratio <=1.05, and projected total <540s.

### Milestone 4: One scored run and verification
- [ ] Confirm no stale log, then run once with `timeout --kill-after=30s 600s uv run train.py > run.log 2>&1`.
- [ ] Extract the summary and verify `best_test_acc >=94.25%`, complete finite fields, `299.9<=training_seconds<=300.2`, and `total_seconds<600` in necessary-condition order.
- [ ] Record switch/first-weak/peak/final/NLL/exposure/evaluations/CutMix/workers plus hard-versus-soft gate diagnostics; do not rerun a valid job.

## Code Changes

- **`train.py`**: Add `ECAGate` with bias-free `Conv1d(1,1,5,padding=2)`, explicitly zero its kernel, and return `residual * (2*sigmoid(channel_conv(GAP(residual))))`. Construct it inside CPU `fork_rng` so default constructor draws do not perturb shared initialization. Add `use_eca=False` to `BasicBlock`/`_make_layer`, enable only all three `layer3` blocks, and apply before unchanged shortcut addition.
- **Ignored diagnostics**: `preflight_eca.py`, `timing_eca.py`, and PT/JSON evidence under experiment 025 only.
- **Ignored identity oracle**: Persist the pre-edit baseline state/RNG hashes under experiment 025; compare the candidate's shared tensors and RNG state to this oracle, not to another model built through the edited candidate path.

## Configuration Changes

- ECA sites: none -> three `layer3` residual branches; kernel size fixed at 5 from the cited ECA rule; scale fixed at `2*sigmoid` for exact unit initialization.
- Parameters: 1,073,962 -> 1,073,977. Keep every accepted optimizer/data/schedule/evaluator/seed/timer setting unchanged.

## Execution Environment

- Method: local safety/timing controllers, then one local scored run.
- Resources: exactly one idle NVIDIA H20 near 97,871 MiB; existing eight workers; no dependencies.
- Estimated runtime: 2-4 minutes diagnostics and 5.5-9 minutes production.
- Log output: production only in `run.log`; controller evidence experiment-local; never `tee` or stream full output.
- Tool skill: none.
- Safeguards: prepend project root in controllers, launch deterministic work with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, condition device in one unscored subprocess.

## Abort Criteria

- Abort on wrong scope/graph/count/dtype/RNG/shared-state/identity behavior or any non-finite state.
- Abort before timing on any candidate-only >95% share, loss-EMA ratio >1.5, missing hard/soft ECA gradients, or gate/weight bounds above.
- Abort before production on any timing/resource projection failure; do not rescue with fewer gates, another kernel/scale, special LR/decay, fusion, memory format, or precision.
- Treat all numeric preflight/timing constants as pre-registered by this plan: controllers must echo them into reports and may not be edited after measurement to rescue a failure.
- Abort production on non-finite loss, CUDA/OOM/worker failure, or timeout; no seed/operating-point rerun.

## Verification Protocol

### Verification Procedure

1. Query baseline with `exp-index.sh baseline` and `rg '^TIME_BUDGET_S = ' prepare.py` (30s); compute threshold `baseline+0.10`, currently 94.25%, and require budget 300.
2. Before every GPU command run the two registered `nvidia-smi` queries (30s); require exactly one idle H20 near 97,871 MiB.
3. Run `uv run python -m py_compile train.py`, Ruff lint/format, `git diff --check`, and `git diff --name-only` (60s); require only `train.py`. Run structural/identity assertions described in Milestone 1.
4. Run `CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout 180s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/025/preflight_eca.py`; require all Milestone-2 gates, persisted corpus SHA/report, production loader/collator assertions, and exact pre-registered thresholds echoed in the report.
5. Run `timeout 300s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/025/timing_eca.py`; require all Milestone-3 gates and raw trials.
6. Require `find . -maxdepth 1 -type f -name 'run*.log' -print` empty, then run the scored command from Milestone 4.
7. Grep all summary fields. First require `best_test_acc>=baseline+0.10`; on failure stop verification as no-improvement. If passed, require exit zero/finite complete summary, then the goal-authorized fixed counted budget and `<600s` wall condition. The single-seed ten-example margin is formally valid but must be reported as weak causal evidence if unsupported by NLL/trajectory; it cannot change the verdict.
8. Verify one near-80% switch, eight stopped workers, hard weak targets, expected CutMix fraction, seed 42, 1,073,977 parameters, no retry, and unique at-most-once-per-epoch evaluations. Trajectory/NLL/gate evidence informs causal confidence but cannot override the formal verdict.

### Informational Metrics (Optional)

- Extract all goal-listed summary metrics from `run.log` after conditions pass.
- Record switch/first-weak, peak/final gap and NLL, updates/epochs/evaluations, CutMix/workers, and hard/soft gate statistics for analysis regardless of metric outcome.

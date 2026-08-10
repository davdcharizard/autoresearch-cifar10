# Plan EXP-026: Exact-Corpus Balanced Mixup/CutMix Retry
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Isolated 50/25/25 production policy
- [x] Create experiment branch `autoresearch/maximize-cifar10-best-test-accuracy-026`; confirm baseline 94.15%, threshold 94.25%, and 300-second budget.
- [x] Modify only `train.py`: one categorical draw yields 25% alpha-1 CutMix, 25% alpha-0.4 Mixup, or 50% hard during the strong phase; weak phase stays hard.
- [x] Add explicit hard/CutMix/Mixup provenance and switch counts; prove common hard and low-quarter CutMix branches preserve accepted semantics/RNG, model/optimizer/schedule/evaluator remain unchanged, and parameter count is 1,073,962.
- [x] Pass compile, Ruff, format, diff, and scope checks.

### Milestone 2: Natural immutable pre-policy corpus and semantics
- [x] Write `preflight_balanced_mix.py`, importing production loader/policies and prepending project root. A top-level source collator performs only default collation then captures its worker CPU RNG state/id/seed.
- [x] Mirror production parent RNG ordering (loader, model, optimizer, then iterator), persist the first 200 unfiltered post-N1/M7 pre-policy batches/states, fsync and SHA-256 the corpus, validate every tensor/state digest, and shut all eight workers.
- [x] Require natural candidate counts >=35 for hard/CutMix/Mixup, identical total-mixed decisions, unchanged surrounding CPU/CUDA RNG, bitwise equal hard and shared CutMix branches, and valid distinct Mixup/CutMix targets; serialize before assertions.

### Milestone 3: Exact-corpus safety and real-loader timing
- [x] Fresh explicit control/candidate processes load identical seed-42 model/SGD state and the immutable corpus; apply their policies from each stored state and run all 200 steps.
- [x] Require finite state, exact 200 BN counters/complete momentum, no candidate-only >95% class share while control <=95%, terminal loss-EMA ratio <=1.5, and all provenance/count gates.
- [x] Pass a 20,000-collation lifecycle/proportion gate: hard 48.5-51.5%, CutMix/Mixup each 23.5-26.5%, strong workers stopped, weak rebuild <5s and hard two-item batch, weak workers then explicitly stopped, and zero live worker children.
- [x] On one idle H20 run conditioning plus five alternating real-loader pairs with explicit distinct control (one draw; `u<0.5` CutMix) and candidate collators; require recorded policy identities/counts, mean ratio <=1.01, every pair <=1.04, CV <3%, projected steps >=26,629, loader headroom/wait gates, wall/count <=1.07 and <=control+0.02, peak <650 MiB, total projection <540s.

### Milestone 4: One scored run
- [x] Confirm no stale log and one idle H20; run exactly once with `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`.
- [x] Verify necessary conditions in order: `best_test_acc>=94.25%`; exit zero and complete finite summary; counted budget near 300 and total <600.
- [x] Record steps, switch/first-weak/peak/final/NLL, geometry counts, evaluations, workers, memory, and corpus/timing hashes; do not rerun.

## Code Changes

- **`train.py`**: Add `MIXUP_ALPHA=0.4`, `MIXUP_PROBABILITY=0.25`, change CutMix probability to 0.25, construct torchvision `v2.MixUp`, define provenance constants and `apply_strong_policy`, and replace the strong collator with a fork-RNG-isolated triple-return collator. Replace `for inputs, targets in train_iterator` with `for batch in train_iterator`, unpack exactly three items while strong and exactly two while weak, validate provenance/target rank, count each geometry, and print them at the existing switch. Weak loader and all model/optimization/evaluation code remain unchanged.
- **Ignored diagnostics**: experiment-local pre-policy corpus, `preflight_balanced_mix.py`, `timing_balanced_mix.py`, and JSON/PT reports only.

## Configuration Changes

- Strong geometry: 50% hard/50% alpha-1 CutMix -> 50% hard/25% alpha-1 CutMix/25% alpha-0.4 Mixup.
- Total soft-target probability remains 0.5. Preserve batch128, FP32, seed42, width-2 ResNet-20, standard momentum/decay, N1/M7, 80% schedule/switch, hard weak tail, workers, timer, and evaluator.

## Execution Environment

- Method: local CPU semantic/lifecycle controller, local paired H20 safety/timing, then one production run.
- Resources: exactly one idle H20 near 97,871 MiB for GPU commands; eight existing workers; no dependencies.
- Estimated runtime: 3-6 minutes diagnostics and 5.5-9 minutes production.
- Log output: production only to `run.log`; controller evidence experiment-local; never `tee` or full-log streaming.
- Tool skill: none.
- Safeguards: root-path prepend, guarded `if __name__ == "__main__"` for every forkserver controller, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms and identical cuDNN benchmark/deterministic plus TF32 flags recorded per arm, serialize/fsync before veto, one unscored device conditioner.

## Abort Criteria

- Abort on scope/model/optimizer/schedule/evaluator/weak-tail drift, more than one categorical draw, non-natural corpus selection, digest/source mutation, wrong parent RNG ordering, or policy provenance/target/RNG mismatch.
- Abort if any natural geometry count <35, shared hard/CutMix branches are not bitwise equal, backend flags differ, safety state is non-finite/incomplete, candidate-only concentration occurs, or loss-EMA ratio >1.5.
- Abort on any proportion/lifecycle/timing/exposure/loader/memory/wall gate. Do not rematerialize corpus or rescue with another alpha, ratio, seed, workers, GPU mixing, threshold, or pure CutMix fallback.
- Abort production on non-finite loss, CUDA/OOM/worker failure, or timeout; no valid-run retry.

## Verification Protocol

### Verification Procedure

1. Query `exp-index.sh baseline` and `rg '^TIME_BUDGET_S = ' prepare.py` (30s); derive current threshold baseline+0.10.
2. Before each GPU command run both registered `nvidia-smi` queries (30s); require exactly one idle H20 near 97,871 MiB.
3. Run compile/Ruff/format/diff/scope checks (60s), plus static policy/model/optimizer/schedule/evaluator assertions; only `train.py` may differ.
4. Run `CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout 240s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/026/preflight_balanced_mix.py`; require all Milestone-2/3 semantic, corpus, safety, proportion, and lifecycle gates with report written before assertion.
5. Run `timeout 300s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/026/timing_balanced_mix.py`; require all registered real-loader timing gates and raw paired trials.
6. Require no `run*.log`, then run the Milestone-4 command once.
7. Before necessary-condition evaluation, establish that the process exited zero and emitted one complete finite final summary; a timeout, partial log, or missing field is crash/invalid rather than a numeric miss. Then compare best accuracy with baseline+0.10 and stop as no-improvement if lower. If passed, evaluate the remaining goal conditions in order: complete summary and `299.9<=training_seconds<=300.2`, `total_seconds<600`.
8. Verify exactly one switch with reported progress in `[79.5,80.5]%`, exactly eight strong workers stopped, hard weak targets, production geometry intervals, 1,073,962 parameters, no retry, and unique at-most-once-per-epoch evaluations. Peak VRAM is informational in production; the pre-production timing gate alone requires `<650 MiB`. Diagnostics cannot override formal verdict.

### Informational Metrics (Optional)

- Extract all goal-listed summary metrics after conditions pass.
- Record geometry counts, switch/first-weak/peak/final/NLL, steps/epochs/evaluations, worker lifecycle, memory, corpus SHA, and timing projections regardless of metric outcome.

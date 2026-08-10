# Plan EXP-024: Depth-Preserving Final-Stage Widening to 160
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Isolated 32/64/160 ResNet-20 implementation
- [x] Create experiment branch `autoresearch/maximize-cifar10-best-test-accuracy-024` from the integration branch and verify the moving baseline is 94.15% (`best_test_acc`, higher).
- [x] Modify only tracked `train.py`: add an explicit final-stage channel override, retain nine residual blocks and widths 32/64 before `layer3`, and instantiate the candidate at 32/64/160.
- [x] Prove 19 convolutions, nine blocks, stage outputs 32/64/160 at 32/16/8 resolution, two unchanged Option-A transitions with pads 32/96, `Linear(160,10)`, 1,507,818 parameters, and FP32 state.
- [x] Pass `py_compile`, Ruff lint/format, `git diff --check`, and a scope check showing only `train.py` differs.

### Milestone 2: Exact-corpus numerical and optimization safety
- [x] Write the ignored controller `experiments/024/preflight_stage_width.py`; prepend the resolved project root to `sys.path` and serialize evidence before applying veto assertions.
- [x] Materialize and hash three production-path buckets once: 100 strong-hard N1/M7 batches, 100 strong-soft alpha-1 CutMix batches, and 100 weak-hard crop/flip batches. Replay the 200 strong buckets byte-identically through fresh explicit `ResNet(3,10,2)` control and `ResNet(3,10,2,160)` candidate processes.
- [ ] Confirm finite hard/soft forward-loss-backward-update state, correct BN/eval behavior, no candidate-only >95% class concentration, and record predeclared per-stage gradient/update diagnostics.

### Milestone 3: Paired H20 timing and exposure gate
- [ ] Confirm exactly one idle NVIDIA H20 with approximately 97,871 MiB, then run one unscored conditioning process followed by five alternating fresh-process control/candidate pairs.
- [ ] Measure 100 warmups plus exactly 400 strong-hard, 400 strong-soft, and 200 weak-hard complete synchronized steps per arm, recording mean/p50/p95, forward/backward/update components, peak allocation, pair order, and CV. Separately time evaluator-shaped batch-256 inference and charge all projected evaluations, startup, and phase-switch overhead to the wall projection.
- [ ] Advance only if weighted ratio is <=1.12, every pair <1.15, both CVs <2%, candidate p95 <1.20x its paired control mean, projected exposure >=24,000 updates, peak <1.0 GiB, and projected total <540 seconds.

### Milestone 4: One scored fixed-budget run
- [ ] Confirm hardware and absence of stale `run*.log`, then launch exactly once with `timeout --kill-after=30s 600s uv run train.py > run.log 2>&1`.
- [ ] Monitor bounded status/GPU activity without streaming full output; kill on timeout, CUDA/OOM/worker error, non-finite training, or loss of GPU activity.
- [ ] Extract the numeric summary and trajectory diagnostics: switch/first-weak checkpoints, peak/final accuracy and NLL, steps/epochs/evaluations, runtime, parameters, VRAM, worker shutdown, and CutMix fraction.
- [ ] Verify `best_test_acc >=94.25%`, complete numeric output, 300-second counted budget, and total runtime below 600 seconds; preserve the log for analysis, then remove it before the following experiment.

## Code Changes

- **`train.py`**: Add `FINAL_STAGE_CHANNELS = 160`. Extend `ResNet.__init__` with an optional `final_stage_channels` argument; derive `c1=32` and `c2=64` from the unchanged width multiplier and set only `c3` from the override. Instantiate production with the override. Keep `BasicBlock`, `_make_layer`, Option-A slicing/padding, initialization, global average pooling, classifier behavior, data pipeline, optimizer, schedule, timing, evaluation, seed, and summary schema unchanged.
- **Ignored experiment artifacts**: Add `preflight_stage_width.py`, `timing_stage_width.py`, and their JSON/PT reports only under `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/024/`. These are diagnostics and must not alter tracked production code or the scored forward path.

## Configuration Changes

- Stage widths: `32/64/128 -> 32/64/160`; depth remains ResNet-20 with three blocks per stage.
- Final classifier input: `128 -> 160`; expected parameter count `1,073,962 -> 1,507,818` (+40.40%).
- Estimated convolution MACs/image: `161.32M -> 189.04M` (+17.18%); paired H20 measurement, not the estimate, decides feasibility.
- Preserve batch 128, FP32/default-TF32, seed 42, standard SGD LR 0.1/momentum 0.9/all-parameter decay `1e-4`, 80% high-LR hold, 0.01-to-`1e-4` cosine tail, N1/M7 plus probability-0.5 alpha-1 CutMix in the strong phase, hard weak tail, workers, timer, and evaluator.

## Execution Environment

- Method: Local single-process safety/timing controllers followed by one local production command from the project root.
- Resources: Exactly one idle NVIDIA H20 with approximately 97,871 MiB; eight existing loader workers; no new dependency or package installation.
- Estimated runtime: Approximately 2-4 minutes for safety/timing and 5.5-9 minutes total for the scored run; every individual production command is capped at 600 seconds.
- Log output: Production stdout/stderr only in project-root `run.log`, never `tee`; controller data stays in experiment-local JSON/PT files. Do not stream the full production log into agent context.
- Tool skill: None; no remote scheduler or WandB is used.
- Infrastructure safeguards: Controllers resolve and prepend project root before importing `train`; deterministic diagnostics launch with `CUBLAS_WORKSPACE_CONFIG=:4096:8`; an unscored fresh subprocess conditions the H20 before alternating timing trials.

## Abort Criteria

- Stop before GPU work if the branch/base/baseline is wrong, a stale `run*.log` exists, tracked scope exceeds `train.py`, or static checks fail.
- Stop before timing/production on an incorrect graph, parameter count, dtype, optimizer/data/evaluator configuration, Option-A geometry, non-finite state, candidate-only >95% prediction concentration, or candidate terminal loss EMA >1.5x control. Record per-stage parameter-normalized gradient/update ratios as diagnostics only; unequal architectures make them unsuitable as an ambiguous post-hoc veto.
- Stop before production if weighted timing ratio >1.12, any pair >=1.15, either CV >=2%, candidate p95 >=1.20x paired control mean, projected exposure <24,000, peak allocation >=1.0 GiB, or projected total >=540 seconds. Do not rescue with width 144/192, learned transitions, precision, memory format, batch/LR/decay changes, or another run.
- Stop production on non-finite loss, OOM/CUDA/worker failure, or 600-second timeout. Use `timeout --kill-after=30s` so a TERM-resistant CUDA process cannot outlive the hard limit. Treat inactivity as a fault only when the process remains alive, the log has not advanced, and no compute application is visible for 180 consecutive seconds; evaluation and the loader switch are exempt. Do not rerun a valid seed-42 job or tune the final width after observing results.

## Verification Protocol

### Verification Procedure

1. Query the moving baseline and fixed budget (timeout 30 seconds):
   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv
   rg '^TIME_BUDGET_S = ' prepare.py
   ```
   Read the numeric baseline from command output and compute the threshold as `baseline + 0.10`; at planning time the expected values are 94.15% and 94.25%. Require `TIME_BUDGET_S = 300`; stop and re-plan if either value changed.

2. Before every GPU command, verify the resource (timeout 30 seconds):
   ```bash
   nvidia-smi --query-gpu=index,name,memory.total,compute_mode --format=csv,noheader
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
   ```
   Pass only with exactly one visible H20 near 97,871 MiB and no competing compute process.

3. Verify implementation and tracked scope (timeout 60 seconds):
   ```bash
   uv run python -m py_compile train.py
   uv run ruff check train.py
   uv run ruff format --check train.py
   git diff --check
   git diff --name-only
   ```
   All commands must exit zero and the last command must print only `train.py`. A static controller must additionally assert exactly nine blocks, 19 `Conv2d`s, stage shapes `(N,32,32,32)/(N,64,16,16)/(N,160,8,8)`, Option-A pads 32/96, `fc.in_features=160`, and 1,507,818 parameters.

4. Run immutable-corpus safety (timeout 180 seconds):
   ```bash
   CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout 180s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/024/preflight_stage_width.py
   ```
   Before constructing each arm, reset CPU/CUDA seed 42 and deterministic settings, record post-construction RNG metadata, and load the same persisted corpus SHA. Pass only when the serialized report proves the three hashed 100-batch buckets, explicit control/candidate constructors, finite hard/soft updates/parameters/buffers/momentum, correct BN/eval state, no candidate-only concentration, and loss-EMA ratio <=1.5. Record gradient/update ratios without using them as efficacy evidence or an undefined veto; this preflight tests numerical safety, not late accuracy.

5. Run paired timing (timeout 300 seconds):
   ```bash
   timeout 300s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/024/timing_stage_width.py
   ```
   Pass only with one unscored conditioning subprocess and five alternating fresh-process pairs. Reset seed/state per arm and replay exactly 400 strong-hard, 400 strong-soft, and 200 weak-hard steps after 100 warmups. Require weighted ratio <=1.12, every pair <1.15, both CVs <2%, candidate p95 <1.20x paired control mean, projected exposure >=24,000 updates, and peak <1.0 GiB. Separately time batch-256 evaluator forwards, then project wall time by charging measured candidate evaluation cost for the accepted checkpoint schedule plus observed accepted startup and phase-switch overhead; require the conservative total <540 seconds. Persist all raw trials and projection terms.

6. Confirm no stale logs and run once (timeout 600 seconds):
   ```bash
   find . -maxdepth 1 -type f -name 'run*.log' -print
   timeout --kill-after=30s 600s uv run train.py > run.log 2>&1
   ```
   The first command must print nothing. The scored command must exit zero; no full-run retry is permitted for a valid result.

7. Verify the three necessary conditions in order. First extract the primary and summary fields:
   ```bash
   grep "^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:" run.log
   ```
   Parse numerically. (a) Require `best_test_acc >= baseline+0.10` (currently 94.25%); on failure classify `no-improvement` and stop checking necessary conditions. (b) If it passes, require exit zero and all expected finite numeric summary fields. (c) If that passes, require `299.9 <= training_seconds <= 300.2` (one-step overshoot plus one-decimal reporting tolerance) and `total_seconds <600`. Do not reinterpret a failure using trajectory diagnostics. The fixed seed and best-over-checkpoints protocol are user-defined; a bare threshold pass is formally valid but causal confidence must be reported as weak unless supported by trajectory/NLL evidence.

8. Verify run integrity and interpret the mechanism: require exactly one near-80% loader switch, eight stopped workers, hard targets after the switch, seed 42, expected CutMix fraction, 1,507,818 parameters, no retry, and at-most-once-per-epoch unique evaluations. Record actual exposure; 24,000 is a pre-run timing floor, while a lower actual count is a diagnostic discrepancy rather than an invented goal condition. Compare switch accuracy to 89.73% and the 87.08% underfit marker, first-weak accuracy to 93.16%, and final NLL to 0.1934. Weak first-tail conversion implicates the joint late-stage intervention (transition geometry, wider convolutions/BN, classifier, initialization stream, and exposure), not the shortcut alone; healthy conversion followed by a miss weighs against static late capacity under this net operating point.

### Informational Metrics (Optional)

- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, `num_params`: extract from the final summary grep in step 7 after all necessary conditions pass.
- Switch/first-weak accuracy, peak epoch/NLL, final-best gap, evaluation count, worker shutdown, and CutMix fraction: parse the registered trajectory/status lines in `run.log`; record regardless of pass/fail for analysis, without using them to rescue the formal verdict.

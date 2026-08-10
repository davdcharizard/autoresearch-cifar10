# Plan EXP-031: Scale-Controlled Max-Residual Global Pooling
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Freeze calibration and implement the bounded descriptor
- [x] On the clean baseline, calibrate `s=min(1,rms(avg)/rms(max-avg))` once from seed-42 initialization and training indices 0-1023 with deterministic weak normalization only; serialize hashes/values and hardcode eight significant digits.
- [x] Create the experiment branch and change only `train.py`: add the frozen scale, replace final average pooling with `avg + 0.10*s*(max-avg)`, and add eval-only descriptor-ratio accumulation/reset/reporting that cannot change logits or coefficient.
- [x] Pass compile, Ruff, format, pre-commit, exact scope/diff, parameter/RNG identity, synthetic formula, target-format, evaluator-call-count, and training-only perturbation-monitor checks. (Static checks passed; exact-corpus dynamic gates later vetoed the candidate.)

### Milestone 2: Pass semantic, trajectory, and timing vetoes
- [x] Reuse and hash the registered 200-batch strong corpus and 64-batch weak corpus; replay identical control/candidate states and require finite geometry, bounded descriptor/logit/gradient/update ratios, no candidate-only class concentration, and recorded ratio drift. (Failed immutable dynamic gates.)
- [x] On one idle H20, run five counterbalanced fresh-process paired full-step trials for strong-hard, strong-CutMix, and weak-hard paths plus evaluator timing; require <=1% weighted mean overhead, without an absolute production step-count gate. (Skipped after safety veto.)
- [x] If any immutable safety or timing gate fails, record this exact point as invalid without changing coefficient, corpus, gates, or implementation.

### Milestone 3: Execute and verify one scored run
- [ ] Only after all vetoes pass, confirm one idle H20 and no stale log, then run seed 42 exactly once with all output redirected to `run.log` under the 595-second process bound.
- [ ] Require clean completion, fixed budget, lifecycle/target/evaluator integrity, and `best_test_acc >=94.25%` for improvement; treat production steps as informational and never rerun a valid result.
- [ ] Compare switch/first-weak/tail/NLL and eval-mode descriptor-ratio drift with EXP010 to determine whether bounded localized evidence helped or amplified artifacts.

## Code Changes
- **`train.py`**: add one frozen `POOL_RESIDUAL_SCALE` constant. In `ResNet.forward`, compute flattened global average and max descriptors and feed `avg + 0.10 * POOL_RESIDUAL_SCALE * (max_ - avg)` to the unchanged `fc` layer. Parameter count, initialization, classifier, optimizer, data, schedule, and timed-step control flow remain unchanged.
- **`train.py`**: add non-persistent Python attributes/methods that, only while `model.training is False`, accumulate detached sums of squared average and max-residual descriptor values and report their aggregate RMS ratio around each existing evaluator call. Reset immediately before evaluation and read immediately afterward. These diagnostics add no train-time operation, do not enter `state_dict`, do not change logits, and must not select or revise the frozen coefficient.
- **`train.py`**: retain detached references to the current training descriptor components and, only at preregistered `step % 1000 == 0`, compute the added-residual/average RMS ratio before the existing synchronization so its reduction cost is counted. Log after synchronization and abort if the ratio exceeds 0.25. This training-only safety cap prevents initialization calibration from masquerading as a lifetime bound; it never reads test metrics and cannot retune the coefficient.

## Configuration Changes
- `POOL_RESIDUAL_SCALE`: absent -> one eight-significant-digit scalar from the preregistered training-only calibration.
- Effective max coefficient: `c=0.10*POOL_RESIDUAL_SCALE`, constrained to `0<c<=0.10`; average coefficient is `1-c`.
- Unchanged: width-2 ResNet-20 and 1,073,962 parameters, batch 128, FP32/default TF32, ordinary momentum 0.9, all-parameter decay `1e-4`, LR 0.1 through 80%, LR-0.01-to-`1e-4` weak tail, N1/M7, CutMix p=0.5/alpha=1, seed 42, workers, timer, and evaluation count/predictions.

## Adversarial Review Response
- The idea review selected pooling for its direct match to the representation/generalization limiter and the named spatial-aggregation open question; momentum reset was cleaner but structurally low-impact, and channels-last depended on two unproven links.
- Adopted scale-drift observability: exact-corpus replay records descriptor ratios throughout, and eval-only accumulation reports the aggregate ratio at every existing evaluation, especially switch and terminal. Drift is diagnostic only and cannot modify `s`, veto a completed metric, or alter model output.
- The plan review correctly rejected initialization calibration as a production-lifetime bound. A sparse training-only monitor now hard-vetoes added-residual/average RMS above 0.25 at fixed 1,000-step intervals; eval/test ratios remain telemetry and can never control execution or tuning.
- Adopted area-insensitivity and runtime controls: production-distribution CutMix replay gates class/update geometry before fresh paired timing, and a >1% weighted overhead blocks production without coefficient rescue.
- EXP014 is not retried: there is no new classifier, parameter, optimizer state, raw max logit, or zero-init recruitment. The shared classifier receives a fixed convex descriptor whose added residual RMS is calibrated and gated.
- Removed absolute production exposure floors. Fresh paired candidate/control timing is the only overhead gate; `num_steps` from the scored wall-clock run is informational, preventing node-speed selection or retry-until-fast behavior.
- Planning verified both reused corpora exist and match their registered hashes. Calibration is pinned to `torch.manual_seed(42)`, CPU model construction, then `.to(cuda)`, exactly matching production initialization order.

## Execution Environment
- Method: ignored local calibration/safety/timing controllers followed conditionally by one local `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`.
- Resources: one idle NVIDIA H20 with approximately 97,871 MiB for CUDA safety/timing/production; CPU/forkserver workers and existing local CIFAR data; no dependency changes.
- Estimated runtime: calibration <2 minutes; exact-corpus safety 2-4 minutes; paired timing approximately 12-20 minutes; scored run approximately 330-350 seconds if authorized.
- Log output: controller reports under `experiments/031/`; production stdout/stderr only in root `run.log`; bounded monitoring only, never `tee`.
- Tool skill: `/research-execute`; no remote submission or W&B.

## Abort Criteria
- Before production, stop for non-reproducible/out-of-range calibration, test-data/evaluator use during calibration, any tracked change beyond `train.py`, state/RNG/parameter mismatch, wrong pooling algebra, changed logits from diagnostics, corpus/hash mismatch, nonfinite state, candidate-only >95% concentration, or any preregistered scale/gradient/update/timing threshold failure.
- During a scored run, terminate for wrong/busy GPU, traceback, OOM/resource failure, non-finite loss/state, training-only added-residual/average RMS above 0.25 at a fixed monitor point, target/lifecycle assertion, no bounded progress for 120 seconds, or the 595-second timeout.
- Do not abort a valid production run for low switch accuracy, descriptor drift, worse NLL, or low intermediate top-1; those diagnose the mechanism and the fixed-budget result must complete.
- One valid seed-42 completion only. One unchanged retry is permitted solely for a documented external infrastructure failure preventing a valid summary. Never rescue with another coefficient/corpus, learned/per-channel scale, GeM, normalization, clipping, extra evaluation, or another seed.

## Verification Protocol

### Verification Procedure
1. **Baseline/source (10s):** query `exp-index.sh baseline` and require 94.15 at `7c1e7d8`; require pristine `train.py` hash equal to that commit and preserve untracked `data/`.
2. **Calibration (120s):** an ignored controller sets CPU and CUDA seed 42, constructs the model on CPU in the exact production order, records the initialized tensor/RNG hash, then moves it to CUDA; BN remains in default running-stat state before `eval()`. Without `Eval.evaluate()`, load training indices 0-1023 in order with `ToTensor` plus accepted normalization, capture final post-ReLU maps, and accumulate float64 `rms_A` and `rms_R`. Repeat independently and require identical model/source/corpus hashes and eight-digit `s=min(1,rms_A/max(rms_R,1e-12))`, `0<s<=1`, `rms_R>0`, and `RMS(0.10*s*R)/RMS(A)<=0.100001`. Freeze the value before candidate safety/timing.
3. **Scope/semantic checks (120s):** run compile, Ruff, format, pre-commit, diff, and AST checks. Require only `train.py`; unchanged state dict/RNG/optimizer identity and 1,073,962 parameters; exact constant-map equality; FP64 random-map formula agreement; finite hard/soft CE; descriptor perturbation RMS <=0.12; logit cosine >=0.995; classifier-gradient and total first-update ratios in `[0.80,1.25]`; and no candidate-only initial concentration. Eval diagnostics must leave logits/state bitwise unchanged and evaluator call count unchanged.
4. **Corpus existence and exact-corpus safety (300s):** before controller construction, require both reused files to exist and verify EXP022's 200 strong-batch corpus SHA-256 `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946` and EXP028's 64 weak-batch corpus SHA-256 `ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032`, plus their schemas. Replay accepted/candidate from independent exact state copies in identical order and serialize evidence before assertions. Require finite logits/loss/gradients/parameters/BN/SGD; exact BN counters and corpus/RNG provenance; no candidate-only >95% class share; strong and weak terminal loss-EMA ratios <=1.10; update p95 <=1.25 and max <=1.50; classifier-gradient p95 <=1.30; maximum per-example descriptor perturbation/average norm <=0.75. Record added-residual/average RMS at initialization, every 20 strong steps, strong terminal, every 16 weak steps, and weak terminal; require every recorded aggregate ratio <=0.25. Drift cannot retune `s`.
5. **Fresh paired timing (1,500s):** on one idle H20 run five counterbalanced fresh-process control/candidate pairs, each with 100 warmups and >=1,000 synchronized full steps per arm, measuring strong-hard/strong-soft/weak-hard paths at 40/40/20 weighting. Require weighted mean candidate/control ratio <=1.0100, every pair <=1.04, per-arm CV <=2%, candidate p95 <=1.04x control, peak <620 MiB, and finite state. Report fresh control/candidate milliseconds and projected 300-second steps from those same trial means, but apply no historical or production absolute-step veto. Separately require evaluator-like mean <=1.05x control, CV <=2%, <=19 unchanged evaluations, and total-wall projection <540s. No candidate-only synchronization or excluded cost.
6. **Production environment/run (595s):** only after all prior gates pass, confirm exactly one idle NVIDIA H20 and no stale completed log; execute once with `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`. Exit zero is required; infrastructure-only failure permits at most one identical retry, never a valid-result rerun.
7. **Integrity (30s):** first import `TIME_BUDGET_S` and require it remains 300 as specified by the goal's established protocol; then require one finite ten-field summary, `TIME_BUDGET_S<=training_seconds<TIME_BUDGET_S+1`, `total_seconds<595`, `num_params=1073962`, one 80% switch/eight stopped workers, 45-55% strong CutMix, hard weak targets, no fatal/nonfinite signal, and 18-19 unique evaluator epochs including terminal with no epoch repeated. Production `num_steps` is informational only. Require one finite eval diagnostic ratio per actual evaluator call without changing logits, adding forwards, or adding `Eval.evaluate()` calls: `ResNet.forward` piggybacks detached accumulation on the evaluator's existing test-set forwards and `train.py` reads it after return. Test-set ratios are telemetry only and can never tune `s`, gate execution, or affect verdict.
8. **Primary verdict (20s):** compare parsed `best_test_acc` with 94.25%. All integrity conditions plus >=94.25% is improvement; a valid lower value is no-improvement without retry. Compare switch with 89.73%/87.08%, first weak with 93.16%, final NLL with 0.1934, and informational steps with 26,898, plus ratio drift from initialization through terminal. These mechanism diagnostics cannot override the metric.

### Informational Metrics (Optional)
- Frozen `s`, effective coefficient, calibration RMS/hash evidence: calibration report.
- Exact-corpus descriptor/logit/gradient/update/concentration and ratio-drift series: safety report.
- Paired full-step/path/evaluator ratios, CV, p95, exposure and wall projections, memory: timing report.
- Production summary, evaluation trajectory, switch provenance, descriptor-ratio trajectory: targeted `run.log` parsing, with values copied into `03-execute.md` before log removal.

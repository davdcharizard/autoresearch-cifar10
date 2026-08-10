# Plan EXP-039: Intrinsically Bounded Average-plus-RMS Readout
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement and prove lifetime pooling bounds
- [x] Create the EXP039 branch; replace only final adaptive average pooling/flattening with fixed `mu + (rms-mu)/64` in tracked `train.py`.
- [ ] Pass static/scope and exact construction/RNG/inventory checks; preserve all accepted parameters, optimizer, data, schedule, timer, and evaluator.
- [ ] Pass FP64 zero/constant/one-hot/random/sparse output and VJP oracles; prove descriptor `[1,71/64]` and Jacobian `[63/64,71/64]` relative bounds for every checked nonnegative map.

### Milestone 2: Qualify activity, safety, and cost
- [ ] On fixed unlabeled train tensors, prove distributed nonzero descriptor/logit activity without changing coefficient; activity is diagnostic and never selected by accuracy or labels.
- [ ] Re-hash EXP022/028, qualify two accepted controls, and replay200 strong+64 weak batches; require exact semantics, finite bounded trajectory, no candidate-specific concentration, and algebraic lifetime bounds at registered looks.
- [ ] Run one conditioner and seven counterbalanced complete-step timing pairs; require stable catastrophic-feasibility bounds and accepted wall/evaluation behavior.

### Milestone 3: Run and verify once
- [ ] Re-query baseline, scope, idle H20, evaluator ceiling, and stale logs; execute seed42 once with output only to `run.log` under595 seconds.
- [ ] Require complete finite protocol summary and compare `best_test_acc` to94.25; record switch fit, first weak, final/NLL, exposure, descriptor activity, and cost without reroll.

## Code Changes

- **`train.py`**: after `layer3`, compute `avg = out.mean((2,3))`, `rms = torch.linalg.vector_norm(out, dim=(2,3)) / 8.0`, and `out = torch.lerp(avg, rms, 1.0/64.0)` before the existing `fc`. Remove only the old adaptive-pool/flatten lines. Installed PyTorch2.9.1/Python3.14 was directly checked in FP32/FP64: zero-map `vector_norm` backward is finite with exact zero gradient. No epsilon, parameter, gate, hook, phase branch, evaluator change, or diagnostic enters production.
- **Ignored artifacts**: experiment-local oracle/activity/replay/timing controllers and durable JSON/log reports only; controllers may read tracked source but write only inside ignored EXP039 paths.

## Configuration Changes

- Final descriptor: global average -> fixed `63/64` average plus `1/64` RMS for the accepted nonnegative8x8 map.
- Unchanged: 1,073,962 parameters, width2 ResNet20, batch128, seed42, FP32/default TF32, N1/M7+CutMix through80%, weak hard tail, SGD/decay/LR, loaders, timer, evaluator cadence, and summary.

## Execution Environment

- Method: ignored local controllers, then conditional `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`.
- Resources: one idle 97,871-MiB H20, existing data/environment and registered corpora.
- Estimated runtime: 5-8 minutes preflight, 5-8 minutes timing, ~335 seconds production.
- Log output: ignored experiment-local logs/JSON; production only root `run.log`, never `tee`.
- Tool skill: `/research-execute`.

## Abort Criteria

- Abort before production for scope/construction/oracle/bound violation, stale corpus, nonfinite state, failed controls, candidate-specific persistent >95% class concentration, global gradient/update/logit ratio >5x qualified controls, update >25% parameter norm or >5x preceding16-step median, phase loss EMA >1.5x, exact-BN failure, catastrophic timing, wrong GPU, changed evaluator, or projected wall >=540s.
- Activity requires nonzero descriptor and logit absolute RMS on both fixed hard/CutMix views and at least25% of descriptor channels changed above FP32 resolution; it is not an accuracy gate and cannot tune `1/64`.
- Timing aborts only for aggregate candidate/control mean >1.05, any pair >1.10, pair-ratio CV>=5%, peak>=650MiB, wall/count>1.10, or total>=540s. Ordinary overhead is priced by the scored fixed-time run; arm CV is recorded but cannot veto when paired ratios are stable.
- Production stops only for fatal/nonfinite/resource/lifecycle/timer faults,120s without progress, or guard timeout—not low finite metrics. No coefficient/GeM/gate/phase rescue or reroll.

## Verification Protocol

### Verification Procedure

1. Query baseline using `exp-index.sh baseline`; require94.15 at `7c1e7d8` and threshold94.25. Check status/diff/ancestry, registered corpus hashes, and preserve `data/`.
2. Run `uv run python -m py_compile train.py`, Ruff check/format, pre-commit, and an ignored construction/oracle controller. Require exact state/RNG/inventory and separately coded FP64 formula/VJPs including zero subgradient and constant-map identity.
3. On immutable unlabeled hard/CutMix tensors, capture accepted final maps and compute accepted/candidate descriptors/logits from identical maps. Require all algebraic bounds, nonzero distributed activity, and operator-norm logit ceiling; do not use targets or top-1.
4. Run two accepted controls then candidate on EXP022 SHA `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946` plus EXP028 SHA `ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032`. Freeze/hash controller thresholds before candidate, serialize before assertions, and apply abort bounds with denominator-safe control envelopes.
5. After one conditioner run seven alternating fresh pairs, each100 warmups+1000 complete synchronized steps at40/40/20 hard/CutMix/weak weighting. Persist raw trials, apply timing bounds, and prove the unchanged evaluator remains at most once per epoch. Evaluation count is informational, not a candidate veto.
6. Recheck one idle H20, only `train.py`, no stale root log, and run production once under the declared timeout/redirection.
7. Parse all ten summary fields with `grep '^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:' run.log`. Require exit0, training `[300,301)`, total<600, params1,073,962, one ~80% switch/eight stopped workers, CutMix45-55%, hard weak targets, and no more than one evaluation per epoch. Record the opportunity count and any baseline difference without treating it as a hard gate.
8. Improvement requires all user goal conditions and `best_test_acc>=94.25`; a complete lower score is no-improvement. Mechanism diagnostics are informational after a valid production result. Never reroll.

### Informational Metrics (Optional)

- Ten final fields and evaluation/switch trajectory from `run.log`.
- Descriptor ratio/activity distributions, Jacobian bounds, absolute logit changes, class shares, loss/gradient/update trajectory, corpus/controller/source hashes.
- Paired timing ratios/CV/memory and projected versus actual exposure/wall.

## Adversarial Review Response

- Uses lifetime algebraic descriptor/Jacobian bounds rather than initialization calibration, directly distinguishing EXP039 from EXP014/031.
- Adds an unlabeled non-scoring activity diagnostic to expose a functional no-op without selecting the coefficient.
- Preserves accepted evaluator code and treats switch fit as attribution evidence, not an added user success condition.
- Confirms the installed `vector_norm` zero-map FP32/FP64 backward is finite and exactly zero; records rather than hard-gates throughput-dependent evaluation count.

# Plan EXP-032: Reset Momentum at the 80% Objective Boundary
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement exact once-only reset semantics
- [x] Create the experiment branch and modify only `train.py`: add a helper that zeroes every live SGD momentum buffer in place and returns the count.
- [x] Call it exactly once after the 80% switch evaluation and successful weak-loader construction, before the first weak update; append `momentum_buffers_reset: 59` to the existing switch line.
- [x] Pass compile, Ruff, format, pre-commit, exact diff/scope, AST ordering, parameter/state/RNG invariance, and analytic PyTorch recurrence checks.

### Milestone 2: Prove copied-boundary trajectory safety
- [x] Verify registered strong/weak corpus existence, schemas, and hashes; train one accepted source state through 200 strong records, clone it exactly, then reset only the candidate's 59 buffers.
- [x] Replay 64 identical weak hard-label records at LR 0.01 and require finite state, exact first/subsequent recurrences, no candidate-only class concentration, and bounded update/loss geometry.
- [x] Serialize/fsync all evidence before assertions; any gate failure invalidates the exact full reset without partial/delayed/per-layer rescue.

### Milestone 3: Execute and verify one scored run
- [x] With safety passed, confirm one idle H20 and no stale log; run seed 42 once under 595 seconds with all output redirected to `run.log`.
- [x] Require clean fixed-budget completion, exactly one 80% reset of 59 buffers, unchanged lifecycle/targets/evaluator, and `best_test_acc >=94.25%` for improvement.
- [x] Compare switch, first weak, tail, NLL, best-final gap, train loss, and exposure with EXP010; never rerun a valid completion.

## Code Changes
- **`train.py`**: add `@torch.no_grad()` helper `reset_sgd_momentum(optimizer)`. Iterate all optimizer group parameters, require each state to contain `momentum_buffer`, zero it in place, and return the count. Do not delete/recreate state, optimizer, parameters, gradients, or groups.
- **`train.py`**: in the existing one-way transition block, call the helper after switch evaluation, strong-worker shutdown, and weak-loader construction, but before `randaugment_enabled=False` and before the next iterator. Extend only the switch provenance line with the returned count.

The first weak recurrence changes from `b1=0.9*bs+d1; w1=w0-lr*b1` to `b0=0; b1=d1; w1=w0-lr*d1`. All later updates remain installed ordinary PyTorch momentum SGD. The direct inherited difference decays as `0.9^k`; this is a bounded state-transition intervention, not a global optimizer replacement.

## Configuration Changes
- SGD momentum buffers at the 80% boundary: retained -> zeroed once in place.
- Expected reset count: exactly 59 parameter tensors.
- Unchanged: momentum scalar 0.9, LR schedule including 0.01 tail start, all-parameter decay `1e-4`, model/parameters, N1/M7, CutMix p=0.5/alpha=1, weak hard tail, batch 128, seed 42, FP32/default TF32, timer, workers, and evaluator cadence.

## Adversarial Review Response
- The primary +0.10 gate and single seed are user-defined goal constraints. A 94.25-94.35 result must therefore be classified as formal improvement and merged, but will be reported as weak evidence whose tiny effect may not reproduce; the plan cannot add a confirmation seed/rerun without violating the goal.
- Accepted the max-over-looks concern: production now requires exactly EXP010's 19 unique evaluation epochs including terminal. More or fewer looks invalidate this run for comparability; the candidate cannot gain an extra maximum draw.
- Planning directly verified both reused corpus files exist and byte-match the registered SHA-256 values before branch work. Missing/drifted corpus would block implementation and classify the preregistered experiment invalid; regeneration/substitution is not permitted.
- Retained copied-state safety despite the reset's low-risk structure because recurring optimizer-path failures make immutable recurrence evidence a project standard. The controller is kept narrowly scoped and controller bugs may receive one code-only correction without changing candidate/gates, as allowed by the execution workflow.
- Worker lifecycle uses imported `NUM_WORKERS` as the expected count rather than a hardcoded eight. Buffer count 59 remains an architecture/state invariant verified before production.

## Execution Environment
- Method: ignored copied-state controller, then conditionally local `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`.
- Resources: exactly one idle NVIDIA H20 with approximately 97,871 MiB for safety and production; existing data/environment only.
- Estimated runtime: copied-state safety under 5 minutes; scored production approximately 330-345 seconds.
- Log output: controller report under `experiments/032/`; production only in root `run.log`; bounded monitoring, never `tee`.
- Tool skill: `/research-execute`; no timing campaign because 59 one-time zeros add no recurring step work and execute at an existing uncounted transition.

## Abort Criteria
- Before production, stop for scope/order/count mismatch, missing buffer, mutation of parameters/BN/gradients/groups/RNG, corpus/hash mismatch, recurrence failure, nonfinite state, candidate-only concentration, update spike, or loss-EMA veto.
- During production, terminate for wrong/busy GPU, traceback, OOM/resource error, nonfinite state, reset count other than 59, target/lifecycle assertion, no bounded progress for 120 seconds, or timeout.
- Do not abort for low intermediate accuracy or slow first-weak recovery; the complete run is required to evaluate whether inherited velocity was useful.
- One valid completion only. One identical retry is allowed solely for a documented external infrastructure failure preventing a valid summary. Never rescue with partial scaling, deletion, selected layers, delayed/repeated reset, LR/warmup/clipping changes, extra evaluation, or another seed.

## Verification Protocol

### Verification Procedure
1. **Baseline/source (10s):** require results-index baseline 94.15 at `7c1e7d8`, pristine baseline `train.py`, and only untracked `data/`; preserve it.
2. **Static/quality (60s):** run compile, Ruff, format, pre-commit, diff, scope, and AST checks. Require one helper/call; call after evaluation and weak-loader construction but before the first weak iterator; no evaluator/model/schedule change; exactly 1,073,962 parameters.
3. **State/algebra (120s):** after one accepted strong update, require all 59 FP32 matching buffers. Reset candidate and require exact zero buffers while parameters, model buffers, gradients, optimizer groups, CPU/CUDA RNG, and logits remain bitwise unchanged. On analytic tensors require the first reset update equals `w-lr*(g+lambda*w)` and later updates equal installed SGD at `atol=1e-7, rtol=1e-6`. Treat a second reset call as controller failure.
4. **Copied-state safety (300s):** require strong corpus SHA `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946` (200 records) and weak corpus SHA `ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032` (64 hard records). Train one source through strong records at LR 0.1, clone model/BN/optimizer/RNG exactly, reset candidate only, then replay weak records at LR 0.01. Serialize before vetoes. Require finite state; exact BN counters/corpus; no candidate >95% class share while control <=95%; candidate terminal loss EMA <=1.5x control; no update >25% of pre-update parameter norm; candidate/control update <=5; candidate update <=5x its preceding 16-step median; and recurrence checks. Record momentum-difference decay, first-update cosine/ratio, losses, histograms, and norms.
5. **Environment/run (595s):** confirm one idle H20, no stale log, exact tracked diff, imported `TIME_BUDGET_S==300`, and imported expected worker count `NUM_WORKERS`. Execute once with the registered timeout/redirection. Require exit zero; one finite ten-field summary; `TIME_BUDGET_S<=training_seconds<TIME_BUDGET_S+1`; `total_seconds<595`; 1,073,962 parameters; one transition near 80%, `workers_stopped==NUM_WORKERS`, exactly 59 reset buffers, 45-55% strong CutMix, hard weak targets, first weak LR near 0.01, and no fatal signal. The copied-state controller must show the reset-bearing first weak step remains far below the one-second final-overshoot allowance. Production steps are informational, not a retry/discard gate.
6. **Evaluator/metric (20s):** require exactly EXP010's 19 unique evaluator epochs, at most one per epoch, unchanged early checkpoints, and terminal coverage. Any count mismatch is invalid because `best_test_acc` is a maximum over looks. `best_test_acc>=94.25%` with integrity is improvement; any valid lower value is no-improvement. A 94.25-94.35 pass is formally valid but weak single-seed evidence.
7. **Mechanism diagnostics (20s):** compare switch with 89.73%, first weak with 93.16%, final NLL with 0.1934, best/final with 94.15%, and informational steps with 26,898 from EXP010. Slower initial recovery suggests inherited velocity was useful; improved NLL/sustained tail supports stale-state removal. Diagnostics cannot override the metric.

### Informational Metrics (Optional)
- Copied-state buffer/update/cosine/concentration/loss series: serialized safety report.
- Production summary, trajectory, reset/lifecycle/CutMix provenance: targeted `run.log` parsing copied into `03-execute.md` before log removal.

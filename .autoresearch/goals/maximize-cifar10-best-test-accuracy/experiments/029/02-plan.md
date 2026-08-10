# Plan EXP-029: Conv2d-Weight-Only Data-Gradient Centralization
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement the isolated projection
- [x] Create experiment branch `autoresearch/maximize-cifar10-best-test-accuracy-029` from the 94.15% integration baseline `7c1e7d8`; verify accepted `train.py` identity before editing.
- [x] Modify only tracked `train.py`: add the reviewed Conv2d-gradient helper and call it exactly once after every backward and before unchanged SGD.
- [x] Preserve model/initialization, optimizer construction/state semantics, all-parameter coupled decay, data/target curriculum, schedule, evaluator, seed, precision, timer, and worker lifecycle.
- [x] Verify syntax, Ruff, whitespace, scope, helper eligibility, and source invariants; `git diff --name-only` must contain only `train.py`.

### Milestone 2: Prove projection, recurrence, and trajectory safety
- [x] Create ignored `experiments/029/preflight_gc.py`, prepending the project root before importing `train` and using deterministic CUDA with `CUBLAS_WORKSPACE_CONFIG=:4096:8` supplied before process start.
- [x] Prove exact 19-Conv eligibility, correct `(1,2,3)` projection, FP64-reference agreement, idempotence, non-increasing raw Conv-gradient norm, untouched FC/BN gradients, unchanged parameters/RNG before SGD, and exact four-step `P(g)+lambda*w` momentum recurrence.
- [x] Reuse without regeneration the registered 200-batch strong and 64-batch weak corpora after validating both file and tensor digests, schemas, target ranks, and immutability.
- [x] Run byte-aligned accepted and GC models through all 264 exact batches, save full evidence before assertions, and require finite state, correct BN counters, no candidate-only class share above 95%, phase loss-EMA ratios at most 1.5, whole-model update ratios at most 2.0, no recent-median update spike above 5.0, and unchanged corpus/start hashes.
- [x] Report raw/projected/removed-mean norms and the removed-mean fraction by stage and phase as a mechanism read. A tiny fraction does not veto safety or authorize a rescue; it prospectively explains a possible null accuracy result.

### Milestone 3: Establish fixed-budget cost viability
- [x] Create ignored `experiments/029/timing_gc.py` whose candidate calls the production `train.centralize_conv_weight_gradients`; do not duplicate the projection implementation.
- [x] On one idle H20, run one unscored conditioning process and five counterbalanced fresh-process control/candidate pairs using real eight-worker production loaders, 100 warmups, and at least 1,000 synchronized measured steps per arm across registered strong-hard, strong-soft, and weak-hard paths.
- [x] Save raw trials before assertions. Require aggregate candidate/control counted-step ratio at most 1.01, every pair at most 1.04, per-arm CV below 3%, peak allocation below 650 MiB, valid loader delivery/lifecycle, integrated wall/count at most 1.07, and projected total below 540 seconds. Report the historical step projection but let the fresh ratio govern the abort. **Triggered:** aggregate 1.019749 and maximum pair 1.042778; production skipped.

The 1% gate is intentionally strict and a timing veto is a likely informative outcome: 38 dependent small GPU operations may cost 2-4% on this short step. Building and running the timing gate is still necessary because fixed-budget attribution requires establishing whether the literal minimal implementation preserves exposure; a likely veto does not authorize a different implementation after measurement.

### Milestone 4: Run the scored candidate once
- [ ] Reconfirm baseline 94.15, only `train.py` tracked-modified, one idle 97,871-MiB H20, and no stale root `run*.log`.
- [ ] Launch exactly once as `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`; monitor process/GPU state, log size, and only short tails on failure.
- [ ] Require exit zero, one finite ten-field summary, 300.0-301.0 counted seconds, total below 600 seconds, 1,073,962 parameters, one 80% switch with eight workers stopped, approximately half strong CutMix, hard weak-tail targets, and at most one evaluation per epoch.

### Milestone 5: Verify and hand off
- [ ] Query the moving baseline again and require `best_test_acc >=94.25%` for improvement; never rerun a valid completion.
- [ ] Record implementation, commands, corpus/report/log hashes, hardware, preflight/timing evidence, full summary, switch/first-weak trajectory, evaluation count, and formal status in `experiments/029/03-execute.md`.
- [ ] Keep `run.log` until analysis captures it; analysis owns indexing, verdict, code retention/revert, log removal, and loop routing.

## Code Changes

- **`train.py` — helper**: Add `@torch.no_grad() def centralize_conv_weight_gradients(model):` which iterates `model.modules()`, selects exactly `nn.Conv2d`, requires `module.weight.grad` to exist, and performs `grad.sub_(grad.mean(dim=(1, 2, 3), keepdim=True))`. Eligibility is module-type based so `fc.weight`, BN affine tensors, and biases remain untouched.
- **`train.py` — call site**: Insert `centralize_conv_weight_gradients(model)` exactly between existing `loss.backward()` and `optimizer.step()`. It stays inside `t0`/CUDA synchronization so all 19 reductions and 19 subtractions count against the fixed training budget.
- **`train.py` — preserved optimizer**: Keep the `optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)` constructor byte-for-byte accepted. Thus Conv recurrence is `P(g_t)+1e-4*w_t` followed by ordinary momentum, while every non-Conv recurrence is unchanged. Do not centralize decay-augmented directions or momentum buffers.
- **`train.py` — no production instrumentation**: Add no counters, tensor-norm reductions, synchronizing finite checks, extra evaluations, parameters, buffers, groups, or RNG calls. Mechanism diagnostics belong only in ignored preflight artifacts so timing and production test the same minimal helper.
- **Ignored experiment-local artifacts**: Controllers, JSON/text reports, and reused-corpus metadata may be added only below `experiments/029/`; these remain ignored and are never committed.

## Configuration Changes

- Eligible gradients: none -> all 19 `nn.Conv2d.weight` loss gradients.
- Projection dimensions: none -> `(1, 2, 3)` for `[C_out,C_in,kH,kW]`, independently zero-centering each output filter.
- Projection strength/cadence: none -> full subtraction exactly once every successful training step from step 1 through completion; there is no tunable coefficient or phase switch.
- Decay order: accepted `g+lambda*w` -> Conv-only `P(g)+lambda*w`; non-Conv parameters remain accepted. This intentionally preserves all-parameter coupled `1e-4` decay and does not claim the paper's `P(g+lambda*w)` invariant-mean theorem.
- Exposure floor: the proposal's 26,629-step prediction is a hypothesis discriminator, not a post-run validity rule. The reviewed pre-production 1.01 timing gate is binding; after a valid fixed-budget production completion, the goal's accuracy/runtime conditions determine the formal verdict even if actual steps unexpectedly miss the prediction.
- Every accepted setting otherwise remains fixed: width-2 postactivation ResNet-20, batch 128, seed 42, FP32/default TF32, LR 0.1 through 80% then 0.01-to-1e-4 cosine, momentum 0.9, N1/M7 plus probability-0.5 alpha-1 CutMix through 80%, crop/flip hard-label tail, checkpoints, and workers.

## Execution Environment

- Method: local controllers followed by one local production process from the project root.
- Resources: exactly one idle NVIDIA H20 near 97,871 MiB; existing eight DataLoader workers; no dependency/package changes.
- Estimated runtime: semantic/trajectory preflight 1-3 minutes, fresh paired timing 2-5 minutes, production about 5.5 minutes; every command is bounded and production must finish within 10 minutes.
- Log output: production stdout/stderr only in root `run.log`; controller stdout/stderr and serialized reports in ignored experiment directory 029. Never use `tee` or stream the full production log.
- Tool skill: none; execution is local rather than a Volcano job.
- Infrastructure safeguards: controller imports use resolved project root; deterministic CUDA starts with the required cuBLAS workspace environment; a device-conditioning process precedes paired timing; all reports are written and fsynced before gate assertions.

## Abort Criteria

- Abort before GPU work for wrong branch/baseline/source, any tracked change outside `train.py`, stale/mismatched corpus digest, wrong eligible tensor/count/dimensions/call placement, changed accepted optimizer/decay/model/data/schedule/evaluator/timer/precision/seed/lifecycle, or unavailable required API/command.
- Abort before timing for projection/reference or recurrence error above `1e-7` absolute / `1e-6` relative; projection non-idempotence/norm increase; any FC/BN gradient mutation; parameter/RNG change before SGD; non-finite state; invalid BN counter/variance; corpus/start mutation; candidate-only class share above 95% while control is at most 95%; phase loss-EMA ratio above 1.5; whole-model candidate/control update ratio above 2.0; candidate Conv update above 25% of its pre-update tensor norm; or candidate update above 5x its prior 16-step median.
- The 264-step corpus gate is an early-trajectory screen, not proof of full-run stability. It directly covers the locally observed failure horizon (EXP020 step 11, EXP022 steps 7/13, EXP028 step 3), while production monitoring remains authoritative for any later instability.
- A small removed-common-mode fraction is not a safety abort and cannot justify changing strength, coverage, dimensions, or decay ordering. It is a preregistered mechanism discriminator for analysis.
- Abort before production if fresh aggregate timing ratio exceeds 1.01, any pair exceeds 1.04, CV/memory/loader/lifecycle/wall gates fail, projected total is at least 540 seconds, or the H20 is contended. Do not rescue with a fused/custom kernel, layer subset, coefficient, delayed/tail-only GC, clipping, warmup, another corpus/seed, or threshold relaxation.
- Stop production for non-finite loss, CUDA/OOM/resource error, worker failure, prolonged absent progress with no GPU activity, or the 595-second timeout. Do not rerun a valid seed-42 completion; a finite below-threshold result is no-improvement.

## Verification Protocol

### Verification Procedure

1. Confirm baseline, source identity, and corpora (timeout: 60 seconds):
   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv
   git rev-parse HEAD
   git show 7c1e7d8:train.py | sha256sum
   sha256sum train.py
   sha256sum .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/022/preflight-corpus.pt .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/028/weak-corpus.pt
   uv run python -c 'from prepare import NUM_WORKERS, TIME_BUDGET_S; print(NUM_WORKERS, TIME_BUDGET_S)'
   ```
   Baseline must be 94.15 at `7c1e7d8`; pre-edit source hashes must both be `29c152...e6aa`; corpus file hashes must be `e04dc2...8946` and `ffefe9...5032`; imported protocol constants must print `8 300`. The controller independently checks registered tensor digests `424204...ad40` and `df97b0...eae`.

2. Verify implementation and tracked scope (timeout: 60 seconds):
   ```bash
   uv run python -m py_compile train.py
   uv run ruff check train.py
   git diff --check
   git diff --name-only
   git diff -- train.py
   ```
   All commands must exit zero, name-only output must be exactly `train.py`, and diff inspection must prove one helper/call, unchanged accepted contracts, and no production diagnostic overhead.

3. Confirm hardware before every GPU controller/production command (timeout: 30 seconds):
   ```bash
   nvidia-smi --query-gpu=name,memory.total,compute_mode --format=csv,noheader
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
   ```
   Pass only with exactly one visible NVIDIA H20 near 97,871 MiB and no competing compute process. A mismatch is infrastructure failure, not an experiment result.

4. Run semantic and exact-corpus safety preflight (timeout: 240 seconds):
   ```bash
   CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout 240s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/029/preflight_gc.py
   ```
   Pass only if `preflight-report.json` was serialized/fsynced first and records every static/algebraic check plus 200 exact strong and 64 exact weak aligned steps; all finite/BN/corpus/start checks; zero candidate-only concentration events; loss-EMA ratios at most 1.5; maximum paired update ratio at most 2.0; maximum recent-median ratio at most 5.0; and raw/projected/removed-mean fractions by stage/phase. A nonzero exit is a research veto unless evidence isolates a controller bug independent of candidate behavior; never regenerate/reorder a corpus or change candidate semantics.

5. Run fresh paired timing (timeout: 360 seconds):
   ```bash
   timeout 360s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/029/timing_gc.py
   ```
   Pass only if `timing-report.json` was serialized before assertions and contains one conditioner plus five counterbalanced pairs, real production loaders, at least 100 warmups/1,000 measured steps per arm, separate hard/soft/weak data, aggregate ratio at most 1.01, every pair at most 1.04, per-arm CV below 3%, peak below 650 MiB, warmed delivery at least 1.2x consumption, iterator wait med/p95 below 10%/20% of candidate step, weak rebuild below five seconds, integrated wall/count at most 1.07, and projected total below 540 seconds. Report `floor(26898*control_mean/candidate_mean)` as a historical consistency estimate; the fresh ratio is authoritative.

6. Ensure no stale log and execute exactly once (timeout: 600 seconds):
   ```bash
   find . -maxdepth 1 -type f -name 'run*.log' -print
   timeout --kill-after=5s 595s uv run train.py > run.log 2>&1
   ```
   The first command must print nothing. The run must exit zero; do not launch another full run after a valid completion.

7. Parse the final summary without streaming the log:
   ```bash
   grep '^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:' run.log
   ```
   If incomplete, inspect only `tail -n 50 run.log`. Parse numerically. Protocol validity requires all ten fields finite, `300.0 <= training_seconds <= 301.0`, `total_seconds <600`, `num_params ==1073962`, and correct run integrity. Improvement requires `best_test_acc >=94.25`; 94.15-94.24 is no-improvement. The frozen CIFAR-10 test set has 10,000 examples, so accuracy is quantized in exact 0.01-point increments: 94.25 means 9,425 correct, exactly ten more than the 9,415-correct baseline, with no hidden 94.245 rounding case. Compare actual steps with the 26,629 hypothesis, but do not override a protocol-valid metric improvement if exposure unexpectedly falls below that advisory floor.

8. Verify protocol and interpret mechanism: exactly one switch near 80%, eight workers stopped, 45-55% strong CutMix, exclusively hard weak targets, no repeated epoch/evaluation and no more than the accepted 19 looks, seed 42, and no rerun. Record switch accuracy against 89.73%, first weak against 93.16%, final NLL against 0.1934, best/final gap, steps, epochs, VRAM, counted/wall time, evaluation count, report/corpus/log hashes, and preflight removed-mean fraction by stage/phase. A tiny removed fraction plus a miss is near-inertness; a lower switch plus a miss means useful high-LR drift was removed; neither permits a GC variant inside EXP029.

### Informational Metrics (Optional)

- final_test_acc, final_test_loss, training_seconds, total_seconds, startup_seconds, peak_vram_mb, num_epochs, num_steps, and num_params: corresponding `run.log` summary lines.
- mechanism: raw/projected/removed Conv-gradient norms and fractions by stage/phase, filter-mean drift, update/momentum ratios, concentration history, strong/weak loss EMAs from `preflight-report.json`.
- trajectory: switch/first-weak accuracy, best/final gap, final NLL, CutMix rate, and evaluation count from `run.log`.

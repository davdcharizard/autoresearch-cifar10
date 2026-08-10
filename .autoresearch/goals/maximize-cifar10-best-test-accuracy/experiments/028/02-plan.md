# Plan EXP-028: Signal-Scale-Matched Positive-Negative Momentum
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement the isolated PNM recurrence
- [x] Create experiment branch `autoresearch/maximize-cifar10-best-test-accuracy-028` from integration commit `7c1e7d8`; verify `train.py` is byte-identical to that baseline before editing.
- [x] Modify only tracked `train.py`: add a local `ScaleMatchedPNM` optimizer, replace the accepted SGD constructor with it, and add final non-evaluative optimizer diagnostics.
- [x] Preserve the accepted model, initialization, seed, batch/data curriculum, worker lifecycle, evaluator, elapsed-time LR schedule, timer boundaries, precision, and all-parameter coupled `1e-4` decay.
- [x] Run syntax, Ruff, whitespace, scope, and source-invariant checks. The only tracked modified path may be `train.py`.

### Milestone 2: Prove algebra and immutable-corpus trajectory safety
- [x] Create ignored controller `experiments/028/preflight_pnm.py`, resolving and prepending the project root before importing `train` and setting `CUBLAS_WORKSPACE_CONFIG=:4096:8` before deterministic subprocess startup.
- [x] Test scalar/manual and multi-tensor recurrences, first-step equality, changing-gradient behavior, coupled-decay ordering, parity, LR changes, state save/load, inactive-buffer preservation, RNG neutrality, and absence of gradient mutation.
- [x] Reuse the immutable 200-batch EXP022 strong corpus only after its SHA-256 exactly matches `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946`; do not regenerate or filter it. Materialize one 64-batch weak hard-label corpus once, hash it, and persist it before either aligned arm runs.
- [x] Run byte-aligned accepted-SGD and PNM models through all 200 strong and 64 weak batches, recording each arm's loss, prediction histogram, update/gradient/state norms, parity/scale, BN state, and finite-state checks before evaluating vetoes.
- [x] Abort on any candidate-only class concentration above 95%, median changing-gradient candidate/control update ratio above 1.30, any candidate update above 5x control or 10x its preceding 16-step median, phase-terminal loss EMA above 1.5x control, or any algebra/state/integrity failure. **Triggered:** concentration and 12.35x paired-update spike; timing/production skipped.

### Milestone 3: Establish counted-cost viability
- [ ] Create ignored `experiments/028/timing_pnm.py` using production `train.ScaleMatchedPNM` for the candidate and installed `torch.optim.SGD` for control; do not maintain a second candidate implementation.
- [ ] Confirm an idle H20, run one unscored conditioning subprocess, then five alternating fresh-process control/candidate pairs with identical real production loaders, 100 warmups, and at least 1,000 synchronized measured steps per arm split into registered strong and weak regions.
- [ ] Save all raw trials before assertions. Require median paired weighted step ratio at most 1.03, every pair at most 1.06, each arm's trial-mean CV below 3%, peak allocation below 650 MiB, stable state/worker lifecycle, integrated wall/count at most 1.10, and projected total runtime below 540 seconds. Report the corresponding historical exposure projection, but let the fresh paired ratio—not that historical constant—govern the abort.

### Milestone 4: Run production exactly once
- [ ] Reconfirm the moving baseline is 94.15, exactly one idle 97,871-MiB H20 is visible, tracked scope is clean except `train.py`, and no stale root `run*.log` exists.
- [ ] Run one seed-42 production candidate with `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`; monitor only process/GPU status, log size, and short tails on failure.
- [ ] Require exit zero, one finite ten-field summary, 300.0-301.0 counted seconds, total below 600 seconds, 1,073,962 parameters, one 80% augmentation/target switch with eight workers stopped, hard weak targets, and at most one evaluation per epoch. Record whether actual exposure reaches the hypothesized 26,091-step floor, but do not override the goal's metric verdict after a protocol-valid fixed-budget run.

### Milestone 5: Record and hand off the result
- [ ] Parse `best_test_acc`, compute delta from 94.15, and require at least 94.25 for improvement; never rerun a valid completion.
- [ ] Record the implementation, commands, preflight/timing hashes, failures or adjustments, hardware, full summary, switch/first-weak trajectory, PNM diagnostics, and formal status in `experiments/028/03-execute.md`.
- [ ] Retain `run.log` until analysis has captured and hashed it; analysis owns indexing, verdict, code retention/revert, log removal, and loop routing.

## Code Changes

- **`train.py` — optimizer definition**: Add `ScaleMatchedPNM(optim.Optimizer)` under `@torch.no_grad()`. It has one parameter group with `lr`, `momentum=0.9`, and coupled `weight_decay=1e-4`; group state stores one global `pnm_step`. Each parameter state lazily owns two zero FP32 buffers with the same shape/device as the parameter. No RNG is consumed.
- **`train.py` — exact update**: On completed step `t = pnm_step + 1`, form out-of-place decay-augmented directions `d = grad + weight_decay * parameter` with `torch._foreach_add`; update only the odd or even buffer as `current = 0.81*current + 0.19*d`; form `raw = 2*current - previous`; compute the registered Python-float scale `q_sgd(t)/q_pnm(t)`; and update parameters with one `torch._foreach_add_` coefficient `-lr*scale/sqrt(5)`. Increment parity once only after the successful update. The other stream must remain byte-identical.
- **`train.py` — optimizer contract**: Reject closures, multiple parameter groups, sparse/missing/non-FP32 gradients, or a momentum outside `(0,1)`; inherit `zero_grad`. Do not mutate `.grad`, clip/norm-match updates, use Nesterov, reset state at evaluation or the 80% transition, or add synchronous per-parameter finite checks to production. This explicitly refines the proposal's conceptual per-step finite requirement: preflight performs synchronized finite scans and the final run must emit finite losses/metrics, while both timing and production omit a costly device-synchronizing scan and execute the same recurrence.
- **`train.py` — integration**: Replace only `optim.SGD(...)` with `ScaleMatchedPNM(...)`; keep the existing per-group LR assignment so elapsed-time scheduling remains authoritative. Keep `optimizer.step()` in its accepted position before the existing CUDA synchronization, ensuring all PNM cost is counted.
- **`train.py` — diagnostics**: After training/evaluation has completed, print `pnm_steps`, the first registered scale, and the final registered scale. Do not add evaluations, alter summary fields, or compute large tensor diagnostics in production.
- **Ignored experiment-local artifacts**: The execute phase may add controllers, PT corpora, JSON reports, and timing text only under `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/028/`. These are research metadata and must never be committed.

## Configuration Changes

- Optimizer: PyTorch SGD momentum -> signal-scale-matched beta0=1 PNM.
- PNM alternating-buffer decay `rho`: absent -> `MOMENTUM**2 = 0.81`.
- PNM buffer injection: absent -> `1-rho = 0.19`.
- PNM combination/noise normalization: absent -> `(2*current - previous)/sqrt(5)`.
- PNM signal scale at step `t`: absent -> `((1-MOMENTUM**t)/(1-MOMENTUM)) / ((2*(1-rho**ceil(t/2))-(1-rho**floor(t/2)))/sqrt(5))`. It begins near 5.884389 and converges to 22.360680; it is fixed analytically and never fit to observed gradients.
- Coupled decay: remains `d = grad + 1e-4*parameter` before the PNM recurrence for every trainable parameter, including BN affine and biases.
- All accepted settings remain unchanged: width-2 postactivation ResNet-20; batch 128; seed 42; FP32/default TF32; LR 0.1 through 80%, then 0.01-to-1e-4 cosine; N1/M7 plus probability-0.5 alpha-1 CutMix through 80%; crop/flip hard-label tail; existing evaluator checkpoints and worker lifecycle.

## Execution Environment

- Method: local single-GPU controllers followed by one local production command from the project root.
- Resources: exactly one idle NVIDIA H20 with approximately 97,871 MiB memory; existing eight DataLoader workers; no package or dependency changes.
- Estimated runtime: algebra/trajectory preflight 1-3 minutes, paired timing 2-4 minutes, production about 5.5 minutes; every individual command has a timeout and production must remain below 10 minutes.
- Log output: production stdout/stderr only to root `run.log`; controller output and serialized reports stay in ignored experiment directory 028. Never use `tee` or stream the full production log.
- Tool skill: none; this is a local run rather than a remote Volcano job.
- Infrastructure safeguards: controllers prepend the project root for imports; deterministic CUDA subprocesses receive `CUBLAS_WORKSPACE_CONFIG=:4096:8`; one unscored device conditioner precedes paired timing; exact tensors, reports, and hashes are serialized before gate assertions.

## Abort Criteria

- Abort before GPU work if baseline/branch/source identity differs, tracked scope includes anything besides `train.py`, the immutable EXP022 corpus hash differs, or an expected command/script/API is unavailable.
- Abort before timing if the optimizer/manual recurrence differs beyond `1e-6` relative or `1e-6` absolute (first-step parameter equality additionally within `1e-7` absolute), parity increments incorrectly, inactive state changes, gradients are mutated, RNG changes, state save/load diverges, decay/LR semantics differ, or any tensor/state is non-finite.
- Abort before timing if the 200-strong plus 64-weak aligned trajectory has any candidate-only class share above 95% while control is at most 95%; median candidate/control **total update-norm** ratio above 1.30; any candidate/control update ratio above 5; any candidate update above 10 times its preceding 16-step median; phase-terminal candidate/control loss-EMA ratio above 1.5; invalid BN variance/counter; or corpus/initial-state mutation. The 1.30 gate does not require PNM's newest-gradient coefficient to be small: that coefficient approaches 3.8x, while the gate measures the norm of the complete coherent-plus-stochastic parameter update. Lower candidate loss cannot override a veto.
- Abort before production if the fresh median paired timing ratio exceeds 1.03, any pair exceeds 1.06, or another registered CV/memory/wall/worker gate fails; the historical step projection is a reported consistency check rather than a second governing threshold. Also abort for GPU contention or projected total at least 540 seconds. Do not rescue with another PNM scale, beta, warmup, clipping, ordinary-momentum tail, decoupled decay, regenerated corpus, fused extension, or threshold relaxation.
- Stop production on non-finite loss, CUDA/OOM/resource error, worker failure, prolonged missing progress with no GPU activity, or the 595-second command timeout. Do not rerun a valid seed-42 completion; a finite below-threshold result is no-improvement.

## Verification Protocol

### Verification Procedure

1. Confirm baseline, integration source, and immutable corpus (timeout: 60 seconds):
   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv
   git rev-parse HEAD
   git show 7c1e7d8:train.py | sha256sum
   sha256sum train.py
   sha256sum .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/022/preflight-corpus.pt
   ```
   Before implementation the two `train.py` hashes must match `29c152320623c28518d93eae27c99cdacd151715e8b7dcfc22288919e0c6e6aa`; the baseline output must be 94.15 at `7c1e7d8`; the corpus digest must be the registered `e04dc...8946`. After branch creation, `HEAD` must descend from `7c1e7d8`.

2. Verify implementation and scope (timeout: 60 seconds):
   ```bash
   uv run python -m py_compile train.py
   uv run ruff check train.py
   git diff --check
   git diff --name-only
   git diff -- train.py
   ```
   Every command must exit zero; the name-only output must be exactly `train.py`. Diff inspection must show no model/data/evaluator/schedule/timer/seed/precision change and exactly one production optimizer step per training batch.

3. Confirm hardware immediately before every GPU controller and production command (timeout: 30 seconds):
   ```bash
   nvidia-smi --query-gpu=name,memory.total,compute_mode --format=csv,noheader
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
   ```
   Pass only with exactly one visible NVIDIA H20 near 97,871 MiB and no competing compute process. Treat mismatch/contention as infrastructure failure, not an experiment verdict.

4. Run algebra and immutable-corpus safety preflight (timeout: 240 seconds):
   ```bash
   CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout 240s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/028/preflight_pnm.py
   ```
   Pass only if `preflight-report.json` was saved first and records: 64-step constant-direction equality; manual changing-gradient equality through at least steps 1-16 and 200-201; LR/decay/state-load/parity/gradient/RNG checks; 200 exact strong plus 64 exact weak aligned model steps; all finite/corpus/BN checks; zero concentration events; phase loss-EMA ratios at most 1.5; median update ratio at most 1.30; maximum paired update ratio at most 5; and maximum recent-median ratio at most 10. A nonzero exit is a research veto unless inspection proves an infrastructure/controller bug independent of candidate behavior; never regenerate the corpora or rerun to seek a passing trajectory.

5. Run paired H20 timing (timeout: 300 seconds):
   ```bash
   timeout 300s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/028/timing_pnm.py
   ```
   Pass only if `timing-report.json` was serialized before assertions and contains the conditioning record, five alternating pairs, at least 100 warmups/1,000 measured steps per arm, separate strong/weak stage times, weighted median paired ratio at most 1.03, every pair at most 1.06, per-arm CV below 3%, peak allocation below 650 MiB, stable workers/state, weak rebuild below five seconds, integrated wall/count at most 1.10, and projected total below 540 seconds. Report `floor(26898*control_mean/candidate_mean)` as a historical consistency estimate, but the fresh ratio is the authoritative cost gate. The controller must use the production optimizer without per-step finite scans.

6. Ensure no stale log and execute once (timeout: 600 seconds):
   ```bash
   find . -maxdepth 1 -type f -name 'run*.log' -print
   timeout --kill-after=5s 595s uv run train.py > run.log 2>&1
   ```
   The first command must print nothing. The production command must exit zero, and no second full run is permitted after a valid completion.

7. Parse the required summary without streaming the log:
   ```bash
   grep '^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:\|^pnm_steps:\|^pnm_first_scale:\|^pnm_final_scale:' run.log
   ```
   If fields are empty/incomplete, inspect only `tail -n 50 run.log`. Parse numerically. Protocol validity requires exit zero, all ten standard fields finite, `300.0 <= training_seconds <= 301.0`, `total_seconds < 600`, `pnm_steps == num_steps`, `num_params == 1073962`, and finite scale diagnostics. Improvement requires `best_test_acc >= 94.25`; 94.15-94.24 is no-improvement, not a rounding rescue. Compare `num_steps` with 26,091 to test the exposure hypothesis, but a protocol-valid accuracy improvement remains an improvement even below that advisory floor. The printed scales are production-integrity diagnostics only; independent recurrence/formula verification comes from preflight, not from comparing code-generated values to themselves.

8. Verify trajectory and protocol integrity from the log and saved reports: exactly one switch near 80%; eight workers stopped; all post-switch targets hard; approximately half of strong batches CutMix; no more than one evaluation per epoch and no duplicate epoch; no candidate-specific extra evaluation; seed 42; and no full-run retry. Record switch accuracy against 89.73%, first weak accuracy against 93.16%, final NLL against 0.1934, best/final gap, PNM preflight norm distribution, steps, runtime, VRAM, corpus/report/log hashes, and evaluation count. A switch in `[88.73,89.73)` followed by a miss is the predeclared harmful-noise signature; diagnostics never alter the formal metric verdict.

### Informational Metrics (Optional)

- final_test_acc, final_test_loss, training_seconds, total_seconds, startup_seconds, peak_vram_mb, num_epochs, num_steps, and num_params: corresponding final `run.log` summary lines.
- optimizer integrity: `pnm_steps`, first/final analytic scale, parity/buffer/state/load checks from `run.log` and `preflight-report.json`.
- mechanism: strong/weak loss EMA, changing-gradient candidate/control update-ratio quantiles, odd/even state norms, concentration events, switch/first-weak accuracy, final/best gap, and NLL from preflight artifacts and `run.log`.

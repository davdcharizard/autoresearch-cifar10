# Plan EXP-035: Weaker Alpha-0.1 Batch-Shared Mixup
- **Created**: 2026-07-27

## Milestones

### Milestone 1: Implement the exact one-constant treatment
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-035` from accepted commit `67c8e98`; modify only `train.py` in production and keep `prepare.py`/evaluator frozen.
- [x] Change exactly `MIXUP_ALPHA = 0.2` to `MIXUP_ALPHA = 0.1`; preserve batch-shared CUDA sampling, pairing, target weighting, the strict 65% cutoff, and every other accepted source line.

### Milestone 2: Prove distribution, RNG, and training semantics
- [x] Create ignored `experiments/035/preflight.py` with an independent `git show 67c8e98:train.py` oracle and guarded evaluator/test data; prove byte-equal initial model state, construction RNG, optimizer groups, loader policy, schedule, 987,098 parameters, and exact one-line scope.
- [x] In a disposable process, verify at least 100,000 alpha-0.1 CUDA draws against preregistered moments/masses and directionally against reset alpha-0.2 draws; prove candidate replay from a restored CUDA state, one scalar per batch, pixel/target alignment, finite paired update, exact hard-path identity, and strict cutoff behavior.

### Milestone 3: Require preserved H20 exposure
- [x] Run balanced accepted/candidate complete-body H20 timing separately for early mixup and hard-label regimes with at least 20 warmups and four counterbalanced windows of at least 50 steps per arm; print raw windows before assertions and require every population CV <=5%.
- [x] Compute fixed-time retention from regime medians and require retention >=0.9774 plus projected passes >=130.0. Require accepted/candidate hard-path medians to differ by no more than 2%; treat a stable miss as closure of this exact alpha-0.1 implementation and never rerun or relax the gate.

### Milestone 4: Run and classify the sole fixed-seed score
- [x] Reconfirm baseline 94.32 at `67c8e98`, one idle H20, local CIFAR-10, frozen evaluator, exact scope, no stale `run.log`, and passing gates; execute exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Require exit 0, one finite summary, 300.0-300.1 counted seconds, total <600, 987,098 parameters, correct ordered transitions, unique accepted-cadence evaluations, and no numerical/CUDA/worker errors. Record realized passes; a valid completion below 130 remains the sole score but is operationally inconclusive.
- [x] Classify improvement solely by `best_test_acc >=94.42%`; report `final_test_acc >=94.32%` and `final_test_loss <=0.2523` only as corroboration. A valid normal-exposure miss closes immediate batch-shared mixup-strength tuning without adjacent alpha, cutoff, RNG, sharing, or seed rescue.

## Code Changes
- **`train.py` / one constant only**: set `MIXUP_ALPHA = 0.1`. Keep the existing symmetric CUDA FP32 concentration tensor, single coefficient shared over the batch, device `randperm`, mixed image expression, two aligned cross-entropies, and `<0.65` activation test byte-identical.
- **`.autoresearch/.../experiments/035/preflight.py`**: ignored verification-only harness for exact source/construction checks, distribution and replay semantics, cutoff/hard-path checks, and balanced complete-step H20 timing. It must replace `prepare.Eval` before importing accepted/candidate modules, forbid `CIFAR10(train=False)`, use fixed synthetic pinned batches, and never create `run.log`.

## Configuration Changes
- Symmetric mixup concentration: `Beta(0.2, 0.2) -> Beta(0.1, 0.1)` through the first 65% of counted training time.
- Intended distribution: mean remains 0.5; theoretical variance rises from 0.178571 to 0.208333; mass outside `[0.1,0.9]` rises from 67.34% to 81.28%.
- Accepted model/data/optimization: unchanged `(2,2,3)` WRN, 987,098 FP32 parameters, batch 256, LR `0.2 -> 0.002`, 5% warmup, Nesterov momentum 0.9, matrix-only decay `5e-4`, early worker-private N1/M5 RandAugment, crop/flip/normalization, eight persistent workers, seed 42, 300 counted seconds, and accepted evaluator cadence.

## Execution Environment
- Method: offline local semantic/distribution/timing preflight, then one local score only on pass; no network, remote, installs, W&B, GitHub, `gh`, fetch, push, or PR action.
- Resources: one NVIDIA H20, local CIFAR-10, installed environment, eight persistent forkserver workers.
- Estimated runtime: preflight under 4 minutes; score about 345-360 seconds wall with a 600-second hard timeout.
- Log output: scored stdout/stderr only in root `run.log`, retained through analysis and then removed.
- Tool skill: none.

## Abort Criteria
- Abort before timing on any scope/frozen-file/syntax failure; construction/model/optimizer/loader/schedule mismatch; wrong topology/parameter count/dtype; wrong concentration shape/device/value; invalid empirical distribution; failed candidate replay; coefficient not batch-shared; image/target mismatch; non-finite update; hard-path mismatch; cutoff error; evaluator/test access; or altered worker/budget/evaluation source.
- Abort before scoring if any timing window is non-finite, any CV exceeds 5%, hard-path median ratio lies outside `[0.98,1.02]`, retention is below 0.9774, projected passes are below 130.0, or H20 memory is unsafe. Emit all raw values and projections before assertions; never repeat a stable miss.
- During score stop/classify on timeout, nonzero exit, OOM/worker/resource error, non-finite loss, no output for 60 seconds, malformed/missing/duplicate summary, wrong topology, invalid/repeated transition, duplicate evaluation epoch, or total >=600. Never rerun a valid completion or react to interim accuracy.

## Verification Protocol

### Verification Procedure

1. Query `exp-index.sh baseline`; require 94.32 at `67c8e98`, so the primary threshold is 94.42.
2. Within 30 seconds query `nvidia-smi`, local CIFAR presence, branch/status, `git diff --check`, `git diff --unified=0 67c8e98 -- train.py`, `git diff --exit-code 67c8e98 -- prepare.py`, and `uv run python -m py_compile train.py .../experiments/035/preflight.py`. Require one idle H20, only tracked `train.py`, and exactly `MIXUP_ALPHA = 0.2 -> 0.1`.
3. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/035/preflight.py semantics`. Guard `prepare.Eval` and test-data construction before importing either source. Require identical topology, byte-equal initial model tensors, optimizer-group membership/settings, loader constants/transforms, LR schedule outputs, post-construction CPU/CUDA RNG states, and 987,098 FP32 trainable parameters. Source comparison must prove only the concentration constant differs and all worker, budget, transition, evaluation, and summary code is unchanged.
4. On the H20 in the disposable preflight process, independently reset to the same fixed CUDA seed before drawing at least 100,000 coefficients per concentration. Print statistics before assertions. Require alpha 0.1 draws finite and in `[0,1]`, mean in `[0.495,0.505]`, population variance in `[0.203,0.214]`, `[0.2,0.8]` mass in `[0.115,0.127]`, and `<=0.1 or >=0.9` mass in `[0.806,0.820]`. Require greater variance/endpoint mass and lower central mass than alpha 0.2.
5. Save a candidate pre-call CUDA state and identical model/optimizer/input state; execute one production-equivalent early step twice after restoring it. Require exact equality of coefficient, permutation, mixed inputs, paired targets, loss, gradients, updated parameters, optimizer tensors, and post-call CUDA state. Prove the sampled coefficient is one scalar consistently applied to every image and to both target losses with a direct interpolation oracle. Do not require accepted/candidate CUDA states to match after their first concentration-dependent draw.
6. From cloned accepted/candidate model, optimizer, input, target, CPU RNG, and CUDA RNG state, execute the hard-label path and require bitwise-equal loss, gradients, parameter/optimizer update, and resulting RNG states. Probe progress immediately below, exactly at, and above 0.65; require mixup only below 0.65 and accepted-identical LR values throughout. Require a finite real-model early update and safe peak allocation.
7. Run `timeout 240s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/035/preflight.py timing`. On the idle H20 compare accepted and candidate modules from equal snapshots using fixed pinned batches. Measure early mixup and hard-label production bodies separately, including nonblocking H2D, LR write, zero-grad, concentration-dependent Beta draw/permutation/interpolation when active, forward, aligned paired or hard loss, finite guard, backward, Nesterov step, and final synchronization.
8. Use at least 20 warmups, then four windows of at least 50 measured steps per arm/regime ordered `accepted/candidate, candidate/accepted` and repeated. Give each paired window a distinct preregistered deterministic CUDA fixture/RNG state; restore equal model/optimizer fixtures within the pair but do not replay one identical draw segment across every pair. Define a window value as synchronized total elapsed milliseconds divided by its measured step count. Print every window mean, regime median over the four window means, population CV over those means, peak allocation, retention, and projected passes before assertions. Require all CVs <=0.05 and hard-path candidate/accepted median ratio in `[0.98,1.02]`. Compute `retention=(0.65/candidate_early_ms+0.35/candidate_hard_ms)/(0.65/accepted_early_ms+0.35/accepted_hard_ms)` and `projected_passes=133.00736*retention`; require retention >=0.9774 and projected passes >=130.0. No real-loader timing is needed because worker source, batch shape, and consumer structure are identical.
9. Reconfirm exact audit and one idle H20, remove stale `run.log`, execute exactly `timeout 600s uv run train.py > run.log 2>&1` once, record start/PID, and never launch a second valid score.
10. Parse with `rg`; require one finite summary, 300.0-300.1 counted seconds, total <600, 987,098 parameters, no traceback/OOM/non-finite/worker error, one mixup transition at the first step with pre-step counted time >=195 seconds, and one exhausted-epoch RandAugment transition at the first iterator boundary observed after the threshold. Usually mixup logs first; also accept RandAugment logging on an epoch-ending crossing step followed by mixup at the next epoch with the same completed-step count. In either ordering require source-faithful times, steps, and epoch boundary. Require evaluation epochs to equal `{5, 10, ...} union {final_epoch}` with no duplicate epoch and no more than one evaluation per epoch; the final epoch need not be distinct or partial.
11. Record realized exposure as `num_steps*256/50000`. A valid completed result below 130 passes remains a nonrepeatable goal result and closes exact alpha 0.1, but does not establish the intended normal-exposure strength comparison. A malformed or invalid run is a crash, not an accuracy miss.
12. Classify success only by `best_test_acc >=94.42%`. Separately report final accuracy versus 94.32, final loss versus 0.2523, best-final gap, steps/epochs/passes, evaluation count, transitions, VRAM, counted/wall time, timing retention, and final source audit. On a valid >=130-pass miss, retain alpha 0.2 and close alpha 0.05/0.15/0.25/0.3, cutoff changes, symmetrization, private-RNG realignment, per-example sampling, and another seed as adjacent rescues.

### Informational Metrics (Optional)
- Final summary: `run.log` - best/final accuracy, final loss, counted/total/startup seconds, epochs, steps, passes, VRAM, and parameters.
- Distribution: preflight stdout - alpha-0.1/0.2 means, variances, central/endpoint masses, replay hashes, and CUDA state checks.
- Timing: preflight stdout - raw early/hard windows, medians/CVs, hard-path ratio, retention/pass projection, and peak allocation.
- Mechanism: best/final/loss deltas from accepted 94.32/94.22/0.2523, best-final gap, and realized versus projected exposure.

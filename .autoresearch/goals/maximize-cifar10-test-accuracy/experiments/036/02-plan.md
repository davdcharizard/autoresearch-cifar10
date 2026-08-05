# Plan EXP-036: Scaled Pooled-Feature Residual MLP Head
- **Created**: 2026-07-27

## Milestones

### Milestone 1: Implement the fixed post-pooling treatment
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-036` from accepted commit `67c8e98`; modify only `train.py` in production and keep `prepare.py`/evaluator frozen.
- [x] Add fixed width 64, scale 0.1, and isolated seed 36036; construct a bias-free `128 -> 64 -> 128` ReLU branch after all accepted initialization under a restoring CPU RNG fork using CPU-only `torch.random.default_generator.manual_seed`, then add `0.1 * branch(pooled)` before the accepted classifier. Add no production diagnostic or other behavior.

### Milestone 2: Prove common-state, head, and training semantics
- [x] Create ignored `experiments/036/preflight.py` with an independent `git show 67c8e98:train.py` oracle and guarded evaluator/test data; prove all 987,098 common tensor bytes and post-construction global RNG states remain accepted, exact branch reconstruction under seed 36036, exact topology, 16,384 new parameters, and total 1,003,482.
- [x] Prove the candidate expression, direct-path identity, optimizer membership, RNG neutrality, finite nonzero first-step gradients/updates, accepted mixup/hard paths and temporal controls. Print initial branch/direct norm ratio, direct/candidate logit perturbation, and backbone/classifier/head gradient norms as fixed diagnostics without gating or tuning from their magnitudes.

### Milestone 3: Require preserved H20 exposure
- [x] Run accepted/candidate complete-body timing for early mixup and hard-label regimes with at least 20 warmups and four counterbalanced windows of at least 50 steps per arm; print every synchronized per-step window before assertions and require every population CV <=5% and peak allocation <2,048 MiB.
- [x] Require fixed-time retention >=0.9774 and projected passes >=130.0 from accepted 133.00736. A stable miss closes exact `128 -> 64 -> 128`, ReLU, scale-0.1, seed-36036 feasibility without rerun or repair.

### Milestone 4: Run and classify the sole fixed-seed score
- [x] Reconfirm baseline 94.32 at `67c8e98`, one idle H20, local CIFAR-10, frozen evaluator, exact source scope, no stale `run.log`, and passing gates; execute exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Require exit 0, one finite summary, 300.0-300.1 counted seconds, total <600, 1,003,482 parameters, correct temporal transitions, unique accepted-cadence evaluations, and no numerical/CUDA/worker errors. Record realized passes; a valid completion below 130 remains the sole goal score but makes the remapping mechanism operationally inconclusive.
- [x] Classify improvement solely by `best_test_acc >=94.42%`; report `final_test_acc >=94.32%` and `final_test_loss <=0.2523` only as corroboration. A valid normal-exposure miss closes nearby pooled-head width, scale, activation, bias, zero-init, learned-scale, initialization-seed, and optimizer rescues.

## Code Changes
- **`train.py` / constants**: define `POOLED_HEAD_WIDTH = 64`, `POOLED_HEAD_SCALE = 0.1`, and `POOLED_HEAD_INIT_SEED = 36036`.
- **`train.py` / model construction**: after the accepted `self.apply(self._weights_init)` has initialized the complete accepted model and classifier, enter `torch.random.fork_rng(devices=[])`, call `torch.random.default_generator.manual_seed(36036)` (not `torch.manual_seed`), register `nn.Sequential(nn.Linear(128,64,bias=False), nn.ReLU(), nn.Linear(64,128,bias=False))`, and apply the accepted Kaiming-normal matrix initializer directly to both bias-free weights. Exiting the fork must restore the accepted global CPU RNG exactly; the CPU-only generator call and CPU module construction must leave CUDA RNG byte-identical.
- **`train.py` / forward**: preserve final BN/ReLU/GAP/flatten, compute `out = out + 0.1 * self.pooled_head(out)`, then call the accepted `self.fc(out)`. The direct pooled path, accepted classifier, and every spatial operation remain intact.
- **`.autoresearch/.../experiments/036/preflight.py`**: ignored evaluator-free harness for exact source/construction/topology, formula/RNG/optimizer/update checks, fixed diagnostics, and balanced H20 timing. It must never load test data, call evaluation, add production logs, or create `run.log`.

## Configuration Changes
- Pooled residual head: none -> bias-free `Linear(128,64) -> ReLU -> Linear(64,128)` at fixed residual scale 0.1.
- New-parameter initialization: fixed isolated CPU seed 36036 through `torch.random.default_generator.manual_seed` and the accepted Kaiming initializer; accepted common tensors and global CPU/CUDA construction RNG remain exact.
- Exposure classification: this reviewed plan supersedes the proposal's below-130 label. A structurally valid completed run below 130 is a valid nonrepeatable goal score, not a crash/invalid result; only the pooled-remapping mechanism becomes operationally inconclusive.
- Parameters: 987,098 -> 1,003,482; all 16,384 new matrix parameters use accepted continuous `5e-4` decay and ordinary LR/momentum/Nesterov settings.
- Accepted recipe unchanged: `(2,2,3)`, batch 256, FP32, LR `0.2 -> 0.002`, momentum 0.9, alpha-0.2 batch-shared mixup through 65%, early worker-private N1/M5 RandAugment, crop/flip, seed 42, worker setup, budget, and evaluator cadence.

## Execution Environment
- Method: offline local semantic/diagnostic/timing preflight, then one local score only on pass; no network, remote, installs, W&B, GitHub, `gh`, fetch, push, or PR action.
- Resources: one NVIDIA H20, local CIFAR-10, installed environment, eight persistent forkserver workers.
- Estimated runtime: preflight under 4 minutes; score about 345-360 seconds wall with a 600-second hard timeout.
- Log output: scored stdout/stderr only in root `run.log`, retained through analysis and then removed.
- Tool skill: none.

## Abort Criteria
- Abort before timing on any scope/frozen-file/syntax failure; common tensor or global construction-RNG mismatch; wrong topology/parameter count/dtype; branch reconstruction mismatch; bias/normalization/dropout/learnable scale; wrong optimizer membership; formula/direct-path/RNG mismatch; non-finite or zero branch contribution/gradient/update; accepted loss/cutoff/worker/evaluation-source mismatch; or evaluator/test access.
- Abort before scoring if any timing value is non-finite, any CV exceeds 5%, retention is below 0.9774, projected passes are below 130.0, or peak allocation is at least 2,048 MiB. Emit raw measurements before assertions and never repeat a stable miss.
- During score stop/classify on timeout, nonzero exit, OOM/worker/resource error, non-finite loss, no output for 60 seconds, malformed/missing/duplicate summary, wrong parameter count, invalid/repeated temporal transition, duplicate evaluation epoch, or total >=600. Never rerun a valid completion or react to interim accuracy.

## Verification Protocol

### Verification Procedure

1. Query `exp-index.sh baseline`; require 94.32 at `67c8e98`, so the primary threshold is 94.42.
2. Within 30 seconds query `nvidia-smi`, local CIFAR presence, branch/status, `git diff --check`, `git diff 67c8e98 -- train.py`, `git diff --exit-code 67c8e98 -- prepare.py`, and `uv run python -m py_compile train.py .../experiments/036/preflight.py`. Require one idle H20, only tracked `train.py`, and exactly the fixed branch constants/construction/forward implementation without production diagnostics.
3. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/036/preflight.py semantics`. Guard `prepare.Eval` and test-data construction before importing accepted/candidate modules. Construct both from cloned seed-42 CPU/CUDA state; require all common named parameters/buffers byte-equal, common object shapes/dtypes exact, post-construction global CPU/CUDA RNG states equal, and accepted constants, transforms, loader, schedule, loss, controls, evaluator cadence, and summary source unchanged.
4. Independently recreate the new branch under a restoring seed-36036 CPU fork using `torch.random.default_generator.manual_seed` and require byte-equal matrices. Require exactly `Linear(128,64,bias=False) -> ReLU -> Linear(64,128,bias=False)`, fixed scale 0.1, 16,384 new FP32 trainable parameters, 1,003,482 total, no other new state, and the same isolated branch bytes across repeated construction while both global CPU and CUDA RNG states remain byte-identical. Explicitly reject `torch.manual_seed` inside the branch fork.
5. On deterministic finite synthetic inputs, hook candidate final pooling and classifier input. Require the branch input equals accepted `BN/ReLU/GAP/flatten`, classifier input equals `pooled + 0.1*head(pooled)` exactly, and candidate logits equal an independent expression. Require `candidate.fc(pooled)` bitwise equals accepted logits from common state. Prove forward consumes no CPU/CUDA RNG. Print finite nonzero branch/direct norm ratio and direct-to-candidate maximum/RMS logit perturbation as diagnostics; impose no magnitude threshold.
6. Enumerate optimizer groups by identity/name. Require each trainable tensor exactly once, all new matrices in the `5e-4` decay group, no change to common membership, and exact accepted LR/momentum/Nesterov settings. Run deterministic early-mixup and hard updates; require finite loss, every present gradient finite, nonzero aggregate backbone and classifier gradient norms, finite nonzero gradients and updates for both new head matrices, and finite optimizer states. Print L2 gradient norms for backbone, accepted classifier, first head matrix, and second head matrix; beyond the aggregate/nonzero rules, their relative magnitudes are diagnostic only. Require mixup scalar/target alignment, RNG replay, strict 65% boundary, and accepted LR samples.
7. Run `timeout 240s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/036/preflight.py timing`. Compare independent accepted/candidate modules from common accepted states using fixed pinned batches, production-equivalent nonblocking H2D, LR write, zeroing, Beta/permutation/interpolation when active, forward, paired or hard loss, finite guard, backward, Nesterov step, and synchronization.
8. Use at least 20 disposable warmups per arm/regime, then four windows of at least 50 measured steps ordered `accepted/candidate, candidate/accepted` and repeated. For every pair, start both arms from fresh cloned common parameter and BN-buffer state, equivalent optimizer state, identical deterministic pinned inputs/targets, and identical CPU/CUDA RNG; use a distinct preregistered fixture seed for each pair. Reset CUDA peak-memory statistics immediately before each candidate window and retain the maximum candidate value. Define each window as synchronized elapsed milliseconds divided by steps; print all means, medians, population CVs, retention, projected passes, and candidate peak before assertions. Require CV <=0.05, candidate peak <2,048 MiB, `retention=(.65/candidate_mixup_ms+.35/candidate_hard_ms)/(.65/accepted_mixup_ms+.35/accepted_hard_ms) >=0.9774`, and `projected_passes=133.00736*retention >=130.0`. No loader timing is needed because source and consumer shape are unchanged.
9. Reconfirm audit and one idle H20, remove stale `run.log`, execute exactly `timeout 600s uv run train.py > run.log 2>&1` once, record start/session, and never launch a second valid score.
10. Parse with `rg`; require one finite summary, 300.0-300.1 counted seconds, total <600, 1,003,482 parameters, no traceback/OOM/non-finite/worker error, one mixup transition at the first pre-step counted time >=195 seconds, and one RandAugment transition at the first iterator boundary after threshold. Accept either source-valid marker ordering if the crossing is the last batch of an epoch.
11. Require evaluation epochs to equal every fifth epoch union the final epoch, with no duplicate and at most one call per epoch. Record realized exposure as `num_steps*256/50000`; below 130 cannot be rerun and is mechanism-inconclusive rather than a seed rescue opportunity.
12. Classify success only by `best_test_acc >=94.42%`. Separately report final accuracy versus accepted 94.22 and the 94.32 corroboration floor, final loss versus 0.2523, best-final gap, steps/epochs/passes, evaluation count, transitions, VRAM, counted/wall time, timing retention, initial diagnostics, and final source audit. A valid >=130-pass miss closes the immediate pooled-head neighborhood preregistered above.

### Informational Metrics (Optional)
- Final summary: `run.log` - best/final accuracy, final loss, counted/total/startup seconds, epochs, steps, passes, VRAM, and parameters.
- Head diagnostics: preflight stdout - branch/direct norm ratio, direct/candidate logit maximum/RMS perturbation, and grouped gradient norms.
- Timing: preflight stdout - raw early/hard windows, medians/CVs, retention/pass projection, and peak allocation.
- Mechanism: best/final/loss deltas from accepted 94.32/94.22/0.2523 and realized versus projected exposure.

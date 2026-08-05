# Plan EXP-015: Per-Example Mixup Strengths
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Implement the isolated coefficient-granularity change
- [x] Modify only `train.py`: sample a `[inputs.size(0)]` Beta vector, broadcast it as `[B,1,1,1]` for image interpolation, and return the original `[B]` vector for target weighting.
- [x] Add a small production `mixup_loss` helper that computes the two cross-entropies with `reduction="none"`, weights each example with its own coefficient, and takes one final mean; make the scored mixup branch call this exact helper so preflight and production cannot diverge.
- [x] Preserve alpha 0.2, the 65% cutoff, ordinary permutation, model, initialization, optimizer, time-based schedule, hard-label branch, loader, seed, and evaluation cadence.
- [x] Create an evaluator-free ignored harness at `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/015/preflight.py` and run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/015/preflight.py --semantics`; allow the module-level dummy `Eval` construction required by `train.py` import, but fail immediately if evaluation is called; require every stated semantic assertion to print `PASS` and exit 0.

### Milestone 2: Pass scope, syntax, semantics, and throughput gates
- [x] Run `uv run python -m py_compile train.py`; require exit 0.
- [x] Run `git diff --name-only eb08811 --`; require exactly `train.py`, audit `git diff --check` and `git diff eb08811 -- train.py`, require no non-ignored untracked files, and require the only project-root Python files to be tracked `prepare.py` and `train.py`.
- [x] Run the evaluator-free matched benchmark with `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/015/preflight.py --throughput`; require stable windows, at least 95% mixup-step retention, and at least 134.8 projected passes.
- [x] Remove the preflight process state and confirm `test ! -e run.log` before scoring.

### Milestone 3: Complete the single fixed-seed scored run
- [x] Confirm exactly one visible NVIDIA H20 with `nvidia-smi --query-gpu=name --format=csv,noheader`; require one line equal to `NVIDIA H20`.
- [x] Execute exactly once from the project root with `timeout 600s uv run train.py > run.log 2>&1`; record the exit code and do not rerun a valid result.
- [x] Require exit 0, a complete final summary, finite losses, one mixup-disable transition near 195 counted seconds, about 300 counted training seconds, and total wall time below 600 seconds.

### Milestone 4: Verify and hand off for analysis
- [x] Parse the scored summary; `best_test_acc=93.79` failed the required `>=94.17` condition.
- [x] Audit that evaluations occur at most once per epoch and that `num_steps * 256 / 50000 = 142.01344` realized dataset-equivalent passes.
- [x] Record implementation, preflight measurements, command/exit status, transition, summary metrics, and any deviations in `03-execute.md`; leave `run.log` available for analysis.

## Code Changes
- **`train.py` / `mixup_batch`**: replace the scalar `distribution.sample()` with `distribution.sample((inputs.size(0),))`; create a separate broadcast view only for pixels. Keep the coefficient vector unsymmetrized and keep the existing one-per-batch `torch.randperm`. This changes coefficient correlation without changing the per-example Beta marginal.
- **`train.py` / `mixup_loss` and mixup branch**: add one stateless production helper that computes `loss_a` and `loss_b` with `reduction="none"`, combines them elementwise using the `[B]` coefficient vector, and calls `.mean()` exactly once. The scored branch must call this helper, and the semantic/timing preflight must import and call the same helper rather than reimplementing candidate loss math. The same coefficient must weight the image and its two labels.
- No production file other than `train.py` may change. The preflight harness is an ignored experiment artifact, must replace `prepare.Eval` with a fail-closed stub before importing `train.py`, and must never instantiate or inspect test data.

## Configuration Changes
- Mixup coefficient granularity: one scalar per batch -> one independent scalar per example (the sole treatment).
- `MIXUP_ALPHA`: 0.2 -> 0.2 (unchanged, preserving the validated marginal distribution).
- `MIXUP_END_FRACTION`: 0.65 -> 0.65 (unchanged, preserving the hard-label tail).
- All architecture, optimizer, schedule, augmentation, batch-size, seed, numerical precision, worker, maximum-step, and evaluation settings remain unchanged from commit `eb08811`.

## Execution Environment
- Method: local, offline execution from the project root; no remote service, network lookup, package install, W&B, GitHub, or `gh` operation.
- Resources: exactly one NVIDIA H20 GPU (97,871 MiB visible), local CIFAR-10 data, existing `uv` environment, 8 persistent DataLoader workers.
- Estimated runtime: preflight under 3 minutes; scored training about 342 seconds wall time including evaluations, with 300 counted training seconds and a hard 600-second timeout.
- Log output: the scored command redirects stdout/stderr only to project-root `run.log`; this file is the primary execution record and remains until the analysis phase removes it.
- Tool skill: none; execution is local.

## Abort Criteria
- Abort before scoring if the only production diff is not `train.py`, if syntax fails, if any semantic preflight assertion fails, or if the harness accesses `Eval`/test data.
- Abort before scoring if `git ls-files --others --exclude-standard` reports any file, or if a root-level Python file other than tracked `prepare.py` and `train.py` exists. Confirm the nested ignored preflight path with `git check-ignore`; it must not be importable as a root module used by `train.py`.
- Abort before scoring if any coefficient is non-finite/outside `[0,1]`, the vector shape is not `[256]`, 4,096 fixed-seed draws have mean outside `[0.47,0.53]` or variance outside `[0.15,0.21]`, or within-batch coefficient diversity is absent.
- Abort before scoring if constant-vector equivalence to the accepted scalar loss/gradient fails at `rtol=1e-5, atol=1e-6`, image/label coefficient alignment fails, or a matched hard-label step differs from the accepted rule.
- Abort before scoring if any throughput window has population CV above 5%, candidate mixup retention is below 95%, projected exposure from 141.9 passes is below 134.8, logits/loss/gradients are non-finite, or the preflight OOMs.
- During scoring, let `timeout 600s` terminate a hung run. Classify nonzero exit, CUDA/OOM/error traceback, non-finite loss, absent final summary, more than one evaluation in an epoch, multiple/missing mixup transitions, or total wall time at least 600 seconds as crash/invalid as appropriate. Do not alter code or rerun after a valid completed score.

## Verification Protocol

### Verification Procedure

1. Query the authoritative baseline and threshold:
   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv
   ```
   Require `baseline=94.07` and `baseline_commit=eb08811`; the necessary accuracy threshold is `94.17` (baseline plus 0.10 percentage points).

2. Verify hardware and scope before any scored run:
   ```bash
   nvidia-smi --query-gpu=name --format=csv,noheader
   git diff --name-only eb08811 --
   git status --short --untracked-files=all
   git ls-files --others --exclude-standard
   find . -maxdepth 1 -type f -name '*.py' -printf '%f\n' | sort
   git check-ignore .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/015/preflight.py
   git diff --check
   uv run python -m py_compile train.py
   ```
   Require exactly one output device named `NVIDIA H20`; diff/status output showing only modified `train.py`; no non-ignored untracked file; root Python output exactly `prepare.py` and `train.py`, both tracked; `git check-ignore` exit 0 for the nested harness; no diff-check errors; and compile exit 0. Timeout: 30 seconds total.

3. Run the evaluator-free semantic harness:
   ```bash
   uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/015/preflight.py --semantics
   ```
   The harness must replace `prepare.Eval` before importing `train.py`, allow only the expected module-level dummy construction, and fail immediately if `evaluate` is called. Every candidate loss assertion must call imported production `train.mixup_loss`; the harness may implement only the accepted scalar reference. It must assert: model initialization and state keys match accepted behavior; coefficients are finite FP32 `[256]` values in range with multiple distinct entries; 4,096 draws satisfy the stated broad Beta statistics; each mixed image and paired loss uses its matching coefficient; a manually constant vector passed through production `mixup_loss` reproduces the accepted scalar loss and all gradients within `rtol=1e-5, atol=1e-6`; candidate random coefficients are not scalar broadcasts; and a cloned hard-label forward/backward/SGD step is bitwise identical. Require exit 0 and an explicit `SEMANTICS PASS`. Timeout: 120 seconds.

4. Run a matched H20 throughput benchmark:
   ```bash
   uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/015/preflight.py --throughput
   ```
   Compare complete accepted scalar and candidate vector mixup steps from cloned model/optimizer state in balanced order, including transfer, LR write, Beta draw, permutation, interpolation, forward, finite check, backward, optimizer step, and synchronization. The candidate path must call imported production `train.mixup_batch` and `train.mixup_loss`; only the accepted scalar reference may live in the harness. Warm at least 25 steps per path and measure at least three 50-step windows per path. Require finite `[256,10]` logits and loss, no OOM, population CV at most 5% for each path, retention `accepted_median_ms / candidate_median_ms >= 0.95`, and projected passes `141.9 * retention >= 134.8`. Require explicit `THROUGHPUT PASS`. Timeout: 180 seconds.

5. Remove any stale log and perform exactly one scored run:
   ```bash
   rm -f run.log
   timeout 600s uv run train.py > run.log 2>&1
   ```
   Require command exit 0 within 600 seconds. A missing `best_test_acc` is a crash; inspect with `tail -n 50 run.log`. Do not rerun a valid result regardless of whether it is close to the threshold.

6. Parse and audit necessary conditions:
   ```bash
   rg '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|startup_seconds|peak_vram_mb|num_epochs|num_steps|num_params):' run.log
   rg '^  eval ep' run.log
   rg 'Mixup disabled' run.log
   rg -n 'Traceback|RuntimeError|CUDA out of memory|Non-finite' run.log
   ```
   Require exactly one complete summary; `best_test_acc >= 94.17`; `training_seconds` approximately 300 seconds (one atomic final step may overshoot slightly); `total_seconds < 600`; `num_params = 691674`; no error match; exactly one mixup-disable line near 195 counted seconds; and no duplicate evaluation epoch. The run is an improvement only if every integrity condition passes and the metric meets the threshold. Timeout: 30 seconds.

7. Audit the implementation rather than trusting log claims:
   ```bash
   git diff eb08811 -- train.py
   ```
   Confirm evaluation remains only at the existing end-of-epoch condition `epoch % EVAL_EVERY == 0 or budget_exhausted`, so it cannot occur more than once per epoch; `TIME_BUDGET_S` still comes from frozen `prepare.py`; seed remains 42; `MIXUP_ALPHA` remains 0.2; the cutoff remains 0.65; and no test/evaluator path was modified. Any violation makes the result invalid even if accuracy passes.

### Informational Metrics (Optional)
- `peak_vram_mb`: final summary line in `run.log`; compare qualitatively with the accepted approximately 1,094 MiB.
- `final_test_acc`: final summary line in `run.log`.
- `final_test_loss`: final summary line in `run.log`.
- `training_seconds`: final summary line; should be about 300 seconds.
- `total_seconds`: final summary line; must remain below 600 seconds.
- `num_epochs`: final summary line; use with evaluation lines for cadence audit.
- `num_steps`: final summary line; compute realized passes as `num_steps * 256 / 50000`.
- `num_params`: final summary line; must remain 691,674.

# Plan EXP-007: Literature-Scale ASAM in the Validated Clean Tail
- **Created**: 2026-08-05

## Goal and Fixed Reference

- Parent node: EXP-004 (`1a8d0de`), `best_test_acc=95.40%`, final loss 0.1654, 25,560 steps, 2,449 SAM pulses.
- Formal acceptance threshold: `Decimal("95.40") + Decimal("0.10") = Decimal("95.50")`.
- Preregistered mechanism-sized target: `best_test_acc >= 95.70%` with at least 25,000 steps. Scores from 95.50% through 95.69% satisfy the frozen tree gate but fall below the chosen idea's +0.30-point evidentiary target.
- Chosen intervention: replace only the parent's late period-two Euclidean SAM perturbation package with element-wise p=2 ASAM using fixed literature values `rho=0.5`, `eta=0.01`.
- Attribution: the result tests the complete ASAM package (radius and scale-aware geometry together) under sparse late cadence. It cannot isolate scale adaptation from the tenfold geometry-specific rho change or extrapolate directly to full-run ASAM.

## Milestones

### Milestone 1: Implement exact named-parameter ASAM geometry
- [x] Replace `SAM_RHO=0.05`/`SAM_EPS` with `ASAM_RHO=0.5`, `ASAM_ETA=0.01`, and `ASAM_EPS=1e-12`; retain start 0.75 and period 2.
- [x] Build an ordered named-parameter inventory with exactly 30 adapted non-bias tensors / 2,747,072 elements and 14 unit-scale bias tensors / 1,818 elements.
- [x] Preallocate exact snapshots, scales, and direction buffers. Snapshot every parameter before deriving any scale; use `abs(snapshot)+eta` for non-bias tensors and one for names ending `.bias`.
- [x] Compute the global FP32 denominator `D=||s*g||_2`, then construct `epsilon=rho*s^2*g/(D+eps)` with the required second scale multiplication. Weight decay remains only in the restored base optimizer update.
- [x] On the first production pulse, reduce the actual constructed epsilon buffers to `||epsilon/s||`, `max|epsilon/s|`, Euclidean norm, maximum scale, and conv/BN/fc/bias denominator/epsilon energy shares.
- [x] Verification: closed-form FP64 toy geometry, missing-second-scale discriminator, exact inventory/name coverage, actual adaptive-coordinate radius `[0.499,0.501]`, actual `max|epsilon/s| <= 0.501`, Euclidean norm `[1e-3, 0.501*max_scale]`, and injected-failure restoration must pass.

### Milestone 2: Preserve the complete EXP-004 training contract
- [x] Keep the first 75% parent path, DataLoader, independent-image stream, CutMix generators/draws, architecture, BF16/channels-last forward, drop path, LR, and Nesterov optimizer unchanged.
- [x] Keep the same upcoming one-based period-two predicate. Save CUDA RNG after the primary pass, clear first gradients, replay RNG, disable BatchNorm tracking only for the second pass, restore flags and exact parameters in `finally`, then perform one optimizer update.
- [x] Keep hard clean targets in both ASAM passes and retain the CutMix/ASAM non-overlap assertion.
- [x] Add startup configuration and final `cutmix:` / `asam:` / `asam_geometry:` audit lines with the last applied CutMix progress, ASAM coverage/cadence, denominator statistics, actual first-pulse geometry/group shares, and zero failure counters.
- [x] Verification: actual-parent early CutMix and unscheduled-late parity, six-drop-mask RNG advancement, one BatchNorm update, exact restoration, one momentum update, and evaluator/default-forward parity must pass.

### Milestone 3: Run one fixed GPU-0 experiment
- [x] Confirm physical GPU 0 is the 97,871 MiB NVIDIA H20, branch EXP-007 is at parent `1a8d0de`, only `train.py` differs, and no stale `run.log` exists.
- [x] Benchmark warm actual-parent SAM versus candidate ASAM pulses on GPU 0 after recording utilization/co-tenancy. Require process teardown, finite execution, and a projected full-run horizon of at least 25,000 steps; do not change rho, eta, cadence, or parameter coverage if the gate fails.
- [x] Launch exactly once with seed 42, a 600-second outer timeout, physical GPU 0 only, and complete stdout/stderr capture in `run.log`.
- [x] Monitor only health/abort conditions. Intermediate accuracy cannot prune, tune, or trigger a retry.

### Milestone 4: Verify and record the preregistered result
- [x] Parse configuration, CutMix, ASAM cadence, geometry, timing, evaluation count, and summary integrity before reading `best_test_acc`.
- [x] Compare the single metric read first with 95.50% for the tree verdict and then with 95.70% for the mechanism-sized hypothesis.
- [x] Report best/final accuracy, final loss, steps, epochs/evaluations, time, VRAM, CutMix dose, ASAM dose/geometry, and unchanged parameter count.
- [x] Preserve evidence in execution/analysis artifacts, obtain Claude's adversarial result audit, then remove `run.log` without rerunning.

## Code Changes

- **`train.py` only**:
  - Rename the user-facing perturbation configuration to ASAM while leaving scheduling constants at 0.75 and period 2.
  - Replace `sam_perturb(parameters, snapshots)` with an ASAM helper that receives ordered `(name, parameter, is_bias)` entries plus parallel snapshots/scales/directions. Assert identical list lengths, names, shapes, dtypes, devices, and preserved memory formats at setup.
  - Under `torch.no_grad()`, require one finite gradient per parameter, copy all parameters to snapshots, fill every scale from the complete snapshots, compute `scaled_gradient=s*g` into preallocated direction buffers, and obtain one global FP32 norm. Reject nonfinite/nonpositive denominators. Use `torch._foreach_*` for copy, abs/add, multiply, and parameter-add stages wherever tensor subsets permit it.
  - Reuse each direction buffer for `epsilon`: multiply by its scale with foreach, multiply the list by `ASAM_RHO/(D+ASAM_EPS)`, then foreach-add to parameters. On any exception, restore all snapshots before re-raising; do not implement the production path as 200+ independent per-tensor kernels.
  - Treat exactly names ending in `.bias` as unit-scale. All convolution, linear, and BatchNorm `.weight` tensors use `abs(snapshot)+0.01`; do not exclude BatchNorm affine weights or add optimizer weight decay to the perturbation gradient.
  - Keep `restore_sam_parameters` semantics (renamed if useful) and the existing `try/finally` second-pass structure. Parameters must equal snapshots bitwise immediately before the sole `optimizer.step()`.
  - Record denominator min/mean/max from the already synchronized denominator scalar. Treat `rho*D/(D+eps)` only as a structural formula invariant, not evidence that epsilon was built correctly.
  - On the first production pulse only, compute from the actual epsilon buffers: normalized buffers `epsilon/s`, their global radius and maximum absolute coordinate, Euclidean epsilon norm, maximum scale, and four parameter-group contributions to both `D^2` and `sum(epsilon^2)` (conv weight, BatchNorm weight, `fc.weight`, unit-scale bias). Materialize these host scalars immediately; the one-time synchronization is charged and occurs before any evaluation output.
  - Require first-pulse actual radius `[0.499,0.501]`, normalized maximum `<=0.501`, and Euclidean norm in `[1e-3, 0.501*max_scale]`. These fixed implementation bounds can abort before metric observation; group shares are diagnostic and never tune or abort the run.
  - Add zero-initialized counters for nonfinite geometry, first-pulse geometry violations, restoration failures, and overlap. Later pulses reuse the identical validated code path and require finite positive `D`; they do not claim an independently measured per-pulse radius.
  - Record the maximum progress at which CutMix was actually applied and print it on the final `cutmix:` line. This makes the cutoff audit an observed timestamp rather than an unreachable late-mix counter.
  - Print `asam:` with applied/eligible ratio and first step/progress, and `asam_geometry:` with rho, eta, adapted/unit tensor and element counts, denominator summaries, actual first-pulse radius/normalized max/Euclidean norm/max scale, four denominator shares, four epsilon shares, and failure counters. Preserve every required final summary key exactly once.
  - Do not change model construction, parameter initialization, transforms, loader, batch size, seeds, CutMix probability/alpha/end/geometry, drop path, optimizer hyperparameters, LR schedule, ASAM start/cadence, evaluator, timer boundaries, or validation frequency.

## Configuration Changes

- Perturbation package: Euclidean SAM `rho=0.05, scale=1` -> p=2 ASAM `rho=0.5, eta=0.01, scale=abs(w)+eta` for non-bias weights and scale 1 for biases.
- Perturbation schedule: unchanged at progress `>=0.75` and every second upcoming one-based step.
- Trainable model: unchanged at 2,748,890 parameters; ASAM buffers are non-trainable state only.
- All EXP-004 data, architecture, CutMix, drop-path, optimizer, LR, seed, and evaluation settings remain unchanged.

## Execution Environment

- Method: local single-process execution from the repository root.
- Resources: physical GPU 0 only via `CUDA_VISIBLE_DEVICES=0`; NVIDIA H20 with 97,871 MiB; existing eight DataLoader workers; no dependencies added.
- Estimated runtime: 300 charged seconds and approximately 455-500 seconds total; hard outer timeout 600 seconds.
- Log output: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log` captures all stdout/stderr and is the metric source of truth.
- Full command: `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
- Monitoring: health patterns only; intermediate test accuracy is not actionable.
- Tool skill: none; this is a local GPU run monitored through its process session and log.

## Abort Criteria

- Before launch, abort if GPU 0 is not the 97,871 MiB H20, base/branch is wrong, any protected/dependency file changed, `run.log` exists, static checks fail, ASAM geometry/inventory/restoration/RNG/BatchNorm/parent-parity smokes fail, or the fixed ASAM pulse latency projects fewer than 25,000 steps.
- During the run, stop on traceback, CUDA/OOM error, nonfinite loss/gradient/scale/denominator/epsilon, missing gradient, first-pulse actual radius outside `[0.499,0.501]`, normalized maximum above 0.501, Euclidean norm outside `[1e-3, 0.501*max_scale]`, parameter restoration failure, CutMix/ASAM overlap, BatchNorm flag restoration failure, or stalled progress making the outer timeout unavoidable.
- Exit 124/nonzero, missing or duplicate summary fields, charged time outside 299.5-301.0 seconds, total time at or above 600 seconds, fewer than 24,000 steps, parameter-count change, evaluation count mismatch, or nonzero discriminating audit failure counters is a protocol failure. A valid 24,000-24,999-step run is final and exposure-degraded; it falsifies the 25,000-step mechanism target but never authorizes a retry.
- Do not abort for weak intermediate accuracy or for finite parameter-group shares. No metric-driven retry is allowed. A deterministic code/infrastructure repair is allowed only when a named non-accuracy failure is detected and documented before any evaluation line or summary from that run is read; preserve the failed log/partial metrics in `03-execute.md`. After any evaluation output has been read, the run is final regardless of later nonfatal defects.

## Verification Protocol

### Verification Procedure

1. **Parent, threshold, scope, and hardware** (20-second timeout):
   - Resolve `PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-/root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.2}"`; require executable `$PLUGIN_ROOT/skills/shared/scripts/tree.sh`; query node 004 and require `metric=95.40`, `commit=1a8d0de`, and extendable true. Compute the formal threshold with `Decimal` as 95.50 and record the separate 95.70 hypothesis target.
   - Require branch `tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-007`, HEAD initially `1a8d0de`, only tracked `train.py` differs, no dependency changes, and no stale log.
   - Run `nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader` plus `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader`. Require physical index 0 to be NVIDIA H20 with 97,871 MiB, at least 90,000 MiB free, no other process from this repository, and pre-launch utilization no greater than 10%; record unrelated co-tenants rather than silently assuming exclusivity.

2. **Static and inventory checks** (60-second timeout):
   - Run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, and `git diff --check`; all must exit 0. Inspect `git diff -- train.py` and require only the approved perturbation helper, state, audit, and naming changes.
   - Instantiate the model without a training run and reconcile 44 trainable tensors / 2,748,890 elements: exactly 30 adapted tensors / 2,747,072 elements and 14 `.bias` tensors / 1,818 elements. Load the parent source under a non-main namespace, construct child and parent from separately restored identical RNG states, copy one common `state_dict`, and require equal model state, initialization, default outputs, and post-construction RNG state.
   - Every smoke/benchmark is an inline `uv run python -` heredoc or a temporary file outside the repository. Do not create repository-root smoke scripts or untracked artifacts.

3. **Closed-form ASAM geometry** (120-second timeout):
   - Run an inline deterministic FP64 toy with named weight and bias tensors. Compare scales, `D`, actual epsilon buffers, Euclidean norm, actual `||epsilon/s||`, and actual `max|epsilon/s|` against independent hand calculations. Require exact unit bias scale, non-bias `abs(snapshot)+0.01`, the second scale multiplication, radius `[0.499,0.501]`, normalized maximum `<=0.501`, finite values, and unchanged global CPU/CUDA RNG states.
   - Explicitly compare against the incorrect one-scale formula and require different epsilon on heterogeneous scales. Test zero, missing, NaN, and Inf gradients; invalid inputs must fail before an optimizer update and restore exact parameters.
   - Inject an exception after partial perturbation and require every parameter bitwise equal to its snapshot afterward.

4. **Parent-path and training semantics** (180-second timeout):
   - Load actual parent source with `git show 1a8d0de:train.py` into a non-main namespace. Avoid import-time comparison drift by snapshotting/restoring global RNG around construction and copying a single common `state_dict`. On matched model/input/RNG states, require an early selected CutMix step and an unscheduled late step to match parent decisions, transformed inputs, losses, outputs, gradients, optimizer state, BatchNorm buffers, and CPU/CUDA RNG states.
   - On a scheduled full-WRN BF16/channels-last ASAM step, require six identical drop-path draws across passes through CUDA RNG replay, one BatchNorm-buffer update, finite distinct perturbed loss, actual epsilon-derived radius `[0.499,0.501]`, normalized maximum `<=0.501`, Euclidean norm within its fixed bound, exact parameter restore including BatchNorm affine weights, first-gradient clearing, and one Nesterov/momentum update from second-pass gradients.
   - Reconcile first-pulse conv/BN/fc/bias shares of `D^2` and epsilon energy: each set must be finite, nonnegative, and sum to one within `1e-5`. These shares diagnose concentration but never select/tune the run.
   - Verify all 30 adapted and 14 unit tensors participate, weight decay is absent from first gradients, and CutMix cannot overlap the ASAM branch.

5. **GPU-0 latency and exposure gate** (180-second timeout):
   - In sequential fixed-seed GPU-0 processes, benchmark warm actual-parent SAM and candidate ASAM scheduled pulses at batch 256, BF16, and channels-last with synchronization. Report median/p90 over at least 100 pulses and peak allocation; require every benchmark PID to exit before rechecking GPU state.
   - Combine the pulse ratio with EXP-004's 2,449/25,560 scheduled fraction and report projection margin above 25,000, not only pass/fail. Require projected 300-second horizon at least 25,000 and total runtime below 600 seconds. A failure rejects the fixed recipe; never alter rho, eta, cadence, or tensor coverage to pass.

6. **Single full run** (610-second timeout):
   - Reconfirm launch preconditions, then run exactly once: `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. A single repair is permitted only for a named non-accuracy failure detected before reading any evaluation line or summary; preserve the failed evidence. Otherwise require exit 0 with no retry.

7. **Protocol integrity before metric** (30-second timeout):
   - **Discriminating ASAM gates:** parse startup config and `asam_geometry:`; require fixed rho/eta, exact 30/14 tensor and 2,747,072/1,818 element coverage, zero nonfinite/first-geometry/restoration failures, finite positive denominator, actual first-pulse epsilon-derived radius `[0.499,0.501]`, normalized maximum `<=0.501`, Euclidean norm in `[1e-3, 0.501*max_scale]`, and two four-group share sets that each sum to one within `1e-5`.
   - **Structural protocol invariants:** parse `cutmix:`, `asam:`, all eval lines, and the complete summary; record last applied CutMix progress below 0.75, ASAM first progress `[0.7500,0.7520]`, even first step, applied/eligible ratio `[0.499,0.501]`, no overlap, 299.5-301.0 charged seconds, total below 600, 2,748,890 parameters, unique summary keys, and eval count equal to epochs. These confirm the frozen branch/cadence but are not evidence that ASAM geometry is correct.
   - Require at least 24,000 actual steps. Record 24,000-24,999 as a valid exposure-degraded miss of the 25,000-step hypothesis, with no retry.

8. **Primary and mechanism verdicts** (10-second timeout):
   - Run `grep '^best_test_acc:' run.log` and parse exactly one percentage only after step 7 passes.
   - `>=95.50%` passes the necessary tree metric condition; `<95.50%` is no-improvement. Independently, `>=95.70%` supports the preregistered mechanism-sized hypothesis. For 95.50-95.69, record exactly: "Formal improvement, but the gain is below the preregistered +0.30 evidentiary bar and remains indistinguishable from observed single-run variation." Do not use final loss, geometry shares, steps, or any informational metric to upgrade that mechanism conclusion; do not rerun or tune.

9. **Evidence and cleanup** (10-second timeout):
   - Record all raw audit/summary values in `03-execute.md` and `04-analysis.md`, obtain Claude's adversarial result review, then remove `run.log`. Run `git status --porcelain --untracked-files=all` and require no transient log, smoke script, or untracked artifact; the only intended tracked difference before analysis commit is `train.py`.

### Informational Metrics (Optional)

- `final_test_acc`, `final_test_loss`: final summary, compared with parent 95.40% / 0.1654. Final-equals-best and loss no worse than 0.1654 may be described only as optimization-behavior diagnostics; they never alter the accuracy verdict or evidentiary tier.
- `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`: final summary, compared with parent 300.0 / 457.3 / 1.2 / 1,190.5.
- `num_epochs`, evaluation count, `num_steps`, `num_params`: final summary/log count, compared with parent 132 / 132 / 25,560 / 2,748,890.
- CutMix/ASAM exposure: final audit lines, compared with parent CutMix 10,252/20,662 and SAM 2,449/4,898, first step 20,664 at progress 0.7500.
- ASAM geometry: denominator min/mean/max; actual first-pulse radius, normalized maximum, Euclidean norm, maximum scale, group shares; tensor/element coverage; and failure counters. These support implementation/attribution diagnosis only and never compensate for an accuracy miss.

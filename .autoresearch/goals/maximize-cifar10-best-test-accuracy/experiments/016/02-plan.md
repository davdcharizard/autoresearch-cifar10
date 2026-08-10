# Plan EXP-016: BF16-Funded Width-3 Postactivation ResNet-20
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Isolated production implementation
- [x] Confirm HEAD `7c1e7d8`, baseline `94.15`, clean tracked `train.py`, sole idle NVIDIA H20, and no stale run-log variant.
- [x] Create `autoresearch/maximize-cifar10-best-test-accuracy-016` from the integration branch.
- [x] Change only `train.py`: set `WIDTH_MULTIPLIER = 3`, assert CUDA BF16 support, and wrap only training forward plus cross-entropy in CUDA BF16 autocast.
- [x] Keep FP32 model/master parameters, backward outside autocast, ordinary SGD without scaler, and unmodified FP32 evaluator.
- [x] Pass syntax, diff-scope, structural, parameter-count, optimizer, data, schedule, lifecycle, and evaluator checks.

### Milestone 2: Disposable semantic and numerical preflight
- [x] Add ignored experiment-only preflight utilities under `experiments/016/`; do not add production hooks or change any tracked file besides `train.py`.
- [x] Submit the completed controller sources and production diff to external Claude for a read-only implementation-addendum review before trusting any result; retry/pause on harness failure, with no fallback reviewer.
- [ ] Compare cloned width-3 FP32 and BF16 arms over at least 20 paired production batches covering hard and CutMix targets, then 200 paired production-distribution training steps.
- [ ] Verify FP32 persistent state, expected autocast dtypes, finite values, close loss/logit/gradient/update behavior, accumulated BN-state and FP32-evaluation alignment, and no skipped updates or collapse.

### Milestone 3: Default-TF32-aware funding gate
- [ ] Record CUDA device, PyTorch/cuDNN versions, BF16 support, and the unchanged `allow_tf32` flags in every timing result.
- [ ] Run one predeclared unscored timing-conditioning subprocess, then five balanced fresh-process triplets for A=width2/default-FP32, B=width3/default-FP32, C=width3/BF16.
- [ ] Require C to beat B by the declared BF16 funding margins and retain at least 22,863 projected updates under both the EXP010 ratio anchor and a conservative absolute candidate-step projection; do not disable TF32 or substitute an idealized FP32 control.
- [ ] Verify stable tails, stage attribution, and memory gates.

### Milestone 4: Loader, evaluator, and wall feasibility
- [ ] Measure 1,000 real strong batches with the candidate and require loader wait headroom plus correct hard/soft provenance.
- [ ] Exercise the strong-to-weak worker shutdown once and require all eight workers to stop before a hard weak batch.
- [ ] Benchmark the unchanged FP32 width-3 evaluator after one separate predeclared inference-conditioning process and project freshly measured startup + 300 counted seconds + switch + expected evaluations below 540 seconds.

### Milestone 5: One fixed-seed production run
- [ ] Reconfirm scope, GPU idleness, baseline, no stale logs, and every conjunctive preflight gate.
- [ ] Run exactly once with seed 42 under the 600-second supervisor, redirecting all output only to `run.log`.
- [ ] Monitor terse progress without streaming full output; stop only on mechanical abort criteria, never on an unfavorable accuracy trajectory.

### Milestone 6: Integrity and metric verification
- [ ] Parse the ten finite summary fields and require 300 counted seconds, total below 600 seconds, 2,412,730 parameters, and the unchanged protocol invariants.
- [ ] Compare `best_test_acc` with the moving 94.15 baseline; formal improvement requires at least 94.25.
- [ ] Record actual exposure, phase trajectory, NLL, evaluation count, CutMix rate, worker lifecycle, memory, and wall time for analysis; never rerun a mechanically valid result.

## Code Changes
- **`train.py`**: change `WIDTH_MULTIPLIER` from `2` to `3`, producing channels `48/96/192` and exactly 2,412,730 parameters while preserving all nine postactivation blocks, Option-A shortcuts, initialization, pooling, and classifier design.
- **`train.py`**: after selecting CUDA, fail closed unless CUDA and `torch.cuda.is_bf16_supported()` are available; emit one provenance line containing BF16 support and the unchanged TF32 flags.
- **`train.py`**: wrap only `outputs = model(inputs)` and `loss = F.cross_entropy(outputs, targets)` in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. Leave input/model construction, `loss.backward()`, SGD, synchronization, timer, evaluation, logging, and lifecycle unchanged.
- **Ignored experiment artifacts**: create disposable semantic/timing controllers under `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/016/`. They may import the candidate module and emit JSON/Markdown evidence, but cannot modify production behavior or tracked files.

## Configuration Changes
- `WIDTH_MULTIPLIER`: `2 -> 3` to test the next capacity point in the locally successful width family.
- Training forward/loss precision: default FP32/TF32 path -> CUDA BF16 autocast; FP32 master state, backward placement, optimizer, and evaluation remain fixed.
- Parameters: `1,073,962 -> 2,412,730`; batch remains 128.
- Unchanged: seed 42; SGD LR `0.1`; momentum `0.9`; coupled all-parameter decay `1e-4`; 80% hold then `0.01`-to-`1e-4` cosine; N1/M7; p=0.5 alpha-1 CutMix; weak hard-label tail; workers; timer; checkpoints; evaluator.
- No GradScaler, manual model/input casting, channels-last, compilation, fused optimizer, altered TF32 flag, alternate width, or fallback precision.

## Execution Environment
- Method: local single-GPU execution from the project root.
- Resources: exactly one idle NVIDIA H20, approximately 97,871 MiB, pinned with `CUDA_VISIBLE_DEVICES=0`; no other GPU process may materially load it during timing or production.
- Estimated runtime: 4-8 minutes for disposable numerical/timing/loader gates and about 5.5-9 minutes for the single production run; each individual command has a hard timeout.
- Log output: preflights write bounded reports under `experiments/016/`; production uses only `run.log` with no `tee`. Poll only `tail -n 5`, summary keys, or process state.
- Tool skill: none; execution is local.

## Abort Criteria
- Before production, abort EXP016 as a preflight no-go if any capability, semantic, dtype/state, numerical, funding, exposure, stage-attribution, stability, memory, loader, lifecycle, or wall-projection gate fails. Do not substitute another operating point.
- Abort on any non-finite logit, loss, gradient, parameter, BN buffer, optimizer state, or skipped update in the paired numerical gate.
- Abort if C does not achieve median step time `<=0.86957x` B, at least `1.12x` speedup over B in every triplet, `(forward+backward) <=0.85x` B, backward `<=0.90x` B, or at least 90% of absolute stage savings from forward/backward.
- Abort if either `floor(26,898 * mean_step_A / mean_step_C)` or the conservative absolute projection `floor(300 / (1.025 * mean_step_C_seconds))` is below 22,863 steps, C is slower than about `1.17647x` A, C p95 exceeds `1.25x` A median, trial-mean CV is at least 3%, ratio CV is at least 2%, or an order trend reverses the funding/exposure conclusion.
- Abort if C peak allocation reaches 2 GiB, allocation grows monotonically, median loader wait reaches 10% of candidate step time, p95 loader wait reaches 20%, workers fail to stop, or projected total wall reaches 540 seconds.
- During production, terminate only for crash, non-finite output, missing progress beyond the measured startup envelope, GPU/resource fault, lifecycle/protocol fault, or the 600-second timeout. Do not early-stop for low accuracy or underfit diagnostics.
- Never rerun a mechanically valid production result. A repair/retry is permitted only for an independently demonstrated harness or environment defect that does not alter the reviewed candidate; no fallback reviewer or experiment is permitted.

## Verification Protocol

### Verification Procedure

1. **Baseline, branch, scope, and GPU (30-second timeout).** Run:
   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv
   git rev-parse --short HEAD
   git status --short --branch
   nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
   ```
   Require baseline `94.15` at `7c1e7d8`, only the known untracked `data/`, and exactly one idle H20 with approximately 97,871 MiB. Remove only known completed log variants with `rm -f run.log`; never clean `data/`.

2. **Static and structural checks (60-second timeout).** Run:
   ```bash
   git diff --check
   uv run python -m py_compile train.py
   git diff --name-only
   CUDA_VISIBLE_DEVICES=0 uv run python - <<'PY'
   import torch
   import train
   m = train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER).cuda()
   assert train.WIDTH_MULTIPLIER == 3
   assert sum(p.numel() for p in m.parameters()) == 2_412_730
   assert all(p.dtype == torch.float32 for p in m.parameters())
   assert torch.cuda.is_bf16_supported()
   print(torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
   print("STRUCTURE_PASS")
   PY
   ```
   Require only `train.py` in the tracked diff, exact structure/count/state, and no modifications to evaluator, seed, augmentation, schedule, timer, or lifecycle. Inspect `git diff -- train.py` directly.

3. **Controller fidelity and external addendum review.** Before running the controllers, inspect their full source and require that they import the candidate definitions, keep contiguous NCHW, leave `torch.backends.cudnn.benchmark`, deterministic settings, TF32 flags, matmul precision, and all other backend state unchanged, reuse identical pinned CPU workloads per paired/triplet comparison, and time the exact production interval from immediately before pinned nonblocking H2D through LR calculation, zero-grad, forward/loss, backward, SGD, and final synchronize. They may not compile, use channels-last, force algorithms, omit H2D, pre-stage scored inputs, or add candidate-only warmup. Save the production diff plus controller-source bundle and run the mandatory external Claude implementation-addendum review with the plan-critic prompt. Record the successful review in `02-plan-review-implementation-addendum.md`; on non-zero/empty output, retry the Claude harness or pause for credentials, never self-review or fall back.

4. **Paired production-distribution numerics (240-second timeout).** Create the ignored preflight controller during execution, then run:
   ```bash
   CUDA_VISIBLE_DEVICES=0 timeout 240s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/016/preflight_bf16.py
   ```
   Require `NUMERICAL_GATE_PASS` and its JSON evidence. Across hard and soft targets require BF16/FP32 relative loss error `<=2%`, logit cosine `>=0.995`, gradient cosine `>=0.99`, gradient/update norm ratios in `[0.90,1.10]`, update cosine `>=0.99`, no more than +1 percentage point exactly-zero gradients, matching BN counters, and normalized running-stat differences `<=2%`. Recheck BN counters and normalized running means/variances after all 200 paired steps, still requiring `<=2%`; then put both models in FP32 eval mode on at least five held-out real batches and require finite outputs, candidate/control logit cosine `>=0.98`, and candidate FP32-eval loss no greater than `1.15x` control. Across training require no BF16 loss above `2x` paired FP32, no candidate-only concentration above 95%, and no skipped update. Do not recalibrate BN. Short loss/eval alignment is a collapse veto, not a selection metric.

5. **Three-arm timing and funding (480-second timeout).** The externally reviewed ignored controller must spawn fresh child processes, use exactly one unscored timing-conditioning subprocess for this benchmark group, then run five order-balanced A/B/C triplets with 100 warm steps and at least 500 measured synchronized complete steps per arm:
   ```bash
   CUDA_VISIBLE_DEVICES=0 timeout 480s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/016/timing_bf16.py
   ```
   Require `TIMING_GATE_PASS`, raw trial JSON, identical recorded backend/layout state in all arms, all timing/stage/exposure/CV/p95 gates from Abort Criteria, both projected-step formulas `>=22,863`, and peak allocation `<2048 MiB`. Use mean complete-step time for projections; medians/p95 remain stability diagnostics. A is width2 default FP32; B is width3 default FP32 with whatever TF32 behavior the accepted environment actually uses; C is width3 BF16. Never disable TF32 to make C look faster.

6. **Loader/lifecycle/wall gate (240-second timeout).** Run controller modes:
   ```bash
   CUDA_VISIBLE_DEVICES=0 timeout 180s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/016/preflight_bf16.py --loader
   CUDA_VISIBLE_DEVICES=0 timeout 60s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/016/preflight_bf16.py --eval-wall
   ```
   Require 1,000 production strong batches, correct hard/soft targets, realized CutMix in `[45%,55%]`, median/p95 iterator wait below 10%/20% of candidate step time, all eight workers stopped before a hard weak batch, unique evaluator behavior, and conservative total projection `<540s`. Compute the projection only from freshly measured width-3 evaluator/startup/switch costs and the expected evaluation count under projected candidate epochs; do not use the stale 17.3-second historical evaluator note. Permit exactly one separate predeclared unscored inference-conditioning subprocess for this benchmark group and no other unscored inference run.

7. **One production run (600-second timeout).** After reconfirming steps 1-6 and removing stale `run.log`, launch exactly once:
   ```bash
   CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1
   ```
   Do not use `tee` or stream full output. Poll the existing process every 30-60 seconds and inspect only bounded log tails. Exit code 124 is a timeout failure.

8. **Necessary-condition verification (60-second timeout).** Run:
   ```bash
   grep -E '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|startup_seconds|peak_vram_mb|num_epochs|num_steps|num_params):' run.log
   grep -E '^augmentation_switch:|eval ep' run.log
   git diff --check
   git diff --name-only
   ```
   Parse values rather than comparing formatted strings. Require all ten numeric summary fields, finite values, `training_seconds` approximately 300 as fixed-budget protocol integrity only, `total_seconds <600`, `num_params=2,412,730`, `best_test_acc >=94.25`, one switch near 80%, eight workers stopped, CutMix fraction `[45%,55%]`, only hard weak targets, unique evaluation epochs, and no more than one evaluation per epoch. Candidate throughput is evidenced by `num_steps`, not by the mechanically fixed training-time total. Query the moving baseline again before rendering the metric verdict.

9. **Exposure attribution and cleanup.** Candidate-mechanism support additionally requires actual `num_steps >=22,863`; a lower jittered count does not authorize a rerun and must be reported separately from the formal accuracy condition. Record switch accuracy, first weak accuracy, best/final, final NLL, strong/tail steps, evaluation count, memory, startup, and total wall versus EXP010. Preserve `run.log` through analysis, then remove it before the next experiment. On no-go/no-improvement, restore only `train.py` and return to the integration branch; never delete `data/`.

### Informational Metrics (Optional)
- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, `num_params`: parsed from the ten final `run.log` summary lines.
- Strong switch accuracy, first weak accuracy, final/best gap, and evaluation count: bounded parse of `eval ep` lines.
- Actual CutMix rate and lifecycle: `augmentation_switch` line.
- Numerical alignment and timing-stage evidence: ignored EXP016 preflight JSON reports.

# Plan EXP-010: Conservative Plateau CutMix
- **Created**: 2026-08-05

## Milestones

### Milestone 1: Implement isolated plateau CutMix
- [x] In `train.py`, import `default_collate` and torchvision v2; define fixed alpha 1.0, probability 0.5, and a module-level forkserver-safe collator.
- [x] Preserve CPU RNG state around only the CutMix gate/transform, add optional loader collation, use it only for the initial N1/M7 loader, and rebuild the weak loader with default hard-label collation.
- [x] Count strong mixed/hard target formats before the timed step and report them at the existing switch without changing model, optimizer, schedule, evaluator, or seed.
- [x] Pass compilation, Ruff, formatting, pre-commit, exact diff, installed-API, target, RNG, and lifecycle checks.

### Milestone 2: Pass fixed-time feasibility gates
- [x] On one idle H20, benchmark interleaved hard versus probability-target synchronized width-2 steps and require projected retention >=97% of EXP-007 (at least 26,329 steps).
- [x] Benchmark the exact eight-worker N1/M7 plus 50% CutMix loader after one warm epoch; require three warmed epochs each at >=120 batches/s, mixed fraction 45-55%, and <=20% epoch-time spread from their median.
- [x] Verify all eight strong workers stop, transition plus first weak batch takes <=5 seconds, and the first weak target is integer `[128]`.
- [x] Run 1,000 real mixed-loader width-2 training steps in a fresh process and require joint wall time no more than 15% above summed synchronized step time, ruling out contention hidden by isolated tests.

### Milestone 3: Execute and verify one fixed-seed run
- [x] Confirm one idle 97,871 MiB H20 and no stale log, then run once under a 600-second timeout with all output redirected to `run.log`.
- [x] Verify summary, timing, scope, one 80% switch, eight stopped workers, target-format provenance, unique evaluation epochs, exact parameter count, and no soft target after the switch.
- [x] Compare `best_test_acc` with 93.55%; improvement requires at least 93.65%. Record a switch checkpoint below 87.08% as compounded-underfit evidence, never as an adaptive retry trigger.

## Code Changes
- **`train.py`**: Add a module-level `v2.CutMix(alpha=1.0, num_classes=10)` and `cutmix_collate`. The collator default-collates a batch, saves/restores CPU RNG with `torch.random.fork_rng(devices=[])`, and with fixed probability 0.5 returns CutMix images/probability targets; otherwise it returns the unchanged images/integer targets.
- **`train.py`**: Extend `make_train_loader` with optional `collate_fn`; pass CutMix only to the strong N1/M7 loader and omit it for the weak loader. Count target dimensionality before `t0`, assert weak targets remain one-dimensional after switching, and add mixed/hard counts to the single existing switch log.

## Configuration Changes
- `CUTMIX_ALPHA = 1.0`: canonical evidence-backed setting; no tuning.
- `CUTMIX_PROBABILITY = 0.5`: bounds compounded N1/M7 plus regional mixing to about half the plateau and 40% of total steps.
- Unchanged: width 2, 1,073,962 parameters, batch 128, hard weak tail, N1/M7 strength/order, 80% phase/LR boundary, all-parameter decay `1e-4`, momentum, LR schedule, seed 42, timer, and evaluator cadence.
- Difference from EXP-006: compose rather than replace N1/M7; donor pixels carry information; labels follow actual pasted area; only half of strong batches mix; width 2 and the full hard tail remain.

## Adversarial Review Response
- Mandatory Claude plan review completed with exit code 0; no fallback reviewer was used.
- Concern 1 is an unavoidable protocol limit: a 93.65-93.75 result is accepted by the user-defined gate but will be reported as weak single-run causal evidence, never as a precise effect estimate. Seed rerolls or replication are prohibited.
- Concerns 2-3 and 6 are the experiment's research question, not preflight-correctness failures. The 80% tripwire occurs after roughly 240 counted seconds; finishing the final 60 seconds preserves the primary metric and tests whether the hard tail repairs soft-target underfit. The verdict analyzes the net CutMix/exposure effect rather than claiming those components are separately identified.
- Accepted concern 4: add a joint 1,000-step real-loader/model contention gate and conservative wall-over-counted ratio, supplementing isolated worker and CE timing.
- Accepted concern 5: verify the installed public behavior empirically with controlled constant images and distinct labels. A semantics mismatch is a candidate no-go; runtime provenance additionally records target format, while no claim depends on a private roll-by-one implementation detail.

## Execution Environment
- Method: local `timeout 600s uv run train.py > run.log 2>&1` after every preflight gate passes.
- Resources: exactly one idle NVIDIA H20 with approximately 97,871 MiB; eight existing DataLoader workers.
- Estimated runtime: 333-370 seconds total; 300 seconds counted training; predicted 26,329-27,200 steps.
- Log output: full stdout/stderr only in project-root `run.log`; monitoring uses bounded targeted patterns and process state, never `tee`.
- Tool skill: none; local execution.

## Abort Criteria
- Do not launch the full run if target/RNG/lifecycle correctness fails, any warmed strong-loader epoch is below 120 batches/s, realized mixed fraction is outside 45-55%, transition exceeds five seconds, or projected steps are below 26,329.
- Do not launch if the integrated 1,000-step contention test has wall time more than 15% above summed synchronized step time or projects total runtime above 540 seconds.
- During execution stop on wrong/busy GPU, traceback, OOM, non-finite loss, malformed/leaked target format, failed worker shutdown, or a process exceeding 600 seconds.
- Do not stop or tune merely because the 80% clean checkpoint is low. A value below 87.08% diagnoses compounded underfit, but the fixed hard tail must finish and the operating point must not be changed or rerun.
- Exactly one valid seed-42 run; no alpha/probability adjustment, custom loss, GPU relocation, or seed reroll after observations.

## Verification Protocol

### Verification Procedure
1. Query the results index with `exp-index.sh baseline`; require baseline 93.55 at `8faf0f3`, hence a 93.65 success threshold.
2. Run compilation, Ruff, formatting, pre-commit, and `git diff --check`; require `git diff --name-only` to contain only `train.py` and confirm all excluded mechanics are unchanged.
3. In a disposable fresh process, force hard and mixed collator outcomes and require image `[128,3,32,32]`, hard integer target `[128]`, mixed finite nonnegative float target `[128,10]`, row sums 1, and donor target mass matching the observed pasted fraction within `1/1024` on controlled constant images with distinct labels. This empirically validates the installed transform's public area-label semantics without assuming a private pairing implementation. Require finite CE/gradients for both formats and byte-identical CPU RNG state before/after a collator call. Exercise distribution through the actual transformed loader, where per-sample transforms advance worker RNG between collations; do not require repeated direct calls from an otherwise unchanged RNG state to vary.
4. Benchmark one warm plus three timed full strong-loader epochs in a fresh process. Require 390 batches/epoch, each timed epoch >=120 batches/s, <=20% deviation from their median, mixed fraction 45-55%, clean shutdown of eight workers, transition/first weak batch <=5 seconds, and hard integer first weak targets.
5. On one idle H20, interleave at least 500 warmed synchronized hard and probability-target width-2 steps including H2D, forward, CE, backward, SGD, and synchronize. Compute `ratio = 0.8 * (2*t_hard/(t_hard+t_soft)) + 0.2`; require ratio >=0.97, projected steps `floor(27143*ratio) >=26329`, finite loss/gradients, and normal memory.
6. In a fresh process, run 1,000 consecutive real N1/M7+CutMix loader and width-2 optimizer steps after warmup while separately accumulating iterator-wait wall time and the existing synchronized `t0` step time. Require total joint wall time <=1.15x summed step time, no malformed target, stable mixed fraction, clean worker shutdown, and conservative projected full total <=540 seconds.
7. Confirm exactly one idle H20 with `nvidia-smi`, no `run.log` variant, then execute `timeout 600s uv run train.py > run.log 2>&1`. Exit 0 is required; 124 is timeout failure; any other non-zero exit is a crash.
8. Parse all ten final summary fields. Require 300.0 counted seconds, total below 600, 1,073,962 parameters, one `randaugment+cutmix->base` switch near 80%, eight workers stopped, strong mixed fraction 45-55%, and no post-switch two-dimensional target assertion failure.
9. Parse evaluation epochs and require every epoch unique. Parse `best_test_acc` and require >=93.65 for improvement; a completed lower value is valid no-improvement and cannot be rerun. If the gain is only 0.10-0.20 points, record that it satisfies the protocol but is not strong evidence beyond single-run trajectory noise.
10. Compare actual steps with EXP-007's 27,143 and the 26,329 mechanism floor. Compare the switch checkpoint with 90.08% and underfit marker 87.08%, first weak checkpoint with 92.96%, and final loss/accuracy with 0.2196/93.49%; these diagnostics cannot override the primary gate.

### Informational Metrics (Optional)
- Final summary metrics: targeted final lines in `run.log`.
- Target provenance: switch log mixed/hard counts and realized fraction.
- Trajectory: evaluation/switch lines for strong checkpoint, first weak result, best epoch, final slope, and best/final gap.
- Feasibility: warmed worker batches/s, transition seconds, hard/soft step times, projected/actual retention, and peak VRAM preserved in `03-execute.md`.

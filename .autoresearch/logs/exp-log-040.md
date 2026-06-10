# EXP-040: cuDNN conv-algorithm autotuning (throughput → more epochs)

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-040.md
- **Plan**: plans/plan-040.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-040
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Single-line change to `train.py` per plan-040 Milestone 1: added `torch.backends.cudnn.benchmark = True`
inside `main()` immediately after the `print(f"Device: {device}")` line (now L156), before model
construction and `torch.compile`. No other change. AST parses OK; exactly one `cudnn.benchmark = True`
line; `git diff --name-only` = train.py only. The model, params (4,299,866), data, optimizer, LR
schedule, seed (42), and eval are all untouched — so any change in num_epochs is purely throughput, and
any best_test_acc change at a different epoch count isolates the convergence-bound-vs-epoch-saturated
question.

### Surprises & Discoveries
- (none at implementation time — the change is a standard one-line backend flag.)

### Decisions
- Placed the flag inside `main()` (not module level) so it is scoped to the run and sits adjacent to
  device setup, before cuDNN first selects conv algorithms (at model build / first compiled forward).
  Functionally equivalent to module-level for a single-process run.

## Experimental Adjustments

<!-- Appended incrementally over runs. -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID / background task — recorded at launch)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09

Description:
- Training the unchanged WideResNet-k4 recipe with `cudnn.benchmark=True` added. Launched on idle H20
  GPU 1 (`CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1`). Expect dt ≤ ~8ms (cuDNN may pick
  faster conv algos for the fixed 128×3×32×32 shapes) → ≥ ~91 epochs in 300s. Hypothesis: if the net is
  convergence-bound, extra epochs lift best_test_acc above the bar 96.32. Key diagnostic: num_epochs and
  steady dt vs baseline (~91 ep / ~8ms); final_test_loss vs 0.195.

Observations:
- Clean, uncontended run: dt distribution = 718 steps @ 8ms + 6 @ 9ms, only warmup outliers (13/30ms),
  no sustained contention band (source: run.log dt samples). 94 epochs (vs baseline ~91 — within
  run-to-run throughput jitter, NOT a real throughput gain).
- **THE KEY DIAGNOSTIC**: steady dt is UNCHANGED at 8ms — `cudnn.benchmark=True` did NOT lower per-step
  dt. However peak_vram_mb jumped 491→971 (≈2×), proving cuDNN benchmark mode DID select different
  (higher-memory) conv algorithms — they were simply no faster. The conv dt floor for these fixed shapes
  was already reached by `torch.compile(reduce-overhead)`. → outcome (c) from the plan: cuDNN-autotune is
  a throughput no-op here.
- final_test_loss 0.2039 (vs baseline 0.195) — mildly inflated; run landed in the low side of the
  ±0.25pp augmented-base noise band.

Key Metrics:
- best_test_acc: 95.91% (source: run.log `best_test_acc:`)
- final_test_loss: 0.2039 | num_epochs: 94 | num_steps: 36422 | total_seconds: 433.1 |
  num_params: 4,299,866 (UNCHANGED) | peak_vram_mb: 971.3 (≈2× baseline 491 — higher-mem conv algos)
  (source: run.log summary block)

## Verification Results

### Conditions Checked

- **Primary metric clears bar (NECESSARY)**: best_test_acc 95.91% < bar 96.32 (and < baseline 96.22 by
  0.31pp) → **FAIL** → no-improvement.
- **Clean completion within budget (NECESSARY)**: summary block present, total_seconds 433.1 < 600,
  exit 0 → PASS.
- **No hard-constraint violations (NECESSARY)**: `git diff --name-only` = train.py only; seed 42
  unchanged; num_params 4,299,866 UNCHANGED (model untouched); eval lines 94 == num_epochs 94
  (≤1 eval/epoch) → PASS. Fairness gate: dt steady 8ms, uncontended, 94 ep ≥ baseline — fair run.

Verdict: **no-improvement** — clean, fair run; cuDNN benchmark was a throughput no-op (dt unchanged), so
no extra epochs were bought; acc 95.91 within the low-noise band, below baseline.

### Informational Metrics

## Errors & Dead Ends

<!-- Append only. -->

## Human Notes

> (none — autopilot)

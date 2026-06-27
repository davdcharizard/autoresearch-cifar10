# EXP-021: Step-time engineering — torch.compile(mode="max-autotune") + SGD(fused=True)

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-021.md
- **Plan**: plans/plan-021.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-021
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (Run 4 clean, completed; necessary condition 1 not met — 96.51 < 96.81 despite +10 epochs)

## Implementation Notes

### Summary

Exactly the plan's two-argument diff, nothing else: `torch.compile(model, mode="max-autotune")` (was default mode) and `fused=True` added to the existing `optim.SGD` call (param groups, lr, momentum, nesterov untouched). `git diff` confirms 2 insertions / 1 deletion in train.py, both inside `main()`, zero constants changed. Syntax checked via ast.parse. No GPU pre-validation possible for these arguments on CPU (fused SGD requires CUDA); validation happens at launch via the params print (must stay 4,286,026 — unchanged model) and the step-100 dt gate.

### Surprises & Discoveries

- None at implementation time — both arguments are documented API surface.

### Decisions

- None beyond the plan at first implementation. After Run 1's crash (see Errors & Dead Ends): added `torch.compiler.cudagraph_mark_step_begin()` before each model invocation in BOTH the warmup loop and the timed training loop — the fix the runtime error itself prescribes. Placed INSIDE the timed region in the training loop (after t0) so its ~µs cost is honestly accounted in dt. This is the single code-error retry; the reduce-overhead performance fallback remains unconsumed.
- The plan's fallback ladder (no steps within 150s → reduce-overhead; dt > 23.5ms → reduce-overhead; one fallback retry total) stays armed.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: bvjccaak3 (composite background script: pre-check + train + startup gate + inline watchdog with per-window dt echo)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log
- **WandB**: N/A
- **Status**: running
- **Started**: 2026-06-10 (GPU 0 confirmed free at launch by the script's pre-check)
- **Ended**: (pending)

Description:
- Single 300s-budget run of the byte-identical baseline recipe with two execution-speed arguments: max-autotune compilation (Triton autotuned kernels + CUDA graphs) and fused SGD. Tests the only in-project validated +pp mechanism — throughput at fixed hyperparameters (EXP-006: +25 epochs → +0.48pp). Expected: params 4,286,026 (unchanged model), startup 40–150s (autotuning), windowed dt ≤ 21.0ms if the hypothesis holds (→ ≥147 epochs), VRAM 1.7–2.5GB (cudagraph buffers), best_test_acc ≥ 96.81. Falsifier: dt > 21.5ms at step 100 means the bar is unreachable; dt > 23.5ms or startup > 150s triggers the one reduce-overhead fallback retry.

Observations:
- Launch 11:35:37, GPU 0 confirmed free (source: task bvjccaak3 output L1). run.log immediately shows max-autotune at work: per-conv "AUTOTUNE convolution" blocks with Triton template choices, and Triton beating the ATen/cuDNN fallback on the stem conv (`best_kernel: triton_convolution2d_4`, 0.187ms) — the autotuning the hypothesis depends on is genuinely happening, with kernel-selection wins already visible (source: run.log head).

Key Metrics:
- None — crashed in warmup before the timed loop (rc=1 at ~105s; traceback in Errors & Dead Ends). Autotuning itself completed and Triton kernels won several convs before the crash.

### Run 2

Metadata:
- **Job ID**: bdry7tizt (same composite launcher; code-error retry 1 of 2)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log (overwritten per run)
- **WandB**: N/A
- **Status**: failed (rc=1)
- **Started**: 2026-06-10 11:39:43
- **Ended**: 2026-06-10 11:40:13 (crash ~30s in, warm cache)

Description:
- Identical experiment with `torch.compiler.cudagraph_mark_step_begin()` added before each model invocation in both loops (the fix the Run 1 error message prescribes).

Observations:
- SAME RuntimeError at `warm_loss.backward()` despite mark_step_begin — root cause refined: gradients accumulate ACROSS the 3 warmup iterations (zero_grad was only after the loop), so iteration N's backward accumulates into `.grad` tensors that are cudagraph-pool outputs of iteration N−1's backward — exactly the dead-tensor access the guard forbids (source: run.log traceback, Run 2).

Key Metrics:
- None — crashed in warmup.

### Run 3

Metadata:
- **Job ID**: b3o3o21xt (same composite launcher; code-error retry 2 of 2)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log (overwritten per run)
- **WandB**: N/A
- **Status**: completed BUT contaminated (pre-condition gate failed — see Observations)
- **Started**: 2026-06-10 11:41:16
- **Ended**: 2026-06-10 11:50:16 (TRAIN_EXIT rc=0)

Description:
- Adds per-iteration `optimizer.zero_grad(set_to_none=True)` INSIDE the warmup loop (drops grad references between iterations so no accumulation into dead cudagraph-pool tensors), keeping mark_step_begin from Run 2. Warmup invariant preserved (no optimizer.step → weights unchanged). If this still crashes, the remaining in-pocket fallback is `mode="max-autotune-no-cudagraphs"` (keeps Triton autotuning, drops graphs — reduce-overhead is NOT a fallback for this failure since it also uses cudagraphs).

Observations:
- **The cudagraph fixes worked and the speedup is REAL**: windows ~20.2–21.4ms (vs baseline 22.4), startup 10.8s (warm autotune cache), 145 epochs / 14,042 steps (+6 vs 139), params 4,286,026 unchanged, VRAM 1728.2MB (+115MB cudagraph buffers) (source: task b3o3o21xt WIN lines; run.log summary).
- **Pre-condition gate FAILED**: post-hoc profile 6 of 279 windows >30ms (watchdog's 15s sampling saw only 2 — 33.0/34.5ms at steps 11700/11900); a contention episode spanned steps ~11500–12200 (~p 83–87%), costing roughly 5–6s of budget ≈ 2 epochs. Mean win 21.4ms; expected-epochs consistency holds (145.8 predicted vs 145 actual) (source: profile command).
- **Accuracy signal (recorded but NOT analyzable per protocol)**: best_test_acc 96.41 = final (converged plateau 96.35–96.41 over the last five evals), −0.30pp vs baseline despite +6 epochs. Mid-run trajectory ran below the baseline family well BEFORE the contention episode (ep60 best 87.98 vs 88.50; ep100 90.39/92.61 vs 93.00/93.16) — suggests the deficit is not contention-driven (contention ≈ 2 epochs ≈ 0.04pp by EXP-006 arithmetic), but a clean rerun is required before drawing that conclusion (source: run.log eval trail).

Key Metrics (contaminated run — for reference only):
- best_test_acc: 96.41% @ ep 145 (final = best) | baseline 96.71, bar 96.81
- mean win 21.4ms | 145 epochs | 14,042 steps | startup 10.8s | total 530.1s | VRAM 1728.2MB | params 4,286,026

### Run 4

Metadata:
- **Job ID**: bv43v08x5 (contention rerun, does not consume code-error retries)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log (overwritten per run)
- **WandB**: N/A
- **Status**: completed (the analyzable run)
- **Started**: 2026-06-10 11:52:05 (GPU 0 confirmed free by pre-check)
- **Ended**: 2026-06-10 12:01:20 (TRAIN_EXIT rc=0)

Description:
- Byte-identical rerun of Run 3 (no code changes) after the contention episode violated the ≤2-slow-windows analyzability gate. Warm autotune cache → startup ~11s expected. Purpose: a clean profile to make the accuracy result attributable — Run 3's −0.30pp at +6 epochs suggests max-autotune/fused numerics may COST accuracy at equal hyperparameters, but that claim needs an uncontaminated run.

Observations:
- Pristine: 0 of 286 windows >30ms, mean 20.9ms; 149 epochs / 14,384 steps — exactly the 149.3 projection; startup 10.4s; total 537.4s; VRAM 1728.2MB; params 4,286,026 (source: task bv43v08x5 WIN lines; run.log; profile command).
- **Throughput hypothesis fully delivered**: dt 22.4 → 20.9ms (−6.7%), epochs 139 → 149 (+10, ≥147 predicted). The mediating variable moved exactly as hypothesized (source: profile + run.log summary).
- **Accuracy did NOT follow — the EXP-006 conversion broke**: best_test_acc 96.51 (−0.20pp vs baseline) with a proper converged plateau (final eight evals 96.39–96.51, final = best). Mid-run trajectory ran below the baseline family long before convergence (ep100 91.42/92.73 vs ~93.0/93.2 family) — replicating Run 3's deficit (96.41 at 145 contaminated epochs; the two runs bracket 96.41–96.51). The +10 epochs (~+0.2pp by EXP-006 arithmetic) were more than cancelled by what appears to be a numerics cost of the faster execution regime: max-autotune's Triton kernels (different reduction orders; ALLOW_TF32 choices visible in autotune logs), cudagraphs, and/or fused SGD (source: run.log eval trail; Run 3 trail).

Key Metrics:
- best_test_acc: 96.51% @ ep 146/149, final = best (source: run.log summary + eval trail) — baseline 96.71, bar 96.81: −0.20pp
- mean win 20.9ms (0/286 >30ms) | num_epochs: 149 | num_steps: 14,384 (source: profile, run.log)
- total_seconds: 537.4 | training_seconds: 300.0 | startup_seconds: 10.4 (warm autotune cache) (source: run.log)
- peak_vram_mb: 1728.2 (+115 vs baseline — cudagraph static buffers) | num_params: 4,286,026 unchanged (source: run.log)
- final_test_acc: 96.51% | final_test_loss: 0.1883 (source: run.log)

## Verification Results

<!-- Filled after the experiment completes successfully.
     If ANY necessary condition fails, remaining conditions are not evaluated. -->

### Conditions Checked

All conditions evaluated on Run 4 (the clean run); Run 3 excluded per the contamination protocol.

- **Pre-condition (contention sanity / signature consistency)**: 0 of 286 windows >30ms, mean 20.9ms; num_epochs 149 vs projection 149.3 (within ±3) — CLEAN, analyzable (source: profile command output).
- **Condition 1 — best_test_acc ≥ 96.81 (baseline 96.71 + 0.1)**: `grep "^best_test_acc:" run.log` → 96.51%. **FAILED** (−0.20pp vs baseline, replicated by Run 3's 96.41).
- **Condition 2 — total ≤ 600s, rc=0**: skipped — aborted after prior failure (informally: 537.4s, rc=0 — would have passed).
- **Condition 3 — eval at most once per epoch**: skipped — aborted after prior failure (informally: 149 eval lines = 149 epochs — would have passed).

### Informational Metrics

Not collected per protocol (necessary condition failed). Informal values from run.log for the report: mean dt 20.9ms (−6.7%); 149 epochs (+10); peak_vram_mb 1728.2 (+115); params 4,286,026 unchanged; startup 10.4s warm / would be ~90–120s cold-autotune.

## Errors & Dead Ends

<!-- Append only. Never delete. Agent reads this before proposing any next action.
     Include source pointers for traceability. -->

### 2026-06-10 — Run 1 crash: cudagraphs overwrite guard in the warmup loop
- Error: `RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run` at `warm_loss.backward()` (train.py L196), after autotuning completed and params printed correctly (4,286,026)
- Root cause: max-autotune enables CUDA graphs; the 3-iteration compile-warmup loop runs forward+backward WITHOUT optimizer.step(), so inductor's cudagraph manager sees no step boundary and the next forward overwrites pool tensors the previous backward still references — exactly the case the error message documents
- Source: run.log L136–167 (traceback); task bvjccaak3 (TRAIN_EXIT rc=1 at 11:37:22, ~105s in)
- Do NOT retry: the same warmup pattern under any cudagraph-enabling compile mode without `torch.compiler.cudagraph_mark_step_begin()` before each invocation; fix applied to both the warmup loop and the timed training loop (the call is ~µs, inside dt for honesty)

### 2026-06-10 — Run 2 crash: same guard, deeper cause (cross-iteration grad accumulation)
- Error: identical `RuntimeError ... overwritten by a subsequent run` at `warm_loss.backward()` WITH mark_step_begin in place
- Root cause: warmup iterations accumulate gradients (zero_grad was only after the loop); iteration N's backward accumulates into `.grad` tensors that are cudagraph-pool outputs of iteration N−1's backward, which mark_step_begin declared dead
- Source: run.log traceback (Run 2); task bdry7tizt (rc=1 at 11:40:13)
- Do NOT retry: any cross-iteration grad-tensor reference under cudagraphs; fix = `optimizer.zero_grad(set_to_none=True)` inside the warmup loop after each backward. If cudagraphs still fail → use `mode="max-autotune-no-cudagraphs"`, NOT reduce-overhead (which also enables graphs)

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->

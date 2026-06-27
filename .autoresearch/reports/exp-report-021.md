# Report EXP-021: Step-time engineering — torch.compile(mode="max-autotune") + SGD(fused=True)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-021.md
- **Plan**: plans/plan-021.md
- **Log**: logs/exp-log-021.md

## Goal

Maximize `best_test_acc` (%) of the CIFAR-10 ResNet within the fixed 300s training budget (higher is better). Baseline at experiment time: **96.71%** @ 1990397. Specific question: does the only in-project validated +pp mechanism — throughput at byte-identical hyperparameters (EXP-006: +25 epochs → +0.48pp) — still pay at its last untried tier (max-autotune kernels + CUDA graphs + fused SGD)?

## Idea & Hypothesis

Chosen idea: two execution-speed arguments, zero hyperparameter changes — `torch.compile(model, mode="max-autotune")` (Triton autotuned conv/matmul templates + CUDA graphs; EXP-006 used default mode) and `optim.SGD(..., fused=True)` (single fused update kernel inside dt). Brainstorm-021 also corrected exp-report-020's follow-up premise: loader stalls live OUTSIDE the timed budget (t0 is set after the loader yields), so GPU-side augmentation adds zero epochs — dt itself is the only honest throughput lever. Hypothesis: dt 22.4 → ≤21.0ms (≥6%) → ≥147 epochs → best_test_acc ≥ 96.81 by the EXP-006 conversion (~+0.02pp/epoch).

## Approach

Two-argument diff plus two cudagraph-compatibility fixes discovered at runtime: `torch.compiler.cudagraph_mark_step_begin()` before each model invocation (warmup + timed loop, ~µs, placed inside dt for honest accounting) and per-iteration `optimizer.zero_grad(set_to_none=True)` inside the compile-warmup loop (cross-iteration grad accumulation into cudagraph-pool tensors trips the overwrite guard). All training constants, augmentation, schedule, and update math byte-identical to baseline.

## Execution

Four runs. Run 1 (b5...crashed): cudagraphs overwrite guard at `warm_loss.backward()` — fixed with mark_step_begin (code retry 1). Run 2: same guard, deeper cause — cross-iteration grad accumulation in the warmup loop — fixed with per-iteration zero_grad (code retry 2). Run 3 (b3o3o21xt): completed rc=0, speedup real (mean 21.4ms, 145 epochs, best 96.41) but CONTAMINATED — 6/279 windows >30ms from a mid-run contention episode (steps ~11500–12200); excluded per protocol. Run 4 (bv43v08x5, the analyzable run): pristine — 0/286 windows >30ms, mean 20.9ms, 149 epochs / 14,384 steps (exactly the 149.3 projection), startup 10.4s (warm autotune cache; cold autotune in Run 1 took ~90s), total 537.4s, VRAM 1728.2MB (+115MB cudagraph buffers), params 4,286,026 unchanged.

## Results

- **Primary metric**: best_test_acc = 96.51% (baseline: 96.71, delta: −0.20pp, −0.21%); bar was 96.81. Replicated by Run 3: 96.41 at 145 contaminated epochs.
- **Observations**: The mediating variable moved EXACTLY as hypothesized — dt −6.7%, +10 epochs, every signature consistent. But the trajectory ran below the baseline family from mid-schedule (ep100 91.42/92.73 vs ~93.0/93.2) and converged flat at −0.20pp (final eight evals 96.39–96.51, final = best). The two halves of the hypothesis decoupled: throughput ✓, conversion ✗.
- **Analysis**: This breaks the EXP-006 conversion law as previously stated. EXP-006's gain came from throughput WITHIN the same arithmetic (default inductor, cuDNN kernels); EXP-021's throughput came WITH different arithmetic — Triton autotuned kernels (different reduction orders, ALLOW_TF32 template choices visible in the autotune logs), CUDA-graph execution, and a fused optimizer kernel. At byte-identical hyperparameters the new numerics produced a measurably worse trajectory (~−0.4pp mid-run) that +10 epochs (~+0.2pp by the conversion arithmetic) could not repay. The conversion law needs an amendment: extra epochs convert only when the per-step computation is numerically equivalent, not merely mathematically equivalent — execution arithmetic is part of the tuned recipe. The certified optimum (EXP-006 recipe) is now shown to be tuned TO its execution regime: both directions away from default-compile numerics (eager was slower in EXP-000…005; faster-but-different is worse here) lose. Which of the three components (Triton kernels, cudagraphs, fused SGD) carries the cost is unresolved — see Unexplored Avenues. Sixteen consecutive misses; the throughput axis is now closed at its last tier unless a numerics-preserving speedup exists.
- **Key Learning**: Throughput converts to accuracy only when the faster execution is numerically equivalent — max-autotune/cudagraphs/fused bought +10 epochs and still lost 0.20pp because the changed arithmetic degraded the trajectory more than the epochs repaid.

## Verification

- **Conditions** (evaluated on Run 4; Run 3 excluded as contaminated): pre-condition CLEAN (0/286 windows >30ms; 149 epochs = 149.3 projection); condition 1 FAILED (best_test_acc 96.51 < 96.81); conditions 2–3 skipped per first-failure stop (informally: 537.4s ≤ 600 rc=0; 149 evals = 149 epochs — both would have passed)
- **Review Notes**: trustworthy — metric matches the eval trail; the negative result REPLICATES across two independent runs (96.41/96.51); params print pins the unchanged model; epoch count exactly matches the measured dt; the speedup itself is verified by two clean projections
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (valid clean run; primary-metric necessary condition not met)

## Unexplored Avenues

- **Component isolation (default-compile + fused SGD only)**: fused SGD alone is the least likely numerics culprit (fp32 update math) and gives ~1–2% dt; an isolated probe would attribute the cost — but its standalone epoch gain (~+2–3 epochs ≈ +0.05pp) cannot clear the +0.1 bar even if free. Only worth running as diagnosis, not as a win attempt.
- **mode="max-autotune-no-cudagraphs"**: isolates Triton-kernel numerics from graph execution; same objection — the residual speedup may not clear the bar, and the trajectory cost likely lives in the Triton kernel arithmetic.
- **Numerics-preserving speedups**: anything that reduces dt while keeping cuDNN/default-inductor kernels (e.g. memory-format or allocator tweaks) — nothing concrete identified; dt 22.4ms at default numerics appears near the floor for this model/hardware.

## Next Steps

1. **Heat-constant momentum trade (0.95 + peak 0.2, lr/(1−β) = 4)** — the only never-touched recipe constant, admissible only as a compensated trade; now the last untried in-recipe candidate of any kind. Confidence: low.
2. **Synthesis check, hardened**: sixteen misses. Every axis is closed: constants (bracketed), structure (4 probes), schedule (all dims), init (both directions), capacity (both directions), batch, smoothing, shortcut topology, and now throughput-at-fixed-hparams (its numerics cost exceeds its epoch gain). The honest assessment is that 96.71 @ EXP-006 is the optimum of this recipe family under this budget within the explored intervention space; remaining ideas are low-prior multi-knob trades. Brainstorms should weigh declaring exploitation exhausted vs. continuing with explicitly low expected value. Confidence: n/a (process note).
3. **If continuing after the momentum trade**: revisit near-miss composites only if a mechanism argument exists that escapes both the deferral law AND the numerics-equivalence requirement — e.g., trajectory-variance shaping within the unchanged execution regime. Confidence: low.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->

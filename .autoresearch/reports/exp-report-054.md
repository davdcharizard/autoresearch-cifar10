# Experiment Report EXP-054: fp16 autocast + GradScaler — the finer-precision direction of the numerics axis

- **Date**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-054.md
- **Plan**: plans/plan-054.md
- **Exp-log**: logs/exp-log-054.md
- **Verdict**: no-improvement
- **Metric**: 96.54 vs baseline 96.71 (bar 96.81)

## Goal

Maximize CIFAR-10 best_test_acc (%) within the fixed 300s charged budget; higher is better. Baseline 96.71 @ 1990397; bar ≥ 96.81. σ context (EXP-027): recipe mean ≈ 96.57, σ ≈ 0.16.

## Idea & Hypothesis

The numerics axis was half-probed: EXP-021 (replicated) showed coarser/reordered arithmetic costs −0.20pp, proving trajectory quality is sensitive to kernel arithmetic at the 0.2–0.4pp scale — but the finer direction had never been measured. fp16 carries 10 mantissa bits vs bf16's 7 (ulp(1.0): 2^-10 vs 2^-7, 8× finer) at identical tensor-core throughput on H20; GradScaler manages the narrower dynamic range. External existence proof: the hlb-CIFAR10/airbench speedrun lineage trains CIFAR in fp16 at aggressive LRs. Hypothesis: if bf16's rounding noise had been costing trajectory quality, fp16 reads above the recipe mean (≥ 96.81 if ≥ +0.3). Pre-registered branches: (i) ≥ 96.81 → replicate-pair escalation (mean decides); (ii) mean band [96.41, 96.73] at family signatures → precision-null, axis closed both directions; (iii) < 96.41 / elevated scatter → negative sign; (iv) NaN/divergence → stability closure.

## Approach

~6 lines in train.py (1 file, +7/−5, 3 hunks): autocast dtype `torch.float16` at both sites (warmup + timed loop); `torch.amp.GradScaler("cuda", enabled=device.type=="cuda")`; timed loop `scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()`; warmup `scaler.scale(warm_loss).backward()` (no step/update). Printed loss is the unscaled fp32 autocast output. CPU sanity all-pass including the mechanism check (ulp ratio exactly 8×) and a GradScaler-disabled passthrough exercising the exact new call sequence. No deviations.

## Execution

Single pristine run, no retries, fp16 fully stable end-to-end (zero NaN/divergence/skip-storm events): GATES_CLEAR poll 1; D0 = 22.7ms (scaler overhead ~0.2–0.4ms over the 22.3–22.5ms family — exactly as priced); windows 22.0–23.3ms; ep1 = 36.47 in the family band; **best 96.54**, final 96.49, final_test_loss 0.1877 (family), 300.0s charged, 483.8s total, **136 epochs / 13,181 steps** (the scaler's unscale+inf-check pass cost ≈ 2–3 epochs ≈ 250 steps vs family 138–140), params 4,286,026, evals 136 ≤ 136, converged-flat tail.

## Results

A textbook branch-(ii) resolution: 96.54 sits 0.03 below the recipe mean (−0.2σ) at fully family signatures — trajectory rejoined the family, plateau at family level, family test_loss. Decomposed: the scaler's ~2.5-epoch toll prices at ≈ −0.05 by the conversion law, so the precision effect itself is ≈ 0 ± noise. The conclusion is two-sided and final:

1. **bf16's coarse rounding was never a hidden cost.** 8× finer mantissa on every autocast tensor moved nothing — the trajectory's quality at this σ resolution is insensitive to half-precision rounding noise. Combined with EXP-021 (coarser arithmetic = −0.20 replicated), the numerics axis is now measured on BOTH sides of the certified regime: the recipe sits at a plateau in precision-space, not on a slope. There is no precision dividend left to collect (fp32 autocast-off would halve throughput — dominated; tf32 already on).
2. **The gradient-noise law extends to rounding noise**: the tuned recipe's noise budget evidently saturates whatever micro-regularization bf16 rounding contributes — removing 87.5% of it (fp16) is as invisible as the EXP-036/037-class absorbed mechanisms. (Or it was simply always negligible; at this resolution the readings are equivalent.)
3. **fp16 stability is a clean engineering datum**: zero scaler skips visible (trajectory uninterrupted, plateau tight), even at PEAK_LR 0.4 with nesterov — the stability tail risk priced in the plan did not materialize. Recorded in case any future construction prefers fp16 plumbing.

Trajectory context: 48 consecutive non-improvements. With this closure, every axis ever named in the project — recipe constants, loss geometry, structure, throughput, noise, averaging, augmentation, data order, eval constants, compounds-of-frees, and now numerics-precision both directions — is measured-closed. The remaining search space is strictly: constructions nobody has enumerated yet, passing all standing laws simultaneously.

## Verification

- Integrity pre-condition: PASS (windows 22.0–23.3ms ≤ 23.5/none > 27; 136 ep ∈ [133,142]; params exact; 300.0s; evals 136 ≤ 136; trajectory criterion satisfied; no NaN/EMA spikes).
- Condition 1 (best ≥ 96.81): FAIL — 96.54. First-failure-stop; escalation branch not triggered. Branch (ii) pre-registered closure.
- Conditions 2–3: PASS informationally (483.8s ≤ 600; 136 ≤ 136).
- Trust review: fresh log, clean parse, signatures cross-checked against the composite watchdog; the 2–3-epoch deficit is fully explained by the measured scaler overhead. Verdict basis: valid result below bar → **no-improvement**.

## Unexplored Avenues

- **fp16 without GradScaler** (raw fp16 + inf-clamping): would recover the 2–3-epoch toll, but with the precision effect measured at ≈ 0 there is no upside to fund the stability risk — closed by this result's arithmetic.
- **Selective precision** (fp32 accumulation for specific layers): autocast already runs BN/softmax/loss in fp32; the remaining surface (conv/matmul) is where the null was just measured. No mechanism survives.
- **fp16 plumbing for future constructions**: stability datum stands — any future radical change that happens to prefer fp16 (e.g., custom kernels) can use it without a stability screen.

## Next Steps

1. **First-principles construction sweep** (medium): the next brainstorm must enumerate genuinely novel constructions passing ALL laws — free in heat/epochs/noise/numerics, mechanism not suppliable by heavy aug, not in any closed class. Candidate seeds to price honestly: per-layer LR structure (never probed; heat-law risk), warmup-free schedules (heat axis closed — skip), training-time class-balanced batch composition (data composition never probed beyond order/coverage).
2. **Honest statement of the measured ceiling** (high): with every named axis closed, the loop's value per run is now dominated by closure information, not improvement probability; brainstorms should weight terminal-branch designs accordingly (replicate-pairs, two-sided brackets).
3. **Do not** revisit precision, compile modes, or any throughput flavor (high confidence — bounded twice).

## Key Learning

The certified recipe sits at a measured PLATEAU in precision-space: 8× finer mantissa (fp16, equal speed, fully stable at peak heat) reads exactly mean-band, while coarser arithmetic (EXP-021) cost −0.20 — so bf16 was never leaving accuracy on the table, and the last never-probed axis closes with both signs measured. After 48 nulls across every nameable axis, the bound is no longer "which knob" but "which function class" — only un-enumerated constructions passing all laws simultaneously remain.

# Brainstorm EXP-054
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external fetches; sources are the project record + previously distilled model knowledge:

- **EXP-021 (the numerics-sensitivity datum)**: max-autotune/cudagraphs/fused-SGD delivered +10 epochs at byte-identical hyperparameters yet lost −0.20pp (replicated) — trajectory quality is measurably sensitive to kernel arithmetic at the ~0.2–0.4pp scale. Crucially, that probe changed arithmetic in the *coarser/reordered* direction. The *finer-mantissa-at-equal-speed* direction has never been measured.
- **Precision arithmetic**: bf16 carries 7 mantissa bits (relative rounding step 2^-8 ≈ 0.4%), fp16 carries 10 (2^-11 ≈ 0.05%) — an 8× finer quantization of every activation and gradient tensor in the autocast hot path, at identical tensor-core throughput on H20. The trade is dynamic range (fp16 max 65504, min-normal 6e-5), which is exactly what `torch.amp.GradScaler` exists to manage (dynamic loss scaling, growth/backoff).
- **Speedrun lineage (model knowledge, cited in prior brainstorms)**: hlb-CIFAR10 and the airbench family train CIFAR in fp16 with loss scaling — the fastest-known CIFAR recipes chose fp16, not bf16, where both were available. Not effect-size evidence, but an existence proof that fp16 trains this exact problem class stably at aggressive LRs.
- **Statistics**: σ ≈ 0.16 (EXP-027); bar = 96.81 = mean + 1.5σ; goal-learnings Protocol Findings (EXP-052): near-bar single draws on low-prior candidates require a replicate-pair mean before a baseline move.

## Experimental History Review

State after 54 indexed experiments: baseline 96.71 @ 1990397, bar ≥ 96.81, 47 consecutive non-improvements. Frontier after EXP-053:

- **Every catalogued single-mechanism class is measured-closed** (recipe constants, loss axis both directions, structural classes, throughput at 99.3% kernel floor, gradient-noise bracket, averaging both kinds, augmentation dose-response, data-order, BN/eval constants).
- **EXP-053 closed the compound-of-frees region**: the only two non-negative free components composed to mean − 0.8σ (96.61/96.28); the prefetch's +87-step saving regressed to zero on both replicates. No certified component pool remains to compound.
- **The numerics axis is HALF-probed**: EXP-021 measured the coarser direction (−0.20 replicated). The finer direction (fp16's 8× smaller mantissa step at equal throughput) is the last never-measured axis in the recipe. The certified recipe was tuned UNDER bf16 rounding noise — whether that noise is (a) a hidden cost, (b) neutral, or (c) an accidental regularizer absorbed by TA+RE is an open empirical question with a clean mechanism on each branch.
- Protocol carry-overs: composite gates (26ms), trajectory numerics criterion (EXP-048: judge by trajectory-rejoins-family + plateau + family test_loss, not single ep1 reads), replicate-pair mean for near-bar reads (EXP-052), step ledger with replication caveat (EXP-053).

## Candidate Ideas

### 1. fp16 autocast + GradScaler in place of bf16 — the never-probed finer-precision direction of the numerics axis
**Summary**: `torch.autocast("cuda", dtype=torch.float16)` in both the warmup and timed loops; `torch.amp.GradScaler("cuda")` wrapping backward/step (`scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()`); warmup mirrors with scale+backward but no step. Everything else byte-identical. Single gated run; pre-registered replicate-pair escalation if the read ≥ 96.81.

**Reasoning**: The mechanism is exact and two-sided, which is what makes it a measurement rather than a coin-flip: fp16 reduces per-tensor rounding noise 8× in every forward/backward of the hot path. If bf16 rounding has been silently costing trajectory quality (the EXP-021 datum proves arithmetic differences of this order CAN cost 0.2–0.4pp), fp16 recovers it at zero throughput cost. If instead the recipe's gradient-noise optimum (EXP-011/022/023/024 bracket) already includes bf16's rounding as part of its tuned noise budget, the finer arithmetic reads null-to-negative — which CLOSES the numerics axis from both sides, the last open direction in the certified recipe. External existence proof: the fastest CIFAR recipes (hlb-CIFAR10/airbench lineage) train fp16 at aggressive LRs. Costs priced: GradScaler adds an unscale+inf-check pass (~0.1–0.3ms/step worst case → ≤ −0.04 by the deferral law) and may skip 1–3 early calibration steps (negligible); dynamic-range risk at PEAK_LR 0.4 is the real tail risk and is covered by the divergence guard + a loss-EMA spike check.

**Sources**: project-insights EXP-021 entry; goal-learnings gradient-noise law + EXP-048 trajectory criterion + EXP-052 replicate protocol; train.py L191/229 (autocast sites); model knowledge: hlb-CIFAR10/airbench fp16 lineage.

**Estimated Effort**: low — ~6 lines changed (autocast dtype ×2, scaler init, 3 scaler calls), CPU sanity via `GradScaler(enabled=False)` passthrough, one gated run (~9 min; +9 if escalation triggers).

**Risk Assessment**: Graceful on all branches: (i) ≥ 96.81 → replicate-pair escalation per the EXP-052 protocol (mean of 2 decides); (ii) mean-band read at family signatures → precision-null; numerics axis closed BOTH directions; (iii) low read or elevated-scatter plateau → bf16's rounding was neutral-or-regularizing; axis closed with sign; (iv) NaN/divergence (scaler fails at peak 0.4) → research failure, axis closed on stability grounds (no blind retry). Worst non-crash case costs one loop and still closes an axis permanently.

### 2. Per-layer weight-decay structure (exclude fc and/or stem from WD) — novel dimension of a bracketed axis
**Summary**: Keep WD 5e-4 on conv weights but exempt the final `fc` weight (and optionally the stem conv) — `decay_params` currently includes every ndim>1 tensor.

**Reasoning (and why not the lead)**: The WD *magnitude* axis is bracketed (EXP-015: half-WD −0.30; the optimum is at 5e-4), but WD *structure* was never probed: under LS 0.1 the fc weight norm sets logit scale, and decaying it fights the margin the loss wants. However: (a) the mechanism overlaps the measured loss-axis closure (EXP-050 proved logit-scale pressure is at an optimum — adding fc norm freedom is the same currency); (b) effect is sub-σ by construction (one layer, 2,570 weights of 4.29M); (c) the bracketing law says single-knob neighbors of a certified optimum re-measure the optimum.

**Sources**: train.py L169–175; goal-learnings EXP-015/050 entries.

**Estimated Effort**: trivial.

**Risk Assessment**: Almost-certain mean-band null; an expensive σ-coin-flip without closing any axis (interactions with LS make interpretation murky).

### 3. Gradient clipping (global-norm 5.0) — standing documented-weak candidate
**Summary**: `clip_grad_norm_(5.0)` before step.

**Reasoning (and why not the lead)**: Carried from brainstorm-052/053: mechanism conditional on instability that does not exist in 40+ clean runs; sub-σ at safe thresholds, heat-reduction (closed axis) at aggressive ones. Kept on the books so it is never re-derived; not run.

**Sources**: brainstorm-052 Idea 2; goal-learnings EXP-049 entry.

**Estimated Effort**: trivial.

**Risk Assessment**: Coin-flip on noise; dominated by Idea 1's axis-closing value.

## Idea Evaluation

- **Evidence strength**: Idea 1 rests on a replicated in-project datum (EXP-021: arithmetic differences move the metric at the 0.2–0.4pp scale) plus an external existence proof of fp16 stability on this exact task class. Ideas 2–3 have no anchor and sub-σ mechanisms.
- **Mechanism clarity**: Idea 1's variable (mantissa quantization step, 8× change) is isolated, signed on every branch, and never measured. Idea 2's mechanism collides with the closed loss axis; Idea 3's is vacuous here.
- **Expected impact**: honest prior for Idea 1 ≈ 15–20% positive-of-some-size / mostly null / fat-negative-tail — but EVERY branch closes the last open axis, which after 47 nulls is the correct currency. Ideas 2–3 close nothing.
- **Risk profile**: Idea 1's tail risk (fp16 overflow at peak heat) is bounded by the watchdog's NaN/divergence guards and pre-registered as a terminal branch, not a retry loop.
- **Feasibility**: ~6 lines, validated sanity pattern, standard composite. Lowest-effort axis-closing candidate available.

Idea 1 dominates: it is the only remaining candidate that is simultaneously never-probed, mechanism-clean, and terminal on every branch. Ideas 2–3 recorded so they are not re-derived.

## Chosen Idea
**Selected**: Idea 1 — fp16 autocast + GradScaler (finer-precision direction of the numerics axis)

**Why this idea**:
It is the last never-measured axis in the certified recipe, with a replicated in-project proof that this axis moves the metric (EXP-021), an isolated one-variable mechanism (8× finer mantissa at identical throughput), and terminal branches in every direction — including the failure branches, which close the axis with a sign instead of wasting the loop.

**Hypothesis**:
If bf16's coarse rounding (2^-8 relative step) has been costing trajectory quality, fp16 autocast (2^-11) at byte-identical hyperparameters reads above the recipe mean, and ≥ 96.81 if the effect is ≥ +0.3 (one-draw detectable). Pre-registered branches: (i) read ≥ 96.81 → escalate to a replicate pair (one more byte-identical run; improvement iff MEAN ≥ 96.81, per the EXP-052 protocol for low-prior near-bar reads); (ii) read in the mean band [96.41, 96.73] at family signatures (trajectory criterion, family test_loss) → precision-null; numerics axis closed in BOTH directions; (iii) read < 96.41 or elevated plateau scatter at clean dt → bf16 rounding was neutral/regularizing; axis closed with negative sign; (iv) NaN/divergence or scaler-skip storm (loss-EMA spikes, step ledger collapse) → fp16 unstable at peak 0.4 under this recipe; axis closed on stability grounds — research failure, no blind retry; (v) GATE_KILL/contention → infra relaunch (max 2). Expected signatures: dt 22.3–22.8ms (scaler overhead ≤ 0.3ms), epochs 135–140, params 4,286,026.

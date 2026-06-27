# Report EXP-059: Late batch-size step 512 → 1024 at p ≥ 0.75, LR unchanged
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-059.md
- **Plan**: plans/plan-059.md
- **Log**: logs/exp-log-059.md

## Goal

Maximize CIFAR-10 best_test_acc (%, higher is better) within the fixed 300s charged training budget by modifying train.py only. Baseline: 96.71 @ commit 1990397 (EXP-006 recipe). Success bar: ≥ 96.81 (baseline + 0.1pp = mean + 1.5σ of the family distribution, σ ≈ 0.16 from EXP-027). Specific question: with the noise LEVEL closed at both endpoints (batch 1024-from-start failed on throughput EXP-012/022; batch 256 failed on noise EXP-something-prior), does a noise SCHEDULE — high SGD noise for exploration early, low noise for sharpening late — extract accuracy the constant-noise recipe leaves on the table?

## Idea & Hypothesis

Chosen from brainstorm-059: **late batch-size step 512 → 1024 at p ≥ 0.75 of the charged budget, LR unchanged** (Smith et al. 2018, "Don't Decay the Learning Rate, Increase the Batch Size", arXiv:1711.00489). Doubling the batch at fixed LR halves the SGD gradient-noise scale — equivalent to an extra implicit LR decay applied only in the tail, stacked on top of the cosine's explicit anneal. Two mechanisms could pay: (1) lower-noise tail → sharper convergence into the minimum; (2) per-image throughput dividend at 1024 (one fused step instead of two) → more epochs inside the fixed budget. The switch was implemented in the UNCHARGED fetch path (pairing two loader batches before t0), so the charged region stays byte-identical — the established legitimate pattern. Hypothesis: best_test_acc ≥ 96.81 if tail noise is a binding constraint; family-band read if the cosine's anneal already provides all the late-phase noise reduction the trajectory can use.

## Approach

Four hunks to train.py, charged region untouched:
1. `BATCH_SWITCH_FRAC = 0.75` constant.
2. `torch.compile(model, dynamic=False)` — REQUIRED, not optional (see Execution): default automatic-dynamic-shapes merges the two warmed shapes into one dynamic graph ~18% slower at BOTH shapes.
3. Second 3-iter compile warmup at `2*BATCH_SIZE` on random data (uncharged) so both static graphs are cached before the timer starts.
4. Fetch-path pairing: explicit `batch_iter = iter(train_loader)`; when `total_training_time >= 0.75 * TIME_BUDGET_S`, pull a second batch and `torch.cat(...).pin_memory()` before t0 (StopIteration → unpaired 512 step, shape pre-warmed). Plus honest `img_per_sec = inputs.size(0)/dt`.

CPU sanity all-pass (params 4,286,026; pairing unit test; eager fwd/bwd both shapes; smoke). Switch keyed on charged seconds, same progress key as lr_at.

## Execution

Three GPU probe attempts, then one clean full run:
- Probe attempts 1–2 read P512 = 26.4ms (out of family band [21.5, 23.5]) at BOTH load 57 and load 37 — identical reads at low load disproved the load-contamination theory. Root cause: torch.compile automatic-dynamic-shapes silently recompiled ONE dynamic graph covering both shapes, ~18% slower at both. Without the P512-anchor check this would have shipped an 18% throughput regression for 75% of the budget.
- Probe attempt 3 with `dynamic=False` (load 16): P512 = 21.73 (in band), P1024 = 40.89 ≤ 46 → full-mechanism launch branch. Per-image dividend confirmed: 40.89 vs 2×21.73 = 43.46 (~6%).
- Run 1 (phase-aware composite launcher, TAIL_THRESH 57): PRISTINE. D0 22.5; pre-switch windows 21.7–22.7ms; clean dt step-change at tick 26 (pct 75.5) to 40.5–42.0ms — mechanism engagement proven by physical signature, no recompile stall at the switch. RC=0 at 513.8s.

## Results

- **Primary metric**: best_test_acc 96.51 @ ep136 (baseline: 96.71, delta: −0.20, −0.21%)
- **Observations**: 142 epochs / 11,933 steps — the ~6% tail dividend delivered roughly +2 equivalent epochs over the 140-ep family norm, exactly as the probe predicted. Trajectory family-shaped throughout (ep1 34.08; plateau 96.44–96.51; final_test_loss 0.1856 ≈ family). final 96.50. VRAM 3,157MB. The read is mean−0.4σ — dead center of the family band [96.41, 96.73].
- **Analysis**: Pre-registered branch (ii) fired — schedule-null. The tail noise halving plus the extra epochs moved NOTHING: the late-phase noise schedule is REDUNDANT with the one-cycle cosine's own anneal, which already drives the effective noise-to-signal ratio down faster than the trajectory can exploit. Combined with EXP-012/022 (level: 1024-from-start loses on throughput; the dividend exists but can't pay for the noise change) the noise axis is now closed in BOTH level and schedule. With this closure, every axis on the documented frontier — heat, noise (level + schedule), loss geometry, numerics, structure, order, width lattice, averaging, regularization dose, per-layer constants — is measured-closed.
- **Key Learning**: Tail gradient-noise reduction stacked on a cosine anneal buys nothing at this depth/budget — the explicit LR schedule already saturates late-phase sharpening; the noise axis offers no schedule-shaped escape from its level-closure.

## Verification

- **Conditions**: Integrity pre-condition PASSED (RC=0, signatures on probe-revised bands, steps 11,933 ∈ [11,400, 12,350], 142 evals ≤ 142 epochs, 513.8s ≤ 600, no NaN). Condition 1 FAILED: 96.51 < 96.81. Conditions 2–3 pass informationally.
- **Review Notes**: Results confirmed trustworthy — watchdog cross-checked the entire run window; the dt step-change at pct 75 proves the mechanism actually engaged (this is not a silent no-op null); step ledger rules out foreign-job contamination (the EXP-058 failure mode).
- **Verdict**: no-improvement
- **Verdict Basis**: Condition failure — valid, trustworthy result inside the family band, below the +0.1pp bar.

## Unexplored Avenues

- **Multi-step batch ramp (512→768→1024)**: inherits this null — if a single clean halving of tail noise reads family-band, a smoothed version of the same noise trajectory has no separate mechanism. Closed by inheritance (pre-registered in brainstorm-059).
- **Batch step DOWN (512→256 late, noise-up tail)**: the mirror image; brainstorm-059 pre-registered that a family-band read here closes it too — sharpening and exploration tails both null means the cosine anneal is the binding schedule. Closed by inheritance.
- **Batch switch with LR rescaling (step LR up ×2 at the switch, true Smith et al. protocol)**: NOT tried — but it is equivalent to a late LR bump, and the one-cycle peak/shape space was closed by the LR-schedule experiments (deferral law + two-sided tail pressure). Low residual potential.

## Next Steps

The documented frontier is now EMPTY: every catalogued axis, class, schedule, and corner has a measured closure. Per the autopilot directive the loop continues — brainstorm-060 must therefore either (a) construct genuinely novel law-passing interventions not in the existing taxonomy (combinations of near-misses, radical-but-legal architectural moves inside the {64,128,256} lattice and 4.29M params, mechanisms from papers not yet tested at this budget), or (b) treat the family mean ≈ 96.57 / max ≈ 96.81 as a measured ceiling and design experiments that could FALSIFY the ceiling rather than incrementally probe it. Confidence that any single remaining construction clears +0.3pp (the one-draw detection threshold): low — which argues for candidates with large hypothesized effects over safe small ones.

Durable instrument from this loop (independent of the null): `torch.compile(model, dynamic=False)` + per-shape warmup is the validated pattern for ANY future multi-shape experiment (progressive resizing, variable batch, curriculum crops) — without it, automatic-dynamic silently taxes every shape ~18%.

## Exit Action Results

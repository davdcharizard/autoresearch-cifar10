# Experiment Report: EXP-042 — Grouped two-member deep ensemble (2 × 3x ResNet-20 via groups=2, sum-CE, logit-mean inference)

- **Date**: 2026-06-10
- **Verdict**: invalid (pre-registered throughput screen — GATE_KILL; metric NaN)
- **Primary metric**: best_test_acc = **NaN** (baseline 96.71, bar 96.81; no fair run exists)
- **Branch**: autoresearch/exp-042 (discarded)
- **Artifacts**: brainstorm/brainstorm-042.md · plans/plan-042.md · logs/exp-log-042.md

## Goal
Maximize CIFAR-10 test accuracy (best_test_acc %, higher is better) within the fixed 300s charged training budget, modifying only `train.py`. Baseline 96.71 @ 1990397; bar ≥ 96.81. σ context (EXP-027): baseline mean ≈96.57, σ ≈0.16.

## Idea & Hypothesis
**Idea**: After EXP-041 every identified mechanism class had a measurement; per its Next Steps, EXP-042 constructed a novel mechanism from validated parts — function-space (multi-mode) prediction averaging, the unmeasured half of the averaging dichotomy whose weight-space half (EMA EXP-011, canonical SWA EXP-032) measured zero and produced the diagnosis "the ceiling is decision-boundary-limited." Design: reallocate capacity into multiplicity — two independently-initialized 3x-width ResNet-20 members (48/96/192 each) co-trained inside one module via groups=2 convs (totals 96/192/384, all per-group channels ≤256), sum-of-CE loss keeping each member's per-step dynamics byte-identical to the certified recipe, inference = mean member logits (a 2-branch architecture satisfying `Eval.evaluate()`'s contract directly — no eval wrapper). FLOPs 1.125×; dt projected 24.1ms via the ∂dt/∂FLOPs ≈ 13.3ms/unit law IF grouped kernels price like dense ones — that unknown was pre-registered to a D0-median dt gate (kill > 28.0ms → verdict `invalid`, screen).

**Hypothesis**: Multi-mode averaging raises the converged plateau MEAN above the single-model family; falsified by (i) gate-kill on grouped pricing or (ii) a plateau in the baseline band.

## Approach
train.py only (66 insertions / 25 deletions): `groups` threaded through `BasicBlock` (both convs) with a per-member pad shortcut (reshape→pad-per-member→reshape; flat end-pad would hand all new channels to member 2); grouped stem (in=6 via `x.repeat(1,2,1,1)`); two per-member fc heads; `forward()` returns (l1, l2) in train mode, `(l1+l2)/2` in eval mode. Five CPU sanities passed pre-launch, including the mandatory MEMBER-ISOLATION test (perturbing every member-2 slice left member-1 logits bit-identical) and gradient isolation (CE(l1) backward touches only member-1 slices). Constructed num_params 4,825,460 (hand estimate exact).

## Execution
Single gated launch (gates clear poll 1: apps=0, load=6; launched 21:10:14). Startup/compile normal; training healthy (sum-CE loss falling 4.58→3.68 by step 250; ep1 eval 39.22% ≈ family ~38 — ensemble-mean inference showed no deferral toll). **GATE_KILL at tick 6 (~90s): three independent 200-step windows read exactly 63.0ms** (steps 300/500/700; printed dt 63ms uniform; host clean throughout) → projected 49 epochs vs the ≥111 the gate required. Composite exit 47 per protocol; no retry, no fallback this loop (pre-registered). Total GPU cost ≈ 2 minutes.

## Results
- **The grouped-conv pricing fact (the screen's product): groups=2 convolutions cost 63.0ms vs the dense baseline's 22.4ms at only 1.125× FLOPs — a 2.8× penalty.** This is WORSE than the dense >256-channel cliff plateau (54ms, EXP-040) despite every per-group width being ≤256 and totals being 32-aligned. The H20 + torch.compile(default) + channels_last + bf16 stack evidently has no efficient grouped-conv kernel path at these shapes (consistent with grouped convs' known poor arithmetic intensity / cuDNN fallback behavior); the EXP-034 dense pricing law (∂dt/∂FLOPs ≈ 13.3ms/unit) does NOT extend to grouped kernels.
- **The ensemble mechanism itself was NOT tested** — 49 projected epochs is deep starvation territory (EXP-002/005/007), so any full run would have measured starvation, not multi-mode averaging. The hypothesis' branch (i) fired: this is a hardware closure of the in-one-kernel IMPLEMENTATION, not evidence against function-space averaging.
- **What little trajectory exists supports the design's training-health claims**: loss fell normally at sum-CE scale and the ep1 ensemble eval was family-equal — no instability, no deferral toll, eval contract worked under the real `Eval.evaluate()`.
- **Trajectory fit**: second productive gate-kill in three loops (with EXP-040) — the dt gate keeps architecture-class screens at ~2 GPU-minutes each while building the hardware pricing map: dense ≤256 fast (22.4ms), dense >256 cliff (54ms), grouped 2.8× (63ms).

## Verification
First-failure-stop per plan-042. **Pre-condition (gate) FAILED on the pre-registered branch**: GATE_DECISION D0 = 63.0ms > 28.0ms. Per the plan's Abort Criteria, the screen verdict `invalid` (NaN metric) applies and Conditions 1–3 were not evaluated — no completed run exists to judge. Screen integrity is high: clean host (load 6, zero foreign GPU-0 apps), three identical windows, params matching the constructed value. No false-kill risk: 63.0 is 2.25× the threshold, far outside any quantization or contention ambiguity.

## Unexplored Avenues
- **Alternating-step two-member ensemble (brainstorm-042 Idea B)** — the pre-registered retry path: two separate full-width (4x) members, one trained per step with dense baseline kernels at 22.4ms; zero kernel risk; per-member step starvation (~70 epochs' worth each) is the cost; needs ensemble gain ≥ ~+0.6 over member plateaus ~96.0–96.2. The mechanism question (does init-diversity function-space averaging raise the plateau?) remains open and this is the only implementation left that prices at dense dt.
- **Member width asymmetry under alternation** (e.g., one 4x member trained 60% of steps + one 3x member 40%): more arithmetic knobs, no new mechanism; only worth considering after a 2-member symmetric read exists.
- **Grouped convs without channels_last or with eager mode**: might dodge the slow kernel path, but abandoning channels_last/compile costs the certified 1.22× throughput — strictly dominated; do not pursue.

## Next Steps
1. **Run Idea B (alternating-step 2 × 4x ensemble) as EXP-043's lead candidate** — it inherits this loop's motivation (decision-boundary diagnosis, multi-mode averaging) at dense-kernel pricing; the open question is whether the ensemble gain can outrun the ~70-epoch member starvation (medium confidence it clears the bar; high confidence the read is mechanism-informative either way: it finally measures function-space averaging).
2. **Record the grouped-conv pricing fact** in project-insights (extends the EXP-040 kernel-cliff entry) and goal-learnings (dt-gate protocol entry ref bump) — done in this loop's distillation (high confidence).
3. **Protocol carry-over**: pre-registered replicate pair remains on file for judging any future promising mid-band read (carried from exp-report-040/041; medium confidence).

## Key Learning
Grouped convolutions on this stack (H20, torch.compile default, channels_last, bf16) cost 2.8× dense step time at near-equal FLOPs (63.0 vs 22.4ms at 1.125×, every per-group width ≤256) — the dense pricing law and the 256-channel cliff are both dense-kernel facts that do not transfer to groups=2, closing all in-one-kernel multi-member designs on hardware grounds at a total cost of ~2 GPU-minutes. The function-space-averaging hypothesis (the program's last unmeasured averaging mechanism) remains open and moves to the alternating-step implementation, which uses only certified dense kernels.

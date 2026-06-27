# Brainstorm EXP-002
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external search this loop — the knowledge base already covers the candidate space (knowledge/README.md § References):
- **WRN (arXiv 1605.07146)**: width gains persist to 8–12× at 16-layer depth; WRN-16-8/WRN-22-8 land ~95.7–96% with full 200-epoch training; the paper uses the SAME base LR across all widths (no LR retuning needed when widening).
- **Cutout (arXiv 1708.04552, cited in exp-report-001 Unexplored Avenues)**: +0.5–1pp on CIFAR-10 for WRN-class models; implemented as torchvision `RandomErasing` (no new dependency).
- **cifar10-fast / airbench**: shallow-wide topologies reach 94% in 10–15 one-cycle epochs — evidence that ~30–50 epochs is NOT automatically undertraining for wide nets under one-cycle.

## Experimental History Review

- **Trajectory**: 91.97 (BASE) → 93.16 (EXP-000, recipe) → 95.23 (EXP-001, 4× width). Both improvements; no failures yet.
- **Key learnings** (goal-learnings):
  - Width gradient is steep: 4× width = +2.07pp; next step (6–8×) flagged high-value with peak-LR caution and undertraining watch below ~50 epochs (Patterns, High).
  - Time-keyed one-cycle self-adapts to any throughput — no schedule retuning when the model gets heavier (Patterns, High; validated twice).
  - Heavier models REDUCE wall-clock risk: EXP-001 finished at 395.8s vs EXP-000's 596.7s (Protocol Findings, High).
- **Current state**: 4× wide ResNet-20 (4.29M params), 114 epochs, 18.7k img/s, peak VRAM 1.6GB.
- **Untried gaps**: more width (8×), Cutout/RandomErasing regularization, ResNet-9 topology, larger batch, GPU-resident data pipeline.

## Candidate Ideas

### 1. Widen to 8× (WRN-16-8-class: stage widths 128/256/512)
**Summary**: Change `WIDTH_MULT` 4 → 8 (~17M params). Recipe byte-identical (peak 0.4 — WRN evidence says no LR retuning across widths; time-keyed schedule self-adapts). Expected throughput ~5–7k img/s → ~30–45 epochs.

**Reasoning**: The width gradient measured in this project is steep (+2.07pp at 4×) and WRN shows monotone gains to 8–12× at this depth: WRN-16-8 ≈ 95.7%, WRN-28-10 ≈ 96.1% with full training. The one-cycle regime compresses most of that into few epochs (cifar10-fast reaches 94% in ~13 epochs on a smaller net). VRAM (~6GB) and wall clock (epochs drop further) are both comfortable. Single-variable continuation of the validated direction.

**Sources**: arXiv 1605.07146 §4 (width study, fixed LR across widths); goal-learnings § Patterns (EXP-001); exp-report-001 § Next Steps.

**Estimated Effort**: trivial (one constant)

**Risk Assessment**: Undertraining at ~30–45 epochs is the main risk — mitigated by one-cycle evidence; if accuracy lands below 95.33 the verdict is a clean no-improvement and the 6× midpoint remains available. Divergence risk low (warmup + WRN's fixed-LR-across-width evidence). VRAM ~4–8GB, trivially fine.

### 2. Add RandomErasing (Cutout) on the 4× net
**Summary**: One transform line after Normalize: `transforms.RandomErasing(p=0.5, scale=(0.02, 0.25))`. Architecture and recipe unchanged.

**Reasoning**: Cutout gives +0.5–1pp on WRN-class CIFAR-10 models with long training. At 4.29M params and 114 epochs the model now has capacity for stronger regularization (EXP-001 final_test_loss 0.2447 with near-zero train loss EMA implies a real train/test gap).

**Sources**: arXiv 1708.04552; torchvision RandomErasing docs; exp-report-001 § Unexplored Avenues.

**Estimated Effort**: trivial (one line)

**Risk Assessment**: Regularization shortens effective capacity use when epochs are few; with only 114 epochs the published gains (200+ epochs) may not fully materialize — could land within noise (+0.1–0.3). Fails gracefully.

### 3. ResNet-9 topology at matched parameter count
**Summary**: Replace the model with the cifar10-fast ResNet-9 (~6.5M params, much shallower) under the current recipe.

**Reasoning**: Speedrun-proven highest accuracy-per-second topology; fewer sequential layers → better GPU utilization per parameter than deep stacks.

**Sources**: knowledge/README.md § References (cifar10-fast, airbench).

**Estimated Effort**: medium (model rewrite)

**Risk Assessment**: Recipe transfer risk (its published recipe differs: logit scaling, different WD treatment); confounds topology with capacity. Width on the known-good topology dominates this loop; revisit when width saturates.

## Idea Evaluation

**Evidence strength**: Idea 1 leads — it extrapolates a gradient measured *in this exact project* (+2.07pp at 4×) backed by WRN's systematic width study showing gains continue to 8×+ at this depth, including the key detail that LR transfers across widths unchanged. Idea 2's evidence is solid but calibrated to 200+-epoch schedules; at ~114 epochs the expected effect shrinks toward the +0.1 bar. Idea 3's evidence is strong for the topology but weak for recipe transfer.

**Mechanism clarity**: Idea 1 — same mechanism that just delivered +2.07pp (more capacity per fixed time budget), with throughput/wall-clock math worked out. Idea 2 — regularization mechanism, plausible but smaller and schedule-dependent. Idea 3 — capacity-per-second mechanism, entangled with recipe risk.

**Expected impact**: Idea 1: +0.3–0.8pp expected (WRN-16-8 full-training reference ~95.7–96% vs our 95.23). Idea 2: +0.1–0.5pp. Idea 3: high variance.

**Risk profile**: Ideas 1 and 2 both fail gracefully; Idea 1's failure (undertraining) cleanly informs the next move (6× midpoint or batch/throughput work), while Idea 2's failure is ambiguous (epochs? strength? placement?). Idea 3 riskiest.

**Feasibility**: Ideas 1–2 are one-line diffs; Idea 3 is a rewrite.

## Chosen Idea
**Selected**: Idea 1 — Widen to 8× (stage widths 128/256/512, ~17M params)

**Why this idea**:
Continues the steepest measured gradient in the project with the strongest setting-specific evidence (WRN width study, LR transfer across widths), is a one-constant change on the validated recipe, fails gracefully and informatively, and keeps wall clock comfortable. Cutout (Idea 2) is the natural follow-up once capacity is maxed.

**Hypothesis**:
Setting WIDTH_MULT = 8 (~17M params) under the unchanged recipe will raise best_test_acc from 95.23% to ≥95.6%, because the project-measured width gradient and WRN's width study both indicate meaningful gains remain between 4× and 8× at this depth, and one-cycle training extracts most of a wide net's accuracy within ~30–45 epochs; total_seconds will drop further (~340–370s) as epoch count falls.

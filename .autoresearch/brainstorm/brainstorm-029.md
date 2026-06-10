# Brainstorm EXP-029
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Loshchilov & Hutter — "SGDR: Stochastic Gradient Descent with Warm Restarts" (ICLR 2017), arXiv:1608.03983**
  Cosine annealing with periodic WARM RESTARTS: anneal LR peak→~0 over a cycle, then jump back to peak and repeat (optionally lengthening cycles). Two reported benefits on CIFAR WRN: (1) faster anytime convergence, and (2) the restart can escape a sharp basin so the *final* cycle's minimum generalizes better; the per-restart snapshots also form a cheap ensemble (Huang et al. "Snapshot Ensembles" 2017) — but ensembling needs averaging multiple snapshots, which overlaps our CLOSED weight-averaging axis (EXP-006/019/020). The single-final-model SGDR benefit is the re-exploration, not the ensemble.
- **Goyal et al. 2017 / general warmup practice**: a longer LR warmup stabilizes early training when gradient noise is high (large batch OR strong augmentation). Our recipe uses strong TrivialAugment+Cutout → noisy early grads; warmup is currently only 5% of budget (~4.5 ep).
- **torch.compile `max-autotune`** (PyTorch docs): autotunes kernel choices for lower per-step latency vs `reduce-overhead`. BUT in this harness the `dt` budget timer wraps the FIRST step with NO untimed warmup (train.py L218-241), so the (much longer) autotune compile would be CHARGED to the 300s budget → likely fewer effective training steps. Demoted to a rejected candidate below.

## Experimental History Review

Current best **96.22%** (EXP-012, commit 6c417a4); bar = 96.32 (+0.1pp). 29 experiments; the plateau is exhaustively mapped (~21 axes closed):
- Scalar knobs bracketed (LR-peak 0.2 interior optimum EXP-016/017, Cutout-16 EXP-013/021, LS-0.1 EXP-023, WD-1e-4 EXP-005, batch-128 EXP-025); aug family closed (TA ceiling; Mixup/CutMix/dropout regress); regularizer-adding fails (convergence-bound); compute-adding hits the epoch wall (k≥5, pre-act, BlurPool); batch-scaling compute-bound; weight-averaging/convergence-polish move loss not top-1 (EMA/SWA/LS-down/Bag-of-Tricks); downsampling/anti-aliasing closed both sides (BlurPool EXP-024 + ResNet-D EXP-027); **activation closed both recipes (SiLU EXP-010 pre-TA + EXP-028 TA)**.
- **CRITICAL budget mechanic (re-confirmed this loop)**: the 300s budget gates on Σ(per-step `dt`), and `dt` includes the first-step compile (no untimed warmup, train.py L218-241). So "buy epochs via a faster compiler" is NOT free — a slow autotune compile is charged to the budget.

**Untried, compute-NEUTRAL levers remaining**: (1) **LR-schedule SHAPE** — only the LR-PEAK was swept (016/017, settled at 0.2) and the cosine FLOOR was swept via SWA (closed); the cosine RESTART/SGDR shape is genuinely untried. (2) **Warmup fraction** — never cleanly swept (EXP-025 changed it 0.05→0.08 but BUNDLED with batch-256, fully confounded). (3) per-channel input std-norm — now correctly assessed as a likely REGRESSION (frozen eval at std=(1,1,1) → train/test BN-scale mismatch), not the clean null earlier memory assumed; demoted.

## Candidate Ideas

### 1. SGDR — cosine annealing with warm restarts (2 cycles)
**Summary**: Change the LR schedule from a single cosine-to-0 over the whole budget to **2 cosine cycles** (SGDR): warmup once over WARMUP_FRAC of the first cycle, then each cycle anneals PEAK_LR→~0 over half the time budget, with a restart to PEAK_LR at the 50% mark. Pure edit to `lr_at_fraction(frac)` (train.py L35-41) — compute-neutral, no model/param change, no epoch-wall risk.

**Reasoning**: The LR-schedule SHAPE is the largest genuinely-untried compute-neutral axis (only PEAK was swept, settled; FLOOR closed via SWA). SGDR's restart can kick the optimizer out of the basin the single cosine settles into and re-anneal into a potentially flatter/better-generalizing one — a *trajectory* change (could find a genuinely different, possibly-better minimum) rather than convergence-polish around a fixed minimum (the closed EMA/SWA class). Documented to help CIFAR WRNs. Mechanism is distinct from every closed axis.

**Sources**: Loshchilov & Hutter 2017 (arXiv:1608.03983); train.py L35-41 `lr_at_fraction`; project-insights (LR-peak settled, weight-averaging closed).

**Estimated Effort**: low (rewrite the 6-line `lr_at_fraction`; one 300s run).

**Risk Assessment**: (a) Budget-splitting — at a short 300s budget each cycle gets ~45 ep; the strong-aug recipe may not fully converge per cycle → the restart's re-exploration may not pay back the lost per-cycle convergence → null/mild-regression. (b) The single-model (no-snapshot-ensemble) SGDR benefit is the weaker of SGDR's two benefits, and the ensemble benefit overlaps the CLOSED weight-averaging axis. (c) Compute-neutral so NO epoch-wall/throughput risk — fails gracefully to no-improvement. Verify throughput-neutral (epochs ~91) as always.

### 2. Longer LR warmup (WARMUP_FRAC 0.05 → 0.10)
**Summary**: Double the warmup from 5%→10% of the time budget (~4.5→~9 ep) in `lr_at_fraction`. One-constant change, compute-neutral.

**Reasoning**: The recipe uses strong TrivialAugment+Cutout → high early gradient noise; a longer warmup can stabilize the high-LR early phase and let the cosine anneal into a better basin. Never cleanly tested (EXP-025's warmup change was confounded by the batch-256 disaster).

**Sources**: Goyal et al. 2017 (warmup stabilizes noisy early training); train.py L24 WARMUP_FRAC, L38-39; EXP-025 (confounded prior warmup change).

**Estimated Effort**: low (one constant; one run).

**Risk Assessment**: Likely NULL — the recipe already trains stably at 0.05 warmup (no instability observed at 96.22), so 0.05 is probably already adequate; doubling it slightly shortens the productive high-LR phase. Compute-neutral, fails gracefully. Lower EV than SGDR (closes a narrower sub-axis).

### 3. torch.compile mode "max-autotune" — REJECTED (compile charged to budget)
**Summary**: Swap `mode="reduce-overhead"`→`"max-autotune"` (train.py L190) to lower per-step dt and buy epochs for the possibly-epoch-hungry TA recipe.

**Reasoning / why REJECTED**: The idea targets the BINDING constraint (dt) directly, but the `dt` timer wraps the first step with NO untimed warmup (train.py L218-241), so max-autotune's much longer autotuning compile (often 1-3 min) is CHARGED to the 300s budget → likely 60-180s of budget lost to compile → FEWER training steps → likely regression, unless the per-step speedup (uncertain on this small net whose default kernels are already good) fully offsets it. Negative-to-neutral EV. Adding an untimed warmup to dodge this would change the timing methodology vs all 28 priors (unfair comparison / budget-gaming risk). DISCARDED — documented so future loops don't naively reconsider it.

**Sources**: train.py L190 (compile mode), L218-241 (dt timer wraps first step, no warmup); EXP-007 (reduce-overhead gained because its compile is fast ~25s).

**Estimated Effort**: n/a (rejected).

**Risk Assessment**: n/a (rejected — likely-negative due to compile-in-budget).

## Idea Evaluation

Candidates 1 & 2 respect all hard constraints (train.py-only, no new deps, single GPU, ≤1 eval/epoch, no seed hacking), are compute-neutral (no epoch-wall risk), and are genuinely untried; candidate 3 is rejected on the compile-in-budget mechanic.

- **Evidence strength**: SGDR (1) has documented CIFAR-WRN gains AND closes the larger untried axis (schedule SHAPE). Warmup (2) has weaker, more generic evidence and closes a narrower sub-axis. Both modest given the depth of the plateau.
- **Mechanism clarity**: SGDR is the crisper "different mechanism" — a restart changes the optimization TRAJECTORY and can land a different minimum, distinct from convergence-polish (closed). Warmup only adjusts early-phase stability.
- **Expected impact**: Both low in absolute terms (plateau). SGDR has the higher ceiling (could find a genuinely better basin); warmup is almost certainly within noise.
- **Risk profile**: Both fail gracefully to no-improvement (compute-neutral). SGDR's risk is budget-splitting under-convergence; warmup's is being a no-op.
- **Feasibility**: Both low-effort one-function edits.

SGDR (1) leads: it is the most substantive genuinely-untried compute-neutral lever (the schedule-SHAPE axis the memory has repeatedly flagged), has a real re-exploration mechanism distinct from every closed axis, and aligns with the directive to try more radical (yet safe) changes. Warmup (2) is the lower-EV fallback. Candidate 3 is rejected.

## Chosen Idea
**Selected**: SGDR — cosine annealing with warm restarts (2 cycles)

**Why this idea**:
With all scalar knobs, the aug family, capacity, batch, activation, downsampling, and weight-averaging closed, the LR-schedule SHAPE (restarts) is the single largest untried compute-neutral lever — only the PEAK (settled at 0.2) and the FLOOR (closed via SWA) were ever explored. SGDR is compute-neutral (zero epoch-wall risk, the dominant failure mode here) and its warm-restart mechanism changes the optimization trajectory — it can escape the basin the single cosine settles into and re-anneal into a flatter/better-generalizing minimum. That is a genuinely different mechanism from convergence-polish (EMA/SWA, closed) which only averages around a fixed minimum. It is the best remaining shot at a real top-1 move without hitting the epoch wall.

**Hypothesis**:
Replacing the single cosine-to-0 with a 2-cycle SGDR schedule (warmup once, then PEAK→0 each half-budget cycle with a restart at 50%) lifts `best_test_acc` above the 96.32 bar at an unchanged ~91 epochs / 4,299,866 params / dt ~8ms / <600s, by re-exploring past the single-cosine basin into a better-generalizing one. Falsifiable: if epochs hold (~91, confirming compute-neutrality) but accuracy lands within ±0.2pp of 96.22, schedule-shape/restarts do not help this well-tuned recipe at the 300s budget (the single long cosine is optimal), and the schedule axis is fully closed (peak + floor + shape) → the 96.22 plateau stands.

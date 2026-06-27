# Brainstorm EXP-010
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new search this loop — the decision rests on in-project measurements plus already-distilled sources:
- **Super-convergence (arXiv 1708.07120)** (knowledge/README.md): one-cycle schedules tolerate and benefit from much larger peak LRs than step-decay intuition suggests — CIFAR ResNet-56 trained stably at peak LR 1.0–3.0, and the large-LR phase acts as additional regularization with faster traversal of the loss landscape. Directly relevant: our PEAK_LR 0.4 came from conservative linear scaling (0.1 × 512/128), not from tuning under one-cycle.
- **cifar10-fast (davidcpage)** (knowledge/README.md): the tuned ResNet-9 recipe at batch 512 used an effective peak LR ≈ 0.4–0.5 with heavy tuning of the LR/WD pair as the final accuracy lever once architecture and augmentation were frozen — the same position this project is now in.
- **goal-learnings § Patterns (updated EXP-009)**: regularization dose-response is saturated (RE +0.83, TA +0.17, mixup −0.46); remaining headroom named as optimization quality, throughput, or base-hyperparameter re-tuning.

## Experimental History Review

- **Trajectory**: 91.97 → 93.16 → 95.23 (4x) → 94.41 (8x fail) → 96.06 (RE) → 96.23 (TA) → 95.12 (5x fail) → 96.71 (compile) → 96.00 (6x fail) → 95.76 (ResNet-14 fail) → 96.25 (mixup fail). Baseline: **96.71 @ 1990397**.
- **State of the axes**: capacity closed bidirectionally (High count 3 + Low entries); regularization saturated (dose-response crossed zero, EXP-009); throughput levers >1.05x exhausted (compile banked; max-autotune projected sub-bar). The recipe's REGULARIZATION and ARCHITECTURE are at their measured optima.
- **The untouched surface**: PEAK_LR=0.4, WARMUP_FRAC=0.15, WEIGHT_DECAY=5e-4, MOMENTUM=0.9 were all set at EXP-000 for an UNAUGMENTED 1x ResNet-20 at 345 epochs. Since then the recipe gained 4x width, LS+TA+RE, compile, and runs 139 epochs — five major changes, zero hyperparameter re-tuning. The optimal (LR, WD) almost certainly moved: heavier augmentation and wider nets both shift optimal peak LR upward (super-convergence; WRN paper trains 8–10x-wide nets at the same nominal LR as thin ones, implying per-parameter effective LR headroom).
- **Stability datapoint**: EXP-002 ran 8x width at PEAK_LR 0.4 with zero instability; no run has ever shown LR-related divergence — suggesting 0.4 sits in the stable interior, possibly below optimum.
- **Protocol findings**: eval overhead counts toward the 600s wall cap (137 evals ≈ 116s); loader stalls land outside the timed budget (fetch happens before t0) — relevant when interpreting total_seconds, not a lever here.

## Candidate Ideas

### 1. PEAK_LR 0.4 → 0.6 (single-constant probe of the LR optimum's direction)
**Summary**: Raise the one-cycle peak from 0.4 to 0.6 (1.5x); everything else byte-identical. Tests whether the never-retuned LR sits below the optimum for the current heavily-regularized wide recipe.

**Reasoning**: First-principles: the binding factor is no longer capacity or regularization dose but how much optimization progress 137 epochs extracts — peak LR is the single most influential constant for that. Three independent arguments point UP rather than down: (a) super-convergence shows one-cycle peaks of 1.0+ are stable and beneficial on CIFAR ResNets, so 0.4 is conservative; (b) heavy augmentation (TA+RE) reduces effective gradient correlation, raising tolerable LR; (c) wider nets (4x) keep per-unit gradient noise lower at fixed batch. The time-keyed schedule guarantees the anneal completes regardless, so a too-hot peak degrades gracefully (depressed mid-schedule, recovered in descent) rather than diverging. 1.5x is a meaningful but safe step — far below the 2.5–7x of super-convergence demonstrations. The result is directional information either way: a gain re-opens a cheap tuning series (0.8 next); a loss says 0.4 was at/above optimum and brackets the search.

**Sources**: arXiv 1708.07120 (knowledge/README.md row); knowledge/README.md cifar10-fast row (LR/WD as the end-game lever); goal-learnings § Patterns (one-cycle time-keyed entry; saturation entry); EXP-002 stability datapoint (experiment-indices row 002).

**Estimated Effort**: trivial (one constant)

**Risk Assessment**: Worst case is a graceful −0.2 to −0.5pp (mid-schedule chaos not fully recovered); divergence is unlikely (bf16 + BN + nesterov at 0.6 is mild; abort criteria cover NaN anyway). Zero throughput/VRAM impact. Clean single-variable attribution.

### 2. EMA weight averaging for evaluation
**Summary**: Per-step EMA (decay 0.995) of all parameters into an eager copy; evaluate the EMA copy each epoch.

**Reasoning**: The final-epoch eval noise of recent runs (±0.1pp epoch-to-epoch at convergence) is exactly what EMA harvests, and the bar is only +0.1pp. But the cosine-to-~0 anneal already averages implicitly, capping expected gain at ~0–0.2pp, and the per-step update costs ~0.5–1ms (~3–6 epochs). Net expectation is marginal.

**Sources**: standard practice (timm/PyTorch recipes); exp-report-009 § Next Steps (low-medium).

**Estimated Effort**: low-medium (parallel copy + lerp + eval switch; keep EMA copy out of the compiled graph)

**Risk Assessment**: Sub-bar likely; clean failure; small bug surface in the eval path (must remain the frozen Eval on clean data).

### 3. WEIGHT_DECAY 5e-4 → 2.5e-4 (decay re-tune under heavy augmentation)
**Summary**: Halve WD on conv/linear weights; rationale: LS+TA+RE now supply the regularization pressure WD was originally providing, so the total regularization budget may be overspent (EXP-009 showed adding more hurts — removing some elsewhere may help).

**Reasoning**: Mirrors the mixup finding from the other side: if the recipe is over-regularized at the margin, reducing WD is the cheapest subtraction. However, WD interacts with LR (effective LR ∝ LR×WD coupling in BN networks), making attribution muddier than the LR probe, and 5e-4 is the near-universal CIFAR WRN setting even WITH heavy augmentation in the literature — weaker prior for a move.

**Sources**: WRN paper recipes (5e-4 standard); EXP-009 over-regularization finding (reports/exp-report-009.md).

**Estimated Effort**: trivial (one constant)

**Risk Assessment**: Graceful failure (overfit slightly → lower test acc); but weaker evidence and worse attribution than Idea 1.

## Idea Evaluation

**Evidence strength**: Idea 1 has the strongest convergent evidence: an external result directly about this schedule family on this dataset/architecture class (super-convergence), a reference implementation that won its benchmark by tuning exactly this pair last (cifar10-fast), an in-project stability datapoint at higher width, and the project's own learning that this is the named untouched surface. Idea 3 contradicts the literature default; Idea 2 fights the schedule's implicit averaging.

**Mechanism clarity**: Idea 1 — more optimization progress per epoch during the high-LR phase plus stronger implicit regularization, both payable within the fixed budget; the failure mode (insufficient recovery in the anneal) is equally crisp. Idea 3 — regularization budget rebalancing, but confounded with effective-LR shift. Idea 2 — variance harvesting at the noise floor.

**Expected impact**: Idea 1: +0.1–0.4pp if 0.4 is below optimum (the three arguments above say it is); −0.2 to −0.5 otherwise — and either result brackets the LR search for one more cheap follow-up. Idea 3: ±0.2pp. Idea 2: 0–0.2pp minus epochs.

**Risk profile**: all fail cleanly; Idea 1 has the best information-per-run (directional bracketing) and zero implementation risk (one constant on infrastructure validated 10 runs deep).

**Feasibility**: Ideas 1 and 3 trivial; Idea 2 moderate. Idea 1 dominates on evidence and information.

## Chosen Idea
**Selected**: Idea 1 — PEAK_LR 0.4 → 0.6

**Why this idea**:
With architecture and regularization both at measured optima, the optimization hyperparameters are the last untouched high-leverage surface, and peak LR is their dominant member. Three independent lines of evidence (super-convergence stability at far higher peaks, augmentation-raised LR tolerance, width-reduced gradient noise) all point to 0.4 being below the current optimum, the time-keyed anneal makes the failure mode graceful, and either outcome brackets the LR search for a cheap follow-up.

**Hypothesis**:
Raising PEAK_LR from 0.4 to 0.6 on the otherwise-frozen compiled 4x TA+RE recipe will raise best_test_acc from 96.71% to ≥96.81%, because the current peak was linearly scaled for an unaugmented 1x net and sits below the optimum for a heavily-augmented 4x net under one-cycle; predicted execution signatures are unchanged dt (~22ms) and epochs (~137–139), deeper mid-schedule accuracy depression than EXP-006 (hotter peak), full recovery in the final anneal, and no NaN/divergence at any point.

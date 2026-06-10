# Brainstorm EXP-011
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new external search. Mixup (Zhang et al. 2018, ICLR), per-channel standardization, and LR tuning are textbook
  and grounded in the project's own history + standard CIFAR practice. Knowledge base empty. Binding evidence is
  internal: the augmentation axis (Cutout) is the largest non-capacity lever this project found.

## Experimental History Review

Current best **96.00%** (EXP-003, commit f59de56): k=4 WideResNet + Cutout(16) GPU-vectorized + bf16/channels_last
+ cosine (peak 0.2) + Nesterov + label smoothing 0.1, seed 42, ~77 epochs, converged (loss 0.204).

12 experiments, 5 improvements. **SEVEN axes exhausted**: width (k=4 sweet spot; k=6/compiled-k5 regress via epoch
wall), weight-decay regularization (EXP-005 marginal), weight-averaging (EXP-006), training-length/throughput
(EXP-007, converged past ~77), channel-attention (EXP-008), compiled-capacity-scaling (EXP-009), activation/
nonlinearity (EXP-010, SiLU null).

**Crucial distinction for this loop**: the "regularization near-saturated" learning (EXP-005) was about *weight
decay / L2*, NOT data augmentation. Data augmentation is a different mechanism and has been the project's **biggest
non-capacity lever**: Cutout added +0.52 (EXP-002) then +0.58 once de-bottlenecked (EXP-003), ~+1.1pp total, and
cut overfit loss 0.25→0.20. But **only Cutout** has been tried on the augmentation axis — Mixup, CutMix,
Rand/TrivialAugment are all untried. goal-learnings § Patterns explicitly flags "augmentation is a live lever…
stacking more (mixup) is promising."

Untried gaps: (1) **complementary augmentation** (Mixup/CutMix) — different mechanism than Cutout's occlusion;
(2) per-channel input std-norm (code uses std=(1,1,1)); (3) LR-schedule micro-tuning (only WD swept).
Key constraint: anything epoch-hungry fights the converged 77-epoch budget (the capacity-axis failure mode).

## Candidate Ideas

First-principles framing: the model is converged and generalization-bound at fixed capacity (EXP-007). The proven
way to improve generalization here without adding capacity or epochs is **better data augmentation** — it's the only
lever that ever moved the metric outside of capacity. Cutout works; the question is whether a *complementary*
augmentation mechanism stacks further. The binding risk for any augmentation is convergence speed: harder-to-fit
augmented data needs more epochs, and we have a fixed budget.

### 1. Mixup augmentation (mild α), GPU-vectorized, stacked on Cutout + compile enabler
**Summary**: Add Mixup (Zhang 2018) to the training loop, GPU-vectorized like the existing `cutout_batch`: per batch
sample λ~Beta(α,α) with mild **α=0.2**, form `x = λ·x + (1−λ)·x[perm]` and the mixed loss
`λ·CE(out,y) + (1−λ)·CE(out,y[perm])` (label smoothing retained). Keep Cutout. Add the validated
`torch.compile(reduce-overhead)` enabler so the run keeps ~85–89 epochs (offsets Mixup's slower convergence).

**Reasoning**: Augmentation is this project's largest non-capacity lever (Cutout ~+1.1pp). Mixup is a *different*
mechanism (convex interpolation of inputs+labels vs Cutout's occlusion), well-evidenced on CIFAR (typically
+0.5–1pp with longer training), and complementary — Mixup+Cutout-family combos are standard strong CIFAR recipes.
It's GPU-vectorizable with no per-sample CPU sync (avoids the EXP-002 dataloader-throttle trap). **Mild α=0.2** is
deliberate: Beta(0.2,0.2) is U-shaped (most samples nearly pure, occasional strong mix), so it regularizes without
drastically slowing convergence — threading the budget constraint that makes strong Mixup risky. Compile gives ~89
epochs (vs 77) to converge; EXP-007 showed compiled-k4 = baseline (null), so any gain is from Mixup.

**Sources**: Mixup (Zhang et al. 2018, ICLR); goal-learnings § Patterns (augmentation live lever, "mixup
promising"); EXP-002/003 (Cutout = biggest non-capacity win); project-insights § Medium (compile enabler).

**Estimated Effort**: low-medium — ~8 GPU-vectorized lines (Beta sample via two Gammas or a fixed-λ-per-batch
approximation, batch permute, mixed loss) + the compile enabler.

**Risk Assessment**: Main risk is convergence: even mild Mixup slows fitting, and at a fixed budget the run could
land slightly under-converged → a soft regression (the capacity-axis failure mode, but milder). α=0.2 + the
compile epoch budget mitigate this. Fails gracefully (no crash). Secondary: if λ is sampled per-sample it needs a
per-sample loss reduction — straightforward but must be implemented carefully (a per-batch scalar λ is the simple,
safe default and standard). Moderate-but-bounded risk; the highest *ceiling* of the remaining ideas.

### 2. Per-channel input standardization (fix `std=(1,1,1)`)
**Summary**: Change `Normalize` std `(1,1,1)` → CIFAR-10 per-channel std `(0.2470,0.2435,0.2616)`. k=4, all else fixed.

**Reasoning**: Textbook input standardization, currently disabled by a faithful-to-original-paper choice
(train.py L152-155). Free (no epoch cost). Mechanism: unit-variance inputs better condition conv1's early gradients.

**Sources**: standard CIFAR constants; codebase train.py:152-163; EXP-010 report Next Steps #1.

**Estimated Effort**: low — one-tuple change.

**Risk Assessment**: Near-certain null: a BatchNorm immediately follows conv1 and re-normalizes activations, so the
input std-scaling benefit is largely absorbed. Cheap and clean but lowest expected impact.

### 3. LR peak micro-tuning (0.2 → 0.3)
**Summary**: Raise `PEAK_LR` 0.2→0.3 (cosine + 5% warmup unchanged). k=4, all else fixed.

**Reasoning**: Peak LR was never swept (only WD, EXP-005). A higher peak with the warmup + BN tolerance can
sometimes improve final accuracy via stronger early exploration. Free.

**Sources**: codebase train.py:23; goal-learnings (LR untried).

**Estimated Effort**: low — one-constant change.

**Risk Assessment**: A single run tests one point of a sweep — could help, be neutral, or destabilize early
training (mitigated by warmup). Low-medium ceiling; the converged recipe is already reasonably tuned. Safe.

## Idea Evaluation

**Evidence strength**: Idea 1 (Mixup) has the best-matched evidence — augmentation is *the* demonstrated
non-capacity lever here (Cutout), and Mixup is a well-evidenced complementary mechanism. Critically, the only prior
"regularization saturated" signal was about WD, not augmentation, so the augmentation axis is NOT actually
exhausted. Idea 2 is textbook-correct but mechanistically neutralized by BN. Idea 3 is a blind one-point sweep.

**Mechanism clarity**: Idea 1 — clear and complementary (interpolation regularization on top of occlusion). Idea 2
— clear but absorbed by BN. Idea 3 — plausible but unprincipled (why 0.3 not 0.15?).

**Expected impact**: Idea 1 has the highest ceiling (the augmentation lever has the only non-capacity track record)
AND the highest variance (convergence risk). Ideas 2/3 are low-ceiling.

**Risk profile**: Idea 2 safest/lowest-impact; idea 3 mild; idea 1 moderate-bounded (soft-regression risk from
under-convergence, mitigated by mild α + compile epochs). All fail gracefully.

**Conclusion**: Idea 1 (Mixup) is the lead — it targets the project's one proven non-capacity lever via an untried
complementary mechanism, with risk explicitly bounded by the mild-α + compile-enabler design. This is the
highest-EV move to actually break the plateau; the cheap probes (2,3) are better held as fallbacks if Mixup fails.

## Chosen Idea
**Selected**: Mixup augmentation (mild α=0.2), GPU-vectorized, stacked on Cutout + compile enabler (Idea 1)

**Why this idea**:
Data augmentation is the only non-capacity lever with a track record on this project (Cutout ~+1.1pp), and the
"regularization saturated" caveat applies to weight decay, not augmentation — so the augmentation axis is genuinely
under-explored, with only Cutout tried. Mixup is the canonical complementary augmentation (interpolation vs
occlusion), GPU-vectorizable (no CPU-bottleneck), and the mild-α=0.2 + compile-enabler design explicitly bounds its
one real risk (convergence slowdown at fixed budget). Higher ceiling than the remaining cheap probes.

**Hypothesis**:
Adding mild Mixup (α=0.2) on top of Cutout, compiled to keep ~85–89 epochs, will improve generalization and lift
`best_test_acc` above the 96.10 bar (expected ~96.1–96.4%, via reduced overfitting / lower test loss). If it lands
near or below 96.0, num_epochs + final_test_loss will distinguish "under-converged (Mixup needs more budget)" from
"augmentation axis also saturated" — either outcome sharpens whether 96.0 is the true plateau.

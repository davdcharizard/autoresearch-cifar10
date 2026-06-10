# Brainstorm EXP-023
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new external search this loop — the lead is a single recipe-constant sweep (label smoothing) whose evidence is
  project-internal, and the decisive new signal is the project's own freshly-distilled insight (below). Label
  smoothing background already in repo context: Szegedy et al. 2016 (Inception-v3) introduced it; it mainly improves
  calibration and can slightly help or slightly hurt top-1 depending on how much other regularization is present.
- Knowledge base (`knowledge/README.md`): trivialaugment, cutmix, swa, wrn-dropout — all the augmentation / averaging
  / feature-dropout mechanisms now closed. BlurPool (Zhang 2019) is noted in Unexplored Avenues but not yet a KB entry.

## Experimental History Review

Current best = **96.22%** (EXP-012, commit 6c417a4). 23 experiments; ~15 axes closed. Binding constraint:
generalization/CONVERGENCE at fixed k=4 capacity in 300s (~84–92 epochs).

**DECISIVE new insight (project-insights Medium, EXP-005/011/018/022):** at this short fixed budget the recipe is
**convergence-bound, not overfit-bound** — every *add-a-regularizer* move regressed or nulled: WD↑ (EXP-005 null),
Mixup (EXP-011 null), CutMix (EXP-018 −1.08pp), in-block dropout (EXP-022 −1.37pp, loss 0.195→0.224 under-fit). The
lone gain (TrivialAugment, EXP-012) was a convergence-NEUTRAL substitution/diversification of input aug, NOT "more
total regularization". **Implication: stop adding regularizers; the only productive remaining directions are
convergence-NEUTRAL changes (aug diversity/substitution, input normalization, schedule shape) or REDUCING a
regularizer.**

**Closed axes (do NOT revisit):** capacity k>4 (EXP-004/009), LR-peak 0.2 interior optimum (EXP-016/017),
block-order/pre-act (EXP-015), activation/SiLU (EXP-010), SE attention (EXP-008), weight-decay-UP (EXP-005),
more-epochs alone (EXP-007), auto-aug policy TA≈RA (EXP-014), occlusion-strength/Cutout-size 16-optimal
(EXP-013/021), label-mixing aug (EXP-011/018), weight-averaging EMA/SWA (EXP-006/019/020), in-block dropout (EXP-022).

**Untested levers consistent with the new insight (no added convergence penalty):**
- **Label smoothing VALUE (fixed 0.1 since EXP-000, NEVER swept)** — REDUCING it is exactly the insight's prescription;
  the single recipe regularizer never probed; clean, convergence-neutral, single-constant.
- Per-channel input std-norm (std=(1,1,1)→true std) — convergence-neutral input rescale; expected null (BN absorbs).
- BlurPool / anti-aliased downsampling (Zhang 2019) — a real generalization mechanism, but restructures the strided
  conv path → EXP-015-style compile-graph epoch-cost / attribution risk; implementation-heavy. Deferred.

## Candidate Ideas

### 1. Lower label smoothing (LABEL_SMOOTHING 0.1 → 0.05)
**Summary**: Change the single constant `LABEL_SMOOTHING = 0.1 → 0.05` in train.py (L27). LS has been fixed at 0.1
since EXP-000 and never swept. Everything else identical to the EXP-012 baseline (k=4, TA+Cutout(16), PEAK_LR 0.2
cosine-to-0, compile, seed 42). Compute- and param-neutral fair test.

**Reasoning**: The project's strongest current signal is that the recipe is convergence-bound and ADDING regularizers
fails (EXP-005/011/018/022), so the productive direction is to REDUCE regularization. Label smoothing is the one
recipe regularizer never swept. With TA + Cutout already providing strong input-space regularization (and aug
effectively perturbs/softens the learning signal), a fixed 0.1 LS may be over-softening targets and capping top-1;
halving it to 0.05 lets the model commit to sharper predictions within the epoch budget. This is the cleanest,
lowest-risk instantiation of the "reduce a regularizer" prescription — single constant, zero confound (no compute or
architecture change), fully attributable.

**Sources**: project-insights Medium ("ADDING regularizers fails — convergence-bound, reduce instead", EXP-022);
train.py L27 (LS fixed 0.1 since EXP-000); Szegedy et al. 2016 (label smoothing).

**Estimated Effort**: low — one constant.

**Risk Assessment**: LS top-1 effects are usually small (it mainly improves calibration), so the change may land
within the ~0.2pp noise band even if directionally correct — clearing a +0.1pp bar is not guaranteed. If 0.1 was
actually load-bearing for this recipe, reducing it could slightly increase overfitting and hurt — but the
convergence-bound evidence makes the "reduce helps or is neutral" direction more likely than "reduce hurts". Fails
gracefully (no-improvement). A follow-up could push to 0.0 if 0.05 helps.

### 2. Per-channel input std-normalization (std (1,1,1) → true CIFAR std)
**Summary**: Normalize inputs by true per-channel std (≈(0.247,0.243,0.261)) instead of (1,1,1) (mean-only), train.py
L152-155. Convergence-neutral (a fixed input rescale).

**Reasoning**: The one untried input-side knob; cheap; definitively closes the input-normalization axis. Consistent
with the new insight (no added penalty).

**Sources**: train.py L152-155 (the `std=(1,1,1)` comment flags this); standard CIFAR practice.

**Estimated Effort**: low — one tuple.

**Risk Assessment**: First layer is Conv→BatchNorm; BN almost certainly absorbs a per-channel input rescale →
expected NULL. Low ceiling; an axis-closer, not a real lead.

### 3. BlurPool / anti-aliased downsampling (Zhang 2019)
**Summary**: Replace the stride-2 downsampling (layer2/layer3 first-block conv1 and the 1×1 projection shortcut) with
anti-aliased downsampling: stride-1 conv → fixed blur filter → subsample (a depthwise binomial BlurPool with a
registered-buffer kernel, no learned params, no new deps).

**Reasoning**: A genuine generalization mechanism (improves shift-invariance), the one architectural generalization
lever flagged in Unexplored Avenues, and it does NOT add a stochastic convergence-slowing penalty — so it fits the
"convergence-neutral generalization lever" prescription better than any regularizer.

**Sources**: Zhang 2019 "Making Convolutional Networks Shift-Invariant Again" (ICML); exp-report-021 Unexplored
Avenues; train.py L80-84, L104-106 (strided downsample path).

**Estimated Effort**: medium — a BlurPool2d module + rewiring two downsample sites in BasicBlock.

**Risk Assessment**: Restructuring the strided conv path risks the EXP-015 confound (less-efficient compiled graph →
fewer epochs → attribution muddied), and the blur conv adds some compute that may cost epochs. BlurPool's documented
CIFAR-10 gain is modest (small 32×32 images, only 2 downsample stages). Higher ceiling than Ideas 1/2 if it works,
but higher implementation + confound risk. Defer until the clean convergence-neutral probes are exhausted.

## Idea Evaluation

**Evidence strength**: Idea 1 has the strongest *current* signal — it is the direct instantiation of the project's
freshly-distilled, multiply-confirmed insight (convergence-bound → reduce, don't add) and probes the single recipe
regularizer never swept. Idea 3 has good external (literature) evidence for a real mechanism but carries an
attribution-confound risk the project has been explicitly burned by (EXP-015). Idea 2 is an expected null.

**Mechanism clarity**: Idea 1 — clear and insight-backed (less target softening → sharper top-1 within the
convergence-bound budget). Idea 3 — clear (anti-aliasing → shift-invariance) but with a competing negative (compute/
epoch cost). Idea 2 — almost certainly nulled by BN.

**Expected impact**: Idea 3 has the highest *ceiling* if it works cleanly, but Idea 1 has the highest
*expected value* right now: low risk, zero confound, directly tests the most-supported hypothesis, and opens a sweep
(→0.0) if it moves. Idea 2 ≈ 0.

**Risk profile**: Idea 1 safest (single constant, fully attributable, graceful). Idea 2 safe but null. Idea 3 riskiest
(confound + implementation).

**Feasibility**: Ideas 1/2 trivial; Idea 3 medium.

Conclusion: **Idea 1 (LS 0.1→0.05)** is the lead — the cleanest, best-motivated, zero-confound test of the project's
strongest current insight, and the discipline of exhausting clean convergence-neutral/reduce-a-regularizer probes
before the risky radical one (Idea 3) is correct. Idea 2 is a cheap follow-up axis-closer; Idea 3 (BlurPool) is the
reserved higher-ceiling radical option for a later loop if 1/2 null.

## Chosen Idea
**Selected**: Lower label smoothing (LABEL_SMOOTHING 0.1 → 0.05)

**Why this idea**:
The project's strongest, multiply-confirmed insight is that the recipe is convergence-bound (not overfit-bound) at the
300s budget, so ADDING regularizers fails and the productive direction is to REDUCE regularization. Label smoothing is
the single recipe regularizer never swept. Reducing it to 0.05 is the cleanest, lowest-risk, fully-attributable,
convergence-neutral instantiation of that prescription — a single-constant fair test.

**Hypothesis**:
Reducing label smoothing 0.1→0.05 will lift best_test_acc above the 96.32 bar by letting the model commit to sharper
predictions (less target over-softening) within the convergence-bound epoch budget, where strong input aug
(TA+Cutout) already supplies regularization. If instead acc falls or stays within noise, either 0.1 was load-bearing
or LS top-1 effects are too small to clear the bar, and the label-smoothing axis is closed (with 0.0 a possible
follow-up if 0.05 helps directionally).

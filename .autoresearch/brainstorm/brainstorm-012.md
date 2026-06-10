# Brainstorm EXP-012
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- **TrivialAugment (Müller & Hutter, ICCV 2021)** (https://arxiv.org/abs/2103.10158): a parameter-free, tuning-free
  automatic-augmentation policy that applies a *single* op per image at a *uniformly random* strength. On CIFAR-10
  with WideResNet it matches/beats the heavily-tuned AutoAugment and RandAugment at near-zero cost. Key relevance:
  it is the strongest "no-knobs" augmentation upgrade and is the canonical SOTA-recipe pairing **TA + Cutout** for
  CIFAR WRNs. It adds *photometric + geometric* diversity (rotate/shear/translate/color/contrast/brightness/
  sharpness/solarize/posterize) — a mechanism orthogonal to Cutout (occlusion) and Mixup (interpolation).
- **torchvision.transforms.TrivialAugmentWide** (verified present, torchvision 0.24.1; torch 2.9.1): drop-in PIL
  transform, no new dependency. Confirmed importable in this env.
- **DeVries & Taylor 2017 (Cutout)** (already in recipe, EXP-002/003): occlusion regularizer; complementary to TA.

## Experimental History Review
- Current best **96.00%** (EXP-003, k=4 WRN + GPU-Cutout + bf16/channels_last/cosine/Nesterov/LS). Bar **≥96.10**.
- **Compiled-k4 ≈ baseline (null)**: EXP-007 (torch.compile reduce-overhead) → 95.92 ≈ 96.00 within the ~0.2pp
  noise band, while buying 77→89 epochs of throughput headroom. ⇒ compile is a clean *execution-only* enabler with
  no standalone accuracy effect; any gain over baseline in a compiled run is attributable to the intervention.
- **Augmentation axis under-tested, not closed**: only Cutout (worked, +1.1pp cumulative) and mild Mixup (EXP-011,
  α=0.2 → 95.86, null) tried. Mixup is a *weak* regularizer; its null shows "a second weak aug on a converged net
  gives diminishing returns" — it does NOT test a *strong, diverse* augmentation policy.
- **Closed axes** (do not revisit): width ≥k5 (EXP-004/009, epoch wall), weight-decay (EXP-005), EMA/SWA with
  cosine-to-0 (EXP-006), more epochs alone (EXP-007), SE/channel-attention (EXP-008), activation/SiLU (EXP-010).
- **Dominant constraint (project-insights High)**: 300s wall-clock epoch wall — any intervention that slows
  convergence risks underfitting (the failure mode behind k=6/k=5). Augmentation that needs the *full* schedule to
  pay off is the key risk for this loop. Mitigation: keep ops cheap (PIL, no GPU sync) + compile for headroom.
- **Untried gaps**: strong auto-augmentation (TA/RandAugment), input std-normalization (std=(1,1,1) currently),
  LR-schedule micro-tuning.

## Candidate Ideas

### 1. TrivialAugment (Wide) added to the input pipeline, kept alongside Cutout (+ compile enabler)
**Summary**: Insert `transforms.TrivialAugmentWide()` into the train transform (before `ToTensor`), keeping the
existing RandomCrop(4)+HorizontalFlip and the GPU-side Cutout(16). This is the canonical SOTA CIFAR-WRN strong-aug
recipe (TA + Cutout). Add the validated `torch.compile(reduce-overhead)` enabler so the freed launch-bound headroom
(EXP-007: 8ms/step, ~89 epochs) absorbs TA's extra CPU augmentation cost and keeps the run from becoming
epoch-starved. Eval stays on the eager handle.
**Reasoning**: Augmentation is the project's ONLY proven non-capacity lever (Cutout drove 94.90→96.00). The axis
was only probed with *weak* Mixup (EXP-011); a *strong, diverse* policy is mechanistically different — it injects
photometric+geometric invariances Cutout/Mixup never touch. TA beats tuned AutoAugment on CIFAR-WRN in the
literature at zero tuning cost, and TA+Cutout is the standard high-accuracy pairing. The k=4 net is converged with a
small overfit gap, so the gain (if any) comes from *better invariance*, not just overfit reduction.
**Sources**: TrivialAugment (arXiv:2103.10158); Cutout EXP-002/003; compile enabler EXP-007 (project-insights
Medium); Mixup null EXP-011 (goal-learnings Failed Approaches Low).
**Estimated Effort**: low (one transform line + 4-line compile enabler; train.py-only).
**Risk Assessment**: Main risk = **budget/underfit**: TA is a stronger regularizer and adds per-sample CPU PIL work;
if the 8-worker dataloader can't keep the launch-bound GPU fed, throughput drops → fewer epochs → underfit (the
k=6/k=5 failure mode). Mitigations: TA applies only ONE cheap PIL op/image (no GPU sync, unlike the EXP-002 Cutout
bottleneck), and compile buys ~15% headroom. Worst case is a graceful no-improvement (lands ~95.5–95.9 like other
converged nulls). Monitor realized epochs/img-s as the abort signal.

### 2. Per-channel input std-normalization (fix std=(1,1,1) → CIFAR per-channel std)
**Summary**: Change `std=(1,1,1)` to the CIFAR-10 per-channel std `(0.2470,0.2435,0.2616)` in `transforms.Normalize`
so inputs are unit-variance, matching the conv1 Kaiming-init assumption. Pure normalization change, zero throughput
cost, no budget risk.
**Reasoning**: Currently inputs are only mean-subtracted (per-channel var ~0.06), so conv1 receives ~4× smaller-
magnitude inputs than its He-init assumes → conv1 effectively trains with a smaller LR (gradient scales with input
magnitude). BN normalizes activations *after* conv1, so the forward pass is fine, but the first layer's optimization
is mildly mis-scaled. Std-norm removes a known deviation from standard practice.
**Sources**: train.py L152-155 (the explicit std=(1,1,1) note); He et al. 2015 Kaiming-init (cited in code).
**Estimated Effort**: low (one-line change).
**Risk Assessment**: Very low risk, but very low ceiling — BN almost certainly absorbs the effect, so the expected
outcome is a noise-scale null. The cheapest clean probe; best as a fallback, not a lead.

### 3. Heavier Cutout (two holes / larger hole)
**Summary**: Increase Cutout regularization (e.g., two 16px holes, or one 20px hole).
**Reasoning**: Cutout is the one aug that worked; more occlusion *might* regularize harder.
**Sources**: DeVries & Taylor 2017; EXP-002/003.
**Estimated Effort**: low.
**Risk Assessment**: Low ceiling and redundant-mechanism risk — it's the *same* occlusion lever already near its
sweet spot at 16px on a converged net; EXP-011 (Mixup) suggests stacking more of the same regularization type gives
diminishing returns. Likely null. Discarded in favor of the orthogonal-mechanism TA.

## Idea Evaluation
Idea 1 (TrivialAugment) has by far the strongest **evidence** (TA is a published SOTA CIFAR-WRN augmentation, and
TA+Cutout is the standard high-accuracy recipe) and the clearest **distinct mechanism** (photometric+geometric
invariance, orthogonal to the occlusion/interpolation augs already tried). Its **expected impact** is the highest of
the three — it is the only candidate with a literature-backed chance of a *non-noise* gain. Its **risk** is the
budget/underfit failure mode, but that is the well-understood dominant constraint here and is mitigated (cheap
single PIL op, no GPU sync, compile headroom) and fails gracefully to no-improvement.

Idea 2 (std-norm) is the safest but has a near-null expected ceiling (BN absorbs it) — it's a clean cheap probe, not
a real shot at +0.1pp. Idea 3 (heavier Cutout) re-pulls the same occlusion lever already at its sweet spot —
diminishing-returns risk, low ceiling.

Mechanism clarity + evidence + the highest (and only literature-backed) upside select Idea 1. Idea 2 is the natural
fallback for the next loop if TA underfits or lands null.

## Chosen Idea
**Selected**: TrivialAugment (Wide) added to the input pipeline, kept alongside Cutout (+ compile enabler)

**Why this idea**:
Augmentation is the project's only proven non-capacity lever, and it has only been tested with weak Mixup — a strong,
diverse auto-augmentation policy (TA) is the highest-evidence, highest-ceiling untried move, and TA+Cutout is the
canonical SOTA CIFAR-WRN recipe. It is parameter-free (no tuning gamble), a train.py-only change with no new
dependency, and its only real risk (throughput→underfit) is directly mitigated by the validated compile enabler and
by TA's cheap single-op, GPU-sync-free design. Compiled-k4 ≈ baseline (EXP-007 null) keeps attribution clean: any
gain over 96.00 is attributable to TA, not compile.

**Hypothesis**:
Adding TrivialAugmentWide to the train pipeline (with Cutout retained, compiled for throughput) will improve
generalization via stronger input-space invariance and lift `best_test_acc` above the 96.10 bar (expected
~96.1–96.5%), PROVIDED the run still fits ≳75 epochs (fair, converged test). If realized epochs fall well below ~70
(throughput-starved), a null/regression would instead indicate TA underfits at this 300s budget rather than that the
augmentation axis is closed.

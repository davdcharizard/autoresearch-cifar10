# Brainstorm EXP-027
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **He et al. — "Bag of Tricks for Image Classification" (CVPR 2019), §4 model tweaks** (knowledge/papers/bag-of-tricks.md)
  **ResNet-D downsample**: the standard residual downsample uses a 1×1 conv with stride 2 on the shortcut, which IGNORES 3/4 of the input feature map (it samples every other pixel). ResNet-D fixes this by inserting a 2×2 average-pool (stride 2) BEFORE a stride-1 1×1 conv, so all input pixels contribute. Reported ~+0.3-0.5% top-1 on ImageNet ResNet-50 at negligible cost. CRITICAL for our budget: this is NEAR-COMPUTE-NEUTRAL — it touches only the cheap 1×1 SHORTCUT (the 1×1 conv FLOPs are unchanged since it still runs at the downsampled resolution; only a trivial avgpool is added). This is the key contrast with EXP-024 BlurPool, which moved the heavy 3×3 MAIN conv to stride-1 and 4×'d FLOPs → epoch wall.

## Experimental History Review

Current best **96.22%** (EXP-012, commit 6c417a4); bar = 96.32 (+0.1pp). 26 experiments; plateau exhaustively mapped (~19 axes closed): all scalar knobs bracketed (LR/Cutout/LS/WD/batch), aug family closed, regularizer-adding fails (convergence-bound), compute-adding hits the epoch wall (k≥5, pre-act, BlurPool), batch-scaling is compute-bound, weight-averaging closed, and compute-neutral convergence-POLISH (EMA/SWA/LS-down/Bag-of-Tricks zero-γ+no-bias-decay) moves test LOSS but not TOP-1 (project-insight Medium).

**Most relevant near-miss**: EXP-024 BlurPool (anti-aliased downsampling) → 95.66 (−0.56pp) but **COMPUTE-CONFOUNDED** (epochs 91→77: it moved the main 3×3 conv to stride-1, 4×'ing FLOPs at the 2 heaviest convs). The report flagged that anti-aliasing's intrinsic merit was masked by the epoch wall and could only be tested fairly with a cheaper implementation. ResNet-D is exactly that cheaper implementation: it improves the lossy downsample WITHOUT touching the main conv, so it should stay ~91 epochs.

**Untried gaps:** (1) **ResNet-D downsample** (the compute-neutral way to fix the lossy stride-2 shortcut — the fair re-test of EXP-024's masked hypothesis); (2) **per-channel input std-norm** (expected BN-absorbed null); (3) test-time flip augmentation (see candidate 3 — rejected on integrity grounds).

Codebase (confirmed this loop): downsample shortcut at train.py L80-84 = `Conv2d(in,out,1,stride=2)+BN`; main path conv1 is already 3×3 stride-2 (ResNet-B style). Only the 2 downsample blocks (layer2/layer3 first block) use the strided shortcut; layer1 block0 uses a stride-1 1×1 (channel change only); all other blocks use Identity.

## Candidate Ideas

### 1. ResNet-D downsample (avgpool-2 + 1×1-stride-1 shortcut)
**Summary**: Replace the lossy strided shortcut `Conv2d(in,out,1,stride=2)+BN` with the ResNet-D form `AvgPool2d(2,stride=2) → Conv2d(in,out,1,stride=1)+BN` for the 2 downsample blocks (stride≠1). The avgpool downsamples (averaging all 4 pixels in each 2×2 window) before the 1×1 conv, so no input information is dropped. The stride-1 channel-change block (layer1 block0, in≠out but stride==1) keeps its plain 1×1; Identity blocks unchanged. Main-path conv1 stays 3×3 stride-2 (untouched).

**Reasoning**: This is the COMPUTE-NEUTRAL re-test of the anti-aliasing/lossy-downsample hypothesis that EXP-024 (BlurPool) couldn't test fairly. BlurPool regressed only because it moved the heavy 3×3 conv to stride-1 (epochs 91→77). ResNet-D touches only the cheap 1×1 shortcut: the 1×1 conv FLOPs are unchanged (still at the downsampled resolution) and the added 2×2 avgpool is trivial (~65k ops at layer2). So epochs should hold ~91 — a fair test. It targets generalization (the residual path no longer discards 3/4 of pixels at the 2 downsample points) which is the binding constraint, and it's documented (~+0.3-0.5% ImageNet). Params unchanged (avgpool is parameter-free, 1×1 conv shape identical) → 4,299,866.

**Sources**: He et al. "Bag of Tricks" §4 ResNet-D (knowledge/papers/bag-of-tricks.md); EXP-024 report (BlurPool compute-confound — the near-miss this fixes); train.py L80-84.

**Estimated Effort**: low (rewrite the 5-line shortcut branch; one 300s run).

**Risk Assessment**: (a) Throughput-neutrality must be VERIFIED — per the EXP-015 lesson, a restructured graph can be less torch.compile-efficient even at equal FLOPs; if epochs drop materially the result is confounded (mitigated: avgpool is a standard fused op on only 2 paths, dt expected ~8ms). (b) On a shallow CIFAR net with only 2 downsample points the gain may be below the ~0.2pp noise floor → no-improvement. (c) The avgpool slightly smooths the shortcut signal (mild low-pass) — could in principle blur useful high-freq info, but averaging is exactly the anti-aliasing benefit. Fails gracefully to no-improvement; compute/param-neutral so no epoch-wall/integrity risk.

### 2. Per-channel input std-normalization
**Summary**: Change `std` in the train `transforms.Normalize` from `(1,1,1)` to the true CIFAR-10 per-channel std `(0.2470, 0.2435, 0.2616)` (mean already subtracted). NOTE: the frozen eval harness (prepare.py) also uses std=(1,1,1), so to keep train/test preprocessing matched this would create a train/test normalization MISMATCH unless the model is robust — BN at the stem largely absorbs global input scale, so accuracy should be ~unchanged, but the mismatch is a subtle risk.

**Reasoning**: The last untouched scalar in the input pipeline; closes the input-normalization axis. Compute-neutral.

**Sources**: train.py L152-155; prepare.py L13 (eval uses std=(1,1,1) — train/test mismatch consideration); memory (flagged as the last cheap probe, expected null).

**Estimated Effort**: low (one-line edit; one run).

**Risk Assessment**: Expected NULL or mild REGRESSION — because eval normalizes with std=(1,1,1), changing only the TRAIN std introduces a train/test input-scale mismatch; BN's running stats are computed on train scale, so eval inputs would be at a different scale. This could actually HURT (unlike the originally-imagined clean null). Lower value than I previously assessed; the mismatch makes it a weaker closer than ResNet-D.

### 3. Test-time augmentation (horizontal-flip logit averaging) — REJECTED on integrity grounds
**Summary**: Gate the model's eval-mode forward (`not self.training`) to return `0.5*(f(x) + f(flip(x)))`, averaging logits over the image and its horizontal flip. Training forward unchanged. Would reliably add ~+0.2-0.5% top-1 on CIFAR.

**Reasoning / why REJECTED**: Although it edits only train.py (model forward) and does not touch prepare.py or call evaluate() more than once, it CHANGES THE EVALUATION PROTOCOL relative to the single-forward 96.22 baseline and all 26 prior experiments. The entire experiment history and the leaderboard baseline assume single-view inference; introducing eval-time TTA makes the comparison apples-to-oranges and effectively circumvents the harness's implicit single-view measurement convention (a hard constraint: "the evaluation harness must not be altered or circumvented"). The gain reflects an inference protocol change, not a better-trained model within the 300s budget. Per the analyze integrity test ("if you were trying to game this metric, would this approach qualify?"), this is too close to the line. DISCARDED — documented here so future loops don't reconsider it naively.

**Sources**: prepare.py L32-47 (eval calls model.eval() + single model(x)); goal Hard Constraints (eval harness not to be circumvented).

**Estimated Effort**: n/a (rejected).

**Risk Assessment**: n/a (rejected — would likely be ruled `invalid`).

## Idea Evaluation

Candidates 1 and 2 respect all hard constraints (train.py-only, no new deps, single GPU, ≤1 eval/epoch, no seed hacking) and are untried; candidate 3 is rejected on integrity grounds (above).

- **Evidence strength**: Idea 1 has the best combination — documented ImageNet gain AND a project-specific rationale (it's the compute-neutral fix for EXP-024's masked anti-aliasing hypothesis, directly addressing why that near-miss failed). Idea 2 is now assessed as expected-null-or-mild-regression (train/test std mismatch), weaker than first thought.
- **Mechanism clarity**: Idea 1 is crisp — stop discarding 3/4 of pixels at the 2 residual downsample points; documented anti-aliasing/information-preservation benefit; compute-neutral so no epoch-wall confound. Idea 2's mechanism predicts ~null and carries a mismatch risk.
- **Expected impact**: Idea 1 is the only remaining lever with BOTH a documented positive top-1 effect AND no epoch-wall/polish-only limitation — it targets generalization (not loss-only) via architecture, the class that hasn't been fairly tested compute-neutrally. Idea 2 is a closer.
- **Risk profile**: Both fail gracefully to no-improvement. Idea 1's main risk is throughput non-neutrality (verifiable via epoch count); Idea 2 risks a train/test mismatch regression.
- **Feasibility**: Both low-effort.

Idea 1 (ResNet-D) clearly leads: it is the fair, compute-neutral re-test of the most promising prior near-miss (anti-aliased/lossless downsampling), targets generalization through architecture, and keeps the 91-epoch baseline. Idea 2 is demoted to a later closer (and its mismatch caveat noted). Idea 3 is rejected.

## Chosen Idea
**Selected**: ResNet-D downsample (avgpool-2 + 1×1-stride-1 shortcut)

**Why this idea**:
It is the compute-neutral re-test of EXP-024's masked hypothesis: BlurPool showed anti-aliased/information-preserving downsampling is plausibly beneficial but couldn't be tested fairly because it 4×'d the main-conv FLOPs and cratered epochs. ResNet-D achieves the same "don't discard pixels at downsampling" benefit by touching ONLY the cheap 1×1 shortcut (avgpool + stride-1 1×1 = identical conv FLOPs + trivial pool), so it should hold ~91 epochs. It is the last untried lever that targets GENERALIZATION through architecture (not optimization polish, which only moves loss) without hitting the epoch wall, and it's documented to help.

**Hypothesis**:
Replacing the lossy stride-2 1×1 shortcut with ResNet-D (avgpool-2 → stride-1 1×1) at the 2 downsample blocks preserves residual-path information and lifts `best_test_acc` above the 96.32 bar at an unchanged ~91 epochs / 4,299,866 params / dt ~8ms / <600s. Falsifiable: if epochs drop materially (compile-graph confound, cf. EXP-015) the test is confounded; if epochs hold but accuracy lands within ±0.2pp of 96.22, the lossy-shortcut effect is negligible on this shallow 2-downsample net and the plateau stands.

# Brainstorm EXP-066
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- (no new external fetch this loop) Progressive resizing / resolution scheduling is a well-established "train fast, finish sharp" recipe (fast.ai DAWNBench CIFAR/ImageNet entries; FFCV ImageNet configs; Howard & Gugger). Core mechanism: train early epochs at REDUCED input resolution (cheaper steps → more steps in a fixed compute budget), then finish at FULL resolution so the converged model and its BN statistics match the (full-res) evaluation distribution. The FixRes line (Touvron 2019) formalizes that a train↔test resolution gap hurts, which is WHY the schedule must END at the eval resolution — a constraint this plan respects (eval is frozen at 32×32; the tail phase trains at 32×32).
- This is the FIRST genuinely-untested mechanism class in many loops: it is NOT a scalar/schedule/aug/capacity/optimizer/norm retune (all mapped) — it is a NEW axis (input spatial resolution as a function of training-time fraction) that directly attacks the binding constraint.

## Experimental History Review
- **Current best: 96.45 (EXP-054)** = k=4 WideResNet-20 + RandomApply(AugMix w3,d-1, p=0.5) + GPU Cutout16 + cosine peak0.2/warmup0.05/Nesterov/LS0.1/WD1e-4 + compile(reduce-overhead), ~91 ep, dt 8ms. 66 experiments, 8 improvements.
- **The binding constraint is convergence/epochs, not regularization** (project-insights High): every compute/layer ADDITION → fewer epochs → underfit; every compute-neutral regularizer/objective/geometry retune moves loss/calibration but NOT top-1. The plateau is mapped across augmentation, capacity (×4 dirs), optimizer, LR, normalization, head, batch, activation, regularizers, weight-averaging. EXP-065 just closed the LS axis from both sides (0.05 and 0.15 both regress; 0.1 optimal).
- **PROVEN winning mechanism = "buy net-new epochs via throughput"**: EXP-003 (move Cutout CPU→GPU) bought epochs and won **+0.58pp** (95.42→96.00) — the single clearest lever class in this project's history. The metric is epoch-starved; anything that buys real epochs WITHOUT hurting per-epoch learning has historically paid off.
- **dt-reduction attempts that FAILED** (do not repeat): cudnn.benchmark (EXP-040, no-op under compile), max-autotune + off-budget warmup (EXP-045/046, no real epoch gain — conv-dt floor reached at the FIXED 32×32 shape). KEY GAP: every dt-reduction attempt so far kept the input shape FIXED at 32×32 and tried to make the SAME convs faster (floor already hit). **NONE reduced the actual FLOPs by shrinking the spatial resolution.** Resolution scheduling reduces FLOPs ~quadratically in side-length — a lever the conv-dt-floor closure does NOT cover.
- **Related-but-distinct closed items**: SE blocks (EXP-008, old recipe, "no accuracy" + cost epochs), CutMix (EXP-018, −1.3pp), ResNet-D downsample (EXP-027), in-block dropout (EXP-022, underfit). Progressive resizing is none of these — it changes the INPUT pixel grid, not the architecture or the augmentation policy.
- **Genuinely UNTESTED**: input-resolution scheduling (this loop); BN momentum (0.1, never tuned); BN eps (1e-5, never tuned).

## Candidate Ideas

### 1. Progressive resolution scheduling — train early at 24×24, finish at 32×32 (buy net-new epochs)
**Summary**: Downscale the training input with `F.interpolate(inputs, size=R, mode="bilinear")` on the GPU, with R driven by training-time fraction: R=24 for roughly the first half of the 300s budget, then R=32 for the remainder. Eval is untouched (frozen 32×32). Single new mechanism; all hyperparameters else byte-identical to EXP-054. Cutout size scales with R (CUTOUT_SIZE·R/32) so the augmentation stays proportionate.

**Reasoning**: The metric is epoch-bound (project-insights High) and the ONE proven lever class is "buy net-new epochs via throughput" (EXP-003, +0.58pp). Every prior throughput attempt kept the 32×32 shape and hit the conv-dt floor (EXP-040/045/046). Resolution scheduling is the untried, higher-leverage throughput knob: a 24×24 step has (24/32)²=0.5625× the conv FLOPs, so early-phase steps are materially cheaper → MORE total steps fit in Σdt=300s → more effective epochs. Low-res early training learns coarse features fine (CIFAR objects are recognizable at 24×24); the full-res tail re-sharpens features and re-adapts BN running stats to the 32×32 eval distribution (FixRes: the schedule MUST end at eval resolution — it does). The architecture needs NO change: global-avg-pool before the FC head accepts any spatial size (24→12→6 through the two stride-2 downsamples vs 32→16→8). If the cheaper early epochs translate into a better-converged final model, this clears the +0.1 bar; even a null cleanly maps a brand-new axis.

**Sources**: train.py L224-246 (input enters loop, GPU Cutout at L231, compiled forward at L240 — resize inserts between L231-cutout and the autocast forward); EXP-003 (throughput→epochs win); EXP-040/045/046 (32×32-fixed dt floor, the gap this fills); project-insights High (epoch-bound); goal-learnings (FixRes/eval-resolution-match constraint).

**Estimated Effort**: Low-moderate (one `F.interpolate` call + a time-fraction resolution selector + proportional Cutout). Params unchanged (4,299,866). cudagraph note: the resize is OUTSIDE the compiled forward, so reduce-overhead captures ONE cudagraph per distinct input shape (2 shapes → 2 graphs, each captured once; the single transition step pays a one-time recapture spike — negligible vs thousands of steps). This respects the EXP-042 rule (no data-dependent branch INSIDE the compiled forward; the shape is fixed within each phase).

**Risk Assessment**: Moderate, but the safest-failing of the radical options. Failure modes: (a) no-improvement if 24×24 early training under-learns fine detail that the short full-res tail can't recover — bracketed cleanly as an axis closure; (b) the 2-graph recapture or interpolate overhead eats the FLOP savings (mitigated: only 2 shapes, interpolate is a cheap kernel); (c) if dt is more launch-bound than compute-bound even under cudagraph, the 24×24 speedup is smaller than the FLOP ratio suggests — still ≥0 epoch gain, not a regression risk. No scope/dep/seed risk (train.py only, GPU op, no new deps). Eval distribution is protected (tail trains at 32×32).

### 2. BN momentum reduction (0.1 → 0.05) — longer EMA window for eval running stats
**Summary**: Set `momentum=0.05` on all `BatchNorm2d` constructors. Single static-arg change.
**Reasoning**: Under heavy AugMix the per-batch BN stats are noisy; a longer EMA window lowers eval-time running-stat estimation variance over the (same) augmented operating distribution — distinct from EXP-061's clean-recalib (which CHANGED the distribution). Compute-/throughput-neutral, cudagraph-safe (static arg).
**Sources**: train.py BN constructors; EXP-061 (BN-stat operating point); brainstorm-065 Idea 2 (carried fallback).
**Estimated Effort**: Trivial.
**Risk Assessment**: Low, low-evidence. With cosine-to-0 the tail is near-frozen-weight so default-momentum running stats are already stable; a longer window folds in slightly-staler higher-LR batches → near-noise, mild-regression-possible.

### 3. BN eps increase (1e-5 → 1e-3) — soft low-variance-channel down-weighting
**Summary**: Set `eps=1e-3` on all `BatchNorm2d`. Single static-arg change.
**Reasoning**: Larger eps shrinks the normalized output of low-variance channels (divide by sqrt(var+eps)), mildly down-weighting less-informative channels — a soft implicit regularizer, untested.
**Sources**: train.py BN constructors; brainstorm-065 Idea 3 (carried fallback).
**Estimated Effort**: Trivial.
**Risk Assessment**: Low, very-low-evidence. On well-activated k=4 channels eps 1e-5 vs 1e-3 is negligible for most channels → near-certain exact null.

## Idea Evaluation
- **Evidence strength**: Idea 1 has by far the strongest grounding — it extends the project's single clearest winning mechanism (throughput→epochs, EXP-003 +0.58pp) along the ONE throughput axis never tried (spatial resolution), and it fills a specific, identified gap (all prior dt attempts kept the 32×32 shape and hit the conv-dt floor). Ideas 2/3 are low-evidence micro-probes on an exhausted plateau.
- **Mechanism clarity**: Idea 1 clear and quantified (FLOPs ∝ side², 24×24 ≈ 0.56× → cheaper early steps → more epochs; FixRes-respecting full-res tail). Idea 2 plausible-but-tail-already-stable. Idea 3 real-but-negligible-magnitude.
- **Expected impact**: Idea 1 is the only candidate that targets the actual binding constraint (epochs) with a real lever rather than retuning a saturated knob — highest ceiling. Ideas 2/3 near-noise.
- **Risk profile**: Idea 1 is moderate (a new mechanism) but fails gracefully to no-improvement (no scope/dep/eval-distribution risk; tail protects the eval resolution). Ideas 2/3 are lower-risk but also near-zero-EV.
- **Feasibility**: Idea 1 low-moderate (well-understood op, one cudagraph caveat already reasoned through); Ideas 2/3 trivial.
- **Conclusion**: Lead with **Idea 1 (progressive resolution scheduling)**. After 65 mapped retunes, the honest path to the +0.1 bar is to attack the epoch bound itself, and resolution scheduling is the highest-leverage, best-evidenced, genuinely-untested way to buy epochs — directly extending this project's proven throughput→epochs win. BN-momentum/BN-eps remain trivial fallbacks for later loops.

## Chosen Idea
**Selected**: Progressive resolution scheduling — train early epochs at 24×24, finish at 32×32, with proportional Cutout, to buy net-new epochs within the fixed Σdt=300s budget.

**Why this idea**: It is the first genuinely-new mechanism class in many loops and it targets the binding constraint directly. The metric is epoch-bound, and the project's single clearest historical win (EXP-003, +0.58pp) came from buying epochs via throughput. Every subsequent throughput attempt kept the input at 32×32 and hit the conv-dt floor (EXP-040/045/046); resolution scheduling reduces conv FLOPs ~quadratically in side-length — the one throughput axis never tried. It needs no architecture change (global-avg-pool head is resolution-agnostic), respects the frozen 32×32 eval (FixRes-correct: the tail trains at full resolution and re-adapts BN), is train.py-only with no new deps, and is cudagraph-safe (resize outside the compiled forward; one stable graph per phase). It fails gracefully to a clean axis-closure.

**Hypothesis**: Training the first ~50% of the 300s budget at 24×24 (proportional Cutout) and the remainder at 32×32 will fit meaningfully more effective epochs than the 91-epoch all-32×32 baseline, and — because the model is epoch-starved — the extra optimization will raise best_test_acc to ≥ 96.55 (baseline 96.45 + 0.1pp). The full-res tail re-aligns features and BN stats to the 32×32 eval distribution, avoiding a FixRes train↔test resolution gap. Most-likely alternative outcome: a within-noise null if 24×24 early training under-learns fine detail the short tail can't fully recover, which cleanly opens-and-brackets the resolution-schedule axis.

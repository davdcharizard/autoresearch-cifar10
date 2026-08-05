# Brainstorm EXP-021
**Created**: 2026-06-30

## Web Search & Literature Review

- **airbench / Keller Jordan, arXiv:2404.00498** (knowledge/references/fast-cifar10-recipes.md; WebSearch 2026-06-30): the path from 95%→96% is **depth + aug**, explicitly — "they add a third convolution to each block — this is the depth trick that increases representational capacity," plus 12px Cutout (we already have) and more epochs (we already have ~4×). Width is the 94→95 trick; **depth is the 95→96 trick.** Also introduces **derandomized "alternating flip"**: "Replacing random flip with alternating flip improves the performance of every training they considered, with the exception of those trainings which do not benefit from horizontal flipping at all" — a deterministic variance-reduction over standard RandomHorizontalFlip.
- **fast-cifar10-recipes.md** (this goal's knowledge base): DavidNet/hlb/airbench lineage; whitening front-end (we have it), GELU (hlb/airbench use it), logits×0.125 (we have it).
- **torch-compile-throughput.md** (this goal's knowledge base): torch.compile gives a validated, math-equivalent **+12% img/s** on this exact net (EXP-014), with off-budget warmup. Banked but unused — available to fund a more expensive backbone's per-step cost.

## Experimental History Review

- **Current best 96.38** (EXP-008, commit 07c3760): whitened ResNet-9/DavidNet + ReZero GatedResidual@layer2 + EMA(0.998) + flip-TTA + Cutout12 + RandomErasing.
- **The only architectural WIN ever: EXP-004** — adding the first ReZero `GatedResidual(256)` at the 8×8/layer2 stage (95.87→96.00). Depth at the proven full-speed stage helped.
- **15 STRAIGHT NULLS (EXP-005, 006, 007, 009–020)** closing every other within-DavidNet axis: capacity-WIDTH (EXP-007/014 — saturated, compile-funded width LOST), optimizer/Muon (009/010), all 3 input-aug mechanisms (008 occlusion won once; 011 mixing/015 transform tied), reg-scalars (012), SAM (013), epochs/throughput (014 — "generalization ceiling, not epoch-bound"), BN-noise (016/017), downsampling/BlurPool (018), channel-attention/SE (019), and LR-schedule-shape/cosine (020). Recurring artifact: a one-session same-session lead that collapses on a confirmation pair (low-control-draw, seen 4× in 016/017/019/020) → **two same-session pairs now mandatory.**
- **The crack in the ceiling diagnosis**: EXP-014's "generalization ceiling, not capacity" verdict tested more EPOCHS and more WIDTH — it NEVER tested more DEPTH. Depth changes the function class (nonlinear composition); width at a saturated stage does not. Depth at 8×8 is the one axis with a positive prior (EXP-004) AND external corroboration (airbench 95→96), and it has never been combined with compile funding (the thing EXP-005/007 lacked → under-anneal).
- **Untried gaps**: (a) compile-funded DEPTH at the proven 8×8 stage; (b) derandomized/alternating flip (a sampling change, not a new aug content — orthogonal to the saturated aug lane); (c) the global-readout head (untouched since EXP-001).

## Collected Ideas

- (history+lit) **Compile-funded depth**: add a 2nd ReZero block (or 3rd conv) at the proven 8×8 layer2 stage, funded by torch.compile +12% so it anneals — synthesis of the only win (EXP-004) + the banked-but-unused lever (EXP-014), targeting the untested DEPTH crack in the ceiling diagnosis.
- (lit, orthogonal sampling) **Alternating/derandomized horizontal flip**: replace i.i.d. RandomHorizontalFlip with antithetic per-image flip parity (airbench: "improves every training considered"). Throughput-free variance reduction.
- (algorithm/readout) **AdaptiveConcatPool head**: concat global avg+max pool → 1024-d → Linear; the one structural component never touched.
- (lit) **GELU activations** (hlb/airbench use them) — smoother nonlinearity; was an EXP-020 finalist, not chosen; cheap.
- (orthogonal) **Whitening front-end tweak**: larger patch / more whitened directions (currently 3×3→54ch).
- (moonshot) **Wholesale different backbone** (pre-activation ResNet / wider-2-stage / ConvNeXt-ish block) — high variance, likely under-trains at 300s.
- (simplification) **Antithetic flip computed off the existing flip** with everything else unchanged — minimal-surface variance cut.

## Combinations

- **Compile-funded depth + alternating flip**: airbench's *actual* 95→96 delta is depth + (we already have) cutout/epochs, and alternating flip improves every training on top — the deeper net has more capacity to benefit from the cleaner (lower-variance) flip signal. Plausibly stronger together, but two variables hurt attribution → keep single-variable this loop; hold the combo as the natural follow-up if either wins.
- **Depth + GELU**: airbench/hlb pair depth with GELU; smoother activation may help a deeper net's gradient flow. Deferred — confounds the depth test.

## Candidate Ideas

### 1. Compile-funded DEPTH at the proven 8×8 stage
**Summary**: Add a second ReZero `GatedResidual(256)` (α init 0 → identity at init) into `layer2` (8×8/256), and fund the per-step cost with torch.compile's validated +12% throughput (off-budget warmup) so the deeper net still anneals in ~135–150 epochs. Asks the sharp question "if one gated 8×8 block helped (EXP-004), does a second?" — now with the compile funding that EXP-005/007 lacked. See proposals/idea-01.md.

**What it targets**: The generalization ceiling at ~96.3–96.5 — specifically the UNTESTED axis in EXP-014's "ceiling not capacity" diagnosis. EXP-014 added epochs and width; depth changes the function class (more nonlinear composition), which is the lever airbench uses for 95→96 and the only lever that ever won here (EXP-004).

**Reasoning**: Three independent evidence sources converge on depth-at-8×8: (a) airbench's explicit 95→96 "third conv per block"; (b) EXP-004's win; (c) EXP-005/007 LOST to under-anneal (capacity useful but unannealed), which compile directly fixes. Identity-init means the deeper net starts bit-equivalent to the proven net → de-risked cold start.

**Sources**: proposals/idea-01.md; arXiv:2404.00498; EXP-004; EXP-014 + knowledge/references/torch-compile-throughput.md; knowledge/references/rezero-identity-init.md.

**Estimated Effort**: Medium — port the EXP-014-validated compile wrapper (off-budget warmup, separate EMA eval path) + add one GatedResidual; smoke num_epochs≥135 with compile ON.

**Risk Assessment**: Primary risk under-anneal if +12% under-funds a 2nd 256-ch block (fallback: single extra conv, or compile-funded headroom check first). Secondary: the ceiling holds even for depth → tie (still closes the axis). Compile fragility (recompile/BN-buffer) mitigated by reusing the validated wrapper + math-equivalence bit-check on the compile-off control.

### 2. Alternating (derandomized) horizontal flip
**Summary**: Replace `transforms.RandomHorizontalFlip()` with antithetic flip parity — each image flipped on odd visits, un-flipped on even (e.g. `flip iff (epoch+idx)%2`), so over any 2 epochs every image is seen once in each orientation. Same marginal distribution, zero sampling variance. Throughput-free, no compile, no architecture change. See proposals/idea-02.md.

**What it targets**: Per-epoch gradient-estimate variance from i.i.d. flip coin-flips — an orthogonal lever to augmentation *content* (which is saturated across occlusion/mixing/transform). Lower-variance gradients can sharpen the final low-LR-tail minimum where this net's accuracy concentrates.

**Reasoning**: airbench makes this a headline result with an unusually strong, broad claim ("improves every training considered" where flip helps — and flip unambiguously helps on CIFAR-10). It is a sampling/variance-reduction change, NOT a new aug class, so the input-aug saturation finding does not apply. Cleanest possible single-variable experiment.

**Sources**: proposals/idea-02.md; arXiv:2404.00498; antithetic-variates rationale.

**Estimated Effort**: Low — wrap the dataset to expose the sample index + epoch parity to a custom flip transform; per-epoch-parity fallback if persistent-worker index plumbing is awkward.

**Risk Assessment**: Primary risk is magnitude — pure variance reduction on an already-present aug may be ≤0.1pp on a heavily-regularized 150-epoch net and fail the +0.1pp bar even if directionally positive. Implementation subtlety with shuffled persistent workers (index-parity plumbing). No under-anneal risk.

### 3. AdaptiveConcatPool readout head
**Summary**: Replace `MaxPool2d(4) → Linear(512,10)` with `concat(globalAvgPool, globalMaxPool) → Linear(1024,10)` (×SCALE_OUT preserved). Gives the linear head both "feature present anywhere" (max) and "how broadly" (avg) statistics over the 4×4/512 map. Near-throughput-free. See proposals/idea-03.md.

**What it targets**: The global-readout — the one structural component untouched since EXP-001. A genuinely-untried axis is likelier to hold residual signal than re-probing saturated ones.

**Reasoning**: Max and avg pool are materially different statistics on a 16-location map; concatenating strictly dominates either for a downstream linear classifier. Flagged as untried in the EXP-018 brainstorm. Throughput-cheap (no new convs) → no under-anneal risk.

**Sources**: proposals/idea-03.md; fast.ai AdaptiveConcatPool2d; Network-in-Network GAP (Lin 2014); EXP-018 brainstorm.

**Estimated Effort**: Low — swap pool+fc, re-init the wider Linear via the existing kaiming path.

**Risk Assessment**: Primary risk is low ceiling — a thin linear head on an already-good 512-d feature; avg-pool may be redundant with max-pool+BN → small/null (cf. the channel-attention null EXP-019). Changes readout, not function-class depth → less likely than idea-01 to break the ceiling.

## Review

Cross-model (Codex) adversarial review in 01-idea-review.md. Scored verdict: **Idea-01 7/10 impact, 5/10 likelihood (PICK)**; Idea-02 3/10 impact, 6/10 likelihood; Idea-03 4/10 impact, 4/10 likelihood. Top concerns + resolutions:
- **"Not truly a different backbone — it's depth-within-DavidNet"** → Accept the reframe: this is the *minimal depth probe* (the cheapest, best-evidenced structural test) that must be run BEFORE spending a loop on a full pre-activation/deeper backbone. If it ties, the evidence then mandates a wholesale backbone swap. Brainstorm/proposal language updated to "minimal depth probe," not "different backbone."
- **"Feasibility hinges on epoch count more than admitted — a 2nd 256-ch 2-conv block may cost more than compile's +12%; EXP-014's 320-width cell only reached 143 ep"** → Make `num_epochs ≥ 135` a HARD pre-run gate (plan must smoke-measure epochs WITH compile ON before the official cells); if below, abort or shrink to a SINGLE extra conv_bn (not a full 2-conv GatedResidual). This is the dominant failure mode (EXP-005/007/013/016) and the #1 thing the plan must instrument.
- **"Evidence is indirect — EXP-014 capacity-saturation cuts against 'more same-stage capacity' unless depth ≠ width"** → True; the bet rests on depth changing the function class where width did not. Mitigate inferential risk with same-session control + a mandatory confirmation pair (the low-control-draw artifact recurred 4× in EXP-016/017/019/020).
- **"Compile integrity is the main reward-hacking/correctness risk"** → Reuse the EXP-014-validated wrapper verbatim: off-budget warmup before `t_start_training`, RNG isolation, BN-buffer restore, and eval on the UNCOMPILED EMA path; add a math-equivalence bit-check (compile-off vs compile-on) on the control at step 2.

## Idea Evaluation

Adopt the reviewer's pick — Idea-01 — without override. It is the only finalist scored with real upside (7/10) and the only one that attacks the diagnosed limiter (the generalization ceiling) by changing nonlinear capacity rather than sampling variance (Idea-02, upside below noise floor) or readout statistics (Idea-03, weakly connected, prior head-side null EXP-019). Idea-02 (alternating flip) is retained as the natural low-risk follow-up / combination partner if Idea-01 wins; Idea-03 demoted to cheap-cleanup. Full scored critique in 01-idea-review.md.

## Chosen Idea
**Selected**: Compile-funded DEPTH at the proven 8×8 stage (a second ReZero GatedResidual at layer2), run as a disciplined minimal depth probe.

**Why this idea**:
It is the single highest-EV move after 15 nulls: three independent evidence sources converge on depth-at-8×8 (airbench's explicit 95→96 "third conv per block"; EXP-004's lone architectural win at exactly this location/block-type; EXP-005/007 losing to *under-anneal* rather than saturation — a constraint compile directly lifts). It is the only finalist that changes the model's function class, and EXP-014's "ceiling not capacity" verdict has a genuine untested crack here: it tested epochs and WIDTH, never DEPTH. torch.compile is banked, validated, and math-equivalent — it funds the per-step cost that doomed prior depth/capacity attempts. Identity-init (ReZero α=0) makes the deeper net start bit-equivalent to the proven net, de-risking the cold start.

**Hypothesis**:
Adding one identity-initialized ReZero GatedResidual(256) at layer2, with torch.compile keeping `num_epochs ≥ 135` (full anneal preserved), will raise `best_test_acc` to **≥ 96.48** and beat its same-session linear-schedule control by **> 0.1pp, replicated on a confirmation pair** — because the extra nonlinear depth at the proven full-speed stage increases representational capacity along the one axis (function-class depth) that the saturated width/epoch/regularization levers could not reach. Falsifiable: if the deeper net ties the same-session control (≤0.1pp) at ≥135 epochs, depth-within-DavidNet is closed and the evidence mandates a wholesale different backbone next.

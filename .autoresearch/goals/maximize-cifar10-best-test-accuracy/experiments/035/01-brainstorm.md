# Brainstorm EXP-035
**Created**: 2026-08-06

## Web Search & Literature Review

- **On the Selection of Initialization and Activation Function for Deep Neural Networks** (`experiments/035/papers/activation-initialization-edge-of-chaos.md`; https://openreview.net/forum?id=H1lJws05K7)
  ICLR 2019 signal-propagation theory identifies Swish among smooth activations that can carry information deeper than ReLU-like functions when initialization is compatible. Transfer to a shallow, BatchNorm-normalized, 300-second CIFAR run is not established, so throughput and trajectory evidence remain load-bearing.
- **When, Where and Why to Average Weights?** (`knowledge/papers/weight-averaging.md`; https://proceedings.mlr.press/v267/ajroldi25a.html)
  ICML 2025 finds modest generalization and efficiency gains from weight averaging, especially with annealing, but requires deliberate window and non-parameter-state handling. EXP018 supplies stronger local negative evidence against uniform weak-tail averaging while leaving recent-weighted EMA untested.
- **Deep Networks with Stochastic Depth** (`knowledge/papers/stochastic-depth.md`; https://arxiv.org/abs/1603.09382)
  Random residual bypass can regularize and reduce expected training work, but the paper's benefits concentrate in very deep networks; one skipped block is a much larger perturbation in the local nine-block ResNet.
- **PyTorch Channels Last Memory Format** (`knowledge/references/pytorch-channels-last.md`; https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
  Official support covers CUDA convolution and BatchNorm while preserving logical NCHW shapes. Tiny FP32 CIFAR kernels and unsupported slicing/padding paths can erase the benefit, so only fresh end-to-end timing can establish value.

## Experimental History Review

- EXP010 remains the 94.15% frontier: width-2 postactivation ResNet-20, N1/M7 plus p0.5 alpha-1 CutMix through the 80% LR boundary, then weak hard-label cosine refinement. Width, RandAugment, CutMix, and the long-high-LR/short-low-LR curriculum are the only repeatably positive ingredients.
- Valid near misses plateaued at 94.22% (EXP012 preactivation and EXP026 partial Mixup), but both worsened strong-phase fit; adjacent stronger regularization, earlier hard labels, tail LR, momentum reset, decay changes, and uniform SWA all lost accuracy. Preserve the accepted curriculum, ordinary SGD state, decay, and boundary.
- Recruited branches, max aggregation, global optimizer paths, abrupt width changes, BF16, sparse erasing, and now fan-out reparameterization repeatedly caused candidate-only one-class transients. Initial equality or lower short loss is insufficient; any new parameter/state path needs immutable multi-step geometry evidence.
- Systems evidence remains unchanged: backward consumes 75.46% of the counted step, loader/transfer/loss/optimizer overhead is small, and memory is abundant. EXP013 showed batch256 gains only 18.91% image throughput and EXP029 showed even a tiny helper can cost 1.97%; exposure proposals need fresh paired timing and an accuracy mechanism beyond “more steps.”
- Still-untried, non-adjacent gaps include changing the activation family without learned parameters, crop-border semantics without deleting class pixels, a recent-weighted rather than uniform tail model average, classifier-bias symmetry, and execution layout. None has local evidence that guarantees a +0.10 point effect.

## Objective Limiter Diagnosis

The primary limiter is not raw training fit alone: the accepted model reaches healthy 89.73% strong-switch accuracy, then converts a short weak hard-label tail into 94.15%, while interventions that fit harder can worsen NLL/generalization. The current quality gap is therefore representation and terminal generalization under a very short fixed horizon, with little tolerance for extra regularization that suppresses the strong phase. The systems limiter is clearer—Conv/BN backward is 75.46% of step time—but 34 experiments have not established that a few percent more exposure moves accuracy at this frontier. EXP035 should prefer a bounded, branch-free representation or data-quality change; pure layout remains secondary unless it clears a margin-bearing throughput gate.

## Collected Ideas

- **Fixed SiLU at every accepted ReLU site** — Replace only the stem, internal-block, and post-add ReLU functions with non-parametric SiLU while preserving graph widths, BN, residual ordering, initialization, and all training policy. It targets feature quality and gradient continuity without recruiting a new branch; ICLR signal-propagation theory is supportive, but sigmoid-backed kernels may consume enough of the 300-second budget to erase a modest gain.
- **Reflection-padded random crops** — Keep the same four-pixel crop geometry in strong and weak loaders but reflect image borders instead of injecting constant-zero pixels. It targets augmentation fidelity at edge objects without deleting class information or changing CutMix labels; it is mechanically narrow and worker-side, though local/published accuracy evidence is weak and reflect work may expose loader stalls.
- **Recent-weighted weak-tail EMA** — Maintain one exponential shadow of parameters and BN state only after the accepted 80% switch, and evaluate a single preregistered model path without adding validation looks. It targets noisy terminal generalization while avoiding EXP018's backward-biased uniform mean; decay lag, online/EMA attribution, BN compatibility, and per-step copy cost are serious design risks.
- **Biasless final classifier** — Construct the final Linear layer without bias, removing random/trainable class-prior offsets from a balanced ten-class problem at zero recurring matrix cost. This simplification preserves hidden representation and avoids amplified parameter scales, but the 10-value bias is likely too small for a 0.10-point effect and removing it is not initial-function exact.
- **End-to-end channels-last execution** — Convert the accepted initialized model and every 4-D input to channels-last, retaining values and logical shapes. It directly attacks Conv/BN backward and could buy more same-recipe exposure; tiny FP32 kernels, Option-A slicing/padding, evaluation-layout normalization, and a weak exposure-to-accuracy link make it a systems experiment rather than a direct quality hypothesis.
- **Fixed small-slope LeakyReLU** — Replace ReLU with a fixed 0.01 negative slope, preserving positive-path arithmetic and adding no learned parameter or sigmoid. This is cheaper and closer to the accepted function than SiLU while preventing exact dead features, but its literature mechanism is weaker and post-add negative residual outputs alter the canonical nonnegative stage interface.
- **High-survival strong-only stochastic depth** — Bypass only same-width residual branches with a small fixed batchwise probability during the strong phase, keeping transitions and the weak tail complete. It jointly regularizes and shortens backward work, but local identity-oriented interventions already suppressed strong fit and the nine-block graph makes each bypass coarse.
- **Moonshot bounded dual-activation mixture** — Replace each ReLU with a fixed convex combination of ReLU and SiLU using one global, non-learned coefficient chosen before execution. This could preserve much of accepted positive-path behavior while importing smooth negative features, but the coefficient is essentially an unsupported interpolation and doubles activation work; it is inferior to testing a canonical endpoint first.

## Combinations

- **SiLU + channels-last**: channels-last might offset SiLU's extra kernel cost while SiLU supplies the missing accuracy mechanism. The cross could beat either alone only if layout accelerates the actual convolution-dominated step enough to subsidize sigmoid work; it is attribution-heavy and should follow independent evidence.
- **Reflection crop + accepted CutMix**: reflection preserves plausible context at crop borders while CutMix continues to provide labeled regional replacement, potentially improving the quality rather than quantity of strong views. This is stronger than reflection alone only because it retains the locally validated class-bearing occlusion prior; it requires no new combination code beyond the padding mode.
- **EMA + biasless classifier**: removing class-prior bias could make the shadow trajectory less sensitive to early class offsets while EMA stabilizes terminal weights. The bias effect is probably negligible and the cross obscures whether averaging helped, so the components should not be combined initially.

## Candidate Ideas

### Weak-Tail Full-State EMA
**Summary**: At the accepted 80% boundary, create one detached FP32 shadow and update all floating parameters and buffers after every weak-tail SGD step with fixed decay 0.999, copying integer BN counters. Continue online SGD for the full budget but route every post-boundary scheduled evaluation exclusively to EMA, keeping exactly one predeclared model per look and 19 looks total. Full specification: `proposals/idea-02.md`.

**What it targets**: Terminal generalization noise during the roughly 60-second low-LR weak tail, while assigning exponentially little weight to the early adapting states that made EXP018's uniform arithmetic average lag.

**Reasoning**: ICML 2025 supports averaging with annealing, and EXP018 explicitly left EMA untested. This point differs in weighting kernel, per-step cadence, full-state BN treatment, uninterrupted SGD, and fixed evaluation routing. A 0.999 decay has a roughly 693-step half-life and leaves about 0.5% boundary-copy mass after the expected tail, offering a concrete recent-state timescale without a sweep. The main uncertainty is whether averaged BN buffers match averaged parameters; evaluating EMA only also deliberately forfeits online-tail scores, so there is no post-hoc oracle advantage.

**Sources**: `knowledge/papers/weight-averaging.md`; EXP018 report/execution; `proposals/idea-02.md`.

**Estimated Effort**: high due to exact recurrence, non-interference, BN-state, evaluator-routing, and timing proofs.

**Risk Assessment**: Medium-high. EMA can still lag a monotonic tail, full-state BN averaging is approximate, foreach updates may lose exposure, and the fixed model stream could underperform online without any allowed fallback.

### Reflection-Padded Strong and Weak Crops
**Summary**: Add `padding_mode="reflect"` to both existing `RandomCrop(32, padding=4)` transforms while preserving crop geometry/order, flip, N1/M7, CutMix, phase boundary, model, optimizer, schedule, seed, and evaluator. Build a paired source/RNG-state corpus because accepted and candidate input tensors must differ while stochastic crop/flip/RandAugment/CutMix decisions remain aligned. Full specification: `proposals/idea-03.md`.

**What it targets**: View quality and train/test boundary mismatch. Constant padding affects roughly 13.4% of a uniformly cropped output in expectation and exposes a fixed negative-color band after normalization; reflection preserves texture and makes crop displacement less trivially recoverable.

**Reasoning**: This is orthogonal to failed occlusion and mixing policies: it deletes no interior information, changes no label, and preserves the validated CutMix/RandAugment curriculum. It also has no GPU-model cost. However, torchvision's installed PIL reflection path adds NumPy conversion/padding work, so strong/weak loader throughput must pass paired non-rollover gates. Evidence for an accuracy gain is mechanistic rather than empirical, and mirrored object fragments may be an equally artificial prior.

**Sources**: torchvision `RandomCrop` documentation and installed source; original ResNet crop geometry; EXP005/006/011/026/027/033; `proposals/idea-03.md`.

**Estimated Effort**: medium; production is two keywords but paired stochastic-data and loader-throughput evidence are nontrivial.

**Risk Assessment**: Medium. The likely effect is near the ten-image threshold, reflection changes about one eighth of most crops, and the PIL/NumPy path may create host stalls despite current prefetch headroom.

### Fixed SiLU Throughout ResNet-20
**Summary**: Replace the three source-level `F.relu` calls—19 dynamic sites across the stem and nine blocks—with parameter-free `F.silu`, preserving accepted Kaiming initialization, residual ordering, shapes, optimizer, data, schedule, and evaluator. Require identical initial state/RNG, immutable strong/weak trajectory safety, and seven fresh paired timing trials; production needs at least 98% accepted exposure. Full specification: `proposals/idea-01.md`.

**What it targets**: The representation/generalization limiter under the short strong phase: smooth signed activation may preserve weak localized evidence and gradients that hard ReLU gates discard, without recruiting a learned branch or changing parameter scale.

**Reasoning**: ICLR 2019 signal-propagation theory gives Swish a concrete mechanism beyond novelty, and no activation-family experiment has been executed locally. It is distinct from preactivation/zero-gamma because every residual branch remains active and BN/order stay accepted. The transfer gap is large: this model is shallow and BN-normalized, no diagnostic identifies dead ReLUs, signed final features may cancel under GAP, and SiLU can cost hundreds of fixed-budget updates. A valid miss retires all-site beta-1 SiLU rather than inviting site/beta/gain tuning.

**Sources**: `experiments/035/papers/activation-initialization-edge-of-chaos.md`; EXP012, EXP015, EXP029, EXP034; `proposals/idea-01.md`.

**Estimated Effort**: high, driven by trajectory and full-step timing evidence despite a three-line production diff.

**Risk Assessment**: High. Strong-phase fit may regress, accepted ReLU sparsity may be beneficial, fixed initialization is not theoretically optimal for SiLU, or the sigmoid-backed backward may fail the 2% timing gate.

## Review

Claude's independent review (`01-idea-review.md`) selected **Fixed SiLU Throughout ResNet-20**, scoring evidence/reasoning 6.5/10 and potential impact 8/10. It judged the apparent EXP012/015 analogy weaker than it first appears because SiLU preserves residual ordering, scale, initialization, and branch activity; the real risks are signed pooled-feature cancellation, loss of useful sparsity, and sigmoid-backed throughput. It downgraded EMA because EXP002's 0.01 best/final gap and EXP018's online-over-average result undermine the terminal-noise premise, and downgraded reflection because its predicted effect sits on the ten-image noise-scale gate.

I adopt the selection. The proposal already records sitewise sign/RMS statistics, pooled-feature sign balance, margins, and switch-fit diagnostics; these remain load-bearing for the reviewer's signed-cancellation and strong-fit concerns. I also retain the strict <=2% weighted timing gate rather than assuming a three-line activation change is free. I do not add a SiLU-specific gain: EXP034 showed that a plausible initialization reparameterization can destabilize relative SGD geometry, and changing both activation and initialization would destroy attribution. The literature supplies a mechanism, not an effect size, so no short-loss benefit or bare threshold pass will be overstated.

## Idea Evaluation

- **Fixed SiLU throughout ResNet-20** — Advance. It is the highest-ceiling genuinely untried representation lever, has a concrete smooth signed-propagation mechanism, and does not repeat the branch-scale/order failures behind prior global architecture changes.
- **Weak-tail full-state EMA** — Defer. It is cleanly differentiated from uniform SWA and protects the strong phase, but the accepted tail is nearly monotonic and three local tail-state interventions have already regressed.
- **Reflection-padded crops** — Defer. It is a narrow, orthogonal boundary-distribution test with low implementation risk, but direct evidence and expected margin are too weak relative to SiLU.

## Chosen Idea
**Selected**: Fixed SiLU Throughout ResNet-20

**Why this idea**:
It changes representation without adding parameters, recruited paths, optimizer-state mechanisms, or data-policy strength, and its main risks are measurable before scoring. Unlike preactivation and zero-gamma, it preserves the accepted postactivation residual topology and active branches; unlike EMA, it targets the diagnosed representation limiter rather than a terminal tail with little measured best/final gap. The all-site canonical endpoint is also cleaner than an unsupported activation mixture or slope/site sweep.

**Hypothesis**:
Replacing all 19 dynamic ReLU operations with fixed beta-1 SiLU while preserving accepted initialization and every training/evaluation setting will pass immutable-corpus class/output/update gates, retain at least 98% of accepted optimizer exposure, maintain at least 89.0% accuracy at the strong-to-weak switch, and raise seed-42 `best_test_acc` from the moving 94.15% baseline to at least 94.25%. A safety/timing veto or one valid sub-threshold run retires this exact all-site SiLU point without beta, site, gain, LR, or kernel rescue.

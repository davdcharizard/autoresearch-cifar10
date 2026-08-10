# Brainstorm EXP-014
**Created**: 2026-08-06

## Web Search & Literature Review

- **PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions** (`papers/poly-loss.md`; `knowledge/papers/polyloss.md`; https://openreview.net/forum?id=gSdSJoenupI)
  Poly-1 is a one-parameter, low-overhead confidence-dependent modification of cross-entropy, but its coefficient is task-dependent. This stack requires a coherent soft-target definition for CutMix and a single accuracy-blind inflation bound.
- **When, Where and Why to Average Weights?** (`papers/when-where-why-average.md`; https://proceedings.mlr.press/v267/ajroldi25a.html)
  Broad ICML 2025 evidence supports checkpoint averaging at minimal state cost and finds averaging complementary to learning-rate annealing. It does not establish that uniform averaging or a different horizon beats EXP-011's already successful charged-time EMA.
- **Group Equivariant Convolutional Networks** (`papers/group-equivariant-convolutions.md`; https://proceedings.mlr.press/v48/cohenc16.html)
  Discrete group convolutions can increase effective capacity through symmetry-aware weight sharing and achieved strong CIFAR-10 results. Horizontal-reflection symmetry is more defensible here than rotations, but a one-file group-convolution implementation is a high-risk architecture moonshot.
- **Existing goal knowledge** (`knowledge/README.md`)
  Wide/pyramidal residual evidence keeps capacity allocation open; SWA requires trajectory diversity and BatchNorm handling; label smoothing can overlap mixed-sample targets; prior SE and augmentation evidence warns that extra kernels or views can erase optimizer exposure.

## Experimental History Review

- The successful lineage is BASE `91.51` -> time-aware WRN EXP-001 `94.62` -> CutMix EXP-002 `95.23` -> clean period-two SAM EXP-004 `95.40` -> full-state charged-time EMA EXP-011 `95.61`. The stable EXP-011 final-16 EMA mean is `95.493125`, so a credible child should target roughly `0.25-0.30` points of plateau lift rather than a selected-max micro-shift.
- EXP-011 retains `25,798` steps at only `1,222.4 MiB` on a `97,871 MiB` H20. Memory is effectively unused; ordinary/SAM steps cost about `10/20 ms`, so extra full forwards are expensive while wider fused convolutions or loss/state operations remain plausible (`02-system-understanding.md`).
- EXP-012's full-probability complementary Cutout reached `95.52` with a `95.418125` tail and missed its dose floor. EXP-013's fixed-scale-40 cosine classifier achieved full preregistered dose but fell to `95.11` with a `95.073750` tail. Do not repeat spatial erasure or fixed-scale cosine geometry from this base.
- EXP-010 moved one equal-MAC block late and gained `9.3%` more steps but no accuracy, rejecting that exact `1-2-3` depth package rather than width generally. WRN-16-5 was previously proposed but never run; architecture initialization and inherited optimizer settings must be treated package-wise.
- Base-independent failures reject identity-halving overlap batches, low-dose substitution of validated CutMix, uncalibrated ASAM, dual-view CPU augmentation, and multi-launch FP32 channel gates. Untried gaps include uniform width expansion, bounded soft-target loss shaping, materially distinct trajectory averaging, cheap multi-head output diversity, and symmetry-aware representations.
- The limiting question is not throughput or memory but whether a new mechanism raises class-boundary generalization across a stable EMA tail. Every finalist must preserve physical GPU 0, one evaluation per epoch, seed-42 protocol, `train.py`-only scope, and one fixed recipe with parent-relative feasibility gating.

## Collected Ideas

- **Calibrated WRN-16-5 width expansion** - Increase stage widths from `64/128/256` to `80/160/320` while retaining the six-block topology and every EXP-011 training mechanism. This attacks representational capacity using extreme memory headroom and a proven CIFAR width axis, but dense convolution work rises about 1.55x, inherited LR/decay may not transfer, and reduced optimizer/data exposure could dominate.
- **Final-stage-only width expansion** - Keep both early stages intact and widen only the `8x8` stage and classifier, spending parameters where activation traffic is cheapest. This is distinct from EXP-010 because it retains all early blocks, but the `128->320` transition may bottleneck the added capacity and the intervention lacks the clean uniform-width prior.
- **Bounded soft-target Poly-1** - Replace hard and CutMix cross-entropy with one coherent Poly-1 objective, using a single coefficient derived from a preregistered maximum constituent-gradient inflation. It targets output-loss weighting at negligible compute and preserves all exposure; task-dependent coefficient transfer, overlap with CutMix, and degeneration toward an effective LR rescale in the confident tail are the main risks.
- **Uniform clean-tail full-state SWA** - Reuse cadence-31 full-state sampling but uniformly average the clean/SAM tail rather than exponentially downweighting older states. This tests a materially different trajectory summary supported by averaging literature, with negligible added compute. The strongly annealed short tail may lack enough trajectory diversity, and uniform inclusion of early-tail states may increase stale bias.
- **Dual shared-backbone affine heads** - Add a second independently initialized linear head, train both against identical hard/CutMix targets, and average logits at evaluation. The extra state and classifier MACs are tiny and head diversity could reduce boundary variance, but a shared representation may drive the heads to near-identical solutions and changing construction RNG must be isolated.
- **Earlier drop-path decay simplification** - Begin removing stochastic depth before the 75% clean boundary while leaving CutMix, SAM, and EMA untouched. This tests whether the mature model is over-regularized and simplifies rather than adds a mechanism, but the lineage supports strong early regularization and supplies no evidence for a new decay onset.
- **Gradient centralization before SGD and SAM** - Center convolutional and classifier gradients per output channel before SAM norm construction and the Nesterov update. It changes optimization geometry without another forward and may regularize features, but adds reduction kernels, changes the validated SAM perturbation, and needs careful all-parameter semantics.
- **Reflection-equivariant residual stem moonshot** - Lift the stem into a two-orientation horizontal-reflection group and preserve/tie orientation-aware filters through at least the first stage before pooling the group axis. It imports a sample-efficient symmetry prior rather than another augmentation view, but is a large custom-kernel architecture change with high initialization, throughput, and CutMix-interaction risk.
- **Late SWA with a non-collapsing LR floor** - Combine uniform tail averaging with a preregistered higher final LR so late checkpoints remain diverse. The combination is more faithful to classic SWA than averaging a collapsing trajectory, but it changes two coupled mechanisms and risks sacrificing the parent's effective clean-tail convergence.

## Combinations

- **WRN-16-5 + Poly-1**: added representation capacity could create a larger train/test gap for Poly-1 to regularize, while the loss adds negligible cost beyond the expensive wider forward. The cross could beat either alone if width is capacity-positive but mildly overfits, yet it confounds the first clean width test and should be sequential rather than bundled.
- **Uniform SWA + non-collapsing LR floor**: maintaining late trajectory diversity gives uniform averaging signal that pure cosine annealing may remove, making the pair mechanistically stronger than either averaging nearly identical states or raising the LR without averaging. It is nevertheless a two-coefficient schedule package with weak direct evidence under the 75-second SAM tail.
- **Dual heads + Poly-1**: head disagreement could preserve multiple boundaries while Poly-1 reweights confident residual errors for both heads. This may offer cheap ensemble-like diversity, but the shared backbone and coupled loss could collapse both heads, and two simultaneous logit interventions would be hard to diagnose.
- **Final-stage width + reflection-equivariant stem**: symmetry-aware early features and cheap late capacity attack sample efficiency and semantics at opposite ends. The combination has larger upside than either narrow change but is too complex and throughput-uncertain for a single isolated experiment.

## Candidate Ideas

### Calibrated Stage-3 Width-5 Expansion
**Summary**: Widen only the final `8x8` stage from 256 to 320 channels, yielding a `64/128/320` six-block taper with `3,827,290` parameters and `461,556,864` MACs/image. Preserve all early/middle blocks and the complete EXP-011 CutMix/SAM/EMA recipe. This is one fixed architecture, not a width sweep.

**What it targets**: Representation capacity at the stable-generalization bottleneck while spending added compute in the low-resolution stage where H20 shape efficiency is favorable (`02-system-understanding.md`; EXP-010).

**Reasoning**: Memory is effectively unused, EXP-001 established width/capacity as the dominant historical gain, and EXP-010's exact failure removed early processing rather than testing added late capacity. The taper adds 39.2% parameters for only 17.6% MACs, keeps both early and middle blocks, and implements by changing existing dense tensor shapes rather than adding launches. Its preflight must accept real exposure loss but require `<=1.15x` median latency, at least 22,000 projected steps, 130 EMA samples, and full architecture/SAM/EMA integrity.

**Sources**: `proposals/idea-01.md`; `knowledge/papers/deep-pyramidal-residual-networks.md`; EXP-001, EXP-010, EXP-011, and EXP-013 reports.

**Estimated Effort**: low code change, medium-high verification.

**Risk Assessment**: The parent may already be capacity-sufficient; fewer updates, images, EMA samples, and evaluation opportunities may outweigh width. Fixed LR/decay and global SAM rho may not transfer, 320-channel kernels may miss projected latency, and shape-dependent initialization makes the result package-level.

### Uniform Full-State Clean-Tail SWA
**Summary**: Replace EXP-011's exponential decay kernel with an arithmetic mean over the identical cadence-31 post-update clean/SAM states. Parameters and floating BatchNorm buffers use cumulative `1/n` averaging, integer buffers copy latest state, and evaluation retains the parent's one-source swap/restore semantics.

**What it targets**: The stable late-checkpoint plateau by centering the full clean/SAM trajectory more broadly without another forward, new coefficient, or changed online training path.

**Reasoning**: Modern ICML evidence supports averaging together with LR annealing, and classic SWA motivates uniform checkpoint weighting. Relative to EXP-011's truncated exponential kernel, uniform weighting raises effective sample count and materially increases early-tail influence while preserving the same support, cadence parity, full-state BN convention, and near-zero overhead. Exact online parity, arithmetic-mean, BN, swap, and dose audits make the comparison clean.

**Sources**: `proposals/idea-03.md`; `papers/when-where-why-average.md`; `knowledge/papers/stochastic-weight-averaging.md`; `knowledge/papers/how-to-scale-your-ema.md`; EXP-011.

**Estimated Effort**: medium code/audit change, medium verification.

**Risk Assessment**: Uniform weighting may amplify stale higher-LR early-tail states, while cosine annealing may leave too little trajectory diversity for SWA. Averaged BN buffers are not recalibrated population statistics, and the likely effect may sit inside known single-run noise even with perfect dose and integrity.

### Bounded Soft-Target Poly-1
**Summary**: Replace every training CE call with `CE(q,p) + 0.25*(1-q dot p)` using the same hard or area-corrected CutMix targets. The coefficient comes from a fixed maximum 1.25 constituent-gradient multiplier and is used identically in ordinary training and both SAM passes; model, optimizer, data, EMA, and evaluation remain unchanged.

**What it targets**: Confidence-dependent loss geometry on every update without sacrificing the optimizer/data exposure that constrains fixed-time descendants (`02-system-understanding.md`).

**Reasoning**: PolyLoss reports broad image-classification gains from a one-line objective, and EXP-011's remaining gap may be decision-boundary rather than capacity limited. The proposal derives exact hard and CutMix gradients, acknowledges that the combined soft-gradient ratio is not bounded, and requires sparse analytic audits plus a single `<=1.01x` latency gate. It is mechanism-distinct from Cutout and cosine normalization and should retain at least 25,300 steps and 155 EMA samples.

**Sources**: `proposals/idea-02.md`; `papers/poly-loss.md`; `knowledge/papers/polyloss.md`; EXP-011 through EXP-013 reports.

**Estimated Effort**: medium code/audit change, medium verification.

**Risk Assessment**: Epsilon 0.25 is optimizer-bounded rather than accuracy-calibrated and may be too weak for a 0.25-point plateau lift. On unequal CutMix targets Poly-1 sharpens toward the majority label, potentially counteracting validated softening; in the confident clean tail it may resemble an effective LR increase. Any outcome is the full loss + CutMix + SAM + EMA package.

## Review

Claude Opus scored the stage-3 taper `7/10` for evidence and `8/10` for impact, ahead of uniform SWA (`5/10`, `5/10`) and bounded Poly-1 (`5/10`, `4/10`). Its arithmetic checks passed. The material concerns adopted are the unverified capacity premise, lower evaluation count, weaker relative SAM perturbation under a larger parameter norm, and an initially too-low scientific tail bar. The proposal now reports train-fit only as a non-causal diagnostic, audits `||epsilon||/||w||` without retuning rho, reports the max-minus-tail premium, and requires final-16 mean `>=95.69%` for mechanism support.

Claude suggested switching to width 288 if width 320 failed the latency gate. That advice is not adopted: an accuracy-blind conditional resize would turn a single isolated package into a timing-selected architecture sweep. The first valid width-320 gate remains decisive and a failure will be recorded. Full review: `01-idea-review.md`.

## Idea Evaluation

The Claude pick is adopted. Uniform SWA exposed an interesting copy-initialized EMA anchor but its proposal miscomputed the parent kernel and replaced validated BN/state weighting; it needs a newly developed bias-correction or hybrid design. Positive Poly-1 has exact calculus and excellent dose preservation but lacks coefficient effect-size evidence, upweights confident examples, and partially sharpens away CutMix soft targets. The stage-3 taper is riskier on exposure but is the only finalist with plausible upside commensurate with the diagnosed `0.25-0.30` plateau lift.

## Chosen Idea
**Selected**: Calibrated Stage-3 Width-5 Expansion

**Why this idea**:
Widening only the `8x8` stage from 256 to 320 channels adds `39.2%` parameters for `17.6%` MAC growth, preserves all validated early/middle processing, and uses existing dense operations rather than launch-heavy modules. EXP-010 explicitly left this direction open after its equal-MAC block relocation failed. The bet is package-level and capacity limitation is not assumed proven; its value is a high-ceiling representation test with decisive latency/dose gates and informative diagnostics.

**Hypothesis**:
On one fixed-seed physical-GPU-0 run, the `64/128/320` taper will pass the first parent-relative preflight at median latency ratio `<=1.15`, realize at least `22,000` optimizer steps and 130 balanced EMA samples, and reach `best_test_acc >=95.71%`. A final-16 EMA mean `>=95.69%` will support a stable representation gain; a formal pass below that mean is max-selected improvement only. Any valid preflight failure or sub-threshold run is recorded without trying another width, LR, SAM radius, or seed.

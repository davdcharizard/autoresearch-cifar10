# Brainstorm EXP-016
**Created**: 2026-08-06

## Web Search & Literature Review

- **When, Where and Why to Average Weights?** (`knowledge/papers/when-where-why-average.md`)
  Averaging complements annealing with low overhead, but kernel mass, support, and BatchNorm semantics determine the actual estimator.
- **How to Scale Your EMA** (`knowledge/papers/how-to-scale-your-ema.md`)
  Time/exposure-derived horizons are preferable to copied decay constants; bias correction and horizon changes are distinct interventions.
- **Lookahead Optimizer** (`knowledge/papers/lookahead-optimizer.md`)
  Online slow/fast interpolation has CIFAR evidence without extra forwards. From EXP-004 it avoids nested evaluation EMA, but canonical alpha 0.5 may contract effective progress under the fixed schedule.
- **Stochastic Weight Averaging** (`knowledge/papers/stochastic-weight-averaging.md`)
  Uniform late averaging can improve generalization when trajectory diversity exists, while BN recalibration is incompatible with this frozen uncharged-data protocol.

## Experimental History Review

- EXP-004 is the 95.40 base: period-two rho-0.05 SAM in the clean final quarter added 0.17 points with 25,560 steps and final-equals-best accuracy. Its local child threshold is 95.50.
- EXP-011 successfully added cadence-31 full-state EMA, reaching 95.61 with a 95.493 tail mean and negligible overhead. Later review found its copy-in first sample retains about 6.3% terminal mass; this is a measured estimator artifact, not a failure of averaging.
- EXP-004's failed direct children reject half-overlap self-distillation, low-dose manifold substitution, uncalibrated 9x-norm ASAM, and CPU-heavy paired RandAugment. Do not repeat them.
- Four children then failed from EXP-011 across complementary erasure, fixed cosine geometry, width-320 latency, and negative Poly-1. This fork should pursue a materially different trajectory/state or representation mechanism, not another narrow output scalar.
- The limiter remains stable generalization under a fixed forward budget. Memory is abundant; extra forwards and loader work are expensive, while sparse state arithmetic is cheap (`02-system-understanding.md`).

## Collected Ideas

- **Activation-anchored bias-corrected EMA** - Add EXP-011's cadence-31, 18.75-second full-state EMA directly to EXP-004 but normalize mass from the exact 225-second activation boundary, eliminating the 6.3% copy-in anchor. This retains the validated support while testing a mathematically corrected estimator.
- **Uniform full-state clean-tail SWA** - Average the same cadence-31 post-SAM/ordinary states uniformly, copying integer buffers latest. It removes coefficient tuning and uses the entire tail, but early higher-LR states and averaged BN buffers may be too stale.
- **Canonical Lookahead without evaluation EMA** - Wrap Nesterov with parameter-only slow weights from step one while retaining EXP-004 evaluation. This targets online variance without nested EMA, but alpha 0.5 may halve five-step displacement and interact with BN/SAM.
- **Late Lookahead only** - Activate slow/fast interpolation with the clean SAM phase, concentrating variance reduction where generalization is shaped. It avoids early progress contraction but introduces an unvalidated phase choice.
- **EMA with latest BN buffers** - Average parameters while copying current BN buffers at evaluation, targeting parameter trajectory without averaging approximate statistics. It differs from the successful full-state package and risks parameter/BN mismatch.
- **Classifier-row affine decorrelation** - Add a cheap Gram penalty without cosine normalization. It targets class separation at negligible backbone cost, but parent row geometry and coefficient evidence are absent.
- **Reflection-equivariant stem moonshot** - Share early kernels across original/reflected features and fuse before the middle stage. It exploits a true CIFAR symmetry but changes activation traffic and augmentation semantics and requires a strict latency gate.

## Combinations

- **Bias-corrected EMA + longer horizon**: correction removes activation-anchor bias and could make a 30-second horizon safer, but the new half-life is an uncalibrated second intervention; isolate correction first.
- **Lookahead + uniform tail SWA**: online variance reduction plus post-hoc averaging may stabilize two scales, but repeats the nested-smoothing concern that motivated forking from EXP-011.
- **Affine decorrelation + corrected EMA**: boundary geometry plus stable trajectory could exceed either alone, but two untested mechanisms prevent useful attribution.

## Candidate Ideas

### Activation-Anchored Bias-Corrected Full-State EMA

**Summary**: Add EXP-011's cadence-31 full-state clean-tail EMA directly to EXP-004, but initialize exponential mass at zero at the exact 225-second activation boundary and normalize each update by accumulated mass. Keep the successful 18.75-second half-life, complete floating-state averaging, latest-copy integer buffers, and one-source EMA evaluation. This isolates removal of EXP-011's unintended 6.30% terminal point mass on its first sampled state (`proposals/idea-01.md`).

**Reasoning**: EXP-011 is the only direct child of EXP-004 that produced a clear gain, reaching 95.61 at essentially unchanged throughput. The corrected recurrence retains that validated support and cost profile while increasing estimated effective sample size from about 79 to 101, moving mean state age from about 25.1 to 21.8 seconds, and eliminating a stale high-learning-rate anchor. This is the strongest lineage-grounded route to clearing EXP-004's local 95.50 threshold, though the anchor may itself have supplied useful regularization.

**Sources**: `knowledge/papers/how-to-scale-your-ema.md`; `knowledge/papers/when-where-why-average.md`; `experiments/011/04-analysis.md`; `experiments/016/proposals/idea-01.md`.

**Estimated Effort**: Medium. Reuse the proven full-state swap/evaluation pattern, add exact time-normalized recurrence and kernel audits, then run the required paired GPU preflight and single metric launch.

**Risk Assessment**: Low implementation risk, medium scientific risk, modest likely effect. It may reproduce EXP-011 rather than improve it, and a local pass from 95.50-95.60 would remain below the current global best.

### Uniform Full-State Clean-Tail SWA

**Summary**: From EXP-004, uniformly average every cadence-31 post-update state throughout the clean final quarter, using one cumulative arithmetic mean for parameters and persistent floating buffers and latest-copy integer buffers. Evaluate only the averaged state after activation, with no BN recalibration or EMA (`proposals/idea-02.md`).

**Reasoning**: Uniform averaging removes both decay tuning and EXP-011's first-state point mass, gives every one of roughly 160 samples equal weight, and doubles effective sample size relative to the implemented EXP-011 kernel. Odd cadence balances ordinary and SAM-derived states. However, its mean state age would be about 37.4 seconds, substantially older than EXP-011, so early higher-learning-rate tail states and full-state BN mismatch are plausible failure modes under the strongly annealed schedule.

**Sources**: `knowledge/papers/stochastic-weight-averaging.md`; `knowledge/papers/when-where-why-average.md`; `experiments/011/04-analysis.md`; `experiments/016/proposals/idea-02.md`.

**Estimated Effort**: Medium. The cumulative recurrence is simple, but full-state inventory, restoration, arithmetic-reference, timing, and trajectory audits remain necessary.

**Risk Assessment**: Low implementation risk, medium-high scientific risk, modest-to-medium upside. Classical SWA evidence does not directly cover this short cosine/SAM tail, and stale states could erase averaging gains.

### CIFAR-Grounded High-Alpha Lookahead Without Evaluation EMA

**Summary**: Wrap EXP-004's Nesterov optimizer with parameter-only Lookahead from step one using fixed `k=5`, `alpha=0.8`, retained momentum, and slow-only evaluation. Every fifth update interpolates slow parameters 80% toward the fast endpoint and copies them back into the optimizer-owned parameter objects. Keep current BN buffers and add no EMA (`proposals/idea-03.md`).

**Reasoning**: Forking from EXP-004 avoids nested evaluation smoothing and lets Lookahead alter the future optimization trajectory. Alpha 0.8 is reported in the original CIFAR experiments and retains substantially more five-step displacement than canonical alpha 0.5; foreach operations should make it inexpensive. The main weakness is diagnostic fit: EXP-004 ended at its best accuracy and does not demonstrate excessive endpoint variance, while retained momentum, current BN buffers, and repeated 20% chord contraction can under-travel or destabilize the Nesterov/SAM path.

**Sources**: `knowledge/papers/lookahead-optimizer.md`; `experiments/004/04-analysis.md`; `experiments/016/proposals/idea-03.md`.

**Estimated Effort**: Medium-high. Parameter/optimizer identity, momentum, SAM ordering, BN semantics, slow-only restoration, recurrence, and charged foreach overhead all need dedicated audits.

**Risk Assessment**: Medium implementation risk, high scientific risk, medium upside. It is a broader trajectory intervention than post-hoc averaging but has weaker local evidence and may reduce effective optimization progress.

## Review

Claude Opus completed the required adversarial review (`01-idea-review.md`) and selected Uniform Full-State Clean-Tail SWA, conditioned on replacing the full-quarter cumulative kernel with a preregistered narrower boxcar. Its central criticism was that full-quarter uniform averaging would raise ESS from roughly 79 to 160 while moving mean state age from 25.1 to 37.4 seconds, conflating kernel shape with substantially greater staleness under a decaying cosine schedule. I accept that criticism.

The selected refinement is a fixed **106-state trailing uniform window** at the unchanged cadence 31. EXP-011's realized 159 intervals spanned 74.7736 seconds, or 0.470274 seconds per cadence sample; a 106-state suffix therefore has predicted mean age `(106-1)/2 * 0.470274 = 24.6894` seconds and ESS exactly 106. This preserves the validated implemented EMA kernel's approximate 25.13-second center while removing its 6.30% first-state point mass and modestly increasing ESS from about 79.2. The fixed count is selected entirely from historical timings before candidate preflight or accuracy and will not change after measurement.

I did not adopt the suggestion to pair corrected EMA with a shorter half-life because it combines de-biasing with a new decay scale and weakens attribution. I also did not rebase Lookahead onto EXP-011 because nested online and evaluation smoothing would introduce a broader two-scale package than this experiment needs. The review's warning that max accuracy benefits from checkpoint variation is retained: report final-16 mean and best-minus-tail premium separately, and do not equate a local pass with a new global best.

## Idea Evaluation

Claude scored corrected EMA at 7/10 evidence and 3/10 impact, full-quarter uniform SWA at 5/10 evidence and 6/10 impact, and high-alpha Lookahead at 4/10 evidence and 4/10 impact. Corrected EMA was judged too close to a known package to resolve its specific effect in one run; Lookahead was judged mechanism-mismatched at alpha 0.8 and handicapped by discarding the validated EMA gain. Uniform SWA won because it changes an untested averaging-kernel axis with low charged cost and gives an informative comparison, provided its stale full-quarter support is removed. The age-limited 106-state refinement directly addresses the decisive weakness without adding an accuracy-selected parameter.

## Chosen Idea
**Selected**: 106-State Trailing Uniform Full-State Clean-Tail SWA

**Why this idea**:
It retains EXP-004's validated training trajectory and the cheap sparse state-averaging class proven by EXP-011, while testing uniform kernel shape without shifting the estimator materially earlier in the tail. A fixed 106-state window removes the anomalous first-state anchor, targets ESS 106, and predicts a 24.69-second mean age. It is more discriminating than corrected EMA and better aligned with the measured limiter than high-alpha Lookahead.

**Hypothesis**:
At cadence 31, a trailing uniform mean of the latest 106 complete post-update states will balance ordinary and SAM-derived iterates, preserve a roughly 25-second state-age center, and reduce dependence on the first sampled clean-tail state. With unchanged optimizer exposure and one averaged-state evaluation per epoch, it will reach `best_test_acc >= 95.50%` from EXP-004; a stronger outcome reaches at least the 95.61% global-best level, while the final-16 averaged-state mean will be reported to distinguish a stable gain from max-selection premium.

# Experiment Learnings: Maximize CIFAR-10 Best Test Accuracy

<!-- Per-goal qualitative distillation, agent-appended during the analyze phase.
     Three categories below. Each bullet follows the
     3-line, budget-strict format:

       - **{statement — ≤150 chars}** ({source refs})
         Evidence: {1-2 lines, MUST cite a source path — report / log / JSON / URL}
         Implication: {1-2 lines}

     HARD CEILING: if a bullet's total length exceeds ~3-4 sentences after composition, REWRITE it
     shorter. Push any detail that does not fit into the experiment report and cite the report
     path from Evidence. Do NOT commit an over-budget bullet. The Failed Approaches section
     uses Mechanism/Insight in place of Evidence/Implication — same budget shape. -->

## Protocol Findings

<!-- Measurement, verification, or benchmarking discoveries specific to this goal's metric.
     Examples:
       - "Stage-1 measurements are contaminated by CUDA autotune warmup — measure warm."
       - "Stored baselines expire same-day; require a same-day pristine reference run."

     Good format (fits budget length, cites report):
       - **Baseline pLDDT drifts ~0.3/day; require same-day pristine reference** (EXP-006)
         Evidence: goals/{slug}/experiments/006/04-analysis.md § Verification Results — 8-case drift table
         Implication: capture pristine baseline same-day as experiment measurement. -->

### High Importance

- **Persistent workers plus bounded evaluation keep fixed-budget runs below the wall limit** (EXP-001)
  Evidence: `goals/maximize-cifar10-best-test-accuracy/experiments/001/04-analysis.md` § Results — 300.0s training, 321.7s total.
  Implication: retain persistent training workers and a small pre-registered evaluation schedule.

- **Strong worker-side augmentation can preserve synchronized optimizer exposure** (EXP-004)
  Evidence: `goals/maximize-cifar10-best-test-accuracy/experiments/004/04-analysis.md` § Results — retained 38,358 steps and finished in 340.7s.
  Implication: preflight host throughput and keep augmentation outside the timed GPU step when possible.

### Medium Importance

- **Each fixed-evaluator pass costs about 17.3s on the current node** (EXP-001)
  Evidence: `goals/maximize-cifar10-best-test-accuracy/experiments/001/02-plan.md` § Configuration Changes — measured planning diagnostic.
  Implication: budget evaluation count explicitly; per-epoch validation can consume the 600s wall limit.

- **Nominally cheap per-step regularizers can materially reduce fixed-budget exposure** (EXP-003, EXP-029)
  Evidence: EXP003 label smoothing lost 6.7% of steps; EXP029's 38 GC operations added 1.97% despite an eleven-line production diff.
  Implication: require fresh paired full-step timing even for small loss/gradient helpers.

- **Ratio safety gates need denominator-safe definitions and accepted-control qualification** (EXP-035)
  Evidence: `experiments/035/04-analysis.md` — zero BN biases produced `9e28` ratios, while controls crossed the 5x site-gradient gate.
  Implication: freeze absolute/relative handling and require controls to pass before running or vetoing a candidate.

### Low Importance

- **Long-horizon generic divergence can lose mechanism identifiability** (EXP-037)
  Evidence: `experiments/037/04-analysis.md` — accepted controls diverged `0.65-0.79` by step 64, leaving stem centering only `1.48-1.61x` above noise.
  Implication: prefer effect-specific survival statistics before chaotic trajectory separation dominates whole-feature or logit comparisons.

- **Rollover-inclusive loader p95 is not a candidate-specific starvation gate** (EXP-033)
  Evidence: `experiments/033/04-analysis.md` — candidate/control p95 waits were 45.014/38.830ms despite 159/179 batches/s and clean workers.
  Implication: compare paired tails or exclude iterator rollover bursts; do not gate against an absolute 1.5ms p95 that the control also violates.

- **Fresh forkserver processes do not replay post-transform batches from seed alone** (EXP-019, EXP-021, EXP-026)
  Evidence: EXP-019 diverged; EXP-026's exact pre-policy corpus made shared branches bitwise equal and safely resolved the same Mixup policy.
  Implication: persist exact augmented tensors and serialize evidence before assertions in paired data-policy gates.

- **Amortize one-time loader transitions over the production horizon in short probes** (EXP-026)
  Evidence: `experiments/026/04-analysis.md` § Execution — charging a 3s rebuild to 13.7s probe work falsely made wall/count 1.25-1.27.
  Implication: separate per-step waits from one-time transitions before projecting full-run wall overhead.

- **Safety probes must use production-distribution inputs** (EXP-015)
  Evidence: `experiments/015/03-execute.md` — Gaussian pixels falsely tripped class concentration while real N1/M7 hard/soft checks passed.
  Implication: use synthetic tensors for structure, but evaluate optimization-collapse gates on materialized production batches.

- **Fresh paired batch timing can overturn a serial exposure estimate** (EXP-013)
  Evidence: `experiments/013/00-paired-timing.md` measured 1.189x image throughput versus the serial probe's 1.284x.
  Implication: gate batch changes with alternating fresh-process pairs, not serial point measurements.

- **Dense final-tail evaluation found only a 0.01-point best-versus-final gap** (EXP-002)
  Evidence: `goals/maximize-cifar10-best-test-accuracy/experiments/002/04-analysis.md` § Results — best 91.83%, final 91.82%.
  Implication: late per-epoch evaluation is reliable but can be reduced if wall-time pressure increases.

- **Paired synthetic timing predicted capacity exposure within 2.5%** (EXP-007, EXP-023)
  Evidence: EXP-007 projected 26,469 versus 27,143 actual steps; EXP-023 projected 23,132 versus 23,465 actual.
  Implication: use paired control/candidate ratios with margin-bearing gates for future capacity changes.

## Failed Approaches

<!-- Approaches tried for this goal's metric that did not work, with a root-cause mechanism.
     Promotion rule: count = 1 → Low; count ≥ 2 → Medium; count ≥ 3 → High. Within each tier,
     sort by count descending (most-hit at top). On promotion to High, add a
     `> **Recurring failure:** ...` callout at the top of this section if not already present.

     Entry format:

       - **{Approach name / root mechanism}** (count: N, EXP-NNN, EXP-MMM)
         Mechanism: {1 line — why the approach fails}
         Insight: {1-2 lines — what alternative approaches might still work} -->

### High Importance

> **Recurring failure:** Global-LR optimizer-path interventions destabilize early class geometry even when first-step or coherent scale is controlled.

- **Global-LR optimizer-path changes repeatedly create class-collapse transients** (count: 3, EXP-020, EXP-022, EXP-028)
  Mechanism: Nesterov amplified step one, Lookahead mismatched velocity/location, and scale-matched PNM produced parity spikes up to 12.35x despite exact coherent scale.
  Insight: Preserve ordinary momentum; immutable-batch output and update-spike gates are mandatory, and constant-direction proofs are insufficient.

### Medium Importance

- **Unlabeled mean-fill occlusion fails as both replacement and composition** (count: 2, EXP-006, EXP-033)
  Mechanism: Every-view 25% Cutout lost 0.67 points; a sparse 1.48% Random Erasing composition triggered candidate-only concentration and 8.93x logit geometry.
  Insight: Retain class-bearing CutMix; any future occlusion idea needs a distinct mechanism, not adjacent p/area/fill tuning.

- **Removing strong-phase regularization before the 80% LR boundary hurts** (count: 2, EXP-005, EXP-027)
  Mechanism: Weak hard views from 75% lost 0.18 points; hard N1/M7 from 70% collapsed fit and lost 0.46 despite full exposure.
  Insight: Keep RandAugment and CutMix together until the simultaneous low-LR hard weak-tail transition.

- **Changing width-2 decay away from all-parameter 1e-4 hurts its short-horizon balance** (count: 2, EXP-008, EXP-009)
  Mechanism: `5e-4` suppressed fit; zeroing BN/bias decay lowered train loss but worsened final NLL and missed baseline.
  Insight: preserve all-parameter `1e-4` and move to a distinct mechanism rather than interpolate this lever.

- **Identity-oriented residual changes suppress the short strong phase** (count: 2, EXP-012, EXP-015)
  Mechanism: full preactivation lowered switch fit 2.85 points; selective zero-gamma lowered it 3.25 despite equal exposure and safe recruitment.
  Insight: preserve accepted residual ordering/initial branch activity; short-fit gains do not establish full-phase representation quality.

- **Pool-first transition shortcuts fail in both learned and deterministic forms** (count: 2, EXP-017, EXP-021)
  Mechanism: learned projection-BN worsened late NLL/top-1; fixed Option-A pooling caused candidate-only one-class transients.
  Insight: retire pool-first shortcuts in this recipe; seek representation changes outside transition downsampling.

### Low Importance

- **RMS-matched cosine logits do not control class-row optimizer geometry** (count: 1, EXP-038)
  Mechanism: Global logits/updates stayed bounded, but normalized classifier rows exceeded `3x` raw-norm dispersion from step4 and peaked at `4.51x`.
  Insight: Retire this fixed-scale/shared-SGD point; future angular heads need prospective intrinsic row-norm control.

- **Stem-only filter mean centering lacks persistent control-relative separation** (count: 1, EXP-037)
  Mechanism: It changed initial pooled/logit features, but after 64 steps its divergence was only `1.48-1.61x` accepted control noise versus the required `5x`.
  Insight: Preserve the accepted stem; any centering revisit needs a stronger distinct scope and an effect-specific survival statistic.

- **Two-phase reflection padding amplifies aligned strong-view logits** (count: 1, EXP-036)
  Mechanism: Exact 13.33% border substitution propagated through strong views and reached 20.72x logits/9.81x gradients despite qualified controls.
  Insight: Preserve constant crop padding; do not rescue with phase, width, or adjacent padding-mode tuning.

- **Conv-only fan-out reparameterization destabilizes early class geometry** (count: 1, EXP-034)
  Mechanism: BN kept initial logits within 0.044%, but smaller norms drove a 13.99% relative stem update, 1.95x whole update, and six late one-class events.
  Insight: Retire subsets/interpolation; future initialization changes need norm/update invariance plus an independent representation hypothesis.

- **Full momentum reset at the 80% objective boundary does not improve the tail** (count: 1, EXP-032)
  Mechanism: It safely halved the first weak update and recovered immediately, but peak/NLL worsened to 93.89%/0.2047 despite 100.52% exposure.
  Insight: Preserve inherited ordinary momentum; partial or conditional resets need new directional evidence, not post-result tuning.

- **Initialization-scaled hard-max residuals do not remain bounded during training** (count: 1, EXP-031)
  Mechanism: A 10%-RMS initialization perturbation drifted to 0.41 aggregate and 4.34 per-example, with candidate-only class collapse and 1.58x updates.
  Insight: Retire fixed hard-max residual pooling; any aggregation revisit needs intrinsic per-example bounds and exact-corpus trajectory gates.

- **Weak-tail start LR 0.02 overfits the hard refinement phase** (count: 1, EXP-030)
  Mechanism: It retained 99.48% exposure and recovered immediately after the switch, but lower train loss coincided with worse NLL, 0.11-point regression, and a 93.90% peak.
  Insight: Preserve the accepted 0.01 tail quench; avoid adjacent LR tuning without a new mechanism for improving weak-tail generalization.

- **Literal all-Conv gradient centralization misses the 99%-exposure gate** (count: 1, EXP-029)
  Mechanism: Nineteen reductions plus subtractions were safe and active but added 1.97% full-step cost, projecting 521 fewer updates.
  Insight: Any GC revisit needs a separately justified fused or sparse-layer formulation; the all-Conv accuracy effect remains unmeasured.

- **Replacing half of accepted CutMix with alpha-0.4 Mixup deepens switch underfit** (count: 1, EXP-026)
  Mechanism: Exposure rose to 27,268 steps, but switch accuracy fell 1.60 points and NLL worsened; the 94.22% peak missed the gate.
  Insight: Retain pure CutMix; any geometry revisit needs a milder split or phase-specific mechanism that preserves strong fit.

- **Global-LR identity-scale ECA saturates its gates during recruitment** (count: 1, EXP-025)
  Mechanism: Three zero-start final-stage gates reached `[0,2]` within 20 steps; late soft block means rose to 1.360 despite finite state.
  Insight: Retire this exact ECA point; any attention revisit needs intrinsically narrower output or independently justified optimizer scaling.

- **Abrupt 32/64/160 late width can create an early class transient** (count: 1, EXP-024)
  Mechanism: On the first CutMix step, the candidate predicted one class for 126/128 samples versus 78/128 control, despite lower terminal loss.
  Insight: Avoid this exact asymmetric width point; prefer capacity changes that preserve the accepted initial function and shortcut ratio.

- **Width-3 ResNet-14 trades too much depth for extra channels** (count: 1, EXP-023)
  Mechanism: The 1.54M-parameter model kept 87.24% of updates and entered the weak tail well, but peaked at 94.00% then regressed.
  Insight: Preserve ResNet-20 depth; test conditional or stage-specific capacity rather than another global width-plus-depth cut.

- **Uniform SWA over the annealed weak tail biases backward from the better online iterate** (count: 1, EXP-018)
  Mechanism: Eight separated snapshots plus 1,624 BN batches still lowered online 94.02% to 93.85% and worsened NLL.
  Insight: Retire this arithmetic window; EMA or constant-LR SWA are distinct hypotheses, not window tweaks.

- **Full-forward BF16 width 3 triggered candidate-only class concentration before timing** (count: 1, EXP-016)
  Mechanism: paired real-batch training crossed >95% one-class predictions while the same-step width-3 FP32 control did not.
  Insight: reject this exact autocast operating point; any FP32-BN or narrower-precision revisit needs a new reviewed hypothesis and safety gate.

- **Raw zero-initialized global-max readout has uncontrolled first-update scale** (count: 1, EXP-014)
  Mechanism: Its first gradient was 4.10x the average classifier's; one update collapsed all predictions to one class and best accuracy to 10.00%.
  Insight: Reject an independent raw-max branch at global LR; any revisit needs normalization or a bounded first-update gate.

- **CutMix probability 0.75 over-regularizes the short strong phase** (count: 1, EXP-011)
  Mechanism: equal exposure still lowered the 80% checkpoint by 2.91 points and best top-1 by 0.15.
  Insight: preserve p=0.5; avoid stronger probability or unsupported interpolation above it.

- **15% LR hold plus cosine/Nesterov likely removes useful high-LR regularization** (count: 1, EXP-001)
  Mechanism: train loss reached 0.0215 while best accuracy was 0.10 points below baseline.
  Insight: preserve a 60-75% high-LR plateau and isolate standard momentum before rejecting time alignment.

- **Built-in label smoothing 0.05 trades lower NLL for unchanged top-1** (count: 1, EXP-003)
  Mechanism: test loss improved 0.2843 to 0.2740, but best accuracy stayed 91.83% while steps fell 6.7%.
  Insight: reject this operating point; only revisit with throughput-equivalent loss or a hard-label refinement tail.

## Patterns

<!-- Goal-specific patterns, mechanisms, and reusable recipes discovered during experiments.
     Examples:
       - "Primary-bucket-only gating by aligning a new gate's threshold with an existing gate's
          threshold keeps long-path behavior bit-identical — composable."
       - "Image-time static override of configs_base.py avoids runtime-mutation side effects."

     Entry format: 3-line statement / evidence / implication (budget-strict). -->

### High Importance
### Medium Importance

- **Long high-LR exploration followed by low-LR refinement improves this fixed-budget model** (EXP-001, EXP-002)
  Evidence: `goals/maximize-cifar10-best-test-accuracy/experiments/002/04-analysis.md` § Results — an 80% hold gained 0.16 points after a 15% hold lost 0.10.
  Implication: preserve roughly four-fifths of the budget at `lr=0.1` before testing added regularization.

- **RandAugment exploration plus a weak hard-label tail raises top-1** (EXP-004)
  Evidence: `goals/maximize-cifar10-best-test-accuracy/experiments/004/04-analysis.md` § Results — 84.60% before switch, 92.30% peak, +0.47 points.
  Implication: preserve one-operation magnitude-7 augmentation through 80%, then crop/flip refinement.

- **Width-2 capacity beats the fixed-time update loss under strong views** (EXP-007)
  Evidence: `goals/maximize-cifar10-best-test-accuracy/experiments/007/04-analysis.md` § Results — 93.55%, +1.25 points with 70.76% of prior steps.
  Implication: use width 2 as the model baseline; tune its regularization before reducing capacity.

- **Plateau-only CutMix complements N1/M7 when followed by a hard weak tail** (EXP-010)
  Evidence: `goals/maximize-cifar10-best-test-accuracy/experiments/010/04-analysis.md` § Results — 94.15%, +0.60 points with 99.10% step retention.
  Implication: preserve 50% alpha-1 regional mixing through 80% and remove it with RandAugment for refinement.

### Low Importance

- **Seven budget checkpoints captured a monotonic annealed trajectory with final equal to best** (EXP-001)
  Evidence: `goals/maximize-cifar10-best-test-accuracy/experiments/001/04-analysis.md` § Results — 82.05% to 91.57% monotonically.
  Implication: checkpointed evaluation is viable for smooth final-focused schedules under this wall limit.

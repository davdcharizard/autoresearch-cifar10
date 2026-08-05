# Experiment Learnings: Maximize CIFAR-10 Test Accuracy

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
### Medium Importance
### Low Importance

- **Delay-free loader CV can reject stable production-paced delivery** (EXP-046)
  Evidence: `experiments/046/04-analysis.md` - service CVs reached13.55% while all11ms-overlap CVs were<=2.11% and projected wall was344.17s.
  Implication: prospectively gate CPU transforms on paired production-paced overlap; retain unconstrained service as diagnostic.

- **Emit preflight measurements before enforcing gates** (EXP-029)
  Evidence: `experiments/029/04-analysis.md` - stable retention failed below 0.9022, but fail-before-print ordering lost exact windows.
  Implication: print or persist measured payloads before assertions while preserving no-rerun decisions.

- **Persistent workers plus sparse evaluation keep high-epoch runs within the wall limit** (EXP-001)
  Evidence: `experiments/001/04-analysis.md` § Execution — timeout at 91% became a 342.5s complete run
  Implication: use persistent DataLoader workers and evaluate periodically plus final.

- **Isolated worker RNG plus an exhausted-epoch flag gives leak-free temporal transforms** (EXP-026)
  Evidence: `experiments/026/04-analysis.md` - exact clean-tail replay and marker checks passed under forkserver workers.
  Implication: preserve base augmentation RNG and flip shared policy state only between fully exhausted iterators.

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

> **Recurring failure:** Removing a stage-3 gate, input dependence, or global cross-channel mixing destroys full SE's small positive signal.

- **Stage-3 SE needs both full conditional gates yet still misses the margin** (count: 5, EXP-017, EXP-018, EXP-019, EXP-024, EXP-025)
  Mechanism: full gates scored 94.16; simplifications lost the signal, while diagnostic-free full gates projected only 136.90 passes.
  Insight: close this attention family; dense global interaction distinguishes its signal but remains too costly for the protected regime.

### Medium Importance

- **Hard-tail LR motion is bracketed around the accepted global cosine** (count: 2, EXP-008, EXP-039)
  Mechanism: a zero endpoint scored 93.80%/0.2629 loss; a 39.46%-larger rephased tail scored 93.98%/0.2661, both at normal exposure.
  Insight: retain the accepted global curve and 0.002 floor; deprioritize tail-shape tuning absent a new diagnosis.

- **Classifier decay is locally bracketed around the accepted 5e-4** (count: 2, EXP-037, EXP-038)
  Mechanism: zero decay scored 94.41%/0.2786 loss; `1e-3` scored 93.82%/0.2598, both at normal exposure versus accepted 94.48%/0.2456.
  Insight: retain `5e-4` and deprioritize nearby static tuning; intermediate values and schedules remain formally untested absent a new mechanism.

- **Batch-shared mixup strength is bracketed around alpha 0.2** (count: 2, EXP-005, EXP-035)
  Mechanism: alpha 0.4 lost 0.50 points on the earlier WRN; alpha 0.1 lost 0.60 on the accepted deeper-plus-RandAugment learner, both at normal exposure with worse loss.
  Insight: retain alpha 0.2; close adjacent strengths, seeds, symmetrization, RNG realignment, and cutoff rescues without a new independent mechanism.

- **Batch-size scaling misses the protected exposure/decision regime** (count: 2, EXP-029, EXP-034)
  Mechanism: batch 128 lost >9.78% image rate; batch 512 gained only 6.10%, insufficient to justify roughly 47% fewer updates.
  Insight: retain batch 256; close adjacent batch/LR scaling absent a new kernel or optimizer mechanism.

- **Late parameter averaging fails across distinct state policies** (count: 2, EXP-013, EXP-033)
  Mechanism: whole-state EMA peaked at 94.10; a short three-point parameter average scored 93.87 at 132.47 passes with worse loss.
  Insight: close immediate averaging windows, decays, and BN-state rescues; lower or smoother loss does not ensure top-1 boundary gains.

- **Early residual feature masking harms the accepted mixup regime** (count: 2, EXP-006, EXP-030)
  Mechanism: broad p=0.10 dropout lost 0.55 points; isolated p=0.05 drop-path on `layer3[2]` lost 0.41 with worse loss at 132.72 passes.
  Insight: preserve full residual contributions during early mixup/RandAugment; close adjacent mask strengths, placements, seeds, and cutoffs.

- **Final-window SAM fails the fixed-budget/top-1 tradeoff** (count: 2, EXP-021, EXP-022)
  Mechanism: dense SAM retained only 89.67% exposure; alternating SAM retained 136.99 passes but scored 93.79 despite 0.2329 loss.
  Insight: close rho-0.05 late SAM; lower terminal loss or flatter geometry does not preserve top-1 refinement.

- **Mixup duration is locally bracketed around the accepted 65% cutoff** (count: 2, EXP-004, EXP-020)
  Mechanism: 50% scored 93.91 and 75% scored 93.82 at normal exposure, versus 94.07 at 65%.
  Insight: retain 65%; both shorter and longer windows misbalance regularization and clean refinement.

### Low Importance

- **Late feature mixup cannot replace accepted input interpolation** (count: 1, EXP-047)
  Mechanism: active post-GAP mixing retained130.09 passes but scored94.20%/0.2619 versus accepted94.48%/0.2456.
  Insight: preserve mixed-pixel backbone training; other feature sites are untested but declined as immediate placement rescues.

- **Fixed averaging in downsampling shortcuts removes useful spatial phase evidence** (count: 1, EXP-045)
  Mechanism: exact two-transition phase averaging retained129.10 passes but scored94.11%/0.2512 versus accepted94.48%/0.2456.
  Insight: preserve single-phase projections; close immediate transition/kernel/order/gain/main-filter rescues absent a new diagnosis.

- **Fixed spatial dispersion is redundant with the accepted pooled representation** (count: 1, EXP-044)
  Mechanism: zero-start std input retained128.71 passes but scored93.95%/0.2637; post-BN/ReLU mean/std correlation was0.835-0.854.
  Insight: preserve GAP plus the accepted MLP; close immediate variance/RMS/max/epsilon/adapter variants absent a new diagnosis.

- **Full-run convolution gradient centralization removes useful common-mode learning** (count: 1, EXP-043)
  Mechanism: `129.81` passes scored 93.88%/0.2661; fixed-fixture projection removed 97.85-98.80% of the stem gradient norm.
  Insight: preserve raw convolution gradients; close the exact all-convolution rule and immediate layer/strength/axis/schedule rescues absent a new diagnosis.

- **Single-query content pooling harms the accepted invariant readout** (count: 1, EXP-042)
  Mechanism: exact-neutral covariance-open attention retained 127.93 passes but scored 93.80% with 0.2787 loss versus accepted 94.48%/0.2456.
  Insight: preserve uniform GAP; decline immediate temperature, initialization, scale, query-count, and cutoff variants as post-result search.

- **Direct-path auxiliary CE weakens the pooled-head frontier** (count: 1, EXP-041)
  Mechanism: exact shared-classifier 90/10 supervision retained 128.54 passes but scored 94.26% with 0.2529 loss despite 0.976-0.989 sampled gradient cosines.
  Insight: preserve sole refined-path CE; close immediate auxiliary-weight, cutoff, detach, separate-head, distillation, and head-scale rescues.

- **Continuous equal classifier radii harm pooled-head boundary quality** (count: 1, EXP-040)
  Mechanism: Frobenius-preserving RMS row normalization retained 127.54 passes but scored 93.91% with 0.2622 loss versus accepted 94.48% and 0.2456.
  Insight: preserve ordinary affine class radii; close the exact RMS map while treating distinct classifier geometries as formally untested.

- **Reflection padding destabilizes active worker delivery** (count: 1, EXP-032)
  Mechanism: active timing had 11.20% CV and a 4.006s outlier versus 0.97% CV for constant padding, so scoring was disallowed.
  Insight: close exact reflection; require stable worker overlap before scoring any CPU-side input-padding change.

- **Channels-last exceeds the accepted numerical-identity envelope** (count: 1, EXP-031)
  Mechanism: logical state matched, but deterministic NHWC logits differed by up to 8.89e-4 and failed fixed bounds before timing.
  Insight: close exact channels-last and its placement/flag/precision rescues; format conversion is not a pure exposure lever here.

- **Late high-resolution prefix freezing trades essential adaptation for exposure** (count: 1, EXP-028)
  Mechanism: freezing 3.4% of parameters cut hard steps 35.9% and delivered 159.10 passes, but scored 93.99 with worse 0.2804 loss.
  Insight: preserve stem/stage-1 gradients through the clean tail; extra upper-layer decisions do not replace late low-level adaptation.

- **Early one-op magnitude-5 RandAugment is below the standalone margin** (count: 1, EXP-026)
  Mechanism: RNG-isolated early augmentation retained 142.45 passes and gained 0.05 points, but scored only 94.12 with 0.2574 loss.
  Insight: close immediate RandAugment tuning; reuse its mild positive invariance signal only under a distinct interaction hypothesis.

- **Selective width plus full SE exceeds the protected exposure envelope** (count: 1, EXP-023)
  Mechanism: diagnostic-free `[32,64,160]` plus two full gates retained 88.94%, projecting 126.21 passes below its 127 floor.
  Insight: close this exact composition; combine near-miss mechanisms only when measured overhead preserves their operating regime.

- **Fixed-MAC late redistribution sacrifices essential high-resolution refinement** (count: 1, EXP-016)
  Mechanism: `[1,2,3]` delivered 171.70 passes but scored 93.82% with 0.2778 loss, despite exact state/RNG control.
  Insight: retain both stage-1 blocks; late capacity helps only when it does not replace early refinement.

- **Per-example mixup removes useful batch-level coefficient coherence** (count: 1, EXP-015)
  Mechanism: independent alpha-0.2 draws preserved 142.01 passes and the marginal law but scored 93.79%, 0.28 below accepted.
  Insight: retain batch-shared coefficients; require a new mechanism before revisiting coefficient or pairing structure.

- **Exact zero residual endpoints mismatch the accepted schedule** (count: 1, EXP-014)
  Mechanism: 142.81 passes scored 93.88% with 0.2660 loss despite correct two-step branch opening and unchanged graph.
  Insight: keep Kaiming residual startup; do not retry zero-BN or adjacent endpoint scales without new evidence.

- **Rank-64 post-stage-3 bottleneck loses the capacity signal** (count: 1, EXP-012)
  Mechanism: 135.49 passes scored 93.74% with 0.2873 loss despite 96.0% throughput retention and exact RNG matching.
  Insight: close this half-width placement; do not assume lower-MAC refinement preserves dense width/depth benefits.

- **One extra 8x8 residual block is below the acceptance margin** (count: 1, EXP-011)
  Mechanism: 132.92 passes scored 94.15% (+0.08), but test loss worsened to 0.2782 and the run missed 94.17%.
  Insight: exact `[2,2,3]` is closed; consider compute-efficient transformation or generalization control, not adjacent depth.

- **Selective 160-channel stage-3 width is below the acceptance margin** (count: 1, EXP-010)
  Mechanism: 132.16 passes scored 94.11% (+0.04) with best loss 0.2435, but missed required 94.17%.
  Insight: low-resolution capacity is promising; close this exact width and require a distinct allocation rationale.

- **BF16 speedup trades accuracy for denser updates** (count: 1, EXP-009)
  Mechanism: 159.07 passes (+12.1%) scored 93.81%, while final loss worsened to 0.2634 despite stable execution.
  Insight: retain FP32 numerics; do not rescue BF16 with batch or LR changes after this joint treatment failed.

- **Removing matrix weight decay at 65% harms generalization** (count: 1, EXP-007)
  Mechanism: 142.5 passes scored 93.74%, while final test loss rose from 0.2432 to 0.3244.
  Insight: keep `5e-4` matrix decay throughout; mixup and weight decay do not share an optimal cutoff.

- **Shared-rectangle CutMix area labels underperform early mixup** (count: 1, EXP-003)
  Mechanism: uniform rectangles pasted 31% of pixels and scored 93.72% despite normal 142.5-pass exposure.
  Insight: retain mixup; revisit CutMix only with smaller patches, per-sample regions, or low mixture probability.

## Patterns

<!-- Goal-specific patterns, mechanisms, and reusable recipes discovered during experiments.
     Examples:
       - "Primary-bucket-only gating by aligning a new gate's threshold with an existing gate's
          threshold keeps long-path behavior bit-identical — composable."
       - "Image-time static override of configs_base.py avoids runtime-mutation side effects."

     Entry format: 3-line statement / evidence / implication (budget-strict). -->

### High Importance
### Medium Importance
### Low Importance

- **A small nonlinear residual remapping after global pooling improves the accepted representation** (EXP-036)
  Evidence: `experiments/036/04-analysis.md` - a scale-0.1 `128 -> 64 -> 128` head scored 94.48% and 0.2456 loss at 130.304 passes.
  Implication: preserve the exact pooled head as the new base; prefer post-spatial capacity over extra feature-map compute, without adjacent scale/width tuning.

- **Early image invariance unlocks extra low-resolution WRN capacity** (EXP-011, EXP-026, EXP-027)
  Evidence: `experiments/027/04-analysis.md` - exact composition scored 94.32 with 0.2523 loss versus standalone 94.15 and 94.12.
  Implication: preserve the deeper-plus-early-RandAugment interaction as the new base; neither component's standalone miss predicts their composition.

- **Moderate WRN capacity and a time-aligned cosine schedule improve the fixed-budget baseline** (EXP-001)
  Evidence: `experiments/001/04-analysis.md` § Results — 91.54% to 93.38% under the same 300s budget
  Implication: keep WRN-16-2 and time-based annealing as the base for controlled refinements.

- **Early mild mixup plus a long hard-label tail improves WRN generalization** (EXP-002)
  Evidence: `experiments/002/04-analysis.md` § Results — 93.38% to 94.07% with 141.9 data passes
  Implication: retain temporal mixup removal and tune its strength or cutoff in controlled trials.

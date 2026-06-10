# Goal Learnings: Maximize CIFAR-10 Best Test Accuracy

**Goal**: goals/maximize-cifar10-best-test-accuracy.md
**Experiment Index**: experiment-indices/maximize-cifar10-best-test-accuracy.tsv

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
         Evidence: reports/exp-report-006.md § Verification Results — 8-case drift table
         Implication: capture pristine baseline same-day as experiment measurement. -->

### High Importance
> **Recurring failure:** Width increases beyond the 28/56/112 anchor have now failed across proportional and final-stage-only variants.

- **Widening beyond 28/56/112 fails under the fixed recipe** (count: 3, EXP-017, EXP-019, EXP-020)
  Mechanism: wider variants either lose useful budget or disrupt the channel balance; final-stage 128 still peaked at only 92.60%.
  Insight: stop prioritizing extra channels above 28/56/112; try sparse averaging, schedule brackets, or non-capacity changes.
### Medium Importance
### Low Importance
- **Missing the 21k first LR drop invalidates clean anchor comparisons** (EXP-047)
  Evidence: reports/exp-report-047.md § Results — contention ended run at 20,321 steps, all LR lines stayed at 0.1000.
  Implication: avoid strong scientific conclusions from step-schedule runs that miss the first milestone.

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
> **Recurring failure:** Schedule-only second LR drops have underperformed across multiple anchors.
> **Recurring failure:** Weight averaging variants have failed through overhead, collapse, or BatchNorm/state mismatch.
> **Recurring failure:** Isolated batch-size deviations from 128 now fail on both smaller and larger sides.
> **Recurring failure:** Deviating from full-run label smoothing 0.05 fails as static changes and post-drop annealing.
> **Recurring failure:** Deviating from the `LR=0.1` startup anchor fails as scalar and short-warmup variants.
> **Recurring failure:** Static CutMix probability moves away from p=0.5 fail across broad and fine brackets.
> **Recurring failure:** Static CutMix alpha moves away from 1.0 fail across broad and fine brackets.

- **CutMix probability brackets away from p=0.5 fail to clear threshold** (count: 3, EXP-065, EXP-066, EXP-090)
  Mechanism: p=0.25 peaked at 94.09%, p=0.75 tied 94.11%, and p=0.4 under the spatial anchor peaked at 94.13%.
  Insight: keep `CUTMIX_PROB=0.5`; only revisit CutMix timing or strength through a clearly distinct coupled mechanism.
- **CutMix alpha brackets away from alpha=1.0 fail to clear threshold** (count: 3, EXP-067, EXP-068, EXP-091)
  Mechanism: alpha 0.5 peaked at 94.07%, alpha 2.0 peaked at 94.00%, and alpha 0.75 under the spatial anchor peaked at 94.34%.
  Insight: keep `CUTMIX_ALPHA=1.0`; only revisit CutMix strength through a distinct scheduled or coupled mechanism.
- **Schedule-only second LR drops fail to improve current anchors** (count: 3, EXP-003, EXP-024, EXP-030)
  Mechanism: LR 0.001 refinement is reachable but either reduces exploration or only flattens late accuracy below the active threshold.
  Insight: stop isolated second-drop tuning; only pair low-LR refinement with an independently motivated non-schedule change.
- **Startup LR deviations from 0.1 weaken the current anchor** (count: 3, EXP-040, EXP-043, EXP-080)
  Mechanism: LR 0.12 peaked at 93.70%, LR 0.08 peaked at 93.49%, and a 500-step 0.02-to-0.1 warmup peaked at 94.08%.
  Insight: keep `LR = 0.1` from startup; revisit optimizer transients only with a distinct coupled mechanism.
- **Weight averaging variants fail under the fixed CIFAR harness** (count: 3, EXP-004, EXP-021, EXP-045)
  Mechanism: per-step EMA lost too many steps, equal averaging collapsed, and sparse late EMA fell from 89.07% to 79.45%.
  Insight: stop EMA/SWA-style averaging unless a plan solves BN/state mismatch without extra validation.
- **Batch-size deviations from 128 lose the anchor tradeoff** (count: 3, EXP-025, EXP-036, EXP-053)
  Mechanism: smaller batches lose useful coverage, while batch 160 preserved the LR drop but peaked at 93.71% below the anchor.
  Insight: stop isolated batch-size retuning; revisit only with a coupled optimizer or schedule rationale.
- **Label-smoothing deviations from 0.05 fail to clear the anchor** (count: 5, EXP-033, EXP-037, EXP-057, EXP-074, EXP-075)
  Mechanism: static 0.03/0.08, post-drop hard labels, hard CutMix endpoints, and the fan-out coupled retry all missed threshold.
  Insight: keep `label_smoothing=0.05` for clean and CutMix endpoint losses; avoid more hard-label endpoint variants.
### Medium Importance
- **Horizontal flip below p=0.4 under-regularizes spatial invariance** (count: 2, EXP-083, EXP-089)
  Mechanism: p=0.35 peaked at 94.17% and p=0.375 under padding 3 peaked at 94.29%, both below their active anchors.
  Insight: keep flip p=0.4 as the spatial anchor; stop isolated lower flip brackets unless coupled to a distinct mechanism.
- **Early CutMix temporal scheduling gives only sub-threshold gains** (count: 2, EXP-073, EXP-079)
  Mechanism: 2000-step clean warmup reached 94.14%, while a 1000-step probability ramp peaked at 94.09%; both missed 94.21%.
  Insight: keep static CutMix for now; revisit early scheduling only with an independent optimizer or stability mechanism.
- **SE channel attention underperforms the current block** (count: 2, EXP-058, EXP-063)
  Mechanism: all-block SE peaked at 93.71%, while layer3-only SE reduced overhead but fell to 93.26%.
  Insight: deprioritize SE gates; revisit attention only with a non-SE mechanism or stronger coupled architecture rationale.
- **Mild mixup variants remain below the current anchor** (count: 2, EXP-055, EXP-060)
  Mechanism: alpha 0.1 peaked at 93.85% with endpoint smoothing and 93.81% without it, both below the 94.07% threshold.
  Insight: deprioritize direct mixup variants; revisit image/label interpolation only with a clearly different mechanism such as regional mixing.
- **Cosine LR schedule variants underperform the 21k step anchor** (count: 2, EXP-046, EXP-052)
  Mechanism: full elapsed-time cosine peaked at 93.01%, and tail-only cosine to 0.002 reached 93.87% but missed 94.07%.
  Insight: avoid isolated cosine schedules; revisit LR tails only when coupled to a distinct non-schedule mechanism.
- **Residual-branch BN down-scaling undertrains the fixed-budget anchor** (count: 2, EXP-028, EXP-051)
  Mechanism: full zero-gamma collapsed to 91.74%, while partial `bn2.weight=0.1` recovered to 93.64% but missed 94.07%.
  Insight: keep default final-BN scale; revisit identity-biased residual initialization only with a stronger coupled rationale.
- **No-decay BatchNorm/bias parameter groups underperform** (count: 2, EXP-027, EXP-056)
  Mechanism: both the older `1e-4` anchor and current `2e-4` anchor lost accuracy when BN affine and bias skipped decay.
  Insight: keep simple global coupled SGD L2 decay; avoid parameter-group decay exceptions without a broader retuning rationale.
- **Classifier-head-only modifications weaken the current anchor** (count: 2, EXP-061, EXP-076)
  Mechanism: final-feature dropout peaked at 93.54%, and Xavier/zero-bias classifier initialization peaked at 93.73%.
  Insight: avoid isolated classifier-head tweaks; revisit only as part of a coupled late-overfit or calibration mechanism.
- **Cutout masking over-regularizes fixed-budget ResNet-20** (count: 2, EXP-005, EXP-009)
  Mechanism: 16x16 cutout peaked at 91.72%, and weaker 8x8 p=0.25 cutout still peaked only at 91.87%.
  Insight: deprioritize erased-patch masking unless paired with a stronger schedule or architecture rationale.
- **Policy augmentation underperforms current anchors** (count: 2, EXP-044, EXP-071)
  Mechanism: mild RandAugment peaked at 93.83%, and CIFAR AutoAugment on CutMix peaked at 93.62%.
  Insight: deprioritize isolated policy augmentation; revisit only with a targeted, lower-cost sub-policy rationale.
- **Conv fan-out initialization gives only sub-threshold or regressive gains** (count: 2, EXP-072, EXP-075)
  Mechanism: isolated fan-out reached 94.16%, but combining it with hard CutMix endpoints regressed to 93.92%.
  Insight: avoid more Conv2d fan-out retries; only revisit initialization through a distinct classifier-specific mechanism.
- **Transition downsampling smoothing weakens the anchor** (count: 2, EXP-059, EXP-077)
  Mechanism: shortcut average-pooling peaked at 93.42%, and residual-branch average-pooling peaked at 93.99%.
  Insight: keep original stride-2 transition behavior; only revisit downsampling through a broader block-topology change.
- **Horizontal flip above p=0.4 restores too much spatial regularization** (count: 2, EXP-084, EXP-087)
  Mechanism: p=0.45 under padding 4 peaked at 94.05%, and p=0.425 under padding 3 peaked at 94.34% below the 94.51 anchor.
  Insight: keep flip p=0.4 as the spatial anchor; only test lower/finer flip if justified by a distinct crop interaction.
- **Weight decay above 2e-4 over-regularizes current anchors** (count: 2, EXP-039, EXP-088)
  Mechanism: `3e-4` peaked at 93.55% on the older anchor, and `2.5e-4` peaked at 94.07% on the spatial anchor.
  Insight: keep `WEIGHT_DECAY = 2e-4`; stronger shrinkage is not the missing complement to padding 3 / flip p=0.4.
- **Weight decay below 2e-4 under-regularizes current anchors** (count: 2, EXP-041, EXP-092)
  Mechanism: `1.5e-4` peaked at 93.61% on the older anchor, and `1.75e-4` peaked at 94.14% on the spatial/CutMix anchor.
  Insight: keep `WEIGHT_DECAY = 2e-4`; isolated lower decay brackets are closed unless paired with a distinct mechanism.
### Low Importance
- **Crop padding below 3 under flip p=0.4 under-regularizes** (count: 1, EXP-086)
  Mechanism: reducing reflect padding from 3 to 2 peaked at 94.22%, below the 94.51% padding-3 / flip-p=0.4 anchor.
  Insight: keep padding 3 as the crop anchor; explore fine flip brackets or coupled mechanisms before more crop reduction.
- **Isolated reflection crop padding 3 gives only a sub-threshold gain** (count: 1, EXP-081)
  Mechanism: reducing reflect padding from 4 to 3 peaked at 94.18%, above baseline but below the 94.21% noise guard.
  Insight: padding 3 is not enough alone, but EXP-085 shows it can work when coupled with flip p=0.4.
- **Standard CIFAR channel std miscalibrates the CutMix anchor** (count: 1, EXP-070)
  Mechanism: scaling inputs by CIFAR std under the tuned LR/BN/CutMix recipe collapsed best accuracy to 75.03%.
  Insight: keep unit-std input scaling; revisit normalization only with a broad LR/optimizer retune, not as an isolated tweak.
- **Post-drop CutMix probability taper weakens the anchor** (count: 1, EXP-069)
  Mechanism: lowering `p` from 0.5 to 0.25 after step 21000 peaked at 93.73%, below the 94.11% CutMix baseline.
  Insight: keep static `CUTMIX_PROB=0.5`; deprioritize temporal CutMix weakening unless coupled to a stronger mechanism.
- **Compact shallow-wide ResNet-14 weakens the anchor** (count: 1, EXP-062)
  Mechanism: `(32,64,128)` ResNet-14 reached 51,471 steps but peaked at 93.51%, below the 94.07% threshold.
  Insight: preserve ResNet-20 depth for capacity work; try localized architecture changes rather than removing blocks.
- **Very mild stochastic depth weakens the current anchor** (count: 1, EXP-054)
  Mechanism: `STOCHASTIC_DEPTH_MAX_P=0.03` preserved the LR drop but plateaued at 93.40%, below the 93.97% anchor.
  Insight: avoid isolated residual drop regularization; revisit only with a much stronger coupled rationale.
- **Clean mild ColorJitter remains below threshold** (count: 1, EXP-050)
  Mechanism: clean ColorJitter reached the 21k LR drop and 41,280 steps but peaked at 93.49%, below the 93.97% anchor.
  Insight: deprioritize isolated photometric jitter; revisit augmentation only with a distinct coupled mechanism.
- **Decoupled weight decay weakens the 2e-4 anchor** (count: 1, EXP-049)
  Mechanism: manual decoupled `2e-4` reached 40,437 steps and the LR drop but peaked at 93.06%, below 93.97%.
  Insight: keep coupled SGD L2 decay for this anchor; revisit decay semantics only with broader retuning.
- **Lower BatchNorm momentum lags the current anchor** (count: 1, EXP-048)
  Mechanism: `momentum=0.05` reached the LR drop cleanly but peaked at 93.48%, well below the 93.97% baseline.
  Insight: keep default BN momentum; only revisit BatchNorm state dynamics with a distinct coupled mechanism.
- **Combining lower smoothing with 22k first drop regresses** (count: 1, EXP-035)
  Mechanism: coupling 0.03 smoothing with a 22k first drop fell to 93.63%, below both single-axis near-misses.
  Insight: do not assume adjacent near-misses are additive; move to a distinct lever such as batch size or bounded late averaging.
- **Combined strong regularization plus slow cosine undertrains ResNet-20** (count: 1, EXP-000)
  Mechanism: exact 16x16 cutout, label smoothing, and cosine over 64k steps held best accuracy to 90.45%.
  Insight: isolate components or calibrate LR decay to the observed ~35k-step budget before retrying.
- **BF16 throughput bundle narrowly misses baseline despite more steps** (count: 1, EXP-001)
  Mechanism: channels-last, BF16 autocast, and compile reached 39,558 steps but only 91.48%.
  Insight: isolate precision-preserving speedups before combining throughput with LR or architecture changes.
- **ResNet-32 with late first LR drop undertrains badly** (count: 1, EXP-006)
  Mechanism: only 23,642 steps completed, missing the 26k first drop and peaking at 88.18%.
  Insight: larger models need measured step-budget calibration or TF32 speedups before capacity scaling is retried.
- **TF32 enablement slows the proven ResNet-20 FP32 path** (count: 1, EXP-007)
  Mechanism: warning disappeared, but the run completed only 37,922 steps and peaked at 91.39%.
  Insight: keep TF32 disabled for this recipe; prioritize schedule or loop changes that preserve step budget.
- **First LR drop at 30k starts LR 0.01 refinement too early** (count: 1, EXP-008)
  Mechanism: `[30000, 64000]` completed 46,331 steps with no LR 0.001 phase but peaked at 91.65%.
  Insight: the original 32k first drop is better calibrated; schedule-only gains around this recipe are unlikely.
- **Nesterov momentum miscalibrates the proven FP32 recipe** (count: 1, EXP-010)
  Mechanism: throughput stayed healthy at 45,163 steps, but post-drop accuracy plateaued at 91.33%.
  Insight: keep classical momentum unless jointly retuning LR schedule or momentum value.
- **Momentum 0.95 worsens the 28/56/112 anchor** (count: 1, EXP-026)
  Mechanism: higher classical momentum preserved throughput but plateaued at 92.90%, below the 93.23% baseline.
  Insight: keep `MOMENTUM = 0.9`; only revisit momentum as part of a broader LR/momentum retuning.
- **First LR drop at 22k is too early for widened ResNet-20** (count: 1, EXP-012)
  Mechanism: more LR 0.01 steps yielded 92.16%, below the 92.22% threshold and worse than EXP-011's 24k drop.
  Insight: keep 24k as the widened-model default; prefer wider capacity or later-drop variants over earlier drops.
- **First LR drop at 23k is too late for 28/56/112 ResNet-20** (count: 1, EXP-015)
  Mechanism: 23k produced 38,274 steps but peaked at 92.88%, below EXP-014's 93.09% with a 22k drop.
  Insight: bracket this width locally with 21k if retuning; do not retry 23k without another coupled change.
- **First LR drop at 22k misses the label-smoothed threshold** (count: 1, EXP-034)
  Mechanism: 22k preserved schedule behavior but peaked at 93.79%, below the 93.80 threshold.
  Insight: keep 21k for the label-smoothed anchor; avoid adjacent first-drop retunes without a distinct coupled change.
- **First LR drop at 20k is slightly too early for 28/56/112 ResNet-20** (count: 1, EXP-022)
  Mechanism: 20k produced 41,876 steps and peaked at 93.18%, just below EXP-016's 93.23% with a 21k drop.
  Insight: keep 21k as the local schedule anchor; move to non-schedule perturbations before finer brackets.
- **Lower weight decay weakens 28/56/112 generalization** (count: 1, EXP-023)
  Mechanism: `5e-5` weight decay completed 42,754 steps but peaked at 92.83%, below the 93.23% anchor baseline.
  Insight: avoid weaker decay; EXP-038 shows the useful direction is stronger shrinkage, now anchored at `2e-4`.
- **Projection shortcuts hurt the 28/56/112 recipe despite preserving steps** (count: 1, EXP-018)
  Mechanism: learned 1x1+BN transition shortcuts reached 38,322 steps but peaked at 92.97%, below the 93.23% baseline.
  Insight: do not prioritize option-B shortcuts here; return to width/schedule calibration or lower-overhead late averaging.
- **Pre-activation blocks weaken the current CutMix anchor** (count: 1, EXP-078)
  Mechanism: full pre-activation BasicBlocks plus final BN/ReLU peaked at 93.92% and completed only 36,288 steps.
  Insight: keep the post-activation block topology; prefer non-topology or coupled mechanisms before more residual rewrites.

## Patterns

<!-- Goal-specific patterns, mechanisms, and reusable recipes discovered during experiments.
     Examples:
       - "Primary-bucket-only gating by aligning a new gate's threshold with an existing gate's
          threshold keeps long-path behavior bit-identical — composable."
       - "Image-time static override of configs_base.py avoids runtime-mutation side effects."

     Entry format: 3-line statement / evidence / implication (budget-strict). -->

### High Importance
### Medium Importance
- **Crop padding 3 plus flip p=0.4 improves the spatial anchor** (EXP-085)
  Evidence: reports/exp-report-085.md, reports/exp-report-086.md, reports/exp-report-087.md — padding 3 reached 94.51%, while padding 2 and p=0.425 regressed.
  Implication: treat padding 3 / flip p=0.4 as the spatial anchor; avoid more crop reduction or upper flip restoration.
- **Flip p=0.4 is locally bracketed by worse nearby probabilities** (EXP-082, EXP-083, EXP-084, EXP-087, EXP-089)
  Evidence: reports/exp-report-083.md, reports/exp-report-084.md, reports/exp-report-087.md, reports/exp-report-089.md — p=0.35, p=0.375, p=0.425, and p=0.45 all missed baseline.
  Implication: preserve p=0.4 as the spatial anchor; move to distinct coupled mechanisms rather than more isolated flip brackets.
- **Lower horizontal flip probability improves the CutMix anchor** (EXP-082)
  Evidence: reports/exp-report-082.md § Results — `RandomHorizontalFlip(p=0.4)` reached 94.36%, +0.25pp over baseline.
  Implication: use flip p=0.4 as the new spatial augmentation anchor; bracket nearby probabilities before adding new regularizers.
- **Probabilistic CutMix improves the 2e-4 label-smoothed anchor** (EXP-064)
  Evidence: reports/exp-report-064.md § Results — `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5` reached 94.11%.
  Implication: keep regional mixing in the anchor; bracket CutMix strength before unrelated augmentation.
- **Mild label smoothing improves the reflection anchor** (EXP-032)
  Evidence: reports/exp-report-032.md § Results — `label_smoothing=0.05` reached 93.70% with 41,823 steps.
  Implication: keep mild label smoothing in the anchor; future regularizers should be milder or coupled carefully.
- **Stronger weight decay improves the label-smoothed reflection anchor** (EXP-038)
  Evidence: reports/exp-report-038.md, reports/exp-report-039.md, reports/exp-report-041.md, reports/exp-report-088.md — `2e-4` improved, while 1.5e-4, 2.5e-4, and 3e-4 regressed.
  Implication: keep `2e-4` as the decay anchor; future regularization should use a distinct mechanism, not more scalar decay.
- **Reflection crop padding beats nearby crop-boundary siblings** (EXP-029, EXP-031)
  Evidence: reports/exp-report-029.md and reports/exp-report-031.md — reflection reached 93.58%; symmetric peaked at 93.48%.
  Implication: keep `padding_mode="reflect"` as the augmentation anchor; deprioritize isolated padding-mode sibling tests.
- **First LR drop at 21k improves 28/56/112 ResNet-20** (EXP-016)
  Evidence: reports/exp-report-016.md § Results — 34,208-step run reached 93.23%, +0.14 points over EXP-014.
  Implication: use 21k as the current schedule anchor for this width; pair wider models with earlier calibrated drops.
- **Width scaling to 28/56/112 works with a 22k first drop** (EXP-014)
  Evidence: reports/exp-report-014.md § Results — 822,790-param ResNet-20 reached 34,259 steps and 93.09%.
  Implication: further width tests should jointly calibrate first LR drop to reachable step budget.
- **Width scaling remains effective at 24/48/96 with a 24k first drop** (EXP-013)
  Evidence: reports/exp-report-013.md § Results — 605k-param ResNet-20 reached 41,825 steps and 92.49%.
  Implication: continue capacity scaling only with explicit schedule/step-budget calibration.
- **Modest width scaling works when the first LR drop is step-budget calibrated** (EXP-011)
  Evidence: reports/exp-report-011.md § Results — 20/40/80 ResNet-20 reached 43,713 steps and 92.12%.
  Implication: capacity experiments should estimate reachable steps and set LR milestones before launch.
### Low Importance
- **FP32 compile plus channels-last improves fixed-budget ResNet-20** (EXP-002)
  Evidence: reports/exp-report-002.md § Results — 43,398 steps and 91.95% best accuracy.
  Implication: preserve FP32 throughput path; tune LR milestones or TF32 next before adding regularization.

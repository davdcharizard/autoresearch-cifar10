# Brainstorm EXP-006
**Created**: 2026-08-05

## Web Search & Literature Review

- **ASAM: Adaptive Sharpness-Aware Minimization** (`papers/asam.md`)
  Element-wise adaptive perturbations remove SAM's parameter-scale sensitivity. Across CIFAR-10 WRN models, ASAM improves over SAM by 0.20-0.30 points using `rho=0.5`, `eta=0.01`, directly motivating a geometry replacement for EXP-004's successful late SAM pulses.
- **Manifold Mixup** (`papers/manifold-mixup.md`)
  One-pass interpolation at randomly selected hidden boundaries substantially improves CIFAR-10 PreActResNet and WRN results over both vanilla training and input Mixup. It preserves images and steps, unlike EXP-005's overlapping stream.
- **Aggregated Residual Transformations / ResNeXt** (`papers/resnext.md`)
  Cardinality can outperform width or depth at matched complexity. CIFAR ResNeXt-29 8x64d reaches 3.65% error versus 4.17% for a similarly sized Wide ResNet, but compact transfer to this short fixed-time regime is uncertain.
- **Squeeze-and-Excitation Networks** (CVPR 2018, https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper)
  Channel recalibration consistently improves residual backbones at minimal arithmetic cost; the evidence is mainly ImageNet and does not match this CIFAR recipe.
- **Improving Sharpness-Aware Minimization by Lookahead** (ICML 2024, https://proceedings.mlr.press/v235/yu24q.html)
  A lookahead mechanism can reduce SAM convergence instability and saddle oscillation, but it is a larger optimizer departure than ASAM and lacks a direct late-periodic mapping.

## Experimental History Review

- The accepted lineage is BASE 91.51% -> EXP-001 94.62% -> EXP-002 95.23% -> EXP-004 95.40%. EXP-004 remains the global-best branch tip and now requires at least 95.50%.
- EXP-001 converted H20 headroom into a BF16, time-scheduled PreAct WRN. EXP-002's front-loaded CutMix added 0.61 points. EXP-004's period-two SAM in the final clean quarter added 0.17 points while retaining 25,560 steps.
- EXP-003 showed that selected scalar CutMix/drop-path gains can reverse by 0.14-0.29 points. A new candidate needs a mechanism-level effect rather than another local scalar search.
- EXP-005 is the only failed child of EXP-004. Its half-overlap DLB recipe retained 25,492 steps and lowered final test loss, but halved new-image introduction and reduced best accuracy to 95.28%. EXP-006 must preserve the parent's independent-image stream.
- EXP-004 still uses only 2.75M parameters and 1,190.5 MiB. The target is a generalization improvement with effect above the 0.10-point gate; memory is abundant, while model-pass overhead and lost independent images are costly.

## Collected Ideas

- **Adaptive clean-tail sharpness minimization:** Replace only the geometry of EXP-004's scheduled SAM pulses with element-wise p=2 ASAM. This directly refines a validated mechanism, keeps identical model-pass cadence and independent images, and has matched CIFAR WRN evidence above the acceptance gate.
- **Shared-budget CutMix/manifold mixup:** Preserve the early 0.5 total mixing gate and split selected batches between unchanged CutMix and one-pass hidden interpolation after the first or second WRN stage. This imports strong representation-flattening evidence without an extra forward, but necessarily reduces validated CutMix exposure.
- **Compact preactivation ResNeXt:** Replace dense two-convolution residual branches with six three-convolution grouped bottlenecks using cardinality eight. The proposed 2.16M-parameter model has slightly fewer estimated MACs than the parent and attacks representation directly, though small grouped kernels may be slow and the published architecture is much larger.
- **Squeeze-and-excitation residual gates:** Add reduction-16 channel recalibration after each residual branch's final convolution. It is a low-cost representation lever stronger than the 18-parameter ECA finalist from EXP-005, but the best evidence is ImageNet and a raw sigmoid gate changes initial residual scale.
- **Lookahead-enhanced late SAM:** Use the ICML lookahead construction to stabilize or extrapolate the final-quarter sharpness trajectory. This follows EXP-004's unexplored optimizer direction but creates more state and a less direct mapping to the current period-two implementation than ASAM.
- **Per-example logits memory bank:** Revisit temporal distillation while retaining ordinary shuffled batches by caching the latest logits for all 50,000 identities. This avoids EXP-005's unique-image loss, but teacher ages become variable and the result would need coverage/age controls before it is a credible accuracy bet.
- **GPU-charged structured Cutout on clean batches:** Mask one region only on a fixed subset of early clean batches inside the timed GPU loop. This preserves the data stream and introduces occlusion diversity, but CutMix already supplies spatial occlusion and the effect is likely close to the noise floor.
- **Drop-path simplification:** Remove stochastic depth now that CutMix and SAM provide two generalization mechanisms, recovering deterministic residual updates. EXP-003's lower-drop-path trials did not confirm, so a complete removal is weakly supported and risks repeating a saturated direction.
- **Moonshot PyramidNet/ShakeDrop rewrite:** Replace the six-block WRN with a gradually widening PyramidNet and ShakeDrop residual regularization. Published CIFAR ceilings are high, but the much deeper serial path is poorly matched to 300 charged seconds and would abandon three validated parent mechanisms at once.

## Combinations

- **ASAM + compact ResNeXt:** The ASAM paper reports a 0.46-point improvement over SAM on ResNeXt29, suggesting adaptive geometry and cardinality can reinforce each other. Combining them now would confound architecture and optimizer effects; ASAM should be validated on the parent first.
- **Manifold mixup + ASAM:** Hidden interpolation can improve the representation learned in the first 75%, while adaptive sharpness can refine it in the clean final quarter. The phases are naturally disjoint, but each mechanism must first prove additive value against EXP-004 alone.
- **SE + ASAM:** Channel recalibration changes feature selectivity at negligible pass cost, while ASAM handles parameter-scale geometry in the tail. This may be stronger than either, but the correct SE initialization and its adaptive perturbation scale need separate evidence.
- **Memory-bank distillation + CutMix-aware cache exclusion:** Cache only clean-example logits from normal independent batches and consume them when identities recur in later epochs. This preserves data diversity but yields sparse, stale targets; profiling teacher age and coverage is required before a full experiment.

## Candidate Ideas

### Compact Pre-Activation ResNeXt-20 8x16d
**Summary**: Replace the six dense WRN blocks with six preactivation grouped bottlenecks: stage outputs 128/256/512, cardinality eight, two blocks per stage, and a 32-channel stem. The proposed model has 2,159,018 parameters and about 0.91x parent MACs while preserving all training logic. A measured H20 latency gate must pass before a full run. Full design: `proposals/idea-03.md`.

**Reasoning**: Cardinality is a genuinely orthogonal representation lever, and the original paper shows it outperforming width at similar CIFAR model size. The compact proposal doubles stage-state widths while using grouped transforms to stay below parent parameter/MAC counts, potentially making better use of the fixed budget than dense residual branches.

**Sources**: `papers/resnext.md`; `proposals/idea-03.md`; EXP-004 and EXP-005 analyses.

**Estimated Effort**: high

**Risk Assessment**: Published CIFAR evidence uses 34M-plus-parameter ResNeXt-29 models, not this compact adaptation. Extra serial convolutions and grouped-kernel launch overhead may reduce steps despite lower MACs, and the new architecture may need optimization retuning that this isolated experiment forbids.

### Adaptive Clean-Tail Sharpness Minimization
**Summary**: Replace EXP-004's plain spherical SAM perturbation with element-wise p=2 ASAM at the exact same final-quarter period-two pulses. Use fixed literature settings `rho=0.5`, `eta=0.01`; adapt convolution, linear, and BatchNorm weights by `abs(w)+eta`, use unit scaling for biases, and preserve every forward, RNG replay, BatchNorm suppression, snapshot restore, and Nesterov update invariant. Full design: `proposals/idea-01.md`.

**Reasoning**: This is the most direct refinement of a validated mechanism. The ASAM paper reports 0.20-0.30-point gains over SAM on CIFAR-10 WRNs and broader gains across residual families, enough to clear the 0.10-point gate. It adds parameterwise arithmetic but no model pass and leaves the independent-image stream, architecture, CutMix phase, and evaluation cadence unchanged.

**Sources**: `papers/asam.md`; `proposals/idea-01.md`; EXP-004 and EXP-005 analyses.

**Estimated Effort**: medium to high

**Risk Assessment**: Published evidence uses full-run ASAM, not 2,450 late pulses; `rho=0.5` may be miscalibrated for a low-LR tail; and the squared weight scaling is easy to implement incorrectly. An adaptive-coordinate norm smoke, bias/BN classification, exact restore, and a 24,500-step diagnostic are mandatory.

### Shared-Budget CutMix and Manifold Mixup
**Summary**: Retain one early `p=0.5` mixing gate, then allocate 75% of selected batches to unchanged CutMix and 25% to `Beta(2,2)` manifold mixup after the 64- or 128-channel stage. Marginal early probabilities are 0.50 clean, 0.375 CutMix, and 0.0625 per hidden boundary. The final quarter remains entirely clean with unchanged SAM. Full design: `proposals/idea-02.md`.

**Reasoning**: Manifold mixup has the largest direct CIFAR effect among finalists and incurs no extra forward. It preserves every independent image and total mixed/clean exposure while diversifying regularization from input pixels into learned features. Direct evidence spans PreActResNet and WRN models, with large error and NLL reductions.

**Sources**: `papers/manifold-mixup.md`; `proposals/idea-02.md`; EXP-002, EXP-004, and EXP-005 analyses.

**Estimated Effort**: medium

**Risk Assessment**: The fixed policy halves a validated CutMix mechanism, so a gain or loss cannot be attributed to manifold mixup alone. The source recipes are longer and weaker than the parent, `Beta(2,2)` mixing may overregularize a shallow network, and extending the forward API creates boundary/label/RNG correctness risks.

## Review

Claude's cross-model review selected shared-budget CutMix/manifold mixup. It rejected ASAM as the lead because full-run 0.20-0.30-point gains plausibly compress to only 0.05-0.10 under the late period-two dose, and rejected compact ResNeXt because its downsizing stacks four extrapolations from much larger evidence. Claude's material refinement is adopted: preserve 75% rather than 50% of selected CutMix exposure, allocate 25% to manifold mixup, and explicitly state that this CutMix/hidden hybrid does not reproduce the paper's strongest linear-Mixup `{0,1,2}` policy. Full review: `01-idea-review.md`.

## Idea Evaluation

The Claude verdict is adopted. Manifold mixup scored 6/10 for evidence and 7/10 for impact because its heavily discounted 0.2-0.5-point ceiling remains above both the gate and observed variability while preserving images and model passes. ASAM scored 6/10 and 3/10 because it is cleanly isolated but likely too small-dose to clear the gate. Compact ResNeXt scored 3/10 and 6/10 because architectural upside is high but the proposed configuration is weakly supported under the frozen WRN recipe.

## Chosen Idea
**Selected**: Shared-Budget CutMix and Manifold Mixup

**Why this idea**:
It is the only finalist with a defensible true-effect ceiling above EXP-003's 0.14-0.29-point variability while preserving the independent-image stream, one-pass early training, total mixed/clean exposure, and validated SAM tail. The asymmetric policy limits downside by retaining 75% of selected CutMix batches.

**Hypothesis**:
During the first 75% of charged time, allocating marginal probabilities 0.375 to CutMix and 0.0625 to manifold mixup at each of the first two stage boundaries will retain at least 24,000 optimizer steps and improve EXP-004's 95.40% `best_test_acc` to 95.55-95.80% in one fixed-seed GPU-0 run. The expected 0.15-0.40-point gain comes from adding hidden-representation interpolation while preserving most validated CutMix exposure; the result measures the combined hybrid policy, not manifold mixup in isolation.
